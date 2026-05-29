include("warfarin_model.jl")

using Optim
using Random
using Base.Threads

const PARAM_NAMES = [
    "logKA", "logCL", "logV",
    "logE0", "logC50", "logKE0", "logitEMAX",
    "logSIGMA_PK", "logSIGMA_PD",
    "logOM_KA", "logOM_CL", "logOM_V",
    "logOM_E0", "logOM_C50", "logOM_KE0",
]

struct FileTime
    low::UInt32
    high::UInt32
end

function process_cpu_seconds()
    if Sys.iswindows()
        handle = ccall((:GetCurrentProcess, "kernel32"), Ptr{Cvoid}, ())
        creation = Ref(FileTime(0, 0))
        exit = Ref(FileTime(0, 0))
        kernel = Ref(FileTime(0, 0))
        user = Ref(FileTime(0, 0))
        ok = ccall((:GetProcessTimes, "kernel32"), Int32,
                   (Ptr{Cvoid}, Ref{FileTime}, Ref{FileTime}, Ref{FileTime}, Ref{FileTime}),
                   handle, creation, exit, kernel, user)
        if ok != 0
            k = (UInt64(kernel[].high) << 32) | UInt64(kernel[].low)
            u = (UInt64(user[].high) << 32) | UInt64(user[].low)
            return Float64(k + u) / 1.0e7
        end
    end
    return time()
end

function primal_float(x)
    y = x
    while y isa ForwardDiff.Dual || y isa ReverseDiff.TrackedReal
        if y isa ForwardDiff.Dual
            y = ForwardDiff.value(y)
        else
            y = ReverseDiff.value(y)
        end
    end
    return Float64(y)
end

primal_vector(v) = [primal_float(x) for x in v]

function shifted_matrix_ad(H; jitter::Float64=1.0e-4)
    q = size(H, 1)
    T = eltype(H)
    H_float = Matrix{Float64}(undef, q, q)
    for i in 1:q, j in 1:q
        H_float[i, j] = primal_float((H[i, j] + H[j, i]) / 2)
    end
    if any(!isfinite, H_float)
        error("non-finite Hessian in shifted_matrix_ad")
    end
    min_eig = minimum(eigvals(Symmetric(H_float)))
    shift = max(jitter, -min_eig + jitter)
    A = Matrix{T}(undef, q, q)
    for i in 1:q, j in 1:q
        A[i, j] = (H[i, j] + H[j, i]) / 2
    end
    for i in 1:q
        A[i, i] += T(shift)
    end
    return A
end

function unroll_newton_step(H, g; jitter::Float64=1.0e-6, floor_rel::Float64=1.0e-4,
                            floor_abs::Float64=1.0e-6, max_step_norm::Float64=3.0)
    q = length(g)
    T = promote_type(eltype(H), eltype(g))
    Hs = Matrix{Float64}(undef, q, q)
    for i in 1:q, j in 1:q
        Hs[i, j] = primal_float((H[i, j] + H[j, i]) / 2)
    end
    if any(!isfinite, Hs)
        error("non-finite Hessian in unrolled Newton step")
    end

    scale = max(mean(abs.(diag(Hs))), 1.0)
    shift = floor_abs + floor_rel * scale + jitter
    Iq = Matrix{Float64}(I, q, q)
    A_float = Hs + shift .* Iq
    shift_ok = false
    for _ in 1:8
        try
            cholesky(Symmetric(A_float))
            shift_ok = true
            break
        catch
        end
        shift *= 10.0
        A_float = Hs + shift .* Iq
    end

    if !shift_ok
        return g
    end

    A = Matrix{T}(undef, q, q)
    for i in 1:q, j in 1:q
        A[i, j] = T((H[i, j] + H[j, i]) / 2)
    end
    for i in 1:q
        A[i, i] += T(shift)
    end
    step = A \ T.(g)

    nrm = sqrt(sum(abs2, primal_vector(step)))
    if !isfinite(nrm)
        error("non-finite unrolled Newton step")
    end
    if nrm > max_step_norm
        step = step .* (max_step_norm / (nrm + 1.0e-12))
    end
    return step
end

mutable struct EtaWarmCache
    etas::Dict{Int, Vector{Float64}}
    best_f::Float64
end

EtaWarmCache() = EtaWarmCache(Dict{Int, Vector{Float64}}(), Inf)

function copy_eta_cache(cache::EtaWarmCache)
    return EtaWarmCache(Dict(k => copy(v) for (k, v) in cache.etas), cache.best_f)
end

function eta_start(cache::Union{Nothing,EtaWarmCache}, subj::SubjectData)
    if cache !== nothing && haskey(cache.etas, subj.sid)
        return copy(cache.etas[subj.sid])
    end
    return zeros(ETA_DIM)
end

function maybe_update_eta_cache!(cache::Union{Nothing,EtaWarmCache},
                                 subjects::Vector{SubjectData}, etas, f::Float64, g)
    cache === nothing && return
    isfinite(f) || return
    any(!isfinite, g) && return
    f <= cache.best_f || return
    for i in eachindex(subjects)
        eta = Vector{Float64}(etas[i])
        all(isfinite, eta) || return
    end
    for i in eachindex(subjects)
        cache.etas[subjects[i].sid] = copy(Vector{Float64}(etas[i]))
    end
    cache.best_f = f
end

function logdet_cholesky_ad(H; jitter::Float64=1.0e-4)
    A = shifted_matrix_ad(H; jitter=jitter)
    T = eltype(A)
    q = size(A, 1)
    L = zeros(T, q, q)
    for i in 1:q
        for j in 1:i
            s = A[i, j]
            for k in 1:(j - 1)
                s -= L[i, k] * L[j, k]
            end
            if i == j
                if primal_float(s) <= 0.0
                    s += T(abs(primal_float(s)) + jitter)
                end
                L[i, j] = sqrt(s)
            else
                L[i, j] = s / L[j, j]
            end
        end
    end
    return 2 * sum(log(L[i, i]) for i in 1:q)
end

function eta_grad_hess_ad(subj::SubjectData, x, eta, representation::Symbol; dt::Float64=0.25)
    f = e -> h_i(subj, x, e, representation; dt=dt)
    return ForwardDiff.gradient(f, eta), ForwardDiff.hessian(f, eta), f(eta)
end

fd_rel_step(base::Float64, z::Float64) = base * max(1.0, abs(z))

function fd_grad_hess_eta(subj::SubjectData, x::Vector{Float64}, eta::Vector{Float64},
                          representation::Symbol; dt::Float64=0.25, h::Float64=1.0e-3)
    q = length(eta)
    f0 = h_i(subj, x, eta, representation; dt=dt)
    g = zeros(q)
    H = zeros(q, q)
    for j in 1:q
        hj = fd_rel_step(h, eta[j])
        ep = copy(eta)
        em = copy(eta)
        ep[j] += hj
        em[j] -= hj
        fp = h_i(subj, x, ep, representation; dt=dt)
        fm = h_i(subj, x, em, representation; dt=dt)
        g[j] = (fp - fm) / (2.0 * hj)
        H[j, j] = (fp - 2.0 * f0 + fm) / (hj * hj)
    end
    for j in 1:q, k in (j + 1):q
        hj = fd_rel_step(h, eta[j])
        hk = fd_rel_step(h, eta[k])
        epp = copy(eta); epm = copy(eta); emp = copy(eta); emm = copy(eta)
        epp[j] += hj; epp[k] += hk
        epm[j] += hj; epm[k] -= hk
        emp[j] -= hj; emp[k] += hk
        emm[j] -= hj; emm[k] -= hk
        H[j, k] = (h_i(subj, x, epp, representation; dt=dt) - h_i(subj, x, epm, representation; dt=dt) -
                   h_i(subj, x, emp, representation; dt=dt) + h_i(subj, x, emm, representation; dt=dt)) / (4.0 * hj * hk)
        H[k, j] = H[j, k]
    end
    return g, H, f0
end

function fd_newton_step(H, g; jitter::Float64=1.0e-6, floor_rel::Float64=1.0e-2,
                        floor_abs::Float64=1.0e-3, max_step_norm::Float64=5.0)
    q = length(g)
    Hs = Matrix{Float64}(0.5 .* (H .+ transpose(H)))
    if any(!isfinite, Hs) || any(!isfinite, g)
        return zeros(q)
    end
    eigs = eigvals(Symmetric(Hs))
    scale = max(1.0, maximum(abs.(eigs)))
    target = max(floor_abs, floor_rel * scale)
    shift = max(0.0, target - minimum(eigs)) + jitter
    Iq = Matrix{Float64}(I, q, q)
    step = nothing
    for k in 0:7
        A = Hs .+ (shift * 10.0^k) .* Iq
        try
            F = cholesky(Symmetric(A); check=false)
            if issuccess(F)
                candidate = F \ Vector{Float64}(g)
                if all(isfinite, candidate)
                    step = candidate
                    break
                end
            end
        catch
        end
    end
    step === nothing && (step = Vector{Float64}(g))
    nrm = norm(step)
    if !isfinite(nrm)
        return zeros(q)
    end
    if nrm > max_step_norm
        step .*= max_step_norm / (nrm + 1.0e-12)
    end
    return step
end

function eta_mode_newton_fd(subj::SubjectData, x::Vector{Float64}, representation::Symbol;
                            dt::Float64=0.25, maxiter::Int=30, tol::Float64=1.0e-7,
                            eps_eta::Float64=1.0e-3,
                            floor_rel::Float64=parse(Float64, get(ENV, "WARFARIN_JULIA_FD_ETA_FLOOR_REL", "1e-6")),
                            floor_abs::Float64=parse(Float64, get(ENV, "WARFARIN_JULIA_FD_ETA_FLOOR_ABS", "1e-3")),
                            max_step_norm::Float64=parse(Float64, get(ENV, "WARFARIN_JULIA_FD_ETA_MAX_STEP", "5.0")),
                            eta0::AbstractVector{<:Real}=zeros(ETA_DIM))
    eta = Float64.(eta0)
    f_curr = h_i(subj, x, eta, representation; dt=dt)
    grad_norm = Inf
    converged = false
    for _ in 1:maxiter
        g, H, f0 = fd_grad_hess_eta(subj, x, eta, representation; dt=dt, h=eps_eta)
        grad_norm = norm(g)
        if !isfinite(f0) || any(!isfinite, g) || any(!isfinite, H)
            break
        end
        if grad_norm < tol
            f_curr = f0
            converged = true
            break
        end
        step = fd_newton_step(H, g; floor_rel=floor_rel, floor_abs=floor_abs,
                              max_step_norm=max_step_norm)
        alpha = 1.0
        accepted = false
        for _ls in 1:14
            trial = eta .- alpha .* step
            f_try = h_i(subj, x, trial, representation; dt=dt)
            if isfinite(Float64(f_try)) && Float64(f_try) <= Float64(f0) + 1.0e-10
                eta = trial
                f_curr = f_try
                accepted = true
                break
            end
            alpha *= 0.5
        end
        if !accepted
            trial = eta .- 0.1 .* step
            f_try = h_i(subj, x, trial, representation; dt=dt)
            if isfinite(Float64(f_try))
                eta = trial
                f_curr = f_try
            else
                break
            end
        end
    end
    return eta, Float64(f_curr), grad_norm, converged
end

function solve_all_etas(subjects::Vector{SubjectData}, x::Vector{Float64}, representation::Symbol;
                        dt::Float64=0.25, maxiter::Int=30, backend::Symbol=:forward,
                        eps_eta::Float64=1.0e-4,
                        eta_cache::Union{Nothing,EtaWarmCache}=nothing)
    etas = Vector{Vector{Float64}}(undef, length(subjects))
    grad_norms = zeros(length(subjects))
    converged = falses(length(subjects))
    @threads for i in eachindex(subjects)
        eta0 = eta_start(eta_cache, subjects[i])
        if backend == :fd
            eta, _, gn, cvg = eta_mode_newton_fd(subjects[i], x, representation; dt=dt,
                                                 maxiter=maxiter, eps_eta=eps_eta, eta0=eta0)
        else
            eta, _, gn, cvg = eta_mode_newton(subjects[i], x, :forward, representation;
                                              dt=dt, maxiter=maxiter, eta0=eta0)
        end
        etas[i] = eta
        grad_norms[i] = gn
        converged[i] = cvg
    end
    return etas, maximum(grad_norms), count(converged)
end

function laplace_subject_fixed_eta(subj::SubjectData, x, eta, representation::Symbol; dt::Float64=0.25)
    f = e -> h_i(subj, x, e, representation; dt=dt)
    H = ForwardDiff.hessian(f, eta)
    hi = f(eta)
    return 2.0 * (hi + 0.5 * logdet_cholesky_ad(H))
end

function laplace_subject_fixed_eta_fd(subj::SubjectData, x::Vector{Float64}, eta::Vector{Float64},
                                      representation::Symbol; dt::Float64=0.25,
                                      eps_eta::Float64=1.0e-3)
    for scale in (1.0, 3.0, 10.0, 30.0, 100.0)
        _, H, hi = fd_grad_hess_eta(subj, x, eta, representation; dt=dt, h=eps_eta * scale)
        ld = strict_fd_logdet(H)
        if isfinite(hi) && isfinite(ld)
            return 2.0 * (hi + 0.5 * ld)
        end
    end
    return BIG
end

function strict_fd_logdet(H; jitter::Float64=1.0e-8)
    Hs = Matrix{Float64}(0.5 .* (H .+ transpose(H)))
    any(!isfinite, Hs) && return Inf
    q = size(Hs, 1)
    F = cholesky(Symmetric(Hs .+ jitter .* Matrix{Float64}(I, q, q)); check=false)
    issuccess(F) || return Inf
    return 2.0 * sum(log, diag(F.L))
end

function outer_gradient(f, x::Vector{Float64})
    mode = lowercase(get(ENV, "WARFARIN_JULIA_OUTER_AD_MODE", "forward"))
    if mode in ("reverse", "backward")
        return ReverseDiff.gradient(f, x)
    elseif mode == "forward"
        return ForwardDiff.gradient(f, x)
    end
    error("Unknown WARFARIN_JULIA_OUTER_AD_MODE=$(mode); use reverse or forward")
end

function stop_value_grad(subjects::Vector{SubjectData}, x::Vector{Float64}, representation::Symbol;
                         dt::Float64=0.25, maxiter_eta::Int=30,
                         eta_cache::Union{Nothing,EtaWarmCache}=nothing)
    etas, max_eta_grad, n_conv = solve_all_etas(subjects, x, representation; dt=dt,
                                                maxiter=maxiter_eta, backend=:forward,
                                                eta_cache=eta_cache)
    vals = zeros(length(subjects))
    grads = [zeros(length(x)) for _ in subjects]
    @threads for i in eachindex(subjects)
        subj = subjects[i]
        eta = etas[i]
        fx = xx -> laplace_subject_fixed_eta(subj, xx, eta, representation; dt=dt)
        vals[i] = Float64(fx(x))
        grads[i] = outer_gradient(fx, x)
    end
    total = sum(vals)
    total_grad = vec(sum(reduce(hcat, grads), dims=2))
    maybe_update_eta_cache!(eta_cache, subjects, etas, total, total_grad)
    return total, total_grad, max_eta_grad, n_conv
end

function ad_population_value(subjects::Vector{SubjectData}, x::Vector{Float64}, representation::Symbol;
                             dt::Float64=0.25, maxiter_eta::Int=30)
    etas, max_eta_grad, n_conv = solve_all_etas(subjects, x, representation; dt=dt, maxiter=maxiter_eta, backend=:forward)
    vals = zeros(length(subjects))
    @threads for i in eachindex(subjects)
        vals[i] = Float64(laplace_subject_fixed_eta(subjects[i], x, etas[i], representation; dt=dt))
    end
    return sum(vals), max_eta_grad, n_conv
end

function safe_ad_population_value(subjects::Vector{SubjectData}, x::Vector{Float64}, representation::Symbol;
                                  dt::Float64=0.25, maxiter_eta::Int=30)
    try
        return ad_population_value(subjects, x, representation; dt=dt, maxiter_eta=maxiter_eta)[1]
    catch err
        @warn "AD endpoint re-evaluation failed; recording NaN" exception=typeof(err)
        return NaN
    end
end

function full_implicit_subject_value_grad(subj::SubjectData, x::Vector{Float64}, eta::Vector{Float64},
                                          representation::Symbol; dt::Float64=0.25)
    H = ForwardDiff.hessian(e -> h_i(subj, x, e, representation; dt=dt), eta)
    ofv = laplace_subject_fixed_eta(subj, x, eta, representation; dt=dt)
    dh_dx = ForwardDiff.gradient(xx -> h_i(subj, xx, eta, representation; dt=dt), x)
    dld_dx = ForwardDiff.gradient(
        xx -> logdet_cholesky_ad(ForwardDiff.hessian(e -> h_i(subj, xx, e, representation; dt=dt), eta)),
        x,
    )
    dld_deta = ForwardDiff.gradient(
        ee -> logdet_cholesky_ad(ForwardDiff.hessian(e -> h_i(subj, x, e, representation; dt=dt), ee)),
        eta,
    )
    lambda = Matrix{Float64}(shifted_matrix_ad(H)) \ (0.5 .* Vector{Float64}(dld_deta))
    term_c = ForwardDiff.gradient(
        xx -> dot(ForwardDiff.gradient(e -> h_i(subj, xx, e, representation; dt=dt), eta), lambda),
        x,
    )
    grad = 2.0 .* (dh_dx .+ 0.5 .* dld_dx .- term_c)
    return Float64(ofv), Vector{Float64}(grad)
end

function full_implicit_value_grad(subjects::Vector{SubjectData}, x::Vector{Float64}, representation::Symbol;
                                  dt::Float64=0.25, maxiter_eta::Int=30,
                                  eta_cache::Union{Nothing,EtaWarmCache}=nothing)
    etas, max_eta_grad, n_conv = solve_all_etas(subjects, x, representation; dt=dt,
                                                maxiter=maxiter_eta, backend=:forward,
                                                eta_cache=eta_cache)
    vals = zeros(length(subjects))
    grads = [zeros(length(x)) for _ in subjects]
    @threads for i in eachindex(subjects)
        vals[i], grads[i] = full_implicit_subject_value_grad(subjects[i], x, etas[i], representation; dt=dt)
    end
    total = sum(vals)
    total_grad = vec(sum(reduce(hcat, grads), dims=2))
    maybe_update_eta_cache!(eta_cache, subjects, etas, total, total_grad)
    return total, total_grad, max_eta_grad, n_conv
end

function eta_newton_unroll(subj::SubjectData, x, eta0::Vector{Float64}, representation::Symbol;
                           dt::Float64=0.25, steps::Int=30, tol::Float64=1.0e-6)
    T = eltype(x)
    eta = T.(eta0)
    for _ in 1:steps
        g, H, hi = eta_grad_hess_ad(subj, x, eta, representation; dt=dt)
        if !isfinite(primal_float(hi)) || any(!isfinite, primal_vector(g))
            error("non-finite unrolled eta gradient")
        end
        step = unroll_newton_step(H, g)
        step_norm = sqrt(sum(abs2, primal_vector(step)))
        step_norm < tol && break

        # Line-search decisions are made on detached/primal values, then the
        # accepted scalar alpha is applied to the differentiable Newton update.
        x_pr = primal_vector(x)
        eta_pr = primal_vector(eta)
        step_pr = primal_vector(step)
        f0 = h_i(subj, x_pr, eta_pr, representation; dt=dt)
        if !isfinite(Float64(f0))
            error("non-finite unrolled eta objective")
        end
        alpha = 1.0
        accepted = false
        for _ls in 1:14
            trial_pr = eta_pr .- alpha .* step_pr
            f_try = h_i(subj, x_pr, trial_pr, representation; dt=dt)
            if isfinite(Float64(f_try)) && Float64(f_try) <= Float64(f0) + 1.0e-10
                eta = eta .- alpha .* step
                accepted = true
                break
            end
            alpha *= 0.5
        end
        accepted || break
        alpha * step_norm < tol && break
        sqrt(sum(abs2, primal_vector(g))) < tol && break
    end
    return eta
end

function full_unroll_subject_value(subj::SubjectData, x, eta0::Vector{Float64}, representation::Symbol;
                                   dt::Float64=0.25, steps::Int=30)
    eta = eta_newton_unroll(subj, x, eta0, representation; dt=dt, steps=steps)
    return laplace_subject_fixed_eta(subj, x, eta, representation; dt=dt)
end

function full_unroll_value_grad(subjects::Vector{SubjectData}, x::Vector{Float64}, representation::Symbol;
                                dt::Float64=0.25, steps::Int=30,
                                eta_cache::Union{Nothing,EtaWarmCache}=nothing)
    vals = zeros(length(subjects))
    grads = [zeros(length(x)) for _ in subjects]
    etas = Vector{Vector{Float64}}(undef, length(subjects))
    @threads for i in eachindex(subjects)
        subj = subjects[i]
        eta0 = eta_start(eta_cache, subj)
        eta = eta_newton_unroll(subj, x, eta0, representation; dt=dt, steps=steps)
        etas[i] = primal_vector(eta)
        vals[i] = Float64(laplace_subject_fixed_eta(subj, x, eta, representation; dt=dt))
        fx = xx -> full_unroll_subject_value(subj, xx, eta0, representation; dt=dt, steps=steps)
        grads[i] = outer_gradient(fx, x)
    end
    total = sum(vals)
    total_grad = vec(sum(reduce(hcat, grads), dims=2))
    maybe_update_eta_cache!(eta_cache, subjects, etas, total, total_grad)
    return total, total_grad, NaN, 0
end

function fd_population_value(subjects::Vector{SubjectData}, x::Vector{Float64}, representation::Symbol;
                             dt::Float64=0.25, maxiter_eta::Int=30,
                             eps_eta::Float64=1.0e-3,
                             eta_cache::Union{Nothing,EtaWarmCache}=nothing)
    etas, _, _ = solve_all_etas(subjects, x, representation; dt=dt, maxiter=maxiter_eta,
                                backend=:fd, eps_eta=eps_eta, eta_cache=eta_cache)
    return fd_population_value_from_etas(subjects, x, etas, representation; dt=dt,
                                         eps_eta=eps_eta)
end

function fd_population_value_from_etas(subjects::Vector{SubjectData}, x::Vector{Float64}, etas,
                                       representation::Symbol; dt::Float64=0.25,
                                       eps_eta::Float64=1.0e-3)
    vals = zeros(length(subjects))
    @threads for i in eachindex(subjects)
        vals[i] = laplace_subject_fixed_eta_fd(subjects[i], x, etas[i], representation;
                                               dt=dt, eps_eta=eps_eta)
    end
    return sum(vals)
end

function fd_value_grad(subjects::Vector{SubjectData}, x::Vector{Float64}, representation::Symbol;
                       dt::Float64=0.25, maxiter_eta::Int=30,
                       eps::Float64=parse(Float64, get(ENV, "WARFARIN_JULIA_FD_EPS_THETA", "1e-3")),
                       eps_eta::Float64=parse(Float64, get(ENV, "WARFARIN_JULIA_FD_EPS_ETA", "1e-3")),
                       eta_cache::Union{Nothing,EtaWarmCache}=nothing)
    etas, max_eta_grad, n_conv = solve_all_etas(subjects, x, representation; dt=dt,
                                                maxiter=maxiter_eta, backend=:fd,
                                                eps_eta=eps_eta, eta_cache=eta_cache)
    f0 = fd_population_value_from_etas(subjects, x, etas, representation; dt=dt,
                                       eps_eta=eps_eta)
    base_cache = EtaWarmCache()
    for i in eachindex(subjects)
        base_cache.etas[subjects[i].sid] = copy(etas[i])
    end
    base_cache.best_f = isfinite(f0) ? f0 : Inf
    g = zeros(length(x))
    lo, hi = theta_bounds()
    @threads for j in eachindex(x)
        xp = copy(x)
        xm = copy(x)
        h = eps * max(1.0, abs(x[j]))
        xp[j] = min(max(x[j] + h, lo[j]), hi[j])
        xm[j] = min(max(x[j] - h, lo[j]), hi[j])
        step_p = xp[j] - x[j]
        step_m = x[j] - xm[j]
        if step_p == 0.0 && step_m == 0.0
            g[j] = 0.0
        elseif step_p == 0.0 || step_m == 0.0
            xone = step_p == 0.0 ? xm : xp
            step = step_p == 0.0 ? -step_m : step_p
            f1 = fd_population_value(subjects, xone, representation; dt=dt,
                                      maxiter_eta=maxiter_eta, eps_eta=eps_eta,
                                      eta_cache=copy_eta_cache(base_cache))
            g[j] = isfinite(f1) ? (f1 - f0) / step : 0.0
        else
            fp = fd_population_value(subjects, xp, representation; dt=dt,
                                     maxiter_eta=maxiter_eta, eps_eta=eps_eta,
                                     eta_cache=copy_eta_cache(base_cache))
            fm = fd_population_value(subjects, xm, representation; dt=dt,
                                     maxiter_eta=maxiter_eta, eps_eta=eps_eta,
                                     eta_cache=copy_eta_cache(base_cache))
            g[j] = (isfinite(fp) && isfinite(fm)) ? (fp - fm) / (step_p + step_m) : 0.0
        end
    end
    fd_cache_tol = parse(Float64, get(ENV, "WARFARIN_JULIA_FD_CACHE_MAX_ETA_GRAD", "1e-3"))
    if n_conv == length(subjects) || (isfinite(max_eta_grad) && max_eta_grad <= fd_cache_tol)
        maybe_update_eta_cache!(eta_cache, subjects, etas, Float64(f0), g)
    end
    return f0, g, max_eta_grad, n_conv
end

function method_evaluator(method::String, subjects::Vector{SubjectData}, representation::Symbol;
                          dt::Float64=0.25, maxiter_eta::Int=30, full_unroll_steps::Int=30)
    eta_cache = EtaWarmCache()
    if method == "STOP"
        return x -> stop_value_grad(subjects, x, representation; dt=dt,
                                    maxiter_eta=maxiter_eta, eta_cache=eta_cache)
    elseif method == "FULL_IMPLICIT"
        return x -> full_implicit_value_grad(subjects, x, representation; dt=dt,
                                             maxiter_eta=maxiter_eta, eta_cache=eta_cache)
    elseif method == "FULL_UNROLL"
        return x -> full_unroll_value_grad(subjects, x, representation; dt=dt,
                                           steps=full_unroll_steps, eta_cache=eta_cache)
    elseif method == "FD"
        return x -> fd_value_grad(subjects, x, representation; dt=dt,
                                  maxiter_eta=maxiter_eta, eta_cache=eta_cache)
    end
    error("unknown method: $method")
end

function canonical_method(method::String)
    m = uppercase(strip(method))
    m == "FULL_IMPLIC" && return "FULL_IMPLICIT"
    m == "STOP+FULL_IMPLIC" && return "STOP+FULL_IMPLICIT"
    m == "FULL_IMPLIC+STOP" && return "FULL_IMPLICIT+STOP"
    return m
end

function stage_plan(method::String, maxiter_outer::Int)
    m = canonical_method(method)
    if m in ("STOP+FULL", "STOP+FULL_IMPLICIT")
        stage1 = parse(Int, get(ENV, "WARFARIN_JULIA_STOP_FULL_STOP_ITERS", "1"))
        stage1 = max(1, min(stage1, maxiter_outer - 1))
        stage2 = parse(Int, get(ENV, "WARFARIN_JULIA_STOP_FULL_FULL_ITERS", string(maxiter_outer - stage1)))
        return "STOP+FULL_IMPLICIT", "STOP", "FULL_IMPLICIT", stage1, max(1, stage2)
    elseif m in ("FULL+STOP", "FULL_IMPLICIT+STOP")
        default_full = string(max(1, min(14, maxiter_outer - 1)))
        stage1 = parse(Int, get(ENV, "WARFARIN_JULIA_FULL_STOP_FULL_ITERS", default_full))
        stage1 = max(1, min(stage1, maxiter_outer - 1))
        stage2 = parse(Int, get(ENV, "WARFARIN_JULIA_FULL_STOP_STOP_ITERS", string(maxiter_outer - stage1)))
        return "FULL_IMPLICIT+STOP", "FULL_IMPLICIT", "STOP", stage1, max(1, stage2)
    end
    return m, "", "", 0, 0
end

is_staged_method(method::String) = stage_plan(method, 50)[2] != ""

function base_x0()
    return [
        log(1.0), log(0.2), log(10.0),
        log(100.0), log(2.0), log(0.05), log(0.9 / 0.1),
        log(0.5), log(5.0),
        fill(log(0.3), 6)...,
    ]
end

function theta_bounds()
    lo = [log(1.0e-3), log(1.0e-4), log(1.0e-2),
          log(1.0), log(1.0e-3), log(1.0e-4), -10.0,
          log(1.0e-4), log(1.0e-4),
          fill(log(1.0e-4), 6)...]
    hi = [log(50.0), log(10.0), log(1.0e3),
          log(200.0), log(1.0e3), log(10.0), 10.0,
          log(50.0), log(200.0),
          fill(log(5.0), 6)...]
    return lo, hi
end

function sample_starts(n::Int, base::Vector{Float64}, lo::Vector{Float64}, hi::Vector{Float64};
                       seed::Int=123579, scale::Float64=0.5)
    rng = MersenneTwister(seed)
    starts = Matrix{Float64}(undef, n, length(base))
    for s in 1:n
        starts[s, :] = min.(max.(base .+ randn(rng, length(base)) .* scale, lo), hi)
    end
    return starts
end

function write_start_bank(path::AbstractString, starts::Matrix{Float64})
    open(path, "w") do io
        println(io, join(vcat(["model", "start_id"], PARAM_NAMES), ","))
        for s in 1:size(starts, 1)
            println(io, join(vcat(["warfarin", string(s - 1)], string.(Vector{Float64}(starts[s, :]))), ","))
        end
    end
end

mutable struct EvalCache
    valid::Bool
    x::Vector{Float64}
    f::Float64
    g::Vector{Float64}
    max_eta_grad::Float64
    n_converged::Int
    evals::Int
    failed::Bool
    has_finite::Bool
    last_finite_x::Vector{Float64}
    last_finite_f::Float64
end

EvalCache(p::Int) = EvalCache(false, zeros(p), Inf, zeros(p), NaN, 0, 0, false,
                              false, zeros(p), Inf)

struct OptimFallbackResult
    x::Vector{Float64}
    iterations::Int
    converged::Bool
end

Optim.minimizer(r::OptimFallbackResult) = r.x
Optim.iterations(r::OptimFallbackResult) = r.iterations
Optim.converged(r::OptimFallbackResult) = r.converged

function ensure_cache!(cache::EvalCache, evaluator, x)
    xx = Vector{Float64}(x)
    if cache.valid && length(cache.x) == length(xx) && all(cache.x .== xx)
        return
    end
    try
        f, g, max_eta_grad, n_conv = evaluator(xx)
        if !isfinite(f) || any(!isfinite, g)
            error("non-finite objective or gradient")
        end
        cache.x = xx
        cache.f = Float64(f)
        cache.g = Vector{Float64}(g)
        cache.max_eta_grad = Float64(max_eta_grad)
        cache.n_converged = Int(n_conv)
        cache.failed = false
        cache.has_finite = true
        cache.last_finite_x = copy(xx)
        cache.last_finite_f = cache.f
    catch err
        cache.x = xx
        if cache.has_finite && length(cache.last_finite_x) == length(xx)
            d = xx .- cache.last_finite_x
            cache.f = BIG + 1.0e6 * sum(abs2, d)
            cache.g = 2.0e6 .* d
        else
            cache.f = BIG
            cache.g = zeros(length(xx))
        end
        cache.max_eta_grad = NaN
        cache.n_converged = 0
        cache.failed = true
        @warn "method evaluation failed; returning penalty" exception=typeof(err)
    end
    cache.valid = true
    cache.evals += 1
end

function optimize_one(method::String, subjects::Vector{SubjectData}, representation::Symbol,
                      theta0::Vector{Float64}, lo::Vector{Float64}, hi::Vector{Float64};
                      dt::Float64=0.25, maxiter_eta::Int=30, full_unroll_steps::Int=30,
                      maxiter_outer::Int=50)
    evaluator = method_evaluator(method, subjects, representation; dt=dt, maxiter_eta=maxiter_eta,
                                 full_unroll_steps=full_unroll_steps)
    cache = EvalCache(length(theta0))
    function bounds_penalty(x)
        xx = Vector{Float64}(x)
        below = max.(lo .- xx, 0.0)
        above = max.(xx .- hi, 0.0)
        d = above .- below
        if any(!iszero, d)
            return true, BIG + 1.0e6 * sum(abs2, d), 2.0e6 .* d
        end
        return false, 0.0, zeros(length(xx))
    end
    f(x) = begin
        outside, fp, _ = bounds_penalty(x)
        outside && return fp
        ensure_cache!(cache, evaluator, x)
        cache.f
    end
    function g!(G, x)
        outside, _, gp = bounds_penalty(x)
        if outside
            G .= gp
            return G
        end
        ensure_cache!(cache, evaluator, x)
        G .= cache.g
        return G
    end
    GC.gc()
    wall0 = time_ns()
    cpu0 = process_cpu_seconds()
    linesearch = lowercase(get(ENV, "WARFARIN_JULIA_LINESEARCH", "hagerzhang"))
    method_obj = linesearch == "backtracking" ? LBFGS(linesearch=Optim.LineSearches.BackTracking()) : LBFGS()
    result = try
        optimize(f, g!, theta0, method_obj,
                 Optim.Options(iterations=maxiter_outer, g_tol=1.0e-6,
                               f_reltol=1.0e-8, show_trace=false))
    catch err
        @warn "outer optimizer failed; returning last finite point" method=method exception=typeof(err)
        fallback_x = cache.has_finite ? copy(cache.last_finite_x) : copy(theta0)
        cache.failed = true
        OptimFallbackResult(fallback_x, 0, false)
    end
    wall = (time_ns() - wall0) / 1.0e9
    cpu = process_cpu_seconds() - cpu0
    theta_hat = min.(max.(Vector{Float64}(Optim.minimizer(result)), lo), hi)
    ensure_cache!(cache, evaluator, theta_hat)
    return result, cache, wall, cpu
end

function optimize_method(method::String, subjects::Vector{SubjectData}, representation::Symbol,
                         theta0::Vector{Float64}, lo::Vector{Float64}, hi::Vector{Float64};
                         dt::Float64=0.25, maxiter_eta::Int=30, full_unroll_steps::Int=30,
                         maxiter_outer::Int=50)
    display_method, first_method, second_method, first_iters, second_iters = stage_plan(method, maxiter_outer)
    if isempty(first_method)
        result, cache, wall, cpu = optimize_one(display_method, subjects, representation, theta0, lo, hi;
                                                dt=dt, maxiter_eta=maxiter_eta,
                                                full_unroll_steps=full_unroll_steps,
                                                maxiter_outer=maxiter_outer)
        theta_hat = Vector{Float64}(Optim.minimizer(result))
        return (
            method=display_method,
            theta=theta_hat,
            iterations=Optim.iterations(result),
            converged=Optim.converged(result),
            objective=cache.f,
            wall=wall,
            cpu=cpu,
            evals=cache.evals,
            max_eta_grad=cache.max_eta_grad,
            n_converged=cache.n_converged,
            failed=cache.failed,
            stage1_method="",
            stage1_outer=0,
            stage2_method="",
            stage2_outer=0,
        )
    end

    result1, cache1, wall1, cpu1 = optimize_one(first_method, subjects, representation, theta0, lo, hi;
                                                dt=dt, maxiter_eta=maxiter_eta,
                                                full_unroll_steps=full_unroll_steps,
                                                maxiter_outer=first_iters)
    theta1 = min.(max.(Vector{Float64}(Optim.minimizer(result1)), lo), hi)
    result2, cache2, wall2, cpu2 = optimize_one(second_method, subjects, representation, theta1, lo, hi;
                                                dt=dt, maxiter_eta=maxiter_eta,
                                                full_unroll_steps=full_unroll_steps,
                                                maxiter_outer=second_iters)
    theta_hat = min.(max.(Vector{Float64}(Optim.minimizer(result2)), lo), hi)
    return (
        method=display_method,
        theta=theta_hat,
        iterations=Optim.iterations(result1) + Optim.iterations(result2),
        converged=Optim.converged(result2),
        objective=cache2.f,
        wall=wall1 + wall2,
        cpu=cpu1 + cpu2,
        evals=cache1.evals + cache2.evals,
        max_eta_grad=cache2.max_eta_grad,
        n_converged=cache2.n_converged,
        failed=cache1.failed || cache2.failed,
        stage1_method=first_method,
        stage1_outer=first_iters,
        stage2_method=second_method,
        stage2_outer=second_iters,
    )
end

function write_results(path::AbstractString, rows)
    open(path, "w") do io
        header = vcat([
            "model", "implementation", "representation", "method", "start_id", "n_subjects",
            "maxiter_eta", "full_unroll_steps", "outer_iterations", "success", "objective",
            "objective_ad_eval", "objective_stop_eval", "wall_sec", "cpu_sec", "method_eval_count",
            "max_eta_grad_norm", "n_eta_converged", "stage1_method", "stage1_outer",
            "stage2_method", "stage2_outer",
        ], PARAM_NAMES)
        println(io, join(header, ","))
        for r in rows
            vals = vcat([
                r.model, r.implementation, r.representation, r.method, r.start_id, r.n_subjects,
                r.maxiter_eta, r.full_unroll_steps, r.outer_iterations, r.success, r.objective,
                r.objective_ad_eval, r.objective_stop_eval, r.wall_sec, r.cpu_sec, r.method_eval_count,
                r.max_eta_grad_norm, r.n_eta_converged, r.stage1_method, r.stage1_outer,
                r.stage2_method, r.stage2_outer,
            ], r.theta)
            println(io, join(string.(vals), ","))
        end
    end
end

split_tokens(s::String) = [String(strip(x)) for x in split(s, ",") if !isempty(strip(x))]

function parse_rep(s::String)
    ss = lowercase(strip(s))
    ss == "ode" && return :ode
    error("unknown representation: $s; this public release supports ode")
end

function main()
    data_path = get(ENV, "WARFARIN_JULIA_DATA", normpath(joinpath(@__DIR__, "..", "data", "warfarin_dat.csv")))
    subjects_all = parse_warfarin_csv(data_path)
    n_subjects = parse(Int, get(ENV, "WARFARIN_JULIA_N_SUBJ", string(length(subjects_all))))
    subjects = subjects_all[1:min(n_subjects, length(subjects_all))]
    n_starts = parse(Int, get(ENV, "WARFARIN_JULIA_N_STARTS", "10"))
    maxiter_eta = parse(Int, get(ENV, "WARFARIN_JULIA_MAXITER_ETA", "50"))
    maxiter_outer = parse(Int, get(ENV, "WARFARIN_JULIA_MAXITER_OUTER", "50"))
    full_unroll_steps = parse(Int, get(ENV, "WARFARIN_JULIA_FULL_UNROLL_STEPS", string(maxiter_eta)))
    dt = parse(Float64, get(ENV, "WARFARIN_JULIA_DT", "0.25"))
    methods = split_tokens(get(ENV, "WARFARIN_JULIA_METHODS", "FD,FULL_IMPLICIT,FULL_UNROLL,STOP,STOP+FULL,FULL+STOP"))
    reps = [parse_rep(x) for x in split_tokens(get(ENV, "WARFARIN_JULIA_REPRESENTATIONS", "ode"))]
    outdir = get(ENV, "WARFARIN_JULIA_OUTDIR", joinpath(@__DIR__, "tables"))
    mkpath(outdir)
    outpath = joinpath(outdir, "warfarin_julia_multistart_methods.csv")
    starts_path = joinpath(outdir, "warfarin_julia_start_bank.csv")

    lo, hi = theta_bounds()
    starts = sample_starts(n_starts, base_x0(), lo, hi)
    write_start_bank(starts_path, starts)

    println("Warfarin Julia multistart methods")
    println("threads=", nthreads(), " subjects=", length(subjects), " starts=", n_starts)
    println("representations=", join(string.(reps), ","), " methods=", join(methods, ","))
    println("maxiter_eta=", maxiter_eta, " full_unroll_steps=", full_unroll_steps,
            " maxiter_outer=", maxiter_outer, " dt=", dt)
    println("output=", outpath)
    println("starts=", starts_path)

    for rep in reps, method in methods
        println("warming ", rep, " / ", method)
        _, first_method, second_method, _, _ = stage_plan(method, maxiter_outer)
        warm_methods = isempty(first_method) ? [canonical_method(method)] : [first_method, second_method]
        for warm_method in warm_methods
            evaluator = method_evaluator(warm_method, subjects, rep; dt=dt,
                                         maxiter_eta=min(maxiter_eta, 2),
                                         full_unroll_steps=min(full_unroll_steps, 2))
            evaluator(Vector{Float64}(starts[1, :]))
        end
    end

    rows = NamedTuple[]
    for rep in reps
        for method in methods
            for s in 1:n_starts
                theta0 = Vector{Float64}(starts[s, :])
                @printf("\n[%s/%s] start %d/%d\n", string(rep), method, s, n_starts)
                outcome = optimize_method(method, subjects, rep, theta0, lo, hi;
                                          dt=dt, maxiter_eta=maxiter_eta,
                                          full_unroll_steps=full_unroll_steps,
                                          maxiter_outer=maxiter_outer)
                theta_hat = outcome.theta
                ad_eval = safe_ad_population_value(subjects, theta_hat, rep; dt=dt, maxiter_eta=maxiter_eta)
                stop_eval = ad_eval
                row = (
                    model="warfarin",
                    implementation="julia_forward_eta",
                    representation=string(rep),
                    method=outcome.method,
                    start_id=s - 1,
                    n_subjects=length(subjects),
                    maxiter_eta=maxiter_eta,
                    full_unroll_steps=full_unroll_steps,
                    outer_iterations=outcome.iterations,
                    success=outcome.converged && !outcome.failed && isfinite(ad_eval),
                    objective=outcome.objective,
                    objective_ad_eval=ad_eval,
                    objective_stop_eval=stop_eval,
                    wall_sec=outcome.wall,
                    cpu_sec=outcome.cpu,
                    method_eval_count=outcome.evals,
                    max_eta_grad_norm=outcome.max_eta_grad,
                    n_eta_converged=outcome.n_converged,
                    stage1_method=outcome.stage1_method,
                    stage1_outer=outcome.stage1_outer,
                    stage2_method=outcome.stage2_method,
                    stage2_outer=outcome.stage2_outer,
                    theta=theta_hat,
                )
                push!(rows, row)
                write_results(outpath, rows)
                @printf("  success=%s objective=%.6f ad_eval=%.6f wall=%.3f cpu=%.3f evals=%d\n",
                        string(row.success), row.objective, row.objective_ad_eval,
                        row.wall_sec, row.cpu_sec, row.method_eval_count)
            end
        end
    end
    println("\nSaved ", length(rows), " rows to ", outpath)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
