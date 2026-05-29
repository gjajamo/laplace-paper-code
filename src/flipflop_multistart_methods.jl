using LinearAlgebra
using Printf
using Random
using Statistics
using ForwardDiff
using ReverseDiff
using Optim
using Base.Threads

const DOSE = 100.0
const TIMES = [0.25, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 24.0, 48.0]
const ETA_DIM = 2
const BIG = 1.0e12
const MIN_POS = 1.0e-12
const PARAM_NAMES = ["logka", "logcl", "logv", "logomega_ka", "logomega_v", "logsigma"]

struct SubjectData
    y::Vector{Float64}
end

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

function base_float(x)
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

function pk_conc(t, ka, cl, v)
    ka = max(ka, MIN_POS)
    cl = max(cl, MIN_POS)
    v = max(v, MIN_POS)
    ke = cl / v
    absd = abs(ka - ke)
    base_rate = min(ka, ke)
    base = exp(-base_rate * t)
    x = absd * t
    ratio = if base_float(x) < 1.0e-6
        t - 0.5 * absd * t^2 + (absd^2) * t^3 / 6.0
    else
        -expm1(-x) / max(absd, MIN_POS)
    end
    return (DOSE / v) * ka * base * ratio
end

function simulate_subjects(theta::Vector{Float64}; n_subj::Int=50, seed::Int=123)
    rng = MersenneTwister(seed)
    logka, logcl, logv, logomega_ka, logomega_v, logsigma = theta
    omega_ka = exp(logomega_ka)
    omega_v = exp(logomega_v)
    sigma = exp(logsigma)
    subjects = SubjectData[]
    for _ in 1:n_subj
        eta_ka = randn(rng) * omega_ka
        eta_v = randn(rng) * omega_v
        ka = exp(logka + eta_ka)
        cl = exp(logcl)
        v = exp(logv + eta_v)
        y = Float64[]
        for t in TIMES
            push!(y, max(pk_conc(t, ka, cl, v) + randn(rng) * sigma, 1.0e-8))
        end
        push!(subjects, SubjectData(y))
    end
    return subjects
end

function read_subjects_long_csv(path::AbstractString)
    lines = readlines(path)
    isempty(lines) && error("empty subject data file: $path")
    header = lowercase.(strip.(split(lines[1], ",")))
    id_idx = findfirst(==("id"), header)
    time_idx = findfirst(==("time"), header)
    dv_idx = findfirst(x -> x in ("dv", "y"), header)
    (id_idx === nothing || time_idx === nothing || dv_idx === nothing) &&
        error("subject CSV must contain id,time,dv columns: $path")

    grouped = Dict{Int, Vector{Tuple{Float64, Float64}}}()
    for line in lines[2:end]
        isempty(strip(line)) && continue
        parts = strip.(split(line, ","))
        sid = parse(Int, parts[id_idx])
        tm = parse(Float64, parts[time_idx])
        dv = parse(Float64, parts[dv_idx])
        push!(get!(grouped, sid, Tuple{Float64, Float64}[]), (tm, dv))
    end

    subjects = SubjectData[]
    for sid in sort(collect(keys(grouped)))
        rows = sort(grouped[sid]; by=x -> x[1])
        if length(rows) != length(TIMES)
            error("subject $sid has $(length(rows)) observations; expected $(length(TIMES))")
        end
        push!(subjects, SubjectData([dv for (_, dv) in rows]))
    end
    return subjects
end

function subject_nll(subj::SubjectData, theta, eta)
    logka, logcl, logv, logomega_ka, logomega_v, logsigma = theta
    ka = exp(logka + eta[1])
    cl = exp(logcl)
    v = exp(logv + eta[2])
    omega_ka = exp(logomega_ka)
    omega_v = exp(logomega_v)
    sigma = exp(logsigma)

    total = zero(ka + cl + v + omega_ka + omega_v + sigma + sum(eta))
    total += length(subj.y) * log(sigma)
    for (j, t) in enumerate(TIMES)
        r = (subj.y[j] - pk_conc(t, ka, cl, v)) / sigma
        total += 0.5 * r * r
    end
    total += log(omega_ka) + log(omega_v)
    total += 0.5 * (eta[1] / omega_ka)^2
    total += 0.5 * (eta[2] / omega_v)^2
    return total
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

function shifted_matrix(H; jitter::Float64=1.0e-6)
    q = size(H, 1)
    T = eltype(H)
    H_float = Matrix{Float64}(undef, q, q)
    for i in 1:q, j in 1:q
        H_float[i, j] = primal_float((H[i, j] + H[j, i]) / 2)
    end
    if any(!isfinite, H_float)
        error("non-finite Hessian in shifted_matrix")
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

mutable struct EtaWarmCache
    etas::Dict{Int, Vector{Float64}}
    best_f::Float64
end

EtaWarmCache() = EtaWarmCache(Dict{Int, Vector{Float64}}(), Inf)

function copy_eta_cache(cache::Union{Nothing,EtaWarmCache})
    cache === nothing && return nothing
    return EtaWarmCache(Dict(k => copy(v) for (k, v) in cache.etas), cache.best_f)
end

function eta_start(cache::Union{Nothing,EtaWarmCache}, idx::Int)
    if cache !== nothing && haskey(cache.etas, idx)
        return copy(cache.etas[idx])
    end
    return zeros(ETA_DIM)
end

function maybe_update_eta_cache!(cache::Union{Nothing,EtaWarmCache},
                                 etas, f::Float64, g)
    cache === nothing && return
    isfinite(f) || return
    any(!isfinite, g) && return
    f <= cache.best_f || return
    for i in eachindex(etas)
        eta = Vector{Float64}(etas[i])
        all(isfinite, eta) || return
    end
    for i in eachindex(etas)
        cache.etas[Int(i)] = copy(Vector{Float64}(etas[i]))
    end
    cache.best_f = f
end

function logdet_cholesky_once(H; jitter::Float64)
    T = eltype(H)
    q = size(H, 1)
    A = Matrix{T}(undef, q, q)
    for i in 1:q, j in 1:q
        A[i, j] = (H[i, j] + H[j, i]) / 2
    end
    for i in 1:q
        A[i, i] += T(jitter)
    end
    L = zeros(T, q, q)
    for i in 1:q
        for j in 1:i
            s = A[i, j]
            for k in 1:(j - 1)
                s -= L[i, k] * L[j, k]
            end
            if i == j
                if base_float(s) <= 0.0 || !isfinite(base_float(s))
                    error("non-positive Cholesky pivot")
                end
                L[i, j] = sqrt(s)
            else
                L[i, j] = s / L[j, j]
            end
        end
    end
    return 2 * sum(log(L[i, i]) for i in 1:q)
end

function logdet_cholesky_python_floor(H; jitter::Float64=1.0e-8, max_tries::Int=20,
                                      floor_rel::Float64=1.0e-4, floor_abs::Float64=1.0e-6)
    q = size(H, 1)
    T = eltype(H)
    Hs = Matrix{T}(undef, q, q)
    for i in 1:q, j in 1:q
        Hs[i, j] = (H[i, j] + H[j, i]) / 2
    end
    diag_scale = mean(abs(primal_float(Hs[i, i])) for i in 1:q)
    diag_scale = max(diag_scale, 1.0)
    shift = floor_abs + floor_rel * diag_scale + jitter
    for _ in 1:max_tries
        try
            return logdet_cholesky_once(Hs; jitter=shift)
        catch
            shift *= 10.0
        end
    end

    max_need = 0.0
    for i in 1:q
        off = sum(abs(primal_float(Hs[i, j])) for j in 1:q if j != i)
        need = off - primal_float(Hs[i, i]) + floor_abs + floor_rel * diag_scale
        max_need = max(max_need, need)
    end
    return logdet_cholesky_once(Hs; jitter=max(max_need, 0.0) + jitter)
end

function logdet_cholesky(H; jitter::Float64=1.0e-8, max_tries::Int=20)
    logdet_mode = lowercase(get(ENV, "FLIPFLOP_JULIA_LOGDET_MODE", "raw"))
    if logdet_mode in ("python", "python_floor", "python-floor")
        return logdet_cholesky_python_floor(H; jitter=jitter, max_tries=max_tries)
    end
    current = jitter
    for _ in 1:max_tries
        try
            return logdet_cholesky_once(H; jitter=current)
        catch
            current *= 10.0
        end
    end
    A = shifted_matrix(H; jitter=current)
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

function grad_hess_forward(subj::SubjectData, theta, eta)
    f = e -> subject_nll(subj, theta, e)
    return ForwardDiff.gradient(f, eta), ForwardDiff.hessian(f, eta), f(eta)
end

fd_rel_step(base::Float64, z::Float64) = base * max(1.0, abs(z))

function fd_grad_hess_eta(subj::SubjectData, theta::Vector{Float64}, eta::Vector{Float64}; h::Float64=1.0e-3)
    q = length(eta)
    f0 = subject_nll(subj, theta, eta)
    g = zeros(q)
    H = zeros(q, q)
    for j in 1:q
        hj = fd_rel_step(h, eta[j])
        ep = copy(eta)
        em = copy(eta)
        ep[j] += hj
        em[j] -= hj
        fp = subject_nll(subj, theta, ep)
        fm = subject_nll(subj, theta, em)
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
        H[j, k] = (subject_nll(subj, theta, epp) - subject_nll(subj, theta, epm) -
                   subject_nll(subj, theta, emp) + subject_nll(subj, theta, emm)) / (4.0 * hj * hk)
        H[k, j] = H[j, k]
    end
    return g, H, f0
end

function fd_newton_step(H, g; jitter::Float64=1.0e-8,
                        floor_rel::Float64=parse(Float64, get(ENV, "FLIPFLOP_JULIA_FD_ETA_FLOOR_REL", "1e-6")),
                        floor_abs::Float64=parse(Float64, get(ENV, "FLIPFLOP_JULIA_FD_ETA_FLOOR_ABS", "1e-8")),
                        max_step_norm::Float64=parse(Float64, get(ENV, "FLIPFLOP_JULIA_FD_ETA_MAX_STEP", "10.0")))
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

function safe_newton_step(H, g; max_step_norm::Float64=10.0, jitter::Float64=1.0e-8)
    A = Matrix{Float64}(shifted_matrix(H; jitter=jitter))
    step = A \ Vector{Float64}(g)
    nrm = norm(step)
    if isfinite(nrm) && nrm > max_step_norm
        step .*= max_step_norm / (nrm + 1.0e-12)
    end
    return step
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

    nrm = sqrt(sum(abs2, [primal_float(s) for s in step]))
    if !isfinite(nrm)
        error("non-finite unrolled Newton step")
    end
    if nrm > max_step_norm
        step = step .* (max_step_norm / (nrm + 1.0e-12))
    end
    return step
end

function eta_mode_newton(subj::SubjectData, theta::Vector{Float64}; maxiter::Int=20,
                         tol::Float64=1.0e-8, backend::Symbol=:forward,
                         eps_eta::Float64=1.0e-3,
                         eta0::Union{Nothing,Vector{Float64}}=nothing)
    eta = eta0 === nothing ? zeros(ETA_DIM) : copy(eta0)
    f_curr = subject_nll(subj, theta, eta)
    grad_norm = Inf
    converged = false
    for _ in 1:maxiter
        g, H, f0 = backend == :fd ? fd_grad_hess_eta(subj, theta, eta; h=eps_eta) : grad_hess_forward(subj, theta, eta)
        grad_norm = norm(g)
        if !isfinite(f0) || any(!isfinite, g) || any(!isfinite, H)
            break
        end
        if grad_norm < tol
            f_curr = f0
            converged = true
            break
        end
        step = backend == :fd ? fd_newton_step(H, g) : safe_newton_step(H, g)
        alpha = 1.0
        accepted = false
        for _ls in 1:12
            trial = eta .- alpha .* step
            f_try = subject_nll(subj, theta, trial)
            if isfinite(f_try) && f_try <= f0 + 1.0e-10
                eta = trial
                f_curr = f_try
                accepted = true
                break
            end
            alpha *= 0.5
        end
        if !accepted
            trial = eta .- 0.1 .* step
            f_try = subject_nll(subj, theta, trial)
            if isfinite(f_try)
                eta = trial
                f_curr = f_try
            else
                break
            end
        end
    end
    return eta, Float64(f_curr), grad_norm, converged
end

function eta_mode_bfgs(subj::SubjectData, theta::Vector{Float64}; maxiter::Int=20,
                       tol::Float64=1.0e-8, backend::Symbol=:forward,
                       eps_eta::Float64=1.0e-4,
                       eta0::Union{Nothing,Vector{Float64}}=nothing)
    cache_eta = zeros(ETA_DIM)
    cache_valid = false
    cache_f = Inf
    cache_g = zeros(ETA_DIM)
    function eval_eta!(eta)
        eta_vec = Vector{Float64}(eta)
        if cache_valid && all(cache_eta .== eta_vec)
            return
        end
        f0 = subject_nll(subj, theta, eta_vec)
        g = backend == :fd ? fd_grad_hess_eta(subj, theta, eta_vec; h=eps_eta)[1] :
                              ForwardDiff.gradient(e -> subject_nll(subj, theta, e), eta_vec)
        cache_eta = eta_vec
        cache_f = Float64(f0)
        cache_g = Vector{Float64}(g)
        cache_valid = true
    end
    f(eta) = begin
        eval_eta!(eta)
        cache_f
    end
    function g!(G, eta)
        eval_eta!(eta)
        G .= cache_g
        return G
    end
    eta_initial = eta0 === nothing ? zeros(ETA_DIM) : copy(eta0)
    result = optimize(f, g!, eta_initial, BFGS(),
                      Optim.Options(iterations=maxiter, g_tol=tol, f_reltol=1.0e-10, show_trace=false))
    eta_hat = Vector{Float64}(Optim.minimizer(result))
    final_g = backend == :fd ? fd_grad_hess_eta(subj, theta, eta_hat; h=eps_eta)[1] :
                               ForwardDiff.gradient(e -> subject_nll(subj, theta, e), eta_hat)
    grad_norm = norm(final_g)
    return eta_hat, Float64(subject_nll(subj, theta, eta_hat)), grad_norm, grad_norm < max(tol, 1.0e-6)
end

function laplace_subject_fixed_eta(subj::SubjectData, theta, eta)
    f = e -> subject_nll(subj, theta, e)
    H = ForwardDiff.hessian(f, eta)
    hi = f(eta)
    return 2.0 * (hi + 0.5 * logdet_cholesky(H))
end

function laplace_subject_fixed_eta_fd(subj::SubjectData, theta::Vector{Float64}, eta::Vector{Float64};
                                      eps_eta::Float64=1.0e-3)
    for scale in (1.0, 3.0, 10.0, 30.0, 100.0)
        _, H, hi = fd_grad_hess_eta(subj, theta, eta; h=eps_eta * scale)
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

function solve_all_etas(subjects::Vector{SubjectData}, theta::Vector{Float64}; maxiter::Int=20,
                        backend::Symbol=:forward, eps_eta::Float64=1.0e-3,
                        solver::Symbol=:newton,
                        eta_cache::Union{Nothing,EtaWarmCache}=nothing)
    etas = Vector{Vector{Float64}}(undef, length(subjects))
    grad_norms = zeros(length(subjects))
    converged = falses(length(subjects))
    @threads for i in eachindex(subjects)
        eta0 = eta_start(eta_cache, Int(i))
        eta, _, gn, cvg = solver == :bfgs ?
            eta_mode_bfgs(subjects[i], theta; maxiter=maxiter, backend=backend, eps_eta=eps_eta, eta0=eta0) :
            eta_mode_newton(subjects[i], theta; maxiter=maxiter, backend=backend, eps_eta=eps_eta, eta0=eta0)
        etas[i] = eta
        grad_norms[i] = gn
        converged[i] = cvg
    end
    return etas, maximum(grad_norms), count(converged)
end

function outer_gradient(f, theta::Vector{Float64})
    mode = lowercase(get(ENV, "FLIPFLOP_JULIA_OUTER_AD_MODE", "forward"))
    if mode in ("reverse", "backward")
        return ReverseDiff.gradient(f, theta)
    elseif mode == "forward"
        return ForwardDiff.gradient(f, theta)
    end
    error("Unknown FLIPFLOP_JULIA_OUTER_AD_MODE=$(mode); use reverse or forward")
end

function parse_eta_solver()
    mode = lowercase(get(ENV, "FLIPFLOP_JULIA_ETA_SOLVER", "newton"))
    mode in ("newton", "hessian", "second_order", "second-order") && return :newton
    mode in ("bfgs", "gradient", "gradient_only", "gradient-only") && return :bfgs
    error("Unknown FLIPFLOP_JULIA_ETA_SOLVER=$(mode); use newton or bfgs")
end

function use_eta_cache()
    mode = lowercase(get(ENV, "FLIPFLOP_JULIA_USE_ETA_CACHE", "true"))
    return !(mode in ("0", "false", "no", "off"))
end

function stop_value_grad(subjects::Vector{SubjectData}, theta::Vector{Float64}; maxiter_eta::Int=20,
                         eta_solver::Symbol=:newton,
                         eta_cache::Union{Nothing,EtaWarmCache}=nothing)
    etas, max_eta_grad, n_conv = solve_all_etas(subjects, theta; maxiter=maxiter_eta,
                                                backend=:forward, solver=eta_solver,
                                                eta_cache=eta_cache)
    vals = zeros(length(subjects))
    grads = [zeros(length(theta)) for _ in subjects]
    @threads for i in eachindex(subjects)
        subj = subjects[i]
        eta = etas[i]
        ftheta = x -> laplace_subject_fixed_eta(subj, x, eta)
        vals[i] = Float64(ftheta(theta))
        grads[i] = outer_gradient(ftheta, theta)
    end
    total = sum(vals)
    total_grad = vec(sum(reduce(hcat, grads), dims=2))
    maybe_update_eta_cache!(eta_cache, etas, total, total_grad)
    return total, total_grad, max_eta_grad, n_conv
end

function ad_population_value(subjects::Vector{SubjectData}, theta::Vector{Float64}; maxiter_eta::Int=20,
                             eta_solver::Symbol=:newton,
                             eta_cache::Union{Nothing,EtaWarmCache}=nothing)
    etas, max_eta_grad, n_conv = solve_all_etas(subjects, theta; maxiter=maxiter_eta,
                                                backend=:forward, solver=eta_solver,
                                                eta_cache=eta_cache)
    vals = zeros(length(subjects))
    @threads for i in eachindex(subjects)
        vals[i] = Float64(laplace_subject_fixed_eta(subjects[i], theta, etas[i]))
    end
    return sum(vals), max_eta_grad, n_conv
end

function safe_ad_population_value(subjects::Vector{SubjectData}, theta::Vector{Float64}; maxiter_eta::Int=20,
                                  eta_solver::Symbol=:newton)
    try
        return ad_population_value(subjects, theta; maxiter_eta=maxiter_eta, eta_solver=eta_solver)[1]
    catch err
        @warn "AD endpoint re-evaluation failed; recording NaN" exception=typeof(err)
        return NaN
    end
end

function full_implicit_subject_value_grad(subj::SubjectData, theta::Vector{Float64}, eta::Vector{Float64})
    H = ForwardDiff.hessian(e -> subject_nll(subj, theta, e), eta)
    ofv = laplace_subject_fixed_eta(subj, theta, eta)
    dh_dx = ForwardDiff.gradient(x -> subject_nll(subj, x, eta), theta)
    dld_dx = ForwardDiff.gradient(x -> logdet_cholesky(ForwardDiff.hessian(e -> subject_nll(subj, x, e), eta)), theta)
    dld_deta = ForwardDiff.gradient(e -> logdet_cholesky(ForwardDiff.hessian(ee -> subject_nll(subj, theta, ee), e)), eta)
    lambda = Matrix{Float64}(shifted_matrix(H)) \ (0.5 .* Vector{Float64}(dld_deta))
    term_c = ForwardDiff.gradient(
        x -> dot(ForwardDiff.gradient(e -> subject_nll(subj, x, e), eta), lambda),
        theta,
    )
    grad = 2.0 .* (dh_dx .+ 0.5 .* dld_dx .- term_c)
    return Float64(ofv), Vector{Float64}(grad)
end

function full_implicit_value_grad(subjects::Vector{SubjectData}, theta::Vector{Float64}; maxiter_eta::Int=20,
                                  eta_solver::Symbol=:newton,
                                  eta_cache::Union{Nothing,EtaWarmCache}=nothing)
    etas, max_eta_grad, n_conv = solve_all_etas(subjects, theta; maxiter=maxiter_eta,
                                                backend=:forward, solver=eta_solver,
                                                eta_cache=eta_cache)
    vals = zeros(length(subjects))
    grads = [zeros(length(theta)) for _ in subjects]
    @threads for i in eachindex(subjects)
        vals[i], grads[i] = full_implicit_subject_value_grad(subjects[i], theta, etas[i])
    end
    total = sum(vals)
    total_grad = vec(sum(reduce(hcat, grads), dims=2))
    maybe_update_eta_cache!(eta_cache, etas, total, total_grad)
    return total, total_grad, max_eta_grad, n_conv
end

function eta_newton_unroll(subj::SubjectData, theta, eta0::Vector{Float64}; steps::Int=20)
    T = eltype(theta)
    eta = T.(eta0)
    for _ in 1:steps
        g, H, hi = grad_hess_forward(subj, theta, eta)
        if !isfinite(primal_float(hi)) || any(!isfinite, [primal_float(s) for s in g])
            error("non-finite unrolled eta gradient")
        end
        step = unroll_newton_step(H, g)
        step_norm = sqrt(sum(abs2, [primal_float(s) for s in step]))
        step_norm < 1.0e-8 && break

        theta_pr = [primal_float(s) for s in theta]
        eta_pr = [primal_float(s) for s in eta]
        step_pr = [primal_float(s) for s in step]
        f0 = subject_nll(subj, theta_pr, eta_pr)
        if !isfinite(Float64(f0))
            error("non-finite unrolled eta objective")
        end
        alpha = 1.0
        accepted = false
        for _ls in 1:14
            trial_pr = eta_pr .- alpha .* step_pr
            f_try = subject_nll(subj, theta_pr, trial_pr)
            if isfinite(Float64(f_try)) && Float64(f_try) <= Float64(f0) + 1.0e-10
                eta = eta .- alpha .* step
                accepted = true
                break
            end
            alpha *= 0.5
        end
        if !accepted
            break
        end
        alpha * step_norm < 1.0e-8 && break
        sqrt(sum(abs2, [primal_float(s) for s in g])) < 1.0e-8 && break
    end
    return eta
end

function full_unroll_subject_value(subj::SubjectData, theta, eta0::Vector{Float64}; steps::Int=20)
    eta = eta_newton_unroll(subj, theta, eta0; steps=steps)
    return laplace_subject_fixed_eta(subj, theta, eta)
end

function full_unroll_value_grad(subjects::Vector{SubjectData}, theta::Vector{Float64}; steps::Int=20,
                                eta_cache::Union{Nothing,EtaWarmCache}=nothing)
    vals = zeros(length(subjects))
    grads = [zeros(length(theta)) for _ in subjects]
    etas = Vector{Vector{Float64}}(undef, length(subjects))
    @threads for i in eachindex(subjects)
        subj = subjects[i]
        eta0 = eta_start(eta_cache, Int(i))
        eta = eta_newton_unroll(subj, theta, eta0; steps=steps)
        etas[i] = [primal_float(s) for s in eta]
        ftheta = x -> full_unroll_subject_value(subj, x, eta0; steps=steps)
        vals[i] = Float64(ftheta(theta))
        grads[i] = outer_gradient(ftheta, theta)
    end
    total = sum(vals)
    total_grad = vec(sum(reduce(hcat, grads), dims=2))
    maybe_update_eta_cache!(eta_cache, etas, total, total_grad)
    return total, total_grad, NaN, 0
end

function fd_population_value(subjects::Vector{SubjectData}, theta::Vector{Float64}; maxiter_eta::Int=20,
                             eps_eta::Float64=1.0e-3, eta_solver::Symbol=:newton,
                             eta_cache::Union{Nothing,EtaWarmCache}=nothing)
    etas, _, _ = solve_all_etas(subjects, theta; maxiter=maxiter_eta, backend=:fd,
                                eps_eta=eps_eta, solver=eta_solver,
                                eta_cache=eta_cache)
    return fd_population_value_from_etas(subjects, theta, etas; eps_eta=eps_eta)
end

function fd_population_value_from_etas(subjects::Vector{SubjectData}, theta::Vector{Float64}, etas;
                                       eps_eta::Float64=1.0e-3)
    vals = zeros(length(subjects))
    @threads for i in eachindex(subjects)
        vals[i] = laplace_subject_fixed_eta_fd(subjects[i], theta, etas[i]; eps_eta=eps_eta)
    end
    return sum(vals)
end

function fd_value_grad(subjects::Vector{SubjectData}, theta::Vector{Float64}; maxiter_eta::Int=20,
                       eps::Float64=parse(Float64, get(ENV, "FLIPFLOP_JULIA_FD_EPS_THETA", "1e-3")),
                       eps_eta::Float64=parse(Float64, get(ENV, "FLIPFLOP_JULIA_FD_EPS_ETA", "1e-3")),
                       eta_solver::Symbol=:newton,
                       eta_cache::Union{Nothing,EtaWarmCache}=nothing)
    etas, max_eta_grad, n_conv = solve_all_etas(subjects, theta; maxiter=maxiter_eta,
                                                backend=:fd, eps_eta=eps_eta,
                                                solver=eta_solver, eta_cache=eta_cache)
    f0 = fd_population_value_from_etas(subjects, theta, etas; eps_eta=eps_eta)
    base_cache = EtaWarmCache()
    for i in eachindex(subjects)
        base_cache.etas[Int(i)] = copy(etas[i])
    end
    base_cache.best_f = isfinite(f0) ? f0 : Inf
    g = zeros(length(theta))
    lo, hi = theta_bounds()
    @threads for j in eachindex(theta)
        xp = copy(theta)
        xm = copy(theta)
        h = eps * max(1.0, abs(theta[j]))
        xp[j] = min(max(theta[j] + h, lo[j]), hi[j])
        xm[j] = min(max(theta[j] - h, lo[j]), hi[j])
        step_p = xp[j] - theta[j]
        step_m = theta[j] - xm[j]
        if step_p == 0.0 && step_m == 0.0
            g[j] = 0.0
        elseif step_p == 0.0 || step_m == 0.0
            xone = step_p == 0.0 ? xm : xp
            step = step_p == 0.0 ? -step_m : step_p
            f1 = fd_population_value(subjects, xone; maxiter_eta=maxiter_eta, eps_eta=eps_eta,
                                     eta_solver=eta_solver, eta_cache=copy_eta_cache(base_cache))
            g[j] = isfinite(f1) ? (f1 - f0) / step : 0.0
        else
            fp = fd_population_value(subjects, xp; maxiter_eta=maxiter_eta, eps_eta=eps_eta,
                                     eta_solver=eta_solver, eta_cache=copy_eta_cache(base_cache))
            fm = fd_population_value(subjects, xm; maxiter_eta=maxiter_eta, eps_eta=eps_eta,
                                     eta_solver=eta_solver, eta_cache=copy_eta_cache(base_cache))
            g[j] = (isfinite(fp) && isfinite(fm)) ? (fp - fm) / (step_p + step_m) : 0.0
        end
    end
    fd_cache_tol = parse(Float64, get(ENV, "FLIPFLOP_JULIA_FD_CACHE_MAX_ETA_GRAD", "1e-3"))
    if n_conv == length(subjects) || (isfinite(max_eta_grad) && max_eta_grad <= fd_cache_tol)
        maybe_update_eta_cache!(eta_cache, etas, Float64(f0), g)
    end
    return f0, g, max_eta_grad, n_conv
end

function theta_bounds()
    lo = log.([1.0e-3, 0.05, 1.0, 0.01, 0.01, 0.01])
    hi = log.([2.0, 30.0, 300.0, 1.0, 1.0, 5.0])
    return lo, hi
end

function clip_to_bounds(theta::Vector{Float64}, lo::Vector{Float64}, hi::Vector{Float64})
    return min.(max.(theta, lo), hi)
end

function sample_starts(n::Int, theta_true1::Vector{Float64}, theta_true2::Vector{Float64},
                       lo::Vector{Float64}, hi::Vector{Float64}; seed::Int=20260322)
    rng = MersenneTwister(seed)
    starts = Matrix{Float64}(undef, n, length(theta_true1))
    scale = [0.5, 0.15, 0.5, 0.15, 0.15, 0.15]
    for s in 1:n
        center = rand(rng) < 0.5 ? theta_true1 : theta_true2
        starts[s, :] = clip_to_bounds(center .+ randn(rng, length(center)) .* scale, lo, hi)
    end
    return starts
end

function read_start_bank_csv(path::AbstractString, lo::Vector{Float64}, hi::Vector{Float64}; n_starts::Int)
    lines = readlines(path)
    isempty(lines) && error("empty start-bank file: $path")
    header = lowercase.(strip.(split(lines[1], ",")))
    idxs = [findfirst(==(name), header) for name in PARAM_NAMES]
    any(isnothing, idxs) && error("start bank must contain $(join(PARAM_NAMES, ", ")): $path")

    rows = Vector{Vector{Float64}}()
    for line in lines[2:end]
        isempty(strip(line)) && continue
        parts = strip.(split(line, ","))
        theta = [parse(Float64, parts[Int(idx)]) for idx in idxs]
        push!(rows, clip_to_bounds(theta, lo, hi))
        length(rows) >= n_starts && break
    end
    length(rows) < n_starts && error("start bank $path has $(length(rows)) rows; expected at least $n_starts")
    starts = Matrix{Float64}(undef, n_starts, length(PARAM_NAMES))
    for i in 1:n_starts
        starts[i, :] .= rows[i]
    end
    return starts
end

function method_evaluator(method::String, subjects::Vector{SubjectData}; maxiter_eta::Int=20,
                          full_unroll_steps::Int=20, eta_solver::Symbol=:newton)
    eta_cache = use_eta_cache() ? EtaWarmCache() : nothing
    if method == "STOP"
        return theta -> stop_value_grad(subjects, theta; maxiter_eta=maxiter_eta,
                                        eta_solver=eta_solver, eta_cache=eta_cache)
    elseif method == "FULL_IMPLICIT"
        return theta -> full_implicit_value_grad(subjects, theta; maxiter_eta=maxiter_eta,
                                                 eta_solver=eta_solver, eta_cache=eta_cache)
    elseif method == "FULL_UNROLL"
        eta_solver == :newton || error("FULL_UNROLL currently requires the differentiable Newton unroll eta solver")
        return theta -> full_unroll_value_grad(subjects, theta; steps=full_unroll_steps,
                                               eta_cache=eta_cache)
    elseif method == "FD"
        return theta -> fd_value_grad(subjects, theta; maxiter_eta=maxiter_eta,
                                      eta_solver=eta_solver, eta_cache=eta_cache)
    end
    error("unknown method: $method")
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

function EvalCache(p::Int)
    return EvalCache(false, zeros(p), Inf, zeros(p), NaN, 0, 0,
                     false, false, zeros(p), Inf)
end

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
            cache.f = BIG + sum(abs2, xx)
            cache.g = 2.0 .* xx
        end
        cache.max_eta_grad = NaN
        cache.n_converged = 0
        cache.failed = true
        @warn "method evaluation failed; returning smooth penalty" exception=typeof(err)
    end
    cache.valid = true
    cache.evals += 1
end

function optimize_one(method::String, subjects::Vector{SubjectData}, theta0::Vector{Float64},
                      lo::Vector{Float64}, hi::Vector{Float64}; maxiter_eta::Int=20,
                      full_unroll_steps::Int=20, maxiter_outer::Int=50,
                      eta_solver::Symbol=:newton)
    evaluator = method_evaluator(method, subjects; maxiter_eta=maxiter_eta,
                                 full_unroll_steps=full_unroll_steps,
                                 eta_solver=eta_solver)
    cache = EvalCache(length(theta0))
    bound_mode = lowercase(get(ENV, "FLIPFLOP_JULIA_BOUND_MODE", "penalty"))
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
    f_box(x) = begin
        ensure_cache!(cache, evaluator, Vector{Float64}(x))
        cache.f
    end
    function g_box!(G, x)
        ensure_cache!(cache, evaluator, Vector{Float64}(x))
        G .= cache.g
        return G
    end
    f_penalty(x) = begin
        outside, fp, _ = bounds_penalty(x)
        outside && return fp
        ensure_cache!(cache, evaluator, x)
        cache.f
    end
    function g_penalty!(G, x)
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
    theta0_box = min.(max.(theta0, lo), hi)
    linesearch = lowercase(get(ENV, "FLIPFLOP_JULIA_LINESEARCH", "hagerzhang"))
    lbfgs = linesearch == "backtracking" ? LBFGS(linesearch=Optim.LineSearches.BackTracking()) : LBFGS()
    result = try
        if bound_mode == "penalty"
            optimize(f_penalty, g_penalty!, theta0_box, lbfgs,
                     Optim.Options(iterations=maxiter_outer, g_tol=1.0e-6,
                                   f_reltol=1.0e-8, show_trace=false))
        elseif bound_mode == "fminbox"
            optimize(f_box, g_box!, lo, hi, theta0_box, Fminbox(lbfgs),
                     Optim.Options(iterations=maxiter_outer, g_tol=1.0e-6,
                                   f_reltol=1.0e-8, show_trace=false))
        else
            error("Unknown FLIPFLOP_JULIA_BOUND_MODE=$(bound_mode); use fminbox or penalty")
        end
    catch err
        @warn "outer optimizer failed; returning last finite point" method=method exception=typeof(err)
        fallback_x = cache.has_finite ? copy(cache.last_finite_x) : copy(theta0_box)
        cache.failed = true
        OptimFallbackResult(fallback_x, 0, false)
    end
    wall = (time_ns() - wall0) / 1.0e9
    cpu = process_cpu_seconds() - cpu0
    theta_hat = min.(max.(Vector{Float64}(Optim.minimizer(result)), lo), hi)
    ensure_cache!(cache, evaluator, theta_hat)
    return result, cache, wall, cpu
end

function write_results(path::AbstractString, rows)
    open(path, "w") do io
        header = [
            "model", "implementation", "method", "start_id", "n_subjects", "maxiter_eta",
            "eta_solver", "full_unroll_steps", "outer_iterations", "success", "objective", "objective_ad_eval", "objective_stop_eval",
            "wall_sec", "cpu_sec", "method_eval_count", "max_eta_grad_norm", "n_eta_converged",
            "theta_logka", "theta_logcl", "theta_logv", "theta_logomega_ka", "theta_logomega_v", "theta_logsigma",
            "ka_pop", "cl_pop", "v_pop", "omega_ka", "omega_v", "sigma",
        ]
        println(io, join(header, ","))
        for r in rows
            theta = r.theta
            vals = [
                r.model, r.implementation, r.method, r.start_id, r.n_subjects, r.maxiter_eta,
                r.eta_solver, r.full_unroll_steps, r.outer_iterations, r.success, r.objective, r.objective_ad_eval, r.objective_stop_eval,
                r.wall_sec, r.cpu_sec, r.method_eval_count, r.max_eta_grad_norm, r.n_eta_converged,
                theta[1], theta[2], theta[3], theta[4], theta[5], theta[6],
                exp(theta[1]), exp(theta[2]), exp(theta[3]), exp(theta[4]), exp(theta[5]), exp(theta[6]),
            ]
            println(io, join(string.(vals), ","))
        end
    end
end

function write_start_bank(path::AbstractString, starts::Matrix{Float64})
    open(path, "w") do io
        header = vcat(["model", "start_id"], PARAM_NAMES, ["ka_pop", "cl_pop", "v_pop", "omega_ka", "omega_v", "sigma"])
        println(io, join(header, ","))
        for s in 1:size(starts, 1)
            theta = Vector{Float64}(starts[s, :])
            vals = vcat(["flipflop", string(s - 1)], string.(theta), string.(exp.(theta)))
            println(io, join(vals, ","))
        end
    end
end

function write_nonmem_data(path::AbstractString, subjects::Vector{SubjectData})
    open(path, "w") do io
        println(io, "ID TIME AMT DV EVID MDV CMT")
        for (i, subj) in enumerate(subjects)
            println(io, join((i, 0.0, DOSE, 0.0, 1, 1, 1), " "))
            for (t, y) in zip(TIMES, subj.y)
                println(io, join((i, t, 0.0, y, 0, 0, 2), " "))
            end
        end
    end
end

function split_methods(s::String)
    return [String(strip(x)) for x in split(s, ",") if !isempty(strip(x))]
end

function main()
    n_subj = parse(Int, get(ENV, "FLIPFLOP_JULIA_N_SUBJ", "50"))
    n_starts = parse(Int, get(ENV, "FLIPFLOP_JULIA_N_STARTS", "100"))
    maxiter_eta = parse(Int, get(ENV, "FLIPFLOP_JULIA_MAXITER_ETA", "50"))
    maxiter_outer = parse(Int, get(ENV, "FLIPFLOP_JULIA_MAXITER_OUTER", "50"))
    full_unroll_steps = parse(Int, get(ENV, "FLIPFLOP_JULIA_FULL_UNROLL_STEPS", string(maxiter_eta)))
    eta_solver = parse_eta_solver()
    methods = split_methods(get(ENV, "FLIPFLOP_JULIA_METHODS", "FULL_IMPLICIT,FULL_UNROLL,STOP,FD"))
    outdir = get(ENV, "FLIPFLOP_JULIA_OUTDIR", joinpath(@__DIR__, "tables"))
    mkpath(outdir)
    outpath = joinpath(outdir, "flipflop_julia_multistart_methods.csv")
    starts_path = joinpath(outdir, "flipflop_julia_start_bank.csv")
    nonmem_data_path = joinpath(outdir, "flipflop_nonmem.dat")

    theta_true1 = log.([0.05, 3.0, 20.0, 0.1, 0.1, 0.1])
    theta_true2 = log.([3.0 / 20.0, 3.0, 3.0 / 0.05, 0.1, 0.1, 0.1])
    lo, hi = theta_bounds()
    subjects_csv = get(ENV, "FLIPFLOP_JULIA_SUBJECTS_CSV", "")
    starts_csv = get(ENV, "FLIPFLOP_JULIA_STARTS_CSV", "")
    subjects = isempty(subjects_csv) ? simulate_subjects(theta_true1; n_subj=n_subj, seed=123) : read_subjects_long_csv(subjects_csv)
    if !isempty(subjects_csv)
        n_subj = length(subjects)
    end
    starts = isempty(starts_csv) ? sample_starts(n_starts, theta_true1, theta_true2, lo, hi) : read_start_bank_csv(starts_csv, lo, hi; n_starts=n_starts)
    write_start_bank(starts_path, starts)
    write_nonmem_data(nonmem_data_path, subjects)

    println("Flip-flop Julia multistart methods")
    println("threads=", nthreads(), " subjects=", n_subj, " starts=", n_starts)
    println("methods=", join(methods, ","), " maxiter_eta=", maxiter_eta,
            " full_unroll_steps=", full_unroll_steps, " maxiter_outer=", maxiter_outer)
    println("eta_solver=", eta_solver, " eta_dim=", ETA_DIM)
    println("bound_mode=", lowercase(get(ENV, "FLIPFLOP_JULIA_BOUND_MODE", "penalty")))
    println("output=", outpath)
    println("starts=", starts_path)
    println("nonmem_data=", nonmem_data_path)
    !isempty(subjects_csv) && println("matched_subjects_csv=", subjects_csv)
    !isempty(starts_csv) && println("matched_starts_csv=", starts_csv)

    # Warm the compiled method paths outside reported timings.
    for method in methods
        println("warming ", method)
        evaluator = method_evaluator(method, subjects; maxiter_eta=min(maxiter_eta, 3),
                                     full_unroll_steps=min(full_unroll_steps, 3),
                                     eta_solver=eta_solver)
        evaluator(Vector{Float64}(starts[1, :]))
    end

    rows = NamedTuple[]
    for method in methods
        for s in 1:n_starts
            theta0 = Vector{Float64}(starts[s, :])
            @printf("\n[%s] start %d/%d\n", method, s, n_starts)
            result, cache, wall, cpu = optimize_one(method, subjects, theta0, lo, hi;
                                                    maxiter_eta=maxiter_eta,
                                                    full_unroll_steps=full_unroll_steps,
                                                    maxiter_outer=maxiter_outer,
                                                    eta_solver=eta_solver)
            theta_hat = Vector{Float64}(Optim.minimizer(result))
            ad_eval = safe_ad_population_value(subjects, theta_hat; maxiter_eta=maxiter_eta,
                                               eta_solver=eta_solver)
            stop_eval = ad_eval
            row = (
                model="flipflop",
                implementation="julia_forward_eta",
                method=method,
                start_id=s - 1,
                n_subjects=n_subj,
                maxiter_eta=maxiter_eta,
                eta_solver=string(eta_solver),
                full_unroll_steps=full_unroll_steps,
                outer_iterations=Optim.iterations(result),
                success=Optim.converged(result) && !cache.failed && isfinite(ad_eval),
                objective=cache.f,
                objective_ad_eval=ad_eval,
                objective_stop_eval=stop_eval,
                wall_sec=wall,
                cpu_sec=cpu,
                method_eval_count=cache.evals,
                max_eta_grad_norm=cache.max_eta_grad,
                n_eta_converged=cache.n_converged,
                theta=theta_hat,
            )
            push!(rows, row)
            write_results(outpath, rows)
            @printf("  success=%s objective=%.6f ad_eval=%.6f wall=%.3f cpu=%.3f evals=%d\n",
                    string(row.success), row.objective, row.objective_ad_eval,
                    row.wall_sec, row.cpu_sec, row.method_eval_count)
        end
    end
    println("\nSaved ", length(rows), " rows to ", outpath)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
