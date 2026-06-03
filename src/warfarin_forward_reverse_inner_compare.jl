using LinearAlgebra
using Printf
using Statistics
using ForwardDiff
using ReverseDiff

const ETA_DIM = 6
const BIG = 1.0e12

struct SubjectData
    sid::Int
    dose_mg::Float64
    pk_times::Vector{Float64}
    pk_obs::Vector{Float64}
    pd_times::Vector{Float64}
    pd_obs::Vector{Float64}
end

function parse_warfarin_csv(path::AbstractString)
    lines = readlines(path)
    header = split(strip(lines[1]), ",")
    col = Dict(name => i for (i, name) in enumerate(header))

    ids = Set{Int}()
    dose = Dict{Int, Float64}()
    pk_times = Dict{Int, Vector{Float64}}()
    pk_obs = Dict{Int, Vector{Float64}}()
    pd_times = Dict{Int, Vector{Float64}}()
    pd_obs = Dict{Int, Vector{Float64}}()

    for line in lines[2:end]
        isempty(strip(line)) && continue
        f = split(strip(line), ",")
        sid = parse(Int, f[col["id"]])
        time = parse(Float64, f[col["time"]])
        amt = parse(Float64, f[col["amt"]])
        dv = parse(Float64, f[col["dv"]])
        dvid = parse(Int, f[col["dvid"]])
        evid = parse(Int, f[col["evid"]])
        push!(ids, sid)

        if evid == 1
            dose[sid] = amt
        elseif evid == 0 && dvid == 1
            push!(get!(pk_times, sid, Float64[]), time)
            push!(get!(pk_obs, sid, Float64[]), dv)
        elseif evid == 0 && dvid == 2
            push!(get!(pd_times, sid, Float64[]), time)
            push!(get!(pd_obs, sid, Float64[]), dv)
        end
    end

    subjects = SubjectData[]
    for sid in sort(collect(ids))
        push!(
            subjects,
            SubjectData(
                sid,
                get(dose, sid, 0.0),
                get(pk_times, sid, Float64[]),
                get(pk_obs, sid, Float64[]),
                get(pd_times, sid, Float64[]),
                get(pd_obs, sid, Float64[]),
            ),
        )
    end
    return subjects
end

sigmoid(z) = one(z) / (one(z) + exp(-z))

function phi_exprel_neg(x)
    ax = abs(x)
    if ax < 1.0e-8
        return one(x) - x / 2 + x^2 / 6 - x^3 / 24
    end
    return -expm1(-x) / x
end

function diffexp_over_diff(a, b, t)
    tt = max(t, 0.0)
    return exp(-a * tt) * tt * phi_exprel_neg((b - a) * tt)
end

function pk_conc_oral_1c(t, dose, ka, cl, v)
    k = cl / v
    return (dose / v) * ka * diffexp_over_diff(k, ka, t)
end

function ce_closed_form(t, dose, ka, cl, v, ke0)
    k = cl / v
    term1 = diffexp_over_diff(k, ke0, t)
    term2 = diffexp_over_diff(ka, ke0, t)
    return dose * ka * ke0 * (term1 - term2) / (v * (ka - k))
end

function ce_ode(times::Vector{Float64}, dose, ka, cl, v, ke0; dt_max::Float64=0.25)
    ce = zero(ka + cl + v + ke0 + dose)
    current_t = 0.0
    out = typeof(ce)[]

    rhs(tt, ce_val) = ke0 * (pk_conc_oral_1c(tt, dose, ka, cl, v) - ce_val)

    for target in times
        interval = max(target - current_t, 0.0)
        if interval > 0.0
            n_steps = max(1, ceil(Int, interval / dt_max))
            h = interval / n_steps
            for _ in 1:n_steps
                half_h = h / 2
                k1 = rhs(current_t, ce)
                k2 = rhs(current_t + half_h, ce + half_h * k1)
                k3 = rhs(current_t + half_h, ce + half_h * k2)
                k4 = rhs(current_t + h, ce + h * k3)
                ce = ce + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
                current_t += h
            end
        end
        push!(out, ce)
    end
    return out
end

function predict_pk(subj::SubjectData, x, eta)
    ka = exp(x[1] + eta[1])
    cl = exp(x[2] + eta[2])
    v = exp(x[3] + eta[3])
    dose = typeof(ka + cl + v)(subj.dose_mg)
    return [pk_conc_oral_1c(t, dose, ka, cl, v) for t in subj.pk_times]
end

function predict_pd(subj::SubjectData, x, eta, representation::Symbol; dt::Float64=0.25)
    ka = exp(x[1] + eta[1])
    cl = exp(x[2] + eta[2])
    v = exp(x[3] + eta[3])
    e0 = exp(x[4] + eta[4])
    c50 = exp(x[5] + eta[5])
    ke0 = exp(x[6] + eta[6])
    emax = sigmoid(x[7])
    dose = typeof(ka + cl + v + e0 + c50 + ke0)(subj.dose_mg)

    ce = if representation == :closed_form
        [ce_closed_form(t, dose, ka, cl, v, ke0) for t in subj.pd_times]
    elseif representation == :ode
        ce_ode(subj.pd_times, dose, ka, cl, v, ke0; dt_max=dt)
    else
        error("unknown representation: $representation")
    end

    return [e0 * (one(c) - emax * c / (c50 + c + 1.0e-12)) for c in ce]
end

function h_i(subj::SubjectData, x, eta, representation::Symbol; dt::Float64=0.25)
    sigma_pk = exp(x[8])
    sigma_pd = exp(x[9])
    omega = exp.(x[10:15])

    total = zero(sigma_pk + sigma_pd + eta[1])
    pk_hat = predict_pk(subj, x, eta)
    total += length(subj.pk_obs) * log(sigma_pk)
    for (y, pred) in zip(subj.pk_obs, pk_hat)
        r = (y - pred) / (sigma_pk + 1.0e-12)
        total += 0.5 * r * r
    end

    pd_hat = predict_pd(subj, x, eta, representation; dt=dt)
    total += length(subj.pd_obs) * log(sigma_pd)
    for (y, pred) in zip(subj.pd_obs, pd_hat)
        r = (y - pred) / (sigma_pd + 1.0e-12)
        total += 0.5 * r * r
    end

    total += sum(log.(omega .+ 1.0e-12))
    for j in 1:ETA_DIM
        total += 0.5 * (eta[j] / (omega[j] + 1.0e-12))^2
    end
    return total
end

function grad_hess_forward(subj::SubjectData, x, eta::Vector{Float64}, representation::Symbol; dt::Float64=0.25)
    f = e -> h_i(subj, x, e, representation; dt=dt)
    return ForwardDiff.gradient(f, eta), ForwardDiff.hessian(f, eta), f(eta)
end

function grad_hess_reverse(subj::SubjectData, x, eta::Vector{Float64}, representation::Symbol; dt::Float64=0.25)
    f = e -> h_i(subj, x, e, representation; dt=dt)
    return ReverseDiff.gradient(f, eta), ReverseDiff.hessian(f, eta), f(eta)
end

function grad_hess(backend::Symbol, subj::SubjectData, x, eta::Vector{Float64}, representation::Symbol; dt::Float64=0.25)
    if backend == :forward
        return grad_hess_forward(subj, x, eta, representation; dt=dt)
    elseif backend == :reverse
        return grad_hess_reverse(subj, x, eta, representation; dt=dt)
    end
    error("unknown backend: $backend")
end

function safe_solve(H, g; jitter::Float64=1.0e-6)
    q = length(g)
    A0 = Matrix{Float64}(0.5 .* (H .+ transpose(H)))
    Iq = Matrix{Float64}(I, q, q)
    min_eig = minimum(eigvals(Symmetric(A0)))
    damp = max(jitter, jitter - min_eig)
    for _ in 1:12
        try
            step = (A0 .+ damp .* Iq) \ Vector{Float64}(g)
            if all(isfinite, step)
                nrm = norm(step)
                if nrm > 3.0
                    step .*= 3.0 / (nrm + 1.0e-12)
                end
                return step
            end
        catch
        end
        damp *= 10.0
    end
    return zeros(q)
end

function eta_mode_newton(subj::SubjectData, x::Vector{Float64}, backend::Symbol,
                         representation::Symbol; dt::Float64=0.25,
                         maxiter::Int=30, tol::Float64=1.0e-7,
                         eta0::AbstractVector{<:Real}=zeros(ETA_DIM))
    eta = Float64.(eta0)
    f_curr = h_i(subj, x, eta, representation; dt=dt)
    converged = false
    grad_norm = Inf

    for _ in 1:maxiter
        g, H, f0 = grad_hess(backend, subj, x, eta, representation; dt=dt)
        grad_norm = norm(g)
        if !isfinite(f0) || any(!isfinite, g) || any(!isfinite, H)
            break
        end
        if grad_norm < tol
            converged = true
            f_curr = f0
            break
        end

        step = safe_solve(H, g)
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
        accepted || break
    end

    return eta, Float64(f_curr), grad_norm, converged
end

function safe_logdet(H; jitter::Float64=1.0e-8)
    q = size(H, 1)
    Hs = Matrix{Float64}(0.5 .* (H .+ transpose(H)))
    Iq = Matrix{Float64}(I, q, q)
    min_eig = minimum(eigvals(Symmetric(Hs)))
    damp = max(jitter, jitter - min_eig)
    for _ in 1:12
        sign, ld = logabsdet(Hs .+ damp .* Iq)
        if sign > 0 && isfinite(ld)
            return ld
        end
        damp *= 10.0
    end
    return log(BIG)
end

function laplace_subject(subj::SubjectData, x::Vector{Float64}, eta::Vector{Float64},
                         backend::Symbol, representation::Symbol; dt::Float64=0.25)
    _, H, hi = grad_hess(backend, subj, x, eta, representation; dt=dt)
    return 2.0 * (hi + 0.5 * safe_logdet(H))
end

function population_laplace(subjects::Vector{SubjectData}, x::Vector{Float64}, backend::Symbol,
                            representation::Symbol; dt::Float64=0.25, maxiter::Int=30)
    etas = Vector{Vector{Float64}}(undef, length(subjects))
    total = 0.0
    max_grad_norm = 0.0
    n_converged = 0
    for (i, subj) in enumerate(subjects)
        eta, _, grad_norm, converged = eta_mode_newton(subj, x, backend, representation; dt=dt, maxiter=maxiter)
        etas[i] = eta
        max_grad_norm = max(max_grad_norm, grad_norm)
        n_converged += converged ? 1 : 0
        total += laplace_subject(subj, x, eta, backend, representation; dt=dt)
    end
    return total, etas, max_grad_norm, n_converged
end

function timed_seconds(f)
    GC.gc()
    t0 = time_ns()
    value = f()
    return value, (time_ns() - t0) / 1.0e9
end

max_eta_absdiff(a, b) = maximum(maximum(abs.(a[i] .- b[i])) for i in eachindex(a))

function write_rows(path::AbstractString, rows)
    open(path, "w") do io
        println(io, "representation,backend,n_subjects,maxiter_eta,dt,seconds,ofv,ofv_absdiff,max_eta_absdiff,max_eta_grad_norm,n_converged,agreement_pass")
        for r in rows
            @printf(io, "%s,%s,%d,%d,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%d,%s\n",
                    r.representation, r.backend, r.n_subjects, r.maxiter_eta, r.dt, r.seconds,
                    r.ofv, r.ofv_absdiff, r.max_eta_absdiff, r.max_eta_grad_norm,
                    r.n_converged, string(r.agreement_pass))
        end
    end
end

function write_summary(path::AbstractString, rows)
    open(path, "w") do io
        println(io, "representation,forward_seconds,reverse_seconds,reverse_over_forward_speed_ratio,ofv_absdiff,max_eta_absdiff,forward_max_eta_grad_norm,reverse_max_eta_grad_norm,agreement_pass")
        for rep in unique([r.representation for r in rows])
            f = first(r for r in rows if r.representation == rep && r.backend == "ForwardDiff")
            b = first(r for r in rows if r.representation == rep && r.backend == "ReverseDiff")
            @printf(io, "%s,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%s\n",
                    rep, f.seconds, b.seconds, b.seconds / f.seconds, b.ofv_absdiff,
                    b.max_eta_absdiff, f.max_eta_grad_norm, b.max_eta_grad_norm,
                    string(b.agreement_pass))
        end
    end
end

function parse_representations(raw::AbstractString)
    reps = Symbol[]
    for item in split(raw, ",")
        v = lowercase(strip(item))
        if v in ("closed", "closed_form", "closed-form", "analytic")
            push!(reps, :closed_form)
        elseif v in ("ode", "rk4")
            push!(reps, :ode)
        else
            error("unknown representation: $item")
        end
    end
    return reps
end

function main()
    data_path = joinpath(@__DIR__, "warfarin_dat.csv")
    subjects_all = parse_warfarin_csv(data_path)
    n_subjects = parse(Int, get(ENV, "WARFARIN_JULIA_N_SUBJ", string(length(subjects_all))))
    subjects = subjects_all[1:min(n_subjects, length(subjects_all))]
    reps = parse_representations(get(ENV, "WARFARIN_JULIA_REPRESENTATIONS", "closed_form,ode"))
    maxiter = parse(Int, get(ENV, "WARFARIN_JULIA_MAXITER_ETA", "50"))
    dt = parse(Float64, get(ENV, "WARFARIN_JULIA_PD_DT", "0.25"))
    outdir = get(ENV, "WARFARIN_JULIA_OUTDIR", joinpath(@__DIR__, "tables"))
    mkpath(outdir)

    # Default to the closed-form Warfarin FULL-implicit single-start estimate.
    # This keeps the eta-mode comparison in a relevant part of parameter space.
    x0 = [
        -0.9210127726653524, -1.9722402766094584, 2.016933619247369,
        4.574268512106518, 0.404725533852134, -3.813983430255588,
        3.7660967865817736, 0.0551221879421654, 1.8605600006302243,
        -0.0145730171168364, -1.2286231256669593, -1.29962914164841,
        -3.727931923757764, -1.2210404901973697, -1.118180862203899,
    ]

    rows = NamedTuple[]
    println("Warfarin Julia inner-loop AD comparison")
    println("subjects=", length(subjects), " maxiter_eta=", maxiter, " dt=", dt)

    for rep in reps
        println("\nRepresentation: ", rep)

        # Compile both paths before the reported timings.
        population_laplace(subjects, x0, :forward, rep; dt=dt, maxiter=maxiter)
        population_laplace(subjects, x0, :reverse, rep; dt=dt, maxiter=maxiter)

        (fwd, sec_fwd) = timed_seconds() do
            population_laplace(subjects, x0, :forward, rep; dt=dt, maxiter=maxiter)
        end
        ofv_fwd, etas_fwd, grad_fwd, conv_fwd = fwd
        push!(rows, (
            representation=String(rep),
            backend="ForwardDiff",
            n_subjects=length(subjects),
            maxiter_eta=maxiter,
            dt=dt,
            seconds=sec_fwd,
            ofv=ofv_fwd,
            ofv_absdiff=0.0,
            max_eta_absdiff=0.0,
            max_eta_grad_norm=grad_fwd,
            n_converged=conv_fwd,
            agreement_pass=true,
        ))

        (rev, sec_rev) = timed_seconds() do
            population_laplace(subjects, x0, :reverse, rep; dt=dt, maxiter=maxiter)
        end
        ofv_rev, etas_rev, grad_rev, conv_rev = rev
        ofv_diff = abs(ofv_rev - ofv_fwd)
        eta_diff = max_eta_absdiff(etas_fwd, etas_rev)
        pass = ofv_diff <= 1.0e-7 && eta_diff <= 1.0e-7
        push!(rows, (
            representation=String(rep),
            backend="ReverseDiff",
            n_subjects=length(subjects),
            maxiter_eta=maxiter,
            dt=dt,
            seconds=sec_rev,
            ofv=ofv_rev,
            ofv_absdiff=ofv_diff,
            max_eta_absdiff=eta_diff,
            max_eta_grad_norm=grad_rev,
            n_converged=conv_rev,
            agreement_pass=pass,
        ))

        @printf("  ForwardDiff OFV=%14.6f time=%9.4f s max|g_eta|=%.3e converged=%d/%d\n",
                ofv_fwd, sec_fwd, grad_fwd, conv_fwd, length(subjects))
        @printf("  ReverseDiff OFV=%14.6f time=%9.4f s ofv_diff=%.3e eta_diff=%.3e max|g_eta|=%.3e pass=%s\n",
                ofv_rev, sec_rev, ofv_diff, eta_diff, grad_rev, string(pass))
    end

    detail_path = joinpath(outdir, "warfarin_forward_reverse_inner_compare.csv")
    summary_path = joinpath(outdir, "warfarin_forward_reverse_inner_summary.csv")
    write_rows(detail_path, rows)
    write_summary(summary_path, rows)
    println("\nWrote: ", detail_path)
    println("Wrote: ", summary_path)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
