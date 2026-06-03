# Experimental NONMEM-like sensitivity-equation path for the Warfarin ODE
# representation. The effect-compartment ODE is integrated together with
# first and second eta sensitivities; the outer population gradient is still
# finite-differenced, matching the intended diagnostic comparison.

function warfarin_sens_index_s(a::Int)
    return 1 + a
end

function warfarin_sens_index_t(a::Int, b::Int)
    return 1 + ETA_DIM + (a - 1) * ETA_DIM + b
end

function warfarin_pk_eta_derivs(subj::SubjectData, x::Vector{Float64},
                                eta::AbstractVector, t::Float64)
    f = e -> begin
        ka = exp(x[1] + e[1])
        cl = exp(x[2] + e[2])
        v = exp(x[3] + e[3])
        pk_conc_oral_1c(t, subj.dose_mg, ka, cl, v)
    end
    val = Float64(f(eta))
    grad = Vector{Float64}(ForwardDiff.gradient(f, eta))
    hess = Matrix{Float64}(ForwardDiff.hessian(f, eta))
    return val, grad, hess
end

function warfarin_ce_sens_rhs(z::Vector{Float64}, t::Float64, subj::SubjectData,
                              x::Vector{Float64}, eta::Vector{Float64})
    q = ETA_DIM
    ce = z[1]
    pk, dpk, d2pk = warfarin_pk_eta_derivs(subj, x, eta, t)
    ke0 = exp(x[6] + eta[6])
    dke0 = zeros(q)
    d2ke0 = zeros(q, q)
    dke0[6] = ke0
    d2ke0[6, 6] = ke0

    dz = zeros(length(z))
    dz[1] = ke0 * (pk - ce)
    for a in 1:q
        s_a = z[warfarin_sens_index_s(a)]
        f_a = dke0[a] * (pk - ce) + ke0 * dpk[a]
        dz[warfarin_sens_index_s(a)] = -ke0 * s_a + f_a
    end

    for a in 1:q, b in 1:q
        t_ab = z[warfarin_sens_index_t(a, b)]
        s_a = z[warfarin_sens_index_s(a)]
        s_b = z[warfarin_sens_index_s(b)]
        f_ab = d2ke0[a, b] * (pk - ce) + dke0[a] * dpk[b] +
               dke0[b] * dpk[a] + ke0 * d2pk[a, b]
        dz[warfarin_sens_index_t(a, b)] = -ke0 * t_ab - dke0[b] * s_a -
                                          dke0[a] * s_b + f_ab
    end
    return dz
end

function warfarin_ce_sens_rk4_step(z::Vector{Float64}, t::Float64, h::Float64,
                                   subj::SubjectData, x::Vector{Float64},
                                   eta::Vector{Float64})
    k1 = warfarin_ce_sens_rhs(z, t, subj, x, eta)
    k2 = warfarin_ce_sens_rhs(z .+ 0.5 .* h .* k1, t + 0.5 * h, subj, x, eta)
    k3 = warfarin_ce_sens_rhs(z .+ 0.5 .* h .* k2, t + 0.5 * h, subj, x, eta)
    k4 = warfarin_ce_sens_rhs(z .+ h .* k3, t + h, subj, x, eta)
    return z .+ (h / 6.0) .* (k1 .+ 2.0 .* k2 .+ 2.0 .* k3 .+ k4)
end

function warfarin_ce_sens_predictions(subj::SubjectData, x::Vector{Float64}, eta::Vector{Float64};
                                      dt::Float64=0.25)
    q = ETA_DIM
    z = zeros(1 + q + q * q)
    current_t = 0.0
    ce_vals = Float64[]
    dce_vals = Vector{Float64}[]
    d2ce_vals = Matrix{Float64}[]
    for target in subj.pd_times
        interval = max(target - current_t, 0.0)
        if interval > 0.0
            n_steps = max(1, ceil(Int, interval / dt))
            h = interval / n_steps
            for _ in 1:n_steps
                z = warfarin_ce_sens_rk4_step(z, current_t, h, subj, x, eta)
                current_t += h
            end
        end
        push!(ce_vals, z[1])
        push!(dce_vals, [z[warfarin_sens_index_s(a)] for a in 1:q])
        push!(d2ce_vals, [z[warfarin_sens_index_t(a, b)] for a in 1:q, b in 1:q])
    end
    return ce_vals, dce_vals, d2ce_vals
end

function warfarin_pd_from_ce_sens(x::Vector{Float64}, eta::Vector{Float64},
                                  ce::Float64, dce::Vector{Float64}, d2ce::Matrix{Float64})
    q = ETA_DIM
    e0 = exp(x[4] + eta[4])
    c50 = exp(x[5] + eta[5])
    emax = sigmoid(x[7])
    den = c50 + ce + 1.0e-12
    phi = 1.0 - emax * ce / den
    pred = e0 * phi

    y_E = phi
    y_q = e0 * emax * ce / den^2
    y_z = -e0 * emax * c50 / den^2
    y_Eq = emax * ce / den^2
    y_Ez = -emax * c50 / den^2
    y_qq = -2.0 * e0 * emax * ce / den^3
    y_qz = e0 * emax * (c50 - ce) / den^3
    y_zz = 2.0 * e0 * emax * c50 / den^3

    grad = zeros(q)
    hess = zeros(q, q)
    for a in 1:q
        dE_a = a == 4 ? e0 : 0.0
        dq_a = a == 5 ? c50 : 0.0
        grad[a] = y_E * dE_a + y_q * dq_a + y_z * dce[a]
    end
    for a in 1:q, b in 1:q
        dE_a = a == 4 ? e0 : 0.0
        dE_b = b == 4 ? e0 : 0.0
        d2E = (a == 4 && b == 4) ? e0 : 0.0
        dq_a = a == 5 ? c50 : 0.0
        dq_b = b == 5 ? c50 : 0.0
        d2q = (a == 5 && b == 5) ? c50 : 0.0
        hess[a, b] = y_E * d2E + y_q * d2q + y_z * d2ce[a, b] +
                     y_Eq * (dE_a * dq_b + dE_b * dq_a) +
                     y_Ez * (dE_a * dce[b] + dE_b * dce[a]) +
                     y_qq * dq_a * dq_b +
                     y_qz * (dq_a * dce[b] + dq_b * dce[a]) +
                     y_zz * dce[a] * dce[b]
    end
    return pred, grad, hess
end

function warfarin_sens_eta_grad_hess(subj::SubjectData, x::Vector{Float64}, eta::Vector{Float64};
                                     dt::Float64=0.25)
    q = ETA_DIM
    sigma_pk = exp(x[8])
    sigma_pd = exp(x[9])
    omega = exp.(x[10:15])
    hi = length(subj.pk_obs) * log(sigma_pk) + length(subj.pd_obs) * log(sigma_pd) +
         sum(log.(omega .+ 1.0e-12))
    grad = zeros(q)
    H = zeros(q, q)

    sig2_pk = (sigma_pk + 1.0e-12)^2
    for (obs, t) in zip(subj.pk_obs, subj.pk_times)
        pred, dp, d2p = warfarin_pk_eta_derivs(subj, x, eta, t)
        diff = pred - obs
        hi += 0.5 * diff^2 / sig2_pk
        for a in 1:q
            grad[a] += diff * dp[a] / sig2_pk
        end
        for a in 1:q, b in 1:q
            H[a, b] += (dp[a] * dp[b] + diff * d2p[a, b]) / sig2_pk
        end
    end

    ce_vals, dce_vals, d2ce_vals = warfarin_ce_sens_predictions(subj, x, eta; dt=dt)
    sig2_pd = (sigma_pd + 1.0e-12)^2
    for j in eachindex(subj.pd_obs)
        pred, dp, d2p = warfarin_pd_from_ce_sens(x, eta, ce_vals[j], dce_vals[j], d2ce_vals[j])
        diff = pred - subj.pd_obs[j]
        hi += 0.5 * diff^2 / sig2_pd
        for a in 1:q
            grad[a] += diff * dp[a] / sig2_pd
        end
        for a in 1:q, b in 1:q
            H[a, b] += (dp[a] * dp[b] + diff * d2p[a, b]) / sig2_pd
        end
    end

    for a in 1:q
        hi += 0.5 * (eta[a] / (omega[a] + 1.0e-12))^2
        grad[a] += eta[a] / ((omega[a] + 1.0e-12)^2)
        H[a, a] += 1.0 / ((omega[a] + 1.0e-12)^2)
    end
    return grad, H, hi
end

function eta_mode_newton_sens(subj::SubjectData, x::Vector{Float64};
                              dt::Float64=0.25, maxiter::Int=30, tol::Float64=1.0e-7,
                              eta0::AbstractVector{<:Real}=zeros(ETA_DIM))
    eta = Float64.(eta0)
    f_curr = Inf
    grad_norm = Inf
    converged = false
    for _ in 1:maxiter
        g, H, f0 = warfarin_sens_eta_grad_hess(subj, x, eta; dt=dt)
        grad_norm = norm(g)
        if !isfinite(f0) || any(!isfinite, g) || any(!isfinite, H)
            break
        end
        f_curr = f0
        if grad_norm < tol
            converged = true
            break
        end
        step = safe_solve(H, g)
        alpha = 1.0
        accepted = false
        for _ls in 1:14
            trial = eta .- alpha .* step
            _, _, f_try = warfarin_sens_eta_grad_hess(subj, x, trial; dt=dt)
            if isfinite(f_try) && f_try <= f0 + 1.0e-10
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

function solve_all_etas_sens(subjects::Vector{SubjectData}, x::Vector{Float64};
                             dt::Float64=0.25, maxiter::Int=30,
                             eta_cache::Union{Nothing,EtaWarmCache}=nothing)
    etas = Vector{Vector{Float64}}(undef, length(subjects))
    grad_norms = zeros(length(subjects))
    converged = falses(length(subjects))
    @threads for i in eachindex(subjects)
        eta0 = eta_start(eta_cache, subjects[i])
        eta, _, gn, cvg = eta_mode_newton_sens(subjects[i], x; dt=dt, maxiter=maxiter, eta0=eta0)
        etas[i] = eta
        grad_norms[i] = gn
        converged[i] = cvg
    end
    return etas, maximum(grad_norms), count(converged)
end

function laplace_subject_fixed_eta_sens(subj::SubjectData, x::Vector{Float64}, eta::Vector{Float64};
                                        dt::Float64=0.25)
    _, H, hi = warfarin_sens_eta_grad_hess(subj, x, eta; dt=dt)
    return 2.0 * (hi + 0.5 * logdet_cholesky_ad(H))
end

function sens_population_value_from_etas(subjects::Vector{SubjectData}, x::Vector{Float64}, etas;
                                         dt::Float64=0.25)
    vals = zeros(length(subjects))
    @threads for i in eachindex(subjects)
        vals[i] = laplace_subject_fixed_eta_sens(subjects[i], x, etas[i]; dt=dt)
    end
    return sum(vals)
end

function sens_population_value(subjects::Vector{SubjectData}, x::Vector{Float64};
                               dt::Float64=0.25, maxiter_eta::Int=30,
                               eta_cache::Union{Nothing,EtaWarmCache}=nothing)
    etas, max_eta_grad, n_conv = solve_all_etas_sens(subjects, x; dt=dt,
                                                     maxiter=maxiter_eta,
                                                     eta_cache=eta_cache)
    return sens_population_value_from_etas(subjects, x, etas; dt=dt), max_eta_grad, n_conv, etas
end

function sens_fd_value_grad(subjects::Vector{SubjectData}, x::Vector{Float64}, representation::Symbol;
                            dt::Float64=0.25, maxiter_eta::Int=30,
                            eps::Float64=parse(Float64, get(ENV, "WARFARIN_JULIA_SENS_FD_EPS_THETA", "1e-4")),
                            eta_cache::Union{Nothing,EtaWarmCache}=nothing)
    representation == :ode || error("SENS_FD is implemented for the Warfarin ODE representation")
    f0, max_eta_grad, n_conv, etas = sens_population_value(subjects, x; dt=dt,
                                                           maxiter_eta=maxiter_eta,
                                                           eta_cache=eta_cache)
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
            f1 = sens_population_value(subjects, xone; dt=dt, maxiter_eta=maxiter_eta,
                                       eta_cache=copy_eta_cache(base_cache))[1]
            g[j] = isfinite(f1) ? (f1 - f0) / step : 0.0
        else
            fp = sens_population_value(subjects, xp; dt=dt, maxiter_eta=maxiter_eta,
                                      eta_cache=copy_eta_cache(base_cache))[1]
            fm = sens_population_value(subjects, xm; dt=dt, maxiter_eta=maxiter_eta,
                                      eta_cache=copy_eta_cache(base_cache))[1]
            g[j] = (isfinite(fp) && isfinite(fm)) ? (fp - fm) / (step_p + step_m) : 0.0
        end
    end
    maybe_update_eta_cache!(eta_cache, subjects, etas, Float64(f0), g)
    return f0, g, max_eta_grad, n_conv
end

function warfarin_pk_x_derivs(subj::SubjectData, x::Vector{Float64}, eta::Vector{Float64}, t::Float64)
    f = z -> begin
        ka = exp(z[1] + eta[1])
        cl = exp(z[2] + eta[2])
        v = exp(z[3] + eta[3])
        pk_conc_oral_1c(t, subj.dose_mg, ka, cl, v)
    end
    z = x[1:3]
    val = Float64(f(z))
    g3 = ForwardDiff.gradient(f, z)
    grad = zeros(length(x))
    grad[1:3] .= Float64.(g3)
    return val, grad
end

function warfarin_ce_param_sens_rhs(z::Vector{Float64}, t::Float64, subj::SubjectData,
                                    x::Vector{Float64}, eta::Vector{Float64})
    p = length(x)
    ce = z[1]
    pk, dpk = warfarin_pk_x_derivs(subj, x, eta, t)
    ke0 = exp(x[6] + eta[6])
    dz = zeros(length(z))
    dz[1] = ke0 * (pk - ce)
    for j in 1:p
        dke0 = j == 6 ? ke0 : 0.0
        s = z[1 + j]
        dz[1 + j] = dke0 * (pk - ce) + ke0 * (dpk[j] - s)
    end
    return dz
end

function warfarin_ce_param_sens_rk4_step(z::Vector{Float64}, t::Float64, h::Float64,
                                         subj::SubjectData, x::Vector{Float64},
                                         eta::Vector{Float64})
    k1 = warfarin_ce_param_sens_rhs(z, t, subj, x, eta)
    k2 = warfarin_ce_param_sens_rhs(z .+ 0.5 .* h .* k1, t + 0.5 * h, subj, x, eta)
    k3 = warfarin_ce_param_sens_rhs(z .+ 0.5 .* h .* k2, t + 0.5 * h, subj, x, eta)
    k4 = warfarin_ce_param_sens_rhs(z .+ h .* k3, t + h, subj, x, eta)
    return z .+ (h / 6.0) .* (k1 .+ 2.0 .* k2 .+ 2.0 .* k3 .+ k4)
end

function warfarin_ce_param_predictions(subj::SubjectData, x::Vector{Float64}, eta::Vector{Float64};
                                       dt::Float64=0.25)
    p = length(x)
    z = zeros(1 + p)
    current_t = 0.0
    ce_vals = Float64[]
    dce_vals = Vector{Float64}[]
    for target in subj.pd_times
        interval = max(target - current_t, 0.0)
        if interval > 0.0
            n_steps = max(1, ceil(Int, interval / dt))
            h = interval / n_steps
            for _ in 1:n_steps
                z = warfarin_ce_param_sens_rk4_step(z, current_t, h, subj, x, eta)
                current_t += h
            end
        end
        push!(ce_vals, z[1])
        push!(dce_vals, [z[1 + j] for j in 1:p])
    end
    return ce_vals, dce_vals
end

function warfarin_pd_param_grad(x::Vector{Float64}, eta::Vector{Float64},
                                ce::Float64, dce::Vector{Float64})
    p = length(x)
    e0 = exp(x[4] + eta[4])
    c50 = exp(x[5] + eta[5])
    emax = sigmoid(x[7])
    den = c50 + ce + 1.0e-12
    phi = 1.0 - emax * ce / den
    y_E = phi
    y_q = e0 * emax * ce / den^2
    y_m = -e0 * ce / den
    y_z = -e0 * emax * c50 / den^2
    grad = zeros(p)
    for j in 1:p
        dE = j == 4 ? e0 : 0.0
        dq = j == 5 ? c50 : 0.0
        dm = j == 7 ? emax * (1.0 - emax) : 0.0
        grad[j] = y_E * dE + y_q * dq + y_m * dm + y_z * dce[j]
    end
    return grad
end

function warfarin_sens_hi_x_grad(subj::SubjectData, x::Vector{Float64}, eta::Vector{Float64},
                                 representation::Symbol; dt::Float64=0.25)
    representation == :ode || error("SENS_PARAM is implemented for the Warfarin ODE representation")
    p = length(x)
    sigma_pk = exp(x[8])
    sigma_pd = exp(x[9])
    omega = exp.(x[10:15])
    grad = zeros(p)
    sig2_pk = (sigma_pk + 1.0e-12)^2
    for (obs, t) in zip(subj.pk_obs, subj.pk_times)
        pred, dp = warfarin_pk_x_derivs(subj, x, eta, t)
        diff = pred - obs
        grad .+= (diff / sig2_pk) .* dp
        grad[8] -= diff^2 / sig2_pk
    end
    grad[8] += length(subj.pk_obs)

    ce_vals, dce_vals = warfarin_ce_param_predictions(subj, x, eta; dt=dt)
    sig2_pd = (sigma_pd + 1.0e-12)^2
    for j in eachindex(subj.pd_obs)
        ce = ce_vals[j]
        e0 = exp(x[4] + eta[4])
        c50 = exp(x[5] + eta[5])
        emax = sigmoid(x[7])
        pred = e0 * (1.0 - emax * ce / (c50 + ce + 1.0e-12))
        dp = warfarin_pd_param_grad(x, eta, ce, dce_vals[j])
        diff = pred - subj.pd_obs[j]
        grad .+= (diff / sig2_pd) .* dp
        grad[9] -= diff^2 / sig2_pd
    end
    grad[9] += length(subj.pd_obs)

    for a in 1:ETA_DIM
        grad[9 + a] += 1.0 - (eta[a] / (omega[a] + 1.0e-12))^2
    end
    return grad
end

function warfarin_sens_logdet_x_grad_fd(subj::SubjectData, x::Vector{Float64}, eta::Vector{Float64},
                                        representation::Symbol, H0::Matrix{Float64};
                                        dt::Float64=0.25, maxiter_eta::Int=30,
                                        eps::Float64=parse(Float64, get(ENV, "WARFARIN_JULIA_SENS_PARAM_HESS_EPS", "1e-4")))
    representation == :ode || error("SENS_PARAM is implemented for the Warfarin ODE representation")
    A = Matrix{Float64}(shifted_matrix_ad(H0))
    Ainv = inv(A)
    grad = zeros(length(x))
    lo, hi = theta_bounds()
    for j in eachindex(x)
        xp = copy(x)
        xm = copy(x)
        h = eps * max(1.0, abs(x[j]))
        xp[j] = min(max(x[j] + h, lo[j]), hi[j])
        xm[j] = min(max(x[j] - h, lo[j]), hi[j])
        step_p = xp[j] - x[j]
        step_m = x[j] - xm[j]
        if step_p == 0.0 && step_m == 0.0
            grad[j] = 0.0
        elseif step_p == 0.0 || step_m == 0.0
            xone = step_p == 0.0 ? xm : xp
            step = step_p == 0.0 ? -step_m : step_p
            _, Hone, _ = warfarin_sens_eta_grad_hess(subj, xone, eta; dt=dt)
            dH = (Hone .- H0) ./ step
            grad[j] = tr(Ainv * dH)
        else
            _, Hp, _ = warfarin_sens_eta_grad_hess(subj, xp, eta; dt=dt)
            _, Hm, _ = warfarin_sens_eta_grad_hess(subj, xm, eta; dt=dt)
            dH = (Hp .- Hm) ./ (step_p + step_m)
            grad[j] = tr(Ainv * dH)
        end
    end
    return grad
end

function sens_param_subject_value_grad(subj::SubjectData, x::Vector{Float64}, eta::Vector{Float64},
                                       representation::Symbol; dt::Float64=0.25,
                                       maxiter_eta::Int=30)
    _, H, hi = warfarin_sens_eta_grad_hess(subj, x, eta; dt=dt)
    ld = logdet_cholesky_ad(H)
    dh = warfarin_sens_hi_x_grad(subj, x, eta, representation; dt=dt)
    dld = warfarin_sens_logdet_x_grad_fd(subj, x, eta, representation, H; dt=dt,
                                         maxiter_eta=maxiter_eta)
    return 2.0 * (hi + 0.5 * ld), 2.0 .* (dh .+ 0.5 .* dld)
end

function sens_param_value_grad(subjects::Vector{SubjectData}, x::Vector{Float64}, representation::Symbol;
                               dt::Float64=0.25, maxiter_eta::Int=30,
                               eta_cache::Union{Nothing,EtaWarmCache}=nothing)
    representation == :ode || error("SENS_PARAM is implemented for the Warfarin ODE representation")
    etas, max_eta_grad, n_conv = solve_all_etas_sens(subjects, x; dt=dt,
                                                     maxiter=maxiter_eta,
                                                     eta_cache=eta_cache)
    vals = zeros(length(subjects))
    grads = [zeros(length(x)) for _ in subjects]
    @threads for i in eachindex(subjects)
        vals[i], grads[i] = sens_param_subject_value_grad(subjects[i], x, etas[i], representation;
                                                          dt=dt, maxiter_eta=maxiter_eta)
    end
    total = sum(vals)
    total_grad = vec(sum(reduce(hcat, grads), dims=2))
    maybe_update_eta_cache!(eta_cache, subjects, etas, Float64(total), total_grad)
    return total, total_grad, max_eta_grad, n_conv
end
