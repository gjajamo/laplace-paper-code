# Experimental NONMEM-like sensitivity-equation path.
# This uses ODE sensitivities for eta derivatives and finite differences only
# for the outer population-parameter gradient.

const FLIPFLOP_SENS_DT_DEFAULT = 0.01

function flipflop_sens_index_s(state::Int, a::Int)
    return 2 + (a - 1) * 2 + state
end

function flipflop_sens_index_t(state::Int, a::Int, b::Int)
    return 2 + 2 * ETA_DIM + ((a - 1) * ETA_DIM + (b - 1)) * 2 + state
end

function flipflop_sens_rhs(z::Vector{Float64}, theta::Vector{Float64}, eta::Vector{Float64})
    logka, logcl, logv, _, _, _ = theta
    ka = exp(logka + eta[1])
    cl = exp(logcl)
    v = exp(logv + eta[2])
    k = cl / v

    dz = zeros(length(z))
    a1 = z[1]
    a2 = z[2]
    dz[1] = -ka * a1
    dz[2] = ka * a1 - k * a2

    function fy_mul(s1, s2)
        return -ka * s1, ka * s1 - k * s2
    end
    function fyp_mul(p::Int, s1, s2)
        if p == 1
            return -ka * s1, ka * s1
        elseif p == 2
            return 0.0, k * s2
        end
        return 0.0, 0.0
    end
    function fp(p::Int)
        if p == 1
            return -ka * a1, ka * a1
        elseif p == 2
            return 0.0, k * a2
        end
        return 0.0, 0.0
    end
    function fpp(a::Int, b::Int)
        if a == 1 && b == 1
            return -ka * a1, ka * a1
        elseif a == 2 && b == 2
            return 0.0, -k * a2
        end
        return 0.0, 0.0
    end

    for a in 1:ETA_DIM
        s1 = z[flipflop_sens_index_s(1, a)]
        s2 = z[flipflop_sens_index_s(2, a)]
        y1, y2 = fy_mul(s1, s2)
        p1, p2 = fp(a)
        dz[flipflop_sens_index_s(1, a)] = y1 + p1
        dz[flipflop_sens_index_s(2, a)] = y2 + p2
    end

    for a in 1:ETA_DIM, b in 1:ETA_DIM
        t1 = z[flipflop_sens_index_t(1, a, b)]
        t2 = z[flipflop_sens_index_t(2, a, b)]
        y1, y2 = fy_mul(t1, t2)

        sa1 = z[flipflop_sens_index_s(1, a)]
        sa2 = z[flipflop_sens_index_s(2, a)]
        sb1 = z[flipflop_sens_index_s(1, b)]
        sb2 = z[flipflop_sens_index_s(2, b)]
        ypa1, ypa2 = fyp_mul(a, sb1, sb2)
        ypb1, ypb2 = fyp_mul(b, sa1, sa2)
        pp1, pp2 = fpp(a, b)

        dz[flipflop_sens_index_t(1, a, b)] = y1 + ypa1 + ypb1 + pp1
        dz[flipflop_sens_index_t(2, a, b)] = y2 + ypa2 + ypb2 + pp2
    end

    return dz
end

function flipflop_sens_rk4_step(z::Vector{Float64}, h::Float64,
                                theta::Vector{Float64}, eta::Vector{Float64})
    k1 = flipflop_sens_rhs(z, theta, eta)
    k2 = flipflop_sens_rhs(z .+ 0.5 .* h .* k1, theta, eta)
    k3 = flipflop_sens_rhs(z .+ 0.5 .* h .* k2, theta, eta)
    k4 = flipflop_sens_rhs(z .+ h .* k3, theta, eta)
    return z .+ (h / 6.0) .* (k1 .+ 2.0 .* k2 .+ 2.0 .* k3 .+ k4)
end

function flipflop_conc_sens_from_state(z::Vector{Float64}, theta::Vector{Float64}, eta::Vector{Float64})
    v = exp(theta[3] + eta[2])
    inv_v = 1.0 / v
    a2 = z[2]
    pred = a2 * inv_v
    grad = zeros(ETA_DIM)
    hess = zeros(ETA_DIM, ETA_DIM)
    for a in 1:ETA_DIM
        dinv_a = a == 2 ? -inv_v : 0.0
        s2a = z[flipflop_sens_index_s(2, a)]
        grad[a] = s2a * inv_v + a2 * dinv_a
    end
    for a in 1:ETA_DIM, b in 1:ETA_DIM
        dinv_a = a == 2 ? -inv_v : 0.0
        dinv_b = b == 2 ? -inv_v : 0.0
        d2inv = (a == 2 && b == 2) ? inv_v : 0.0
        s2a = z[flipflop_sens_index_s(2, a)]
        s2b = z[flipflop_sens_index_s(2, b)]
        t2ab = z[flipflop_sens_index_t(2, a, b)]
        hess[a, b] = t2ab * inv_v + s2a * dinv_b + s2b * dinv_a + a2 * d2inv
    end
    return pred, grad, hess
end

function flipflop_sens_predictions(theta::Vector{Float64}, eta::Vector{Float64};
                                   dt::Float64=FLIPFLOP_SENS_DT_DEFAULT)
    nz = 2 + 2 * ETA_DIM + 2 * ETA_DIM * ETA_DIM
    z = zeros(nz)
    z[1] = DOSE
    current_t = 0.0
    preds = Float64[]
    grads = Vector{Float64}[]
    hesses = Matrix{Float64}[]
    for target in TIMES
        interval = max(target - current_t, 0.0)
        if interval > 0.0
            n_steps = max(1, ceil(Int, interval / dt))
            h = interval / n_steps
            for _ in 1:n_steps
                z = flipflop_sens_rk4_step(z, h, theta, eta)
                current_t += h
            end
        end
        pred, grad, hess = flipflop_conc_sens_from_state(z, theta, eta)
        push!(preds, pred)
        push!(grads, grad)
        push!(hesses, hess)
    end
    return preds, grads, hesses
end

function flipflop_sens_eta_grad_hess(subj::SubjectData, theta::Vector{Float64}, eta::Vector{Float64};
                                     dt::Float64=FLIPFLOP_SENS_DT_DEFAULT)
    _, _, _, logomega_ka, logomega_v, logsigma = theta
    omega = [exp(logomega_ka), exp(logomega_v)]
    sigma = exp(logsigma)
    preds, dp, d2p = flipflop_sens_predictions(theta, eta; dt=dt)

    hi = length(subj.y) * log(sigma) + sum(log.(omega))
    grad = zeros(ETA_DIM)
    H = zeros(ETA_DIM, ETA_DIM)
    sig2 = (sigma + 1.0e-12)^2
    for j in eachindex(subj.y)
        diff = preds[j] - subj.y[j]
        hi += 0.5 * diff^2 / sig2
        for a in 1:ETA_DIM
            grad[a] += diff * dp[j][a] / sig2
        end
        for a in 1:ETA_DIM, b in 1:ETA_DIM
            H[a, b] += (dp[j][a] * dp[j][b] + diff * d2p[j][a, b]) / sig2
        end
    end
    for a in 1:ETA_DIM
        hi += 0.5 * (eta[a] / omega[a])^2
        grad[a] += eta[a] / (omega[a]^2 + 1.0e-24)
        H[a, a] += 1.0 / (omega[a]^2 + 1.0e-24)
    end
    return grad, H, hi
end

function eta_mode_newton_sens(subj::SubjectData, theta::Vector{Float64};
                              maxiter::Int=20, tol::Float64=1.0e-8,
                              dt::Float64=FLIPFLOP_SENS_DT_DEFAULT)
    eta = zeros(ETA_DIM)
    grad_norm = Inf
    converged = false
    f_curr = Inf
    for _ in 1:maxiter
        g, H, f0 = flipflop_sens_eta_grad_hess(subj, theta, eta; dt=dt)
        grad_norm = norm(g)
        if !isfinite(f0) || any(!isfinite, g) || any(!isfinite, H)
            break
        end
        f_curr = f0
        if grad_norm < tol
            converged = true
            break
        end
        step = safe_newton_step(H, g)
        alpha = 1.0
        accepted = false
        for _ls in 1:12
            trial = eta .- alpha .* step
            _, _, f_try = flipflop_sens_eta_grad_hess(subj, theta, trial; dt=dt)
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

function solve_all_etas_sens(subjects::Vector{SubjectData}, theta::Vector{Float64};
                             maxiter::Int=20, dt::Float64=FLIPFLOP_SENS_DT_DEFAULT)
    etas = Vector{Vector{Float64}}(undef, length(subjects))
    grad_norms = zeros(length(subjects))
    converged = falses(length(subjects))
    @threads for i in eachindex(subjects)
        eta, _, gn, cvg = eta_mode_newton_sens(subjects[i], theta; maxiter=maxiter, dt=dt)
        etas[i] = eta
        grad_norms[i] = gn
        converged[i] = cvg
    end
    return etas, maximum(grad_norms), count(converged)
end

function laplace_subject_fixed_eta_sens(subj::SubjectData, theta::Vector{Float64}, eta::Vector{Float64};
                                        dt::Float64=FLIPFLOP_SENS_DT_DEFAULT)
    _, H, hi = flipflop_sens_eta_grad_hess(subj, theta, eta; dt=dt)
    return 2.0 * (hi + 0.5 * logdet_cholesky(H))
end

function sens_population_value(subjects::Vector{SubjectData}, theta::Vector{Float64};
                               maxiter_eta::Int=20, dt::Float64=FLIPFLOP_SENS_DT_DEFAULT)
    etas, max_eta_grad, n_conv = solve_all_etas_sens(subjects, theta; maxiter=maxiter_eta, dt=dt)
    vals = zeros(length(subjects))
    @threads for i in eachindex(subjects)
        vals[i] = laplace_subject_fixed_eta_sens(subjects[i], theta, etas[i]; dt=dt)
    end
    return sum(vals), max_eta_grad, n_conv
end

function sens_fd_value_grad(subjects::Vector{SubjectData}, theta::Vector{Float64};
                            maxiter_eta::Int=20,
                            eps::Float64=parse(Float64, get(ENV, "FLIPFLOP_JULIA_SENS_FD_EPS_THETA", "2e-4")),
                            dt::Float64=parse(Float64, get(ENV, "FLIPFLOP_JULIA_SENS_DT", string(FLIPFLOP_SENS_DT_DEFAULT))))
    f0, max_eta_grad, n_conv = sens_population_value(subjects, theta; maxiter_eta=maxiter_eta, dt=dt)
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
            f1 = sens_population_value(subjects, xone; maxiter_eta=maxiter_eta, dt=dt)[1]
            g[j] = isfinite(f1) ? (f1 - f0) / step : 0.0
        else
            fp = sens_population_value(subjects, xp; maxiter_eta=maxiter_eta, dt=dt)[1]
            fm = sens_population_value(subjects, xm; maxiter_eta=maxiter_eta, dt=dt)[1]
            g[j] = (isfinite(fp) && isfinite(fm)) ? (fp - fm) / (step_p + step_m) : 0.0
        end
    end
    return f0, g, max_eta_grad, n_conv
end

function flipflop_param_index_s(state::Int, k::Int)
    return 2 + (k - 1) * 2 + state
end

function flipflop_param_sens_rhs(z::Vector{Float64}, theta::Vector{Float64}, eta::Vector{Float64})
    p = length(theta)
    logka, logcl, logv, _, _, _ = theta
    ka = exp(logka + eta[1])
    cl = exp(logcl)
    v = exp(logv + eta[2])
    k_elim = cl / v

    dz = zeros(length(z))
    a1 = z[1]
    a2 = z[2]
    dz[1] = -ka * a1
    dz[2] = ka * a1 - k_elim * a2

    for j in 1:p
        s1 = z[flipflop_param_index_s(1, j)]
        s2 = z[flipflop_param_index_s(2, j)]
        fx1 = 0.0
        fx2 = 0.0
        if j == 1
            fx1 = -ka * a1
            fx2 = ka * a1
        elseif j == 2
            fx2 = -k_elim * a2
        elseif j == 3
            fx2 = k_elim * a2
        end
        dz[flipflop_param_index_s(1, j)] = -ka * s1 + fx1
        dz[flipflop_param_index_s(2, j)] = ka * s1 - k_elim * s2 + fx2
    end
    return dz
end

function flipflop_param_sens_rk4_step(z::Vector{Float64}, h::Float64,
                                      theta::Vector{Float64}, eta::Vector{Float64})
    k1 = flipflop_param_sens_rhs(z, theta, eta)
    k2 = flipflop_param_sens_rhs(z .+ 0.5 .* h .* k1, theta, eta)
    k3 = flipflop_param_sens_rhs(z .+ 0.5 .* h .* k2, theta, eta)
    k4 = flipflop_param_sens_rhs(z .+ h .* k3, theta, eta)
    return z .+ (h / 6.0) .* (k1 .+ 2.0 .* k2 .+ 2.0 .* k3 .+ k4)
end

function flipflop_param_predictions(theta::Vector{Float64}, eta::Vector{Float64};
                                    dt::Float64=FLIPFLOP_SENS_DT_DEFAULT)
    p = length(theta)
    z = zeros(2 + 2 * p)
    z[1] = DOSE
    current_t = 0.0
    preds = Float64[]
    grads = Vector{Float64}[]
    for target in TIMES
        interval = max(target - current_t, 0.0)
        if interval > 0.0
            n_steps = max(1, ceil(Int, interval / dt))
            h = interval / n_steps
            for _ in 1:n_steps
                z = flipflop_param_sens_rk4_step(z, h, theta, eta)
                current_t += h
            end
        end
        v = exp(theta[3] + eta[2])
        inv_v = 1.0 / v
        pred = z[2] * inv_v
        grad = zeros(p)
        for j in 1:p
            s2 = z[flipflop_param_index_s(2, j)]
            grad[j] = s2 * inv_v
        end
        grad[3] -= pred
        push!(preds, pred)
        push!(grads, grad)
    end
    return preds, grads
end

function flipflop_sens_hi_theta_grad(subj::SubjectData, theta::Vector{Float64}, eta::Vector{Float64};
                                     dt::Float64=FLIPFLOP_SENS_DT_DEFAULT)
    _, _, _, logomega_ka, logomega_v, logsigma = theta
    omega = [exp(logomega_ka), exp(logomega_v)]
    sigma = exp(logsigma)
    sig2 = (sigma + 1.0e-12)^2
    preds, dp = flipflop_param_predictions(theta, eta; dt=dt)
    grad = zeros(length(theta))
    for j in eachindex(subj.y)
        diff = preds[j] - subj.y[j]
        grad .+= (diff / sig2) .* dp[j]
        grad[6] -= diff^2 / sig2
    end
    grad[6] += length(subj.y)
    grad[4] += 1.0 - (eta[1] / (omega[1] + 1.0e-12))^2
    grad[5] += 1.0 - (eta[2] / (omega[2] + 1.0e-12))^2
    return grad
end

function flipflop_sens_logdet_theta_grad_fd(subj::SubjectData, theta::Vector{Float64},
                                            eta::Vector{Float64}, H0::Matrix{Float64};
                                            maxiter_eta::Int=20,
                                            dt::Float64=FLIPFLOP_SENS_DT_DEFAULT,
                                            eps::Float64=parse(Float64, get(ENV, "FLIPFLOP_JULIA_SENS_PARAM_HESS_EPS", "1e-4")))
    A = Matrix{Float64}(shifted_matrix(H0))
    Ainv = inv(A)
    grad = zeros(length(theta))
    lo, hi = theta_bounds()
    for j in eachindex(theta)
        xp = copy(theta)
        xm = copy(theta)
        h = eps * max(1.0, abs(theta[j]))
        xp[j] = min(max(theta[j] + h, lo[j]), hi[j])
        xm[j] = min(max(theta[j] - h, lo[j]), hi[j])
        step_p = xp[j] - theta[j]
        step_m = theta[j] - xm[j]
        if step_p == 0.0 && step_m == 0.0
            grad[j] = 0.0
        elseif step_p == 0.0 || step_m == 0.0
            xone = step_p == 0.0 ? xm : xp
            step = step_p == 0.0 ? -step_m : step_p
            _, Hone, _ = flipflop_sens_eta_grad_hess(subj, xone, eta; dt=dt)
            dH = (Hone .- H0) ./ step
            grad[j] = tr(Ainv * dH)
        else
            _, Hp, _ = flipflop_sens_eta_grad_hess(subj, xp, eta; dt=dt)
            _, Hm, _ = flipflop_sens_eta_grad_hess(subj, xm, eta; dt=dt)
            dH = (Hp .- Hm) ./ (step_p + step_m)
            grad[j] = tr(Ainv * dH)
        end
    end
    return grad
end

function sens_param_subject_value_grad(subj::SubjectData, theta::Vector{Float64}, eta::Vector{Float64};
                                       maxiter_eta::Int=20,
                                       dt::Float64=parse(Float64, get(ENV, "FLIPFLOP_JULIA_SENS_DT", string(FLIPFLOP_SENS_DT_DEFAULT))))
    _, H, hi = flipflop_sens_eta_grad_hess(subj, theta, eta; dt=dt)
    ld = logdet_cholesky(H)
    dh = flipflop_sens_hi_theta_grad(subj, theta, eta; dt=dt)
    dld = flipflop_sens_logdet_theta_grad_fd(subj, theta, eta, H; maxiter_eta=maxiter_eta, dt=dt)
    return 2.0 * (hi + 0.5 * ld), 2.0 .* (dh .+ 0.5 .* dld)
end

function sens_param_value_grad(subjects::Vector{SubjectData}, theta::Vector{Float64};
                               maxiter_eta::Int=20,
                               dt::Float64=parse(Float64, get(ENV, "FLIPFLOP_JULIA_SENS_DT", string(FLIPFLOP_SENS_DT_DEFAULT))))
    etas, max_eta_grad, n_conv = solve_all_etas_sens(subjects, theta; maxiter=maxiter_eta, dt=dt)
    vals = zeros(length(subjects))
    grads = [zeros(length(theta)) for _ in subjects]
    @threads for i in eachindex(subjects)
        vals[i], grads[i] = sens_param_subject_value_grad(subjects[i], theta, etas[i];
                                                          maxiter_eta=maxiter_eta, dt=dt)
    end
    return sum(vals), vec(sum(reduce(hcat, grads), dims=2)), max_eta_grad, n_conv
end
