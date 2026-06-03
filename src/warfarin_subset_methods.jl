include("warfarin_multistart_methods.jl")

using Printf

function read_start_bank(path::AbstractString)
    lines = readlines(path)
    length(lines) > 1 || error("start bank has no rows: $path")
    header = split(lines[1], ",")
    start_idx = findfirst(==("start_id"), header)
    start_idx === nothing && error("start_id column missing in $path")
    param_indices = [findfirst(==(name), header) for name in PARAM_NAMES]
    any(isnothing, param_indices) && error("one or more parameter columns missing in $path")
    starts = Dict{Int, Vector{Float64}}()
    for line in lines[2:end]
        isempty(strip(line)) && continue
        parts = split(line, ",")
        sid = parse(Int, parts[start_idx])
        starts[sid] = [parse(Float64, parts[i]) for i in param_indices]
    end
    return starts
end

function parse_start_ids(s::String, available_ids)
    ss = strip(s)
    if isempty(ss) || lowercase(ss) == "all"
        return sort(collect(available_ids))
    end
    return [parse(Int, strip(x)) for x in split(ss, ",") if !isempty(strip(x))]
end

function subset_main()
    root = normpath(joinpath(@__DIR__, ".."))
    data_path = get(ENV, "WARFARIN_JULIA_DATA", joinpath(@__DIR__, "warfarin_dat.csv"))
    subjects_all = parse_warfarin_csv(data_path)
    n_subjects = parse(Int, get(ENV, "WARFARIN_JULIA_N_SUBJ", string(length(subjects_all))))
    subjects = subjects_all[1:min(n_subjects, length(subjects_all))]

    maxiter_eta = parse(Int, get(ENV, "WARFARIN_JULIA_MAXITER_ETA", "30"))
    maxiter_outer = parse(Int, get(ENV, "WARFARIN_JULIA_MAXITER_OUTER", "50"))
    full_unroll_steps = parse(Int, get(ENV, "WARFARIN_JULIA_FULL_UNROLL_STEPS", string(maxiter_eta)))
    dt = parse(Float64, get(ENV, "WARFARIN_JULIA_DT", "0.25"))
    methods = split_tokens(get(ENV, "WARFARIN_JULIA_METHODS", "FULL_IMPLICIT,FULL_UNROLL,STOP,FD"))
    reps = [parse_rep(x) for x in split_tokens(get(ENV, "WARFARIN_JULIA_REPRESENTATIONS", "ode"))]
    outdir = get(ENV, "WARFARIN_JULIA_OUTDIR", joinpath(@__DIR__, "tables", "warfarin_subset"))
    mkpath(outdir)
    outpath = joinpath(outdir, "warfarin_julia_multistart_methods.csv")
    full_unroll_auto_retry = lowercase(get(ENV, "WARFARIN_JULIA_FULL_UNROLL_AUTO_RETRY", "true")) in ("1", "true", "yes")
    full_unroll_retry_delta = parse(Float64, get(ENV, "WARFARIN_JULIA_FULL_UNROLL_RETRY_DELTA", "0.5"))

    default_start_bank = joinpath(@__DIR__, "tables", "warfarin_ode_10starts_allmethods_20260522", "warfarin_julia_start_bank.csv")
    start_bank_path = get(ENV, "WARFARIN_JULIA_START_BANK", default_start_bank)
    starts = read_start_bank(start_bank_path)
    start_ids = parse_start_ids(get(ENV, "WARFARIN_JULIA_START_IDS", "all"), keys(starts))

    lo, hi = theta_bounds()
    println("Warfarin Julia subset methods")
    println("threads=", nthreads(), " subjects=", length(subjects))
    println("representations=", join(string.(reps), ","), " methods=", join(methods, ","))
    println("start_ids=", join(start_ids, ","), " start_bank=", start_bank_path)
    println("maxiter_eta=", maxiter_eta, " full_unroll_steps=", full_unroll_steps,
            " maxiter_outer=", maxiter_outer, " dt=", dt)
    println("output=", outpath)

    for rep in reps, method in methods
        println("warming ", rep, " / ", method)
        _, first_method, second_method, _, _ = stage_plan(method, maxiter_outer)
        warm_methods = isempty(first_method) ? [canonical_method(method)] : [first_method, second_method]
        for warm_method in warm_methods
            evaluator = method_evaluator(warm_method, subjects, rep; dt=dt,
                                         maxiter_eta=min(maxiter_eta, 2),
                                         full_unroll_steps=min(full_unroll_steps, 2))
            evaluator(copy(starts[start_ids[1]]))
        end
    end

    rows = NamedTuple[]
    for rep in reps
        for method in methods
            for sid in start_ids
                theta0 = copy(starts[sid])
                @printf("\n[%s/%s] start_id %d\n", string(rep), method, sid)
                outcome = optimize_method(method, subjects, rep, theta0, lo, hi;
                                          dt=dt, maxiter_eta=maxiter_eta,
                                          full_unroll_steps=full_unroll_steps,
                                          maxiter_outer=maxiter_outer)
                theta_hat = outcome.theta
                ad_eval = safe_ad_population_value(subjects, theta_hat, rep; dt=dt, maxiter_eta=maxiter_eta)
                if full_unroll_auto_retry && canonical_method(method) == "FULL_UNROLL"
                    current_linesearch = lowercase(get(ENV, method_env_key("FULL_UNROLL", "LINESEARCH"),
                                                       get(ENV, "WARFARIN_JULIA_LINESEARCH", "hagerzhang")))
                    mismatch = !isfinite(ad_eval) || !isfinite(outcome.objective) ||
                               abs(ad_eval - outcome.objective) > full_unroll_retry_delta
                    if current_linesearch != "backtracking" && mismatch
                        @printf("  FULL_UNROLL retry: ad_eval-objective mismatch %.6g, trying backtracking\n",
                                ad_eval - outcome.objective)
                        retry = with_env_value(method_env_key("FULL_UNROLL", "LINESEARCH"), "backtracking") do
                            optimize_method(method, subjects, rep, theta0, lo, hi;
                                            dt=dt, maxiter_eta=maxiter_eta,
                                            full_unroll_steps=full_unroll_steps,
                                            maxiter_outer=maxiter_outer)
                        end
                        retry_ad_eval = safe_ad_population_value(subjects, retry.theta, rep; dt=dt,
                                                                 maxiter_eta=maxiter_eta)
                        use_retry = isfinite(retry_ad_eval) && (!isfinite(ad_eval) || retry_ad_eval < ad_eval)
                        outcome = account_full_unroll_retry(outcome, retry; use_retry=use_retry)
                        theta_hat = outcome.theta
                        ad_eval = use_retry ? retry_ad_eval : ad_eval
                    end
                end
                row = (
                    model="warfarin",
                    implementation="julia_forward_eta",
                    representation=string(rep),
                    method=outcome.method,
                    start_id=sid,
                    n_subjects=length(subjects),
                    maxiter_eta=maxiter_eta,
                    full_unroll_steps=full_unroll_steps,
                    outer_iterations=outcome.iterations,
                    success=outcome.converged && !outcome.failed && isfinite(ad_eval),
                    objective=outcome.objective,
                    objective_ad_eval=ad_eval,
                    objective_stop_eval=ad_eval,
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
    subset_main()
end
