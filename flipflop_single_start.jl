ENV["FLIPFLOP_JULIA_N_STARTS"] = get(ENV, "FLIPFLOP_JULIA_N_STARTS", "1")
ENV["FLIPFLOP_JULIA_SUBJECTS_CSV"] = get(ENV, "FLIPFLOP_JULIA_SUBJECTS_CSV", joinpath(@__DIR__, "data", "flipflop", "flipflop_python_subjects.csv"))
ENV["FLIPFLOP_JULIA_STARTS_CSV"] = get(ENV, "FLIPFLOP_JULIA_STARTS_CSV", joinpath(@__DIR__, "data", "flipflop", "flipflop_python_start_bank_6param.csv"))
ENV["FLIPFLOP_JULIA_OUTDIR"] = get(ENV, "FLIPFLOP_JULIA_OUTDIR", joinpath(@__DIR__, "outputs", "FlipFlopSingleStart"))

include(joinpath(@__DIR__, "src", "flipflop_multistart_methods.jl"))
main()
