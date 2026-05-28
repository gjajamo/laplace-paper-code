ENV["FLIPFLOP_JULIA_N_STARTS"] = get(ENV, "FLIPFLOP_JULIA_N_STARTS", "1")
ENV["FLIPFLOP_JULIA_OUTDIR"] = get(ENV, "FLIPFLOP_JULIA_OUTDIR", joinpath(@__DIR__, "outputs", "FlipFlopSingleStart"))

include(joinpath(@__DIR__, "src", "flipflop_multistart_methods.jl"))
main()
