ENV["FLIPFLOP_JULIA_OUTDIR"] = get(ENV, "FLIPFLOP_JULIA_OUTDIR", joinpath(@__DIR__, "outputs", "FlipFlopMultistart"))

include(joinpath(@__DIR__, "src", "flipflop_multistart_methods.jl"))
main()
