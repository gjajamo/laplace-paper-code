ENV["WARFARIN_JULIA_N_STARTS"] = get(ENV, "WARFARIN_JULIA_N_STARTS", "1")
ENV["WARFARIN_JULIA_REPRESENTATIONS"] = "ode"
ENV["WARFARIN_JULIA_DATA"] = get(ENV, "WARFARIN_JULIA_DATA", joinpath(@__DIR__, "data", "warfarin_dat.csv"))
ENV["WARFARIN_JULIA_OUTDIR"] = get(ENV, "WARFARIN_JULIA_OUTDIR", joinpath(@__DIR__, "outputs", "WarfarinSingleStart"))

include(joinpath(@__DIR__, "src", "warfarin_multistart_methods.jl"))
main()
