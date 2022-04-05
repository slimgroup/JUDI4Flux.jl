using Test, JUDI4Flux, JUDI

const ftol = sqrt(eps(1f0))

include("ad_gradient_test_ext_src.jl")
include("ad_gradient_test_fix_src.jl")
include("gradient_test_extended.jl")
include("gradient_test_extended_adjoint.jl")