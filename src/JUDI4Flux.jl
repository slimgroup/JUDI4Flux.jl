module JUDI4Flux
    using JUDI.TimeModeling: judiJacobian
    using JUDI.TimeModeling: judiPDEfull
    using JUDI.TimeModeling: judiVector
    using Tracker
    import Base.*
    import Tracker.@grad
    export modeling

    # Linearized Born scattering
    *(x::judiJacobian,y::TrackedVector) = Tracker.track(*, x, y)

    @grad a::judiJacobian * b::AbstractVecOrMat =
        Tracker.data(a)*Tracker.data(b), Δ -> (nothing, transpose(a) * Δ)

    # Nonlinear forward modeling
    function modeling(F::judiPDEfull, m; q=nothing)
        Flocal = deepcopy(F)    # don't overwrite model of original operator
        Flocal.model.m = reshape(m, F.model.n)  # reshape from Tensor into 2D/3D array
        return Flocal*q
    end

    modeling(F::judiPDEfull, m::TrackedArray; q=nothing) = Tracker.track(modeling, F, m, q=q)

    @grad function modeling(F::judiPDEfull, m::TrackedArray; q=nothing)
        J = judiJacobian(F, q)
        return modeling(F, Tracker.data(m); q=q), Δ -> (nothing, adjoint(J) * Δ, nothing)
    end

end
