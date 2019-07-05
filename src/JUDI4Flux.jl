module JUDI4Flux
    using JUDI.TimeModeling: judiJacobian
    using Tracker
    import Base.*
    import Tracker.@grad

    *(x::judiJacobian,y::TrackedVector) = Tracker.track(*, x, y)

    @grad a::judiJacobian * b::AbstractVecOrMat =
        Tracker.data(a)*Tracker.data(b), Δ -> (0., transpose(a) * Δ)

end
