module JUDI4Flux
    using JUDI.TimeModeling: judiJacobian
    using JUDI.TimeModeling: judiPDEextended
    using JUDI.TimeModeling: judiWeights
    using JUDI.TimeModeling: judiPDEfull
    using JUDI.TimeModeling: judiVector
    using Tracker, Flux, JOLI
    import Base.*
    import Tracker.@grad
    using Flux: @treelike
    export ForwardModel, ExtendedQForward

####################################################################################################

    # Linearized Born scattering
    *(x::judiJacobian,y::TrackedVector) = Tracker.track(*, x, y)

    @grad a::judiJacobian * b::AbstractVecOrMat =
        Tracker.data(a)*Tracker.data(b), Δ -> (nothing, transpose(a) * Δ)

####################################################################################################

    # Layer for non-linear modeling
    struct ForwardModel{T1, T2}
        F::T1
        q::T2
    end

    @treelike ForwardModel

    # Non-linear modeling: forward mode
    function (FM::ForwardModel)(m::AbstractArray)

        Flocal = deepcopy(FM.F)    # don't overwrite model of original operator
        Flocal.model.m = reshape(m, Flocal.model.n)  # reshape from Tensor into 2D/3D array
        if typeof(FM.q) == judiVector{Float32} || typeof(FM.q) == judiWeights{Float32}
            out = Flocal*FM.q
        else
            out = Flocal*vec(FM.q)
        end
        print("Typeof(out): ", typeof(out), "\n")
        return out
    end

    function show(io::IO, FM::ForwardModel)
        print(io, "ForwardModel(", size(FM.F.n), ", ", size(FM.F.m))
        print(io, ")")
    end

    # Non-linear modeling: backward mode
    (FM::ForwardModel)(m::TrackedArray) = Tracker.track(FM, m)

    @grad function (FM::ForwardModel)(m::TrackedArray)
        J = judiJacobian(FM.F, FM.q)
        return FM(Tracker.data(m)), Δ -> (reshape(adjoint(J) * Δ, J.model.n[1], J.model.n[2]), nothing)
    end


####################################################################################################


    # Extended source forward modeling
    struct ExtendedQForward{T1, T2}
        F::T1   # forward modeling operator
        m::T2   # velocity model
    end

    # Constructor
    function ExtendedQForward(F::judiPDEextended)
        Fcp = deepcopy(F)
        m = Fcp.model.m
        return ExtendedQForward(Fcp, param(m))
    end

    @treelike ExtendedQForward

    # Extended source forward modeling: forward mode
    function (EQF::ExtendedQForward)(w::AbstractArray)
        m = EQF.m

        ℱ = ForwardModel(EQF.F, w)
        out = ℱ(m)
        print("Typeof(out): ", typeof(out), "\n")

        return out
    end

    function (EQF::ExtendedQForward)(w::TrackedArray)
        m = EQF.m

        ℱ = ForwardModel(EQF.F, w)
        out = ℱ(m)
        print("Typeof(out): ", typeof(out), "\n")

        return out
    end

    function show(io::IO, EQF::ExtendedQForward)
        print(io, "ExtendedQForward(", size(EQF.F.n), ", ", size(EQF.F.m))
        print(io, ")")
    end

    # Extended source forward modeling: backward mode
    function (EQF::ExtendedQForward)(w::TrackedArray)
        return Tracker.track(EQF, w)
    end

    @grad function (EQF::ExtendedQForward)(w::TrackedArray)
        return EQF(Tracker.data(w)), Δ -> (reshape(adjoint(EQF.F) * Δ, EQF.F.model.n[1], EQF.F.model.n[2]), nothing)
    end

end
