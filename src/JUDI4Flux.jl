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
    export ForwardModel, ExtendedQForward, ExtendedQAdjoint

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
        Flocal.model.m = m[:,:,1,1]
        if typeof(FM.q) == judiVector{Float32} || typeof(FM.q) == judiWeights{Float32}
            out = Flocal*FM.q
        else
            out = Flocal*vec(FM.q)
        end
        nt = Flocal.recGeometry.nt[1]
        nrec = length(Flocal.recGeometry.xloc[1])
        return reshape(out, nt, nrec, 1, 1)
    end

    function show(io::IO, FM::ForwardModel)
        print(io, "ForwardModel(", size(FM.F.n), ", ", size(FM.F.m))
        print(io, ")")
    end

    # Non-linear modeling: backward mode
    (FM::ForwardModel)(m::TrackedArray) = Tracker.track(FM, m)

    @grad function (FM::ForwardModel)(m::TrackedArray)
        J = judiJacobian(FM.F, FM.q)
        return FM(Tracker.data(m)), Δ -> (reshape(adjoint(J) * vec(Δ), J.model.n[1], J.model.n[2], 1, 1), nothing)
    end

####################################################################################################

    # Extended source forward modeling
    struct ExtendedQForward{T1}
        F::T1   # forward modeling operator
    end

    @treelike ExtendedQForward

    # Extended source forward modeling: forward mode
    function (EQF::ExtendedQForward)(w::AbstractArray, m::AbstractArray)
        Flocal = deepcopy(EQF.F)
        Flocal.model.m = m[:, :, 1,1]
        out = Flocal*vec(w)
        nt = Flocal.recGeometry.nt[1]
        nrec = length(Flocal.recGeometry.xloc[1])
        return reshape(out, nt, nrec, 1, 1)
    end

    function show(io::IO, EQF::ExtendedQForward)
        print(io, "ExtendedQForward(", size(EQF.F.n), ", ", size(EQF.F.m))
        print(io, ")")
    end

    # Extended source forward modeling: backward mode
    (EQF::ExtendedQForward)(w::TrackedArray, m::TrackedArray) = Tracker.track(EQF, w, m)
    (EQF::ExtendedQForward)(w::TrackedArray, m::AbstractArray)= Tracker.track(EQF, w, m)
    (EQF::ExtendedQForward)(w::AbstractArray, m::TrackedArray) =  Tracker.track(EQF, w, m)


    function grad_w(EQF::ExtendedQForward, m, Δd)
        Flocal = deepcopy(EQF.F)
        Flocal.model.m = m
        Δw = adjoint(Flocal) * vec(Δd)
        return reshape(Δw, EQF.F.model.n[1], EQF.F.model.n[2], 1, 1)
    end

    function grad_m(EQF::ExtendedQForward, w, m, Δd)
        Flocal = deepcopy(EQF.F)
        Flocal.model.m = m
        J = judiJacobian(Flocal, w)
        Δm = adjoint(J) * vec(Δd)
        return reshape(Δm, EQF.F.model.n[1], EQF.F.model.n[2], 1, 1)
    end

    @grad function (EQF::ExtendedQForward)(w::TrackedArray, m::TrackedArray)
        m =  Tracker.data(m[:,:,1,1])
        w = Tracker.data(w[:,:,1,1])
        return EQF(Tracker.data(w), Tracker.data(m)), Δ -> (grad_w(EQF, m, Δ), grad_m(EQF, w, m, Δ), nothing)
    end

    @grad function (EQF::ExtendedQForward)(w::TrackedArray, m::AbstractArray)
        m = m[:,:,1,1]
        w = Tracker.data(w[:,:,1,1])
        return EQF(Tracker.data(w), Tracker.data(m)), Δ -> (grad_w(EQF, m, Δ), nothing, nothing)
    end

    @grad function (EQF::ExtendedQForward)(w::AbstractArray, m::TrackedArray)
        m = Tracker.data(m[:,:,1,1])
        w = w[:,:,1,1]
        return EQF(Tracker.data(w), Tracker.data(m)), Δ -> (nothing, grad_m(EQF, w, m, Δ), nothing)
    end


####################################################################################################


    # Extended source adjoint modeling
    struct ExtendedQAdjoint{T1}
        F::T1   # adjoint modeling operator
    end

    @treelike ExtendedQAdjoint

    # Extended source adjoint modeling: forward mode
    function (EQT::ExtendedQAdjoint)(d::AbstractArray, m::AbstractArray)
        Flocal = deepcopy(EQT.F)
        Flocal.model.m = m[:,:,1,1]
        out = adjoint(Flocal)*vec(d)
        out = reshape(out, Flocal.model.n[1], Flocal.model.n[2], 1, 1)
        return out
    end

    function show(io::IO, EQT::ExtendedQAdjoint)
        print(io, "ExtendedQAdjoint(", size(EQT.F.m), ", ", size(EQT.F.n))
        print(io, ")")
    end

    # Extended source adjoint modeling: backward mode
    (EQT::ExtendedQAdjoint)(d::TrackedArray, m::TrackedArray) = Tracker.track(EQT, d, m)
    (EQT::ExtendedQAdjoint)(d::TrackedArray, m::AbstractArray)= Tracker.track(EQT, d, m)
    (EQT::ExtendedQAdjoint)(d::AbstractArray, m::TrackedArray) =  Tracker.track(EQT, d, m)

    function grad_d(EQT::ExtendedQAdjoint, m, Δw)
        Flocal = deepcopy(EQT.F)
        Flocal.model.m = m
        Δd = Flocal * vec(Δw)
        nt = Flocal.recGeometry.nt[1]
        nrec = length(Flocal.recGeometry.xloc[1])
        return reshape(Δd, nt, nrec, 1, 1)
    end

    function grad_m(EQT::ExtendedQAdjoint, d, m, Δw)
        Flocal = deepcopy(EQT.F)
        Flocal.model.m = m
        J = judiJacobian(Flocal, Δw[:,:,1,1])
        Δm = adjoint(J) * vec(d)
        return reshape(Δm, EQT.F.model.n[1], EQT.F.model.n[2], 1, 1)
    end

    @grad function (EQT::ExtendedQAdjoint)(d::TrackedArray, m::TrackedArray)
        d = Tracker.data(d[:,:,1,1])
        m = Tracker.data(m[:,:,1,1])
        return EQT(Tracker.data(d), Tracker.data(m)), Δ -> (grad_d(EQT, m, Δ), grad_m(EQT, d, m, Δ), nothing)
    end

    @grad function (EQT::ExtendedQAdjoint)(d::TrackedArray, m::AbstractArray)
        d = Tracker.data(d[:,:,1,1])
        m = m[:,:,1,1]
        return EQT(Tracker.data(d), m), Δ -> (grad_d(EQT, m, Δ), nothing, nothing)
    end

    @grad function (EQT::ExtendedQAdjoint)(d::AbstractArray, m::TrackedArray)
        d = d[:,:,1,1]
        m = Tracker.data(m[:,:,1,1])
        return EQT(d, Tracker.data(m)), Δ -> (nothing, grad_m(EQT, d, m, Δ), nothing)

    end

end
