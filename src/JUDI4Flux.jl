module JUDI4Flux

    using JUDI.TimeModeling: judiJacobian
    using JUDI.TimeModeling: judiPDEfull
    using JUDI.TimeModeling: judiVector
    using Zygote, Flux, JOLI
    using Zygote: @adjoint
    import Base.*
    
    export ForwardModel, ExtendedQForward, ExtendedQAdjoint

    function my_norm(x; dt=1, p=2)
        x = dt * sum(abs.(vec(x)).^p)
        return x^(1.f0/p)
    end

    function convert_to_cell(w)
        nsrc = size(w, 4)
        w_cell = Array{Any}(undef, nsrc)
        for j=1:nsrc
            w_cell[j] = w[:,:,1,j]
        end
        return w_cell
    end

####################################################################################################

    # Linearized Born scattering
    @adjoint *(J::judiJacobian, x::AbstractVecOrMat) = *(J, x), Δ -> (nothing, transpose(J) * Δ)

####################################################################################################

   # Layer for non-linear modeling
   struct ForwardModel{T1, T2}
       F::T1
       q::T2
   end

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
       return reshape(out, nt, nrec, 1, Flocal.info.nsrc)
   end

   @adjoint function (FM::ForwardModel)(m::AbstractArray)
        J = judiJacobian(FM.F, FM.q)
        return FM(m), Δ -> (nothing, reshape(adjoint(J) * vec(Δ), J.model.n[1], J.model.n[2], 1, 1))
    end


####################################################################################################

    # Extende source modeling

    # Extended source forward modeling
    struct ExtendedQForward{T1}
        F::T1   # forward modeling operator
    end


    # Extended source forward modeling: forward mode
    function (EQF::ExtendedQForward)(w::AbstractArray, m::AbstractArray)
        Flocal = deepcopy(EQF.F)
        Flocal.model.m = m[:, :, 1,1]
        out = Flocal*vec(w)
        nt = Flocal.recGeometry.nt[1]
        nrec = length(Flocal.recGeometry.xloc[1])
        return reshape(out, nt, nrec, 1, Flocal.info.nsrc)
    end

    function grad_w(EQF::ExtendedQForward, m, Δd)
        print("Compute w grad")
        Flocal = deepcopy(EQF.F)
        Flocal.model.m = m[:,:,1,1]
        Δw = adjoint(Flocal) * vec(Δd)
        return reshape(Δw, EQF.F.model.n[1], EQF.F.model.n[2], 1, EQF.F.info.nsrc)
    end

    function grad_m(EQF::ExtendedQForward, w, m, Δd)
        print("Compute m grad")
        Flocal = deepcopy(EQF.F)
        Flocal.model.m = m[:,:,1,1]
        J = judiJacobian(Flocal, w[:,:,1,1])
        Δm = adjoint(J) * vec(Δd)
        return reshape(Δm, EQF.F.model.n[1], EQF.F.model.n[2], 1, 1)
    end

    @adjoint function (EQF::ExtendedQForward)(w::AbstractArray, m::AbstractArray)
        print("typeof(w)", typeof(w))
        print("typeof(m)", typeof(m))
        return EQF(w, m), Δ -> (nothing, grad_w(EQF, m, Δ), grad_m(EQF, w, m, Δ))
    end


####################################################################################################

    # Adjoint extende source modeling

    # Extended source adjoint modeling
    struct ExtendedQAdjoint{T1}
        F::T1   # adjoint modeling operator
    end

    # Extended source adjoint modeling: forward mode
    function (EQT::ExtendedQAdjoint)(d::AbstractArray, m::AbstractArray)
        Flocal = deepcopy(EQT.F)
        Flocal.model.m = m[:,:,1,1]
        out = adjoint(Flocal)*vec(d)
        out = reshape(out, Flocal.model.n[1], Flocal.model.n[2], 1, Flocal.info.nsrc)
        return out
    end

    function grad_d(EQT::ExtendedQAdjoint, m, Δw)
        Flocal = deepcopy(EQT.F)
        Flocal.model.m = m[:,:,1,1]
        Δd = Flocal * vec(Δw)
        nt = Flocal.recGeometry.nt[1]
        nrec = length(Flocal.recGeometry.xloc[1])
        return reshape(Δd, nt, nrec, 1, EQT.F.info.nsrc)
    end

    function grad_m(EQT::ExtendedQAdjoint, d, m, Δw)
        Flocal = deepcopy(EQT.F)
        Flocal.model.m = m[:,:,1,1]
        J = judiJacobian(Flocal, Δw[:,:,1,1])
        Δm = adjoint(J) * vec(d)
        return reshape(Δm, EQT.F.model.n[1], EQT.F.model.n[2], 1, 1)
    end

    @adjoint function (EQT::ExtendedQAdjoint)(d::AbstractArray, m::AbstractArray)
        return EQT(d, m), Δ -> (nothing, grad_d(EQT, m, Δ), grad_m(EQT, d, m, Δ))
    end

end