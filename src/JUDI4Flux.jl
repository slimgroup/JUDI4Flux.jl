module JUDI4Flux

    using JUDI.TimeModeling: judiJacobian
    using JUDI.TimeModeling: judiPDEfull
    using JUDI.TimeModeling: judiVector
    using Zygote, Flux, JOLI
    using Zygote: @adjoint
    import Base.*
    
    export ExtendedQForward, ExtendedQAdjoint, Forward

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


    # Extended source forward modeling
    struct ExtendedQForward{T1}
        F::T1   # forward modeling operator
        m_grad::Bool
    end

    ExtendedQForward(F; m_grad=true) = ExtendedQForward(F, m_grad)

    # Extended source forward modeling: forward mode
    function (EQF::ExtendedQForward)(w::AbstractArray, m::AbstractArray)
        Flocal = deepcopy(EQF.F)
        Flocal.model.m .= m[:, :, 1,1]
        out = Flocal * vec(w)
        nt = Flocal.recGeometry.nt[1]
        nrec = length(Flocal.recGeometry.xloc[1])
        return reshape(out, nt, nrec, 1, Flocal.info.nsrc)
    end

    function grad_w(EQF::ExtendedQForward, m, Δd)
        Flocal = deepcopy(EQF.F)
        Flocal.model.m .= m[:,:,1,1]
        Δw = adjoint(Flocal) * vec(Δd)
        return reshape(Δw, EQF.F.model.n[1], EQF.F.model.n[2], 1, EQF.F.info.nsrc)
    end

    function grad_m(EQF::ExtendedQForward, w, m, Δd)
        Flocal = deepcopy(EQF.F)
        Flocal.model.m .= m[:,:,1,1]
        J = judiJacobian(Flocal, w[:,:,1,:])
        Δm = adjoint(J) * vec(Δd)
        return reshape(Δm, EQF.F.model.n[1], EQF.F.model.n[2], 1, 1)
    end

    @adjoint function (EQF::ExtendedQForward)(w::AbstractArray, m::AbstractArray)
        if EQF.m_grad
            return EQF(w, m), Δ -> (nothing, grad_w(EQF, m, Δ), grad_m(EQF, w, m, Δ))
        else
            return EQF(w, m), Δ -> (nothing, grad_w(EQF, m, Δ), nothing)
        end
    end


####################################################################################################

    # Extended source adjoint modeling
    struct ExtendedQAdjoint{T1}
        F::T1   # adjoint modeling operator
        m_grad::Bool
    end

    ExtendedQAdjoint(F; m_grad=true) = ExtendedQAdjoint(F, m_grad)

    # Extended source adjoint modeling: forward mode
    function (EQT::ExtendedQAdjoint)(d::AbstractArray, m::AbstractArray)
        Flocal = deepcopy(EQT.F)
        Flocal.model.m .= m[:,:,1,1]
        out = adjoint(Flocal)*vec(d)
        out = reshape(out, Flocal.model.n[1], Flocal.model.n[2], 1, Flocal.info.nsrc)
        return out
    end

    function grad_d(EQT::ExtendedQAdjoint, m, Δw)
        Flocal = deepcopy(EQT.F)
        Flocal.model.m .= m[:,:,1,1]
        Δd = Flocal * vec(Δw)
        nt = Flocal.recGeometry.nt[1]
        nrec = length(Flocal.recGeometry.xloc[1])
        return reshape(Δd, nt, nrec, 1, EQT.F.info.nsrc)
    end

    function grad_m(EQT::ExtendedQAdjoint, d, m, Δw)
        Flocal = deepcopy(EQT.F)
        Flocal.model.m .= m[:,:,1,1]
        J = judiJacobian(Flocal, Δw[:,:,1,:])
        Δm = adjoint(J) * vec(d)
        return reshape(Δm, EQT.F.model.n[1], EQT.F.model.n[2], 1, 1)
    end

    @adjoint function (EQT::ExtendedQAdjoint)(d::AbstractArray, m::AbstractArray)
        if EQT.m_grad
            return EQT(d, m), Δ -> (nothing, grad_d(EQT, m, Δ), grad_m(EQT, d, m, Δ))
        else
            return EQT(d, m), Δ -> (nothing, grad_d(EQT, m, Δ), nothing)
        end
    end

    # Fixed source forward modeling
    struct Forward{T1}
        F::T1   # forward modeling operator
        q::judiVector
    end

    # Fixed source forward modeling: forward mode
    function (FWD::Forward)(m::AbstractArray)
        Flocal = deepcopy(FWD.F)
        Flocal.model.m .= m[:, :, 1,1]
        out = Flocal * vec(Forward.q)
        nt = Flocal.recGeometry.nt[1]
        nrec = length(Flocal.recGeometry.xloc[1])
        return reshape(out, nt, nrec, 1, Flocal.info.nsrc)
    end

    @adjoint function (FWD::Forward)(m::AbstractArray)
        J = judiJacobian(FWD.F,FWD.q)
        return FWD(m), Δ -> (nothing, transpose(J) * Δ)
    end

end