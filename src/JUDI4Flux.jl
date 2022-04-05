module JUDI4Flux

    using JUDI: judiJacobian, judiPDEfull, judiVector, judiModeling, judiPDEextended
    using ChainRulesCore
    import Base.*

####################################################################################################

    # Linearized Born scattering
    function ChainRulesCore.rrule(::typeof(*), J::judiJacobian, x::AbstractVecOrMat)
        y = J * x
        function pullback(Δy)
            return (NoTangent(), NoTangent(), J' * Δy)
        end
        return y, pullback
    end

####################################################################################################

    # Extended source forward modeling: forward mode
    function (EQF::judiPDEextended)(m::AbstractArray, w::AbstractArray)
        out = EQF(;m=m[:, :, 1, 1]) * vec(w)
        nt = EQF.recGeometry.nt[1]
        nrec = length(EQF.recGeometry.xloc[1])
        return reshape(out, nt, nrec, 1, EQF.info.nsrc)
    end

    function grad_w(EQF::judiPDEextended, m, Δd)
        Δw = EQF(;m=m[:, :, 1, 1])' * vec(Δd)
        return reshape(Δw, EQF.model.n..., 1, EQF.info.nsrc)
    end

    function grad_m(EQF::judiPDEextended, w, m, Δd)
        J = judiJacobian(EQF, ww)
        Δm = J(;m=m[:, :, 1, 1])' * vec(Δd)
        return reshape(Δm, EQF.model.n..., 1, 1)
    end

    function ChainRulesCore.rrule(EQF::judiPDEextended, m::AbstractArray, w::AbstractArray)
        ww = length(w[:, :, 1, 1]) == size(EQF, 2) ? w[:, :, 1, 1] : w[:, :, 1, :]
        y = EQF(;m=m[:, :, 1, 1]) * vec(ww)
        function pullback(Δy)
            gw = @thunk(grad_w(EQF, m, Δy))
            gm = @thunk(grad_m(EQF, ww, m, Δy))
            return (NoTangent(), gm, gw)
        end
        return y, pullback
    end

    # Fixed source forward modeling: forward mode
    (FWD::judiModeling)(m::AbstractArray, q::judiVector) = FWD(;m=m)* q

    function  ChainRulesCore.rrule(F::judiModeling, m::AbstractArray, q::judiVector)
        y = FWD(;m=m[:, :, 1, 1]) * q
        function pullback(Δy)
            dq = @thunk(F'(;m=m)*Δy)
            dm = reshape(judiJacobian(FWD(;m=m), q)' * vec(Δ), size(m))
            return (NoTangent(), dm, dq)
        end
        return y, pullback
    end
end
