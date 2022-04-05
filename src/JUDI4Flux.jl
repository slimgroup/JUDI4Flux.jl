module JUDI4Flux

    using JUDI: judiJacobian, judiPDEfull, judiVector, judiModeling, judiPDEextended
    using ChainRulesCore
    import Base.*

    squeeze(A::AbstractArray{T, N}) where {T, N} = dropdims(A, dims = (findall(size(A) .== 1)...,))

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
    function rma(v::Vector{T}, dsize::NTuple{4, Integer}, msize::NTuple{4, Integer}) where {T}
        if length(v) == prod(dsize)
            return reshape(v, dsize...)::Array{T, 4}
        else
            return reshape(v, msize...)::Array{T, 4}
        end
    end

    # Extended source forward modeling: forward mode
    function (EQF::judiPDEextended)(m::AbstractArray, w::AbstractArray)
        out = EQF(;m=squeeze(m)) * vec(w)
        nt = EQF.recGeometry.nt[1]
        nrec = length(EQF.recGeometry.xloc[1])
        return rma(out, (nt, nrec, 1, EQF.info.nsrc), (EQF.model.n..., 1, EQF.info.nsrc))
    end

    function grad_w(EQF::judiPDEextended, m, Δd)
        Δw = EQF(;m=squeeze(m))' * vec(Δd)
        nt = EQF.recGeometry.nt[1]
        nrec = length(EQF.recGeometry.xloc[1])
        return rma(Δw, (nt, nrec, 1, EQF.info.nsrc), (EQF.model.n..., 1, EQF.info.nsrc))
    end

    function grad_m(EQF::judiPDEextended, w, m, Δd)
        ww, d = size(w) == (EQF.model.n..., 1, EQF.info.nsrc) ? (w, Δd) : (Δd, w)
        J = judiJacobian(EQF, squeeze(ww))
        Δm = J(;m=squeeze(m))' * vec(d)
        return reshape(Δm, EQF.model.n..., 1, 1)
    end

    function ChainRulesCore.rrule(EQF::judiPDEextended, m::AbstractArray, w::AbstractArray)
        y = EQF(squeeze(m), squeeze(w))
        function pullback(Δy)
            gw = @thunk(grad_w(EQF, m, Δy))
            gm = @thunk(grad_m(EQF, w, m, Δy))
            return (NoTangent(), gm, gw)
        end
        return y, pullback
    end

    # Fixed source forward modeling: forward mode
    (F::judiPDEfull)(m::AbstractArray, q::judiVector) = F(;m=m) * q

    function  ChainRulesCore.rrule(F::judiPDEfull, m::AbstractArray, q::judiVector)
        y = F(;m=squeeze(m)) * q
        function pullback(Δy)
            dq = @thunk(F'(;m=squeeze(m))*Δy)
            dm = reshape(judiJacobian(F(;m=squeeze(m)), q)' * vec(Δy), size(m))
            return (NoTangent(), dm, dq)
        end
        return y, pullback
    end
end
