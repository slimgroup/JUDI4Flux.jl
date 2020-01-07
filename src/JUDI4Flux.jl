module JUDI4Flux
    using JUDI.TimeModeling: judiJacobian
    using JUDI.TimeModeling: judiPDEfull
    using JUDI.TimeModeling: judiVector
    using Zygote, Flux, JOLI
    using Zygote: @adjoint
    import Base.*
    export ForwardModel

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

end
