# Gradient tests for adjoint extended modeling
# Author: Ziyi Yin, ziyi.yin@gatech.edu
# Date: Aug 2020
#
using JUDI4Flux, JUDI, Flux
using Test, LinearAlgebra, Random

Random.seed!(11)

# Set up model structure
n = (120, 100)   # (x,y,z) or (x,z)
d = (10., 10.)
o = (0., 0.)

# Velocity [km/s]
v = ones(Float32,n) .+ 0.5f0
v[:, Int(round(end/2)):end] .= 4f0
v0 = 1f0 .* v
v0[:, 2:end-1] .= 1f0/3f0 .* (v0[:, 1:end-2] + v0[:, 2:end-1] + v0[:, 3:end])

# Slowness squared [s^2/km^2]
m = (1f0 ./ v).^2
m0 = (1f0 ./ v0).^2

# Setup info and model structure
nsrc = 4	# number of sources
model = Model(n, d, o, m)
model0 = Model(n, d, o, m0)
m = reshape(m,n[1],n[2],1,1)
m0 = reshape(m0,n[1],n[2],1,1)

# Set up receiver geometry
nxrec = 120
xrec = range(50f0, stop=1150f0, length=nxrec)
yrec = 0f0
zrec = range(50f0, stop=50f0, length=nxrec)

# receiver sampling and recording time
time = 1000f0   # receiver recording time [ms]
dt = 1f0    # receiver sampling interval [ms]
nt = Int(time/dt)+1
# Set up receiver structure
recGeometry = Geometry(xrec, yrec, zrec; dt=dt, t=time, nsrc=nsrc)

# setup wavelet
f0 = 0.01f0     # MHz
wavelet = ricker_wavelet(time, dt, f0)

# Set up info structure for linear operators
ntComp = get_computational_nt(recGeometry, model)
info = Info(prod(n), nsrc, ntComp)

# return linearized data as Julia array
opt = Options(return_array=true, sum_padding=true)

# Linear operators
Pr = judiProjection(info, recGeometry)
A_inv = judiModeling(info, model; options=opt)
A_inv0 = judiModeling(info, model0; options=opt)
Pw = judiLRWF(info, wavelet)
F = Pr*A_inv*adjoint(Pw)
F0 = Pr*A_inv0*adjoint(Pw)


function GenSimSourceMulti(xsrc_index, zsrc_index, nsrc, n)
	weights = zeros(Float32, n[1], n[2], 1, nsrc)

	for j=1:nsrc
        weights[xsrc_index[j], zsrc_index[j], 1, j] = 1f0
    end
    
    return weights
end


#####################################################################################

function misfit_objective(d_obs, q0, m0, F)

	# Data residual and source residual
  	r_data = F(m0, q0) - d_obs

	# Function value
	fval = .5f0 *sum(r_data.^2)
	
	print("Function value: ", fval, "\n")
	return fval
end

if nsrc == 1
    xsrc_index = Int.(round.((range(50,stop=50,length=nsrc))))   # indices where sim source weights are 1
    zsrc_index = Int.(round.((range(10,stop=10,length=nsrc))))
else
    xsrc_index = Int.(round.((range(1,stop=n[1]-1,length=nsrc))))   # indices where sim source weights are 1
    zsrc_index = Int.(round.((range(10,stop=10,length=nsrc))))
end

q = GenSimSourceMulti(xsrc_index, zsrc_index, nsrc, model.n);
q0 = deepcopy(q)
for i = 1:nsrc
    q0[:,:,1,i] = q[:,:,1,i] .* (1f0 .+ randn(Float32, size(q[:,:,1,i])))
end

d_obs = F(m, q)

J = judiJacobian(F0,q0)
gradient_m = adjoint(J)*vec(F(m0, q0)-d_obs)

gs_inv = gradient(x -> misfit_objective(d_obs, q0, x, F), m0)

g1 = vec(gradient_m)
g2 = vec(gs_inv[1])

@test isapprox(norm(g1 - g2) / norm(g1 + g2), 0f0; atol=ftol)
@test isapprox(dot(g1, g2)/norm(g1)^2,1f0;rtol=ftol)
@test isapprox(dot(g1, g2)/norm(g2)^2,1f0;rtol=ftol)
