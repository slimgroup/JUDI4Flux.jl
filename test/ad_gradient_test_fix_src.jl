# Gradient tests for adjoint extended modeling
# Author: Ziyi Yin, ziyi.yin@gatech.edu
# Date: Mar 2021
#
using JUDI4Flux, JUDI, Flux
using Test, ImageFiltering, LinearAlgebra, Random

Random.seed!(11)

# Set up model structure
n = (120, 100)   # (x,y,z) or (x,z)
d = (10., 10.)
o = (0., 0.)

# Velocity [km/s]
v = ones(Float32,n) .+ 0.4f0
v[:, Int(round(end/2)):end] .= 4f0

# Slowness squared [s^2/km^2]
m = (1f0 ./ v).^2
m0 = imfilter(m, Float32.(Kernel.gaussian(10)))

# Setup info and model structure
nsrc = 4	# number of sources
model = Model(n, d, o, m)
model0 = Model(n, d, o, m0)
m = reshape(m,n[1],n[2],1,1)
m0 = reshape(m0,n[1],n[2],1,1)

# Set up receiver geometry
nxrec = 120

xsrc = convertToCell(range(50f0, stop=1150f0,length=nsrc))
ysrc = convertToCell(range(0f0,stop=0f0,length=nsrc))
zsrc = convertToCell(range(20f0, stop=20f0,length=nsrc))

xrec = range(50f0, stop=1150f0, length=nxrec)
yrec = 0f0
zrec = range(50f0, stop=50f0, length=nxrec)

# receiver sampling and recording time
time = 1000f0   # receiver recording time [ms]
dt = 1f0    # receiver sampling interval [ms]
nt = Int(time/dt)+1
# Set up receiver structure
srcGeometry = Geometry(xsrc, ysrc, zsrc; dt=dt, t=time)
recGeometry = Geometry(xrec, yrec, zrec; dt=dt, t=time, nsrc=nsrc)

# setup wavelet
f0 = 0.01f0     # MHz
wavelet = ricker_wavelet(time, dt, f0)

q = judiVector(srcGeometry, wavelet)

# Set up info structure for linear operators
ntComp = get_computational_nt(srcGeometry, recGeometry, model)
info = Info(prod(n), nsrc, ntComp)

# return linearized data as Julia array
opt = Options(return_array=true, sum_padding=true)

# Linear operators
Pr = judiProjection(info, recGeometry)
A_inv = judiModeling(info, model; options=opt)
A_inv0 = judiModeling(info, model0; options=opt)
Ps = judiProjection(info, srcGeometry)
F = Pr*A_inv*adjoint(Ps)
F0 = Pr*A_inv0*adjoint(Ps)

#####################################################################################

function misfit_objective(d_obs, m0, G)

	# Data residual and source residual
  	r_data   = F(m0, q) - d_obs

	# Function value
	fval = .5f0 *sum(r_data.^2)
	
	print("Function value: ", fval, "\n")
	return fval
end

d_obs = G(m)

J = judiJacobian(F0, q)

gradient_m = adjoint(J)*vec(F(m0, q)-d_obs)

gs_inv = gradient(x -> misfit_objective(d_obs, x, G), m0)

g1 = vec(gradient_m)
g2 = vec(gs_inv[1])

@test isapprox(norm(g1-g2) / norm(g1), 0f0; atol=1f-1)
@test isapprox(dot(g1,g2)/norm(g1)/norm(g2),1f0;rtol=1f-1)
