# Gradient tests for adjoint extended modeling
# Author: Ziyi Yin, ziyi.yin@gatech.edu
# Date: Aug 2020
#
using JUDI4Flux
using JUDI.TimeModeling, SegyIO, JOLI, Flux
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

# Setup info and model structure
nsrc = 4	# number of sources
model = Model(n, d, o, m)
m = reshape(m,n[1],n[2],1,1)

model = Model(n, d, o, m0)
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
Pw = judiLRWF(info, wavelet)
F = Pr*A_inv*adjoint(Pw)


function GenSimSourceMulti(xsrc_index, zsrc_index, nsrc, n)
	weights = zeros(Float32, n[1], n[2], 1, nsrc)

	for j=1:nsrc
        weights[xsrc_index[j], zsrc_index[j], 1, j] = 1f0
    end
    
    return weights
end


#####################################################################################

G = ExtendedQForward(F)

function misfit_objective(d_obs, q0, m0, G)

	# Data residual and source residual
  	r_data   = G(q0, m0) - d_obs

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

q1 = GenSimSourceMulti(xsrc_index, zsrc_index, nsrc, model.n);
q = deepcopy(q1)
for i = 1:nsrc
    q[:,:,1,i] = imfilter(q1[:,:,1,i], Float32.(Kernel.gaussian(10)))
end

d_obs = G(q1,m)

J = judiJacobian(F,q)
gradient_m = adjoint(J)*vec(G(q,m)-d_obs)

Loss(m)=misfit_objective(d_obs, q, m, G)
p = params(m)
gs_inv = gradient(() -> Loss(m),p)

g1 = vec(gradient_m)
g2 = vec(gs_inv[m])

@test isapprox(norm(g1-g2) / norm(g1), 0f0; atol=2f-1)
@test isapprox(dot(g1,g2)/norm(g1)/norm(g2),1f0;rtol=2f-1)