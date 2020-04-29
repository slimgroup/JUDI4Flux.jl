# Example for extended source modeling with Flux
# Author: Philipp Witte, pwitte3@gatech.edu
# Date: April 2020
#


using JUDI.TimeModeling, SegyIO, JOLI, Flux, JUDI4Flux


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
nsrc = 1	# number of sources
model = Model(n, d, o, m)

# Set up receiver geometry
nxrec = 120
xrec = range(50f0, stop=1150f0, length=nxrec)
yrec = 0f0
zrec = range(50f0, stop=50f0, length=nxrec)

# receiver sampling and recording time
time = 1000f0   # receiver recording time [ms]
dt = 1f0    # receiver sampling interval [ms]

# Set up receiver structure
recGeometry = Geometry(xrec, yrec, zrec; dt=dt, t=time, nsrc=nsrc)

# setup wavelet
f0 = 0.01f0     # MHz
wavelet = ricker_wavelet(time, dt, f0)

# Set up info structure for linear operators
ntComp = get_computational_nt(recGeometry, model)
info = Info(prod(n), nsrc, ntComp)

# return linearized data as Julia array
opt = Options(return_array=true, dt_comp=dt)

# Linear operators
Pr = judiProjection(info, recGeometry)
A_inv = judiModeling(info, model; options=opt)
Pw = judiLRWF(info, wavelet)
F = Pr*A_inv*adjoint(Pw)

# Extended source weight
w = randn(Float32, model.n)

#####################################################################################
# Extended source forward

# Build CNN
G = ExtendedQForward(F)

w = reshape(w, n[1], n[2], 1, 1)
m = reshape(m, n[1], n[2], 1, 1)
y = randn(Float32, recGeometry.nt[1], nxrec, 1, 1)

loss(w, m, y) = Flux.mse(G(w, m), y)

p =  params(w, m, y)
gs = gradient(() -> loss(w, m, y), p)

#####################################################################################
# Extended source adjoint

G = ExtendedQAdjoint(F)

w = reshape(w, n[1], n[2], 1, 1)
m = reshape(m, n[1], n[2], 1, 1)
y = randn(Float32, recGeometry.nt[1], nxrec, 1, 1)

loss(y, m, w) = Flux.mse(G(y, m), w)

p =  params(y, m, w)
gs = gradient(() -> loss(y, m, w), p)

 