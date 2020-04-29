# Gradient tests for adjoint extended modeling
# Author: Philipp Witte, pwitte3@gatech.edu
# Date: April 2020
#

using JUDI.TimeModeling, SegyIO, JOLI, Flux, JUDI4Flux
using Test, ImageFiltering, LinearAlgebra, Random

Random.seed!(11)

# Set up model structure
n = (120, 100)   # (x,y,z) or (x,z)
d = (10., 10.)
o = (0., 0.)

# Velocity [km/s]
v = ones(Float32,n) .+ 0.4f0
v[:, Int(round(end/2)):end] .= 4f0
v0 = imfilter(v, Float32.(Kernel.gaussian(10)))

# Slowness squared [s^2/km^2]
m = (1f0 ./ v).^2
m0 = (1f0 ./ v0).^2

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
opt = Options(return_array=true, dt_comp=dt, sum_padding=true)

# Linear operators
Pr = judiProjection(info, recGeometry)
A_inv = judiModeling(info, model; options=opt)
Pw = judiLRWF(info, wavelet)
F = Pr*A_inv*adjoint(Pw)


#####################################################################################

G = ExtendedQForward(F)

misfit(x, m, y) = Flux.mse(G(x, m), y)

function loss(x, m, y)
    f = misfit(x, m, y)
    gs = gradient(() -> misfit(x, m, y), params(x, m))
    return f, gs[x], gs[m]
end

# Nonlinear modeling
m = reshape(m, n[1], n[2], 1, 1)
x = randn(Float32, n[1], n[2], 1, 1)
x0 = randn(Float32, n[1], n[2], 1, 1)
dx = x - x0
y = randn(Float32, recGeometry.nt[1], nxrec, 1, 1)

# Gradient test for extended modeling: weights
f0, g = loss(x0, m, y)[1:2]
h = .1f0
maxiter = 6

err1 = zeros(Float32, maxiter)
err2 = zeros(Float32, maxiter)

print("\nGradient test extended source weights\n")
for j=1:maxiter
    f = loss(x0 + h*dx, m, y)[1]
    err1[j] = abs(f - f0)
    err2[j] = abs(f - f0 - h*dot(dx, g))
    print(err1[j], "; ", err2[j], "\n")
    global h = h/2f0
end

@test isapprox(err1[end] / (err1[1]/2^(maxiter-1)), 1f0; atol=1f1)
@test isapprox(err2[end] / (err2[1]/4^(maxiter-1)), 1f0; atol=1f1)


# Gradient test for extended modeling: model
x = randn(Float32, n[1], n[2], 1, 1)
m = reshape(m, n[1], n[2], 1, 1)
m0 = reshape(m0, n[1], n[2], 1, 1)
dm = m - m0

f0, gw, gm = loss(x, m0, y)
h = .1f0
maxiter = 4

err3 = zeros(Float32, maxiter)
err4 = zeros(Float32, maxiter)

print("\nGradient test extended source model\n")
for j=1:maxiter
    f = loss(x, m0 + h*dm, y)[1]
    err3[j] = abs(f - f0)
    err4[j] = abs(f - f0 - h*dot(dm, gm))
    print(err3[j], "; ", err4[j], "\n")
    global h = h/2f0
end

@test isapprox(err3[end] / (err3[1]/2^(maxiter-1)), 1f0; atol=1f1)
@test isapprox(err4[end] / (err4[1]/4^(maxiter-1)), 1f0; atol=1f1)
