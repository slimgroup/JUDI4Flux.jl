# Gradient tests for adjoint extended modeling
# Author: Philipp Witte, pwitte3@gatech.edu
# Date: April 2020
#

using JUDI, Flux, JUDI4Flux, Test
using LinearAlgebra, Random

stol = 5f-2

Random.seed!(11)

mean(x)= sum(x)/length(x)

# Set up model structure
n = (120, 100)   # (x,y,z) or (x,z)
d = (10., 10.)
o = (0., 0.)

# Velocity [km/s]
v = ones(Float32,n) .+ 0.4f0
v[:, Int(round(end/2)):end] .= 3f0
v0 = 1f0 .* v
v0[:, 2:end-1] .= 1f0/3f0 .* (v0[:, 1:end-2] + v0[:, 2:end-1] + v0[:, 3:end])

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
F = adjoint(Pr*A_inv*adjoint(Pw))


#####################################################################################

misfit(y, m, x) = Flux.mse(F(m, y), x)

function loss(y, m, x)
    f = misfit(y, m, x)
    gs = gradient(() -> misfit(y, m, x), params(y, m))
    return f, gs[y], gs[m]
end

# Nonlinear modeling
m = reshape(m, n[1], n[2], 1, 1)
x = randn(Float32, n[1], n[2], 1, 1)
y = randn(Float32, recGeometry.nt[1], nxrec, 1, 1)
y0 = randn(Float32, recGeometry.nt[1], nxrec, 1, 1)
dy = y - y0

# Gradient test for extended modeling: weights
f0, g = loss(y0, m, x)[1:2]
h = .1f0
maxiter = 6

err1 = zeros(Float32, maxiter)
err2 = zeros(Float32, maxiter)

print("\nGradient test extended source weights\n")
for j=1:maxiter
    f = loss(y0 + h*dy, m, x)[1]
    err1[j] = abs(f - f0)
    err2[j] = abs(f - f0 - h*dot(dy, g))
    print(err1[j], "; ", err2[j], "\n")
    global h = h/2f0
end

rate1 = err1[1:end-1]./err1[2:end]
rate2 = err2[1:end-1]./err2[2:end]
@show rate1, rate2
@test isapprox(mean(rate1), 2f0; atol=stol)
@test isapprox(mean(rate2), 4f0; atol=stol)

# Gradient test for extended modeling: model
y = randn(Float32, recGeometry.nt[1], nxrec, 1, 1)
m = reshape(m, n[1], n[2], 1, 1)
m0 = reshape(m0, n[1], n[2], 1, 1)
dm = m - m0

f0, gy, gm = loss(y, m0, x)
h = .1f0
maxiter = 4

err3 = zeros(Float32, maxiter)
err4 = zeros(Float32, maxiter)

print("\nGradient test extended source model\n")
for j=1:maxiter
    f = loss(y, m0 + h*dm, x)[1]
    err3[j] = abs(f - f0)
    err4[j] = abs(f - f0 - h*dot(dm, gm))
    print(err3[j], "; ", err4[j], "\n")
    global h = h/2f0
end

rate1 = err1[1:end-1]./err1[2:end]
rate2 = err2[1:end-1]./err2[2:end]
@show rate1, rate2
@test isapprox(mean(rate1), 2f0; atol=stol)
@test isapprox(mean(rate2), 4f0; atol=stol)
