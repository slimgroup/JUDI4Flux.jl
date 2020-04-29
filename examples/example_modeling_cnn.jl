# Example for basic 2D modeling:
# The receiver positions and the source wavelets are the same for each of the four experiments.
# Author: Philipp Witte, pwitte@eos.ubc.ca
# Date: January 2017
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

## Set up source geometry (cell array with source locations for each shot)
xsrc = convertToCell([600f0])
ysrc = convertToCell([0f0])
zsrc = convertToCell([20f0])

# Set up source structure
srcGeometry = Geometry(xsrc, ysrc, zsrc; dt=dt, t=time)

# setup wavelet
f0 = 0.01f0     # MHz
wavelet = ricker_wavelet(time, dt, f0)
q = judiVector(srcGeometry, wavelet)

# Set up info structure for linear operators
ntComp = get_computational_nt(srcGeometry, recGeometry, model)
info = Info(prod(n), nsrc, ntComp)

# return linearized data as Julia array
opt = Options(return_array=true)

# Non-linear forward modeling operator
F = judiModeling(info, model, srcGeometry, recGeometry; options=opt)
num_samples = recGeometry.nt[1] * nxrec


#####################################################################################

# Build CNN
n_in = 10
n_out = 8
batchsize = nsrc
ℱ = ForwardModel(F, q)
conv1 = Conv((3, 3), n_in => 1, stride=1, pad=1)
conv2 = Conv((3, 3), 1 => n_out, stride=1, pad=1)

function network(x)
    x = conv1(x)
    x = ℱ(abs.(x))
    x = conv2(x)
    return x
end

# Nonlinear modeling

x = ones(Float32, n[1], n[2], n_in, batchsize)
y = randn(Float32, recGeometry.nt[1], nxrec, n_out, batchsize)

# Evaluate MSE loss
loss(x, y) = Flux.mse(network(x), y)

# Compute gradient w.r.t. x and y
Δx, Δy = gradient(loss, x, y)

# Compute gradient for x, y and CNN weights
p = params(x, y, conv1, conv2)
gs = gradient(() -> loss(x, y), p)

# Access gradients
Δx = gs[x]
ΔW1 = gs[conv1.weight]
Δb1 = gs[conv1.bias]

# and so on...

