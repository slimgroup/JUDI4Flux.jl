## JUDI4Flux: Seismic modeling for deep learning

JUDI4Flux enables compositions of seismic modeling operators with (convolutional) neural networks. JUDI4Flux is an extension of [JUDI](https://github.com/slimgroup/JUDI.jl), a framework for seismic modeling and inversion with automatic code generation and performance optimization based on [Devito](https://www.devitoproject.org/). JUDI4Flux integrates JUDI's linear and non-linear modeling operators into the [Flux](https://github.com/FluxML/Flux.jl) deep learning library, thus allowing the implementation of *physics-driven* neural networks. For backpropagation, JUDI4Flux calls JUDI's adjoint PDE solvers, thus making it possible to backpropagate effieciently through single or multiple PDE layers and scale to large problem sizes.

Features:

 - Compatibility with the Julia [Flux](https://github.com/FluxML/Flux.jl) deep learning library. Both Flux and JUDI are based on abstract high-level mathematical expressions that enable *clean* coding.

 - Blazingly fast seismic modeling routines using stencil-based finite-difference C code, which is automatically generated and optimized using [Devito](https://www.devitoproject.org/).

 - Supported operators: forward/adjoint modeling, linearized Born scattering/RTM, forward/adjoint extended source modeling.


## Linear and nonlinear JUDI operators with Flux

JUDI4Flux enables compositions of neural network layers (e.g. convolutional or fully-connected layers) with operators for seismic modeling. Instead of having to re-implement seismic modeling operators with convolutions from machine learning libraries, this makes it possible to use existing modeling operators, namely JUDI operators for Born- and nonlinear modeling. Even more importantly, we can evaluate these operators during backpropagation by calling the corresponding adjoint operators, but fully integrate them into Flux's automatic differentiation (AD) module (Flux.Tracker).

This allows combining JUDI operators with Flux misfit functions to compute gradients for least squares RTM (LS-RTM) or full waveform inversion (FWI) and combinations of modeling and neural network layers. For example, the following code show an example of combining the Born scattering operator with a dense neural network layer (check here for the full example):

```{#example_lin}
# Test demigration operator w/ Flux Dense Layer
y = randn(Float32, 100)
W = randn(Float32, 100, length(y))
b = randn(Float32, 100)

# Linearized Born operator
J = judiJacobian(F, q)

# Example image
x = vec(image)

predict(x) = W*(J*x) .+ b
loss(x,y) = Flux.mse(predict(x), y)

# Compute gradient w/ Flux
gs = Tracker.gradient(() -> loss(x, y), params(W, b, x))
gs[x]   # evalute gradient of x
```

Using non-linear modeling operators and convolutional layers is possible as well! For example, the following code shows how to integrate a nonlinear forward modeling JUDI operator into a shallow CNN, consisting of two convolutional layers, with a nonlinear forward modeling layer ℱ in-between them. As before, we can define a loss function using Flux utilities and compute derivatives with respect to various parameters, such as the squared slowness vector `m`. Once again, gradients of layers containing JUDI operators are computed using the corresponding adjoints or JUDI gradients, instead of Flux's Tracker module (full example here):


```{#example_nonlin}
# Nonlinear JUDI modeling operator
model = Model(n, d, o, m)
F = judiModeling(info, model, rec_geometry, src_geometry)

# Network layers
ℱ = ForwardModel(F, q)
conv1 = Conv((3, 3), 1=>1, pad=1, stride=1)
conv2 = Conv((3, 3), 1=>1, pad=1, stride=1)

# Network and loss
predict(x) = conv2(ℱ(conv1(x)))
loss(x, y) = Flux.mse(predict(x), y)

# Compute gradient w/ Flux
gs = Tracker.gradient(() -> loss(x, y), params(m))
gs[m]   # evalute gradient w.r.t. m
```

## Example applications

A possible application of JUDI4Flux is the implementation of loop-unrolled LS-RTM algorithms - physics-augmented convolutional neural networks for seismic imaging. By training a loop unrolled LS-RTM network using pairs of true images and observed data, this makes it possible to obtain high-fidelity images from noisy simultaneous shot records. The below figure compares RTM, standard LS-RTM with gradient descent and loop unrolled LS-RTM. Each image is obtained from a single simultaneous shot record only. (*Full preprint to be added to arXiv shortly.*)

![](docs/loop_unrolling.png)


## Related work

For a similar framework in Python that interfaces PyTorch (and actually predates `JUDI4Flux`), please check out Alan Richardson very cool package [deepwave](https://github.com/ar4/deepwave). Similar to `JUDI4Flux`, this package integrates seismic modeling and linearized modeling functions based on finite-difference stencil code into a deep learning framework. Alan's package supports seismic modeling on both CPUs and GPUs.

## Author

This package was written by [Philipp Witte](https://www.slim.eos.ubc.ca/philipp) from the Seismic Laboratory for Imaging and Modeling (SLIM) at the Georgia Institute of Technology.
