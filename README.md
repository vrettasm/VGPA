# VGPA

**Variational Gaussian Process Approximation**

This project contains a python3 implementation of the original VGPA
algorithm for approximate inference in SDEs. It can be directly applied
to solve (perform inference) to four (stochastic) dynamical systems,
namely:

1. [Double Well](https://en.wikipedia.org/wiki/Double-well_potential)
2. [Ornstein-Uhlenbeck](https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process)
3. [Lorenz63 (3D)](https://en.wikipedia.org/wiki/Lorenz_system)
4. [Lorenz96](https://en.wikipedia.org/wiki/Lorenz_96_model)

For any other dynamical system, one has to write the required code
(and inherit from the *stochastic_process.py*) to generate the sample
paths and compute the required energy terms.

The forward-backward ODEs can be solved with four different solvers:

1. [Euler's method](https://en.wikipedia.org/wiki/Euler_method): 1st order
2. [Heun's method](https://en.wikipedia.org/wiki/Heun%27s_method): 1st order (predictor-corrector)
3. [Runge-Kutta 2](https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods): 2nd order
4. [Runge-Kutta 4](https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods): 4th order

### Required packages
---

The recommended version is **Python3.7**. The implementation is independent from
third-party libraries since all the optimization (SCG) and integration routines
(Fwd / Bwd) are custom made. However, you need to have installed basic packages
such as:

> 
> numpy, scipy, numba, h5py, json 
>

## How to run
---

To execute the program, first navigate to the main directory of the project
(i.e. where the vgpa_main.py is located), and then run the following command:

    $ python3 vgpa_main.py --params path/to/sim_params.json

The models parameters should be given in the 'sim_params.json' file as follows:

```
{ ...

  "Time-window": {
    "t0": 0.00,
    "tf": 10.0,
    "dt": 0.01
  },

  "Noise": {
    "sys": 0.80,
    "obs": 0.04
  },

  ...
}
```

## References
---

The code refers to the initial algorithms as published in:

1. [C. Archambeau, D. Cornford, M. Opper, J. Shawe-Taylor (2007)](
http://proceedings.mlr.press/v1/archambeau07a.html).
"Gaussian process approximations of stochastic differential equations",
Journal of Machine Learning Research, Workshop and Conference Proceedings.
vol. 1, 2007, pp. 1–16.

2. [C. Archambeau, M. Opper, Y. Shen D. Cornford,  J. Shawe-Taylor (2007)](
https://papers.nips.cc/paper/3282-variational-inference-for-diffusion-processes.pdf).
"Variational Inference for Diffusion Processes",
Neural Information Processing Systems (NIPS), vol. 20.

### Note

Some of the optimizations are adopted (translated) from NETLAB with the
following message:

    NOTE: This code is adopted from NETLAB (a free MATLAB library)
    
    Reference Book:
    (1) Ian T. Nabney (2001): Netlab: Algorithms for Pattern Recognition.
    Advances in Pattern Recognition, Springer.

All the copyrights of this algorithm remain with the original author of the
book (Ian T. Nabney).

### Contact
---

For any questions / comments please contact me at: vrettasm@gmail.com