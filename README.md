# VGPA

**Variational Gaussian Process Approximation**

This project contains a python implementation of the basic VGPA algorithm
for approximate inference in SDEs. The code refers to the initial algorithms
as published in:

* C. Archambeau, D. Cornford, M. Opper, J. Shawe-Taylor, _Gaussian process approximations of stochastic differential equations_,
in: Journal of Machine Learning Research, Workshop and Conference Proceedings. vol. 1, 2007, pp. 1–16.

The code can deal with both 1-D and N-D systems. Examples include:

0. Ornstein-Uhlenbeck (1-D)
0. Double-Well (1-D)
0. Lorenz-63 (3-D)
0. Lorenz-98 (40-D)

If someone is interested in applying the algorithm on other dynamical systems is fully responible to (re)-write the relevant
files for the necessary expectation functions.

Note
----

Some of the optimizations in the 'auxiliary/optimize.py' are adopted (translated) from NETLAB with the following message:

    NOTE: This code is adopted from NETLAB (a free MATLAB library)
    
    Reference Book:
    (1) Ian T. Nabney (2001): Netlab: Algorithms for Pattern Recognition. Advances in Pattern Recognition, Springer.

All the copyrights of this algorithm remain with the original author of the book (Ian T. Nabney).
