# Propagators
Lattice Green's functions used by other repositories. Includes several modules:

### recursive
These are implementations of the recursive algorithm for calculating Green's functions on a square lattice derived by T. Morita in [J. Math. Phys. **12**, 1744 (1975)](https://iopscience.iop.org/article/10.1088/0305-4470/8/4/008) and [J. Phys. Soc. Jpn. **30**, 957 (1996)](https://www.jstage.jst.go.jp/article/iis/2/1/2_1_63/_pdf). Also handles 1D and 0D cases.

### recursive3site
Propagators for special 3-site processes arising from exchange interactions.

### general
A general algorithm for calculating cubic lattice Green's functions in principle for any dimensionality, but only calculation up to 3D is practical. Based on [M. Berciu and A. M. Cook, EPL **92**, 40003 (2010)](https://iopscience.iop.org/article/10.1209/0295-5075/92/40003)

### integrated
Direct numerical integration of k-space Green's functions. Useful for debugging and testing of the recursive functions.

### utils
Various utility functions useful in dealing with nearest neighbor sites.
