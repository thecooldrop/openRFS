# openRFS
A training ground for implementing RFS based tracking algorithms and components in Python

Code guidelines:

In this repository numpy vectorization is heavily used to circumvent looping in Python code. In order to remain consistent across repository here some guidelines to writing such vectorized code are given. In case that multi-dimensional matrices are used to perform a computation for some vector in loop, for example compute difference between each input vector and each stored vector, then the first index of the 3D matrix would represent an index over input vector, the second index would represent the index of stored vector. Led by this example we define:

- When iterating over measurements and vector the first index of multi-dimensional array iterates over measurements, while the second index always iterates over vector

- If not iterating over usual quantities a comment is always due to indicate which set of values is represented by which of the indices of the multi-dimensional matrix.

- It is always to be specified which part of multi-dimensional matrix represents the result of computation.
