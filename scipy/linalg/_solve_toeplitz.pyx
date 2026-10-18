# Author: Robert T. McGibbon, December 2014
#
# cython: boundscheck=False, wraparound=False, cdivision=True
from numpy import zeros, asarray, complex64, complex128, float32, float64
from numpy.linalg import LinAlgError
from numpy cimport complex64_t, complex128_t, float32_t, float64_t


cdef fused dz:
    float32_t
    float64_t
    complex64_t
    complex128_t


def levinson(const dz[::1] a, const dz[:, :] b):
    """Solve a linear Toeplitz system using Levinson recursion.

    Parameters
    ----------
    a : array, dtype=float, double, complex64 or complex128, shape=(2n-1,)
        The first column of the matrix in reverse order (without the diagonal)
        followed by the first (see below)
    b : array, dtype=float, double, complex64  or complex128, shape=(n,)
        The right hand side vector. Both a and b must have the same type
        (float, double, complex64 or complex128).

    Notes
    -----
    For example, the 5x5 toeplitz matrix below should be represented as
    the linear array ``a`` on the right ::

        [ a0    a1   a2  a3  a4 ]
        [ a-1   a0   a1  a2  a3 ]
        [ a-2  a-1   a0  a1  a2 ] -> [a-4  a-3  a-2  a-1  a0  a1  a2  a3  a4]
        [ a-3  a-2  a-1  a0  a1 ]
        [ a-4  a-3  a-2  a-1 a0 ]

    Returns
    -------
    x : array, shape=(n,)
        The solution vector
    reflection_coeff : array, shape=(n+1,)
        Toeplitz reflection coefficients. When a is symmetric Toeplitz and
        ``b`` is ``a[n:]``, as in the solution of autoregressive systems,
        then ``reflection_coeff`` also correspond to the partial
        autocorrelation function.
    """
    # Adapted from toeplitz.f90 by Alan Miller, accessed at
    # http://jblevins.org/mirror/amiller/toeplitz.f90
    # Released under a Public domain declaration.

    if dz is float32_t:
        dtype = float32
    elif dz is float64_t:
        dtype = float64
    elif dz is complex64_t:
        dtype = complex64
    else:
        dtype = complex128

    cdef ssize_t n, m, i, j, nmj, k, m2
    n, nrhs = b.shape[0], b.shape[1]
    cdef dz g_num, h_num, x_den, g_den
    cdef dz gj, gk, hj, hk, c1, c2
    cdef dz[:, ::1] x = zeros((n, nrhs), dtype=dtype, order="C")  # result
    cdef dz[:] x_num = zeros(nrhs, dtype=dtype) # workspace
    cdef dz[:] g = zeros(n, dtype=dtype)  # workspace
    cdef dz[:] h = zeros(n, dtype=dtype)  # workspace
    cdef dz[:, ::1] reflection_coeff = zeros((n+1, nrhs), dtype=dtype, order="C")  # history
    assert len(a) == (2*n) - 1

    if a[n-1] == 0:
        raise LinAlgError('Singular principal minor')

    for i in range(nrhs):
        x[0, i] = b[0, i] / a[n-1]

    reflection_coeff[0, :] = 1
    reflection_coeff[1, :] = x[0, :]

    if (n == 1):
        return asarray(x), asarray(reflection_coeff)

    g[0] = a[n-2] / a[n-1]
    h[0] = a[n] / a[n-1]

    for m in range(1, n):
        # Compute numerator and denominator of x[m]
        for i in range(nrhs):
            x_num[i] = -b[m, i]
        x_den = -a[n-1]
        for j in range(m):
            nmj = n + m - (j+1)
            for i in range(nrhs):
                x_num[i] = x_num[i] + a[nmj] * x[j, i]
            x_den = x_den + a[nmj] * g[m-j-1]
        if x_den == 0:
            raise LinAlgError('Singular principal minor')

        for i in range(nrhs):
            x[m, i] = x_num[i] / x_den
            reflection_coeff[m+1, i] = x[m, i]

        # Compute x
        for j in range(m):
            for i in range(nrhs):
                x[j, i] = x[j, i] - x[m, i] * g[m-j-1]
        if m == n-1:
            return asarray(x), asarray(reflection_coeff)

        # Compute the numerator and denominator of g[m] and h[m]
        g_num = -a[n-m-2]
        h_num = -a[n+m]
        g_den = -a[n-1]
        for j in range(m):
            g_num = g_num + a[n+j-m-1] * g[j]
            h_num = h_num + a[n+m-j-1] * h[j]
            g_den = g_den + a[n+j-m-1] * h[m-j-1]

        if g_den == 0.0:
            raise LinAlgError("Singular principal minor")

        # Compute g and h
        g[m] = g_num / g_den
        h[m] = h_num / x_den
        k = m - 1
        m2 = (m + 1) >> 1
        c1 = g[m]
        c2 = h[m]
        for j in range(m2):
            gj = g[j]
            gk = g[k]
            hj = h[j]
            hk = h[k]
            g[j] = gj - (c1 * hk)
            g[k] = gk - (c1 * hj)
            h[j] = hj - (c2 * gk)
            h[k] = hk - (c2 * gj)
            k -= 1
