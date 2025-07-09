import numpy as np

def multiply(A, B):
    """
    Multiplies two polynomials

    Arguments:
    A: Coefficients of the first polynomial
    B: Coefficients of the second polynomial

    Returns:
    C: The coefficients of A*B
    """
    n = len(A)
    m = len(B)

    # Initialize result array of size n + m - 1
    C = np.zeros(n + m - 1, dtype=A.dtype)

    # Polynomial multiplication logic
    for i in range(n):
        for j in range(m):
            C[i + j] += A[i] * B[j]

    # Remove trailing zeros if any (optional)
    while len(C) > 0 and C[-1] == 0:
        C = C[:-1]

    return C

# Test
A = np.array([1, 2])
B = np.array([3, 4])
C_exp = np.array([3, 10, 8])
np.testing.assert_allclose(multiply(A, B), C_exp, rtol=1e-5, atol=1e-10)

A = np.array([5, 6])
B = np.array([1, 3, 5, 9])
C_exp = np.array([5, 21, 43, 75, 54])
np.testing.assert_allclose(multiply(A, B), C_exp, rtol=1e-5, atol=1e-10)
np.testing.assert_allclose(multiply(B, A), C_exp, rtol=1e-5, atol=1e-10)

print("All tests passed!")