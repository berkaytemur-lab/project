import numpy as np
def diagonalize(A):
    import numpy as np
    """
    Diagonalizes the input matrix A

    Arguments:
    A: A two dimensional Numpy array which is guaranteed to be diagonalizable

    Returns:
    S, D, S_inv: As explained above
    """

    ### BEGIN SOLUTION

    # Retrieve the number of rows in A
    n = len(A)

    # Get the eigenvalues and eigenvectors of A
    eig_vals, S =  np.linalg.eig(A)

    # Start by initializing D to a matrix of zeros of the appropriate shape
    D = np.zeros((n,n))

    # Set the diagonal element of D to be the eigenvalues
    for i in range(n):
        
        D[i][i]=eig_vals[i]

    # Compute the inverse of S
    S_inv = np.linalg.inv(S)
    
    a_c = S @ D @ S_inv

    ### END SOLUTION

    return S, D, S_inv, a_c

A = np.array([[1, 5],
              [2, 4]])

print(diagonlize(A))