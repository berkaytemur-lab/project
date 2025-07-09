import numpy as np

def sigmoid(x):
    """
    Computes the sigmoid of x

    Arguments:
    x: A real number or a Numpy array

    Returns:
    s: The sigmoid of x
    """

    ### BEGIN SOLUTION
    if type(x) == float or type(x) == int:
        s = 1 / (1 + np.exp(-x))
    
    else:
        s = []
        for i in range(len(x)):
            row = []
            for col in x[i]:
                value = 1 / (1 + np.exp(-col))
                row.append(value)
            s.append(row)
    ### END SOLUTION

    return s


