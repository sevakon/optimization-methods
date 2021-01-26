import numpy as np

from scipy.sparse import csr_matrix, dok_matrix, lil_matrix


def get_LU(A):
    L = np.eye(*A.shape)
    U = np.zeros(A.shape)
    
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if i <= j:
                U[i, j] = A[i, j] - L[i, :i] @ U[:i, j]
            else:
                if U[j, j] == 0.0:
                    raise ValueError('LU decomposition does not exist!')
                
                L[i, j] = (A[i, j] - L[i, :j] @ U[:j, j]) / U[j, j]

    return L, U


def solve_triangular_system(A, b, lower):
    n = A.shape[0]

    x = np.zeros(n)

    if lower:
        for i in range(n):
            x[i] = (b[i] - A[i, :i] @ x[:i]) / A[i, i]

    else:
        for i in range(n - 1, -1, -1):
            x[i] = (b[i] - A[i, i+1:] @ x[i+1:]) / A[i, i]

    return x


def solve_system(A, b):
    n = A.shape[0]
    
    L, U = get_LU(A)
    
    y = solve_triangular_system(L, b, lower=True)
    x = solve_triangular_system(U, y, lower=False)
    
    return x


def get_inverse(A):
    n = A.shape[0]

    L, U = get_LU(A)

    E = np.eye(n)
    Y = []

    for i in range(n):
        y = solve_triangular_system(L, E[:, i], lower=True)
        Y.append(y)

    AI = np.zeros(A.shape)
    for i in range(n):
        AI[:, i] = solve_triangular_system(U, Y[i], lower=False)

    return AI


def solve_matrix_equation(A, B):
    n = A.shape[0]
    
    AI = np.zeros(A.shape)
    for i in range(n):
        AI[:, i] = solve_triangular_system(A, B[i], lower=False)
        
    return AI
