import numpy as np
import scipy.sparse as sp


from scipy.sparse import csr_matrix, dok_matrix, lil_matrix


def get_LU_sparse(A):
    EPS = 1e-7

    L = dok_matrix(A.shape)
    U = dok_matrix(A.shape)

    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if i <= j:
                t = A[i, j] - (L[i, :i] @ U[:i, j]).sum()

                if abs(t) > EPS:
                    U[i, j] = t

                if i == j:
                    L[i, i] = 1.0
                    
            else:
                if U[j, j] == 0.0:
                    raise ValueError('LU decomposition does not exist!')
    
                t = (A[i, j] - (L[i, :j] @ U[:j, j]).sum()) / U[j, j]
                if abs(t) > EPS:
                    L[i, j] = t

    return L.tocsr(), U.tocsr()


def solve_triangular_system_sparse(A, b, lower):
    n = A.shape[0]

    x = dok_matrix((n, 1))

    if lower:
        for i in range(n):
            x[i] = (b[i].sum() - (A[i, :i] @ x[:i]).sum()) / A[i, i]

    else:
        for i in range(n - 1, -1, -1):
            x[i] = (b[i].sum() - (A[i, i+1:] @ x[i+1:]).sum()) / A[i, i]

    return x


def solve_system_sparse(A, b):
    n = A.shape[0]
    
    L, U = get_LU_sparse(A)
    
    y = solve_triangular_system_sparse(L, b, lower=True)
    x = solve_triangular_system_sparse(U, y, lower=False)
    
    return x


def get_inverse_sparse(A):
    n = A.shape[0]

    L, U = get_LU_sparse(A)

    Y = []

    for i in range(n):
        y = solve_triangular_system_sparse(
            L, sp.eye(10, 1, k=-i, format='dok'), lower=True)
        Y.append(y)

    AI = lil_matrix(A.shape)
    for i in range(n):
        AI[:, i] = solve_triangular_system_sparse(
            U, Y[i], lower=False)

    return AI
