import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix, dok_matrix, lil_matrix

def solve_system(A, b, lower):
	n = A.shape[0]

	x = np.zeros(n)

	if lower:
		for i in range(n):
			x[i] = (b[i] - A[i, :i] @ x[:i]) / A[i, i]

	else:
		for i in range(n - 1, -1, -1):
			x[i] = (b[i] - A[i, i+1:] @ x[i+1:]) / A[i, i]

	return x

def solve_system_sparse(A, b, lower):
	n = A.shape[0]

	x = dok_matrix((n, 1))

	if lower:
		for i in range(n):
			x[i] = (b[i].sum() - (A[i, :i] @ x[:i]).sum()) / A[i, i]

	else:
		for i in range(n - 1, -1, -1):
			x[i] = (b[i].sum() - (A[i, i+1:] @ x[i+1:]).sum()) / A[i, i]

	return x

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

def get_inverse(A):
	n = A.shape[0]

	L, U = get_LU(A)

	E = np.eye(n)
	Y = []

	for i in range(n):
		y = solve_system(L, E[:, i], lower=True)
		Y.append(y)

	AI = np.zeros(A.shape)
	for i in range(n):
		AI[:, i] = solve_system(U, Y[i], lower=False)

	return AI


def get_inverse_sparse(A):
	n = A.shape[0]

	L, U = get_LU_sparse(A)

	Y = []

	for i in range(n):
		y = solve_system_sparse(L, sp.eye(10, 1, k=-i, format='dok'), lower=True)
		Y.append(y)

	AI = lil_matrix(A.shape)
	for i in range(n):
		AI[:, i] = solve_system_sparse(U, Y[i], lower=False).toarray().flatten()

	return AI


def main():
	A = np.array([
		[1, 2, 3],
		[4, 5, 6],
		[7, 8, 9]
	])

	L, U = get_LU(A)

	print(L)
	print(U)

def main2():
	L = np.array([
		[1, 0, 0],
		[1, 2, 0],
		[1, 2, 3]
	])


	b = np.array([4, 6, 9])

	x = solve_system(L.T, b, lower=False)

	print(x)

def main3():
	A = np.array([
		[1, 2, 3],
		[4, 5, 3], 
		[7, 8, 1], 
	])

	print("Actual:\n", np.linalg.inv(A))

	AI = get_inverse(A)

	print("Got:\n", AI)
	print("A^-1 @ A:\n", AI @ A)

def main4():
	A = np.array([
		[1, 0, 1],
		[2, 4, 5],
		[0, 0, 5],
	])

	L, U = get_LU(A)
	print(L, U, sep='\n')
	print('\n\n')

	S = csr_matrix(A)

	LS, US = get_LU_sparse(S)
	print(LS, US, sep='\n-----\n')

def main5():
	A = np.array([
		[1, 1, 0],
		[2, 4, 0],
		[0, 0, 5],
	])

	print("Actual:\n", np.linalg.inv(A))

	AI = get_inverse_sparse(csr_matrix(A))

	print("Got:\n", AI.toarray())
	print("A^-1 @ A:\n", AI @ A)

main3()
