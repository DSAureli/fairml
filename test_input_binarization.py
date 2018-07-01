import numpy as np
import scipy.sparse
import timeit

def lib(M = np.array([1,3,2,0])): # PRE: values in M start at 0
	row_ind = np.arange(M.size)
	data = np.ones(M.size)
	out = scipy.sparse.csr_matrix((data, (row_ind, M)))
	#print(out.toarray())

def man(M = np.array([1,3,2,0]), vals = 4):
	M_b = np.zeros((M.size,vals))
	for idx,i in enumerate(M):
		M_b[idx,i] = 1
	#print(M_b)

def man2(M = np.array([1,3,2,0]), vals = 4):
	M_c = M.reshape((-1, 1)) # to column
	M_b = np.broadcast_to(M_c, [M.size,vals])
	lbs = np.arange(vals) #labels
	M_l = np.broadcast_to(lbs, (M.size,lbs.size))
	#print((M_b == M_l).astype(float))



#M = np.array([1,3,2,0])
#lib(M)
#man(M,4)
#man2(M,4)


#print(timeit.timeit(lib, number=10000))    # 1.1516489889667136
print(timeit.timeit(man, number=10000))    # 0.015874800239648712 !
#print(timeit.timeit(man2, number=10000))   # 0.06581852881850447



def cman(M = np.array([[1],[3],[2],[0]]), vals = 4):
	M_b = np.zeros((M.size,vals))
	for idx,i in enumerate(M):
		M_b[idx,i] = 1
	#print(M_b)

#M = np.array([[1],[3],[2],[0]])
#cman(M,4)

print(timeit.timeit(cman, number=10000))    # 0.060895513998873295



def cmant(M = np.array([[1],[3],[2],[0]]), vals = 4):
	M_b = np.zeros((M.size,vals))
	for idx,i in enumerate(M.transpose()[0]):
		M_b[idx,i] = 1.
	#print(M_b)
	return M_b

#cmant()

print(timeit.timeit(cmant, number=10000))    # 0.018565353411351113



'''

a = np.array([[1,2,3],[4,5,6]])

array([[1, 2, 3],
	   [4, 5, 6]])

b = np.array([[7],[8]])

array([[7],
	   [8]])

np.append(a, b, axis=1)

array([[1, 2, 3, 7],
	   [4, 5, 6, 8]])

'''