import numpy as np
#from test_input_binarization import cmant
from fairml.orthogonal_projection import obtain_orthogonal_transformed_matrix


def cmant(M, vals):
	M_b = np.zeros((M.size,vals))
	for idx,i in enumerate(M.transpose()[0]):
		M_b[idx,int(i)] = 1.
	return M_b


#a = np.array([4,5,1,3,2,1])
#ca = np.transpose([a])

a = np.array([[1,0,1],[3,1,2],[0,0,1]], dtype=float)
ca = np.swapaxes(a,0,1)
print("ca")
print(ca)

col = ca[:,[1]]
# [:,1] -> get column as 1D array
# [:,[1]] -> get column as column
# also [:,["column 1 name","column 2 name"]] works
print("col")
print(col)

col_b = cmant(col,4)
print("col_b")
print(col_b)

ca = np.delete(ca, 1, axis=1)

cat = np.insert(ca, [1,1,1,1], col_b, axis=1) # position array length = col_b columns number
print("cat")
print(cat)

cat2 = np.array(cat)

# remove first column
#cat = cat[:,1:]
#print(cat)

ref = np.array(cat[:,0])
print("ref")
print(ref)

b = obtain_orthogonal_transformed_matrix(cat, ref)
print("b")
print(b)

# il dato è sempre consistente poichè ogni riga ha un solo 1
# l'unica colonna che cambia con l'ortogonalizzazione è la baseline
# tutte le altre sono già ortogonali

# questo però è vero ortogonalizzando per tutte le colonne della feature
# ortogonalizzando su altre feature si mantiene la consistenza?

# direi che non si avranno mai molteplici 1 su una riga, però può succedere
# che si azzerino tutte le colonne binarie di una feature non numerica

ref = np.array(cat2[:,5])
print("ref")
print(ref)

b = obtain_orthogonal_transformed_matrix(cat2, ref)
print("b")
print(b)

# in example.py non vengono mai reali perchè tronca a intero