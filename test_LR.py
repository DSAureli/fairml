import numpy as np
import pandas as pd
from fairml.orthogonal_projection import obtain_orthogonal_transformed_matrix
from fairml import audit_model
from fairml import plot_dependencies
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt


def bb(data):
	print("BB")
	print(data)
	out = np.zeros((data.shape[0],1))
	print("BB_out")
	print(out)
	for idx,row in enumerate(data):
		out[idx,0] = (3 * row[0]) ** 3 + (row[1] ** 2) + 9 * row[2]
		#out[idx,0] = 6 * row[0] + 6 * row[2] + row[1]
	return out


def cmant(M, vals):
	M_b = np.zeros((M.size,vals))
	for idx,i in enumerate(M.transpose()[0]):
		M_b[idx,int(i)] = 1.
	return M_b


a = np.array([	[1,0,1,1,0,1,0,0,1,1,0,1],
				[3,1,2,7,4,5,2,6,2,3,4,5],
				[0,0,1,0,1,0,1,0,0,0,1,1]], dtype=float)
ca = np.swapaxes(a,0,1)

print("ca")
print(ca)

cout = bb(ca)

print("out")
print(cout)

ca = pd.DataFrame(ca)
imp, dir_imp = audit_model(bb, ca)

fig = plot_dependencies(
	imp.median(),
	reverse_values=False,
	title="test_LR_1"
)

file_name = "test_LR_1.png"
plt.savefig(file_name, transparent=False, bbox_inches='tight', dpi=250)

fig = plot_dependencies(
	dir_imp.median(),
	reverse_values=False,
	title="test_LR_1 direct perturbation"
)

file_name = "test_LR_1_dir.png"
plt.savefig(file_name, transparent=False, bbox_inches='tight', dpi=250)


clf = LogisticRegression(penalty='l2', C=0.01)
clf.fit(ca.values, cout)

imp, dir_imp = audit_model(clf.predict, ca)

fig = plot_dependencies(
	imp.median(),
	reverse_values=False,
	title="test_LR_2"
)

file_name = "test_LR_2.png"
plt.savefig(file_name, transparent=False, bbox_inches='tight', dpi=250)

fig = plot_dependencies(
	dir_imp.median(),
	reverse_values=False,
	title="test_LR_2 direct perturbation"
)

file_name = "test_LR_2_dir.png"
plt.savefig(file_name, transparent=False, bbox_inches='tight', dpi=250)


'''
col_idx = 1

col = ca[:,[col_idx]]
# [:,1] -> get column as 1D array
# [:,[1]] -> get column as column
# also [:,["column 1 name","column 2 name"]] works
print("col")
print(col)

col_b = cmant(col,4)
print("col_b")
print(col_b)

ca = np.delete(ca, col_idx, axis=1)

cat = np.insert(ca, [1,1,1,1], col_b, axis=1) # position array length = col_b columns number
print("cat")
print(cat)

cat2 = np.array(cat)

# remove first column
#cat = cat[:,1:]
#print(cat)

ref = np.array(cat[:,1])
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

ref = np.array(cat2[:,6])
print("ref")
print(ref)

b = obtain_orthogonal_transformed_matrix(cat2, ref)
print("b")
print(b)

# in example.py non vengono mai reali perchè tronca a intero
'''