# reverse feature binarization #

# pre: values in binarized and orthogonalized columns range from -1 to 1
# input: matrix with feature binary columns

# three methods for orthogonal vectors approximation:
# 1) minimize divergences sum -> balance divergences
# 2) minimize distance between divergences -> minimize divergences sum -> balance divergences
# 3) weighted random choice, with values themselves as weights

# further possible improvement: put flat lines (all column equal) in a pool, to be considered at last
# remember to square the difference for the divergence (vectorial distance)!

import numpy
import math

''' 1)
choose the max, if more than one then base choice on balancing divergences
	init divergences array
	for each row
		get indexes of occurrences of max value in row
		if max value >= 0.5
			choose index of max divergence between indexes of max value
		else
			choose index of min divergence...
		generate row with chosen column = 1 and others = 0
		update divergences array
		replace row with generated row
	return matrix
'''

def approx_orth_min_divs_sum(M):
	divs = [0] * M.shape[1]
	for row_idx,row in enumerate(row.tolist() for row in M): # generator expression
		max_val = max(row) # don't move into list comprehension!
		max_val_idxs = [idx for idx,val in enumerate(row) if val == max_val]
		max_val_idxs_divs = [math.sqrt(divs[idx]) for idx in max_val_idxs]
		choice_div = max(max_val_idxs_divs) if max_val >= 0.5 else min(max_val_idxs_divs)
		choice = max_val_idxs[max_val_idxs_divs.index(choice_div)]
		new_row = numpy.zeros(M.shape[1])
		new_row[choice] = 1
		divs = [div + (new_row[i] - row[i])**2 for i,div in enumerate(divs)]
		#print([math.sqrt(div) for div in divs])
		M[row_idx] = new_row
	return M

''' test
a = numpy.array([[0.5,0.25,1],
				 [1,1,0.5],
				 [0,0,0],
				 [0,0,0],
				 [-1,-0.5,-0.25]])

print(approx_orth_min_divs_sum(a))
'''

''' 2)

'''

def approx_orth_min_divs_dist(M):
	pass

''' 3)

'''

def approx_orth_weighted_rand(M):
	pass

def rev_bin(M):
	pass

'''
import numpy
import timeit
rnd = numpy.random.random_integers(0, 10, (1, 100000))[0].tolist()
#print(rnd)

def dio():
	max_idx = [idx for idx,val in enumerate(rnd) if val == max(rnd)]
	#print(max_idx)

def dio2():
	max_val = max(rnd)
	max_idx = [idx for idx,val in enumerate(rnd) if val == max_val]
	#print(max_idx)

print(timeit.timeit(dio2, number=100))
print(timeit.timeit(dio, number=100))
'''