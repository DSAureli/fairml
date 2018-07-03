# reverse feature binarization #

# pre: values in binarized and orthogonalized columns range from -1 to 1
# input: matrix with feature binary columns

# three methods for orthogonal vectors approximation:
# 1) minimize divergences sum -> balance divergences
# 2) minimize distance between divergences -> minimize divergences sum -> balance divergences
# 3) weighted semi-random choice, with values themselves as weights

# further possible improvement: put flat lines (all column equal) in a pool, to be considered at last
# remember to square the difference for the divergence (vectorial distance)!

import numpy as np
import math
import random

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
'''

def approx_orth_min_divs_sum(M):
	divs = [0] * M.shape[1]
	for row_idx,row in enumerate(row.tolist() for row in M): # generator expression
		max_val = max(row) # don't move into list comprehension!
		max_val_idxs = [idx for idx,val in enumerate(row) if val == max_val]
		max_val_idxs_divs = [math.sqrt(divs[idx]) for idx in max_val_idxs]
		choice_div = max(max_val_idxs_divs) if max_val >= 0.5 else min(max_val_idxs_divs)
		choice = max_val_idxs[max_val_idxs_divs.index(choice_div)]
		new_row = np.zeros(M.shape[1])
		new_row[choice] = 1
		divs = [div + (new_row[i] - row[i])**2 for i,div in enumerate(divs)]
		#print([math.sqrt(div) for div in divs])
		M[row_idx] = new_row

''' test
a = np.array([[0.5,0.25,1],
			  [1,1,0.5],
			  [0,0,0],
			  [0,0,0],
			  [-1,-0.5,-0.25]])

approx_orth_min_divs_sum(a)
print(a)
'''

''' 2)
	init row of divergences
	for each row
		init new row
		init best value of difference between divergences
		init best sum of divergences
		for each col
			generate row with column col = 1 and others = 0
			generate array of divergences
			sum row of divergences to array of divergences
			calculate difference between max and min divergences
			if first iteration or
			   calculated difference < best value of difference or
			   (calculated difference == best value of difference and sum of divergences < best sum of divergences)
				assign generated row to new row
				assign calculated difference to best value of difference
				assign sum of divergences to best sum of divergences
		update row of divergences
		replace row with new row
'''

def approx_orth_min_divs_dist(M):
	divs = [0] * M.shape[1]
	for row_idx,row in enumerate(row.tolist() for row in M): # generator expression
		new_row = None
		best_divs = None
		best_divs_diff = 0
		best_divs_sum = 0
		for col_idx,_ in enumerate(row):
			gen_row = np.zeros(M.shape[1])
			gen_row[col_idx] = 1
			new_divs = [div + (gen_row[i] - row[i])**2 for i,div in enumerate(divs)]
			new_divs_diff = max(new_divs) - min(new_divs)
			new_divs_sum = sum(new_divs)
			if col_idx == 0 or \
			   new_divs_diff < best_divs_diff or \
			   (new_divs_diff == best_divs_diff and new_divs_sum < best_divs_sum):
				new_row = gen_row
				best_divs = new_divs
				best_divs_diff = new_divs_diff
				best_divs_sum = new_divs_sum
		divs = best_divs
		#print(divs)
		M[row_idx] = new_row

''' test
b1 = np.array([[0,0,0],
			   [0,0,1]])
b2 = np.copy(b1)

approx_orth_min_divs_sum(b1)
approx_orth_min_divs_dist(b2)
print(b1)
print(b2)
'''

''' 3)
	for each row
		if at least one value > 0
			generate array of weights [x for x in row if x > 0 else 0]
			random choice with weights
		else
			choose max in row
		generate new row with chosen index = 1 and others = 0
		replace row with new row
'''

def approx_orth_weighted_rand(M):
	for row_idx,row in enumerate(row.tolist() for row in M): # generator expression
		if any(x > 0 for x in row): # generator expression
			weights = [x if x > 0 else 0 for x in row]
			choice = random.choices([i for i,_ in enumerate(row)], weights)
		else:
			max_val = max(row) # don't move into list comprehension!
			max_idxs = [idx for idx,val in enumerate(row) if val == max_val]
			choice = random.choice(max_idxs)
		new_row = np.zeros(M.shape[1])
		new_row[choice] = 1
		M[row_idx] = new_row

''' test
c = np.array([[-1,-0.5,-0.25],
			  [-1,-0.5,0.25],
			  [0.25,0.5,1]])

approx_orth_weighted_rand(c)
print(c)
'''

''' perf test

'''

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