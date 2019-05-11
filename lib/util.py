import torch
import numpy as np
from scipy.stats import levy_stable

def calc_correct(Ahat,A):
	'''
	Calculating the correlation between the columns of A and Ahat
	'''
	AT = A.transpose(1,0)
	correlations = AT.mm(Ahat).abs()
	correct = correlations.max(dim=1)[0] 
	return correct


def Generate_A(m, n, max_correlation=0.8):
	'''
	Generating a matrix A (m X n) randomly
	such that its columns have corolation less than cor
	
	Arguments
	-----------------------------
	m : int
		number of rows
	n : int
		number of columns
	max_correlation : float
		Maximum correlation between columns
	'''
	A = torch.randn([m, 1], dtype=torch.float64)
	A /= A.norm()
	counter = 1
	while counter < n:
	    Column = torch.randn([m, 1], dtype=torch.float64)
	    Column /= Column.norm()
	    cor = Column.transpose(1, 0).mm(A).abs().max()
	    if cor < max_correlation:
	    	A = torch.cat([A, Column], dim=1)
	    	counter += 1
	return A


def Generate_alpha_random(alpha, beta, shape=1):
    """ 
    Generating random variables from an alpha-stable distribution.
    """   
    X = levy_stable.rvs(alpha, beta, size=shape)
    return X







