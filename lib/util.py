import torch


def calc_correct(Ahat,A):
	'''
	Calculating the correlation between the columns of A and Ahat
	'''
	AT = A.transpose(1,0)
	correlations = AT.mm(Ahat).abs()
	correct = correlations.max(dim=1)[0] 
	return correct

