

import autograd.numpy as N
from  autograd import grad
import scipy.special as func
import copy

# Implementation of Dictionary Learning Based on Sparse Distribution Tomography


# Compute the corrolation
def calc_correct(Ahat,A):
    AT =N.transpose(A)
    M = N.dot(AT,Ahat)
    correct = []
    for i in range(A.shape[1]):
        num = []
        for j in range(A.shape[1]):
            num.append(abs(M[i][j]))
        correct.append(N.max(num))
    return correct

def gamma_hat2(alpha,U,Y):
    """
    :return: gamma
    """
    p = 0.4
    # beta = 0
    UT = N.transpose(U)
    h = abs(N.dot(UT,Y))**p
    EXP = N.mean(h,axis=1)
    c = (2**(p+1)) * func.gamma((p+1)/2.0) * func.gamma(-p/alpha) /(alpha*(N.pi**0.5) * func.gamma(-p/2.0) )
    gamma = (EXP/c)**(1/p)
    return gamma

def findU(Y,m,size,alpha):
    l = 0
    while l < size:
        U =  N.random.normal(0,1,[m,1])
        U = Normalize2(U)
        UT = N.transpose(U)
        sig = N.log(abs(N.dot(UT,Y)))
        k1 = N.mean(sig,axis=1)
        k2 = N.mean((sig[:]-k1)**2)
        alpha_sq = 2*N.pi**2 /(12*k2*(1-N.pi**2/(12*k2)))
        if alpha**2-0.2<alpha_sq<alpha**2+0.2 :
            if l == 0:
                Uall = U
            else:
                Uall = N.append(Uall,U,1)
            l += 1
    return Uall

def Normalize2(M):
    Mn = M/N.linalg.norm(M,ord=2,axis=0)
    return  Mn

def norm_alpha(A,U,alpha):
    AT = N.transpose(A)
    B = abs(N.dot(AT,U))**alpha
    norm = N.sum(B,axis=0)
    return norm

def cost_func(A,U,gamma, alpha):
    L = U.shape[1]
    norm = norm_alpha(A,U,alpha)
    cost = N.sum((norm-gamma**alpha) **2)/L
    return cost


def main(X, Y, Ahat, num_iteration, alpha, updateU = 1 ,randU = True, stepinit = 0.1, momentum = 0.7, num_col = None):
    """
    :param X: X is the signal vector
    :param Y: Y is the observations
    :param Ahat: Ahat is the starting point dictionary
    :param num_iteration: num_iteration is number of steps of SGD
    :param alpha: alpha is the estimated alpha
    :param updateU: update U every updateU iterations
    :param randU: randU is a flag showing that U is generated randomly or based on their estimated alpha
    :param stepinit: The initial stepsize of the SGD
    :param stepinit: Momentum used for SGD
    :return Ahat: The trained dictionaries Ahat and Ahat_avg
    :return: The trained dictionaries Ahat and Ahat_avg
    """
    n , K = X.shape # n is the number of atoms of the dictionary
    m = Y.shape[0]
    if num_col == None:
        num_col = 2 * m * n # Number of cloumns of U
    G_old = N.zeros([m,n])
    gradiant = grad(cost_func) # Compute the gradient
    step = stepinit
    Ahat_avg = copy.copy(Ahat)
    for it in range(num_iteration):
        ####### Update U
        if it % updateU ==0:
            if randU:
                """
                Find U such that estimated alpha for U*Y is close to the estimated alpha
                """
                U = findU(Y,m,num_col,alpha)
            else:
                """
                Generate U randomly
                """
                U =  N.random.normal(0,1,[m,num_col])
            gamma= gamma_hat2(alpha,U,Y) # Compute the gamma_hat
            cost_old = cost_func(Ahat,U,gamma, alpha) # Compute the cost function used for step_size adjustment.1

        G_new = gradiant(Ahat, U, gamma, alpha) # Compute gradient using autograd
        G = momentum * G_new + (1-momentum) * G_old
        G_old = G
        Ahat2 = Ahat - step * G
        cost_new = cost_func(Ahat2,U,gamma,alpha)
        # Line searching
        if cost_new <= cost_old:
            step *= 1.1
            if cost_new/cost_old < 0.75:
                step /= 2
            Ahat = Ahat2
            cost_old = cost_new
            # Ahat_avg is the moving avergae of A_hat trained over the course of optimization
            Ahat_avg = Ahat_avg * (it - 1) / (it + 1) + 2 * Ahat / (it + 1)
        else:
            step /= 2.0
    return Ahat, Ahat_avg
