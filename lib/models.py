

import torch
import numpy as np
import scipy.special as func
from torch import optim
import copy

# Implementation of Dictionary Learning Based on Sparse Distribution Tomography

class ModelSparseDT:
    def __init__(self, n_cols, Y, A=None, alpha=None, num_samples=1e5, batch=2000, device='cpu',
                tol=1e-5):
        """
        n_cols : int
            Number of columns of the dictionary
        Y : torch.tensor (m_rows x K)
            K Observations with m_rows
        A : torch.tensor (m_rows x n_cols)
            the initial value for dictionary A
        alpha: float
             estimated alpha
        num_samples : int
            number of samples in estimation of alpha
        batch : int
            minibatch of samples used in estimation of alpha
        """    
        if not torch.cuda.is_available() and  device == 'cuda':
            raise ValueError("cuda is not available")
        if Y.type() != 'torch.FloatTensor':
            raise TypeError("Y must be torch.FloatTensor")
        if len(Y.size()) != 2:
            raise ValueError("Y must be a matrix of dimention 2 with K observation")
        self.n_cols = n_cols
        self.m_rows = Y.size(0)
        self.K = Y.size(1) # Number of observations
        self.device = device
        ### Setting A
        self.A = A if A is not None else torch.rand([self.m_rows,self.n_cols],dtype=torch.float32)
        self.A = self.A / self.A.data.norm(dim=0)
        self.A = self.A.to(self.device).requires_grad_()
        self.Y = Y.to(self.device)
        self.batch = batch
        self.alpha = alpha if alpha is not None else self._estim_alpha(num_samples)
        self.p = 0.4
        self.c = (2**(self.p+1)) * func.gamma((self.p+1)/2.0) * \
            func.gamma(-self.p/self.alpha) /(self.alpha*(np.pi**0.5) * func.gamma(-self.p/2.0) )


    def set_optimizer(self, name_optim, args):
        '''
        Setting the optimizer
        
        Arguments:
        
        name_optim: name of the optimizer (string)
        args: arguments of the optimizer (dict)
        '''
        self.optimizer = getattr(optim, name_optim)([self.A], **args)


    def _estim_alpha(self, num_samples):
        """
        Estimating the alpha using the algorithm in
        "Reconstruction of Ultrasound RF Echoes Modeled as Stable Random Variables"
        returns alpha_hat
        """
        print('================= estimating alpha =================')
        alphahat = 0
        count = 0
        while count <= num_samples:
            UT = torch.randn([self.batch,self.m_rows], device=self.device)
            sig = (1e-7 + UT.mm(self.Y).abs()).log()         # (batch X k)
            k1 = sig.mean(dim=1).unsqueeze(1)       # (batch X 1)
            k2 = ((sig-k1)**2).mean(dim=1)          # (batch X 1)
            alpha_sq = (2 * np.pi**2 /(12*k2*(1 - np.pi**2/(12*k2))))
            #if 0<alpha_sq<4 :
            alpha_sq[alpha_sq<0] = 0
            alpha = alpha_sq ** 0.5
            alphahat += alpha.sum() 
            count += self.batch
            print(f"Initialize estimation {100 * round(min((count/num_samples),1),1)}% | "
                f"estimated_alpha: {alphahat/count:.2f}", end="\r")
        alpha = alphahat/count
        return alpha

    def _estimate_gamma(self, U):
        """
        estimating the dispersion gamma for the given U and Y
        U: torch.tensor
            A random U
        """
        UT = U.transpose(1,0)
        h = UT.mm(self.Y).abs()**self.p
        EXP = h.mean(dim=1)
        gamma = (EXP/self.c)**(1/self.p)
        return gamma


    def _findU(self, size):
        """
        Finding U such that their estimated alpha for U*Y is close to the given alpha of the dataset
        size : int
            size of the U that is being returned
        """
        count = 0
        Uall = torch.tensor([], device=self.device)
        while count < size:
            UT = torch.randn([self.batch, self.m_rows], device=self.device)
            UT /= UT.norm(dim=1).unsqueeze(dim=1)
            sig = UT.mm(self.Y).abs().log()         # (batch X k)
            k1 = sig.mean(dim=1).unsqueeze(1)       # (batch X 1)
            k2 = ((sig-k1)**2).mean(dim=1)          # (batch X 1)
            alpha_sq = (2 * np.pi**2 /(12*k2*(1 - np.pi**2/(12*k2))))
            idx =  (self.alpha**2-0.2<alpha_sq) * (alpha_sq<self.alpha**2+0.2)
            U = UT[idx,:].transpose(1,0) 
            Uall = torch.cat([Uall,U],dim=1)
            count += U.size(1)
        return Uall

    def _norm_alpha(self, A, U):
        AT = A.transpose(1,0)
        B = AT.mm(U).abs()**self.alpha
        norm = B.sum(dim=0)
        return norm

    def _check_convergence(self):
        if torch.abs(self.A - self.A_prev).max() < self.tol:
            return True
        self.coeffs_prev = self.coeffs.detach().clone()
        return False

    def Loss(self, U, gamma):
        norm = self._norm_alpha(self.A, U)
        cost = ((norm-gamma**self.alpha) **2).mean()
        return cost

    def fit(self, max_iter=1e4, rand_U=False, num_col = None, verbose=True):
        '''
        Fitting the model

        -----------------------
        Arguments:
        max_iter ; int
            maximum number of iterations
        rand_U : boolian
            specify if U are selected randomly or found such that 
            their estimated alpha is close to the initial estimation
        num_col : int
            Number of columns of U

        '''
        num_col = num_col if num_col else  2*self.m_rows*self.n_cols
        self.A_prev = self.A.detach().clone()

        for it in range(int(max_iter)):
            U = self._findU(num_col) if rand_U else torch.randn([self.m_rows, num_col], device=self.device)
            gamma = self._estimate_gamma(U)
            
            self.optimizer.zero_grad()
            loss = self.Loss(U, gamma)
            loss.backward()
            self.optimizer.step()
            if verbose:
                print(f'it : {it} | loss : {loss}', end = '\r')
            # Check that the optimization did not fail
            if torch.isnan(self.A).any():
                raise ValueError('NaNs in coeffs! Stop optimization...')
            # Check convergence
            if self._check_convergence():
                print('Converged')
                break
        self.A = self.A / self.A.data.norm(dim=0)
        return self.A.data





