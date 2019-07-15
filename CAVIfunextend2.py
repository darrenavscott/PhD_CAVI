# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 17:35:41 2019

@author: darre

Functions for CAVI updates 

"""


import numpy as np
import matplotlib.pyplot as plt
import math
from numpy.random import rand 
import matplotlib.pyplot as plt
import random
import scipy as sy
from scipy import linalg 
from scipy.stats import invgamma
from scipy.stats import beta
from scipy.stats import gamma
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import statsmodels.formula.api as smf
import scipy as sy
from scipy import linalg 
from scipy.stats import invgamma
from scipy.stats import beta
from scipy.stats import gamma




def usumhlk(Y, X, beta_1, rho_mu, inv_sigma2_1, T, t, s): 
    """"
	Purpose
	-------
	(Calculate the Atk**2 in the mu_beta calculation. Returns a column vector)  
     
 	Parameters
	----------
	(Y = Matrix of data, 
     X = Design matrix, 
     mu_beta = Matrix of mean parameters for beta, 
     beta_1 = Vector of updates for beta,
     rho_mu = Vector of mean parameters for E_q[rho], 
     inv_sigma2_1 = Vector of mean parameters E_q[sigma2**(-1)],
     T = Total number of responses,
     t = Response,
     s = Covariate column,
	
	Returns
	-------
	(Scalar sum for Atk_2 for reponse t and covariate s.
	"""    
    ind = []
    
    for k in np.arange(t+1,T,1):
        index = np.delete(np.arange(0,T,1),t)[np.delete(np.arange(0,T,1),t)<k]   
        U = Y - np.dot(X, beta_1.T)  
        ind.append(inv_sigma2_1[k] * np.dot(U[:,index], rho_mu[index, k]) * rho_mu[t,k]) #append adds to the arrary
            
    out = np.sum(ind, axis = 0) #Because list so axis is 0
    
    return out


def sigmab_beta_gamma_updateu(Y, X, mu_beta, beta_1, sigma2_beta, gamma_1,
                              inv_sigma2_1, log_inv_sigma2_1, rho_mu, 
                              rho_2, inv_w_1, log_inv_w_1, log_omega_1,
                              log_minomega_1, T):
    """"
	Purpose
	-------
	(Calculate CAVI update for the parameters in the approximating densities 
     of beta and gamma.)  
     
	Parameters
	----------
	(Y = Matrix of data, 
     X = Design matrix, 
     mu_beta = Matrix of mean parameters for beta, 
     beta_1 = mu_beta x gamma_1
     sigma2_beta = Matrix of variance parameters for beta, 
     gamma_1 = Matrix of paramter values for E_q[gamma],
     inv_sigma2_1 = Vector of values for E_q[1/sigma2], 
     log_inv_sigma2_1 = Vector of values for E_q[1/(log sigma2)], 
     rho_mu = Vector of mean parameters for E_q[rho], 
     rho_2 = Vector of values for E_q[rho**2], 
     inv_w_1 = Scalar for 1/ E_q[w], 
     log_inv_w_1 = Scalar 1 / E_q[log w]),
     log_omega_1 = log Eq[omega],
     log_minomega_1 = log Eq[log(1 - omega)],
     T = Total number of reponses. 
     
	Returns
	-------
	(mu_beta = E_q[beta] T x p, 
     sigma2_beta = Var_q[beta] T x p,
     gamma_1 = E_q[gamma] T x p, 
     beta_1 = mu_beta * gamma_1 T x p,
     beta_2 = (sigma2_beta + mu_beta**2)(gamma_1) T x p)
	"""
    sigma2_beta = (np.sum(X**2, axis = 0) * ((inv_sigma2_1 + np.sum(rho_2 * inv_sigma2_1, axis = 1))[:, np.newaxis]) 
                  + inv_w_1[:, np.newaxis]) ** (-1)
  
    for t in np.arange(T):
        for s in np.arange(p):
            
            if t < (T - 1):
                  
                ut_s =  Y[:,t] - np.dot(np.delete(X, s, axis = 1), np.delete(beta_1[t, :], s))               
                ukl = Y[:,:t] - np.dot(X, beta_1[:t,:].T)     #sum over k<t                     
                uku = Y[:,t+1:] - np.dot(X, beta_1[t+1:,:].T) #sum over k>t
                
                A = inv_sigma2_1[t] * ut_s - inv_sigma2_1[t] * np.sum(ukl * rho_mu[:t, t], axis = 1)
                B = np.sum((inv_sigma2_1[t+1:] * rho_2[t, t+1:])[:,np.newaxis] * ut_s, axis = 0)
                C = np.sum(inv_sigma2_1[t+1:] * rho_mu[t,t+1:] * uku , axis = 1) 
                D = usumhlk(Y = Y, X = X, beta_1 = beta_1, rho_mu = rho_mu, inv_sigma2_1 = inv_sigma2_1,
                            T = T, t = t, s = s)
                
                mu_beta[t, s] = sigma2_beta[t, s] * (np.dot(X[:,s], A + B - C + D)) 
                                    
                gamma_1[t, s] = 1 / (1 + (1 / np.sqrt(sigma2_beta[t, s])) * \
                                    np.exp(log_minomega_1[s] - log_omega_1[s] - \
                                    (1 / 2) * log_inv_w_1[t] - \
                                    ((1 / 2) * mu_beta[t, s]**2) * (1 / sigma2_beta[t, s])))                         
                                                  
                beta_1[t, s] = mu_beta[t, s] * gamma_1[t, s]
                beta_2[t, s] = (sigma2_beta[t, s] + mu_beta[t, s]**2) * gamma_1[t, s]  
                                          
               
            else:
                ut_s = Y[:,t] - np.dot(np.delete(X, s, axis = 1), np.delete(beta_1[t, :], s))               
                ukl = Y[:,:t] - np.dot(X, beta_1[:t,:].T)     #sum over k<t                     
                
                A = inv_sigma2_1[t] * ut_s - inv_sigma2_1[t] * np.sum(ukl * rho_mu[:t, t], axis = 1)
                
                mu_beta[t, s] = sigma2_beta[t, s] * (np.dot(X[:,s], A))                            
                
                gamma_1[t, s] = 1 / (1 + (1 / np.sqrt(sigma2_beta[t, s])) * \
                                    np.exp(log_minomega_1[s] - log_omega_1[s] - \
                                    (1 / 2) * log_inv_w_1[t] - \
                                    ((1 / 2) * mu_beta[t, s]**2) * (1 / sigma2_beta[t, s])))  
                
                beta_1[t, s] = mu_beta[t, s] * gamma_1[t, s]
                beta_2[t, s] = (sigma2_beta[t, s] + mu_beta[t, s]**2) * gamma_1[t, s]    
                                           
    return np.array([mu_beta, sigma2_beta, gamma_1, beta_1, beta_2])   


def kprodaccu(Y, X, beta_1, rho_mu, T):
    """"
	Purpose
	-------
	(Calculate part expression b_star_sigma2 to be called in the update. To
     be run before sigma_2 update)  
     	
	Parameters
	----------
	(Y = Data,
     X = Design matrix,
     beta_1 = Vector of updates for beta,
     rho_mu = Vector of updates for rho,
     T = Total number of responses
	
	Returns
	-------
	(Part k of the update equation 3.2.19 in upgrade report.)
	"""
    
    total = np.zeros(T-2)
    p = 0
    
    u = Y - np.dot(X, beta_1.T)
    
    for n in np.arange(2,T):
        ix = np.indices((n,n))[:, ~np.tri(n, k=0, dtype=bool)]
        total[p] = np.sum(u[:,ix.T[:,0]] * u[:,ix.T[:,1]] * rho_mu[ix.T[:,0],n] * rho_mu[ix.T[:,1],n])     
            
        p = p + 1

    tot = np.concatenate([np.zeros(T-len(total)),total])
    return (np.array([tot]))
                


def sumexself(X, beta_1, p):
    """"
	Purpose
	-------
	(Calculate part expression ||u_t||**2)  
     	
	Parameters
	----------
	(X = Design matrix,
     beta_1 = Vector of updates for beta,
     p = Total number of covariates
	
	Returns
	-------
	(Part UD of the update in notes.)
	"""
    isx =  np.indices((p, p))[:, ~np.tri(p, k=0, dtype=bool)]
    
    Dm  = np.sum(X[:,isx[0]] * X[:,isx[1]], axis = 0) 
    Out = np.sum(Dm * beta_1[:,isx[0]] *  beta_1[:,isx[1]], axis = 1) 

    return (Out)



def sigma2_updateu2(Y, X, nu, T, n, gamma_1, tau_1, rho_mu, rho_2, inv_w_1, beta_1, beta_2):
    """"
	Purpose
	-------
	(Calculate CAVI update for the parameters in the approximating density 
     of sigma2.)  
     	
	Parameters
	----------
	(Y = Data,
     X = Design matrix,
     nu = Scalar hyperparameter for sigma2,
     T = Total number of responses,
     n = Number of individuals,
     gamma_1 = Vector of updates for gamma E_q[gamma],
     tau_1 = Scalar update for tau E_q[tau],
     rho_mu = Vector of updates for rho E_q[rho],
     rho_2 = Vector of updates for E_q[rho**2],
     inv_w_1 = Scaler update for w**(-1) E_q[1/w], 
     beta_1 = Matrix of mean parameters for E_q[beta] = mu_beta * gamma_1, 
     beta_2 = Matrix of beta parameters sigma2_beta + mu_beta**2, 
	
	Returns
	-------
	(a_star_sigma2 1 x T,
     b_star_sigma2 1 x T,
     inv_sigma2_1 1 x T,
     log_inv_sigma2_1 1 x T)
	"""
    
    a_star_sigma2 = (np.arange(T) / 2) + ((nu - T + np.arange(1,T+1)) / 2) \
                   + (n / 2) 
    
    U = Y - np.dot(X, beta_1.T)  
    
    #U_2 = <U_t, U_t> for all t
    UA = np.sum(Y**2, axis = 0)
    UB = 2 * np.sum(Y * np.dot(X, beta_1.T), axis = 0) 
    UC = np.dot(np.sum(X**2, axis = 0), beta_2.T)
    UD = 2 * sumexself(X, beta_1, p)
    
    U_2 = UA - UB + UC + UD 
    
    A = tau_1 + tau_1 * np.sum(rho_mu, axis = 0)
    B = U_2 
    C = np.concatenate([np.zeros(1), np.dot(U_2[:T-1], rho_2[:T-1,1:])]) 
    D = kprodaccu(Y = Y, X = X, beta_1 = beta_1, rho_mu = rho_mu, T = T)
    E = np.concatenate([np.zeros(1), np.sum(U[:,1:] * np.dot(U[:,:T-1], rho_mu[:T-1,1:]), axis = 0)])
       
    b_star_sigma2 = (1/ 2) * (A + B + C) + D - E 
        
    inv_sigma2_1 = a_star_sigma2 / b_star_sigma2                        

    log_inv_sigma2_1 = sy.special.digamma(a_star_sigma2) - np.log(b_star_sigma2)

    return np.array([a_star_sigma2, b_star_sigma2, inv_sigma2_1, log_inv_sigma2_1])       


                               
def sigma2_updateu1(Y, X, nu, T, n, gamma_1, tau_1, rho_mu, rho_2, inv_w_1, beta_1, beta_2):
    """"
	Purpose
	-------
	(Calculate CAVI update for the parameters in the approximating density 
     of sigma2. Might be quicker.)  
     	
	Parameters
	----------
	(Y = Data,
     X = Design matrix,
     nu = Scalar hyperparameter for sigma2,
     T = Total number of responses,
     n = Number of individuals,
     gamma_1 = Vector of updates for gamma E_q[gamma],
     tau_1 = Scalar update for tau E_q[tau],
     rho_mu = Vector of updates for rho E_q[rho],
     rho_2 = Vector of updates for E_q[rho**2],
     inv_w_1 = Scaler update for w**(-1) E_q[1/w], 
     beta_1 = Matrix of mean parameters for E_q[beta] = mu_beta * gamma_1, 
     beta_2 = Matrix of beta parameters sigma2_beta + mu_beta**2, 
	
	Returns
	-------
	(a_star_sigma2 1 x T,
     b_star_sigma2 1 x T,
     inv_sigma2_1 1 x T,
     log_inv_sigma2_1 1 x T)
	"""
    
    a_star_sigma2 = (np.arange(T) / 2) + ((nu - T + np.arange(1,T+1)) / 2) \
                   + (n / 2) 
    
    U = Y - np.dot(X, beta_1.T)  
    
    #U_2 = <U_t, U_t> for all t
    UA = np.sum(Y**2, axis = 0)
    UB = 2 * np.sum(Y * np.dot(X, beta_1.T), axis = 0) 
    UC = np.dot(np.sum(X**2, axis = 0), beta_2.T)
    UD = 2 * np.sum(np.sum((beta_1[:,:p-1][:,np.newaxis,:] * X[:,:p-1]) \
              * ((np.cumsum(((X * beta_1[:,np.newaxis])[:,:,1:])[:,:,::-1],axis = 2))[:,:,::-1]),axis = 1), axis = 1)
    
    U_2 = UA - UB + UC + UD 
    
          
    A = tau_1 + tau_1 * np.sum(rho_mu, axis= 0)
    B = U_2 
    C = np.concatenate([np.zeros(1), np.dot(U_2[:T-1], rho_2[:T-1,1:])]) 
    D = kprodaccu(Y = Y, X = X, beta_1 = beta_1, rho_mu = rho_mu, T = T)
    E = np.concatenate([np.zeros(1), np.sum(U[:,1:] * np.dot(U[:,:T-1], rho_mu[:T-1,1:]), axis = 0)])
    
    b_star_sigma2 = (1 / 2) * (A + B + C) + D - E 
    
    
    inv_sigma2_1 = a_star_sigma2 / b_star_sigma2                        

    log_inv_sigma2_1 = sy.special.digamma(a_star_sigma2) - np.log(b_star_sigma2)

    return np.array([a_star_sigma2, b_star_sigma2, inv_sigma2_1, log_inv_sigma2_1])       



def rho_updateu(Y, X, beta_1, beta_2, inv_sigma2_1, tau_1, T, rho_mu):
    """"
	Purpose
	-------
	(Calculate CAVI update for the parameters in the approximating densities 
     of rho.)  
     
	Parameters
	----------
	(Y = Matrix of data, 
     X = Design matrix, 
     beta_1 = Matrix of mean parameters for E_q[beta] = mu_beta * gamma_1, 
     beta_2 = Matrix of beta parameters sigma2_beta + mu_beta**2, 
     inv_sigma2_1 = Vector of values for E_q[1/sigma2], 
     tau_1 = Scalar update for tau E_q[tau],
     rho_mu = Triangular matrix E_q[rho].
     
	Returns
	-------
	(rho_mu = E_q[rho] p x T, 
     rho_sigma2 = Var_q[rho] p x T,
     rho_2 = E_q[rho**2] p x T)
	"""
    
    U = Y - np.dot(X, beta_1.T)   

    #U_2 = <U_t, U_t> for all t
    UA = np.sum(Y**2, axis = 0)
    UB = 2 * np.sum(Y * np.dot(X, beta_1.T), axis = 0) 
    UC = np.dot(np.sum(X**2, axis = 0), beta_2.T)
    UD = 2 * sumexself(X, beta_1, p)
    
    U_2 = UA - UB + UC + UD 
    
    #Inverse inside of the tril function
    rho_sigma2 = (inv_sigma2_1**(-1)) * np.tril(np.concatenate([(tau_1 + U_2[:T-1])**(-1), np.zeros(1)]), k = -1).T  

    for t in np.arange(1,T):
        for k in np.arange(t):
            rho_mu[k,t] =  ((np.dot(U[:,t], U[:,k]) - \
                           np.sum(np.dot(U[:,k], np.delete(U[:,:t], k, axis=1)) * \
                           np.delete(rho_mu[:t,t], k)))) / (tau_1 + U_2[k])
    
    rho_2  = rho_sigma2**2 + rho_mu**2

    return np.array([rho_mu, rho_sigma2, rho_2])   


                     
def omega_update(gamma_1, T, a_omega, b_omega):
    """"
	Purpose
	-------
	(Calculate CAVI update for the parameters in the approximating densities 
     of omega.)  
     	
	Parameters
	----------
	(T = Total number of responses,
     a_omega = Vector of hyperparameters for omega,
     b_omega = Vector of hyperparameters for omega
     gamma_1 = Vector of gamma updates)
	
	Returns
	-------
	(omega_1 = E_q[omega]  1 x p,
     log_omega_1 = E_q[log omega] 1 x p,
     log_minomega_1 = E_q[log(1-omega)] 1 x p,
     a_omega_star 1 x p,
     b_omega_star 1 x p)
	"""
    
    a_omega_star = a_omega + np.sum(gamma_1,axis=0) 
    b_omega_star = b_omega + T - np.sum(gamma_1,axis=0)                          
    
    omega_1 = a_omega_star / (a_omega + b_omega + T)
    log_omega_1 = sy.special.digamma(a_omega_star) - sy.special.digamma(a_omega_star + b_omega_star)
    log_minomega_1 = sy.special.digamma(b_omega_star) - sy.special.digamma(a_omega_star + b_omega_star)
    
    return np.array([a_omega_star, b_omega_star, omega_1, log_omega_1, log_minomega_1])   




def w_update(a_w, b_w_1, gamma_1, beta_2):
    """"
	Purpose
	-------
	(Calculate CAVI update for the parameters in the approximating density 
     of w_t.)  
     	
	Parameters
	----------
	(a_w = Vector of hyperparameters t x 1,
     b_w_1 = Vector of hyperparameters t x 1,
     gamma_1 = Matrix of update gamma parameters,
     beta_2 = Matrix of updated parameters E_q[beta**2]).
	
	Returns
	-------
	(a_w_star (1),
     b_w_star (1),
     inv_w_1 (1),
     log_inv_w_1 (1))
	"""
    a_w_star = a_w + (1 / 2) * np.sum(gamma_1, axis = 1)
    b_w_star = b_w_1 + (1 / 2) * np.sum(beta_2, axis = 1)                               
    
    inv_w_1 = a_w_star / b_w_star
    log_inv_w_1 = sy.special.digamma(a_w_star) - np.log(b_w_star)                               
                                  
    return np.array([a_w_star, b_w_star, inv_w_1, log_inv_w_1]) 
  
   
    
def bw_update(a_w, a_b, b_b, T, inv_w_1):
    """"
	Purpose
	-------
	(Calculate CAVI update for the parameters in the approximating density 
     of b_w.)  
     	
	Parameters
	----------
	(a_b = Scalar hyperparameter
     b_b Scalar hyperparamter
     a_b = Vector of hyperparameters t x 1,
     b_w = Vector of hyperparameters t x 1,
     inv_w_1 = Vector of updates for 1/ E_q[w])
	 T = Total number of respnses. 
    
	Returns
	-------
	(a_b_star (1),
     b_b_star (1),
     bw_1 (1),
     log_bw_1 (1))
	"""
    
    a_b_star = T * a_w + a_b
    b_b_star = np.sum(inv_w_1) + b_b
    
    b_w_1 = a_b_star / b_b_star
    log_b_w_1 = sy.special.digamma(a_b_star) - np.log(b_b_star)
    
    return np.array([a_b_star, b_b_star, b_w_1, log_b_w_1])



def tau_update(a_tau, b_tau, T, nu, inv_sigma2_1, rho_2):
    """"
	Purpose
	-------
	(Calculate CAVI update for the parameters in the approximating density 
     of tau.)  
     	
	Parameters
	----------
	(a_tau = Scalar hyperparameter for tau,
     b_tau = Scalar hyperparameter for tau,
     T = Number of responses,
     nu =  Scalar hyperparameter for sigma2 ,
     inv_sigma2_1 = Vector of updates for sigma2 E_q[1/sigma2],
     rho_2 = Vector of updates for rho E_q[rho**2])
	 inv_w_1 = Vector of updates for 1/ E_q[w]).

	Returns
	-------
	(a_tau_star (1),
     b_tau_star (1),
     tau_1 (1),
     log_tau_1 (1))
	"""
    
    a_tau_star = a_tau + T * nu / 2
    b_tau_star = b_tau + (1 / 2) * np.sum(inv_sigma2_1 * (1 + ( np.sum(rho_2, axis=0))))

    tau_1 =  a_tau_star / b_tau_star
    log_tau_1 = sy.special.digamma(a_tau_star) - np.log(b_tau_star)   

    return np.array([a_tau_star, b_tau_star, tau_1, log_tau_1])    

######



def ELBO_A_y_cq(n, X, b_star_sigma2, inv_sigma2_1, tau_1, rho_2):    
    """"
	Purpose
	-------
	(Calculate the ELBO E_q[log p(y,z)] = E_q[log q(z)] componet involving Y 
     in the most computationally efficient approach.)  
     
	Parameters
	----------
	(n = number of individuals in the dataset,
     X = Design matrix, 
     b_star_sigma2 = Vector of values for sigma2 parameter update,
     inv_sigma2_1 = Vector of values for E_q[1/sigma2],
     tau_1 = Scalar update for tau E_q[tau],
     rho_2 = Vector of updates for E_q[rho**2].) 
           
	Returns
	-------
	(sum_t(A(y_t| beta_t, sigma^2_t, rho_t)).)
	"""
    ELBO_A_t = - (n / 2) * np.log(2 * math.pi) + (n / 2) * np.log(inv_sigma2_1) - (1 / 2) * inv_sigma2_1 * \
               (b_star_sigma2 - (tau_1 * (1 / 2) + tau_1 * (1 / 2) * np.sum(rho_2, axis= 0)))             
        
    ELBO_A = np.sum(ELBO_A_t)
    
    return np.array([ELBO_A])      



def ELBO_B_beta_gamma(gamma_1, log_inv_w_1, log_inv_sigma2_1, inv_sigma2_1, inv_w_1, beta_2, sigma2_beta, log_omega_1, log_minomega_1):
    """"
	Purpose
	-------
	(Calculate the ELBO E_q[log p(y,z)] = E_q[log q(z)] componet involving beta_ts
     and gamma_ts B.)
     
    Parameters
	----------
	(gamma_1 = Matrix of paramter values for E_q[gamma],
     log_inv_w_1 = Scalar 1 / E_q[log w]), 
     log_inv_sigma2_1 = Vector of values for E_q[1/(log sigma2)], 
     inv_sigma2_1 = Vector of values for E_q[1/sigma2], 
     inv_w_1 = Scalar for 1/ E_q[w],  
     beta_2 = Matrix of beta parameters sigma2_beta + mu_beta**2,
     sigma2_beta = Matrix of variance parameters for beta, 
     log_omega__1 = Vector of values for E_q[log omega],
     log_minomega_1 = Vector of values for E_q[log(1-omega)],
     ) 
           
	Returns
	-------
	(sum_ts(B(beta_ts, gamma_ts| sigma^2_t, w, omega_s)).)
        
	"""
    ##Due to potential / by 0
    indx1 = np.where(gamma_1 == 1)
    indx0 = np.where(gamma_1 == 0)
    indx = np.where( (gamma_1 != 1) & (gamma_1 != 0))
       
    ELBO_B_bg_ne01a = (gamma_1[indx[0],indx[1]] / 2) *  \
                          (log_inv_w_1[indx[0]] + 2 * log_omega_1[indx[1]] +  \
                          np.log(sigma2_beta[indx[0],indx[1]]) + 1 -  \
                          np.log(gamma_1[indx[0],indx[1]])) - \
                      beta_2[indx[0],indx[1]] * inv_w_1[indx[0]] 
                                   
    ELBO_B_bg_ne01b =   (1 - gamma_1[indx[0],indx[1]]) * (log_minomega_1[indx[1]] *  \
                                np.log(1 - gamma_1[indx[0],indx[1]]))
    
    ELBO_B_bg_e1 = (gamma_1[indx1[0],indx1[1]] / 2) *  \
                           (log_inv_w_1[indx1[0]] + 2 * log_omega_1[indx1[1]] +  \
                           np.log(sigma2_beta[indx1[0],indx1[1]]) + 1 -  \
                           np.log(gamma_1[indx1[0],indx1[1]])) - \
                   beta_2[indx1[0],indx1[1]] * inv_w_1[indx1[0]]
                                 
    ELBO_B_bg_e0 = (1 - gamma_1[indx0[0],indx0[1]]) * (log_minomega_1[indx0[1]] *  \
                                np.log(1 - gamma_1[indx0[0],indx0[1]]))                      
                    
    ELBO_B_beta_gamma = np.sum(ELBO_B_bg_ne01a) + np.sum(ELBO_B_bg_ne01b) + \
                        np.sum(ELBO_B_bg_e1) + np.sum(ELBO_B_bg_e0) 

    return np.array([ELBO_B_beta_gamma])


def ELBO_C_omega(a_omega, b_omega, a_omega_star, b_omega_star, log_omega_1, log_minomega_1):
    """"
	Purpose
	-------
	(Calculate the ELBO E_q[log p(y,z)] = E_q[log q(z)] componet involving 
     omega_s.) 
     	
	Parameters
	----------
	(T = Total number of responses,
     a_omega = Vector of hyperparameters for omega,
     b_omega = Vector of hyperparameters for omega
     gamma_1 = Vector of gamma updates)
	
	Returns
	-------
	(sum_s(C(omega_s)).)
	"""
    ELBO_C_omega_s = np.log(sy.special.beta(a_omega_star, b_omega_star)) - np.log(sy.special.beta(a_omega, b_omega)) \
                     +  (a_omega_star - a_omega) * log_omega_1 + (b_omega_star - b_omega) * log_minomega_1      

    ELBO_C_omega = np.sum(ELBO_C_omega_s)

    return np.array([ELBO_C_omega])


def ELBO_D_w(a_w, b_w_1, a_w_star, b_w_star, log_inv_w_1, inv_w_1, log_b_w_1):
    """"
	Purpose
	-------
	(Calculate the ELBO E_q[log p(y,z)] = E_q[log q(z)] componet involving 
     w.) 
     	
	Parameters
	----------
	(a_w = Scalar hyperparameter for w,
     b_w = Scalar hyperparameter for w,
     a_w_star = Scalar update for q(w),
     b_w_star = Scalar update for q(w),
     log_inv_w_1 = Scalar E_q[log 1/w],
     inv_w_1 = Scaler E_q[1/w].)
	
	Returns
	-------
	(D(w).)
	"""
    ELBO_D_w_t = a_w * log_b_w_1 - a_w_star * np.log(b_w_star) + np.log(sy.special.gamma(a_w_star)) \
                 - np.log(sy.special.gamma(a_w)) + (a_w - a_w_star) * log_inv_w_1 + (b_w_star - b_w_1) * inv_w_1  
        
    ELBO_D_w = np.sum(ELBO_D_w_t)
        
    return np.array([ELBO_D_w])  


def ELBO_F_sigma2(nu, T, tau_1, log_tau_1, a_star_sigma2, b_star_sigma2, inv_sigma2_1, log_inv_sigma2_1):
    """"
	Purpose
	-------
	(Calculate the ELBO E_q[log p(y,z)] = E_q[log q(z)] componet involving 
     sigma2.) 
     	
	Parameters
	----------
	(nu = Scalar hyperparameter for sigma2,
     T = Total number of responses,
     tau_1 = Scalar update E_q[tau,]
     log_tau_1 = Scalar update E_q[log tau],
     a_star_sigma2 = Vector of updates for q(sigma),
     b_star_sigma2 = Vector of updates for q(sigma),
     inv_sigma2_1 = Vector of values for E_q[1/sigma2],
     log_inv_sigma2_1 = Vector of values for E_q[log 1/sigma2].)
	
	Returns
	-------
	(F(sigma2).)
	"""
    ##issue come back t
    ELBO_F_sigma_t = ((nu - T + np.arange(1,T+1,1)) / 2) * (log_tau_1 - np.log(2)) - \
                     a_star_sigma2 * np.log(b_star_sigma2) - \
                     np.log(sy.special.gamma((nu - T + np.arange(1,T+1,1)) / 2)) + \
                     sy.special.gammaln(a_star_sigma2) + \
                     log_inv_sigma2_1 * (((nu - T + np.arange(1,T+1,1)) / 2)- a_star_sigma2) \
                     + inv_sigma2_1 * (b_star_sigma2 - tau_1 / 2) 

    ELBO_F_sigma = np.sum(ELBO_F_sigma_t)

    return np.array([ELBO_F_sigma])


sy.special.gammaln


def upper_tri(a, T):
    """"
	Purpose
	-------
	(Create upper triange from the array extracted from rho matrix.  
     For ELBO calculation.)
     	
	Parameters
	----------
	(T = Total number of reponses,
     a = array)
	
	Returns
	-------
	(U triangle with 0 on diagonal)
	"""
    mask = np.tri(T, dtype=bool, k = - 1).T 
    out = np.zeros((T, T), dtype = float)
    out[mask] = a
    
    return out


def ELBO_G_rho(tau_1, log_tau_1, inv_sigma2_1, log_inv_sigma2_1, rho_2, rho_sigma2, T):
    """"
	Purpose
	-------
	(Calculate the ELBO E_q[log p(y,z)] = E_q[log q(z)] componet involving 
     rho.) 
     	
	Parameters
	----------
	(tau_1 = Scalar update E_q[tau,]
     log_tau_1 = Scalar update E_q[log tau],
     inv_sigma2_1 = Vector of values for E_q[1/sigma2],
     log_inv_sigma2_1 = Vector of values for E_q[log 1/sigma2],
     rho_2 = Matrix of updates E_q[rho**2],
     rho_sigma2 = Matrix of updates for variance parameter q[rho]).)
	
	Returns
	-------
	(G(rho).)
	"""
    #Extract log of the upper triangle for function. Put in once working.
    a_log_rho_sigma2 = np.log(rho_sigma2[np.triu_indices(T, k = 1)])
    
    ELBO_G_rho_tk = (1 / 2) * (log_tau_1 + log_inv_sigma2_1[:,np.newaxis] \
                    - tau_1 * inv_sigma2_1[:,np.newaxis] * rho_2 \
                    + upper_tri(a_log_rho_sigma2, T) \
                    - 1)  

    ELBO_G_rho = np.sum(ELBO_G_rho_tk)

    return np.array([ELBO_G_rho])


def ELBO_H_tau(a_tau, b_tau, a_tau_star, b_tau_star, log_tau_1, tau_1):
    """"
	Purpose
	-------
	(Calculate the ELBO E_q[log p(y,z)] = E_q[log q(z)] componet involving 
     tau.) 
     	
	Parameters
	----------
	(a_tau = Scalar hyperparameter for p(tau),
     b_tau = Scalar hyperparameter for p(tau),
     a_tau_star = Scalar parameter for q(tau),
     b_tau_star = Scalar parameter for q(tau),
     log_tau_1 = Scalar update E_q[log tau],
     tau_1 = Scaler E_q[tau].)
	
	Returns
	-------
	(H(tau).)
	"""
    ELBO_H_tau = a_tau * np.log(b_tau) - a_tau_star * np.log(b_tau_star) \
                 + np.log(sy.special.gamma(a_tau_star)) - np.log(sy.special.gamma(a_tau)) \
                 + (a_tau_star - a_tau) * log_tau_1 + (b_tau_star - b_tau) * tau_1

    return np.array([ELBO_H_tau])



def ELBO_I_b_w(a_b, b_b, a_b_star, b_b_star, log_b_w_1, b_w_1):
    """"
	Purpose
	-------
	(Calculate the ELBO E_q[log p(y,z)] = E_q[log q(z)] componet involving 
     b_w.) 
     	
	Parameters
	----------
	(a_b = Scalar hyperparameter for p(b_w),
     b_b = Scalar hyperparameter for p(b_w),
     a_b_star = Scalar update for q(b_w),
     b_b_star = Scalar update for q(b_w),
     b_w_1 = Scalar E_q[b_w],
     log_b_w_1 = Scalar E_q[log b_w].)
	
	Returns
	-------
	(H(tau).)
	"""
    ELBO_H_tau = a_b * np.log(b_b) - a_b_star * np.log(b_b_star) - np.log(sy.special.gamma(a_b)) \
                 + np.log(sy.special.gamma(a_b_star)) + log_b_w_1 * (a_b - a_b_star) \
                 + b_w_1 * (b_b_star - b_b)

    return np.array([ELBO_H_tau])



def cov2corr(A):
    """"
	Purpose
	-------
	(Covariance matrix to correlation matrix computationally quicker 
     than D^-1 Sigma D^-1) 
     	
	Parameters
	----------
	(A = Covariance matrix.)
	
	Returns
	-------
	(Correlation matrix.)
	"""

    d = np.sqrt(A.diagonal())
    A = ((A.T/d).T)/d

    return A




