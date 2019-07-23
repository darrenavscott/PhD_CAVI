# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 08:38:13 2019

@author: darre
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
from pylab import *
from numba import jit
from sklearn.datasets import make_sparse_spd_matrix



###################################################################################################################
#Use python package to generate a densly packed matrix with lots of correlations
#This suggests this is only a problem when the VCv matrix is populated with large values accross
# more than 80% of its values

#Define the sizes
n = 300
p = 50
T = 30

##Random State is seed
prec = make_sparse_spd_matrix(T, alpha=.15,
                              smallest_coef=.4,
                              largest_coef=.7, random_state=1)

##This approach always produces a very dense VCV matrix as we are treating the PSD random
##matrix as the precision. Research at the bottom. 
cov = linalg.inv(prec)
d = np.sqrt(np.diag(cov))
cov /= d
cov /= d[:, np.newaxis]
precc = linalg.inv(cov)

D =  linalg.inv(np.diag(d))
R = np.dot(np.dot(D,cov),D)
##True reparam
backtransform(VCV = cov) 



##Design matrix
out = np.array([0,1,2])
np.random.seed(0)
Xu = np.random.choice(out, size = n * p, replace = True, p = ([1/3,1/3,1/3]))
Xu = Xu.reshape(n,p)

Ys = np.zeros(T * n).reshape(n,T)
Xtrack = np.zeros(T*4,dtype=int).reshape(T,4)
Btrack = np.zeros(T*4).reshape(T,4)
for j in np.arange(T):
    covar = np.sort(np.random.choice(np.arange(p), size=4, replace=False, p=None))
    betas = np.random.normal(0,3,4)
    Ys[:,j] = Xu[:,covar[0]]*betas[0] + Xu[:,covar[1]]*betas[1] + Xu[:,covar[2]]*betas[2] + Xu[:,covar[3]]*betas[3] 
    Xtrack[j,:] = covar
    Btrack[j,:] = betas

np.random.seed(17)
error = np.random.multivariate_normal(np.zeros(T), cov,(n)) 
np.cov(error,rowvar = False)

#Data 
Yu = Ys + error
Xu # Design Matrix


##Data
# Y reponses T = 30
# X = design matrix
Y = Yu - np.mean(Yu, axis = 0) 
X = Xu - np.mean(Xu, axis = 0)

##Intialise
# Choose priors and intialise the VI updates

#Beta and gamma T X p
#Intialise
mu_beta = np.reshape(np.repeat(0, T*p),(T,p)) #Intial T x p
sigma2_beta = np.reshape(np.repeat(5, T*p),(T,p))  #Initial T x p 
gamma_1 = np.reshape(np.repeat(2/9, T * p),(T, p))  #Initial T x p
# Choose gamma_1 = 0.22.. which is the mean of a beta distribution Beta(2,7) - quite diffusive

beta_1 = mu_beta * gamma_1  #Initial
beta_2 = (mu_beta + sigma2_beta) * gamma_1  #Initial

#Tau
a_tau = 1  #Parms
b_tau = 1  #Parms
tau_1 = a_tau / b_tau  #Initial
log_tau_1 = sy.special.digamma(a_tau) - np.log(b_tau)  #Initial


#Sigma # Use IG(0.5, 0.5 * MSE)
a_sigma2 = np.repeat(0.5, T)  #Initial
#b_sigma2 = np.repeat((results1.mse_resid * 0.5), T)  #Initial
b_sigma2 = np.repeat((5 * 0.5), T)  #Initial


#Distribution with mean = mean squared error and diffuse IG(1, 1 / (mse * 1))
nu = 2 * a_sigma2[0] - 1 + T  # Parms
inv_sigma2_1 = a_sigma2 / b_sigma2   #Initial
log_inv_sigma2_1 = sy.special.digamma(a_sigma2) - np.log(b_sigma2)  #Initial


#Make upper triangle matrix for rho_mu k x T
tri = np.reshape(np.zeros(T*T), (T,T))
rho_mu = tri 
tri2 = np.triu(np.repeat(2,T),k = 1)
rho_sigma2 = tri2 
rho_2 = rho_mu**2 + rho_sigma2**2  #Initial


#Omega 
a_omega = np.repeat(2,p)  #Parms
b_omega = np.repeat(7,p)  #Parms
omega_1 = a_omega / (b_omega + a_omega) #Initial
log_omega_1 = sy.special.digamma(a_omega) - sy.special.digamma(a_omega + b_omega)  #Initial
log_minomega_1 = sy.special.digamma(b_omega) - sy.special.digamma(a_omega + b_omega)  #Initial

#bw 
a_b = 1  #Parms
b_b = 2  #Parms
b_w_1 = a_b / b_b  #Initial
log_bw_1 = sy.special.digamma(a_b) - np.log(b_b) #Intial

#w_t
a_w = 0.5  #Intial value
inv_w_1 = np.repeat(a_w / b_w_1, T)  #Initial T x 1
log_inv_w_1 = np.repeat(sy.special.digamma(a_w) - np.log(b_w_1), T)  #Initial


#CAVI

def CAVI(Y, X, mu_beta, sigma2_beta, gamma_1, beta_1, beta_2, a_omega, b_omega, 
         omega_1, log_omega_1, log_minomega_1, a_w, b_w_1, inv_w_1, log_inv_w_1,
         a_sigma2, b_sigma2, inv_sigma2_1, log_inv_sigma2_1, rho_mu, rho_sigma2, 
         rho_2, a_tau, b_tau, tau_1, log_tau_1, nu):

    ##Intialise
    T = len(Y[0,:])
    p = len(X[0,:])
    n = len(Y[:,0])
    
     #Intialise
    e = 1
    ELBO = [-10000] #list 
    ELBO_diff = 100
    max_diff = 1
    
    while(ELBO_diff > max_diff):

        #Mu            
        out_beta_gamma = sigmab_beta_gamma_updateu(Y = Y, X = X, mu_beta= mu_beta, beta_1 = beta_1,
                                                   sigma2_beta = sigma2_beta, gamma_1 = gamma_1, 
                                                   inv_sigma2_1 = inv_sigma2_1, 
                                                   log_inv_sigma2_1 = log_inv_sigma2_1, 
                                                   rho_mu = rho_mu, rho_2 = rho_2, 
                                                   inv_w_1 = inv_w_1, 
                                                   log_inv_w_1 = log_inv_w_1, 
                                                   log_omega_1 = log_omega_1,
                                                   log_minomega_1 = log_minomega_1,
                                                   T = T, p = p, beta_2 = beta_2)    
          
        mu_beta = out_beta_gamma[0] 
        sigma2_beta = out_beta_gamma[1]                 
        gamma_1 = out_beta_gamma[2]   
        beta_1 = out_beta_gamma[3]        
        beta_2 = out_beta_gamma[4]
    
        #Omega
        out_omega = omega_update(gamma_1 = gamma_1, T = T, a_omega = a_omega, 
                                 b_omega = b_omega)
        a_omega_star = out_omega[0]
        b_omega_star = out_omega[1]
        omega_1 = out_omega[2]
        log_omega_1 = out_omega[3]
        log_minomega_1 = out_omega[4]
        
        #W
        out_w = w_update(a_w = a_w, b_w_1 = b_w_1, gamma_1 = gamma_1, 
                         beta_2 = beta_2)
        a_w_star = out_w[0]
        b_w_star = out_w[1]
        inv_w_1 = out_w[2]
        log_inv_w_1 = out_w[3]
        
        #b_w
        out_bw = bw_update(a_w = a_w, a_b = a_b, b_b = b_b, T = T, 
                           inv_w_1 = inv_w_1)
        a_b_star = out_bw[0]
        b_b_star = out_bw[1]
        b_w_1 = out_bw[2]
        log_b_w_1 = out_bw[3]
        
        #Tau
        out_tau = tau_update(a_tau = a_tau, b_tau = b_tau, T = T, nu = nu, 
                                inv_sigma2_1 = inv_sigma2_1, rho_2 = rho_2)
        a_tau_star = out_tau[0]
        b_tau_star = out_tau[1]
        tau_1 = out_tau[2]
        log_tau_1 = out_tau[3]
       
        
        #Sigma
        out_sigma2 = sigma2_updateu2(Y = Y, X = X, nu = nu, T =T, n = n, gamma_1 = gamma_1, 
                                      tau_1 = tau_1, rho_mu = rho_mu, rho_2 = rho_2, 
                                      inv_w_1 = inv_w_1, beta_1 = beta_1, 
                                      beta_2 = beta_2, p = p)
        a_star_sigma2 = out_sigma2[0].flatten()
        b_star_sigma2 = out_sigma2[1].flatten()
        inv_sigma2_1 = out_sigma2[2].flatten()
        log_inv_sigma2_1 = out_sigma2[3].flatten()
    
        #Rho
        out_rho = rho_updateu(tau_1 = tau_1, Y = Y, X = X, beta_1 = beta_1, 
                              beta_2 = beta_2, inv_sigma2_1 = inv_sigma2_1, T = T,
                              rho_mu = rho_mu, p = p)
        rho_mu = out_rho[0]
        rho_sigma2 = out_rho[1]
        rho_2 = out_rho[2]
    
    
        #ELBO
        
        A = ELBO_A_y_cq(n = n, X = X, b_star_sigma2 = b_star_sigma2, 
                           inv_sigma2_1 = inv_sigma2_1, tau_1 = tau_1, 
                           rho_2 = rho_2, log_inv_sigma2_1 = log_inv_sigma2_1) 
        B = ELBO_B_beta_gamma(gamma_1 = gamma_1, log_inv_w_1 = log_inv_w_1, 
                                 log_inv_sigma2_1 = log_inv_sigma2_1, 
                                 inv_sigma2_1 = inv_sigma2_1,  
                                 inv_w_1 = inv_w_1, beta_2 = beta_2, 
                                 sigma2_beta = sigma2_beta, log_omega_1 = log_omega_1,  
                                 log_minomega_1 = log_minomega_1) 
        C = ELBO_C_omega(a_omega = a_omega, b_omega = b_omega, a_omega_star = a_omega_star, 
                            b_omega_star = b_omega_star, log_omega_1 = log_omega_1, 
                            log_minomega_1 = log_minomega_1) 
        D = ELBO_D_w(a_w = a_w, b_w_1 = b_w_1, a_w_star = a_w_star, 
                     b_w_star = b_w_star, log_inv_w_1 = log_inv_w_1, 
                     inv_w_1 = inv_w_1, log_b_w_1 = log_b_w_1)
        F = ELBO_F_sigma2(nu = nu, T = T, tau_1 = tau_1, log_tau_1 = log_tau_1, 
                             a_star_sigma2 = a_star_sigma2, b_star_sigma2 = b_star_sigma2, 
                             inv_sigma2_1 = inv_sigma2_1, log_inv_sigma2_1 = log_inv_sigma2_1)
        G = ELBO_G_rho(tau_1 = tau_1, log_tau_1 = log_tau_1, inv_sigma2_1 = inv_sigma2_1, 
                          log_inv_sigma2_1 = log_inv_sigma2_1, rho_2 = rho_2, 
                          rho_sigma2 = rho_sigma2, T = T) 
        H = ELBO_H_tau(a_tau = a_tau, b_tau = b_tau, a_tau_star = a_tau_star,
                       b_tau_star = b_tau_star, log_tau_1 = log_tau_1, 
                       tau_1 = tau_1) 
        I = ELBO_I_b_w(a_b = a_b, b_b = b_b, a_b_star = a_b_star, 
                       b_b_star = b_b_star, log_b_w_1 = log_b_w_1, b_w_1 = b_w_1)
        
        ELBO.append(np.sum([A, B, C, D, F, G, H, I]))
    
        ELBO_diff = np.abs(ELBO[e] - ELBO[e-1])
   
        e = e + 1      
    
    return [mu_beta, sigma2_beta, gamma_1, beta_1, beta_2, 
                     a_star_sigma2, b_star_sigma2, rho_mu, rho_sigma2, 
                     inv_sigma2_1, ELBO]



CAVout = CAVI(
        Y = Y, X = X, mu_beta = mu_beta, sigma2_beta = sigma2_beta, 
        gamma_1 = gamma_1, beta_1 = beta_1, beta_2 = beta_2, a_omega = a_omega, 
        b_omega = b_omega, omega_1 = omega_1, log_omega_1 = log_omega_1, 
        log_minomega_1 = log_minomega_1, a_w = a_w, 
        b_w_1 = b_w_1, inv_w_1 = inv_w_1, log_inv_w_1 = log_inv_w_1,
        a_sigma2 = a_sigma2, b_sigma2 = b_sigma2, 
        inv_sigma2_1 = inv_sigma2_1, log_inv_sigma2_1 = log_inv_sigma2_1, 
        rho_mu = rho_mu, rho_sigma2 = rho_sigma2, 
        rho_2 = rho_2, a_tau = a_tau, 
        b_tau = b_tau, tau_1 = tau_1, log_tau_1 = log_tau_1, nu = nu)


##Change output to list as the np.array had a broadcasting issue (not sure why as this should be fine)

##Plot of the ELBO
ELBO = CAVout[10]
np.array([ELBO])
ELBOplot = ELBO[0:len(ELBO)]
itera = np.arange(0, len(ELBO), 1)
plot(itera, ELBOplot)
xlabel('Iteration (i)')
ylabel('ELBO')
title('ELBO')
grid(True)
show()


#Estimated mean sigma2_t values
1 / (CAVout[9])
##True rho and sigma2_t
backtransform(VCV = cov)  


#Paramter - point estimates 
mu_beta = CAVout[0]
gamma = CAVout[2] 

##Estimates of Beta
rowi = np.repeat(np.arange(T),4)
mu_beta[rowi,Xtrack.flatten()].reshape(T,4)
##Truth
Btrack

##Estimates of gamma
gamma[rowi,Xtrack.flatten()]
##Truth 
Xtrack



#############VCV check

##Here I scale the Random matrix to be the correlation matrix rather than the precision matrix 
prng = np.random.RandomState(1)
R1 = make_sparse_spd_matrix(30, alpha=.65,
                              smallest_coef=.4,
                              largest_coef=.7)
d1 = np.sqrt(np.diag(R1))
R1 /= d1
R1 /= d1[:, np.newaxis]
R1 ##This is the correlation matrix 

D = np.diag(np.repeat(2,30))
VCV = np.dot(np.dot(D, R1), D) 
backtransform(VCV = VCV) 
  



































