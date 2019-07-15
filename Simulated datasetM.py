# -*- coding: utf-8 -*-
"""
Created on Mon May 27 14:41:46 2019

@author: darre

Generating a dataset 
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
#from CAVIfun import *


#######Create Dataset with correlated noise###############################################

#Covariate values
out = np.array([0,1,2])

##Design matrix
np.random.seed(0)
Xu = np.random.choice(out, size=2500, replace=True, p=([1/3,1/3,1/3]))
Xu = Xu.reshape(100,25)

#Responses
y_1 = 2.785 * Xu[:, 5] + 3.455 * Xu[:, 17] + 4.678 * Xu[:, 24]
y_2 = 8.988 * Xu[:, 15] + 4 * Xu[:, 20] - 3.5 * Xu[:, 21]
y_3 = -2.55 * Xu[:, 4] - 15 * Xu[:, 10] - 3.77 * Xu[:, 23]
y_4 = -1.55 * Xu[:, 9] + 5 * Xu[:, 14] - 3.2 * Xu[:, 20]

#Noise 
#Correlation
R = np.array([1,   0.4, 0.9, 0.35, 
              0.4,   1, 0.5,  0.5,
              0.9, 0.5,   1,  0.2,
              0.35,0.5, 0.2,    1]).reshape(4,4)
D = np.diag([1, 1, 1, 1])



#Covariance
C = np.dot(np.dot(D, R), D) 
np.random.seed(17)
error = np.random.multivariate_normal(([0,0,0,0]),C,(100)) 

#Data 
Yu = np.column_stack((y_1,y_2,y_3,y_4)) + error
Xu # Design Matrix


#Correlation check
Dt = np.sqrt(np.diag(C))
DInv = linalg.inv(np.diag(Dt))
Rcheck = np.dot(np.dot(DInv, C), DInv)

##Check Covariance
Ymu = np.column_stack((y_1,y_2,y_3,y_4))
tuerror = Yu-Ymu
tucov = np.cov(tuerror.T)
Dt = np.sqrt(np.diag(tucov))
DInv = linalg.inv(np.diag(Dt))

np.dot(np.dot(DInv, tucov), DInv) 

##Data
# Y reponses T = 4
# X = design matrix
Y = Yu - np.mean(Yu, axis = 0) 
X = Xu - np.mean(Xu, axis = 0)

##Intialise
T = len(Y[0,:])
p = len(X[0,:])
n = len(Y[:,0])

##Response 1
model1 = sm.OLS(Y[:,0],X)
results1 = model1.fit()
model2 = sm.OLS(Y[:,1],X)
results2 = model2.fit()
model3 = sm.OLS(Y[:,2],X)
results3 = model3.fit()
model4 = sm.OLS(Y[:,3],X)
results4 = model4.fit()

##Residuals used for rho
results1.resid

#Summary 
print(results1.summary())
print(results2.summary())
print(results3.summary())





#Beta and gamma T X p
#mu_beta =  np.row_stack((results1.params, results2.params, results3.params,results4.params))
mu_beta = np.reshape(np.repeat(0, T*p),(T,p))
#sigma2_beta = np.row_stack((results1.bse, results2.bse, results3.bse, results4.bse))
sigma2_beta = np.reshape(np.repeat(5, T*p),(T,p))  #Initial T x p 
gamma_1 = np.reshape(np.repeat(2/9, T * p),(T, p))  #Initial
# Choose gamma_1 = 0.22.. which is the mean of a beta distribution Beta(2,7) - quite diffusive



beta_1 = mu_beta * gamma_1  #Initial
beta_2 = (mu_beta + sigma2_beta) * gamma_1  #Initial


#Sigma # Use IG(0.5, 0.5 * MSE)
a_sigma2 = np.repeat(0.5, T)  #Initial
b_sigma2 = np.repeat((results1.mse_resid * 0.5), T)  #Initial
#Distribution with mean = mean squared error and diffuse IG(1, 1 / (mse * 1))
nu = 2 * a_sigma2[0] - 1 + T  # Parms
inv_sigma2_1 = a_sigma2 / b_sigma2   #Initial
log_inv_sigma2_1 = sy.special.digamma(a_sigma2) - np.log(b_sigma2)  #Initial


#Rho
modelrh1 = sm.OLS(Y[:,1], np.column_stack((X, results1.resid)))
resultsrh1 = modelrh1.fit()
rho_mu_1 = np.concatenate([np.array([resultsrh1.params[25]]), np.zeros(3)])
rho_sigma_1 = np.concatenate([np.array([resultsrh1.bse[25]]), np.zeros(3)])

modelrh2 = sm.OLS(Y[:,2], np.column_stack((X, results1.resid, results2.resid)))
resultsrh2 = modelrh2.fit()
rho_mu_2 = np.concatenate([resultsrh2.params[25:27], np.zeros(2)])
rho_sigma_2 = np.concatenate([resultsrh2.bse[25:27], np.zeros(2)])

modelrh3 = sm.OLS(Y[:,3], np.column_stack((X, results1.resid, results2.resid, results3.resid)))
resultsrh3 = modelrh3.fit()
rho_mu_3 = np.concatenate([resultsrh3.params[25:28], np.zeros(1)])
rho_sigma_3 = np.concatenate([resultsrh3.bse[25:28], np.zeros(1)])



#Make upper triangle matrix for rho_mu
tri = np.reshape(np.zeros(T*T), (T,T))
tri[:,1] = rho_mu_1
tri[:,2] = rho_mu_2
tri[:,3] = rho_mu_3
rho_mu = tri 

tri = np.reshape(np.zeros(T*T), (T,T))
tri[:,1] = rho_sigma_1
tri[:,2] = rho_sigma_2
tri[:,3] = rho_sigma_3
rho_sigma2 = tri 

rho_2 = rho_mu**2 + rho_sigma2**2  #Initial


#Omega 
a_omega = np.repeat(2,p)  #Parms
b_omega = np.repeat(7,p)  #Parms
omega_1 = a_omega / (b_omega + a_omega) #Initial
log_omega_1 = sy.special.digamma(a_omega) - sy.special.digamma(a_omega + b_omega)  #Initial
log_minomega_1 = sy.special.digamma(b_omega) - sy.special.digamma(a_omega + b_omega)  #Initial


 #Tau
a_tau = 1  #Parms
b_tau = 1  #Parms
tau_1 = a_tau / b_tau  #Initial
log_tau_1 = sy.special.digamma(a_tau) - np.log(b_tau)  #Initial

#w
a_w = 1  #Parms
b_w = 1 #Parms
inv_w_1 = a_w/ b_w  #Initial
log_inv_w_1 = sy.special.digamma(a_w) - np.log(b_w)  #Initial


########################################################################################################

##Function with functions inside =- Problem 

#CAVI

def CAVI(Y, X, mu_beta, sigma2_beta, gamma_1, beta_1, beta_2, a_omega, b_omega, 
         omega_1, log_omega_1, log_minomega_1, a_w, b_w, inv_w_1, log_inv_w_1,
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
    max_diff = 0.001
    
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
                                                   T = T)    
          
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
        out_w = w_update(a_w = a_w, b_w = b_w, gamma_1 = gamma_1, 
                         beta_2 = beta_2)
        a_w_star = out_w[0]
        b_w_star = out_w[1]
        inv_w_1 = out_w[2]
        log_inv_w_1 = out_w[3]
        
        #Tau
        out_tau = tau_update(a_tau = a_tau, b_tau = b_tau, T = T, nu = nu, 
                                inv_sigma2_1 = inv_sigma2_1, rho_2 = rho_2)
        a_tau_star = out_tau[0]
        b_tau_star = out_tau[1]
        tau_1 = out_tau[2]
        log_tau_1 = out_tau[3]
        
        #Sigma
        out_sigma2 = sigma2_updateu(Y = Y, X = X, nu = nu, T =T, n = n, gamma_1 = gamma_1, 
                                      tau_1 = tau_1, rho_mu = rho_mu, rho_2 = rho_2, 
                                      inv_w_1 = inv_w_1, beta_1 = beta_1, 
                                      beta_2 = beta_2)
        a_star_sigma2 = out_sigma2[0].flatten()
        b_star_sigma2 = out_sigma2[1].flatten()
        inv_sigma2_1 = out_sigma2[2].flatten()
        log_inv_sigma2_1 = out_sigma2[3].flatten()
    
        #Rho
        out_rho = rho_updateu(tau_1 = tau_1, Y = Y, X = X, beta_1 = beta_1, 
                            beta_2 = beta_2, inv_sigma2_1 = inv_sigma2_1, T = T,
                            rho_mu = rho_mu)
        rho_mu = out_rho[0]
        rho_sigma2 = out_rho[1]
        rho_2 = out_rho[2]
    
    
        #ELBO
        
        A = ELBO_A_y_cq(n = n, X = X, b_star_sigma2 = b_star_sigma2, 
                           inv_sigma2_1 = inv_sigma2_1, tau_1 = tau_1, 
                           rho_2 = rho_2) 
        B = ELBO_B_beta_gamma(gamma_1 = gamma_1, log_inv_w_1 = log_inv_w_1, 
                                 log_inv_sigma2_1 = log_inv_sigma2_1, 
                                 inv_sigma2_1 = inv_sigma2_1,  
                                 inv_w_1 = inv_w_1, beta_2 = beta_2, 
                                 sigma2_beta = sigma2_beta, log_omega_1 = log_omega_1,  
                                 log_minomega_1 = log_minomega_1) 
        C = ELBO_C_omega(a_omega = a_omega, b_omega = b_omega, a_omega_star = a_omega_star, 
                            b_omega_star = b_omega_star, log_omega_1 = log_omega_1, 
                            log_minomega_1 = log_minomega_1) 
        D = ELBO_D_w(a_w = a_w, b_w = b_w, a_w_star = a_w_star, b_w_star = b_w_star, 
                        log_inv_w_1 = log_inv_w_1, inv_w_1 = inv_w_1) 
        E = ELBO_F_sigma2(nu = nu, T = T, tau_1 = tau_1, log_tau_1 = log_tau_1, 
                             a_star_sigma2 = a_star_sigma2, b_star_sigma2 = b_star_sigma2, 
                             inv_sigma2_1 = inv_sigma2_1, log_inv_sigma2_1 = log_inv_sigma2_1)
        F = ELBO_G_rho(tau_1 = tau_1, log_tau_1 = log_tau_1, inv_sigma2_1 = inv_sigma2_1, 
                          log_inv_sigma2_1 = log_inv_sigma2_1, rho_2 = rho_2, 
                          rho_sigma2 = rho_sigma2, T = T) 
        G = ELBO_H_tau(a_tau = a_tau, b_tau = b_tau, a_tau_star = a_tau_star, 
                          b_tau_star = b_tau_star, log_tau_1 = log_tau_1, 
                          inv_w_1 = inv_w_1, b_w = b_w, b_w_star = b_w_star) 
        
        ELBO.append(np.sum([A,B,C,D,E,F,G]))
    
        ELBO_diff = np.abs(ELBO[e] - ELBO[e-1])
   
        e = e + 1      
    
    return np.array([mu_beta, sigma2_beta, gamma_1, beta_1, beta_2, 
                     a_star_sigma2, b_star_sigma2, rho_mu, rho_sigma2, ELBO])


CAVout = CAVI(Y = Y, X = X, mu_beta = mu_beta, sigma2_beta = sigma2_beta,
              gamma_1 = gamma_1, beta_1 = beta_1, beta_2 = beta_2, a_omega = a_omega, 
              b_omega = b_omega, omega_1 = omega_1, log_omega_1 = log_omega_1, 
              log_minomega_1 = log_minomega_1, a_w = a_w, 
              b_w = b_w, inv_w_1 = inv_w_1, log_inv_w_1 = log_inv_w_1,
              a_sigma2 = a_sigma2, b_sigma2 = b_sigma2, 
              inv_sigma2_1 = inv_sigma2_1, log_inv_sigma2_1 = log_inv_sigma2_1, 
              rho_mu = rho_mu, rho_sigma2 = rho_sigma2, 
              rho_2 = rho_2, a_tau = a_tau, 
              b_tau = b_tau, tau_1 = tau_1, log_tau_1 = log_tau_1, nu = nu)




##Plot of the ELBO
ELBO = CAVout[9]
ELBOplot = ELBO[1:len(ELBO)]
itera = np.arange(1, len(ELBO), 1)
plot(itera, ELBOplot)
xlabel('Iteration (i)')
ylabel('ELBO')
title('ELBO')
grid(True)
show()


#Paramter - point estimates 
mu_beta = CAVout[0]
gamma = CAVout[2] 

##Quick check
mu_beta_1 = np.array([mu_beta[0, 5], mu_beta[0, 17], mu_beta[0, 24]])
gamma_1 = np.array([gamma[0, 5], gamma[0, 17], gamma[0, 24]])
# y_1 = 2.785 * Xu[:, 5] + 3.455 * Xu[:, 17] + 4.678 * Xu[:, 24]

mu_beta_2 = np.array([mu_beta[1, 15], mu_beta[1, 20], mu_beta[1, 21]])
gamma_2 = np.array([gamma[1, 15], gamma[1, 20], gamma[1, 21]])
# y_2 = 8.988 * Xu[:, 15] + 4 * Xu[:, 20] - 3.5 * Xu[:, 21]

mu_beta_3 = np.array([mu_beta[2, 4], mu_beta[2, 10], mu_beta[2, 23]])
gamma_3 = np.array([gamma[2, 4], gamma[2, 10], gamma[2, 23]])
#y_3 = -2.55 * Xu[:, 4] - 15 * Xu[:, 10] - 3.77 * Xu[:, 23]

mu_beta_4 = np.array([mu_beta[3, 9], mu_beta[3, 14], mu_beta[3, 20]])
gamma_4 = np.array([gamma[3, 9], gamma[3, 14], gamma[3, 20]])
#y_4 = -1.55 * Xu[:, 9] + 5 * Xu[:, 14] - 3.2 * Xu[:, 20]




##recover the VCV matrix 

a_star_sigma2 = CAVout[5]
b_star_sigma2 = CAVout[6]
rho_mu = CAVout[7]
rho_sigma2 = CAVout[8]

##Number of samples 
itera = 30000

#3d array
outcv = np.zeros([itera,4,4])
outcorr = np.zeros([itera,4,4])

for i in np.arange(itera):
    
    #Compute C matrix 
    
    #C1
    sigma2_1 = sy.stats.invgamma.rvs(a = a_star_sigma2[0], scale = b_star_sigma2[0])
    c1 = sigma2_1
    #rho_1 = 0
    
    #C2
    sigma2_2 = sy.stats.invgamma.rvs(a = a_star_sigma2[1], scale = b_star_sigma2[1])    
    rho_2 = np.random.normal(rho_mu[0,1], np.sqrt(rho_sigma2[0,1]))
    
    c12 = rho_2 * sigma2_1
    c2 = sigma2_2 + c12 * (c1**(-1)) * c12
    ix = np.indices((2,2))[:, ~np.tri(2, k=0, dtype=bool)]
    C2 = np.diag([c1,c2])
    C2[ix[0], ix[1]] = c12
    C2[ix[1], ix[0]] = c12
    
    #C3
    sigma2_3 = sy.stats.invgamma.rvs(a = a_star_sigma2[2], scale = b_star_sigma2[2])
    rho_31 = np.random.normal(rho_mu[0,2], np.sqrt(rho_sigma2[0,2]))
    rho_32 = np.random.normal(rho_mu[1,2], np.sqrt(rho_sigma2[1,2]))
    rho_3 = np.array([rho_31, rho_32])
    
    c3v = np.dot(C2.T, rho_3) 
    c3 = np.array([sigma2_3 + np.dot(np.dot(c3v.T, np.linalg.inv(C2)), c3v)])
    C3 = np.hstack((C2,c3v.reshape(2,1)))
    C3 = np.vstack((C3,np.concatenate([c3v, c3])))
    
    #C4
    sigma2_4 = sy.stats.invgamma.rvs(a = a_star_sigma2[3], scale = b_star_sigma2[3])        
    rho_41 = np.random.normal(rho_mu[0,3], np.sqrt(rho_sigma2[0,3]))
    rho_42 = np.random.normal(rho_mu[1,3], np.sqrt(rho_sigma2[1,3]))
    rho_43 = np.random.normal(rho_mu[2,3], np.sqrt(rho_sigma2[2,3]))
    rho_4 = np.array([rho_41, rho_42, rho_43])
    
    c4v = np.dot(C3.T, rho_4)
    c4 = np.array([sigma2_4 + np.dot(np.dot(c4v.T, np.linalg.inv(C3)), c4v)])
    C4 = np.hstack((C3,c4v.reshape(3,1)))
    C4 = np.vstack((C4,np.concatenate([c4v, c4])))
    
    ##Each outcv is the variance covariance matrix
    outcv[i,:,:] = C4

    ##Each outcorr is the correlation matrix
    outcorr[i,:,:] = cov2corr(C4)
    




outcorr.mean(axis=0)
outcv.mean(axis=0)
























