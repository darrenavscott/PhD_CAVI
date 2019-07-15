# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 13:31:38 2019

@author: Darren
"""

#######Create Dataset with correlated noise##################

#Covariate values

out = np.array([0,1,2])

##Intialise
T = 4
p = 25
n = 100

##Design matrix
np.random.seed(0)
X = np.random.choice(out, size=2500, replace=True, p=([1/3,1/3,1/3])).reshape(100,25)
sigmat = np.array([1,1,1,1])
rho = np.array([0, 0.5, 1.3,  2.3, 
                0,   0, 2,  1.4,
                0,   0,   0,  3.17,
                0,   0,   0,     0]).reshape(4,4)

#Responses
y_1 = 2.785 * Xu[:, 5] + 3.455 * Xu[:, 17] + 4.678 * Xu[:, 24]
y_e1 = np.random.normal(0, sigmat[0], n)
y_d1 = y_1 + y_e1

y_2 = 8.988 * Xu[:, 15] + 4 * Xu[:, 20] - 3.5 * Xu[:, 21] + rho[0,1] * y_e1
y_e2 = np.random.normal(0, sigmat[1], n)
y_d2 = y_2 + y_e2

y_3 = -2.55 * Xu[:, 4] - 15 * Xu[:, 10] - 3.77 * Xu[:, 23] + rho[0,2] * y_e1 + rho[1,2] * y_e2 
y_e3 = np.random.normal(0, sigmat[2], n)
y_d3 = y_3 + y_e3

y_4 = -1.55 * Xu[:, 9] + 5 * Xu[:, 14] - 3.2 * Xu[:, 20] + rho[0,3] * y_e1 + rho[1,3] * y_e2 + rho[2,3] * y_e3
y_e4 = np.random.normal(0,sigmat[3], n)
y_d4 = y_4 + y_e4

Yu = np.column_stack([y_d1, y_d2, y_d3, y_d4])

#Centre Y
Y = Yu - np.mean(Yu, axis = 0)
X = Xu - np.mean(Xu, axis = 0)

##Intialise
T = len(Y[0,:])
p = len(X[0,:])
n = len(Y[:,0])

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



########################################################################################################

##Function with functions inside =- Problem 

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
    max_diff = 0.1
    
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
    
        ELBO_diff = (ELBO[e] - ELBO[e-1])
   
        e = e + 1      
    
    return np.array([mu_beta, sigma2_beta, gamma_1, beta_1, beta_2, 
                     inv_sigma2_1, rho_mu, rho_sigma2, ELBO])


CAVout = CAVI(Y = Y, X = X, mu_beta = mu_beta, sigma2_beta = sigma2_beta,
              gamma_1 = gamma_1, beta_1 = beta_1, beta_2 = beta_2, a_omega = a_omega, 
              b_omega = b_omega, omega_1 = omega_1, log_omega_1 = log_omega_1, 
              log_minomega_1 = log_minomega_1, a_w = a_w, 
              b_w_1 = b_w_1, inv_w_1 = inv_w_1, log_inv_w_1 = log_inv_w_1,
              a_sigma2 = a_sigma2, b_sigma2 = b_sigma2, 
              inv_sigma2_1 = inv_sigma2_1, log_inv_sigma2_1 = log_inv_sigma2_1, 
              rho_mu = rho_mu, rho_sigma2 = rho_sigma2, 
              rho_2 = rho_2, a_tau = a_tau, 
              b_tau = b_tau, tau_1 = tau_1, log_tau_1 = log_tau_1, nu = nu)


E = (CAVout[5] / CAVout[6])**(-1)

##Plot of the ELBO
ELBO = CAVout[9]
np.array([ELBO])
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
mu_beta1 = np.array([mu_beta[0, 5], mu_beta[0, 17], mu_beta[0, 24]])
gamma1 = np.array([gamma[0, 5], gamma[0, 17], gamma[0, 24]])
#y_1 = 2.785 * Xu[:, 50] + 3.455 * Xu[:, 170] + 4.678 * Xu[:, 24]


mu_beta2 = np.array([mu_beta[1, 15], mu_beta[1, 20], mu_beta[1, 21]])
gamma2 = np.array([gamma[1, 15], gamma[1, 20], gamma[1, 21]])
# y_2 = 8.988 * Xu[:, 15] + 4 * Xu[:, 20] - 3.5 * Xu[:, 21]

mu_beta3 = np.array([mu_beta[2, 4], mu_beta[2, 10], mu_beta[2, 23]])
gamma3 = np.array([gamma[2, 4], gamma[2, 10], gamma[2, 23]])
#y_3 = -2.55 * Xu[:, 4] - 15 * Xu[:, 100] - 3.77 * Xu[:, 23]

mu_beta4 = np.array([mu_beta[3, 9], mu_beta[3, 14], mu_beta[3, 20]])
gamma4 = np.array([gamma[3, 9], gamma[3, 14], gamma[3, 20]])
#y_4 = -1.55 * Xu[:, 9] + 5 * Xu[:, 14] - 3.2 * Xu[:, 20]

rho_mu = CAVout[7]
rho




















