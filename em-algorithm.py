# -*- coding: utf-8 -*-
import numpy as np
import random
import statistics 
def gaussian_pdf(x, mu, sigma):
    return (1 / np.sqrt(2 * np.pi * sigma**2)) * np.exp(-((x - mu)**2 / (2 * sigma**2))) 

def exponential_pdf(x, lambd):
    if x >= 0:
        return lambd * np.exp(-(lambd * x)) 
    else:
        return 0 

def mixture_pdf(x, alpha, mu, sigma, lambd):
    return alpha * gaussian_pdf(x, mu, sigma) + (1 - alpha) * exponential_pdf(x, lambd)


def log_likelihood(F, T, alpha, mu, sigma, lambd):
    n = len(F)
    log_likelihood = 0
    for i in range(n):
        fi = F[i]
        ti = T[i]
        log_likelihood += np.log(alpha * gaussian_pdf(fi, mu, sigma)) + np.log((1 - alpha) * exponential_pdf(fi, lambd))
    return log_likelihood




def em_algorithm(F, alpha, mu, sigma, lambd, max_iter=1000, epsilon=1e-6):
    print("alpha", alpha)
    ll_old = 0
    for iter in range(max_iter):
        # E-step
        t = np.zeros(len(F))
        for i in range(len(F)):
            fi = F[i]
            
            #posterior
            p = mixture_pdf(fi, alpha, mu, sigma, lambd)
            t[i] = (alpha * gaussian_pdf(fi, mu, sigma)) / (p)
        
        
        # M-step
        alpha = np.mean(t)

        mu = np.sum(t * F) / np.sum(t)
        sigma = np.sqrt(np.sum(t * (F - mu)**2) / np.sum(t))
        lambd = np.sum(1-t) / np.sum((1-t) * (F))
        
        # Check convergence
        ll = log_likelihood(F, t, alpha, mu, sigma, lambd)

        
        if abs(ll - ll_old) < epsilon:
            break
        ll_old = ll
    return alpha, mu, sigma, lambd, t, ll

 


def converting_outlier_score (samples):
    #initialization
    max_ll = float('-inf')
    
    #iteration for different random inizialization values
    for iter in range (10):

        #randomly inizialization mean by a uniform distrubtion
        mu_rand = random.uniform(min(samples), max(samples))
        
        #randomly inizialization sigma by a sample vairance of a sub-set equal to the 10% of the entire sample
        sort_samp = sorted(samples)
        sigma_rand = np.sqrt(statistics.variance(random.sample(sort_samp, round(0.1*len(samples)))))
        #randomly inizialization lambda by a uniform distrubtion
        lambda_rand = random.uniform(0.5, 3)
        
        alpha, mu, sigma, lambd, t, ll = em_algorithm(F = samples, alpha = 0.5, mu = mu_rand, sigma = sigma_rand, lambd = lambda_rand)
  
        #save best estimation
        if ll > max_ll:
            max_ll = ll
            alpha_final, mu_final, sigma_final, lambd_final, t_final, ll_final = alpha, mu, sigma, lambd, t, ll

    return alpha_final, mu_final, sigma_final, lambd_final, t_final, ll_final 







