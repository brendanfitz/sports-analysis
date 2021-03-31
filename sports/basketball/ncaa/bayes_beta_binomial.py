# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 16:19:22 2021

@author: Brendan
"""

import numpy as np
from scipy.stats import beta
import matplotlib.pyplot as plt

class BayesBetaBinomial(object):
    
    def __init__(self, name, a_prior, b_prior):
        self.name = name
        self.a_prior = a_prior
        self.b_prior = b_prior
        self.prior_dist = beta(a_prior, b_prior)
        self.posterior_dist = None
        
        
    def update(self, x, n):
        self.a_posterior = self.a_prior + x
        self.b_posterior = self.b_prior + n - x
        
        self.posterior_dist = beta(self.a_posterior, self.b_posterior)
    
    def credible_interval(self, quantile=0.95):
        l = (1 - quantile)/2
        u = 1 - l   
        
        credible_interval = self.posterior_dist.ppf((l, u))
            
        return credible_interval

    def plot_posterior(self, title, ax=None):
        plt.style.use('ggplot')
        
        ls = "solid"
        
        fig, ax = plt.subplots(figsize=(12.8, 9.6))
        
        x = np.arange(0, 1.01, 0.001)
        y = self.posterior_dist.pdf(x)
        ax.plot(x, y, lw=2, ls=ls, color='navy', alpha=0.5)
        
        yrange_min, yrange_max = ax.get_ylim()
        yrange = yrange_max - yrange_min
        
        credible_interval = self.credible_interval()
        credible_interval_x = np.arange(*credible_interval, 0.001)
        credible_interval_y = self.posterior_dist.pdf(credible_interval_x)
        ax.fill_between(credible_interval_x, 0, credible_interval_y, color='lightskyblue', alpha=0.5)
        
        l, u = credible_interval
        
        ax.set_title(title, fontsize=24, fontweight='bold')
        ax.set_xlabel("Higher Ranked Team Win Probability", fontsize=16)
        
        ymin = (0 - yrange_min) / yrange
        
        posterior_mean = self.posterior_dist.mean()
        ymax = (self.posterior_dist.pdf(posterior_mean) - yrange_min) / yrange
        ax.axvline(x=self.posterior_dist.mean(), ymin=ymin, ymax=ymax, lw=2, alpha=0.5, color='navy', ls="dashed")
        
        ymax = (self.posterior_dist.pdf(l) - yrange_min) / yrange
        ax.axvline(x=l, ymin=ymin, ymax=ymax, lw=2, alpha=0.5, color='navy', ls="dashed")
        
        ymax = (self.posterior_dist.pdf(u) - yrange_min) / yrange
        ax.axvline(x=u, ymin=ymin, ymax=ymax, lw=2, alpha=0.5, color='navy', ls="dashed")
        
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.tick_params(axis='x', labelsize=14)
        ax.yaxis.set_ticklabels([])
    
        s = f"Lower Bound: {l:>6.1%}\nUpper Bound: {u:>6.1%}"
        text_y = ax.get_ylim()[1] * 0.8
        ax.text(x=0.05, y=text_y, s=s, fontsize=18)
        
        s = f"Posterior Mean: {posterior_mean:>6.1%}"
        ax.text(x=posterior_mean, y=-0.15, s=s, fontsize=14, ha="center", va="top")
        
        return ax