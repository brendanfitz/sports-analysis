# -*- coding: utf-8 -*-
"""
Created on Sun Feb 14 11:20:57 2021

@author: Brendan Non-Admin
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import t

class SingleMeanHypothesisTester(object):
    """ Theoretical Method """
    
    def __init__(self, data, xname, yname, conf_level=0.95):
        self.data = data
        self.xname, self.yname = xname, yname
        self.conf_level = conf_level
        
        self.calcs = (self.data.groupby(self.xname)
            .agg({
                self.yname: ['mean', 'std', 'count']
            })
            .xs(self.yname, axis=1)
        )
        
        self.y_bar1, self.y_bar2 = self.calcs.loc[:, 'mean']
        self.n1, self.n2 = self.calcs.loc[:, 'count']
        self.s1, self.s2 = self.calcs.loc[:, 'std']
        
        self.y_bar_diff = self.y_bar2 - self.y_bar1
        self.degf = min(self.n1 - 1, self.n2 - 1)
        
        self.test_stats = self.create_test_stats_df()
        
        self.perc_crit_value = self.conf_level + ((1 - self.conf_level)  / 2)
        
        self.t_star = t.ppf(self.perc_crit_value, self.degf)
        self.se = ((self.s1**2 / self.n1) + (self.s2**2 / self.n2))**(1/2)
        
        self.t_stat = self.y_bar_diff / self.se
        self.p_value = t.cdf(-abs(self.t_stat), self.degf) * 2
        
    def run_hypotheis_test(self, y_bar1=None, y_bar2=None, n1=None, n2=None, s1=None, s2=None):
        y_bar1 = y_bar1 or self.y_bar1
        y_bar2 = y_bar2 or self.y_bar2
        
        n1 = n1 or self.n1
        n2 = n2 or self.n2
        
        s1 = s1 or self.s1
        s2 = s2 or self.s2
        
        y_bar_diff = y_bar1 - y_bar2
        se = ((s1**2 / n1) + (s2**2 / n2))**(1/2)
        degf = min(n1 - 1, n2 - 1)
        
        t_stat = y_bar_diff / se
        p_value = t.cdf(-abs(t_stat), degf) * 2
        
        return p_value

    def conf_int(self):
        self.margin_o_err = self.t_star * self.se
        self.ci = self.y_bar_diff + np.array([-1, 1]) * self.margin_o_err
        
        return self.ci + self.y_bar1

    def create_test_stats_df(self):
        test_stats_data = [
            [self.y_bar1, self.y_bar2],
            [self.n1, self.n2], 
            [self.s1, self.s2]
        ]
        columns = ["Sample 1", "Sample 2"]
        # columns = ["Sample 1 (Pre-Bubble)", "Sample 2 (Bubble)"]
        index=["mu", "n", "std"]
        test_stats = (pd.DataFrame(data=test_stats_data, columns=columns, index=index)
            .T
            .assign(
                mu=lambda x: x.mu.map("{:,.2f}".format),
                n=lambda x: x.n.map("{:,.0f}".format),
                std=lambda x: x['std'].map("{:,.2f}".format),
            )
        )
        return test_stats
    
    def calc_per_increase(self):
        per_inc = (self.y_bar1[1] / self.y_bar2[0] - 1) * 1
        return f"Percent Change: {per_inc:.2%}"
    
    def test_visual(self, title):
        l, u = -3 * self.se, 3 * self.se
        x = np.arange(l, u, (u - l) / 1000)
        y = t.pdf(x, self.degf)
        
        x_below_p = x[x < self.t_stat]
        y_below_p = y[0:len(x_below_p)]
        x_above_p = x[len(x_below_p):]
        y_above_p = y[len(x_below_p):]
        
        ylim_top_adj = 1.05
        
        ax = sns.lineplot(x=x_below_p, y=y_below_p)
        ax.fill_between(x_below_p, y_below_p)
        
        ax.plot(x_above_p, y_above_p, color='r', alpha=0.3)
        ax.fill_between(x_above_p, y_above_p, color='r', alpha=0.3)
        ax.vlines(self.t_stat, 0, max(y), colors='r', alpha=0.9, linewidth=2)
        ax.set(title=title)
        #ax.axvspan(l, t_stat, alpha=0.3, color='red')
        ax.set_ylim(0, max(y) * ylim_top_adj)
        font = {
            'family': 'arial',
            'color':  'black',
            'weight': 'normal',
            'size': 14,
        }
        s = f"p-value: {self.p_value:0.2%}"
        ax.text(
            x=np.percentile(x, 35),
            y=np.percentile(y, 92.5),
            s=s,
            fontdict=font,
            ha="right"
        )
        return ax
    
    def simulation(self, nsims=1000):
        np.random.seed(11)
        sims = np.empty(nsims)
        y = self.data.loc[:, self.yname]
        x = self.data.loc[:, self.xname]
        n = len(y)
        for i in range(nsims):
            y_sim = y.sample(n, replace=True).values
            y_sim_bar1, y_sim_bar2 = (pd.DataFrame(np.column_stack((x.values, y_sim)), columns=['bubble', 'PTS'])
                .groupby('bubble')['PTS'].mean()
            )
            sims[i] = y_sim_bar2 - y_sim_bar1
        
        p_value = sum(sims >= self.y_bar_diff) / nsims
        
        return p_value, sims
