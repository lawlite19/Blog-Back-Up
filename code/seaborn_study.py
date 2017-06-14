#-*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
#%matplotlib inline  # 为了在jupyter notebook里作图，需要用到这个命令

def displot():
    tips = sns.load_dataset('tips')
    print(tips.shape)    
    #sns.distplot(tips['total_bill'], bins=None, hist=True, kde=False, rug=True, fit=None, 
                #hist_kws=None, kde_kws=None, rug_kws=None, 
                #fit_kws=None, color=None, vertical=False, 
                #norm_hist=False, axlabel='total_bill', label='dis plot', ax=None)
    from scipy import stats
    sns.distplot(tips.total_bill, fit=stats.gamma, kde=False)
    sns.plt.show()
def kdeplot():
    tips = sns.load_dataset('tips')
    print(tips.shape)    
    ax = sns.kdeplot(tips['total_bill'], data2=None, shade=False, vertical=False, 
                    kernel="gau", bw="scott", 
                    gridsize=100, cut=3, clip=None, 
                    legend=True, cumulative=False, 
                    shade_lowest=True, ax=None)
    sns.plt.show()
def pairplot():
    iris = sns.load_dataset('iris')
    g = sns.pairplot(iris, hue='species', hue_order=None, palette=None, 
                     vars=list(iris.columns[0:-1]), 
                     x_vars=None, y_vars=None, 
                     kind="reg", diag_kind="hist", 
                     markers=None, size=1.5, aspect=1, 
                     dropna=True, plot_kws=None, 
                     diag_kws=None, grid_kws=None)
    sns.plt.show()
def stripplot():
    tips = sns.load_dataset('tips')
    ax = sns.stripplot(x='sex', y='total_bill', hue='day', data=tips, order=None, 
                      hue_order=None, jitter=True, 
                      split=False, orient=None, 
                      color=None, palette=None, size=5, 
                      edgecolor="gray", linewidth=0, 
                      ax=None)
    sns.plt.show()
def swarmplot():
    tips = sns.load_dataset('tips')
    ax = sns.swarmplot(x='sex', y='total_bill', hue='day', data=tips)
    sns.plt.show()
def boxplot():
    tips = sns.load_dataset('tips')
    ax = sns.boxplot(x='day', y='total_bill', hue=None, data=tips, order=None, 
                    hue_order=None, orient=None, 
                    color=None, palette=None, 
                    saturation=.75, width=.8, 
                    fliersize=5, linewidth=None, 
                    whis=1.5, notch=False, ax=None)
    sns.stripplot(x='day', y='total_bill', hue=None, data=tips, order=None, 
                 hue_order=None, jitter=True, split=False, 
                 orient=None, color=None, palette=None, 
                 size=5, edgecolor="gray", linewidth=0, 
                 ax=None)
    sns.plt.show()
def jointplot():
    tips = sns.load_dataset('tips')
    from scipy import stats
    g = (sns.jointplot(x='total_bill', y='tip',data=tips).plot_joint(sns.kdeplot))
    sns.plt.show()
def violinplot():
    tips = sns.load_dataset('tips')
    ax = sns.violinplot(x='day', y='total_bill', 
                        hue='smoker', data=tips, order=None, 
                        hue_order=None, bw="scott", 
                        cut=2, scale="area", 
                        scale_hue=True, gridsize=100, 
                        width=.8, inner="quartile", 
                        split=False, orient=None, 
                        linewidth=None, color=None, 
                        palette='muted', saturation=.75, 
                        ax=None) 
    #sns.violinplot(x=tips['total_bill'])
    sns.plt.show()
def pointplot():
    tips = sns.load_dataset('tips')
    sns.pointplot(x='time', y='total_bill', hue='smoker', data=tips, order=None, 
                 hue_order=None, estimator=np.mean, ci=95, 
                 n_boot=1000, units=None, markers="o", 
                 linestyles="-", dodge=False, join=True, 
                 scale=1, orient=None, color=None, 
                 palette=None, ax=None, errwidth=None, 
                 capsize=None)
    sns.plt.show()
def barplot():
    tips = sns.load_dataset('tips')
    sns.barplot(x='day', y='total_bill', hue='sex', data=tips, order=None, 
               hue_order=None, estimator=np.mean, ci=95, 
               n_boot=1000, units=None, orient=None, 
               color=None, palette=None, saturation=.75, 
               errcolor=".26", errwidth=None, capsize=None, 
               ax=None)
    sns.plt.show()
def countplot():
    tips = sns.load_dataset('tips')
    sns.countplot(x='day', hue='sex', data=tips)
    sns.plt.show()    
def factorplot():
    titanic = sns.load_dataset('titanic')
    sns.factorplot(x='age', y='embark_town', 
                   hue='sex', data=titanic,
                   row='class', col='sex', 
                   col_wrap=None, estimator=np.mean, ci=95, 
                   n_boot=1000, units=None, order=None, 
                   hue_order=None, row_order=None, 
                   col_order=None, kind="box", size=4, 
                   aspect=1, orient=None, color=None, 
                   palette=None, legend=True, 
                   legend_out=True, sharex=True, 
                   sharey=True, margin_titles=False, 
                   facet_kws=None)
    sns.plt.show()
def heatmap():
    flight = sns.load_dataset('flights')
    flights = flight.pivot('month','year','passengers')
    sns.heatmap(flights, annot=True, fmt='d')
    sns.plt.show()
def tsplot():
    gammas = sns.load_dataset('gammas')
    sns.tsplot(data=gammas, time='timepoint', unit='subject', 
               condition='ROI', value='BOLD signal', 
               err_style="ci_band", ci=68, interpolate=True, 
               color=None, estimator=np.mean, n_boot=5000, 
               err_palette=None, err_kws=None, legend=True, 
               ax=None)
    sns.plt.show()
tsplot()