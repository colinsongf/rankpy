import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from matplotlib.ticker import MaxNLocator


def plot_lambdas_andrews_curves(lambdas, relevance_scores):
    columns = ['Tree %d' % i for i in range(1, 1 + lambdas.shape[0])]
    columns.append('Relevance')
    
    data = pd.DataFrame(np.r_[lambdas, relevance_scores.reshape(1, -1).astype(int)].T, columns=columns)
    pd.tools.plotting.andrews_curves(data, 'Relevance')
    
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.gca().legend(handles, map(lambda s: 'Relevance ' + s, labels))

   
def plot_lambdas_parallel_coordinates(lambdas, relevance_scores, individual=False, cumulative=False):
    unique_scores = sorted(np.unique(relevance_scores).astype(int), reverse=True)
    colors = ['r', 'g', 'b', 'y', 'c', 'm', 'y', 'k']
    
    if not individual:
        plt.figure()

        legend_handles = []
        legend_labels = []

        for c, r in enumerate(unique_scores):
            legend_handles.append(mlines.Line2D([], [], color=colors[c], linewidth=2))
            legend_labels.append('Relevance %d' % r)

    if cumulative:
        lambdas_cumsum = lambdas.cumsum(axis=0)
        ymin, ymax = lambdas_cumsum.min(), lambdas_cumsum.max()
    else:
        ymin, ymax = lambdas.min(), lambdas.max()
        
    for c, r in enumerate(unique_scores):
        if individual:
            plt.figure()

        if cumulative:
            plt.plot(lambdas[:, relevance_scores == r].cumsum(axis=0), '-', marker='o', markersize=4, c=colors[c])
        else:
            plt.plot(lambdas[:, relevance_scores == r], '-', marker='o', markersize=4, c=colors[c])

        if individual:
            plt.gca().get_xaxis().set_major_locator(MaxNLocator(integer=True))
            plt.gca().set_ylim([ymin, ymax])

            plt.title('Paralell Coordinates for%sLambdas (Relevance %d)' % (' Cumulative ' if cumulative else ' ', r))
            plt.xlabel('Trees')
            plt.ylabel('Cumulative Lambda Values' if cumulative else 'Lambda Values')
            plt.show()

    if not individual:
        plt.gca().get_yaxis().set_major_locator(MaxNLocator(integer=True))
        plt.gca().set_ylim([ymin, ymax])
        
        plt.title('Paralell Coordinates for%sLambdas (Relevance %d)' % (' Cumulative ' if cumulative else ' ', r))
        plt.xlabel('Trees')
        plt.ylabel('Cumulative Lambda Values' if cumulative else 'Lambda Values')    

        plt.legend(legend_handles, legend_labels, loc='best')
        plt.show()
