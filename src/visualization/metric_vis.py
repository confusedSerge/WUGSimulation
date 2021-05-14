import numpy as np
import matplotlib.pyplot as plt

"""
This module contains various visualizations for metrics
"""

def bar_metric(labels, title: str, y_label: str, save_flag: bool = False, save_path: str = None, **metrics) -> None:
    """
    Creates a bar plot containing len(labels) seperated bargraph-groups, 
        where the nth bargraph-group contains the nth metric 
    Args:
        :param label: label of each bargraph-group
        :param title: title of plot
        :param y_label: y axis label 
        :param save_flag: if to save
        :param save_path: where to save
        :param metrics: the different metric points, where the length needs to match up with label.
    """
    for k, v in metrics.items():
        assert len(labels) == len(v)

    x_point = np.arange(len(labels))
    width = 0.8 / len(metrics)

    fig, ax = plt.subplots()

    for i, (k, v) in enumerate(metrics.items()):
        ax.bar(x_point + (width * (1 - len(v)) / 2) + i * width, v, width, label=k)

    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.set_ylim((0.0, 1.0))

    ax.set_xticks(x_point)
    ax.set_xticklabels(labels)
    ax.legend()

    if save_flag:
        assert save_path != None and type(save_path) == str
        plt.savefig(save_path)
    else:
        plt.show()

def line_ploter(steps, title: str, x_label: str, y_label: str, save_flag: bool = False, save_path: str = None, **data) -> None:
    """
    Creates a line plot of the data 
    Args:
        :param steps: step labels
        :param title: title of plot
        :param x_label: x axis label 
        :param y_label: y axis label 
        :param save_flag: if to save
        :param save_path: where to save
        :param metrics: the different metric points, where the length needs to match up with label.
    """
    for k, v in data.items():
        assert len(steps) == len(v)

    
    x_point = np.arange(len(steps))

    fig, ax = plt.subplots()

    for k, v in data.items():
        ax.plot(x_point, v, label=k)

    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    ax.set_title(title)
    
    ax.set_xlim(min(x_point), max(x_point))
    ax.set_ylim((0.0, 1.0))

    ax.set_xticks(x_point)
    ax.set_xticklabels(steps)
    ax.legend()

    if save_flag:
        assert save_path != None and type(save_path) == str
        plt.savefig(save_path)
    else:
        plt.show()
    