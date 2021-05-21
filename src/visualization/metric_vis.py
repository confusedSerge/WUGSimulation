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
    

def threed_line_ploter(x_labels: list, x_label: str, y_labels: list, y_label: str, z_label:str, title: str, save_flag: bool = False, save_path: str = None, **data) -> None:
    """
    Creates a 3d line plot based on data. 
    The line is fixed on the y achsis!

    Expected input for data:
        metric1 = {y1: [z1=f(x1), z2=f(x2), ...], y2: [...], ...}, metric2 = 

    Args:
        :param x_labels: x axis labels
        :param x_label: x axis label
        :param y_labels: y axis labels
        :param y_label: y axis label
        :param z_label: z axis label
        :param title: title of plot
        :param save_flag: if to save
        :param save_path: where to save
        :param data: the different metric points
    """
    colors = 'bgrcmyk'

    ax = plt.figure().add_subplot(projection='3d')

    x_points = np.arange(len(x_labels))
    _y_points = np.arange(len(y_labels))

    for y_point, _y_label in enumerate(y_labels):
        for i, (k, v) in enumerate(data.items()):
            ax.plot(x_points, [y_point]*len(x_points), v[_y_label], label='(Metric: {}, k: {})'.format(k, _y_label), color=colors[i % len(colors)])

    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    ax.set_zlabel(z_label)
    ax.set_title(title)
    
    ax.set_xlim(min(x_points), max(x_points))
    ax.set_ylim(min(_y_points), max(_y_points))

    ax.set_xticks(x_points)
    ax.set_xticklabels(x_labels)    
    ax.set_yticks(_y_points)
    ax.set_yticklabels(y_labels)
    ax.legend()

    if save_flag:
        assert save_path != None and type(save_path) == str
        plt.savefig(save_path)
    else:
        plt.show()

