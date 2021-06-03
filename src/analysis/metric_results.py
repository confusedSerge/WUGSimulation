import numpy as np

class MetricResults():

    def __init__(self, name: str = None, info: str = None):
        self.name = name
        self.info = info
        self.metric_dict = {}
        self.metric_info_dict = {}

    # ===Util Functions===
    # TODO: Add function?
    def add_metric(self, name_metric: str, size: tuple, axis_info: tuple = None):
        if name_metric in self.metric_dict.keys():
            raise KeyError('Key exist already')

        new_metric_matrix = np.full(size, np.NaN)
        self.metric_dict[name_metric] = new_metric_matrix
        self.metric_info_dict[name_metric] = axis_info

    def remove_metric(self, name_metric: str):
        return self.metric_dict.pop(name_metric, None)

    def get_value(self, name_metric: str, position: tuple):
        if name_metric not in self.metric_dict.keys():
            raise KeyError
        return self.metric_dict.get[name_metric][position] if position != None else self.metric_dict[name_metric]

    def update_value(self, name_metric: str, position: tuple, new_values):
        if name_metric not in self.metric_dict.keys():
            raise KeyError
        
        if position == None:
            old_values = self.metric_dict[name_metric]
            self.metric_dict[name_metric] = new_values
            return old_values
        
        old_values = self.metric_dict[name_metric][position]

        if len(old_values) != len(new_values):
            raise ValueError

        self.metric_dict[name_metric][position] = new_values
        return old_values

    def get_axis_info(self, name_metric: str):
        if name_metric not in self.metric_info_dict.keys():
            raise KeyError
        return self.metric_info_dict[name_metric]

    #===Eval Functions===