import numpy as np


class MetricResults():

    def __init__(self, name: str = None, info: str = None):
        self.name = name
        self.info = info
        self.metric_dict = {}
        self.metric_info_dict = {}

    # ===Util Functions===
    def add_metric(self, name_metric: str, size: tuple, axes_info: tuple = None):
        if name_metric in self.metric_dict.keys():
            return

        new_metric_matrix = np.full(size, np.NaN)
        self.metric_dict[name_metric] = new_metric_matrix
        self.metric_info_dict[name_metric] = axes_info

    def remove_metric(self, name_metric: str):
        return self.metric_dict.pop(name_metric, None)

    def get_values(self, name_metric: str, *axes):
        """
        Gets the values for given point on the axes. 
        The n-th axis value describes the n-th axis.
        >>>args=(None, 1, 1)
            is going to be the same as a[:, 1, 1, ...]
        args can be of type slice(), int (=position), None (all values along axis), or tuple describing (start, stop [, step]),  
        """
        if name_metric not in self.metric_dict.keys():
            raise KeyError

        assert len(axes) <= len(self.metric_dict[name_metric].shape)
        if len(axes) == 0:
            return self.metric_dict[name_metric]

        axes = self._gen_slice_tuple(name_metric, axes)
        return self.metric_dict[name_metric][axes]

    def update_value(self, name_metric: str, new_values, *axes):
        if name_metric not in self.metric_dict.keys():
            raise KeyError

        axes = self._gen_slice_tuple(name_metric, axes)
        self.metric_dict[name_metric][axes] = new_values

    def extend_dim(self, name_metric: str, axis: int = 0):
        if name_metric not in self.metric_dict.keys():
            raise KeyError

        self.metric_dict[name_metric] = np.expand_dims(self.metric_dict[name_metric], axis=axis)

    def append_values(self, name_metric: str, values, axis: int = 0):
        if name_metric not in self.metric_dict.keys():
            raise KeyError

        self.metric_dict[name_metric] = np.append(self.metric_dict[name_metric], values, axis=axis)


    def get_axes_info(self, name_metric: str):
        if name_metric not in self.metric_info_dict.keys():
            raise KeyError
        return self.metric_info_dict[name_metric]

    def set_axes_info(self, name_metric: str, axes_info: tuple = None):
        if name_metric not in self.metric_info_dict.keys():
            raise KeyError
        self.metric_info_dict[name_metric] = axes_info

    # ===Eval Functions===
    def mean(self, name_metric: str, axes_value: tuple, axis_mean: int):
        return np.nanmean(self.get_values(name_metric, axes_value), axis=axis_mean)
    
    def median(self, name_metric: str, axes_value: tuple, axis_mean: tuple):
        return np.nanmedian(self.get_values(name_metric, axes_value), axis=axis_mean)

    def std(self, name_metric: str, axes_value: tuple, axis_mean: tuple):
        return np.nanstd(self.get_values(name_metric, axes_value), axis=axis_mean)
    
    def var(self, name_metric: str, axes_value: tuple, axis_mean: tuple):
        return np.nanvar(self.get_values(name_metric, axes_value), axis=axis_mean)
    
    def max(self, name_metric: str, axes_value: tuple, *axes):
        """
        Note, argmax only works, if no NaNs in the given array
        """
        val = self.get_values(name_metric, *axes)
        return np.nanmax(val), np.argmax(val)

    def min(self, name_metric: str, *axes):
        """
        Note, argmax only works, if no NaNs in the given array
        """
        val = self.get_values(name_metric, *axes)
        return np.nanmin(val), np.argmin(val)
    
    #===Util Functions===
    def _gen_slice_tuple(self, name_metric: str, axes: tuple):
        axes = list(axes)
        for c, v in enumerate(axes):
            if type(axes[c]) is int or isinstance(axes[c], slice):
                pass
            elif axes[c] == None:
                axes[c] = slice(self.metric_dict[name_metric].shape[c])
            elif len(axes[c]) == 2:
                axes[c] = slice(axes[c][0], axes[c][1])
            elif len(axes[c]) == 3:
                axes[c] = slice(axes[c][0], axes[c][1], axes[c][2])
        return tuple(axes)

if __name__ == "__main__":
    res = MetricResults()
    res.add_metric('randInd', (3, 3))
    print(res.get_values('randInd').flatten())

    res.update_value('randInd', [1, 2, 2], 0)

    print(res.get_values('randInd').flatten())
    print(res.std('randInd', slice(0, 1), 0))
    print(res.max('randInd', 0, 0))
    print(res.min('randInd', 0, 0))

    res.update_value('randInd', 1)
    print(res.get_values('randInd'))
