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

    def get_values(self, name_metric: str, *axis):
        """
        Gets the values for given point on the axes. 
        The n-th axis value describes the n-th axis.
        >>>args=(None, 1, 1)
            is going to be the same as a[:, 1, 1, ...]
        """
        if name_metric not in self.metric_dict.keys():
            raise KeyError

        assert len(axis) <= len(self.metric_dict[name_metric].shape)
        if len(axis) == 0:
            return self.metric_dict[name_metric]

        axis = self._gen_slice_tuple(name_metric, axis)
        return self.metric_dict[name_metric][axis]

    def update_value(self, name_metric: str, new_values, *axis):
        if name_metric not in self.metric_dict.keys():
            raise KeyError

        axis = self._gen_slice_tuple(name_metric, axis)
        self.metric_dict[name_metric][axis] = new_values

    def get_axis_info(self, name_metric: str):
        if name_metric not in self.metric_info_dict.keys():
            raise KeyError
        return self.metric_info_dict[name_metric]

    # ===Eval Functions===
    def mean(self, name_metric: str, *axis):
        return np.nanmean(self.get_values(name_metric, *axis))
    
    def median(self, name_metric: str, *axis):
        return np.nanmedian(self.get_values(name_metric, *axis))

    def std(self, name_metric: str, *axis):
        return np.nanstd(self.get_values(name_metric, *axis))
    
    def var(self, name_metric: str, *axis):
        return np.nanvar(self.get_values(name_metric, *axis))
    
    def max(self, name_metric: str, *axis):
        """
        Note, argmax only works, if no NaNs in the given array
        """
        val = self.get_values(name_metric, *axis)
        return np.nanmax(val), np.argmax(val)

    def min(self, name_metric: str, *axis):
        """
        Note, argmax only works, if no NaNs in the given array
        """
        val = self.get_values(name_metric, *axis)
        return np.nanmin(val), np.argmin(val)
    
    #===Util Functions===
    def _gen_slice_tuple(self, name_metric: str, axis: tuple):
        axis = list(axis)
        for c, v in enumerate(axis):
            if type(axis[c]) is int:
                pass
            elif axis[c] == None:
                axis[c] = slice(self.metric_dict[name_metric].shape[c])
            elif len(axis[c]) == 2:
                axis[c] = slice(axis[c][0], axis[c][1])
            elif len(axis[c]) == 3:
                axis[c] = slice(axis[c][0], axis[c][1], axis[c][2])
        return tuple(axis)

if __name__ == "__main__":
    res = MetricResults()
    res.add_metric('randInd', (2, 2, 2))
    print(res.get_values('randInd').flatten())

    res.update_value('randInd', [1, 2], 0, 0)

    print(res.get_values('randInd').flatten())
    print(res.std('randInd', 0, 0))
    print(res.max('randInd', 0, 0))
    print(res.min('randInd', 0, 0))

    res.update_value('randInd', 1)
    print(res.get_values('randInd'))
