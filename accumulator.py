""" Accumulate a series of floating point values for statistics """

import math

class Accumulator(object):
    """ Compute and store running averages and other statistics """
    def __init__(self):
        """ Return a new object for accumulating statistics """
        self._data = {"sum" : 0., "sumSq" : 0., "nValues" : 0.}

    def accumulate(self, value):
        """ Add value to running average """
        self._data["sum"] += value
        self._data["sumSq"] += value*value
        self._data["nValues"] += 1.

    def average(self):
        """ Return the average value """
        return self._data["sum"]/self._data["nValues"]

    def std(self):
        """ Return the standard deviation """
        fluct = self._data["sumSq"]/self._data["nValues"] - self.average()**2
        return math.sqrt(fluct*self._data["nValues"]/(self._data["nValues"]-1))
