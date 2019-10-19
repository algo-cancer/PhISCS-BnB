import sys
from Utils.const import *
from Utils.interfaces import *
from . import LP
from . import MWM


class HybridBounding(BoundingAlgAbstract):
    def __init__(self, first_bounding=None, second_bounding=True, threshold_n_flips=None):
        """
        :param first_bounding:
        :param second_bounding:
        :param threshold_n_flips:
        """
        self.matrix = None
        self.threshold_n_flips = threshold_n_flips
        self.first_bounding = first_bounding
        self.second_bounding = second_bounding
        self.times = None

    def get_name(self):
        return (
            type(self).__name__
            + f"_{self.first_bounding.get_name()}_{self.second_bounding.getـname()}_{self.threshold_n_flips}"
        )

    def reset(self, matrix):
        self.times = {"modelPreperationTime": 0, "optimizationTime": 0}
        self.first_bounding.reset(matrix)
        self.second_bounding.reset(matrix)

    def get_bound(self, delta):
        flips = delta.count_nonzero()
        if flips < self.threshold_n_flips:
            bound = self.first_bounding.get_bound(delta)
        else:
            bound = self.second_bounding.getBound(delta)
        return bound


if __name__ == "__main__":

    n, m = 15, 15
    x = np.random.randint(2, size=(n, m))
    delta = sp.lil_matrix((n, m))
    delta[0, 0] = 1

    b1 = LP.SemiDynamicLPBounding(ratio=None, continuous=True)
    b2 = MWM.DynamicMWMBounding()
    hybridBounding = HybridBounding(first_bounding=b1, second_bounding=b2, threshold_n_flips=5)
    hybridBounding.reset(x)
    print(hybridBounding.get_bound(delta))

    print(b1.get_bound(delta))
    print(b2.get_bound(delta))

