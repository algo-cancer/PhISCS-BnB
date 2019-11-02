import sys
from Utils.const import *
from Utils.interfaces import *
from . import LP
from . import MWM


class RandomHybridBounding(BoundingAlgAbstract):
    def __init__(self, boundings = [], weight = None, n_algs_per_node = 1):
        """
        randomized choice of hybrid
        """
        assert len(boundings) > 0, "There is no bounding provided!"
        self.matrix = None
        self.boundings = boundings
        self.n_algs_per_node = n_algs_per_node
        self.weight = weight
        if self.weight is None:
            self.weight = np.ones(len(self.boundings))
        assert len(self.weight) == len(self.boundings), "Number of weights should be equal to boundings"
        assert self.n_algs_per_node <= len(self.boundings), f"should be " \
            f"self.n_algs_per_node <= len(self.boundings) but {self.n_algs_per_node} <= {len(self.boundings)}"
        assert 0 <= self.n_algs_per_node , f"should be 0 <= self.n_algs_per_node but {0} <= {self.n_algs_per_node}"
        self.weight /= len(self.weight)
        self._times = None
        self._extraInfo = None

    def get_name(self):
        return (
            type(self).__name__
            + f"_{len(self.boundings)}_{self.weight[0]}_{self.n_algs_per_node}"
        )

    def reset(self, matrix):
        for bounding in self.boundings:
            bounding.reset(matrix)

    def get_bound(self, delta):
        # flips = delta.count_nonzero()
        if len(self.boundings) == self.n_algs_per_node:
            chosen_bnd_indices = np.arange(self.n_algs_per_node)
        else:
            chosen_bnd_indices = np.random.choice(len(self.boundings), self.n_algs_per_node, p=self.weight)
        bound = np.max([self.boundings[ind].get_bound(delta) for ind in chosen_bnd_indices])
        return bound

    def get_times(self):
        times_list = [bounding.get_times() for bounding in self.boundings]
        times_list = [x for x in times_list if x is not None]
        df = pd.DataFrame(times_list)
        df = df.sum()
        self._times = df.to_dict()
        return super().get_times()


if __name__ == "__main__":
    n, m = 15, 15
    x = np.random.randint(2, size=(n, m))
    delta = sp.lil_matrix((n, m))

    #######  Change one random coordinate ######
    nnz_ind = np.nonzero(1 - (x + delta))
    a, b = nnz_ind[0][0], nnz_ind[1][0]
    delta[a, b] = 1
    ############################################

    boundings = [
        LP.SemiDynamicLPBounding(ratio=None, continuous=True),
        LP.SemiDynamicLPBounding(ratio=None, continuous=True),
        MWM.DynamicMWMBounding(),
    ]
    hybrid_bounding = RandomHybridBounding(boundings, n_algs_per_node=1)
    hybrid_bounding.reset(x)
    print(hybrid_bounding.get_bound(delta))
    print()
    for bnd in boundings:
        print(bnd.get_bound(delta))
        print(bnd.get_times())
    print(hybrid_bounding.get_times())

