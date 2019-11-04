import sys
from Utils.const import *
from Utils.interfaces import *
from . import LP_APX



class AdaptiveLPSubsampleBounding(BoundingAlgAbstract):
    def __init__(self):
        """
        Adaptive LP Subsample
        """
        self.matrix = None
        self.strength = 0.9
        self.init_number = 1
        self.boundings = [LP_APX.SubsampleLPBounding(strength = self.strength) for _ in range(self.init_number)]
        self._times = None
        self._extraInfo = None
        self.weights = None # this will be as long as constr. Initialized at reset()
        self.last_weights = None # this will be as long as constr. Initialized at reset()
        self.from_last_add_bounding = None
        self.n_init_runs = 4
        self.eps = 0.1

    def get_name(self):
        return (
            type(self).__name__
            + f"_{self.init_number}"
        )

    def reset(self, matrix):
        self.matrix = matrix
        self.from_last_add_bounding = 0
        for bounding in self.boundings:
            bounding.reset(self.matrix)
            if self.weights is None:
                self.weights = np.ones(len(bounding.constraints))

    def update_weights(self, mask):
        assert len(mask) == len(self.weights), "shoule be len(mask) == len(self.weights) but " \
                                               f"{len(mask)} == {len(self.weights)}"
        for ind in range(len(mask)):
            if not mask[ind]:
                self.weights[ind] *=2

        self.weights = self.weights / self.weights.sum()

    def add_bounding(self):
        self.last_weights = self.weights.copy()
        new_bounding = LP_APX.SubsampleLPBounding(strength=self.strength, weights = self.weights)
        new_bounding.reset(self.matrix)
        self.boundings.append(new_bounding)

    def get_bound(self, delta):
        # flips = delta.count_nonzero()
        # TODO: try another version later
        # self.n_algs_per_node = len(self.boundings)
        # if len(self.boundings) == self.n_algs_per_node:
        #     chosen_bnd_indices = np.arange(self.n_algs_per_node)
        # else:
        #     chosen_bnd_indices = np.random.choice(len(self.boundings), self.n_algs_per_node, p=self.weight)

        #choose lastone:
        chosen_bnd_indices = [len(self.boundings) - 1]
        for ind in chosen_bnd_indices:
            bound = self.boundings[ind].get_bound(delta)
            mask = self.boundings[ind].get_unsat_mask()
            break # todo: only one for now
        self.update_weights(mask)
        if self.last_weights is None:
            if self.from_last_add_bounding > self.n_init_runs:
                self.add_bounding()
        elif np.linalg.norm(self.last_weights-self.weights) > self.eps:
            self.add_bounding()
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

    main_bounding = AdaptiveLPSubsampleBounding()
    main_bounding.reset(x)
    print(main_bounding.get_bound(delta))
    print()
    print(main_bounding.get_times())
    print()
    print(main_bounding.weights)
