from Utils.const import *
from Utils.interfaces import *
from operator import add
from functools import reduce
from Utils.instances import *
from operator import itemgetter
plg = print_line_iter()
def high_degree(G):
    degs = list(G.degree())
    degs.sort(key=itemgetter(1), reverse=True)
    n_edges = G.number_of_edges()
    # todo: binary search
    deg_sum = 0
    for ind, (node, deg) in enumerate(degs):
        deg_sum += deg
        if deg_sum > n_edges:
            break
    return ind
    # print(degs)

def Min_vertex_cover(G):
    rc2 = RC2(WCNF())
    for node in G.nodes():
        rc2.add_clause([-node], weight = 1)
    for edge in G.edges():
        rc2.add_clause(edge)
    variables = rc2.compute()
    ans = 0
    for var in variables:
        if var > 0:
            ans += 1
    return ans


def make_graph(matrix, coloring=None, inside=None):
    # print(type(coloring))
    # exit(0)
    G = nx.Graph()
    F = - np.ones(matrix.shape, dtype=np.int64)
    num_var_F = 0
    map_f2ij = {}
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i, j] == 0:
                num_var_F += 1
                map_f2ij[num_var_F] = (i, j)
                F[i, j] = num_var_F
    if isinstance(coloring, np.ndarray):
        color = coloring
        bipartite = True
    elif coloring == True:  # color the columns
        color = np.random.randint(2, size=matrix.shape[1])
        bipartite = True
    elif coloring is None:
        bipartite = False
    for p in range(matrix.shape[1]):
        for q in range(p + 1, matrix.shape[1]):
            if bipartite:
                if inside == (color[p] != color[q]):
                    continue
            if np.any(np.logical_and(matrix[:, p] == 1, matrix[:, q] == 1)):
                r01 = np.nonzero(np.logical_and(matrix[:, p] == 0, matrix[:, q] == 1))[0]
                r10 = np.nonzero(np.logical_and(matrix[:, p] == 1, matrix[:, q] == 0))[0]
                # print(len(r01), len(r10))
                for a, b in itertools.product(r01, r10):
                    G.add_edge(F[a, p], F[b, q])
                    # print(F[a, p], F[b, q])
    return G, F, map_f2ij


class GraphicBounding(BoundingAlgAbstract):
    def __init__(self, priority_version = 7, version = 0):
        self.matrix = None
        self.priority_version = priority_version
        self._times = None
        self.version = version

    def get_name(self):
        return f"{type(self).__name__}_{self.priority_version}_{self.version}"

    def reset(self, matrix):
        self.matrix = matrix # make the model here and do small alterations later
        self._times = {"model_preparation_time": 0, "optimization_time": 0}

    # def get_init_node(self):
    #     raise NotImplementedError("To be implemented")

    def get_bound(self, delta):
        from networkx.algorithms.approximation import min_weighted_vertex_cover
        self._extraInfo = None
        current_matrix = np.array(self.matrix + delta) # todo: make this dynamic

        inside = False
        if self.version == 0:
            coloring = None
        elif self.version == 1:
            coloring = True
        elif self.version == 2:
            coloring = True
        elif self.version == 3:
            coloring = None
        elif self.version == 4:
            coloring = None
        elif self.version == 5:
            coloring = None
        elif self.version == 6:
            coloring = True
        elif self.version == 7:
            coloring = get_clustering(current_matrix)
            inside = True
        elif self.version == 8:
            inside = True
            coloring = True
        else:
            coloring = None

        model_time = time.time()
        G, F, map_f2ij = make_graph(current_matrix, coloring = coloring, inside=inside)
        model_time = time.time() - model_time
        self._times["model_preparation_time"] += model_time


        opt_time = time.time()
        if self.version == 0:
            mwvc = min_weighted_vertex_cover(G)
            bnd = len(mwvc)
        elif self.version == 1:
            mwvc = min_weighted_vertex_cover(G)
            bnd = len(mwvc)
        elif self.version == 2:
            matching = nx.bipartite.maximum_matching(G)
            vertex_cover = nx.bipartite.to_vertex_cover(G, matching)
            bnd = len(vertex_cover)
        elif self.version == 3:
            T = nx.minimum_spanning_tree(G)
            mwvc = min_weighted_vertex_cover(T)
            bnd = len(mwvc)
        elif self.version == 4:
            from dimod.reference.samplers import ExactSolver
            import dwave_networkx as dnx
            sampler = ExactSolver()
            mwvc = dnx.min_vertex_cover(G, sampler)
            bnd = len(mwvc)
        elif self.version == 5:
            bnd =  Min_vertex_cover(G)
        elif self.version == 6:
            bnd =  Min_vertex_cover(G)
        elif self.version == 7:
            # matching = nx.bipartite.maximum_matching(G)
            # print(matching)
            # vertex_cover = nx.bipartite.to_vertex_cover(G, matching)
            # bnd = len(vertex_cover)
            bnd =  Min_vertex_cover(G)
        elif self.version == 8:
            bnd =  Min_vertex_cover(G)
        elif self.version == "cycles":
            # sim_cyc = nx.simple_cycles(G)
            # print(len(sim_cyc))
            bas_cyc = nx.cycle_basis(G)
            print(len(bas_cyc))
            ls = []
            for c in bas_cyc:
                ls.append(len(c))
            print(np.bincount(ls))
            bnd = -1

        opt_time = time.time() - opt_time
        self._times["optimization_time"] += opt_time

        result = bnd

        return result


    def get_priority(self, till_here, this_step, after_here, icf=False):
        if icf:
            return self.matrix.shape[0] * self.matrix.shape[1] + 10
        else:
            sgn = np.sign(self.priority_version)
            pv_abs = self.priority_version * sgn
            if pv_abs == 1:
                return sgn * (till_here + this_step + after_here)
            elif pv_abs == 2:
                return sgn * (this_step + after_here)
            elif pv_abs == 3:
                return sgn * (after_here)
            elif pv_abs == 4:
                return sgn * (till_here + after_here)
            elif pv_abs == 5:
                return sgn * (till_here)
            elif pv_abs == 6:
                return sgn * (till_here + this_step)
            elif pv_abs == 7:
                return 0


if __name__ == "__main__":
    # n, m = 20, 20
    # x = np.random.randint(2, size=(n, m))
    # x = I_small
    # x = read_matrix_from_file("test2.SC.noisy")
    # xx = read_matrix_from_file("../noisy_simp/simNo_2-s_4-m_50-n_50-k_50.SC.noisy")
    xx = np.loadtxt("fivehmatrix.txt")
    # x = xx[:100, :100]
    # x = x[:, :30]
    # x = np.hstack((x, x, x, x, x, x, x))
    # x = np.vstack((x, x))
    from Boundings.two_sat import two_sat
    from Boundings.LP import SemiDynamicLPBounding
    algos =[
        # GraphicBounding(version=0),
        # GraphicBounding(version=1),
        # GraphicBounding(version=2),
        # GraphicBounding(version=2),
        # GraphicBounding(version=2),
        # GraphicBounding(version=3),
        # GraphicBounding(version=5),
        # GraphicBounding(version=6),
        # GraphicBounding(version=7),
        # GraphicBounding(version=8),
        two_sat(),
        GraphicBounding(version="cycles"),
        # SemiDynamicLPBounding()
    ]
    from networkx.algorithms.approximation.vertex_cover import min_weighted_vertex_cover
    from networkx.algorithms.community import k_clique_communities
    from networkx.algorithms.approximation.kcomponents import k_components

    for i in range(50, 501, 50):
        x = xx[:100, :i]
        G, F, map_f2ij = make_graph(x)
        # mm = nx.maximal_matching(G)
        # mwm = nx.max_weight_matching(G)
        next(plg)
        mvc = min_weighted_vertex_cover(G)
        next(plg)
        # hd = high_degree(G)
        # print(len(mm), len(mwm), len(mvc), hd)
        # print("-------------")
        # kc = nx.k_components(G)
        # print(kc)
        # print("-------------")
        # kc2 = k_components(G, min_density=0.5)
        # print(kc2)
        # continue
        # nx.draw(G)
        # plt.show()
        print("number_of_edges:", G.number_of_edges())
        print("number_of_nodes:", G.number_of_nodes())
        # bas_cyc = nx.cycle_basis(G)
        # print(len(bas_cyc))
        # ls = []
        # for c in bas_cyc:
        #     ls.append(len(c))
        # bc = np.bincount(ls)
        # print(i, bc)
        # print("sum:", np.sum(bc))
        # ws = np.inner(np.arange(bc.shape[0]), bc)
        # print("ws:", ws)
        # print("example:", bas_cyc[0])
        # print()
        # nodes = set()
        # edges = set()
        # print("-------------")
        # for c in bas_cyc:
        #     if len(c) > 3:
        #         continue
        #     # print(c)
        #     # nodes.update(c)
        #     for i in range(len(c)):
        #         aa, bb = c[i], c[(i + 1) % len(c)]
        #         a = min(aa, bb)
        #         b = max(aa, bb)
        #         # if (a, b) not in edges:
        #         #     print(a,b)
        #         edges.add((a, b))
        # sG = nx.Graph(list(edges))
        # print(sG.number_of_edges())
        # print(sG.number_of_nodes())
        print("-------------")
        # print(len(nodes))
        # print(len(edges))
        next(plg)
        print(Min_vertex_cover(G), len(mvc))
        next(plg)
        # print(Min_vertex_cover(sG))
        # print(len(mm), len(mwm), len(mvc), len(kcc))
        print("-------------=======================------------------------")
        continue
        print(x.shape)
        delta = sp.lil_matrix(x.shape)
        for algo in algos:
            a = time.time()
            algo.reset(x)
            bnd = algo.get_bound(delta)
            b = time.time()
            print(algo.get_name().ljust(30), bnd, b - a, algo._times["model_preparation_time"], algo._times["optimization_time"], sep="\t")
        # print(bnd, algo._times)
    # print(bnd)
    # print(algo.get_priority(0,0,bnd, False))
