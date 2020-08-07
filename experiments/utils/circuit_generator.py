import networkx as nx
import numpy as np
import torch
from itertools import permutations
from functools import partial
from typing import Tuple, Hashable, List
import itertools

def has_cycle(g: nx.DiGraph) -> bool:
    try:
        nx.find_cycle(g)
        return True
    except nx.NetworkXNoCycle:
        return False


def to_one_hot(arr: torch.tensor, mx: int):
    oh = torch.zeros((arr.shape[0], mx))
    for i, a in enumerate(arr):
        oh[i, a] = 1.0
    return oh


def sigmoid(x, a, kd, n, offset, inv):
    return a - (a) / (1. + np.exp((-x + kd) * n * inv)) + offset

# class Augment:
#     @staticmethod
#     def add_all_edges(g):
#         nodes = list(g.nodes())
#         for n1, n2 in itertools.product(nodes, repeat=2):
#             if n2 not in g[n1] or g[n1][n2] is None or len(g[n1][n2]) == 0:
#                 g.add_edge(n1, n2, features=torch.tensor([0.]), target=torch.tensor([0.]))
#         return g

class CircuitGenerator(object):

    functions = {
        'sigmoid': sigmoid
    }

    def __init__(self, n_parts: int):
        self.n_parts = n_parts
        self.func_name = 'sigmoid'
        assert self.func_name in self.functions
        params, labels = self.random_params(self.n_parts)
        self.params = params
        self.param_labels = params

    @property
    def func(self):
        return self.functions[self.func_name]

    def random_params(self, num):
        A = np.random.uniform(1, 20, size=(num))
        K = np.random.uniform(1, 20, size=(num))
        n = np.random.uniform(1, 2, size=(num))
        o = np.random.uniform(0, A.max() / 10., size=(num))

        A = np.expand_dims(A, 1)
        K = np.expand_dims(K, 1)
        n = np.expand_dims(n, 1)
        o = np.expand_dims(o, 1)

        # choose repressor (1) or activator (-1)
        i = np.random.choice([1], (num, 1))

        # [n_parts, n_params]
        params = np.hstack([A, K, n, o, i])
        labels = np.array(['A', 'K', 'n', 'o', 'i'])
        return params, labels

    def steady_state(self, g, acc='sum', node_to_part=lambda x: int(x)):
        acc_dict = {
            'sum': lambda x: np.sum(np.concatenate(x))
        }

        # in topological order, we evaluate the sigmoid function at each node
        for node in nx.topological_sort(g):

            idx = node_to_part(node)
            # gather all parents
            # accumulate outputs 'y' using the provided accumulation function
            parents = list(g.predecessors(node))
            if not parents:
                p = np.expand_dims(self.params[idx:idx + 1].T, 2)
                x = np.array([[0.]])
            else:
                a = []
                for p in parents:
                    _x = g.nodes[p]['y']
                    a.append(_x)
                if len(a) and len(a[0]):
                    x = acc_dict[acc](a)
                    x = x.reshape(_x.shape)
                else:
                    x = torch.tensor([], dtype=torch.float)
            y = self._partial_func(self.func_name, x, idx)
            g.nodes[node]['y'] = y

    def _partial_func(self, fname, x, node: Hashable):
        return self.functions[fname](x, *tuple(np.expand_dims(self.params[node:node + 1].T, 2)))

    def annotate_graph_with_features(self, g: nx.DiGraph, include_target: bool = True):
        # one-hot encode the graph nodes
        one_hot_encoded = to_one_hot(torch.arange(0, self.n_parts), self.n_parts)

        new_g = nx.DiGraph()
        for n, data in g.nodes(data=True):
            new_g.add_node(n, **data)
        for n1, n2, edata in g.edges(data=True):
            edata['features'] = np.array([0.])
            if include_target:
                edata['target'] = np.array([1.])
            new_g.add_edge(n1, n2, **edata)
        if include_target:
            self.steady_state(new_g, node_to_part=lambda x: x[-1])
        for n, ndata in new_g.nodes(data=True):
            # convert this to ONE HOT!
            ndata['features'] = one_hot_encoded[list(n)[-1]]
            if include_target:
                ndata['target'] = torch.tensor([ndata['y'].flatten()], dtype=torch.float)
        new_g.data = {
            'features': torch.tensor([0])
        }
        if include_target:
            new_g.data['target'] = torch.tensor([0])

        return new_g

    @staticmethod
    def graph_from_nodes(nodes: List[Tuple[Hashable, Hashable, Hashable]]) -> nx.DiGraph:
        g = nx.DiGraph()
        for n1, n2 in permutations(nodes, r=2):
            if n1[-1] in (n2[0], n2[1]):
                g.add_edge(tuple(n1), tuple(n2))
        return g

    def random_circuit(self, part_range: Tuple[int, int]):
        num_nodes = np.random.randint(*part_range)
        nodes = np.random.randint(0, self.n_parts, size=(num_nodes, 3))
        g = self.graph_from_nodes(nodes)
        return g

    def iter_random_circuit(self, limit: int, part_range: Tuple[int, int], cycles: bool = True, annotate: bool = False) -> nx.DiGraph:
        new_circuit = partial(self.random_circuit, part_range=part_range)
        for i in range(limit):
            c = new_circuit()
            if cycles is True:
                pass
            else:
                while has_cycle(c):
                    c = new_circuit()
            if annotate:
                c = self.annotate_graph_with_features(c)
            yield c