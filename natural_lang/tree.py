import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


# tree object from stanfordnlp/treelstm
class Tree(object):
    def __init__(self):
        self.parent = None
        self.num_children = 0
        self.children = list()

    def add_child(self, child):
        child.parent = self
        self.num_children += 1
        self.children.append(child)

    def size(self):
        if hasattr(self, '_size'):
            return self._size
        count = 1
        for i in range(self.num_children):
            count += self.children[i].size()
        self._size = count
        return self._size

    def data(self):
        """
        :return: list of tree nodes as a plain list
        """
        assert self._data is not None, "Only root node contains the tree list!"
        return self._data

    def depth(self):
        if getattr(self, '_depth'):
            return self._depth
        count = 0
        if self.num_children > 0:
            for i in range(self.num_children):
                child_depth = self.children[i].depth()
                if child_depth > count:
                    count = child_depth
            count += 1
        self._depth = count
        return self._depth

    def get_relations(self, rels=None):
        if rels is None:
            rels = []

        for ch in self.children:
            rels.append((self.idx, ch.idx))
            ch.get_relations(rels)

        return rels

    def plot(self):
        G = nx.DiGraph()
        G.add_edges_from(self.get_relations())
        p = nx.drawing.nx_pydot.to_pydot(G)
        p.write_png('example.png')
