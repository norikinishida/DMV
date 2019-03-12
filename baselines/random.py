import copy

import numpy as np

import treetk

class Random(object):

    def __init__(self):
        pass

    def parse(self, tokens):
        """
        :type tokens: list of str
        :rtype: list of (int, int)
        """
        assert tokens[0] == "<root>" # NOTE

        if len(tokens) == 2:
            return [(0, 1)]

        arcs = []

        # Construct a ctree in a bottom-up manner
        x = copy.deepcopy(tokens[1:])
        while len(x) > 1:
            i = np.random.randint(0, len(x)-1)
            merged = "( %s %s )" % (x[i], x[i+1])
            x[i] = merged
            x.pop(i+1)
        sexp = x[0]
        sexp = sexp.split()
        ctree = treetk.sexp2tree(sexp, with_nonterminal_labels=False, with_terminal_labels=False)

        # Random head assignment
        ctree.calc_heads(func_head_child_rule=lambda node: np.random.randint(0, 2)) # Random heading

        # Convert ctree to dtree
        dtree = treetk.ctree2dtree(ctree, func_label_rule=None)
        arcs = dtree.tolist(labeled=False)
        return arcs

