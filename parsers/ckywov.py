from collections import defaultdict

import numpy as np

import treetk

from globalvars import RIGHT, LEFT, STOP, CONT

class CKYParserWithoutValence(object):

    def __init__(self):
        pass

    def parse(self, postags, model):
        """
        :type postags: list of str
        :type: model: DMV
        :rtype: list of (int, int)

        P(STOP | h, ->): R_s[h] -> L_s[h]
        P(CONT | h, ->): R_s[h] -> R_a[h]

        P(STOP | h, <-): L_s[h] -> h
        P(CONT | h, <-): L_s[h] -> L_a[h]

        P(d | h, ->): R_a[h] -> R_s[h] R_s[d]
        P(d | h, <-): L_a[h] -> R_s[d] R_s[h]

        P(h | S, ->) = S -> R_s[h]
        """
        assert postags[0] != "<root>" # NOTE

        n_postags = len(postags)

        # Initialize charts
        chart = defaultdict(float) # {(int, int, str, int): float}
        back_ptr = {} # {(int, int, str, int): [int, str, int, str, int, int]

        # Base case
        for i in range(0, n_postags):
            # "L_s[h] -> h"
            chart[i, i, "L_s", i] \
                    = model.get_stop_prob(head=postags[i],
                                          direction=LEFT,
                                          valency=0,
                                          stop=STOP)
            back_ptr[i, i, "L_s", i] = [None, None, None, None, None, None]

            # Handle unaries
            added = True
            while added:
                added = False
                # "R_s[h] -> L_s[h]"
                score = model.get_stop_prob(head=postags[i],
                                            direction=RIGHT,
                                            valency=0,
                                            stop=STOP) \
                         * chart[i, i, "L_s", i]
                if chart[i, i, "R_s", i] < score:
                    chart[i, i, "R_s", i] = score
                    back_ptr[i, i, "R_s", i] = [i, "L_s", i]
                    added = True

        # General case
        for d in range(1, n_postags):
            for i1 in range(0, n_postags - d):
                i3 = i1 + d
                for i2 in range(i1, i3):
                    for l_i in range(i1, i2+1):
                        for r_i in range(i2+1, i3+1):
                            # Left attachment "L_a[r] -> R_s[l] L_s[r]
                            score = model.get_attach_prob(head=postags[r_i],
                                                         direction=LEFT,
                                                         dep=postags[l_i]) \
                                   * chart[i1, i2, "R_s", l_i] \
                                   * chart[i2+1, i3, "L_s", r_i]
                            if chart[i1, i3, "L_a", r_i] < score:
                                chart[i1, i3, "L_a", r_i] = score
                                back_ptr[i1, i3, "L_a", r_i] = [r_i, "R_s", l_i, "L_s", r_i, i2]

                            # Right attachment "R_a[l] -> R_s[l] R_s[r]"
                            score = model.get_attach_prob(head=postags[l_i],
                                                         direction=RIGHT,
                                                         dep=postags[r_i]) \
                                   * chart[i1, i2, "R_s", l_i] \
                                   * chart[i2+1, i3, "R_s", r_i]
                            if chart[i1, i3, "R_a", l_i] < score:
                                chart[i1, i3, "R_a", l_i] = score
                                back_ptr[i1, i3, "R_a", l_i] = [l_i, "R_s", l_i, "R_s", r_i, i2]

                # Handle unaries
                added = True
                while added:
                    added = False
                    for h_i in range(i1, i3+1):
                        # "L_s[h] -> h" (can be ignored)

                        # "L_s[h] -> L_a[h]"
                        score = model.get_stop_prob(head=postags[h_i],
                                                    direction=LEFT,
                                                    valency=0,
                                                    stop=CONT) \
                                * chart[i1, i3, "L_a", h_i]
                        if chart[i1, i3, "L_s", h_i] < score:
                            chart[i1, i3, "L_s", h_i] = score
                            back_ptr[i1, i3, "L_s", h_i] = [h_i, "L_a", h_i]
                            added = True

                        # "R_s[h] -> L_s[h]"
                        score = model.get_stop_prob(head=postags[h_i],
                                                    direction=RIGHT,
                                                    valency=0,
                                                    stop=STOP) \
                                * chart[i1, i3, "L_s", h_i]
                        if chart[i1, i3, "R_s", h_i] < score:
                            chart[i1, i3, "R_s", h_i] = score
                            back_ptr[i1, i3, "R_s", h_i] = [h_i, "L_s", h_i]
                            added = True

                        # "R_s[h] -> R_a[h]"
                        score = model.get_stop_prob(head=postags[h_i],
                                                    direction=RIGHT,
                                                    valency=0,
                                                    stop=CONT) \
                                * chart[i1, i3, "R_a", h_i]
                        if chart[i1, i3, "R_s", h_i] < score:
                            chart[i1, i3, "R_s", h_i] = score
                            back_ptr[i1, i3, "R_s", h_i] = [h_i, "R_a", h_i]
                            added = True

        # "S -> R_s[h]"
        # Note that this part should be excluded from the general case loop
        # to deal with sentences of length 1
        for h_i in range(0, n_postags):
            score = model.get_root_prob(dep=postags[h_i]) \
                    * chart[0, n_postags-1, "R_s", h_i]
            chart[0, n_postags-1, "S", h_i] = score
            back_ptr[0, n_postags-1, "S", h_i] = [h_i, "R_s", h_i]

        # Find the best-scoring head for the entire sentence
        max_head = None
        max_score = -np.inf
        for h_i in range(0, n_postags):
            if max_score < chart[0, n_postags-1, "S", h_i]:
                max_head = h_i
                max_score = chart[0, n_postags-1, "S", h_i]

        if chart[0, n_postags-1, "S", max_head] > 0.0:
            sexp = self.recover_tree(postags, back_ptr, 0, n_postags-1, "S", max_head)
            sexp = treetk.preprocess(sexp)
            arcs = self.convert_cfgsexp_to_arcs(sexp)
            return arcs
        else:
            raise ValueError("Unable to find a complete dependency-CFG tree to %s" % postags)

    def recover_tree(self, postags, back_ptr, i1, i3, A, h_i):
        """
        :type postags: list of str
        :type back_ptr: {(int, int, str, int): [int, str, int, str, int, int]/[int, str, int]}
        :type i1: int
        :type i3: int
        :type A: str
        :type h_i: int
        :rtype: str
        """
        if len(back_ptr[i1, i3, A, h_i]) == 3:
            # Handle unaries
            _, B, d_i = back_ptr[i1, i3, A, h_i]
            inner = self.recover_tree(postags, back_ptr, i1, i3, B, d_i)
            if A != "S":
                return "( %s[%s] %s )" % (A, postags[h_i], inner)
            else:
                return "( %s %s )" % (A, inner)
        else:
            if i1 == i3:
                return "( %s[%s] %s )" % (A, postags[h_i], postags[i1])
            else:
                h_i, B, B_i, C, C_i, i2 = back_ptr[i1, i3, A, h_i]
                inner1 = self.recover_tree(postags, back_ptr, i1, i2, B, B_i)
                inner2 = self.recover_tree(postags, back_ptr, i2+1, i3, C, C_i)
                return "( %s[%s] %s %s )" % (A, postags[h_i], inner1, inner2)

    def convert_cfgsexp_to_arcs(self, sexp):
        """
        :type sexp: list of str
        :rtype: list of (int, int)
        """
        tree = treetk.sexp2tree(sexp, with_nonterminal_labels=True, with_terminal_labels=True)
        tree.calc_heads(func_head_child_rule=self.func_head_child_rule)
        arcs = self.aggregate_arcs(tree)
        return arcs

    def func_head_child_rule(self, node):
        """
        :type node: NonTerminal
        :rtype: int
        """
        if node.label.startswith("R_a"):
            # R_a[h] -> R_s1[h] R_s0[d]
            assert len(node.children) == 2
            return 0
        elif node.label.startswith("L_a"):
            # L_a[h] -> R_s0[d] L_s1[h]
            assert len(node.children) == 2
            return 1
        else:
            assert len(node.children) == 1
            return 0

    def aggregate_arcs(self, node, arcs=None):
        """
        :type node: NonTerminal/Terminal:
        :type arcs: list of (int, int), or None
        :rtype: (int, int), or None
        """
        if arcs is None:
            arcs = []

        if node.is_terminal():
            return arcs

        if node.label == "S":
            # S -> R_s[d] = (S, d)
            head = 0
            dep = node.head_token_index + 1
            arcs.append((head, dep))
        elif node.label.startswith("R_a"):
            # R_a[h] -> R_s[h] R_s[d] = (h, d)
            head = node.head_token_index + 1
            dep = node.children[1].head_token_index + 1
            arcs.append((head, dep))
        elif node.label.startswith("L_a"):
            # L_a[h] -> R_s[d] L_s[h] = (h, d)
            head = node.head_token_index + 1
            dep = node.children[0].head_token_index + 1
            arcs.append((head, dep))

        for c in node.children:
            arcs = self.aggregate_arcs(c, arcs)
        return arcs

