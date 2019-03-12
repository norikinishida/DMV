from collections import defaultdict, OrderedDict

import numpy as np
import chainer
from chainer import cuda, serializers
import chainer.functions as F
import chainer.links as L
import pyprind

import utils
import treetk

import parsers
from .templatefeatureextractor import TemplateFeatureExtractor
from globalvars import RIGHT, LEFT, STOP, CONT
from globalvars import ITERS_AT_INIT_M
# from globalvars import COARSE_POSTAG_MAP

class LogLinearDMV(chainer.Chain):

    def __init__(self, vocab, optimizer_name, weight_decay):
        """
        :type vocab: {str: int}
        :type optimizer_name: str
        :type weight_decay: float

        P(STOP | h, ->, 0): R_s0[h] -> L_s0[h]
        P(CONT | h, ->, 0): R_s0[h] -> R_a[h]

        P(STOP | h, ->, 1): R_s1[h] -> L_s0[h]
        P(CONT | h, ->, 1): R_s1[h] -> R_a[h]

        P(STOP | h, <-, 0): L_s0[h] -> h
        P(CONT | h, <-, 0): L_s0[h] -> L_a[h]

        P(STOP | h, <-, 1): L_s1[h] -> h
        P(CONT | h, <-, 1): L_s1[h] -> L_a[h]

        P(d | h, ->): R_a[h] -> R_s1[h] R_s0[d]
        P(d | h, <-): L_a[h] -> R_s0[d] R_s1[h]

        P(h | S, ->) = S -> R_s0[h]
        """
        self.vocab = vocab
        self.optimizer_name = optimizer_name
        self.weight_decay = weight_decay

        self.stop_probs = OrderedDict() # {(str, str, int, str): float}
        self.attach_probs = OrderedDict() # {(str, str, str): float}
        self.root_probs = OrderedDict() # {str: float}

        for head in self.vocab.keys():
            for direction in [RIGHT, LEFT]:
                for valency in [0, 1]:
                    for stop in [STOP, CONT]:
                        self.stop_probs[head, direction, valency, stop] = 0.5
        for head in self.vocab.keys():
            for direction in [RIGHT, LEFT]:
                for dep in self.vocab.keys():
                    self.attach_probs[head, direction, dep] = 1.0 / float(len(self.vocab))
        for dep in self.vocab.keys():
            self.root_probs[dep] = 1.0 / float(len(self.vocab))

        rule_i = 0
        for head in self.vocab.keys():
            for direction in [RIGHT, LEFT]:
                for valency in [0, 1]:
                    for stop in [STOP, CONT]:
                        utils.writelog("LogLinearDMV", "Rule %04d: STOP(head=%s, dir=%s, val=%s, stop=%s)" % (rule_i, head, direction, valency, stop))
                        rule_i += 1
        for head in self.vocab.keys():
            for direction in [RIGHT, LEFT]:
                for dep in self.vocab.keys():
                    utils.writelog("LogLinearDMV", "Rule %04d: ATTACH(head=%s, dir=%s, dep=%s)" % (rule_i, head, direction, dep))
                    rule_i += 1
        for dep in self.vocab.keys():
            self.root_probs[dep] = 1.0 / float(len(self.vocab))
            utils.writelog("LogLinearDMV", "Rule %04d: ROOT(dep=%s)" % (rule_i, dep))
            rule_i += 1

        self.template_feature_extractor = TemplateFeatureExtractor(self.vocab)

        links = {}
        links["W"] = L.Linear(self.template_feature_extractor.feature_size, 1)
        super(LogLinearDMV, self).__init__(**links)

        self.opt = utils.get_optimizer(optimizer_name)
        self.opt.setup(self)
        self.opt.add_hook(chainer.optimizer.WeightDecay(weight_decay))

        # Filtering
        # self.mask_to_root_dist = np.ones((1, len(self.vocab)), dtype=np.float32)
        # for dep_i, dep in enumerate(self.vocab.keys()):
        #     if not ((dep in COARSE_POSTAG_MAP["Noun"]) or
        #             (dep in COARSE_POSTAG_MAP["Verb"]) or
        #             (dep in COARSE_POSTAG_MAP["Pronoun"]) or
        #             (dep in COARSE_POSTAG_MAP["Auxiliary"])):
        #         self.mask_to_root_dist[0, dep_i] = 0.0

    #########################

    def get_stop_prob(self, head, direction, valency, stop):
        """
        :type head: str
        :type direction: str
        :type valency: int
        :type stop: str
        :rtype: float
        """
        return self.stop_probs[head, direction, valency, stop]

    def get_attach_prob(self, head, direction, dep, head_i=None, dep_i=None, sigma=None):
        """
        :type head: str
        :type direction: str
        :type dep: str
        :type head_i: int
        :type dep_i: int
        :type sigma: float
        :rtype: float
        """
        prob = self.attach_probs[head, direction, dep]
        if (head_i is not None) and (dep_i is not None) and (sigma is not None):
            prob = prob * np.exp(sigma * np.abs(head_i - dep_i))
        return prob

    def get_root_prob(self, dep):
        """
        :type dep: str
        :rtype: float
        """
        return self.root_probs[dep]

    #########################
    # log-linear computation for multinomial distributions

    def compute_stop_dist(self, head, direction, valency):
        """
        :type head: str
        :type direction: str
        :type valency: int
        :rtype: Variable(shape=(2,D), dtype=np.float32)
        """
        feats = self.template_feature_extractor.extract_stop_features(head, direction, valency, stop=None) # (2, D)
        feats = utils.convert_ndarray_to_variable(feats, seq=False) # (2, D)
        logits = self.W(feats) # (2, 1)
        logits = F.transpose(logits) # (1, 2)
        dist = F.softmax(logits) # (1, 2)
        dist = F.transpose(dist) # (2, 1)
        return dist

    def compute_attach_dist(self, head, direction):
        """
        :type head: str
        :type direction: str
        :rtype: Variable(shape=(|V|,D), dtype=np.float32)
        """
        feats = self.template_feature_extractor.extract_attach_features(head, direction, dep=None) # (|V|, D)
        feats = utils.convert_ndarray_to_variable(feats, seq=False) # (|V|, D)
        logits = self.W(feats) # (|V|, 1)
        logits = F.transpose(logits) # (1, |V|)
        dist = F.softmax(logits) # (1, |V|)
        dist = F.transpose(dist) # (|V|, 1)
        return dist

    def compute_root_dist(self):
        """
        :rtype: Variable(shape=(|V|,D), dtype=np.float32)
        """
        # C = 100000.0
        # eps = 1e-9

        feats = self.template_feature_extractor.extract_root_features(dep=None) # (|V|, D)
        feats = utils.convert_ndarray_to_variable(feats, seq=False) # (|V|, D)
        logits = self.W(feats) # (|V|, 1)
        logits = F.transpose(logits) # (1, |V|)

        # Filtering (1)
        # mask = utils.convert_ndarray_to_variable(self.mask_to_root_dist, seq=False) # (1, |V|)
        # logits = logits - (1.0 - mask) * C

        dist = F.softmax(logits) # (1, |V|)
        dist = F.transpose(dist) # (|V|, 1)

        # Filtering (2)
        # dist = dist + eps # to avoid nan, i.e., log(0)

        return dist

    #########################
    # Initialization

    def init_params(self, init_method, databatch):
        """
        :type init_method: str
        :type databatch: DataBatch
        :rtype: None
        """
        if init_method == "uniform":
            self.uniform_initialization()
        elif init_method == "km":
            self.km_initialization(databatch)
        elif init_method == "noah":
            self.noah_initialization(databatch)
        elif init_method == "oracle":
            self.oracle_initialization(databatch)
        else:
            raise ValueError("Invalid init_method=%s" % init_method)

    def uniform_initialization(self):
        """
        :rtype: None
        """
        stop_counts = defaultdict(float) # {(str, str, int, str): float}
        attach_counts = defaultdict(float) # {(str, str, str): float}
        root_counts = defaultdict(float) # {str: str}

        # Smoothing
        stop_counts = self.apply_smoothing_to_stop_counts(stop_counts, 1.0)
        attach_counts = self.apply_smoothing_to_attach_counts(attach_counts, 1.0)
        root_counts = self.apply_smoothing_to_root_counts(root_counts, 1.0)

        # Normalization
        self.m_step(stop_counts, attach_counts, root_counts)

    def km_initialization(self, databatch):
        """
        :type: databatch
        :rtype: None
        """
        SMOOTHING = 1.0

        stop_counts = defaultdict(float) # {(str, str, int, str): float}
        attach_counts = defaultdict(float) # {(str, str, str): float}
        root_counts = defaultdict(float) # {str: str}

        # Smoothing
        stop_counts = self.apply_smoothing_to_stop_counts(stop_counts, 1.0) # const
        attach_counts = self.apply_smoothing_to_attach_counts(attach_counts, SMOOTHING)
        root_counts = self.apply_smoothing_to_root_counts(root_counts, 1.0) # const

        # Counting (only attachment)
        for postags in databatch.batch_postags:
            postags = postags[1:] # "<root>"を省く
            n_postags = len(postags)
            for h_i in range(0, n_postags):
                for d_i in range(0, n_postags):
                    if h_i == d_i:
                        continue
                    distance = float(np.abs(h_i - d_i))
                    if h_i < d_i:
                        attach_counts[postags[h_i], RIGHT, postags[d_i]] += 1.0 / distance
                    elif d_i < h_i:
                        attach_counts[postags[h_i], LEFT, postags[d_i]] += 1.0 / distance
                    else:
                        raise Exception("Never happen.")

        # Normalization
        for it in range(ITERS_AT_INIT_M):
            loss = self.m_step(stop_counts, attach_counts, root_counts)
            utils.writelog("init-m", "iter=%d, loss=%f" % (it+1, loss))

    # def noah_initialization(self, databatch):
    #     """
    #     :type: databatch
    #     :rtype: None
    #     """
    #     stop_counts = defaultdict(float) # {(str, str, int, str): float}
    #     attach_counts = defaultdict(float) # {(str, str, str): float}
    #     root_counts = defaultdict(float) # {str: str}
    #
    #     cc_stop_counts = defaultdict(float) # {(str, str, int, str): float}
    #     cc_attach_counts = defaultdict(float) # {(str, str, str): float}
    #     cc_root_counts = defaultdict(float) # {str: str}
    #
    #     # BLOCK A
    #     for postags in databatch.batch_postags:
    #         postags = postags[1:] # "<root>"を省く
    #
    #         n_postags = len(postags)
    #
    #         change_r = np.zeros((n_postags,))
    #         change_l = np.zeros((n_postags,))
    #
    #         # Update root counts
    #         for i in range(0, n_postags):
    #             root_counts[postags[i]] += 1.0 / float(n_postags)
    #
    #         # Update attachment counts
    #         for j in range(0, n_postags):
    #             sum_ = 0.0
    #
    #             for i in range(0, n_postags):
    #                 if i != j:
    #                     sum_ += 1.0 / np.abs(i - j)
    #
    #             n = float(n_postags)
    #             scale = ((n - 1.0) / n) * (1.0 / sum_)
    #
    #             for i in range(0, j):
    #                 update = scale * (1.0 / np.abs(j - i))
    #                 change_r[i] += update
    #                 attach_counts[postags[i], RIGHT, postags[j]] += update
    #
    #             for i in range(j+1, n_postags):
    #                 update = scale * (1.0 / np.abs(i - j))
    #                 change_l[i] += update
    #                 attach_counts[postags[i], LEFT, postags[j]] += update
    #
    #         # Update stop/continue counts
    #         for i in range(0, n_postags):
    #             if change_l[i] > 0.0:
    #                 stop_counts[postags[i], LEFT, 0, CONT]      += 0.0
    #                 cc_stop_counts[postags[i], LEFT, 0, CONT]   += 1.0
    #                 stop_counts[postags[i], LEFT, 1, CONT]      += change_l[i]
    #                 cc_stop_counts[postags[i], LEFT, 1, CONT]   += -1.0
    #
    #                 stop_counts[postags[i], LEFT, 0, STOP]      += 1.0
    #                 cc_stop_counts[postags[i], LEFT, 0, STOP]   += -1.0
    #                 stop_counts[postags[i], LEFT, 1, STOP]      += 0.0
    #                 cc_stop_counts[postags[i], LEFT, 1, STOP]   += 1.0
    #             else:
    #                 stop_counts[postags[i], LEFT, 0, STOP]      += 1.0
    #
    #             if change_r[i] > 0.0:
    #                 stop_counts[postags[i], RIGHT, 0, CONT]     += 0.0
    #                 cc_stop_counts[postags[i], RIGHT, 0, CONT]  += 1.0
    #                 stop_counts[postags[i], RIGHT, 1, CONT]     += change_r[i]
    #                 cc_stop_counts[postags[i], RIGHT, 1, CONT]  += -1.0
    #
    #                 stop_counts[postags[i], RIGHT, 0, STOP]     += 1.0
    #                 cc_stop_counts[postags[i], RIGHT, 0, STOP]  += -1.0
    #                 stop_counts[postags[i], RIGHT, 1, STOP]     += 0.0
    #                 cc_stop_counts[postags[i], RIGHT, 1, STOP]  += 1.0
    #             else:
    #                 stop_counts[postags[i], RIGHT, 0, STOP]     += 1.0
    #
    #     # BLOCK B (smoothing)
    #     for head in self.vocab.keys():
    #         for direction in [RIGHT, LEFT]:
    #             for valency in [0, 1]:
    #                 for stop in [STOP, CONT]:
    #                     stop_counts[head, direction, valency, stop] += 0.1
    #
    #     for head in self.vocab.keys():
    #         for direction in [RIGHT, LEFT]:
    #             for dep in self.vocab.keys():
    #                 attach_counts[head, direction, dep] += 0.1
    #
    #     for dep in self.vocab.keys():
    #         root_counts[dep] += 0.1
    #
    #     # BLOCK C
    #     max_e = 1.0
    #     min_e = 0.0
    #     for head in self.vocab.keys():
    #         for direction in [RIGHT, LEFT]:
    #             for valency in [0, 1]:
    #                 for stop in [STOP, CONT]:
    #                     num = stop_counts[head, direction, valency, stop]
    #                     if num > 0:
    #                         denom = cc_stop_counts[head, direction, valency, stop]
    #                         if denom < 0.0 and num > 0.0:
    #                             ratio = -1.0 * num / denom
    #                             if ratio < max_e:
    #                                 max_e = ratio
    #                         if 0.0 < denom and 0.0 < num:
    #                             ratio = -1.0 * num / denom
    #                             if min_e < ratio:
    #                                 min_e = ratio
    #                                 raise Exception("Never happen.")
    #     assert min_e == 0.0
    #
    #     # BLOCK D
    #     pr_first_kid = 0.9 * max_e + 0.1 * min_e
    #
    #     for head in self.vocab.keys():
    #         for direction in [LEFT, RIGHT]:
    #             for valency in [0, 1]:
    #                 for stop in [STOP, CONT]:
    #                     cc_stop_counts[head, direction, valency, stop] *= pr_first_kid
    #     for head in self.vocab.keys():
    #         for direction in [LEFT, RIGHT]:
    #             for valency in [0, 1]:
    #                 for stop in [STOP, CONT]:
    #                     stop_counts[head, direction, valency, stop] \
    #                             += cc_stop_counts[head, direction, valency, stop]
    #
    #     for head in self.vocab.keys():
    #         for direction in [RIGHT, LEFT]:
    #             for dep in self.vocab.keys():
    #                 attach_counts[head, direction, dep] \
    #                         += cc_attach_counts[head, direction, dep] * pr_first_kid
    #
    #     for dep in self.vocab.keys():
    #         root_counts[dep] += cc_root_counts[dep] * pr_first_kid
    #
    #     # BLOCK E (convert real to log)
    #     for head in self.vocab.keys():
    #         for direction in [RIGHT, LEFT]:
    #             for valency in [0, 1]:
    #                 for stop in [STOP, CONT]:
    #                     stop_counts[head, direction, valency, stop] \
    #                             = np.log(stop_counts[head, direction, valency, stop])
    #
    #     for head in self.vocab.keys():
    #         for direction in [RIGHT, LEFT]:
    #             for dep in self.vocab.keys():
    #                 attach_counts[head, direction, dep] \
    #                         = np.log(attach_counts[head, direction, dep])
    #
    #     for dep in self.vocab.keys():
    #         root_counts[dep] = np.log(root_counts[dep])
    #
    #     # BLOCK F (normalization in log space)
    #     def logsum(lx, ly):
    #         if lx == -np.inf:
    #             return ly
    #         if ly == -np.inf:
    #             return lx
    #         d = lx - ly
    #         if d >= 0:
    #             if d > 745:
    #                 return lx
    #             else:
    #                 return lx + np.log(1.0 + np.exp(-d))
    #         else:
    #             if d < -745:
    #                 return ly
    #             else:
    #                 return ly + np.log(1.0 + np.exp(d))
    #
    #     def logdivide(lx, ly):
    #         return lx - ly
    #
    #     for head in self.vocab.keys():
    #         for direction in [RIGHT, LEFT]:
    #             for valency in [0, 1]:
    #                 # Sum over stop of continue actions
    #                 Z = -np.inf
    #                 c1 = stop_counts[head, direction, valency, STOP]
    #                 c2 = stop_counts[head, direction, valency, CONT]
    #                 Z = logsum(c1, c2)
    #                 if Z == -np.inf:
    #                     self.stop_probs[head, direction, valency, STOP] = np.log(0.5)
    #                     self.stop_probs[head, direction, valency, CONT] = np.log(0.5)
    #                 else:
    #                     self.stop_probs[head, direction, valency, STOP] = logdivide(c1, Z)
    #                     self.stop_probs[head, direction, valency, CONT] = logdivide(c2, Z)
    #
    #     for head in self.vocab.keys():
    #         for direction in [RIGHT, LEFT]:
    #             # Sum over all children
    #             Z = -np.inf
    #             for dep in self.vocab.keys():
    #                 c = attach_counts[head, direction, dep]
    #                 Z = logsum(Z, c)
    #             if Z == -np.inf:
    #                 for dep in self.vocab.keys():
    #                     self.attach_probs[head, direction, dep] = np.log(1.0 / float(len(self.vocab)))
    #             else:
    #                 for dep in self.vocab.keys():
    #                     c = attach_counts[head, direction, dep]
    #                     self.attach_probs[head, direction, dep] = logdivide(c, Z)
    #
    #     # Sum over all children
    #     Z = -np.inf
    #     for dep in self.vocab.keys():
    #         c = root_counts[dep]
    #         Z = logsum(Z, c)
    #     if Z == -np.inf:
    #         for dep in self.vocab.keys():
    #             self.root_probs[dep] = np.log(1.0 / float(len(self.vocab)))
    #     else:
    #         for dep in self.vocab.keys():
    #             c = root_counts[dep]
    #             self.root_probs[dep] = logdivide(c, Z)
    #
    #     # BLOCK G (convert log to real)
    #     for head in self.vocab.keys():
    #         for direction in [RIGHT, LEFT]:
    #             for valency in [0, 1]:
    #                 for stop in [STOP, CONT]:
    #                     self.stop_probs[head, direction, valency, stop] \
    #                             = np.exp(self.stop_probs[head, direction, valency, stop])
    #
    #     for head in self.vocab.keys():
    #         for direction in [RIGHT, LEFT]:
    #             for dep in self.vocab.keys():
    #                 self.attach_probs[head, direction, dep] \
    #                         = np.exp(self.attach_probs[head, direction, dep])
    #
    #     for dep in self.vocab.keys():
    #         self.root_probs[dep] = np.exp(self.root_probs[dep])

    def oracle_initialization(self, databatch):
        """
        :type databatch: DataBatch
        :rtype: None
        """
        SMOOTHING = 1.0

        stop_counts = defaultdict(float) # {(str, str, int, str): float}
        attach_counts = defaultdict(float) # {(str, str, str): float}
        root_counts = defaultdict(float) # {str: str}

        # Smoothing
        stop_counts = self.apply_smoothing_to_stop_counts(stop_counts, SMOOTHING)
        attach_counts = self.apply_smoothing_to_attach_counts(attach_counts, SMOOTHING)
        root_counts = self.apply_smoothing_to_root_counts(root_counts, SMOOTHING)

        # Counting
        for postags, arcs in zip(databatch.batch_postags, databatch.batch_arcs):
            stop_rules, attach_rules, root_rules = self.convert_arcs_to_cfgrules(arcs, postags)
            for key in stop_rules:
                assert len(key) == 4 # head, direction, valency, stop
                stop_counts[key] += 1.0
            for key in attach_rules:
                assert len(key) == 3 # head, direction, dep
                attach_counts[key] += 1.0
            for key in root_rules:
                assert isinstance(key, str) # dep
                root_counts[key] += 1.0

        # Normalization
        for it in range(ITERS_AT_INIT_M):
            loss = self.m_step(stop_counts, attach_counts, root_counts)
            utils.writelog("init-m", "iter=%d, loss=%f" % (it+1, loss))

    #########################
    # Functions for E step

    def e_step(self, databatch, em_type, smoothing_param):
        """
        :type databatch: DataBatch
        :type em_type: str
        :type smoothing_param: float
        :rtype: {(str, str, int, str): float}, {(str, str, str): float}, {str: str}
        """
        if em_type == "standard":
            return self.standard_e_step(databatch, smoothing_param)
        elif em_type == "viterbi":
            return self.viterbi_e_step(databatch, smoothing_param)
        else:
            raise ValueError("Unknown em_type=%s" % em_type)

    def standard_e_step(self, databatch, smoothing_param):
        """
        :type databatch: DataBatch
        :type smoothing_param: float
        :rtype: {(str, str, int, str): float}, {(str, str, str): float}, {str: str}

        Standard EM using an inside-outside algorithm
        """
        # Initialize the expected counts of rules
        stop_counts = defaultdict(float)
        attach_counts = defaultdict(float)
        root_counts = defaultdict(float)

        # Smoothing
        stop_counts = self.apply_smoothing_to_stop_counts(stop_counts, smoothing_param)
        attach_counts = self.apply_smoothing_to_attach_counts(attach_counts, smoothing_param)
        root_counts = self.apply_smoothing_to_root_counts(root_counts, smoothing_param)

        # Counting
        for postags in pyprind.prog_bar(databatch.batch_postags):
            postags = postags[1:] # "<root>"を省く

            n_postags = len(postags)

            inside = self.compute_inside_prob(postags)
            outside = self.compute_outside_prob(postags, inside)

            # Attachment Rules / Binary Rules
            for d in range(1, n_postags):
                for i1 in range(0, n_postags-d):
                    i3 = i1 + d
                    for i2 in range(i1, i3):
                        for l_i in range(i1, i2+1):
                            for r_i in range(i2+1, i3+1):
                                # Count L_a[r] -> R_s0[l] L_s1[r]
                                attach_counts[postags[r_i], LEFT, postags[l_i]] \
                                        += self.get_attach_prob(head=postags[r_i],
                                                                direction=LEFT,
                                                                dep=postags[l_i]) \
                                            * outside[i1, i3, "L_a", r_i] \
                                            * inside[i1, i2, "R_s0", l_i] \
                                            * inside[i2+1, i3, "L_s1", r_i] \
                                            / inside[0, n_postags-1, "S", -1]
                                # Count R_a[l] -> R_s1[l] R_s0[r]
                                attach_counts[postags[l_i], RIGHT, postags[r_i]] \
                                        += self.get_attach_prob(head=postags[l_i],
                                                                direction=RIGHT,
                                                                dep=postags[r_i]) \
                                            * outside[i1, i3, "R_a", l_i] \
                                            * inside[i1, i2, "R_s1", l_i] \
                                            * inside[i2+1, i3, "R_s0", r_i] \
                                            / inside[0, n_postags-1, "S", -1]

            # Stop Rules / Unary Rules
            # Count stop(h, <-, v): L_sv[h] -> h
            for p in range(0, n_postags):
                stop_counts[postags[p], LEFT, 0, STOP] += self.get_stop_prob(head=postags[p],
                                                                             direction=LEFT,
                                                                             valency=0,
                                                                             stop=STOP) \
                                                          * outside[p, p, "L_s0", p] \
                                                          / inside[0, n_postags-1, "S", -1]
                stop_counts[postags[p], LEFT, 1, STOP] += self.get_stop_prob(head=postags[p],
                                                                             direction=LEFT,
                                                                             valency=1,
                                                                             stop=STOP) \
                                                          * outside[p, p, "L_s1", p] \
                                                          / inside[0, n_postags-1, "S", -1]

            for i1 in range(0, n_postags):
                for i3 in range(i1, n_postags):
                    for p in range(i1, i3+1):
                        # Count stop(h, ->, v): R_sv[h] -> L_s0[h]
                        stop_counts[postags[p], RIGHT, 0, STOP] += self.get_stop_prob(head=postags[p],
                                                                                      direction=RIGHT,
                                                                                      valency=0,
                                                                                      stop=STOP) \
                                                                    * outside[i1, i3, "R_s0", p] \
                                                                    * inside[i1, i3, "L_s0", p] \
                                                                    / inside[0, n_postags-1, "S", -1]
                        stop_counts[postags[p], RIGHT, 1, STOP] += self.get_stop_prob(head=postags[p],
                                                                                      direction=RIGHT,
                                                                                      valency=1,
                                                                                      stop=STOP) \
                                                                    * outside[i1, i3, "R_s1", p] \
                                                                    * inside[i1, i3, "L_s0", p] \
                                                                    / inside[0, n_postags-1, "S", -1]

                        # Count continue(h, ->, v): R_sv[h] -> R_a[h]
                        stop_counts[postags[p], RIGHT, 0, CONT] += self.get_stop_prob(head=postags[p],
                                                                                       direction=RIGHT,
                                                                                       valency=0,
                                                                                       stop=CONT) \
                                                                    * outside[i1, i3, "R_s0", p] \
                                                                    * inside[i1, i3, "R_a", p] \
                                                                    / inside[0, n_postags-1, "S", -1]
                        stop_counts[postags[p], RIGHT, 1, CONT] += self.get_stop_prob(head=postags[p],
                                                                                       direction=RIGHT,
                                                                                       valency=1,
                                                                                       stop=CONT) \
                                                                    * outside[i1, i3, "R_s1", p] \
                                                                    * inside[i1, i3, "R_a", p] \
                                                                    / inside[0, n_postags-1, "S", -1]
                        # Count continue(h, <-, v): L_sv[h] -> L_a[h]
                        stop_counts[postags[p], LEFT, 0, CONT] += self.get_stop_prob(head=postags[p],
                                                                                      direction=LEFT,
                                                                                      valency=0,
                                                                                      stop=CONT) \
                                                                    * outside[i1, i3, "L_s0", p] \
                                                                    * inside[i1, i3, "L_a", p] \
                                                                    / inside[0, n_postags-1, "S", -1]
                        stop_counts[postags[p], LEFT, 1, CONT] += self.get_stop_prob(head=postags[p],
                                                                                      direction=LEFT,
                                                                                      valency=1,
                                                                                      stop=CONT) \
                                                                    * outside[i1, i3, "L_s1", p] \
                                                                    * inside[i1, i3, "L_a", p] \
                                                                    / inside[0, n_postags-1, "S", -1]

            # ROOT Rules / Unary Rules
            # Count S -> R_s0[d]
            for p in range(0, n_postags):
                root_counts[postags[p]] += self.get_root_prob(dep=postags[p]) \
                                            * outside[0, n_postags-1, "S", p] \
                                            * inside[0, n_postags-1, "R_s0", p] \
                                            / inside[0, n_postags-1, "S", -1]

        return stop_counts, attach_counts, root_counts

    def compute_inside_prob(self, postags):
        """
        :type postags: list of str
        :rtype: {(int, int, str, int): float}
        """
        n_postags = len(postags)

        inside = defaultdict(float)
        # e.g., inside[begin, end, nonterminal_type, head] = float

        # Base case
        for i in range(0, n_postags):
            # "L_s0[h] -> h"
            inside[i, i, "L_s0", i] += self.get_stop_prob(head=postags[i],
                                                          direction=LEFT,
                                                          valency=0,
                                                          stop=STOP)
            # "L_s1[h] -> h"
            if 0 < i:
                inside[i, i, "L_s1", i] += self.get_stop_prob(head=postags[i],
                                                              direction=LEFT,
                                                              valency=1,
                                                              stop=STOP)

            # Handle unaries
            # "R_s0[h] -> L_s0[h]"
            inside[i, i, "R_s0", i] += self.get_stop_prob(head=postags[i],
                                                          direction=RIGHT,
                                                          valency=0,
                                                          stop=STOP) \
                                        * inside[i, i, "L_s0", i]
            # "R_s1[h] -> L_s0[h]"
            if i < n_postags-1:
                inside[i, i, "R_s1", i] += self.get_stop_prob(head=postags[i],
                                                              direction=RIGHT,
                                                              valency=1,
                                                              stop=STOP) \
                                            * inside[i, i, "L_s0", i]

        # General case
        for d in range(1, n_postags):
            for i1 in range(0, n_postags - d):
                i3 = i1 + d
                for i2 in range(i1, i3):
                    for l_i in range(i1, i2+1):
                        for r_i in range(i2+1, i3+1):
                            # "L_a[r] -> R_s0[l] L_s1[r]"
                            inside[i1, i3, "L_a", r_i] += self.get_attach_prob(head=postags[r_i],
                                                                                direction=LEFT,
                                                                                dep=postags[l_i]) \
                                                            * inside[i1, i2, "R_s0", l_i] \
                                                            * inside[i2+1, i3, "L_s1", r_i]

                            # "R_a[l] -> R_s1[l] R_s0[r]"
                            inside[i1, i3, "R_a", l_i] += self.get_attach_prob(head=postags[l_i],
                                                                                direction=RIGHT,
                                                                                dep=postags[r_i]) \
                                                            * inside[i1, i2, "R_s1", l_i] \
                                                            * inside[i2+1, i3, "R_s0", r_i]

                # Handle unaries
                for h_i in range(i1, i3+1):
                    # "L_s0[h] -> h" (can be ignored)
                    # "L_s1[h] -> h" (can be ignored)

                    # "L_s0[h] -> L_a[h]"
                    inside[i1, i3, "L_s0", h_i] += self.get_stop_prob(head=postags[h_i],
                                                                      direction=LEFT,
                                                                      valency=0,
                                                                      stop=CONT) \
                                                    * inside[i1, i3, "L_a", h_i]
                    # "L_s1[h] -> L_a[h]"
                    inside[i1, i3, "L_s1", h_i] += self.get_stop_prob(head=postags[h_i],
                                                                      direction=LEFT,
                                                                      valency=1,
                                                                      stop=CONT) \
                                                    * inside[i1, i3, "L_a", h_i]

                    # "R_s0[h] -> L_s0[h]"
                    inside[i1, i3, "R_s0", h_i] += self.get_stop_prob(head=postags[h_i],
                                                                      direction=RIGHT,
                                                                      valency=0,
                                                                      stop=STOP) \
                                                    * inside[i1, i3, "L_s0", h_i]
                    # "R_s1[h] -> L_s0[h]"
                    inside[i1, i3, "R_s1", h_i] += self.get_stop_prob(head=postags[h_i],
                                                                      direction=RIGHT,
                                                                      valency=1,
                                                                      stop=STOP) \
                                                    * inside[i1, i3, "L_s0", h_i]

                    # "R_s0[h] -> R_a[h]"
                    inside[i1, i3, "R_s0", h_i] += self.get_stop_prob(head=postags[h_i],
                                                                      direction=RIGHT,
                                                                      valency=0,
                                                                      stop=CONT) \
                                                    * inside[i1, i3, "R_a", h_i]
                    # "R_s1[h] -> R_a[h]"
                    inside[i1, i3, "R_s1", h_i] += self.get_stop_prob(head=postags[h_i],
                                                                      direction=RIGHT,
                                                                      valency=1,
                                                                      stop=CONT) \
                                                    * inside[i1, i3, "R_a", h_i]

        # "S -> R_s0[h]"
        for h_i in range(0, n_postags):
            inside[0, n_postags-1, "S", h_i] += self.get_root_prob(dep=postags[h_i]) \
                                                * inside[0, n_postags-1, "R_s0", h_i]
        for h_i in range(0, n_postags):
            inside[0, n_postags-1, "S", -1] += inside[0, n_postags-1, "S", h_i]
        return inside

    def compute_outside_prob(self, postags, inside):
        """
        :type postags: list of str
        :type inside: {(int, int, str, int): float}
        :rtype: {(int, int, str, int): float}
        """
        n_postags = len(postags)

        outside = defaultdict(float)

        # Base case
        for i in range(0, n_postags):
            outside[0, n_postags-1, "S", i] = 1.0

            # Handle unaries
            # "S -> R_s0[h]"
            outside[0, n_postags-1, "R_s0", i] += self.get_root_prob(dep=postags[i]) \
                                                  * outside[0, n_postags-1, "S", i]
            # "R_s0[h] -> L_s0[h]"
            outside[0, n_postags-1, "L_s0", i] += self.get_stop_prob(head=postags[i],
                                                                     direction=RIGHT,
                                                                     valency=0,
                                                                     stop=STOP) \
                                                    * outside[0, n_postags-1, "R_s0", i]
            # "L_s0[h] -> L_a[h]"
            if 0 < i: # If not leftmost token, can have left children
                outside[0, n_postags-1, "L_a", i] += self.get_stop_prob(head=postags[i],
                                                                        direction=LEFT,
                                                                        valency=0,
                                                                        stop=CONT) \
                                                    * outside[0, n_postags-1, "L_s0", i]
            # "R_s0[h] -> R_a[h]"
            if i < n_postags-1: # If not rightmost token, can have right children
                outside[0, n_postags-1, "R_a", i] += self.get_stop_prob(head=postags[i],
                                                                        direction=RIGHT,
                                                                        valency=0,
                                                                        stop=CONT) \
                                                    * outside[0, n_postags-1, "R_s0", i]

        # General case
        for i1 in range(0, n_postags):
            for i3 in range(n_postags-1, -1, -1):
                if (i1 == 0 and i3 == n_postags-1):
                    continue
                if i3 < i1:
                    continue

                # i0:i3 -> i0:(i1-1)  i1:i3
                for i0 in range(0, i1):
                    for l_i in range(i0, i1):
                        for r_i in range(i1, i3+1):
                            # "R_a[l] -> R_s1[l] R_s0[r]"
                            outside[i1, i3, "R_s0", r_i] += self.get_attach_prob(head=postags[l_i],
                                                                                 direction=RIGHT,
                                                                                 dep=postags[r_i]) \
                                                            * inside[i0, i1-1, "R_s1", l_i] \
                                                            * outside[i0, i3, "R_a", l_i]
                            # "L_a[r] -> R_s0[l] L_s1[r]
                            outside[i1, i3, "L_s1", r_i] += self.get_attach_prob(head=postags[r_i],
                                                                                 direction=LEFT,
                                                                                 dep=postags[l_i]) \
                                                            * inside[i0, i1-1, "R_s0", l_i] \
                                                            * outside[i0, i3, "L_a", r_i]
                # i1:i4 -> i1:i3  (i3+1):i4
                for i4 in range(i3+1, n_postags):
                    for l_i in range(i1, i3+1):
                        for r_i in range(i3+1, i4+1):
                            # "R_a[l] -> R_s1[l] R_s0[r]"
                            outside[i1, i3, "R_s1", l_i] += self.get_attach_prob(head=postags[l_i],
                                                                                 direction=RIGHT,
                                                                                 dep=postags[r_i]) \
                                                            * inside[i3+1, i4, "R_s0", r_i] \
                                                            * outside[i1, i4, "R_a", l_i]
                            # "L_a[r] -> R_s0[l] L_s1[r]
                            outside[i1, i3, "R_s0", l_i] += self.get_attach_prob(head=postags[r_i],
                                                                                 direction=LEFT,
                                                                                 dep=postags[l_i]) \
                                                            * inside[i3+1, i4, "L_s1", r_i] \
                                                            * outside[i1, i4, "L_a", r_i]

                # Handle unaries
                for h_i in range(i1, i3+1):
                    # "R_s0[h] -> L_s0[h]"
                    outside[i1, i3, "L_s0", h_i] += self.get_stop_prob(head=postags[h_i],
                                                                       direction=RIGHT,
                                                                       valency=0,
                                                                       stop=STOP) \
                                                    * outside[i1, i3, "R_s0", h_i]
                    # "R_s1[h] -> L_s0[h]"
                    if h_i < n_postags-1:
                        outside[i1, i3, "L_s0", h_i] += self.get_stop_prob(head=postags[h_i],
                                                                           direction=RIGHT,
                                                                           valency=1,
                                                                           stop=STOP) \
                                                        * outside[i1, i3, "R_s1", h_i]
                    # "R_s0[h] -> R_a[h]"
                    outside[i1, i3, "R_a", h_i] += self.get_stop_prob(head=postags[h_i],
                                                                      direction=RIGHT,
                                                                      valency=0,
                                                                      stop=CONT) \
                                                        * outside[i1, i3, "R_s0", h_i]
                    # "R_s1[h] -> R_a[h]"
                    if h_i < n_postags-1:
                        outside[i1, i3, "R_a", h_i] += self.get_stop_prob(head=postags[h_i],
                                                                          direction=RIGHT,
                                                                          valency=1,
                                                                          stop=CONT) \
                                                        * outside[i1, i3, "R_s1", h_i]
                    # "L_s0[h] -> L_a[h]"
                    outside[i1, i3, "L_a", h_i] += self.get_stop_prob(head=postags[h_i],
                                                                      direction=LEFT,
                                                                      valency=0,
                                                                      stop=CONT) \
                                                    * outside[i1, i3, "L_s0", h_i]
                    # "L_s1[h] -> L_a[h]"
                    if 0 < h_i:
                        outside[i1, i3, "L_a", h_i] += self.get_stop_prob(head=postags[h_i],
                                                                          direction=LEFT,
                                                                          valency=1,
                                                                          stop=CONT) \
                                                        * outside[i1, i3, "L_s1", h_i]

        return outside

    def viterbi_e_step(self, databatch, smoothing_param):
        """
        :type databatch: DataBatch
        :type smoothing_param: float
        :rtype: {(str, str, int, str): float}, {(str, str, str): float}, {str: str}

        Viterbi EM
        """
        stop_counts = defaultdict(float) # {(str, str, int, str): float}
        attach_counts = defaultdict(float) # {(str, str, str): float}
        root_counts = defaultdict(float) # {str: str}

        # Smoothing
        stop_counts = self.apply_smoothing_to_stop_counts(stop_counts, smoothing_param)
        attach_counts = self.apply_smoothing_to_attach_counts(attach_counts, smoothing_param)
        root_counts = self.apply_smoothing_to_root_counts(root_counts, smoothing_param)

        # Counting
        parser = parsers.CKYParser()
        for postags in pyprind.prog_bar(databatch.batch_postags):
            arcs = parser.parse(postags[1:], self) # best-scoring dependency tree
            stop_rules, attach_rules, root_rules = self.convert_arcs_to_cfgrules(arcs, postags)
            for key in stop_rules:
                assert len(key) == 4 # head, direction, valency, stop
                stop_counts[key] += 1.0
            for key in attach_rules:
                assert len(key) == 3 # head, direction, dep
                attach_counts[key] += 1.0
            for key in root_rules:
                assert isinstance(key, str) # dep
                root_counts[key] += 1.0

        return stop_counts, attach_counts, root_counts

    def apply_smoothing_to_stop_counts(self, stop_counts, smoothing_param):
        """
        :type smoothing_param: float
        :rtype: {(str, str, int, str): float}
        """
        for head in self.vocab.keys():
            for direction in [RIGHT, LEFT]:
                for valency in [0, 1]:
                    for stop in [STOP, CONT]:
                        stop_counts[head, direction, valency, stop] = smoothing_param
        return stop_counts

    def apply_smoothing_to_attach_counts(self, attach_counts, smoothing_param):
        """
        :type smoothing_param: float
        :rtype: {(str, str, str): float}
        """
        for head in self.vocab.keys():
            for direction in [RIGHT, LEFT]:
                for dep in self.vocab.keys():
                    attach_counts[head, direction, dep] = smoothing_param
        return attach_counts

    def apply_smoothing_to_root_counts(self, root_counts, smoothing_param):
        """
        :type smoothing_param: float
        :rtype: {str: str}
        """
        for dep in self.vocab.keys():
            root_counts[dep] = smoothing_param
        return root_counts

    #########################
    # Functions for M step

    def m_step(self, stop_counts, attach_counts, root_counts):
        """
        :type stop_counts: {(str, str, int, str): float}
        :type attach_counts: {(str, str, str): float}
        :type root_counts: {str: float}
        :rtype: float
        """
        for head in self.vocab.keys():
            for direction in [RIGHT, LEFT]:
                for valency in [0, 1]:
                    for stop in [STOP, CONT]:
                        stop_counts[head, direction, valency, stop] = float(stop_counts[head, direction, valency, stop])
        for head in self.vocab.keys():
            for direction in [RIGHT, LEFT]:
                for dep in self.vocab.keys():
                    attach_counts[head, direction, dep] = float(attach_counts[head, direction, dep])
        for dep in self.vocab.keys():
            root_counts[dep] = float(root_counts[dep])

        # Filtering
        # for dep in self.vocab.keys():
        #     if not ((dep in COARSE_POSTAG_MAP["Noun"]) or
        #             (dep in COARSE_POSTAG_MAP["Verb"]) or
        #             (dep in COARSE_POSTAG_MAP["Pronoun"]) or
        #             (dep in COARSE_POSTAG_MAP["Auxiliary"])):
        #         root_counts[dep] = 0.0

        # Update parameters
        with chainer.using_config("train", True):
            loss = 0.0 # negative log-likelihood
            # Stop
            for head in self.vocab.keys():
                for direction in [RIGHT, LEFT]:
                    for valency in [0, 1]:
                        dist = self.compute_stop_dist(head, direction, valency) # (2, 1)
                        dist = F.log(dist) # (2, 1)
                        for stop_i, stop in enumerate([STOP, CONT]):
                            loss += - stop_counts[head, direction, valency, stop] * dist[stop_i]
            # Attachment
            for head in self.vocab.keys():
                for direction in [RIGHT, LEFT]:
                    dist = self.compute_attach_dist(head, direction) # (|V|, 1)
                    dist = F.log(dist) # (|V|, 1)
                    for dep_i, dep in enumerate(self.vocab.keys()):
                        loss += - attach_counts[head, direction, dep] * dist[dep_i]
            # ROOT
            dist = self.compute_root_dist() # (|V|, 1)
            dist = F.log(dist) # (|V|, 1)
            for dep_i, dep in enumerate(self.vocab.keys()):
                loss += - root_counts[dep] * dist[dep_i]
            # Update
            self.zerograds()
            loss.backward()
            self.opt.update()
            loss_data = float(cuda.to_cpu(loss.data))

        # Compute caches
        with chainer.using_config("train", False), chainer.no_backprop_mode():
            # Stop
            for head in self.vocab.keys():
                for direction in [RIGHT, LEFT]:
                    for valency in [0, 1]:
                        dist = self.compute_stop_dist(head, direction, valency) # (2, 1)
                        dist = cuda.to_cpu(dist.data)
                        for stop_i, stop in enumerate([STOP, CONT]):
                            self.stop_probs[head, direction, valency, stop] = float(dist[stop_i])
            # Attachment
            for head in self.vocab.keys():
                for direction in [RIGHT, LEFT]:
                    dist = self.compute_attach_dist(head, direction) # (|V|, 1)
                    dist = cuda.to_cpu(dist.data)
                    for dep_i, dep in enumerate(self.vocab.keys()):
                        self.attach_probs[head, direction, dep] = float(dist[dep_i])
            # Root
            dist = self.compute_root_dist() # (|V|, 1)
            dist = cuda.to_cpu(dist.data)
            for dep_i, dep in enumerate(self.vocab.keys()):
                self.root_probs[dep] = float(dist[dep_i])

            return loss_data

    #########################
    # Saving/Loading

    def load_params(self, path):
        """
        :type path: str
        :rtype: None
        """
        for line in open(path):
            items = line.strip().split("\t")
            if items[0] == "STOP":
                # Set stop probs
                assert len(items) == 5 # "STOP", head, direction, valency, prob
                head = items[1]
                direction = items[2]
                valency = int(items[3])
                prob = float(items[4])
                assert (head, direction, valency, STOP) in self.stop_probs
                assert (head, direction, valency, CONT) in self.stop_probs
                self.stop_probs[head, direction, valency, STOP] = prob
                self.stop_probs[head, direction, valency, CONT] = 1.0 - prob
            elif items[0] == "ATTACH":
                # Set attachment probs
                assert len(items) == 5 # "ATTACH", head, direction, dep, prob
                head = items[1]
                direction = items[2]
                dep = items[3]
                prob = float(items[4])
                assert (head, direction, dep) in self.attach_probs
                self.attach_probs[head, direction, dep] = prob
            elif items[0] == "ROOT":
                # Set ROOT probs
                assert len(items) == 4 # "ROOT", "S", dep, prob
                assert items[1] == "S"
                dep = items[2]
                prob = float(items[3])
                assert dep in self.root_probs
                self.root_probs[dep] = prob

        serializers.load_npz(path + ".model", self)

    def save(self, path):
        """
        :type path: str
        :rtype: None
        """
        with open(path, "w") as f:
            for key in self.stop_probs.keys():
                assert len(key) == 4 # head, direction, valency, stop
                if key[-1] == CONT:
                    # Skip "continue"
                    continue
                prob = self.stop_probs[key]
                items = ["STOP"] + [str(x) for x in key[:-1]] + [str(prob)]
                assert len(items) == 5 # "STOP", head, direction, valency, prob
                f.write("%s\n" % "\t".join(items))

            for key in self.attach_probs.keys():
                assert len(key) == 3 # head, direction, dep
                prob = self.attach_probs[key]
                items = ["ATTACH"] + [str(x) for x in key] + [str(prob)]
                assert len(items) == 5 # "ATTACH", head, direction, dep, prob
                f.write("%s\n" % "\t".join(items))

            for key in self.root_probs.keys():
                assert isinstance(key, str) # dep
                prob = self.root_probs[key]
                items = ["ROOT", "S"] + [key] + [str(prob)]
                assert len(items) == 4 # "ROOT", "S", dep, prob
                f.write("%s\n" % "\t".join(items))

        serializers.save_npz(path + ".model", self)

    #########################
    # Others

    def convert_arcs_to_cfgrules(self, arcs, postags):
        """
        :type arcs: list of (int, int) or list of (int, int, str)
        :rtype: list of (str, str, int, str), list of (str, str, str), list of str
        """
        stop_rules = []
        attach_rules = []
        root_rules = []

        dtree = treetk.arcs2dtree(arcs=arcs, tokens=postags)

        for head in range(len(postags)):
            dependents = dtree.get_dependents(head)
            if head == 0:
                # ROOT arc
                if len(dependents) != 1:
                    utils.writelog("model", "Skipped an instance with multiple ROOT's dependents (%d)" % len(dependents))
                    treetk.pretty_print_dtree(dtree)
                    return [], [], []
                dep, _ = dependents[0]
                # S -> R_s0[d]
                root_rules.append(postags[dep])
            else:
                left_dependents = [dep for dep,_ in dependents if dep < head]
                right_dependents = [dep for dep,_ in dependents if dep > head]
                left_dependents = sorted(left_dependents, key=lambda x: -x)
                right_dependents = sorted(right_dependents)

                if len(right_dependents) == 0:
                    # R_s0[h] -> L_s0[h]
                    stop_rules.append( (postags[head], RIGHT, 0, STOP) )
                else:
                    right_1st_child = True
                    for dep in right_dependents:
                        if right_1st_child:
                            # R_s0[h] -> R_a[h]
                            stop_rules.append( (postags[head], RIGHT, 0, CONT) )
                            # R_a[h] -> R_s1[h] R_s0[d]
                            attach_rules.append( (postags[head], RIGHT, postags[dep]) )
                        else:
                            # R_s1[h] -> R_a[h]
                            stop_rules.append( (postags[head], RIGHT, 1, CONT) )
                            # R_a[h] -> R_s1[h] R_s0[d]
                            attach_rules.append( (postags[head], RIGHT, postags[dep]) )
                        right_1st_child = False
                    # R_s1[h] -> L_s0[h]
                    stop_rules.append( (postags[head], RIGHT, 1, STOP) )

                if len(left_dependents) == 0:
                    # L_s0[h] -> h
                    stop_rules.append( (postags[head], LEFT, 0, STOP) )
                else:
                    left_1st_child = True
                    for dep in left_dependents:
                        if left_1st_child:
                            # L_s0[h] -> L_a[h]
                            stop_rules.append( (postags[head], LEFT, 0, CONT) )
                            # L_a[h] -> R_s0[d] L_s1[h]
                            attach_rules.append( (postags[head], LEFT, postags[dep]) )
                        else:
                            # L_s1[h] -> L_a[h]
                            stop_rules.append( (postags[head], LEFT, 1, CONT) )
                            # L_a[h] -> R_s0[d] L_s1[h]
                            attach_rules.append( (postags[head], LEFT, postags[dep]) )
                        left_1st_child = False
                    # L_s1[h] -> h
                    stop_rules.append( (postags[head], LEFT, 1, STOP) )
        return stop_rules, attach_rules, root_rules


