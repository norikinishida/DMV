from collections import OrderedDict

import numpy as np

import utils

from globalvars import RIGHT, LEFT, STOP, CONT
from globalvars import COARSE_POSTAG_MAP, IS_NOUN, IS_VERB

class TemplateFeatureExtractor(object):

    def __init__(self, vocab):
        """
        :type vocab: {str: int}
        """

        self.vocab = vocab # list of str

        self.patterns = [] # list of str
        self.pattern2dim = None # {str: int}
        self.feature_size = None # int
        self.pattern2vector = None # {str: numpy.ndarray(shape=(1, feature_size), dtype=np.float32)}

        for coarse_postag in COARSE_POSTAG_MAP.keys():
            postags = " ".join(COARSE_POSTAG_MAP[coarse_postag])
            utils.writelog("TemplateFeatureExtractor", "COARSE_POSTAG_MAP: %s -> %s" % (coarse_postag, postags))
        utils.writelog("TemplateFeatureExtractor", "IS_NOUN: %s" % " ".join(IS_NOUN))
        utils.writelog("TemplateFeatureExtractor", "IS_VERB: %s" % " ".join(IS_VERB))

        self.aggregate_patterns()
        self.make_pattern2dim()
        self.make_pattern2vector()

        for pat_i, pat in enumerate(self.patterns):
            utils.writelog("TemplateFeatureExtractor", "Feature %04d: %s" % (pat_i, pat))
        utils.writelog("TemplateFeatureExtractor", "Feature size=%d" % self.feature_size)

    ##########################
    # Functions for generating pattern symbols

    def convert_to_stop_pattern(self, head, direction, valency, stop):
        """
        :type head: str
        :type direction: str
        :type valency: int
        :type stop: str
        :rtype: str
        """
        return "STOP(head=%s, dir=%s, val=%s, stop=%s)" % (head, direction, valency, stop)

    def convert_to_attach_pattern(self, head, direction, dep):
        """
        :type head: str
        :type direction: str
        :type dep: str
        :rtype: str
        """
        return "ATTACH(head=%s, dir=%s, dep=%s)" % (head, direction, dep)

    def convert_to_root_pattern(self, dep):
        """
        :type dep: str
        :rtype: str
        """
        return "ROOT(dep=%s)" % dep

    ##########################
    # Functions for aggregating patterns

    def aggregate_patterns(self):
        # Aggregate all patterns.
        # We add BASIC, NOUN, VERB, and NOUN-VERB patterns.
        # We also add BACK-OFF versions (ignoring direction and valency) for each pattern.

        ##########################
        # Features on stop/continue decisions
        # STOP(head=*, dir=*, val=*, stop=*)
        for head in self.vocab.keys():
            for direction in [RIGHT, LEFT]:
                for valency in [0, 1]:
                    for stop in [STOP, CONT]:
                        patterns = self.generate_stop_patterns(head, direction, valency, stop)
                        for pattern in patterns:
                            self.add_one_pattern(pattern)
        ##########################

        ##########################
        # Features on attachment decisions
        # ATTACH(head=*, dir=*, dep=*)
        for head in self.vocab.keys():
            for direction in [RIGHT, LEFT]:
                for dep in self.vocab.keys():
                    patterns = self.generate_attach_patterns(head, direction, dep)
                    for pattern in patterns:
                        self.add_one_pattern(pattern)
        ##########################

        ##########################
        # Features on root-attachment decisions
        # ROOT(dep=*)
        for dep in self.vocab.keys():
            patterns = self.generate_root_patterns(dep)
            for pattern in patterns:
                self.add_one_pattern(pattern)
        ##########################

        assert len(self.patterns) == len(set(self.patterns))

    def add_one_pattern(self, pattern):
        """
        :type pattern: str
        :rtype: None
        """
        if not pattern in self.patterns:
            self.patterns.append(pattern)

    ##########################
    # Functions for making the mapping from a pattern to a dimension index

    def make_pattern2dim(self):
        self.pattern2dim = {pat: dim for dim, pat in enumerate(self.patterns)}
        self.feature_size = len(self.patterns)

    ##########################
    # Functions for making the mapping from a pattern to a feature vector

    def make_pattern2vector(self):
        # Make mapping from a feature pattern to a feature vector
        self.pattern2vector = OrderedDict()

        for head in self.vocab.keys():
            for direction in [RIGHT, LEFT]:
                for valency in [0, 1]:
                    for stop in [STOP, CONT]:
                        self.add_stop_vector(head, direction, valency, stop)

        for head in self.vocab.keys():
            for direction in [RIGHT, LEFT]:
                for dep in self.vocab.keys():
                    self.add_attach_vector(head, direction, dep)

        for dep in self.vocab.keys():
            self.add_root_vector(dep)

    def add_stop_vector(self, head, direction, valency, stop):
        """
        :type head: str
        :type direction: str
        :type valency: int
        :type stop: str
        :rtype: None
        """
        patterns = self.generate_stop_patterns(head, direction, valency, stop)
        pattern_dims = [self.pattern2dim[pat] for pat in patterns] # list of int
        vector = utils.make_multihot_vectors(self.feature_size, [pattern_dims]) # (1, feature_size)
        query_pattern = self.convert_to_stop_pattern(head, direction, valency, stop)
        self.pattern2vector[query_pattern] = vector

    def add_attach_vector(self, head, direction, dep):
        """
        :type head: str
        :type direction: str
        :type dep: str
        :rtype: None
        """
        patterns = self.generate_attach_patterns(head, direction, dep)
        pattern_dims = [self.pattern2dim[pat] for pat in patterns] # list of int
        vector = utils.make_multihot_vectors(self.feature_size, [pattern_dims]) # (1, feature_size)
        query_pattern = self.convert_to_attach_pattern(head, direction, dep)
        self.pattern2vector[query_pattern] = vector

    def add_root_vector(self, dep):
        """
        :type dep: str
        :rtype: None
        """
        patterns = self.generate_root_patterns(dep)
        pattern_dims = [self.pattern2dim[pat] for pat in patterns] # list of int
        vector = utils.make_multihot_vectors(self.feature_size, [pattern_dims]) # (1, feature_size)
        query_pattern = self.convert_to_root_pattern(dep)
        self.pattern2vector[query_pattern] = vector

    ##########################
    # Functions for generating patterns from an input

    def postag2superclass(self, postag):
        """
        :type pos: str
        :rtype: str
        """
        for coarse_postag in IS_NOUN:
            if postag in COARSE_POSTAG_MAP[coarse_postag]:
                return "Noun"
        for coarse_postag in IS_VERB:
            if postag in COARSE_POSTAG_MAP[coarse_postag]:
                return "Verb"
        return postag

    def postag2hyperclass(self, postag):
        """
        :type pos: str
        :rtype: str
        """
        for coarse_postag in IS_NOUN:
            if postag in COARSE_POSTAG_MAP[coarse_postag]:
                return "NounOrVerb"
        for coarse_postag in IS_VERB:
            if postag in COARSE_POSTAG_MAP[coarse_postag]:
                return "NounOrVerb"
        return postag

    def generate_stop_patterns(self, head, direction, valency, stop):
        """
        :type head: str
        :type direction: str
        :type valency: int
        :type stop: str
        :rtype: list of str
        """
        patterns = []
        # BASIC
        patterns.append(self.convert_to_stop_pattern(head, direction, valency, stop))
        patterns.append(self.convert_to_stop_pattern(head, None, valency, stop))
        patterns.append(self.convert_to_stop_pattern(head, direction, None, stop))
        patterns.append(self.convert_to_stop_pattern(head, None, None, stop))
        # NOUN, VERB
        patterns.append(self.convert_to_stop_pattern(self.postag2superclass(head), direction, valency, stop))
        patterns.append(self.convert_to_stop_pattern(self.postag2superclass(head), None, valency, stop))
        patterns.append(self.convert_to_stop_pattern(self.postag2superclass(head), direction, None, stop))
        patterns.append(self.convert_to_stop_pattern(self.postag2superclass(head), None, None, stop))
        # NOUN-VERB
        patterns.append(self.convert_to_stop_pattern(self.postag2hyperclass(head), direction, valency, stop))
        patterns.append(self.convert_to_stop_pattern(self.postag2hyperclass(head), None, valency, stop))
        patterns.append(self.convert_to_stop_pattern(self.postag2hyperclass(head), direction, None, stop))
        patterns.append(self.convert_to_stop_pattern(self.postag2hyperclass(head), None, None, stop))
        return patterns

    def generate_attach_patterns(self, head, direction, dep):
        """
        :type head: str
        :type direction: str
        :type dep: str
        :rtype: list of str
        """
        patterns = []
        # BASIC
        patterns.append(self.convert_to_attach_pattern(head, direction, dep))
        patterns.append(self.convert_to_attach_pattern(head, None, dep))
        # NOUN, VERB
        patterns.append(self.convert_to_attach_pattern(self.postag2superclass(head), direction, dep))
        patterns.append(self.convert_to_attach_pattern(head, direction, self.postag2superclass(dep)))
        patterns.append(self.convert_to_attach_pattern(self.postag2superclass(head), direction, self.postag2superclass(dep)))
        patterns.append(self.convert_to_attach_pattern(self.postag2superclass(head), None, dep))
        patterns.append(self.convert_to_attach_pattern(head, None, self.postag2superclass(dep)))
        patterns.append(self.convert_to_attach_pattern(self.postag2superclass(head), None, self.postag2superclass(dep)))
        # NOUN-VERB
        patterns.append(self.convert_to_attach_pattern(self.postag2hyperclass(head), direction, dep))
        patterns.append(self.convert_to_attach_pattern(head, direction, self.postag2hyperclass(dep)))
        patterns.append(self.convert_to_attach_pattern(self.postag2hyperclass(head), direction, self.postag2hyperclass(dep)))
        patterns.append(self.convert_to_attach_pattern(self.postag2hyperclass(head), None, dep))
        patterns.append(self.convert_to_attach_pattern(head, None, self.postag2hyperclass(dep)))
        patterns.append(self.convert_to_attach_pattern(self.postag2hyperclass(head), None, self.postag2hyperclass(dep)))
        return patterns

    def generate_root_patterns(self, dep):
        """
        :type dep: str
        :rtype: list of str
        """
        patterns = []
        # BASIC
        patterns.append(self.convert_to_root_pattern(dep))
        # NOUN, VERB
        patterns.append(self.convert_to_root_pattern(self.postag2superclass(dep)))
        # NOUN-VERB
        patterns.append(self.convert_to_root_pattern(self.postag2hyperclass(dep)))
        return patterns

    ##########################
    # Functions for extracting feature vectors

    def extract_stop_features(self, head, direction, valency, stop=None):
        """
        :type head: str
        :type direction: str
        :type valency: int
        :type stop: str or None
        :rtype: numpy.ndarray(shape=(1, feature_size)/(2, feature_size), dtype=np.float32)
        """
        if stop is not None:
            return self.pattern2vector[self.convert_to_stop_pattern(head, direction, valency, stop)]
        else:
            return np.vstack([self.pattern2vector[self.convert_to_stop_pattern(head, direction, valency, stop_)] for stop_ in [STOP, CONT]])

    def extract_attach_features(self, head, direction, dep=None):
        """
        :type head: str
        :type direction: str
        :type dep: str or None
        :rtype: numpy.ndarray(shape=(1, feature_size)/(|V|, feature_size), dtype=np.float32)
        """
        if dep is not None:
            return self.pattern2vector[self.convert_to_attach_pattern(head, direction, dep)]
        else:
            return np.vstack([self.pattern2vector[self.convert_to_attach_pattern(head, direction, dep_)] for dep_ in self.vocab.keys()])

    def extract_root_features(self, dep=None):
        """
        :type dep: str or None
        :rtype: numpy.ndarray(shape=(1, feature_size)/(|V|, feature_size), dtype=np.float32)
        """
        if dep is not None:
            return self.pattern2vector[self.convert_to_root_pattern(dep)]
        else:
            return np.vstack([self.pattern2vector[self.convert_to_root_pattern(dep_)] for dep_ in self.vocab.keys()])

