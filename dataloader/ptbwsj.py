import os

import numpy as np

import utils
import treetk

def read_ptbwsj(split, min_length, max_length):
    """
    :type split: str
    :type min_length: int
    :type max_length: int
    :rtype: DataBatch
    """
    config = utils.Config()

    path_root = os.path.join(config.getpath("data"), "ptbwsj-dependencies")

    # Reading
    batch_tokens = utils.read_lines(os.path.join(path_root, "%s.tokens" % split),
                                    lambda line: ["<root>"] + line.split())
    batch_postags = utils.read_lines(os.path.join(path_root, "%s.postags" % split),
                                    lambda line: ["<root>"] + line.split())
    batch_arcs = utils.read_lines(os.path.join(path_root, "%s.arcs" % split),
                                    lambda line: treetk.hyphens2arcs(line.split()))
    assert len(batch_tokens) == len(batch_postags) == len(batch_arcs)

    # Filtering
    def condition_function(x):
        if min_length <= len(x)-1 <= max_length:
            return True
        else:
            return False
    batch_postags = utils.filter_by_condition(batch_tokens, batch_postags, condition_function)
    batch_arcs = utils.filter_by_condition(batch_tokens, batch_arcs, condition_function)
    batch_tokens = utils.filter_by_condition(batch_tokens, batch_tokens, condition_function)

    # Conversion to numpy.ndarray
    batch_tokens = np.asarray(batch_tokens, dtype="O")
    batch_postags = np.asarray(batch_postags, dtype="O")
    batch_arcs = np.asarray(batch_arcs, dtype="O")

    # Conversion to DataBatch
    databatch = utils.DataBatch(batch_tokens=batch_tokens,
                                batch_postags=batch_postags,
                                batch_arcs=batch_arcs)

    total_arcs = 0
    for postags in batch_postags:
        total_arcs += len(postags[:1])
    utils.writelog("dataloader.read_ptbwsj", "split=%s" % split)
    utils.writelog("dataloader.read_ptbwsj", "minimum length=%d" % min_length)
    utils.writelog("dataloader.read_ptbwsj", "maximum length=%d" % max_length)
    utils.writelog("dataloader.read_ptbwsj", "# of instances=%d" % len(databatch))
    utils.writelog("dataloader.read_ptbwsj", "# of arcs=%d" % total_arcs)
    return databatch

