import argparse
import os
import time

import numpy as np
from chainer import cuda
import pyprind

import utils
import treetk

import dataloader
import models
import parsers
import baselines
from globalvars import CORPUS
from globalvars import MAX_EPOCH, MAX_PATIENCE

###################
# Training

def train(model,
          parser,
          em_type, smoothing_param, n_iters_per_m_step,
          train_databatch, dev_databatch,
          path_snapshot):
    """
    :type model: Model
    :type parser: CKYParser
    :type em_type: str
    :type smoothing_param: float
    :type n_iters_per_m_step: int
    :type train_databatch: DataBatch
    :type dev_databatch: DataBatch
    :type path_snapshot: str
    :rtype: None
    """
    bestscore_holder = utils.BestScoreHolder(scale=100.0)
    bestscore_holder.init()

    # Initial validation
    dda, uda, info1, info2 = evaluate(model=model,
                                      parser=parser,
                                      databatch=dev_databatch)
    utils.writelog("dev", "epoch=0, Directed=%.02f%% (%s), Undirected=%.02f%% (%s)" \
            % (dda * 100.0, info1, uda * 100.0, info2))

    for epoch in range(1, MAX_EPOCH+1):
        # EM for one iteration

        ###############
        # E Step
        utils.writelog("training", "epoch=%d, E step (%s), processing %d instances" % (epoch, em_type, len(train_databatch)))
        stop_counts, attach_counts, root_counts \
                            = model.e_step(
                                        databatch=train_databatch,
                                        em_type=em_type,
                                        smoothing_param=smoothing_param)
        ###############

        ###############
        # M Step
        for m_step_iter in range(n_iters_per_m_step):
            loss = model.m_step(stop_counts, attach_counts, root_counts)
            utils.writelog("training", "epoch=%d, M step %d/%d, loss=%f" % (epoch, m_step_iter+1, n_iters_per_m_step, loss))
        ###############

        # Validation
        dda, uda, info1, info2 = evaluate(model=model,
                                          parser=parser,
                                          databatch=dev_databatch)
        utils.writelog("dev", "epoch=%d, Directed=%.02f%% (%s), Undirected=%.02f%% (%s)" \
                % (epoch, dda * 100.0, info1, uda * 100.0, info2))

        # Saving
        did_update = bestscore_holder.compare_scores(dda, epoch)
        if did_update:
            model.save(path_snapshot)
            utils.writelog("model", "Saved the model to %s" % path_snapshot)

        # Finished?
        if bestscore_holder.ask_finishing(max_patience=MAX_PATIENCE):
            utils.writelog("info", "Patience %d is over. Training finished successfully." \
                    % bestscore_holder.patience)
            return

###################
# Evaluation

def evaluate(model, parser, databatch):
    """
    :type model: DMV
    :type: parser: CKYParser
    :type databatch: DataBatch
    :rtype: float, float, str, str
    """
    golds = databatch.batch_arcs
    preds = []

    # Aggregation
    for postags in pyprind.prog_bar(databatch.batch_postags):
        arcs = parser.parse(postags[1:], model) # NOTE
        preds.append(arcs)

    # Directed/Undirected Dependency Accuracy
    dda, uda, info1, info2 = compute_dependency_accuracy(preds, golds, databatch.batch_postags)
    return dda, uda, info1, info2

def evaluate_baseline(parser, databatch):
    """
    :type parser: BaselineParser
    :type databatch: DataBatch
    :rtype: float, float, str, str
    """
    golds = databatch.batch_arcs
    preds = []

    # Aggregation
    for postags in pyprind.prog_bar(databatch.batch_postags):
        arcs = parser.parse(postags)
        preds.append(arcs)

    # Directed/Undirected Dependency Accuracy
    dda, uda, info1, info2 = compute_dependency_accuracy(preds, golds, databatch.batch_postags)
    return dda, uda, info1, info2

def compute_dependency_accuracy(preds, golds, batch_postags):
    """
    :type preds: list of list of (int, int)
    :type golds: list of list of (int, int, str)
    :type batch_postags: list of list of str
    :rtype: float, float, str, str
    """
    assert len(preds) == len(golds) == len(batch_postags)

    REMOVAL_PUNCTS = ["``", "''", ":", ",", "."]

    total_ok_dir = 0.0
    total_ok_undir = 0.0
    total_arcs = 0.0

    for pred_arcs, gold_arcs, postags in zip(preds, golds, batch_postags):
        assert len(pred_arcs) == len(gold_arcs) == len(postags) - 1

        n_ok_dir = 0.0
        n_ok_undir = 0.0
        n_arcs = 0.0

        pred_dtree = treetk.arcs2dtree(pred_arcs)
        gold_dtree = treetk.arcs2dtree(gold_arcs)

        undir_pred_arcs = []
        undir_gold_arcs = []
        for d in range(len(postags)):
            if d == 0:
                continue # Ignore ROOT
            if postags[d] in REMOVAL_PUNCTS:
                continue # Ignore removal punctuations
            pred_h, _ = pred_dtree.get_head(d)
            gold_h, _ = gold_dtree.get_head(d)

            n_arcs += 1.0

            if pred_h == gold_h:
                n_ok_dir += 1.0

            undir_pred_arcs.append(sorted((pred_h, d)))
            undir_gold_arcs.append(sorted((gold_h, d)))

        for undir_pred_arc in undir_pred_arcs:
            if undir_pred_arc in undir_gold_arcs:
                n_ok_undir += 1.0

        total_ok_dir += n_ok_dir
        total_ok_undir += n_ok_undir
        total_arcs += n_arcs

    dda = total_ok_dir / total_arcs
    uda = total_ok_undir / total_arcs
    info1 = "%d/%d" % (total_ok_dir, total_arcs)
    info2 = "%d/%d" % (total_ok_undir, total_arcs)
    return dda, uda, info1, info2

###################
# Analysis

def dump_outputs(path, model, parser, databatch):
    """
    :type path: str
    :type mode: DMV
    :type parser: CKYParser
    :type databatch: DataBatch
    :rtype: None
    """
    with open(path, "w") as f:
        i = 0
        prog_bar = pyprind.ProgBar(len(databatch))
        for tokens, postags, gold_arcs in zip(
                                            databatch.batch_tokens,
                                            databatch.batch_postags,
                                            databatch.batch_arcs):
            pred_arcs = parser.parse(postags[1:], model) # NOTE

            tokpos = ["%s/%s" % (token,pos) for token,pos in zip(tokens, postags)]
            pred_dtree = treetk.arcs2dtree(arcs=pred_arcs, tokens=tokpos)
            gold_dtree = treetk.arcs2dtree(arcs=gold_arcs, tokens=tokpos)
            pred_result = ["%s-%s-%s" % (x[0], x[1], x[2]) for x in pred_dtree.tolist()]
            gold_result = ["%s-%s-%s" % (x[0], x[1], x[2]) for x in gold_dtree.tolist()]

            f.write("[%d] [tokens] %s\n" % (i, " ".join(tokens)))
            f.write("[%d] [postags] %s\n" % (i, " ".join(postags)))
            f.write("[%d] [gold-arcs] %s\n" % (i, " ".join(gold_result)))
            f.write("[%d] [pred-arcs] %s\n" % (i, " ".join(pred_result)))
            f.write("[%d] [gold-tree]\n%s\n" % (i, treetk.pretty_print_dtree(gold_dtree, return_str=True)))
            f.write("[%d] [pred-tree]\n%s\n" % (i, treetk.pretty_print_dtree(pred_dtree, return_str=True)))
            f.write("############################\n")
            i += 1
            prog_bar.update()

###################
# Main

def main(args):
    ###################
    # Arguments & random seed
    gpu = args.gpu
    model_name = args.model
    path_config = args.config
    trial_name = args.name
    actiontype = args.actiontype

    assert actiontype in ["baseline", "train", "evaluation", "dump_outputs"]
    if actiontype == "baseline":
        assert os.path.basename(path_config) == "baseline.ini"

    if trial_name is None or trial_name == "None":
        trial_name = utils.get_current_time()

    ###################
    # Path setting
    config = utils.Config(path_config)

    basename = "%s.%s.%s" % (model_name,
                             utils.get_basename_without_ext(path_config),
                             trial_name)

    path_snapshot = os.path.join(config.getpath("snapshot"), basename + ".scores.tsv")
    path_log = os.path.join(config.getpath("log"), basename + ".log")
    path_eval = os.path.join(config.getpath("evaluation"), basename + ".eval")
    path_anal = os.path.join(config.getpath("analysis"), basename)

    if actiontype == "train":
        utils.set_logger(path_log)
    elif actiontype == "evaluation":
        utils.set_logger(path_eval)
    elif actiontype == "baseline":
        utils.set_logger(path_eval)

    ###################
    # Random seed
    # random_seed = 1234 # if you specify the random seed
    random_seed = trial_name
    random_seed = utils.hash_string(random_seed)
    np.random.seed(random_seed)
    cuda.cupy.random.seed(random_seed)

    ###################
    # Log so far
    utils.writelog("args", "gpu=%s" % gpu)
    utils.writelog("args", "model_name=%s" % model_name)
    utils.writelog("args", "path_config=%s" % path_config)
    utils.writelog("args", "trial_name=%s" % trial_name)
    utils.writelog("args", "actiontype=%s" % actiontype)

    utils.writelog("path", "path_snapshot=%s" % path_snapshot)
    utils.writelog("path", "path_log=%s" % path_log)
    utils.writelog("path", "path_eval=%s" % path_eval)
    utils.writelog("path", "path_anal=%s" % path_anal)

    utils.writelog("seed", "random_seed=%d" % random_seed)

    ###################
    # Data preparation
    begin_time = time.time()

    if CORPUS == "ptbwsj":
        train_databatch = dataloader.read_ptbwsj("train", min_length=2, max_length=10)
        dev_databatch = dataloader.read_ptbwsj("dev", min_length=-1, max_length=10)
        test_databatch = dataloader.read_ptbwsj("test", min_length=-1, max_length=10)
        vocab = utils.read_vocab(os.path.join(config.getpath("data"), "ptbwsj-vocab", "postags.vocab.txt"))
        assert not "<root>" in vocab
    else:
        raise ValueError("Invalid CORPUS=%s" % CORPUS)

    end_time = time.time()
    utils.writelog("corpus", "Loaded the data. %f [sec.]" % (end_time - begin_time))

    ###################
    # Evaluation of the baseline parser
    if actiontype == "baseline":
        # Baseline parser
        if model_name == "random":
            parser = baselines.Random()
        elif model_name == "left_headed":
            parser = baselines.LeftHeaded()
        elif model_name == "right_headed":
            parser = baselines.RightHeaded()
        else:
            raise ValueError("Invalid model_name=%s" % model_name)
        # Dev
        dda, uda, info1, info2 = evaluate_baseline(parser=parser,
                                                   databatch=dev_databatch)
        utils.writelog("dev", "Directed=%.02f%% (%s), Undirected=%.02f%% (%s)" \
                % (dda * 100.0, info1, uda * 100.0, info2))
        # Test
        dda, uda, info1, info2 = evaluate_baseline(parser=parser,
                                                   databatch=test_databatch)
        utils.writelog("test", "Directed=%.02f%% (%s), Undirected=%.02f%% (%s)" \
                % (dda * 100.0, info1, uda * 100.0, info2))
        utils.writelog("info", "Done. basename=%s" % basename)
        return

    ###################
    # Hyper parameters
    init_method = config.getstr("init_method")
    em_type = config.getstr("em_type")
    smoothing_param = config.getfloat("smoothing_param")
    optimizer_name = config.getstr("optimizer_name")
    weight_decay = config.getfloat("weight_decay")
    n_iters_per_m_step = config.getstr("n_iters_per_m_step")

    utils.writelog("hyperparams", "init_method=%s" % init_method)
    utils.writelog("hyperparams", "em_type=%s" % em_type)
    utils.writelog("hyperparams", "smoothing_param=%f" % smoothing_param)
    utils.writelog("hyperparams", "optimizer_name=%s" % optimizer_name)
    utils.writelog("hyperparams", "weight_decay=%f" % weight_decay)
    utils.writelog("hyperparams", "n_iters_per_m_step=%d" % n_iters_per_m_step)

    ###################
    # Model preparation
    if model_name == "dmv":
        model = models.DMV(vocab=vocab)
    elif model_name == "dmwov":
        model = models.DMwoV(vocab=vocab)
    elif model_name == "loglineardmv":
        cuda.get_device(gpu).use()
        model = models.LogLinearDMV(
                    vocab=vocab,
                    optimizer_name=optimizer_name,
                    weight_decay=weight_decay)
        model.to_gpu()
    else:
        raise ValueError("Unknown model_name=%s" % model_name)

    # Initialize/Load parameters
    if actiontype == "train":
        model.init_params(init_method=init_method, databatch=train_databatch)
    elif actiontype in ["evaluation", "dump_outputs"]:
        model.load_params(path_snapshot)
    else:
        raise ValueError("Unknown actiontype=%s" % actiontype)

    ###################
    # Parser preparation
    parser = parsers.CKYParser()
    if model_name == "dmwov":
        parser = parsers.CKYParserWithoutValence()

    ###################
    # Training, Evaluation, Analysis
    if actiontype == "train":
        train(model=model,
              parser=parser,
              em_type=em_type,
              smoothing_param=smoothing_param,
              n_iters_per_m_step=n_iters_per_m_step,
              train_databatch=train_databatch,
              dev_databatch=dev_databatch,
              path_snapshot=path_snapshot)

    elif actiontype == "evaluation":
        # Dev
        dda, uda, info1, info2 = evaluate(model=model,
                                          parser=parser,
                                          databatch=dev_databatch)
        utils.writelog("dev", "Directed=%.02f%% (%s), Undirected=%.02f%% (%s)" \
                % (dda * 100.0, info1, uda * 100.0, info2))
        # Test
        dda, uda, info1, info2 = evaluate(model=model,
                                          parser=parser,
                                          databatch=test_databatch)
        utils.writelog("test", "Directed=%.02f%% (%s), Undirected=%.02f%% (%s)" \
                % (dda * 100.0, info1, uda * 100.0, info2))

    elif actiontype == "dump_outputs":
        dump_outputs(path_anal + ".outputs", model, parser, dev_databatch)

    utils.writelog("info", "Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--actiontype", type=str, required=True)
    args = parser.parse_args()
    try:
        main(args)
    except Exception as e:
        utils.logger.error(e, exc_info=True)

