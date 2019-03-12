import os

import utils
import textpreprocessor.create_vocabulary
import treetk

def read_flatten_arcs(path):
    """
    :type path: str
    :rtype: list of (int, int, str)
    """
    batch_arcs = utils.read_lines(path, lambda line: treetk.hyphens2arcs(line.split())) # list of list of (int, int, str)
    flat_arcs = utils.flatten_lists(batch_arcs) # list of (int, int, str)
    return flat_arcs

def main():
    config = utils.Config()

    utils.mkdir(os.path.join(config.getpath("data"), "ptbwsj-vocab"))

    # Vocabulary for words
    textpreprocessor.create_vocabulary.run(
                os.path.join(config.getpath("data"),
                             "ptbwsj-dependencies",
                             "train.tokens.preprocessed"),
                os.path.join(config.getpath("data"),
                             "ptbwsj-vocab",
                             "words.vocab.txt"),
                prune_at=50000,
                min_count=3,
                # special_words=["<root>"],
                special_words=[],
                with_unk=True)

    # Vocabulary for POS tags
    textpreprocessor.create_vocabulary.run(
                os.path.join(config.getpath("data"),
                             "ptbwsj-dependencies",
                             "train.postags"),
                os.path.join(config.getpath("data"),
                             "ptbwsj-vocab",
                             "postags.vocab.txt"),
                prune_at=100000000000,
                min_count=0,
                # special_words=["<root>"],
                special_words=[],
                with_unk=False)

    # Vocabulary for dependency relations
    train_flat_arcs = read_flatten_arcs(os.path.join(config.getpath("data"), "ptbwsj-dependencies", "train.arcs"))
    dev_flat_arcs = read_flatten_arcs(os.path.join(config.getpath("data"), "ptbwsj-dependencies", "dev.arcs"))
    test_flat_arcs = read_flatten_arcs(os.path.join(config.getpath("data"), "ptbwsj-dependencies", "test.arcs"))
    flat_arcs = train_flat_arcs + dev_flat_arcs + test_flat_arcs
    labels = [l for h,d,l in flat_arcs] # list of str
    counter = utils.get_word_counter(lines=[labels])
    labels = counter.most_common() # list of (str, int)
    with open(os.path.join(config.getpath("data"), "ptbwsj-vocab", "labels.vocab.txt"), "w") as f:
        for label_i, (label, freq) in enumerate(labels):
            f.write("%s\t%d\t%d\n" % (label, label_i, freq))

    print("Done.")

if __name__ == "__main__":
    main()
