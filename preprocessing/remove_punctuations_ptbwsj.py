import os

import utils
import treetk

REMOVAL_PUNCTS = treetk.ptbwsj.PUNCTUATIONS
# REMOVAL_PUNCTS = REMOVAL_PUNCTS + ["(", ")", "{", "}"]

def remove_punct(sentence):
    """
    :type sentence: list of {str: str}
    :rtype: list of {str: str}
    """
    i = 0
    while i < len(sentence):
        l = sentence[i]
        pos = l["POSTAG"]
        if pos in REMOVAL_PUNCTS:
            parent = int(l["HEAD"])
            if (parent == i + 1): # NOTE that head index is 1-based, while loop is 0-based.
                raise ValueError("Invalid CoNLL line %s" % l) # head = dep; this should never happen
            sentence = sentence[:i] + sentence[i+1:]
            for j, m in enumerate(sentence):
                d = int(m["HEAD"])
                if d == i + 1:
                    d = parent
                    m["HEAD"] = str(d)
                assert(d != i + 1)
                if d > i + 1:
                    m["HEAD"] = str(int(m["HEAD"]) - 1)
                if j >= i:
                    m["ID"] = str(int(m["ID"]) - 1)
            i -= 1
        i += 1
    return sentence

def main():
    config = utils.Config()

    path_src = os.path.join(config.getpath("data"), "ptbwsj-conllx.concat.split")
    path_dst = os.path.join(config.getpath("data"), "ptbwsj-conllx.concat.split.filtered")

    utils.mkdir(path_dst)

    for split in ["train", "dev", "test"]:
        sentences = utils.read_conll(
                        os.path.join(path_src, "%s.conllx" % split),
                        keys=["ID",
                              "FORM", "LEMMA",
                              "POSTAG", "_1",
                              "_2",
                              "HEAD", "DEPREL",
                              "_3", "_4"])
        sentences = [remove_punct(s) for s in sentences]
        utils.write_conll(os.path.join(path_dst, "%s.conllx" % split), sentences)
        print("Processed %s, output in %s" % \
                (os.path.join(path_src, "%s.conllx" % split),
                 os.path.join(path_dst, "%s.conllx" % split)))

if __name__ == "__main__":
    main()
