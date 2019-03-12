import os

import utils

def main():
    config = utils.Config()

    path_src = os.path.join(config.getpath("data"), "ptbwsj-conllx.concat.split.filtered")
    path_dst = os.path.join(config.getpath("data"), "ptbwsj-dependencies")

    utils.mkdir(path_dst)

    for split in ["train", "dev", "test"]:
        path_conll = os.path.join(path_src, "%s.conllx" % split)
        path_tokens = os.path.join(path_dst, "%s.tokens" % split)
        path_postags = os.path.join(path_dst, "%s.postags" % split)
        path_arcs = os.path.join(path_dst, "%s.arcs" % split)

        batch_tokens, batch_postags, batch_arcs = \
            utils.convert_conll_to_linebyline_format(
                path_conll=path_conll,
                keys=["ID",
                      "FORM", "LEMMA",
                      "POSTAG", "_1",
                      "_2",
                      "HEAD", "DEPREL",
                      "_3", "_4"],
                ID="ID",
                FORM="FORM",
                POSTAG="POSTAG",
                HEAD="HEAD",
                DEPREL="DEPREL")

        with open(path_tokens, "w") as ft,\
             open(path_postags, "w") as fp,\
             open(path_arcs, "w") as fa:
            for tokens, postags, arcs in zip(batch_tokens, batch_postags, batch_arcs):
                ft.write("%s\n" % " ".join(tokens))
                fp.write("%s\n" % " ".join(postags))
                fa.write("%s\n" % " ".join(["%s-%s-%s" % (h,d,l) for h,d,l in arcs]))

        print("Processed %s, output in %s" % \
                (os.path.join(path_src, "%s.conllx" % split),
                 os.path.join(path_dst, "%s.{tokens,postags,arcs}" % split)))

if __name__ == "__main__":
    main()

