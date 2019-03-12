import os

import utils
import textpreprocessor.lowercase

def main():
    config = utils.Config()

    path_root = os.path.join(config.getpath("data"), "ptbwsj-dependencies")
    for split in ["train", "dev", "test"]:
        filename = "%s.tokens" % split
        textpreprocessor.lowercase.run(
                os.path.join(path_root, filename),
                os.path.join(path_root, filename + ".preprocessed"))

if __name__ == "__main__":
    main()
