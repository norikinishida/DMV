CORPUS = "ptbwsj"

MAX_EPOCH = 200
MAX_PATIENCE = 10
ITERS_AT_INIT_M = 120
# ITERS_AT_INIT_M = 500

RIGHT = "RIGHT"
LEFT = "LEFT"
STOP = "STOP"
CONT = "CONT"

COARSE_POSTAG_MAP = {
    # "Noun": ["NN", "NNS", "NNP", "NNPS", "VBG"],
    # "Verb": ["VB", "VBD", "VBN", "VBP", "VBZ"],
    "Noun": ["NN", "NNS", "NNP", "NNPS"],
    "Verb": ["VB", "VBD", "VBN", "VBP", "VBZ", "VBG"],
    "Pronoun": ["PRP", "WP"],
    "Auxiliary": ["MD"],
    "Adjective": ["JJ", "JJR", "JJS"],
    "Adverb": ["RB", "RBS", "WRB", "RBR"],
    "Determiner": ["WDT", "DT", "WP$", "PRP$"],
    # "Adposition": ["IN", "TO"],
    "Preposition": ["IN", "TO"],
    "Conjunction": ["CC"],
    "Number": ["CD"],
    "Extra": ["EX", "FW", "PDT", "POS", "RP", "UH", "SYM", "LS"],
    "Root": ["<root>"],
}
IS_NOUN = ["Noun", "Pronoun"]
IS_VERB = ["Verb", "Auxiliary"]

