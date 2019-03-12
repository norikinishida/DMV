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
    # "Noun": ["NN", "NNS", "NNP", "NNPS", "VBG"], # -> Noun, Nounorverb
    # "Verb": ["VB", "VBD", "VBN", "VBP", "VBZ"], # -> Verb, NounOrVerb
    "Noun": ["NN", "NNS", "NNP", "NNPS"], # -> Noun, NounOrVerb
    "Verb": ["VB", "VBD", "VBN", "VBP", "VBZ", "VBG"], # -> Verb, NounOrVerb
    "Pronoun": ["PRP", "WP"], # -> Noun, NounOrVerb
    "Auxiliary": ["MD"], # -> Verb, NounOrVerb
    "Adjective": ["JJ", "JJR", "JJS"],
    "Adverb": ["RB", "RBS", "WRB", "RBR"],
    # "Demonstrative": ["WDT", "DT", "WP$", "PRP$"],
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

