# Constant
UNK_TOKEN = '<UNK>'
PAD_TOKEN = '<PAD>'
START_TAG = "<START>"
STOP_TAG = "<STOP>"

# Tag set
# tag_to_ix = {PAD_TOKEN: 0, "B": 1, "I": 2, "O": 3}
# ix_to_tag = {0: PAD_TOKEN, 1: "B", 2: "I", 3: "O"}

tag_to_ix = {PAD_TOKEN: 0, "B": 1, "I": 2, "O": 3, START_TAG: 4, STOP_TAG: 5}
ix_to_tag = {0: PAD_TOKEN, 1: "B", 2: "I", 3: "O", 4: START_TAG, 5: STOP_TAG}

