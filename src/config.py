import torch
import os

codepath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LEARNED_DIM = 20
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

FLOAT="float"
LONG="long"

POS_DIM = 5
AMINO_ACID = 21

THRESHOLD = 500
THRESHOLDS = [100,500,1000,5000]

PAIR_LEVEL = "pair_level"
PAIR_LEVEL_IN = "pair_level_in"
NPAIR = "npair"
PAIR = "pair"

blosum_path = codepath + '/data/blosum62.txt'
PEPTIDE_LENGTH = 15
ENCODING_SIZE = 20

