"""
Lichess UCI dataset (for srs pretraining)
https://huggingface.co/datasets/austindavis/lichess_uci


example game to highlight the structure of the dataset:
{
  "site": "j1dkb5dw",
  "transcript": "e2e4 e7e6 d2d4 b7b6 a2a3 c8b7 b1c3 g8h6 c1h6 g7h6 f1e2 d8g5 e2g4 h6h5 g1f3 g5g6 f3h4 g6g5 g4h5 g5h4 d1f3 e8d8 f3f7 b8c6 f7e8"
}

Example of downloading the June 2023 moves dataset of Lichess UCI, from root directory:
python dev/data/lichess_uci.py -f -v 202306-moves 202305-moves 
The larger datasets run for small few hours, depending on your internet and computer.
"""
import os
import argparse
import multiprocessing as mp

import numpy as np
from datasets import load_dataset, concatenate_datasets, Dataset
from tqdm import tqdm

from lichess_uci_dates import VALID_LICHESS_MONTHS
from uci_tokenizers import UciTileTokenizer

from data_common import write_datafile
# ------------------------------------------

parser = argparse.ArgumentParser(description="Lichess UCI dataset preprocessing")
parser.add_argument("-v", "--version", type=str, nargs='+', default=["202306-moves"], help="Lichess UCI month(s). Provide multiple months separated by space.")
parser.add_argument("-f", "--filtered", action='store_true', help="Filter dataset to only transcript with promotion tokens.")
parser.add_argument("-m", "--model_desc", type=str, default="gpt-2", help="Model descriptor, gpt-2|llama-3")
parser.add_argument("-s", "--shard_size", type=int, default=10**7, help="Size of each data shard in the output .bin files, in tokens")
args = parser.parse_args()

# The Lichess UCI dataset has many possible subsamples available
assert all(v in VALID_LICHESS_MONTHS for v in args.version), f"Version must be one of:\n{VALID_LICHESS_MONTHS}"

# Convert the `args.version` list to a binary integer representation
version_int = sum(1 << i for i, month in enumerate(VALID_LICHESS_MONTHS) if month in args.version)
version_hex = hex(version_int)

local_dir = os.join('lichess_uci', version_hex) 


# create the cache the local directory if it doesn't exist yet
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

dataset_parts = []
for remote_name in args.version: 
    ds_subset = load_dataset("austindavis/lichess_uci", name=remote_name, split="train")
    
    if args.filtered:
        #filter the datset to only those transcripts that include a promotion token
        # a space exists after the bishop's `b` to disambiguate from the `b`` file
        filtered_subset = ds.filter(lambda x: any(p in x['transcript'].lower() for p in ['q','n','b ','r']))

    dataset_parts.append(filtered_subset)

lichess_uci: Dataset = concatenate_datasets(filteredataset_partsd_datasets)
name = "lichess_uci"


def tokenize_uci(game):
    # tokenizes a single game and returns a numpy array of uint32 tokens
    tokenizer = UciTileTokenizer()
    encode = lambda s: tokenizer.encode(s, add_special_tokens=True)
    tokens = encode(game['transcript'])
    tokens_np = np.array(tokens)
    tokens_np_uint = tokens_np.astype(np.uint16)
    return tokens_np_uint


token_dtype = np.uint16

# tokenize all documents and write output shards, each of shard_size tokens (last shard has remainder)
nprocs = max(1, os.cpu_count() - 2) # don't hog the entire system
with mp.Pool(nprocs) as pool:
    shard_index = 0
    # preallocate buffer to hold current shard
    all_tokens_np = np.empty((args.shard_size,), dtype=token_dtype)
    token_count = 0
    progress_bar = None

    tokenize = tokenize_uci

    for tokens in pool.imap(tokenize, lichess_uci, chunksize=16):

        # is there enough space in the current shard for the new tokens?
        if token_count + len(tokens) < args.shard_size:
            # simply append tokens to current shard
            all_tokens_np[token_count:token_count+len(tokens)] = tokens
            token_count += len(tokens)
            # update progress bar
            if progress_bar is None:
                progress_bar = tqdm(total=args.shard_size, unit="tokens", desc=f"Shard {shard_index}")
            progress_bar.update(len(tokens))
        else:
            # write the current shard and start a new one
            split = "val" if shard_index == 0 else "train"
            filename = os.path.join(DATA_CACHE_DIR, f"{name}_{split}_{shard_index:06d}.bin")
            # split the document into whatever fits in this shard; the remainder goes to next one
            remainder = args.shard_size - token_count
            progress_bar.update(remainder)
            all_tokens_np[token_count:token_count+remainder] = tokens[:remainder]
            write_datafile(filename, all_tokens_np.tolist(), args.model_desc)
            shard_index += 1
            progress_bar = None
            # populate the next shard with the leftovers of the current doc
            all_tokens_np[0:len(tokens)-remainder] = tokens[remainder:]
            token_count = len(tokens)-remainder

    # write any remaining tokens as the last shard
    if token_count != 0:
        split = "val" if shard_index == 0 else "train"
        filename = os.path.join(DATA_CACHE_DIR, f"{name}_{split}_{shard_index:06d}.bin")
        write_datafile(filename, (all_tokens_np[:token_count]).tolist(), args.model_desc)
