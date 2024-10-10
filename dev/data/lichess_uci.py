"""
Lichess UCI dataset
https://huggingface.co/datasets/austindavis/lichess_uci

example doc to highlight the structure of the dataset:
{
  "site": "ik1q7qh0",
  "transcript": "b2b3 g7g6 c1b2 g8f6 e2e4 f8g7 e4e5 f6d5 d2d4 e8g8 c2c4 e7e6 c4d5 e6d5...",
}

Example of downloading the 201301-moves dataset of Lichess UCI, from root directory:
python dev/data/fineweb.py -d 201301

Tokenization time varies based on the number of games played that month, later months
have more games. Here are the times on a single AMD Ryzen 9 5900HS (16-core):
-d=201301 (120k games) runs in <10 seconds
-d=201506 (2.3M games) runs in ~4 minutes
-d=202401 (90.2M games) runs in ~2.5 hours
"""
import argparse
import multiprocessing as mp
import os

import datasets
import numpy as np
from data_common import write_datafile
from datasets import load_dataset
from tqdm import tqdm
from uci_tokenizers import chessGptTokenizer

# ------------------------------------------

ds_builder = datasets.load_dataset_builder('austindavis/lichess_uci')
lichess_months = [subset.name[:-6] for subset in ds_builder.BUILDER_CONFIGS if subset.name.endswith('-moves')]

supported_model_types = ['gpt-2']

parser = argparse.ArgumentParser(description="Lichess UCI dataset preprocessing")
parser.add_argument("-d", "--date", choices=lichess_months, default="201301", help="Date (format: 'yyyymm') to be tokenized.")
parser.add_argument("-m", "--model_desc", choices=supported_model_types, default=supported_model_types[0], help="Model descriptor.")
parser.add_argument("-s", "--shard_size", type=int, default=10**8, help="Size of each data shard in the output .bin files, in tokens")
args = parser.parse_args()

local_dir = remote_name = args.date + "-moves"

# create the cache the local directory if it doesn't exist yet
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

# download the dataset

fw = load_dataset("austindavis/lichess_uci", name=remote_name, split='train')

def tokenize_gpt2(doc):
    # tokenizes a single document and returns a numpy array of uint16 tokens
    enc = chessGptTokenizer()
    encode = lambda s: enc.encode_ordinary(s)
    eot = enc._special_tokens['<|endoftext|>'] # end of text token
    tokens = [eot] # the special <|endoftext|> token delimits all documents
    tokens.extend(encode(doc["transcript"]))
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "token dictionary too large for uint16"
    tokens_np_uint = tokens_np.astype(np.uint16)
    return tokens_np_uint

token_dtype = {
    "gpt-2": np.uint16,
}[args.model_desc]

# tokenize all documents and write output shards, each of shard_size tokens (last shard has remainder)
nprocs = max(1, os.cpu_count() - 2) # don't hog the entire system
with mp.Pool(nprocs) as pool:
    shard_index = 0
    # preallocate buffer to hold current shard
    all_tokens_np = np.empty((args.shard_size,), dtype=token_dtype)
    token_count = 0
    progress_bar = None

    tokenize = lambda x: None
    if args.model_desc == "gpt-2":
        tokenize = tokenize_gpt2
    else:
        raise ValueError(f"unknown model {args.model_desc}")

    for tokens in pool.imap(tokenize, fw, chunksize=16):

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
            filename = os.path.join(DATA_CACHE_DIR, f"{args.date}_{split}_{shard_index:06d}.bin")
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
        filename = os.path.join(DATA_CACHE_DIR, f"{args.date}_{split}_{shard_index:06d}.bin")
        write_datafile(filename, (all_tokens_np[:token_count]).tolist(), args.model_desc)
