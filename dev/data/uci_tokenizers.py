import chess
import tiktoken


def chessGptTokenizer() -> tiktoken.Encoding:
    """
    Defines a tiktoken-based BPE encoder for UCI chess moves. This
    tokenizer effectively tokenizes UCI moves by the square names.
    One notable variation is that promotions must be in upper-case.

    Vocabulary:
    Special Tokens (4): "\<|pad|\>", "\<|startoftext|\>", "\<|endoftext|\>", "\<|unknown|\>"
    Square Tokens (64): a1 through h8
    Promote Tokens (4): Q, B, R, N
    UNUSED (8120): Need 8192-4-64-4=8120 unused tokens of the form <|unused####|>
    """
    special_tokens = ["<|pad|>", "<|startoftext|>", "<|endoftext|>", "<|unknown|>"]
    unused_tokens = [f"<|unused{i:04d}" for i in range(8120)]
    chess_vocab = special_tokens + chess.SQUARE_NAMES + list("QBRN") + unused_tokens
    mergeable_ranks = {k.encode():v for (v,k) in enumerate(chess_vocab)}
    chess_pat_str = r'[a-h][1-8]|[QBRN]'

    enc = tiktoken.Encoding(
        name="chess_enc",
        pat_str=chess_pat_str, # or \d|\s
        mergeable_ranks=mergeable_ranks,
        special_tokens={k:v for (v,k) in enumerate(special_tokens)},
    )

    return enc

