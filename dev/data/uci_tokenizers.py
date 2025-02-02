import tiktoken

SQUARE_NAMES = [
    "a1",
    "b1",
    "c1",
    "d1",
    "e1",
    "f1",
    "g1",
    "h1",
    "a2",
    "b2",
    "c2",
    "d2",
    "e2",
    "f2",
    "g2",
    "h2",
    "a3",
    "b3",
    "c3",
    "d3",
    "e3",
    "f3",
    "g3",
    "h3",
    "a4",
    "b4",
    "c4",
    "d4",
    "e4",
    "f4",
    "g4",
    "h4",
    "a5",
    "b5",
    "c5",
    "d5",
    "e5",
    "f5",
    "g5",
    "h5",
    "a6",
    "b6",
    "c6",
    "d6",
    "e6",
    "f6",
    "g6",
    "h6",
    "a7",
    "b7",
    "c7",
    "d7",
    "e7",
    "f7",
    "g7",
    "h7",
    "a8",
    "b8",
    "c8",
    "d8",
    "e8",
    "f8",
    "g8",
    "h8",
]


def chessGptTokenizer() -> tiktoken.Encoding:
    """
    Defines a tiktoken-based BPE encoder for UCI chess moves. This
    tokenizer effectively tokenizes UCI moves by the square names.
    One notable variation is that promotions must be in upper-case.

    Vocabulary:
    Special Tokens (4): "<|pad|>", "<|startoftext|>", "<|endoftext|>", "<|unknown|>"
    Square Tokens (64): a1 through h8
    Promote Tokens (4): Q, B, R, N
    UNUSED (8120): Need 8192-4-64-4=8120 unused tokens of the form <|unused####|>
    """
    special_tokens = ["<|pad|>", "<|startoftext|>", "<|endoftext|>", "<|unknown|>"]
    unused_tokens = [f"<|unused{i:04d}" for i in range(8120)]
    chess_vocab = special_tokens + SQUARE_NAMES + list("QBRN") + unused_tokens
    mergeable_ranks = {k.encode(): v for (v, k) in enumerate(chess_vocab)}
    chess_pat_str = r"[a-h][1-8]|[QBRN]"

    enc = tiktoken.Encoding(
        name="chess_enc",
        pat_str=chess_pat_str,  # or \d|\s
        mergeable_ranks=mergeable_ranks,
        special_tokens={k: v for (v, k) in enumerate(special_tokens)},
    )

    return enc
