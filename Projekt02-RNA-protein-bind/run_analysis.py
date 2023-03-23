from pathlib import Path
import pandas as pd


DATA_DIR = Path("../data/P2/HNRNPA2B1")
FSHAPE_DIR = DATA_DIR / "hnrnpa2b1_binding_sites_fshape"
EXPECTED_PATTERN_FILE = DATA_DIR / "hnrnpa2b1_expected_pattern.txt"


def load_expected_pattern() -> pd.DataFrame:
    return pd.read_csv(EXPECTED_PATTERN_FILE, delimiter="\t", header=None)


def process_fshape_files(length: int) -> list[pd.DataFrame]:
    window_sizes = [length, length + 1, length + 2]
    possible_seqs = {l: [] for l in window_sizes}
    for file in FSHAPE_DIR.iterdir():
        seq_df = pd.read_csv(file, delimiter="\t", header=None)
        for window_size in window_sizes:
            for i in range(0, len(seq_df) - window_size):
                subseq = seq_df.iloc[i : i + window_size]
                if (subseq.iloc[:, 0] > 1.0).any():
                    possible_seqs[window_size].append(subseq)
    return possible_seqs


def main() -> None:
    expected_pattern = load_expected_pattern()
    pattern_length = expected_pattern.shape[0]

    possibe_seqs = process_fshape_files(pattern_length)


if __name__ == "__main__":
    main()
