from __future__ import annotations

from typing import Optional

import numpy as np


class Pattern:
    def __init__(
        self,
        sequence: str,
        fSHAPE: list[float],
        file: Optional[str] = None,
        start: int = -1,
        end: int = -1,
    ):
        """
        Creates instance of Pattern class.
        Args:
            sequence: a string with nucleotide sequence, e.g. "TNT"
            fSHAPE: a list of float values representing reactivity of each nucleotide in sequence
            file: an optional name of the origin file of the pattern
            start: starting position of a sequence in a file
            end: ending position of a sequence in a file
        """
        self.sequence = sequence
        self.seq_len = len(sequence)
        self.fSHAPE = fSHAPE
        self.file = file
        self.start = start
        self.end = end
        self.associated_pattern: Optional[str] = None
        self.zned = 0.0
        self.ssf = 0.0
        self.aS = 0.0

    def matches_sequence(self, other_sequence: str, wildcard: str = "N") -> bool:
        """Check if two sequences are matching; `wildcard` can be any character"""
        if len(self.sequence) != len(other_sequence):
            print(f"Skipping a comparison: length mismatch")
            return False

        for ch1, ch2 in zip(self.sequence, other_sequence):
            if ch1 not in [ch2, wildcard] and ch2 != wildcard:
                return False
        return True

    def znormalized_euclidean_distance(self, other_fshape: list[float]):
        """Calculates the z-normalized Euclidean distance between two fSHAPE vectors"""
        norm = np.linalg.norm(np.array(self.fSHAPE) - np.array(other_fshape))
        if norm == 0:
            self.zned = 0.0
        else:
            self.zned = norm / np.sqrt(len(self.fSHAPE))
        self.update_aS()

    def similarity_score(self, other_sequence: str):
        """Calculates the similarity score between two sequences"""
        score = 0
        for i in range(self.seq_len):
            if self.matches_sequence(self.sequence[i], other_sequence[i]):
                score += 2
            elif any(
                self.sequence[i] in x and other_sequence[i] in x
                for x in ["AG", "CU", "CT", "UT"]
            ):
                score += 1
        self.ssf = score / self.seq_len
        self.update_aS()

    def update_aS(self):
        self.aS = 10 * self.zned - self.ssf

    def associate(self, other_pattern: Pattern):
        """Links the pattern with another matching pattern and updates similarity metrics"""
        self.associated_pattern = other_pattern
        self.znormalized_euclidean_distance(other_pattern.fSHAPE)
        self.similarity_score(other_pattern.sequence)

    def to_dict(self) -> dict:
        repr = vars(self)
        del repr["associated_pattern"]
        return repr
