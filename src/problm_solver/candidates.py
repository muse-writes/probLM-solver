"""Internal representation of token & log-probability data."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass
class CandidateTokens:
    """Dataclass for token IDs and assigned log-probabilities."""

    candidate_ids: npt.NDArray[np.int32]
    candidate_logprobs: npt.NDArray[np.float64]


class CandidateGeneratorFactory:
    """Gets candidates from a vocabulary."""

    _TOP_P_HYBRID_THRESHOLD = np.float64(0.5)

    def get_candidate_generator(self, top_k: int, top_p: float) -> Callable:
        """Get a candidate generator function."""
        if top_k < 1:
            msg = 'top_k must be greater than 0'
            raise ValueError(msg)

        top_p_f64 = np.float64(top_p)
        if not np.float64(0.0) < top_p_f64 <= np.float64(1.0):
            msg = f'top_p must be in (0, 1], got {top_p}'
            raise ValueError(msg)

        if top_p_f64 == np.float64(1.0):
            if top_k == 1:
                return self._argmax_token
            return lambda lp: self._top_k_tokens(lp, top_k)

        if top_k > 1:
            return lambda lp: self._top_k_p_tokens(lp, top_k, top_p_f64)

        if top_p_f64 >= self._TOP_P_HYBRID_THRESHOLD:
            return lambda lp: self._top_p_tokens_high(lp, top_p_f64)

        return lambda lp: self._top_p_tokens_low(lp, top_p_f64)

    @staticmethod
    def _argmax_token(logprobs: npt.NDArray[np.float64]) -> CandidateTokens:
        """Obtain single highest logprob token as candidate."""
        token_id = np.argmax(logprobs)
        return CandidateTokens(
            candidate_ids=np.array([token_id], dtype=np.int32),
            candidate_logprobs=logprobs[token_id : token_id + 1],
        )

    @staticmethod
    def _top_k_tokens(logprobs: npt.NDArray[np.float64], top_k: int) -> CandidateTokens:
        """Obtain top-k highest logprob tokens as candidates."""
        if top_k < 1:
            msg = 'top_k must be greater than 0'
            raise ValueError(msg)
        if len(logprobs) == 0:
            return CandidateTokens(
                candidate_ids=np.empty(0, dtype=np.int32),
                candidate_logprobs=np.empty(0, dtype=np.float64),
            )

        n = min(top_k, len(logprobs))
        if n == 1:
            return CandidateGeneratorFactory._argmax_token(logprobs)

        cutoff = len(logprobs) - n
        top_ids = np.argpartition(logprobs, cutoff)[-n:]
        top_lp = logprobs[top_ids]
        order = np.argsort(top_lp)[::-1]
        top_ids = top_ids[order]
        return CandidateTokens(
            candidate_ids=top_ids.astype(np.int32, copy=False),
            candidate_logprobs=logprobs[top_ids],
        )

    @staticmethod
    def _top_p_tokens_high(logprobs: npt.NDArray[np.float64], top_p: np.float64) -> CandidateTokens:
        """Top-p selection optimized for high ``top_p`` via a single full sort."""
        if len(logprobs) == 0:
            return CandidateTokens(
                candidate_ids=np.empty(0, dtype=np.int32),
                candidate_logprobs=np.empty(0, dtype=np.float64),
            )

        sorted_ids = np.argsort(logprobs)[::-1]
        sorted_lp = logprobs[sorted_ids]
        cumulative = np.cumsum(np.exp(sorted_lp))
        keep_n = min(
            int(np.searchsorted(cumulative, top_p, side='right')) + 1,
            len(sorted_ids),
        )
        keep_ids = sorted_ids[:keep_n]
        return CandidateTokens(
            candidate_ids=keep_ids.astype(np.int32, copy=False),
            candidate_logprobs=logprobs[keep_ids],
        )

    @staticmethod
    def _top_p_tokens_low(logprobs: npt.NDArray[np.float64], top_p: np.float64) -> CandidateTokens:
        """Top-p selection optimized for low ``top_p`` via adaptive top-k growth."""
        if len(logprobs) == 0:
            return CandidateTokens(
                candidate_ids=np.empty(0, dtype=np.int32),
                candidate_logprobs=np.empty(0, dtype=np.float64),
            )

        vocab_size = len(logprobs)
        k = min(64, vocab_size)
        while True:
            cutoff = vocab_size - k
            top_ids = np.argpartition(logprobs, cutoff)[-k:]
            top_lp = logprobs[top_ids]
            order = np.argsort(top_lp)[::-1]
            top_ids = top_ids[order]
            top_lp = top_lp[order]

            cumulative = np.cumsum(np.exp(top_lp))
            if cumulative[-1] > top_p or k == vocab_size:
                keep_n = min(int(np.searchsorted(cumulative, top_p, side='right')) + 1, k)
                keep_ids = top_ids[:keep_n]
                return CandidateTokens(
                    candidate_ids=keep_ids.astype(np.int32, copy=False),
                    candidate_logprobs=logprobs[keep_ids],
                )

            k = min(k * 2, vocab_size)

    @staticmethod
    def _top_k_p_tokens(logprobs: npt.NDArray[np.float64], top_k: int, top_p: np.float64) -> CandidateTokens:
        """Obtain top-k candidates and truncate to the smallest set exceeding ``top_p``."""
        if top_k < 1:
            msg = 'top_k must be greater than 0'
            raise ValueError(msg)
        if not np.float64(0.0) < top_p <= np.float64(1.0):
            msg = f'top_p must be in (0, 1], got {top_p}'
            raise ValueError(msg)
        if len(logprobs) == 0:
            return CandidateTokens(
                candidate_ids=np.empty(0, dtype=np.int32),
                candidate_logprobs=np.empty(0, dtype=np.float64),
            )

        n = min(top_k, len(logprobs))
        if n == 1:
            return CandidateGeneratorFactory._argmax_token(logprobs)

        cutoff = len(logprobs) - n
        top_ids = np.argpartition(logprobs, cutoff)[-n:]
        top_lp = logprobs[top_ids]
        order = np.argsort(top_lp)[::-1]
        top_ids = top_ids[order]
        top_lp = top_lp[order]

        cumulative = np.cumsum(np.exp(top_lp))
        keep_n = min(int(np.searchsorted(cumulative, top_p, side='right')) + 1, len(top_ids))
        keep_ids = top_ids[:keep_n]
        return CandidateTokens(
            candidate_ids=keep_ids.astype(np.int32, copy=False),
            candidate_logprobs=logprobs[keep_ids],
        )


## -- debug helpers -- ##

def candidates_valid(c: CandidateTokens) -> bool:
    """Verify lengths of candidate dataclass members."""
    return len(c.candidate_ids) == len(c.candidate_logprobs)
