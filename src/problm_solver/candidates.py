"""Internal representation of token & log-probability data."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt


@dataclass
class CandidateTokens:
    """Dataclass for token IDs and assigned log-probabilities."""

    candidate_ids: npt.NDArray[np.int32]
    candidate_logprobs: npt.NDArray[np.float64]


class CandidateGeneratorFactory:
    """Gets candidates from a vocabulary.

    Every public generator accepts the raw full-vocabulary logits array,
    normalises it to log-probabilities and returns a :class:`CandidateTokens`
    subset.
    """

    _TOP_P_HYBRID_THRESHOLD = np.float64(0.5)

    def get_candidate_generator(
        self,
        top_k: int | None = None,
        top_p: float | None = None,
        alpha: float = 1.0,
    ) -> Callable[[npt.NDArray[np.float32]], CandidateTokens]:
        """Get a candidate generator function.

        A ``None`` ``top_k`` or ``top_p`` (and also a ``top_p`` of 1.0)
        disables that truncation stage and falls back to no op.

        :param top_k: Number of top candidate tokens to retrieve. ``None``
            disables top-k truncation.
        :param top_p: Threshold total probability of retrieved tokens. Both
            ``None`` and 1.0 disable top-p truncation.
        :param alpha: Inverse temperature (``temperature = 1 / alpha``). Must be
            strictly positive and finite.
        :returns: A callable mapping raw logits to a ``CandidateTokens`` set.
        :raises ValueError: If ``alpha`` is non-positive or non-finite,
            ``top_k < 1``, or ``top_p`` outside (0, 1].
        """
        alpha_f64 = np.float64(alpha)
        if not np.isfinite(alpha_f64) or alpha_f64 <= np.float64(0.0):
            msg = (
                'Infinite or negative temperature is impermissible, expected alpha > 0, '
                f'got {alpha:.4f}'
            )
            raise ValueError(msg)

        if top_k is not None and top_k < 1:
            msg = 'top_k must be greater than 0'
            raise ValueError(msg)

        top_p_f64: np.float64 | None = None
        if top_p is not None:
            top_p_f64 = np.float64(top_p)
            if not np.float64(0.0) < top_p_f64 <= np.float64(1.0):
                msg = f'top_p must be in (0, 1], got {top_p}'
                raise ValueError(msg)

        is_identity = alpha_f64 == np.float64(1.0)
        mode = self._resolve_truncation_route(top_k, top_p_f64)
        return self._build_generator(
            mode,
            top_k,
            top_p_f64,
            alpha_f64,
            is_identity=is_identity,
        )

    @staticmethod
    def _resolve_truncation_route(
        top_k: int | None,
        top_p_f64: np.float64 | None,
    ) -> str:
        """Resolve which truncation generator to use.

        :param top_k: Requested top-k cap (``None`` disables the stage).
        :param top_p_f64: Requested nucleus threshold (``None`` or 1.0 disables).
        :returns: One of ``'no_op'``, ``'argmax'``, ``'top_k'``,
            ``'top_p_high'``, ``'top_p_low'``, ``'top_k_p'``.
        """
        if top_p_f64 is None or top_p_f64 == np.float64(1.0):
            if top_k is None:
                return 'no_op'
            return 'argmax' if top_k == 1 else 'top_k'
        if top_k is None:
            if top_p_f64 >= CandidateGeneratorFactory._TOP_P_HYBRID_THRESHOLD:
                return 'top_p_high'
            return 'top_p_low'
        return 'argmax' if top_k == 1 else 'top_k_p'

    def _build_generator(
        self,
        mode: str,
        top_k: int | None,
        top_p_f64: np.float64 | None,
        alpha_f64: np.float64,
        *,
        is_identity: bool,
    ) -> Callable[[npt.NDArray[np.float32]], CandidateTokens]:
        """Assemble the generator callable for a resolved route.

        :param mode: Route name from :meth:`_resolve_truncation_route`.
        :param top_k: Requested top-k cap (used by the top-k routes).
        :param top_p_f64: Requested nucleus threshold (used by the top-p routes).
        :param alpha_f64: Inverse temperature, appended to the scaled variants.
        :param is_identity: Whether ``alpha`` is 1.0; selects the temperature-
            free variant, which skips the full-vocabulary scaling operation.
        :returns: A callable mapping raw logits to a ``CandidateTokens`` set.
        """
        routes: dict[str, tuple[
            Callable[..., CandidateTokens],
            Callable[..., CandidateTokens],
            tuple[Any, ...],
        ]] = {
            'no_op': (self._no_op, self._no_op_temp, ()),
            'argmax': (self._argmax_token, self._argmax_token_temp, ()),
            'top_k': (self._top_k_tokens, self._top_k_tokens_temp, (top_k,)),
            'top_p_high': (self._top_p_high_tokens, self._top_p_high_tokens_temp, (top_p_f64,)),
            'top_p_low': (self._top_p_low_tokens, self._top_p_low_tokens_temp, (top_p_f64,)),
            'top_k_p': (self._top_k_p_tokens, self._top_k_p_tokens_temp, (top_k, top_p_f64)),
        }
        unscaled, scaled, params = routes[mode]
        if is_identity:
            return lambda lg: unscaled(lg, *params)
        return lambda lg: scaled(lg, *params, alpha_f64)

    # -- No truncation -- #

    @staticmethod
    def _no_op(logits: npt.NDArray[np.float32]) -> CandidateTokens:
        """Candidate set spanning the full vocabulary (no truncation, alpha = 1)."""
        return CandidateGeneratorFactory._no_op_from_logprobs(log_softmax(logits))

    @staticmethod
    def _no_op_temp(logits: npt.NDArray[np.float32], alpha: np.float64) -> CandidateTokens:
        """Candidate set spanning the full vocabulary, temperature-scaled by alpha."""
        return CandidateGeneratorFactory._no_op_from_logprobs(log_softmax(logits * alpha))

    @staticmethod
    def _no_op_from_logprobs(logprobs: npt.NDArray[np.float64]) -> CandidateTokens:
        """Build the full-vocabulary candidate set from log-probabilities."""
        return CandidateTokens(
            candidate_ids=np.arange(len(logprobs), dtype=np.int32),
            candidate_logprobs=logprobs,
        )

    # -- Argmax -- #

    @staticmethod
    def _argmax_token(logits: npt.NDArray[np.float32]) -> CandidateTokens:
        """Obtain single highest logprob token as candidate (alpha = 1)."""
        return CandidateGeneratorFactory._argmax_from_logprobs(log_softmax(logits))

    @staticmethod
    def _argmax_token_temp(logits: npt.NDArray[np.float32], alpha: np.float64) -> CandidateTokens:
        """Obtain single highest logprob token, temperature-scaled by alpha.

        The selected token is identical to the unscaled argmax for any
        positive alpha (logit scaling is monotone); only the reported
        log-probability changes.
        """
        return CandidateGeneratorFactory._argmax_from_logprobs(log_softmax(logits * alpha))

    @staticmethod
    def _argmax_from_logprobs(logprobs: npt.NDArray[np.float64]) -> CandidateTokens:
        """Build the single-candidate set from log-probabilities."""
        token_id = np.argmax(logprobs)
        return CandidateTokens(
            candidate_ids=np.array([token_id], dtype=np.int32),
            candidate_logprobs=logprobs[token_id : token_id + 1],
        )

    # -- Top-k -- #

    @staticmethod
    def _top_k_tokens(logits: npt.NDArray[np.float32], top_k: int) -> CandidateTokens:
        """Obtain top-k highest logprob tokens as candidates (alpha = 1)."""
        return CandidateGeneratorFactory._top_k_from_logprobs(log_softmax(logits), top_k)

    @staticmethod
    def _top_k_tokens_temp(
        logits: npt.NDArray[np.float32],
        top_k: int,
        alpha: np.float64,
    ) -> CandidateTokens:
        """Obtain top-k highest logprob tokens, temperature-scaled by alpha."""
        return CandidateGeneratorFactory._top_k_from_logprobs(
            log_softmax(logits * alpha), top_k
        )

    @staticmethod
    def _top_k_from_logprobs(logprobs: npt.NDArray[np.float64], top_k: int) -> CandidateTokens:
        """Keep all tokens whose log-probability reaches the ``top_k``-th largest value.

        :param logprobs: Full-vocabulary log-probabilities.
        :param top_k: Requested number of top tokens.
        :returns: The candidate set, ordered descending by log-probability
            (ties broken by ascending token id).
        :raises ValueError: If ``top_k < 1``.
        """
        if top_k < 1:
            msg = 'top_k must be greater than 0'
            raise ValueError(msg)
        if len(logprobs) == 0:
            return CandidateTokens(
                candidate_ids=np.empty(0, dtype=np.int32),
                candidate_logprobs=np.empty(0, dtype=np.float64),
            )

        n = min(top_k, len(logprobs))
        kth_largest = np.partition(logprobs, len(logprobs) - n)[len(logprobs) - n]
        keep_ids = np.flatnonzero(logprobs >= kth_largest)
        order = np.argsort(-logprobs[keep_ids], kind='stable')
        keep_ids = keep_ids[order]
        return CandidateTokens(
            candidate_ids=keep_ids.astype(np.int32, copy=False),
            candidate_logprobs=logprobs[keep_ids],
        )

    # -- Top-p -- #

    @staticmethod
    def _top_p_high_tokens(
        logits: npt.NDArray[np.float32],
        top_p: np.float64,
    ) -> CandidateTokens:
        """Top-p selection optimised for high ``top_p`` via a single full sort (alpha = 1)."""
        return CandidateGeneratorFactory._top_p_high_from_logprobs(log_softmax(logits), top_p)

    @staticmethod
    def _top_p_high_tokens_temp(
        logits: npt.NDArray[np.float32],
        top_p: np.float64,
        alpha: np.float64,
    ) -> CandidateTokens:
        """High-``top_p`` selection, temperature-scaled by alpha."""
        return CandidateGeneratorFactory._top_p_high_from_logprobs(
            log_softmax(logits * alpha), top_p
        )

    @staticmethod
    def _top_p_high_from_logprobs(
        logprobs: npt.NDArray[np.float64],
        top_p: np.float64,
    ) -> CandidateTokens:
        """Keep the smallest full-vocabulary prefix whose cumulative probability reaches ``top_p``."""
        if len(logprobs) == 0:
            return CandidateTokens(
                candidate_ids=np.empty(0, dtype=np.int32),
                candidate_logprobs=np.empty(0, dtype=np.float64),
            )

        sorted_ids = np.argsort(logprobs)[::-1]
        sorted_lp = logprobs[sorted_ids]
        cumulative = np.cumsum(np.exp(sorted_lp))
        keep_n = min(
            int(np.searchsorted(cumulative, top_p, side='left')) + 1,
            len(sorted_ids),
        )
        keep_ids = sorted_ids[:keep_n]
        return CandidateTokens(
            candidate_ids=keep_ids.astype(np.int32, copy=False),
            candidate_logprobs=logprobs[keep_ids],
        )

    @staticmethod
    def _top_p_low_tokens(logits: npt.NDArray[np.float32], top_p: np.float64) -> CandidateTokens:
        """Top-p selection optimised for low ``top_p`` via adaptive top-k growth (alpha = 1)."""
        return CandidateGeneratorFactory._top_p_low_from_logprobs(log_softmax(logits), top_p)

    @staticmethod
    def _top_p_low_tokens_temp(
        logits: npt.NDArray[np.float32],
        top_p: np.float64,
        alpha: np.float64,
    ) -> CandidateTokens:
        """Low-``top_p`` selection, temperature-scaled by alpha."""
        return CandidateGeneratorFactory._top_p_low_from_logprobs(
            log_softmax(logits * alpha), top_p
        )

    @staticmethod
    def _top_p_low_from_logprobs(
        logprobs: npt.NDArray[np.float64],
        top_p: np.float64,
    ) -> CandidateTokens:
        """As :meth:`_top_p_high_from_logprobs`, via adaptive top-k growth for small ``top_p``."""
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
            if cumulative[-1] >= top_p or k == vocab_size:
                keep_n = min(int(np.searchsorted(cumulative, top_p, side='left')) + 1, k)
                keep_ids = top_ids[:keep_n]
                return CandidateTokens(
                    candidate_ids=keep_ids.astype(np.int32, copy=False),
                    candidate_logprobs=logprobs[keep_ids],
                )

            k = min(k * 2, vocab_size)

    # -- Combined top-k and top-p -- #

    @staticmethod
    def _top_k_p_tokens(
        logits: npt.NDArray[np.float32],
        top_k: int,
        top_p: np.float64,
    ) -> CandidateTokens:
        """Obtain top-k candidates truncated to the smallest set exceeding ``top_p`` (alpha = 1)."""
        return CandidateGeneratorFactory._top_k_p_from_logprobs(
            log_softmax(logits), top_k, top_p
        )

    @staticmethod
    def _top_k_p_tokens_temp(
        logits: npt.NDArray[np.float32],
        top_k: int,
        top_p: np.float64,
        alpha: np.float64,
    ) -> CandidateTokens:
        """Obtain combined top-k/top-p candidates, temperature-scaled by alpha."""
        return CandidateGeneratorFactory._top_k_p_from_logprobs(
            log_softmax(logits * alpha), top_k, top_p
        )

    @staticmethod
    def _top_k_p_from_logprobs(
        logprobs: npt.NDArray[np.float64],
        top_k: int,
        top_p: np.float64,
    ) -> CandidateTokens:
        """Truncate the renormalised top-k kept set to the smallest prefix reaching ``top_p``.

        :param logprobs: Full-vocabulary log-probabilities.
        :param top_k: Requested number of top tokens.
        :param top_p: Nucleus threshold.
        :returns: The candidate set, ordered descending by log-probability
            (ties broken by ascending token id).
        :raises ValueError: If ``top_k < 1`` or ``top_p`` outside (0, 1].
        """
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
        kth_largest = np.partition(logprobs, len(logprobs) - n)[len(logprobs) - n]
        top_ids = np.flatnonzero(logprobs >= kth_largest)
        order = np.argsort(-logprobs[top_ids], kind='stable')
        top_ids = top_ids[order]

        # Renormalise the top-k mass to 1 before the nucleus threshold, as HF's
        # TopPLogitsWarper softmaxes over the top-k-filtered vector.
        renorm_lp = logprobs[top_ids]
        renorm_lp = renorm_lp - np.log(np.exp(renorm_lp).sum())
        cumulative = np.cumsum(np.exp(renorm_lp))
        keep_n = min(int(np.searchsorted(cumulative, top_p, side='left')) + 1, len(top_ids))
        keep_ids = top_ids[:keep_n]
        return CandidateTokens(
            candidate_ids=keep_ids.astype(np.int32, copy=False),
            candidate_logprobs=logprobs[keep_ids],
        )


## -- Numerical Utilities -- ##

def log_softmax(logits: npt.NDArray[np.float32]) -> npt.NDArray[np.float64]:
    """Apply numerically stable log-softmax to a 1-D logits vector.

    Subtracts the maximum logit before exponentiation to prevent float
    overflow (common with raw LLM logits which can exceed ±300), then
    uses the log-sum-exp identity:

    .. code-block:: text

        log_softmax(x_i) = (x_i − max x) − log Σ_j exp(x_j − max x)

    The result satisfies ``exp(result).sum() ≈ 1`` and all values are ≤ 0.

    :param logits: 1-D array of raw model logits for the full vocabulary.
    :returns: 1-D float64 array of log-probabilities.
    """ # noqa: RUF002
    x = logits.astype(np.float64)
    shifted = x - x.max()
    return shifted - np.log(np.exp(shifted).sum())


## -- Debug helpers -- ##

def candidates_valid(c: CandidateTokens) -> bool:
    """Verify lengths of candidate dataclass members."""
    return len(c.candidate_ids) == len(c.candidate_logprobs)
