"""Tests for adjust_probs.py."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from problm_solver.adjust_probs import (
    AdjustFn,
    BranchSampler,
    GenerationContext,
    MetropolisSampler,
    SampleLowTemp,
    SamplePowerDist,
    adjust_identity,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def basic_context() -> GenerationContext:
    """Return a minimal GenerationContext for testing simple adjust functions."""
    return GenerationContext(
        token_probs={' hello': -0.5, ' world': -1.2},
        prev_probs=[],
        context_tokens=[1, 2, 3],
        query_next=MagicMock(return_value={' a': -0.5, ' b': -1.0}),
        query_branch=MagicMock(return_value=-1.5),
        tokenize_token=MagicMock(return_value=[99]),
    )


@pytest.fixture()
def context_with_prev(basic_context: GenerationContext) -> GenerationContext:
    """Return a new GenerationContext identical to basic_context but with non-empty prev_probs."""
    from dataclasses import replace
    return replace(basic_context, prev_probs=[0.9, 0.3])


@pytest.fixture()
def power_dist_context() -> GenerationContext:
    """Return a GenerationContext suitable for SamplePowerDist tests.

    Two candidate tokens, query_branch always returns a fixed log-prob,
    tokenize_token always returns [99].
    """
    return GenerationContext(
        token_probs={' hello': -0.5, ' world': -1.2},
        prev_probs=[],
        context_tokens=[1, 2, 3],
        query_next=MagicMock(return_value={' a': -0.5, ' b': -1.0}),
        query_branch=MagicMock(return_value=-1.5),
        tokenize_token=MagicMock(return_value=[99]),
    )


# ---------------------------------------------------------------------------
# TestGenerationContext
# ---------------------------------------------------------------------------

class TestGenerationContext:
    """Tests for GenerationContext dataclass."""

    def test_stores_token_probs(self, basic_context: GenerationContext) -> None:
        """token_probs is stored correctly."""
        assert basic_context.token_probs == {' hello': -0.5, ' world': -1.2}

    def test_stores_prev_probs(self, basic_context: GenerationContext) -> None:
        """prev_probs is stored correctly."""
        assert basic_context.prev_probs == []

    def test_stores_context_tokens(self, basic_context: GenerationContext) -> None:
        """context_tokens is stored correctly."""
        assert basic_context.context_tokens == [1, 2, 3]

    def test_query_next_is_callable(self, basic_context: GenerationContext) -> None:
        """query_next field is callable."""
        assert callable(basic_context.query_next)

    def test_query_branch_is_callable(self, basic_context: GenerationContext) -> None:
        """query_branch field is callable."""
        assert callable(basic_context.query_branch)

    def test_tokenize_token_is_callable(self, basic_context: GenerationContext) -> None:
        """tokenize_token field is callable."""
        assert callable(basic_context.tokenize_token)


# ---------------------------------------------------------------------------
# TestAdjustIdentity
# ---------------------------------------------------------------------------

class TestAdjustIdentity:
    """Tests for adjust_identity."""

    def test_returns_dict(self, basic_context: GenerationContext) -> None:
        """adjust_identity() returns a dict."""
        assert isinstance(adjust_identity(basic_context), dict)

    def test_returns_token_probs_unchanged(self, basic_context: GenerationContext) -> None:
        """adjust_identity() returns context.token_probs unmodified."""
        assert adjust_identity(basic_context) == basic_context.token_probs

    def test_ignores_prev_probs(
        self,
        basic_context: GenerationContext,
        context_with_prev: GenerationContext,
    ) -> None:
        """adjust_identity() produces the same result regardless of prev_probs."""
        assert adjust_identity(basic_context) == adjust_identity(context_with_prev)

    def test_satisfies_adjust_fn_signature(self, basic_context: GenerationContext) -> None:
        """adjust_identity is a valid AdjustFn (callable with correct signature)."""
        assert callable(adjust_identity)
        assert isinstance(adjust_identity(basic_context), dict)


# ---------------------------------------------------------------------------
# TestSampleLowTempInit
# ---------------------------------------------------------------------------

class TestSampleLowTempInit:
    """Tests for SampleLowTemp.__init__."""

    def test_stores_alpha(self) -> None:
        """alpha is stored as an instance attribute."""
        assert SampleLowTemp(alpha=2).alpha == 2

    def test_different_alpha_values_stored(self) -> None:
        """Different alpha values are stored independently."""
        adj1, adj2 = SampleLowTemp(alpha=1), SampleLowTemp(alpha=3)
        assert adj1.alpha == 1
        assert adj2.alpha == 3

    def test_is_callable(self) -> None:
        """SampleLowTemp instances are callable."""
        assert callable(SampleLowTemp(alpha=2))

    def test_satisfies_adjust_fn_signature(self, basic_context: GenerationContext) -> None:
        """SampleLowTemp instances satisfy the AdjustFn interface."""
        assert isinstance(SampleLowTemp(alpha=2)(basic_context), dict)


# ---------------------------------------------------------------------------
# TestSampleLowTempCall
# ---------------------------------------------------------------------------

class TestSampleLowTempCall:
    """Tests for SampleLowTemp.__call__."""

    @pytest.fixture()
    def adj(self) -> SampleLowTemp:
        """Return a SampleLowTemp instance with alpha=2."""
        return SampleLowTemp(alpha=2)

    def test_returns_dict(self, adj: SampleLowTemp, basic_context: GenerationContext) -> None:
        """__call__() returns a dict."""
        assert isinstance(adj(basic_context), dict)

    def test_output_keys_match_input_keys(
        self, adj: SampleLowTemp, basic_context: GenerationContext
    ) -> None:
        """The returned dict has the same keys as context.token_probs."""
        assert set(adj(basic_context).keys()) == set(basic_context.token_probs.keys())

    def test_output_values_are_floats(
        self, adj: SampleLowTemp, basic_context: GenerationContext
    ) -> None:
        """All values in the returned dict are floats."""
        assert all(isinstance(v, float) for v in adj(basic_context).values())

    def test_empty_prev_probs_gives_finite_output(
        self, adj: SampleLowTemp, basic_context: GenerationContext
    ) -> None:
        """With no previous tokens, output log-probs are finite for non-zero probs."""
        assert all(np.isfinite(v) for v in adj(basic_context).values())

    def test_with_prev_probs_changes_output(
        self,
        adj: SampleLowTemp,
        basic_context: GenerationContext,
        context_with_prev: GenerationContext,
    ) -> None:
        """Providing non-empty prev_probs produces a different result."""
        assert adj(basic_context) != adj(context_with_prev)

    def test_relative_order_preserved_with_empty_prev_probs(
        self, adj: SampleLowTemp
    ) -> None:
        """With no previous tokens, the most likely token remains most likely."""
        ctx = GenerationContext(
            token_probs={' likely': -0.1, ' unlikely': -5.0},
            prev_probs=[],
            context_tokens=[],
            query_next=MagicMock(),
            query_branch=MagicMock(),
            tokenize_token=MagicMock(),
        )
        result = adj(ctx)
        assert result[' likely'] > result[' unlikely']

    def test_alpha_one_preserves_relative_order(self) -> None:
        """With alpha=1 and no prev_probs, relative token ordering is unchanged."""
        ctx = GenerationContext(
            token_probs={' a': -0.2, ' b': -0.8, ' c': -2.0},
            prev_probs=[],
            context_tokens=[],
            query_next=MagicMock(),
            query_branch=MagicMock(),
            tokenize_token=MagicMock(),
        )
        result = SampleLowTemp(alpha=1)(ctx)
        assert result[' a'] > result[' b'] > result[' c']

    def test_different_alpha_produces_different_output(self) -> None:
        """Different alpha values produce different adjusted distributions."""
        ctx = GenerationContext(
            token_probs={' a': -0.2, ' b': -1.0},
            prev_probs=[],
            context_tokens=[],
            query_next=MagicMock(),
            query_branch=MagicMock(),
            tokenize_token=MagicMock(),
        )
        assert SampleLowTemp(alpha=1)(ctx) != SampleLowTemp(alpha=3)(ctx)

    def test_matches_exact_power_scaling_without_history(self) -> None:
        """With empty history, output equals log(exp(lp-lp_max) ** alpha)."""
        ctx = GenerationContext(
            token_probs={' a': -0.2, ' b': -1.2},
            prev_probs=[],
            context_tokens=[],
            query_next=MagicMock(),
            query_branch=MagicMock(),
            tokenize_token=MagicMock(),
        )
        out = SampleLowTemp(alpha=2.0)(ctx)
        assert out == pytest.approx({' a': 0.0, ' b': -2.0})

    def test_matches_exact_power_scaling_with_history_factor(self) -> None:
        """History contributes a constant log(prod(prev_probs ** alpha)) shift."""
        ctx = GenerationContext(
            token_probs={' a': -0.2, ' b': -1.2},
            prev_probs=[0.5],
            context_tokens=[],
            query_next=MagicMock(),
            query_branch=MagicMock(),
            tokenize_token=MagicMock(),
        )
        out = SampleLowTemp(alpha=2.0)(ctx)
        expected_shift = float(np.log(0.5 ** 2))
        assert out == pytest.approx({' a': expected_shift, ' b': expected_shift - 2.0})


# ---------------------------------------------------------------------------
# TestBranchSampler
# ---------------------------------------------------------------------------

class TestBranchSampler:
    """Tests for the BranchSampler ABC."""

    def test_reset_is_no_op_by_default(self) -> None:
        """The default reset() implementation does not raise and returns None."""
        class MinimalSampler(BranchSampler):
            def step(self, proposed_log_prob: float, alpha: float = 1.0, **_) -> float:
                return proposed_log_prob
            def should_continue(self, branch_log_probs) -> bool:
                return False
            def future_logprob(self, alpha, branch_log_probs) -> np.float64:
                return np.float64(0.0)

        MinimalSampler().reset()  # must not raise

    def test_subclass_must_implement_abstract_methods(self) -> None:
        """BranchSampler cannot be instantiated without implementing abstract methods."""
        with pytest.raises(TypeError):
            BranchSampler()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# TestMetropolisSampler
# ---------------------------------------------------------------------------

class TestMetropolisSampler:
    """Tests for MetropolisSampler."""

    @pytest.fixture()
    def sampler(self) -> MetropolisSampler:
        """Return a fresh MetropolisSampler."""
        return MetropolisSampler()

    def test_is_branch_sampler(self, sampler: MetropolisSampler) -> None:
        """MetropolisSampler is a BranchSampler subclass."""
        assert isinstance(sampler, BranchSampler)

    def test_first_step_always_initialises_chain(self, sampler: MetropolisSampler) -> None:
        """First step always accepts and initialises _current_log_prob."""
        accepted = sampler.step(-2.0, alpha=2.0)
        assert accepted == -2.0
        assert sampler._current_log_prob == -2.0

    def test_reset_clears_state(self, sampler: MetropolisSampler) -> None:
        """reset() clears chain state."""
        sampler.step(-2.0, alpha=2.0)
        sampler.reset()
        assert sampler._current_log_prob is None

    def test_step_returns_float(self, sampler: MetropolisSampler) -> None:
        """step() returns a float accepted log-probability."""
        assert isinstance(sampler.step(-1.5, alpha=2.0), float)

    def test_more_likely_proposal_has_high_acceptance(self, sampler: MetropolisSampler) -> None:
        """A much higher log-prob proposal is accepted with alpha > 1."""
        sampler.step(-10.0, alpha=2.0)
        accepted = sampler.step(-1.0, alpha=2.0)
        assert accepted == -1.0

    def test_alpha_one_always_accepts(self, sampler: MetropolisSampler) -> None:
        """With alpha=1 acceptance ratio is 0, so every proposal is accepted."""
        sampler.step(-5.0, alpha=1.0)
        # (alpha-1)*(proposed-current) = 0 regardless of values, so always accept
        for _ in range(20):
            proposal = -10.0
            accepted = sampler.step(proposal, alpha=1.0)
            assert accepted == proposal

    def test_should_continue_returns_true_below_min_branches(self) -> None:
        """should_continue returns True when fewer than equil_branches collected."""
        sampler = MetropolisSampler(equil_branches=5)
        for n in range(1, 5):
            lp = np.zeros(n)
            assert sampler.should_continue(lp) is True

    def test_should_continue_returns_false_at_max_branches(self) -> None:
        """should_continue returns False when max_branches is reached."""
        sampler = MetropolisSampler(equil_branches=1, max_branches=10)
        assert sampler.should_continue(np.zeros(10)) is False

    def test_init_sets_defaults_and_seeded_rng(self) -> None:
        """__init__ stores defaults and seeds Generator deterministically."""
        sampler = MetropolisSampler(rng=123)
        assert sampler._equil_branches == 5
        assert sampler._max_branches == 30
        assert sampler._tolerance == pytest.approx(1e-1)
        expected_first = np.random.default_rng(123).random()
        assert sampler._rng.random() == pytest.approx(expected_first)

    def test_step_rejects_when_uniform_draw_above_acceptance_threshold(self) -> None:
        """Proposal is rejected when log(u) exceeds min(0, log_accept_ratio)."""
        sampler = MetropolisSampler()
        sampler._current_log_prob = -1.0
        sampler._rng = MagicMock(random=MagicMock(return_value=0.9))

        accepted = sampler.step(proposed_log_prob=-2.0, alpha=2.0)

        assert accepted == pytest.approx(-1.0)
        assert sampler._current_log_prob == pytest.approx(-1.0)

    def test_step_accepts_when_uniform_draw_below_acceptance_threshold(self) -> None:
        """Proposal is accepted when log(u) is below min(0, log_accept_ratio)."""
        sampler = MetropolisSampler()
        sampler._current_log_prob = -1.0
        sampler._rng = MagicMock(random=MagicMock(return_value=0.1))

        accepted = sampler.step(proposed_log_prob=-2.0, alpha=2.0)

        assert accepted == pytest.approx(-2.0)
        assert sampler._current_log_prob == pytest.approx(-2.0)

    def test_future_logprob_uses_post_equilibration_slice_exactly(self) -> None:
        """future_logprob() computes log-mean-exp over branch_log_probs[equil_branches:]."""
        sampler = MetropolisSampler(equil_branches=2)
        branch_log_probs = np.array([-10.0, -9.0, -2.0, -1.0], dtype=np.float64)

        result = sampler.future_logprob(alpha=2.0, branch_log_probs=branch_log_probs)

        post_eq = np.array([-2.0, -1.0], dtype=np.float64)
        scaled = 2.0 * post_eq
        max_lp = scaled.max()
        expected = np.log(np.mean(np.exp(scaled - max_lp))) + max_lp
        assert result == pytest.approx(expected)


# ---------------------------------------------------------------------------
# TestSamplePowerDistInit
# ---------------------------------------------------------------------------

class TestSamplePowerDistInit:
    """Tests for SamplePowerDist.__init__."""

    def test_stores_alpha(self) -> None:
        """alpha is stored correctly."""
        s = SamplePowerDist(alpha=2.0, lookahead_depth=4,
                            branch_sampler=MetropolisSampler())
        assert s.alpha == 2.0

    def test_stores_lookahead_depth(self) -> None:
        """lookahead_depth is stored correctly."""
        s = SamplePowerDist(alpha=1.0, lookahead_depth=7,
                            branch_sampler=MetropolisSampler())
        assert s.lookahead_depth == 7

    def test_stores_branch_sampler(self) -> None:
        """branch_sampler is stored correctly."""
        bs = MetropolisSampler()
        s = SamplePowerDist(alpha=1.0, lookahead_depth=2, branch_sampler=bs)
        assert s.branch_sampler is bs

    def test_does_not_store_n_branches(self) -> None:
        """SamplePowerDist has no n_branches attribute."""
        s = SamplePowerDist(alpha=1.0, lookahead_depth=2,
                            branch_sampler=MetropolisSampler())
        assert not hasattr(s, 'n_branches')

    def test_is_callable(self) -> None:
        """SamplePowerDist instances are callable."""
        assert callable(SamplePowerDist(alpha=1.0, lookahead_depth=1,
                                        branch_sampler=MetropolisSampler()))


# ---------------------------------------------------------------------------
# TestSamplePowerDistCall
# ---------------------------------------------------------------------------

class TestSamplePowerDistCall:
    """Tests for SamplePowerDist.__call__."""

    @pytest.fixture()
    def mock_sampler(self) -> MagicMock:
        """Return a mock BranchSampler that accepts proposals and stops after one branch."""
        s = MagicMock(spec=BranchSampler)
        s.step.side_effect = lambda proposed_log_prob, **_: proposed_log_prob
        s.should_continue.return_value = False  # stop after the first branch
        s.future_logprob.return_value = 0.0
        return s

    @pytest.fixture()
    def spd(self, mock_sampler: MagicMock) -> SamplePowerDist:
        """Return a SamplePowerDist(alpha=1, lookahead_depth=2) with mock sampler."""
        return SamplePowerDist(
            alpha=1.0, lookahead_depth=2, branch_sampler=mock_sampler
        )

    def test_returns_dict(
        self, spd: SamplePowerDist, power_dist_context: GenerationContext
    ) -> None:
        """__call__() returns a dict."""
        assert isinstance(spd(power_dist_context), dict)

    def test_output_keys_match_input_keys(
        self, spd: SamplePowerDist, power_dist_context: GenerationContext
    ) -> None:
        """The returned dict has the same keys as context.token_probs."""
        result = spd(power_dist_context)
        assert set(result.keys()) == set(power_dist_context.token_probs.keys())

    def test_output_values_are_floats(
        self, spd: SamplePowerDist, power_dist_context: GenerationContext
    ) -> None:
        """All values in the returned dict are floats."""
        result = spd(power_dist_context)
        assert all(isinstance(v, float) for v in result.values())

    def test_query_branch_called_for_each_candidate_token(
        self, spd: SamplePowerDist, power_dist_context: GenerationContext
    ) -> None:
        """query_branch is called once per branch per candidate token."""
        spd(power_dist_context)
        # 2 candidates * 1 branch each = 2 calls
        assert power_dist_context.query_branch.call_count == 2

    def test_branch_sampler_reset_called_per_branch(
        self, spd: SamplePowerDist, mock_sampler: MagicMock,
        power_dist_context: GenerationContext,
    ) -> None:
        """reset() is called once at the start of each branch."""
        spd(power_dist_context)
        # 2 candidates * 1 branch = 2 resets
        assert mock_sampler.reset.call_count == 2

    def test_branch_sampler_step_called_per_branch(
        self, spd: SamplePowerDist, mock_sampler: MagicMock,
        power_dist_context: GenerationContext,
    ) -> None:
        """step() is called once per completed branch proposal."""
        spd(power_dist_context)
        # 2 candidates * 1 branch = 2 MH steps
        assert mock_sampler.step.call_count == 2

    def test_should_continue_checked_after_each_branch(
        self, spd: SamplePowerDist, mock_sampler: MagicMock,
        power_dist_context: GenerationContext,
    ) -> None:
        """should_continue is called once after each completed branch."""
        spd(power_dist_context)
        # called once per candidate (one branch each, then stops)
        assert mock_sampler.should_continue.call_count == 2

    def test_continues_until_should_continue_false(
        self, mock_sampler: MagicMock, power_dist_context: GenerationContext
    ) -> None:
        """Sampling runs multiple branches until should_continue returns False."""
        # Return True for first 2 calls per candidate, then False
        mock_sampler.should_continue.side_effect = [True, True, False, True, True, False]
        spd = SamplePowerDist(alpha=1.0, lookahead_depth=1, branch_sampler=mock_sampler)
        spd(power_dist_context)
        # 2 candidates * 3 branches each = 6 should_continue calls
        assert mock_sampler.should_continue.call_count == 6

    def test_query_branch_result_passed_to_step(
        self, mock_sampler: MagicMock
    ) -> None:
        """The float returned by query_branch is passed directly to branch_sampler.step."""
        mock_sampler.step.side_effect = lambda proposed_log_prob, **_: proposed_log_prob
        spd = SamplePowerDist(alpha=1.0, lookahead_depth=5, branch_sampler=mock_sampler)
        ctx = GenerationContext(
            token_probs={' hello': -0.5},
            prev_probs=[],
            context_tokens=[1, 2],
            query_next=MagicMock(),
            query_branch=MagicMock(return_value=-2.5),
            tokenize_token=MagicMock(return_value=[99]),
        )
        spd(ctx)
        mock_sampler.step.assert_called_once_with(
            proposed_log_prob=-2.5, alpha=1.0
        )

    def test_query_branch_called_with_correct_depth(
        self, mock_sampler: MagicMock
    ) -> None:
        """query_branch is called with lookahead_depth as its depth argument."""
        spd = SamplePowerDist(alpha=1.0, lookahead_depth=4, branch_sampler=mock_sampler)
        ctx = GenerationContext(
            token_probs={' hello': -0.5},
            prev_probs=[],
            context_tokens=[1],
            query_next=MagicMock(),
            query_branch=MagicMock(return_value=-1.0),
            tokenize_token=MagicMock(return_value=[99]),
        )
        spd(ctx)
        args, _ = ctx.query_branch.call_args
        assert args[1] == 4

    def test_branch_loop_has_runaway_guard(self) -> None:
        """Guard fails fast if branch sampling runs beyond expected iterations."""
        mock_sampler = MagicMock(spec=BranchSampler)
        mock_sampler.step.side_effect = lambda proposed_log_prob, **_: proposed_log_prob
        # For each candidate token: run exactly 2 branches, then stop.
        mock_sampler.should_continue.side_effect = [True, False, True, False]
        mock_sampler.future_logprob.return_value = 0.0

        query_calls = 0

        def guarded_query_branch(*_: object) -> float:
            nonlocal query_calls
            query_calls += 1
            if query_calls > 4:
                msg = 'runaway branch loop detected'
                raise AssertionError(msg)
            return -1.0

        ctx = GenerationContext(
            token_probs={' hello': -0.5, ' world': -1.2},
            prev_probs=[],
            context_tokens=[1],
            query_next=MagicMock(),
            query_branch=MagicMock(side_effect=guarded_query_branch),
            tokenize_token=MagicMock(return_value=[99]),
        )

        spd = SamplePowerDist(alpha=1.0, lookahead_depth=2, branch_sampler=mock_sampler)
        spd(ctx)

        assert query_calls == 4


# ---------------------------------------------------------------------------
# TestAdjustFnTypeAlias
# ---------------------------------------------------------------------------

class TestAdjustFnTypeAlias:
    """Tests for the AdjustFn type alias."""

    def test_adjust_identity_is_valid_adjust_fn(
        self, basic_context: GenerationContext
    ) -> None:
        """adjust_identity satisfies the AdjustFn calling convention."""
        fn: AdjustFn = adjust_identity
        assert fn(basic_context) == basic_context.token_probs

    def test_sample_low_temp_is_valid_adjust_fn(
        self, basic_context: GenerationContext
    ) -> None:
        """A SampleLowTemp instance satisfies the AdjustFn calling convention."""
        fn: AdjustFn = SampleLowTemp(alpha=2)
        assert isinstance(fn(basic_context), dict)

    def test_sample_power_dist_is_valid_adjust_fn(
        self, basic_context: GenerationContext
    ) -> None:
        """A SamplePowerDist instance satisfies the AdjustFn calling convention."""
        fn: AdjustFn = SamplePowerDist(
            alpha=1.0, lookahead_depth=1,
            branch_sampler=MetropolisSampler(),
        )
        assert isinstance(fn(basic_context), dict)
