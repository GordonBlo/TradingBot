"""V3.2.2 adapter over the established multi-regime orchestration."""

from __future__ import annotations

from src.backtest.models import BacktestConfig
from src.hypotheses.mechanisms import (
    MechanismHypothesisId,
    MechanismSuiteConfig,
    build_mechanism_candidate,
)
from src.hypotheses.models import HypothesisSuiteConfig
from src.research.multiregime.models import MultiRegimeConfig
from src.research.multiregime.runner import MultiRegimeResearchRunner
from src.research.multiregime.stability import classify_mechanism_candidate
from src.strategy.models import TrendMomentumConfig


class MechanismResearchRunner(MultiRegimeResearchRunner):
    """Run H0/H1/H5/H6/H7 with the unchanged window and diagnostics engine."""

    def __init__(
        self,
        *,
        multiregime_config: MultiRegimeConfig,
        v32_config: HypothesisSuiteConfig,
        mechanism_config: MechanismSuiteConfig,
        strategy_config: TrendMomentumConfig,
        backtest_config: BacktestConfig,
        minimum_trades_warning: int,
    ) -> None:
        def factory(hypothesis_id: object):
            return build_mechanism_candidate(
                MechanismHypothesisId(getattr(hypothesis_id, "value")),
                strategy_config,
                mechanism_config,
                v32_config,
                backtest_config,
            )

        super().__init__(
            multiregime_config=multiregime_config,
            hypothesis_config=mechanism_config,
            strategy_config=strategy_config,
            backtest_config=backtest_config,
            minimum_trades_warning=minimum_trades_warning,
            candidate_ids=tuple(MechanismHypothesisId),
            candidate_factory=factory,
            classification_resolver=classify_mechanism_candidate,
            eligible_classification_value="NEXT_STAGE_ELIGIBLE",
            trade_count_ratio_warning=mechanism_config.trade_count_ratio_warning,
        )
