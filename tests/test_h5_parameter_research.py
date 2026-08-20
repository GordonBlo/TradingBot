from decimal import Decimal

import pytest

from src.hypotheses.h5_parameter_research import (
    H5ParameterCandidateId,
    build_h5_parameter_candidate,
    get_h5_definition,
    h5_parameter_registry,
    validate_h5_parameter_registry,
)
from src.hypotheses.mechanisms import VolatilityBandStrategy
from src.strategy.models import TrendMomentumConfig


def test_registry_is_exactly_preregistered() -> None:
    definitions = h5_parameter_registry()

    assert tuple(
        item.candidate_id
        for item in definitions
    ) == (
        H5ParameterCandidateId.H5_Q25,
        H5ParameterCandidateId.H5_Q35,
        H5ParameterCandidateId.H5_Q45,
        H5ParameterCandidateId.H5_Q50,
    )

    assert tuple(
        item.minimum_percentile
        for item in definitions
    ) == (
        Decimal("25"),
        Decimal("35"),
        Decimal("45"),
        Decimal("50"),
    )


def test_upper_percentile_and_lookback_are_frozen() -> None:
    for definition in h5_parameter_registry():
        assert definition.lookback == 100
        assert definition.maximum_percentile == Decimal("75")


def test_q25_is_the_only_frozen_reference() -> None:
    references = [
        item
        for item in h5_parameter_registry()
        if item.is_frozen_reference
    ]

    assert len(references) == 1
    assert (
        references[0].candidate_id
        is H5ParameterCandidateId.H5_Q25
    )


def test_definition_lookup() -> None:
    definition = get_h5_definition(
        H5ParameterCandidateId.H5_Q45
    )

    assert definition.minimum_percentile == Decimal("45")
    assert definition.maximum_percentile == Decimal("75")
    assert definition.lookback == 100


def test_candidate_reuses_existing_h5_strategy() -> None:
    strategy = build_h5_parameter_candidate(
        H5ParameterCandidateId.H5_Q35,
        TrendMomentumConfig(),
    )

    assert isinstance(strategy, VolatilityBandStrategy)


def test_registry_validation_passes() -> None:
    validate_h5_parameter_registry()


def test_unknown_candidate_is_rejected() -> None:
    with pytest.raises(ValueError):
        get_h5_definition("H5_Q99")  # type: ignore[arg-type]