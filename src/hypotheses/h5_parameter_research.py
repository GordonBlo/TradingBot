"""Pre-registered V3.3 controlled parameter research for H5."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import Enum

from src.hypotheses.mechanisms import VolatilityBandStrategy
from src.strategy.base import BaseStrategy
from src.strategy.models import TrendMomentumConfig


class H5ParameterCandidateId(str, Enum):
    """Fixed V3.3 H5 candidate set."""

    H5_Q25 = "H5_Q25"
    H5_Q35 = "H5_Q35"
    H5_Q45 = "H5_Q45"
    H5_Q50 = "H5_Q50"


@dataclass(frozen=True, slots=True)
class H5BandDefinition:
    candidate_id: H5ParameterCandidateId
    lookback: int
    minimum_percentile: Decimal
    maximum_percentile: Decimal
    is_frozen_reference: bool = False


def h5_parameter_registry() -> tuple[H5BandDefinition, ...]:
    """
    Return the complete pre-registered V3.3 search space.

    Only the LOWER H5 ATR-percentile boundary changes.

    Frozen:
        lookback = 100
        upper percentile = Q75

    Tested lower boundaries:
        Q25, Q35, Q45, Q50
    """

    return (
        H5BandDefinition(
            candidate_id=H5ParameterCandidateId.H5_Q25,
            lookback=100,
            minimum_percentile=Decimal("25"),
            maximum_percentile=Decimal("75"),
            is_frozen_reference=True,
        ),
        H5BandDefinition(
            candidate_id=H5ParameterCandidateId.H5_Q35,
            lookback=100,
            minimum_percentile=Decimal("35"),
            maximum_percentile=Decimal("75"),
        ),
        H5BandDefinition(
            candidate_id=H5ParameterCandidateId.H5_Q45,
            lookback=100,
            minimum_percentile=Decimal("45"),
            maximum_percentile=Decimal("75"),
        ),
        H5BandDefinition(
            candidate_id=H5ParameterCandidateId.H5_Q50,
            lookback=100,
            minimum_percentile=Decimal("50"),
            maximum_percentile=Decimal("75"),
        ),
    )


def get_h5_definition(
    candidate_id: H5ParameterCandidateId,
) -> H5BandDefinition:
    candidate = H5ParameterCandidateId(candidate_id)

    for definition in h5_parameter_registry():
        if definition.candidate_id is candidate:
            return definition

    raise AssertionError("Unreachable H5 V3.3 candidate.")


def build_h5_parameter_candidate(
    candidate_id: H5ParameterCandidateId,
    baseline_config: TrendMomentumConfig,
) -> BaseStrategy:
    """
    Build one V3.3 candidate using the existing causal H5 implementation.

    No H5 mathematics are duplicated here.
    """

    definition = get_h5_definition(candidate_id)

    return VolatilityBandStrategy(
        baseline_config,
        lookback=definition.lookback,
        minimum_percentile=definition.minimum_percentile,
        maximum_percentile=definition.maximum_percentile,
    )


def validate_h5_parameter_registry() -> None:
    """Fail if the pre-registered search space is accidentally changed."""

    definitions = h5_parameter_registry()

    expected_ids = (
        H5ParameterCandidateId.H5_Q25,
        H5ParameterCandidateId.H5_Q35,
        H5ParameterCandidateId.H5_Q45,
        H5ParameterCandidateId.H5_Q50,
    )

    if tuple(item.candidate_id for item in definitions) != expected_ids:
        raise ValueError("V3.3 H5 candidate registry changed.")

    if len(definitions) != 4:
        raise ValueError("V3.3 must contain exactly four H5 candidates.")

    expected_lower_bounds = (
        Decimal("25"),
        Decimal("35"),
        Decimal("45"),
        Decimal("50"),
    )

    if (
        tuple(item.minimum_percentile for item in definitions)
        != expected_lower_bounds
    ):
        raise ValueError("V3.3 H5 lower-percentile search space changed.")

    for item in definitions:
        if item.lookback != 100:
            raise ValueError("V3.3 H5 lookback is frozen at 100.")

        if item.maximum_percentile != Decimal("75"):
            raise ValueError("V3.3 H5 upper percentile is frozen at Q75.")

        if not (
            Decimal("0")
            <= item.minimum_percentile
            < item.maximum_percentile
            <= Decimal("100")
        ):
            raise ValueError("Invalid H5 percentile band.")

    references = [
        item
        for item in definitions
        if item.is_frozen_reference
    ]

    if len(references) != 1:
        raise ValueError(
            "Exactly one frozen H5 reference candidate is required."
        )

    reference = references[0]

    if (
        reference.candidate_id
        is not H5ParameterCandidateId.H5_Q25
    ):
        raise ValueError(
            "H5_Q25 must remain the frozen V3.2.2 reference."
        )