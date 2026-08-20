"""Chronological, offline strategy research framework."""

from src.research.dataset_split import (
    ResearchPeriod,
    ResearchPeriodRange,
    ResearchSplit,
    split_dataset,
)
from src.research.runner import ResearchConfig, ResearchResult, StrategyResearchRunner

__all__ = [
    "ResearchConfig",
    "ResearchPeriod",
    "ResearchPeriodRange",
    "ResearchResult",
    "ResearchSplit",
    "StrategyResearchRunner",
    "split_dataset",
]

