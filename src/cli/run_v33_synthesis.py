"""Create the final H0-H9 consumed-data research synthesis.

No backtest is executed.
No strategy is evaluated.
The blind holdout is never loaded.
"""

from __future__ import annotations

import json
from pathlib import Path


MULTIREGIME = Path(
    "reports/multiregime/3db30d3c8eacef4e/summary.json"
)
MECHANISMS = Path(
    "reports/mechanisms/bc2496aed05555b5/summary.json"
)
V33 = Path(
    "reports/v3_3_h5/b104ea78b4c88bcf/summary.json"
)
H8 = Path(
    "reports/h8/369586c25375c0ef/summary.json"
)
H9 = Path(
    "reports/h9/eb7a43ccaad63dfc/summary.json"
)

OUTPUT = Path(
    "reports/synthesis/v33_research_synthesis"
)


NAMES = {
    "H0": "Frozen baseline",
    "H1": "High-volatility guard",
    "H2": "EMA-extension guard",
    "H3": "Pullback confirmation",
    "H4": "1R target",
    "H5": "ATR volatility band",
    "H6": "Trend persistence",
    "H7": "Cost/opportunity guard",
    "H5_Q25": "Frozen H5 reference",
    "H5_Q35": "H5 lower ATR Q35",
    "H5_Q45": "H5 lower ATR Q45",
    "H5_Q50": "H5 lower ATR Q50",
    "H8": "+1R break-even protection",
    "H9": "Early-failure exit",
}


def load(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing research report: {path}"
        )

    return json.loads(
        path.read_text(encoding="utf-8")
    )


def verify_holdout(payload: dict, key: str) -> None:
    holdout = payload.get(key)

    if not isinstance(holdout, dict):
        raise ValueError(
            f"Missing holdout integrity block: {key}"
        )

    if holdout.get("status") != "LOCKED_BLIND_HOLDOUT":
        raise ValueError(
            "Blind holdout is not locked."
        )

    for field in (
        "revealed",
        "consumed",
        "evaluated",
    ):
        if holdout.get(field) is not False:
            raise ValueError(
                f"Blind holdout integrity failed: {field}"
            )

    if "loaded" in holdout and holdout["loaded"] is not False:
        raise ValueError(
            "Blind holdout was loaded."
        )


def screening_row(item: dict) -> dict:
    hypothesis = item["hypothesis"]

    return {
        "id": hypothesis,
        "name": NAMES[hypothesis],
        "trades": item["combined_trades"],
        "gross_expectancy": item[
            "combined_frictionless_expectancy"
        ],
        "net_expectancy": item[
            "combined_net_expectancy"
        ],
        "profit_factor": item[
            "combined_profit_factor"
        ],
        "stress_net_expectancy": item[
            "stress_expectancy"
        ],
        "gross_better_windows": item[
            "frictionless_better_windows"
        ],
        "net_better_windows": item[
            "net_better_windows"
        ],
        "eligible_windows": item[
            "eligible_windows"
        ],
        "trade_count_ratio": item[
            "trade_count_ratio"
        ],
        "classification": item[
            "classification"
        ],
    }


def v33_row(
    candidate_id: str,
    candidate: dict,
) -> dict:
    base = candidate["base"]
    stress = candidate["stress"]
    gate = candidate["gate"]

    return {
        "id": candidate_id,
        "name": NAMES[candidate_id],
        "trades": candidate["trades"],
        "gross_expectancy_r": base[
            "frictionless_expectancy_r"
        ],
        "net_expectancy_r": base[
            "net_expectancy_r"
        ],
        "profit_factor_r": base[
            "profit_factor_r"
        ],
        "stress_net_expectancy_r": stress[
            "net_expectancy_r"
        ],
        "gross_better_windows": gate[
            "frictionless_better_windows"
        ],
        "net_better_windows": gate[
            "net_better_windows"
        ],
        "positive_net_windows": gate[
            "positive_net_windows"
        ],
        "eligible_windows": gate[
            "eligible_windows"
        ],
        "trade_count_ratio": gate[
            "trade_count_ratio"
        ],
        "classification": candidate[
            "classification"
        ],
    }


def mechanism_row(
    hypothesis_id: str,
    payload: dict,
) -> dict:
    section_key = hypothesis_id.lower()

    if section_key not in payload:
        raise ValueError(
            f"Missing {hypothesis_id} result section."
            f"(expected key: {section_key})."
        )
    
    section = payload[section_key]
    base = section["base"]
    stress = section["stress"]
    gate = payload["gate"]

    return {
        "id": hypothesis_id,
        "name": NAMES[hypothesis_id],
        "trades": section["trades"],
        "gross_expectancy_r": base[
            "frictionless_expectancy_r"
        ],
        "net_expectancy_r": base[
            "net_expectancy_r"
        ],
        "profit_factor_r": base[
            "profit_factor_r"
        ],
        "stress_net_expectancy_r": stress[
            "net_expectancy_r"
        ],
        "gross_better_windows": gate[
            "frictionless_better_windows"
        ],
        "net_better_windows": gate[
            "net_better_windows"
        ],
        "positive_net_windows": gate[
            "positive_net_windows"
        ],
        "eligible_windows": gate[
            "eligible_windows"
        ],
        "trade_count_ratio": gate[
            "trade_count_ratio"
        ],
        "classification": (
            "SUPPORTED"
            if payload["all_support_gates_met"]
            else "NOT_SUPPORTED"
        ),
        "v3_4_eligible": payload[
            "v3_4_eligible"
        ],
    }


def markdown_table(
    headers: tuple[str, ...],
    rows: list[tuple],
) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]

    for row in rows:
        lines.append(
            "| "
            + " | ".join(str(value) for value in row)
            + " |"
        )

    return "\n".join(lines)


def main() -> int:
    multiregime = load(MULTIREGIME)
    mechanisms = load(MECHANISMS)
    v33 = load(V33)
    h8 = load(H8)
    h9 = load(H9)

    verify_holdout(
        multiregime,
        "holdout",
    )
    verify_holdout(
        h8,
        "blind_holdout",
    )
    verify_holdout(
        h9,
        "blind_holdout",
    )

    multi_by_id = {
        item["hypothesis"]: item
        for item in multiregime[
            "candidate_stability"
        ]
    }

    mechanism_by_id = {
        item["hypothesis"]: item
        for item in mechanisms[
            "candidate_stability"
        ]
    }

    screening = [
        screening_row(
            multi_by_id[hypothesis]
        )
        for hypothesis in (
            "H0",
            "H1",
            "H2",
            "H3",
            "H4",
        )
    ]

    screening.extend(
        screening_row(
            mechanism_by_id[hypothesis]
        )
        for hypothesis in (
            "H5",
            "H6",
            "H7",
        )
    )

    focused = [
        v33_row(
            candidate_id,
            v33["candidates"][candidate_id],
        )
        for candidate_id in (
            "H5_Q25",
            "H5_Q35",
            "H5_Q45",
            "H5_Q50",
        )
    ]

    focused.append(
        mechanism_row(
            "H8",
            h8,
        )
    )

    focused.append(
        mechanism_row(
            "H9",
            h9,
        )
    )

    if v33.get("run_id") != "b104ea78b4c88bcf":
        raise ValueError(
            "Unexpected frozen V3.3 run."
        )

    if h8.get("run_id") != "369586c25375c0ef":
        raise ValueError(
            "Unexpected H8 run."
        )

    if h9.get("run_id") != "eb7a43ccaad63dfc":
        raise ValueError(
            "Unexpected H9 run."
        )

    if any(
        float(row["net_expectancy_r"]) > 0
        for row in focused
    ):
        branch_decision = (
            "REVIEW_POSITIVE_NET_CANDIDATE"
        )
    else:
        branch_decision = (
            "CLOSE_CURRENT_H5_TUNING_BRANCH_ON_CONSUMED_DATA"
        )

    payload = {
        "version": "3.3-research-synthesis",
        "dataset_status": "CONSUMED_RESEARCH_DATA",

        "metric_separation": {
            "screening_table": (
                "BACKTEST_QUOTE_CURRENCY_PER_TRADE"
            ),
            "focused_h5_table": (
                "R_NORMALIZED_PER_TRADE"
            ),
            "warning": (
                "Do not compare raw expectancy magnitudes "
                "between these two tables."
            ),
        },

        "screening_h0_h7": screening,
        "focused_h5_h9": focused,

        "branch_decision": branch_decision,

        "conclusions": [
            (
                "H5 was the strongest broad mechanism "
                "relative to H0."
            ),
            (
                "H5_Q25 has positive frictionless edge "
                "but negative net R expectancy."
            ),
            (
                "Q35 and Q45 were not supported."
            ),
            (
                "Q50 was mixed and failed frozen "
                "consistency/trade-retention gates."
            ),
            (
                "H8 break-even protection was not supported."
            ),
            (
                "H9 early-failure exit was not supported."
            ),
            (
                "No H5-family candidate achieved positive "
                "net R expectancy or PF > 1."
            ),
            (
                "Further threshold tuning on the same "
                "consumed data should stop."
            ),
        ],

        "blind_holdout": {
            "status": "LOCKED_BLIND_HOLDOUT",
            "loaded": False,
            "revealed": False,
            "consumed": False,
            "evaluated": False,
        },
    }

    OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    json_path = OUTPUT / "summary.json"
    md_path = OUTPUT / "RESEARCH_SYNTHESIS.md"

    json_path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    screening_md = markdown_table(
        (
            "ID",
            "Trades",
            "Gross",
            "Net",
            "PF",
            "Stress",
            "Gross wins",
            "Net wins",
            "Class",
        ),
        [
            (
                row["id"],
                row["trades"],
                row["gross_expectancy"],
                row["net_expectancy"],
                row["profit_factor"],
                row["stress_net_expectancy"],
                (
                    f'{row["gross_better_windows"]}/'
                    f'{row["eligible_windows"]}'
                ),
                (
                    f'{row["net_better_windows"]}/'
                    f'{row["eligible_windows"]}'
                ),
                row["classification"],
            )
            for row in screening
        ],
    )

    focused_md = markdown_table(
        (
            "ID",
            "Trades",
            "Gross R",
            "Net R",
            "PF",
            "Stress R",
            "Gross wins",
            "Net wins",
            "+Net windows",
            "Class",
        ),
        [
            (
                row["id"],
                row["trades"],
                row["gross_expectancy_r"],
                row["net_expectancy_r"],
                row["profit_factor_r"],
                row["stress_net_expectancy_r"],
                (
                    f'{row["gross_better_windows"]}/'
                    f'{row["eligible_windows"]}'
                ),
                (
                    f'{row["net_better_windows"]}/'
                    f'{row["eligible_windows"]}'
                ),
                (
                    f'{row["positive_net_windows"]}/'
                    f'{row["eligible_windows"]}'
                ),
                row["classification"],
            )
            for row in focused
        ],
    )

    markdown = f"""# V3.3 Research Synthesis

Dataset: **CONSUMED_RESEARCH_DATA**

Blind holdout: **LOCKED / UNTOUCHED**

## H0-H7 multi-regime screening

Metric basis: **quote currency per trade**.

{screening_md}

## H5 focused R-normalized research

Metric basis: **R per trade**.

{focused_md}

## Decision

**{branch_decision}**

The current H5 tuning branch has no candidate with
positive net R expectancy and PF > 1.

H8 and H9 both failed their preregistered support gates.

Further threshold tuning on the same consumed research
data should stop.

The blind holdout remains locked and has not been used.
"""

    md_path.write_text(
        markdown,
        encoding="utf-8",
    )

    print()
    print("V3.3 RESEARCH SYNTHESIS COMPLETE")
    print()
    print(
        "Screening H0-H7: "
        "BACKTEST QUOTE CURRENCY / TRADE"
    )
    print(
        "Focused H5-H9: R-NORMALIZED / TRADE"
    )
    print()
    print(f"Decision: {branch_decision}")
    print()
    print(f"JSON: {json_path}")
    print(f"Markdown: {md_path}")
    print()
    print(
        "Blind holdout: LOCKED | NOT LOADED | "
        "NOT REVEALED | NOT CONSUMED | NOT EVALUATED"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())