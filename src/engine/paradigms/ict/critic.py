from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

import polars as pl

from engine.paradigms.registry import RunContextLike

log = logging.getLogger(__name__)


def score_trades(
    ctx: RunContextLike,
    trade_paths: pl.DataFrame,
    decisions_hypotheses: pl.DataFrame,
    critic_cfg: Mapping[str, Any],
) -> pl.DataFrame:
    """
    Phase A ICT critic: noop scorer.
    Returns decisions_critic (one row per trade_id).
    """
    if trade_paths is None or trade_paths.is_empty():
        log.info(
            "ict.critic: no trades for snapshot_id=%s run_id=%s; skipping critic.",
            ctx.snapshot_id,
            ctx.run_id,
        )
        return pl.DataFrame(
            {
                "snapshot_id": pl.Series([], dtype=pl.Utf8),
                "run_id": pl.Series([], dtype=pl.Utf8),
                "mode": pl.Series([], dtype=pl.Utf8),
                "paradigm_id": pl.Series([], dtype=pl.Utf8),
                "principle_id": pl.Series([], dtype=pl.Utf8),
                "instrument": pl.Series([], dtype=pl.Utf8),
                "trade_id": pl.Series([], dtype=pl.Utf8),
                "critic_score_at_entry": pl.Series([], dtype=pl.Float64),
                "critic_reason_tags_at_entry": pl.Series([], dtype=pl.Utf8),
                "critic_reason_cluster_id": pl.Series([], dtype=pl.Utf8),
            }
        )

    base_columns = [
        "snapshot_id",
        "run_id",
        "mode",
        "paradigm_id",
        "principle_id",
        "instrument",
        "trade_id",
    ]
    base = trade_paths.select(base_columns).unique()
    sentinel = "unknown"
    context_columns = [
        "vol_regime",
        "macro_state",
        "corr_cluster_id",
        "tod_bucket",
    ]

    if decisions_hypotheses is None or decisions_hypotheses.is_empty():
        context_df = base.select("trade_id").with_columns(
            [pl.lit(sentinel).alias(name) for name in context_columns]
        )
    else:
        context_exprs: list[pl.Expr] = []
        for name in context_columns:
            if name in decisions_hypotheses.columns:
                context_exprs.append(
                    pl.col(name).cast(pl.Utf8, strict=False).fill_null(sentinel).alias(name)
                )
            else:
                context_exprs.append(pl.lit(sentinel).alias(name))
        context_df = decisions_hypotheses.select(
            [
                pl.col("trade_id").cast(pl.Utf8, strict=False).alias("trade_id"),
                *context_exprs,
            ]
        ).unique(subset=["trade_id"])

    decisions_critic = (
        base.join(context_df, on="trade_id", how="left")
        .with_columns(
            pl.lit(0.0).alias("critic_score_at_entry"),
            pl.lit('["ict_noop"]').alias("critic_reason_tags_at_entry"),
            pl.concat_str(
                [pl.col(name).fill_null(sentinel) for name in context_columns],
                separator="|",
            ).alias("critic_reason_cluster_id"),
        )
        .select(
            [
                *base_columns,
                "critic_score_at_entry",
                "critic_reason_tags_at_entry",
                "critic_reason_cluster_id",
            ]
        )
    )

    return decisions_critic
