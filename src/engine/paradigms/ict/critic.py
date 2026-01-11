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

    base = trade_paths.select(
        [
            "snapshot_id",
            "run_id",
            "mode",
            "paradigm_id",
            "principle_id",
            "instrument",
            "trade_id",
        ]
    ).unique()

    decisions_critic = base.with_columns(
        pl.lit(0.0).alias("critic_score_at_entry"),
        pl.lit("ict_noop").alias("critic_reason_tags_at_entry"),
        pl.lit("ict_noop").alias("critic_reason_cluster_id"),
    )

    return decisions_critic
