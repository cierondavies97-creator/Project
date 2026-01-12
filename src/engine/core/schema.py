from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl
import yaml

# ---------------------------------------------------------------------------
# Snapshot manifest (snapshots/<SNAPSHOT_ID>.json)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SnapshotConfigRefs:
    retail_config_path: str
    features_registry_path: str
    features_auto_path: str
    rails_auto_path: str
    portfolio_auto_path: str
    path_filters_auto_path: str
    paradigms_dir: str
    principles_dir: str


@dataclass(frozen=True)
class SnapshotParadigmRef:
    paradigm_id: str
    paradigm_config_path: str


@dataclass(frozen=True)
class SnapshotPrincipleRef:
    paradigm_id: str
    principle_id: str
    principle_config_path: str


@dataclass(frozen=True)
class SnapshotDataSlice:
    start_dt: str
    end_dt: str
    instruments: list[str]
    anchor_tfs: list[str]
    tf_entries: list[str]
    contexts_filter: str | None


@dataclass(frozen=True)
class SnapshotManifest:
    snapshot_id: str
    description: str
    created_ts: str
    config_refs: SnapshotConfigRefs
    paradigms: list[SnapshotParadigmRef]
    principles: list[SnapshotPrincipleRef]
    data_slice: SnapshotDataSlice


def load_snapshot_manifest(path: Path | str) -> SnapshotManifest:
    """
    Load snapshots/<SNAPSHOT_ID>.json (stored as YAML/JSON) into a SnapshotManifest.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Snapshot manifest not found at: {p}")

    with p.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    cfg_refs_raw = raw["config_refs"]
    cfg_refs = SnapshotConfigRefs(
        retail_config_path=cfg_refs_raw["retail_config_path"],
        features_registry_path=cfg_refs_raw["features_registry_path"],
        features_auto_path=cfg_refs_raw["features_auto_path"],
        rails_auto_path=cfg_refs_raw["rails_auto_path"],
        portfolio_auto_path=cfg_refs_raw["portfolio_auto_path"],
        path_filters_auto_path=cfg_refs_raw["path_filters_auto_path"],
        paradigms_dir=cfg_refs_raw["paradigms_dir"],
        principles_dir=cfg_refs_raw["principles_dir"],
    )

    paradigms = [
        SnapshotParadigmRef(
            paradigm_id=item["paradigm_id"],
            paradigm_config_path=item["paradigm_config_path"],
        )
        for item in raw.get("paradigms", [])
    ]

    principles = [
        SnapshotPrincipleRef(
            paradigm_id=item["paradigm_id"],
            principle_id=item["principle_id"],
            principle_config_path=item["principle_config_path"],
        )
        for item in raw.get("principles", [])
    ]

    ds_raw = raw["data_slice"]
    data_slice = SnapshotDataSlice(
        start_dt=ds_raw["start_dt"],
        end_dt=ds_raw["end_dt"],
        instruments=list(ds_raw.get("instruments", [])),
        anchor_tfs=list(ds_raw.get("anchor_tfs", [])),
        tf_entries=list(ds_raw.get("tf_entries", [])),
        contexts_filter=ds_raw.get("contexts_filter"),
    )

    return SnapshotManifest(
        snapshot_id=raw["snapshot_id"],
        description=raw.get("description", ""),
        created_ts=raw["created_ts"],
        config_refs=cfg_refs,
        paradigms=paradigms,
        principles=principles,
        data_slice=data_slice,
    )


# ---------------------------------------------------------------------------
# Table schema helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TableSchema:
    """
    Logical schema for a Parquet-backed table.

    Type strings:
      - "string", "int", "double", "boolean", "timestamp", "date"
      - "array<string>"
    """

    name: str
    partition_cols: list[str]
    columns: dict[str, str]


def polars_dtype(type_str: str) -> pl.DataType:
    t = type_str.strip().lower()
    if t in ("string", "utf8", "str"):
        return pl.Utf8
    if t in ("int", "int64", "i64"):
        return pl.Int64
    if t in ("double", "float", "float64", "f64"):
        return pl.Float64
    if t in ("boolean", "bool"):
        return pl.Boolean
    if t in ("timestamp", "datetime"):
        return pl.Datetime("us", time_zone="UTC")
    if t in ("date",):
        return pl.Date
    if t in ("array<string>", "list<string>"):
        return pl.List(pl.Utf8)
    return pl.Utf8


def empty_frame(schema: TableSchema) -> pl.DataFrame:
    cols: dict[str, pl.Series] = {}
    for name, t in schema.columns.items():
        cols[name] = pl.Series(name=name, values=[], dtype=polars_dtype(t))
    return pl.DataFrame(cols)


def enforce_schema(
    df: pl.DataFrame,
    schema: TableSchema,
    *,
    allow_extra: bool = True,
    reorder: bool = False,
) -> pl.DataFrame:
    """
    Ensure df contains all schema.columns with correct dtypes.

    - Adds missing schema columns as null (typed).
    - Casts existing schema columns using strict=False.
    - If allow_extra=False, drops columns not in schema.
    - If reorder=True, orders schema cols first (extras appended).

    Note: legacy columns (e.g. "trading_day") are tolerated via allow_extra=True
    but are not required by Phase B.
    """
    if df is None or df.is_empty():
        out = empty_frame(schema)
        return out if allow_extra else out.select(list(schema.columns.keys()))

    out = df

    missing: list[pl.Expr] = []
    for col, t in schema.columns.items():
        if col not in out.columns:
            missing.append(pl.lit(None).cast(polars_dtype(t)).alias(col))
    if missing:
        out = out.with_columns(missing)

    cast_exprs: list[pl.Expr] = []
    for col, t in schema.columns.items():
        if col in out.columns:
            cast_exprs.append(pl.col(col).cast(polars_dtype(t), strict=False).alias(col))
    if cast_exprs:
        out = out.with_columns(cast_exprs)

    if not allow_extra:
        out = out.select(list(schema.columns.keys()))
        return out

    if reorder:
        schema_cols = [c for c in schema.columns.keys() if c in out.columns]
        extras = [c for c in out.columns if c not in schema.columns]
        out = out.select(schema_cols + extras)

    return out


# ---------------------------------------------------------------------------
# Concrete table schemas (dt is canonical; trading_day is legacy/optional)
# ---------------------------------------------------------------------------

WINDOWS_SCHEMA = TableSchema(
    name="data/windows",
    partition_cols=["run_id", "instrument", "anchor_tf", "dt"],
    columns={
        "snapshot_id": "string",
        "run_id": "string",
        "mode": "string",
        "dt": "date",
        "paradigm_id": "string",
        "principle_id": "string",
        "instrument": "string",
        "anchor_tf": "string",
        "anchor_ts": "timestamp",
        "tf_entry": "string",
        "tod_bucket": "string",
        "dow_bucket": "string",
        "vol_regime": "string",
        "trend_regime": "string",
        "macro_state": "string",
        "macro_is_blackout": "boolean",
        "macro_blackout_max_impact": "int",
        "micro_corr_regime": "string",
        "corr_cluster_id": "string",
        "zone_behaviour_type_bucket": "string",
        "zone_freshness_bucket": "string",
        "zone_stack_depth_bucket": "string",
        "zone_htf_confluence_bucket": "string",
        "zone_vp_type_bucket": "string",
        "unsup_regime_id": "string",
        "entry_profile_id": "string",
        "management_profile_id": "string",
    },
)

TRADE_PATHS_SCHEMA = TableSchema(
    name="data/trade_paths",
    partition_cols=["run_id", "paradigm_id", "principle_id", "candidate_id", "experiment_id", "instrument", "dt"],
    columns={
        "snapshot_id": "string",
        "run_id": "string",
        "mode": "string",
        "dt": "date",
        "paradigm_id": "string",
        "principle_id": "string",
        "candidate_id": "string",
        "experiment_id": "string",
        "trade_id": "string",
        "instrument": "string",
    },
)

DECISIONS_SCHEMA = TableSchema(
    name="data/decisions",
    partition_cols=["run_id", "paradigm_id", "principle_id", "candidate_id", "experiment_id", "instrument", "stage", "dt"],
    columns={
        "snapshot_id": "string",
        "run_id": "string",
        "mode": "string",
        "dt": "date",
        "stage": "string",
        "paradigm_id": "string",
        "principle_id": "string",
        "candidate_id": "string",
        "experiment_id": "string",
        "instrument": "string",
        "trade_id": "string",
    },
)


# ---------------------------------------------------------------------------
# Decisions (stage-specific schemas)
#
# NOTE:
# - Missing columns are auto-added as typed nulls by enforce_schema(...),
#   so you can land the contract first and fill producers later.
# - Arrays are represented as JSON-encoded strings for portability.
# ---------------------------------------------------------------------------

_DECISIONS_HYPOTHESES_EXTRA_COLS: dict[str, str] = {
    # Timeframes / structure
    "anchor_tf": "string",
    "anchor_ts": "timestamp",
    "tf_entry": "string",

    # Trade intent
    "side": "string",
    "setup_logic_id": "string",
    "entry_profile_id": "string",
    "management_profile_id": "string",

    # Price hints
    "entry_hint_price": "double",
    "stop_hint_price": "double",
    "tp_hint_prices": "string",  # JSON array of floats

    # Context tuple + regimes
    "tod_bucket": "string",
    "dow_bucket": "string",
    "vol_regime": "string",
    "trend_regime": "string",
    "macro_state": "string",
    "macro_is_blackout": "boolean",
    "macro_blackout_max_impact": "int",
    "micro_corr_regime": "string",
    "corr_cluster_id": "string",
    "unsup_regime_id": "string",

    # Portable payloads
    "context_keys_json": "string",
    "hypothesis_features_json": "string",
}

_DECISIONS_CRITIC_EXTRA_COLS: dict[str, str] = {
    "critic_score_at_entry": "double",
    "critic_reason_tags_at_entry": "string",  # JSON array of strings
    "critic_reason_cluster_id": "string",
}

_DECISIONS_PRETRADE_EXTRA_COLS: dict[str, str] = {
    "macro_is_blackout": "boolean",
    "macro_blackout_max_impact": "int",
    "macro_blackout_tag": "string",
    "blocked_by_macro": "boolean",
    "blocked_by_spread": "boolean",
    "micro_risk_scale": "double",
    "pretrade_notes": "string",
}

_DECISIONS_GATEKEEPER_EXTRA_COLS: dict[str, str] = {
    # Keep both during migration
    "gatekeeper_status": "string",
    "gate_status": "string",

    "status": "string",
    "rails_passed_flag": "boolean",
    "rails_config_key": "string",
    "gate_rails_version": "string",
    "gate_reason": "string",
    "risk_mode": "string",
    "risk_per_trade_bps": "double",
}

_DECISIONS_PORTFOLIO_EXTRA_COLS: dict[str, str] = {
    # Current dev stub
    "portfolio_risk_bps_at_entry": "double",

    # Target allocator outputs
    "allocated_notional": "double",
    "allocated_risk_bps": "double",
    "instrument_weight_fraction": "double",
    "cluster_weight_fraction": "double",
    "portfolio_risk_bps_after": "double",

    "allocation_mode": "string",
    "drop_reason": "string",
}

# Optional: make portfolio decisions sufficient as a fallback source for brackets
_DECISIONS_BRACKET_FALLBACK_COLS: dict[str, str] = {
    "entry_ts": "timestamp",
    "entry_px": "double",
    "sl_px": "double",
    "tp_px": "double",
    "entry_mode": "string",
    "order_type": "string",
}

DECISIONS_HYPOTHESES_SCHEMA = TableSchema(
    name=DECISIONS_SCHEMA.name,
    partition_cols=DECISIONS_SCHEMA.partition_cols,
    columns={**dict(DECISIONS_SCHEMA.columns), **_DECISIONS_HYPOTHESES_EXTRA_COLS},
)

DECISIONS_CRITIC_SCHEMA = TableSchema(
    name=DECISIONS_SCHEMA.name,
    partition_cols=DECISIONS_SCHEMA.partition_cols,
    columns={
        **dict(DECISIONS_SCHEMA.columns),
        **_DECISIONS_HYPOTHESES_EXTRA_COLS,
        **_DECISIONS_CRITIC_EXTRA_COLS,
    },
)

DECISIONS_PRETRADE_SCHEMA = TableSchema(
    name=DECISIONS_SCHEMA.name,
    partition_cols=DECISIONS_SCHEMA.partition_cols,
    columns={
        **dict(DECISIONS_SCHEMA.columns),
        **_DECISIONS_HYPOTHESES_EXTRA_COLS,
        **_DECISIONS_CRITIC_EXTRA_COLS,
        **_DECISIONS_PRETRADE_EXTRA_COLS,
    },
)

DECISIONS_GATEKEEPER_SCHEMA = TableSchema(
    name=DECISIONS_SCHEMA.name,
    partition_cols=DECISIONS_SCHEMA.partition_cols,
    columns={
        **dict(DECISIONS_SCHEMA.columns),
        **_DECISIONS_HYPOTHESES_EXTRA_COLS,
        **_DECISIONS_CRITIC_EXTRA_COLS,
        **_DECISIONS_PRETRADE_EXTRA_COLS,
        **_DECISIONS_GATEKEEPER_EXTRA_COLS,
    },
)

DECISIONS_PORTFOLIO_SCHEMA = TableSchema(
    name=DECISIONS_SCHEMA.name,
    partition_cols=DECISIONS_SCHEMA.partition_cols,
    columns={
        **dict(DECISIONS_SCHEMA.columns),
        **_DECISIONS_HYPOTHESES_EXTRA_COLS,
        **_DECISIONS_CRITIC_EXTRA_COLS,
        **_DECISIONS_PRETRADE_EXTRA_COLS,
        **_DECISIONS_GATEKEEPER_EXTRA_COLS,
        **_DECISIONS_PORTFOLIO_EXTRA_COLS,
        **_DECISIONS_BRACKET_FALLBACK_COLS,
    },
)

BRACKETS_SCHEMA = TableSchema(
    name="data/brackets",
    partition_cols=["run_id", "paradigm_id", "principle_id", "candidate_id", "experiment_id", "instrument", "dt"],
    columns={
        "snapshot_id": "string",
        "run_id": "string",
        "mode": "string",
        "dt": "date",
        "paradigm_id": "string",
        "principle_id": "string",
        "candidate_id": "string",
        "experiment_id": "string",
        "instrument": "string",
        "trade_id": "string",
    },
)

ORDERS_SCHEMA = TableSchema(
    name="data/orders",
    partition_cols=["run_id", "paradigm_id", "principle_id", "candidate_id", "experiment_id", "instrument", "dt"],
    columns={
        "snapshot_id": "string",
        "run_id": "string",
        "mode": "string",
        "dt": "date",
        "paradigm_id": "string",
        "principle_id": "string",
        "candidate_id": "string",
        "experiment_id": "string",
        "instrument": "string",
    },
)

FILLS_SCHEMA = TableSchema(
    name="data/fills",
    partition_cols=["run_id", "paradigm_id", "principle_id", "candidate_id", "experiment_id", "instrument", "dt"],
    columns={
        "snapshot_id": "string",
        "run_id": "string",
        "mode": "string",
        "dt": "date",
        "paradigm_id": "string",
        "principle_id": "string",
        "candidate_id": "string",
        "experiment_id": "string",
        "instrument": "string",
    },
)

RUN_REPORTS_SCHEMA = TableSchema(
    name="data/run_reports",
    partition_cols=["run_id", "dt", "cluster_id"],
    columns={
        "snapshot_id": "string",
        "run_id": "string",
        "mode": "string",
        "dt": "date",
        "cluster_id": "string",
    },
)

PRINCIPLES_CONTEXT_SCHEMA = TableSchema(
    name="data/principles_context",
    partition_cols=["run_id", "paradigm_id", "principle_id", "candidate_id", "experiment_id", "dt", "cluster_id"],
    columns={
        "snapshot_id": "string",
        "run_id": "string",
        "mode": "string",
        "dt": "date",
        "cluster_id": "string",
        "paradigm_id": "string",
        "principle_id": "string",
        "candidate_id": "string",
        "experiment_id": "string",
        "trade_count": "int",
    },
)

TRADE_CLUSTERS_SCHEMA = TableSchema(
    name="data/trade_clusters",
    partition_cols=["run_id", "paradigm_id", "principle_id", "candidate_id", "experiment_id", "dt", "cluster_id"],
    columns={
        "snapshot_id": "string",
        "run_id": "string",
        "mode": "string",
        "dt": "date",
        "cluster_id": "string",
        "paradigm_id": "string",
        "principle_id": "string",
        "candidate_id": "string",
        "experiment_id": "string",
        "trade_count": "int",
    },
)

# ---------------------------------------------------------------------------
# Additional table schemas (dev stubs)
# ---------------------------------------------------------------------------

FEATURES_SCHEMA = TableSchema(
    name="data/features",
    partition_cols=["run_id", "instrument", "anchor_tf", "dt"],
    columns={
        "snapshot_id": "string",
        "run_id": "string",
        "mode": "string",
        "dt": "date",

        "instrument": "string",
        "anchor_tf": "string",
        "ts": "timestamp",

        # ict_struct (dev-stub)
        "fvg_direction": "string",
        "fvg_gap_ticks": "int",
        "fvg_origin_tf": "string",
        "fvg_origin_ts": "timestamp",
        "fvg_fill_state": "string",
        "fvg_location_bucket": "string",

        "ob_type": "string",
        "ob_high": "double",
        "ob_low": "double",
        "ob_origin_ts": "timestamp",
        "ob_freshness_bucket": "string",

        "eqh_flag": "bool",
        "eql_flag": "bool",
        "liq_grab_flag": "bool",

        "atr_anchor": "double",
        "atr_z": "double",
    },
)

ZONES_STATE_SCHEMA = TableSchema(
    name="data/zones_state",
    partition_cols=["instrument", "anchor_tf", "dt"],
    columns={
        "dt": "date",
        "instrument": "string",
        "anchor_tf": "string",
        "zone_id": "string",
        "ts": "timestamp",

        # minimal ZMF/VP buckets (dev-stub)
        "zone_behaviour_type_bucket": "string",
        "zone_freshness_bucket": "string",
        "zone_stack_depth_bucket": "string",
        "zone_htf_confluence_bucket": "string",
        "zone_vp_type_bucket": "string",
    },
)

PCRA_SCHEMA = TableSchema(
    name="data/pcr_a",
    partition_cols=["instrument", "anchor_tf", "dt"],
    columns={
        "dt": "date",
        "instrument": "string",
        "anchor_tf": "string",
        "pcr_window_ts": "timestamp",

        # typical bar-level microstructure fields (dev-stub)
        "pcr_range_value": "double",
        "pcr_micro_vol_value": "double",
        "pcr_ofi_value": "double",
        "pcr_clv_value": "double",
    },
)

TABLE_SCHEMAS: dict[str, TableSchema] = {
    "windows": WINDOWS_SCHEMA,
    "trade_paths": TRADE_PATHS_SCHEMA,
    "decisions_hypotheses": DECISIONS_HYPOTHESES_SCHEMA,
    "decisions_critic": DECISIONS_CRITIC_SCHEMA,
    "decisions_pretrade": DECISIONS_PRETRADE_SCHEMA,
    "decisions_gatekeeper": DECISIONS_GATEKEEPER_SCHEMA,
    "decisions_portfolio": DECISIONS_PORTFOLIO_SCHEMA,
    "brackets": BRACKETS_SCHEMA,
    "orders": ORDERS_SCHEMA,
    "fills": FILLS_SCHEMA,
    "reports": RUN_REPORTS_SCHEMA,
    "principles_context": PRINCIPLES_CONTEXT_SCHEMA,
    "trade_clusters": TRADE_CLUSTERS_SCHEMA,
    "features": FEATURES_SCHEMA,
    "zones_state": ZONES_STATE_SCHEMA,
    "pcr_a": PCRA_SCHEMA,
}


def get_table_schema(key: str) -> TableSchema:
    try:
        return TABLE_SCHEMAS[key]
    except KeyError as e:
        raise KeyError(f"No TableSchema registered for key={key!r}") from e


def enforce_table(
    df: pl.DataFrame,
    key: str,
    *,
    allow_extra: bool = True,
    reorder: bool = False,
) -> pl.DataFrame:
    return enforce_schema(df, get_table_schema(key), allow_extra=allow_extra, reorder=reorder)



