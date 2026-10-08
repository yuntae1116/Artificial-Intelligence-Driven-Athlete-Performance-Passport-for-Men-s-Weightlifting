

# ========================================================================
# Configuration and dependencies
# ========================================================================

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Scientific Reports APP revision: one reproducible local workflow.

Standalone:
    python APP_revision_workflow.py --raw "D:/APP/lifting_npj_raw.CSV" --out "D:/APP/results"
    python APP_revision_workflow.py --raw "D:/APP/lifting_npj_raw.CSV" --audit-only
    python APP_revision_workflow.py --self-test --out "D:/APP/software_test"

Python 3.11+ is recommended. See DEPENDENCIES below. Notebook contains the same
implementation and does not require this .py file to run.

This is a REVISED workflow, not a claim to reproduce the submitted estimates.
Transferred hyperparameters come from saved outputs of the supplied notebooks;
no new hyperparameter search is silently claimed. RNN epoch cap is configurable.
No original AUC/MAE/RMSE or earlier conversational estimates enter calculations.

Design:
* Full DOB + nationality is only a provisional key; optional existing IDs or a
  source-row linkage map can replace it. Names are audited, NOT fuzzy-merged.
* Final recorded year <=2019: development; later final year: temporal holdout.
* Development 80/20 stratified split; 20% of the 80% reserved for calibration
  AND threshold selection. Fitting/calibration/internal/temporal are disjoint.
* RNN and random-intercept + random-slope LMM forecast the identical final
  observation from earlier observations, excluding the final outcome from inputs.
* By default reference fitting uses recorded-label-0 athletes (not proven clean).
  Fitting-set residuals are athlete-level 5-fold cross-fitted. Evaluation athletes
  are never used to fit population coefficients, early stopping, or encoders.
* Sigmoid recalibration is logistic regression on XGBoost's raw margin, fitted
  on the independent calibration partition at natural label prevalence.
* No LOCF or ridge benchmark. Main, no-nationality, no-residual, LMM-residual XGB;
  >=5-record and <=2018-cutoff refits use the same pipeline.
* Bootstrap intervals condition on fitted models and observed labels; they do
  not include retraining, linkage uncertainty, label error, or causal uncertainty.
* Only competition YEAR is available in the supplied CSV. Source-order tie
  breaking is NOT verified chronology and is explicitly logged. Provide actual
  event dates/order or use the optional strict-earlier-year sensitivity.

Documentation consulted (accessed 2026-10-06):
https://scikit-learn.org/stable/modules/calibration.html
https://www.statsmodels.org/stable/generated/statsmodels.regression.mixed_linear_model.MixedLM.predict.html
https://imbalanced-learn.org/stable/common_pitfalls.html
https://xgboost.readthedocs.io/en/stable/gpu/
"""


# %% 1. Imports, configuration and reproducibility
from __future__ import annotations
import argparse
import copy
import hashlib
import importlib.metadata
import json
import logging
import math
import os
import platform
import random
import re
import sys
import time
import traceback
import unicodedata
import warnings
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Optional

DEPENDENCIES = [
    "numpy>=1.26,<3", "pandas>=2.2,<3", "scipy>=1.11,<2",
    "scikit-learn>=1.4,<2", "xgboost>=2.1,<4", "torch>=2.3,<3",
    "statsmodels>=0.14.4,<0.15", "matplotlib>=3.8,<4",
    "joblib>=1.3,<2", "threadpoolctl>=3.1,<4",
]
try:
    import numpy as np
    import pandas as pd
    from scipy.special import expit, logit
    from scipy.stats import beta as beta_distribution, mannwhitneyu, rankdata
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.model_selection import train_test_split, StratifiedKFold
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (roc_auc_score, average_precision_score, auc,
                                 precision_recall_curve, roc_curve, confusion_matrix,
                                 mean_squared_error, mean_absolute_error, r2_score,
                                 brier_score_loss, log_loss)
    import statsmodels.api as sm
    import torch
    from torch import nn
    from torch.utils.data import Dataset, DataLoader
    import xgboost as xgb
    import joblib
    from threadpoolctl import threadpool_limits
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise ImportError(
        "필요 패키지가 없습니다. Jupyter 설치 셀을 실행하거나 다음을 실행하세요:\n"
        + sys.executable + " -m pip install " + " ".join(f'\"{x}\"' for x in DEPENDENCIES)
    ) from exc

VERSION = "1.0.0"
LOG = logging.getLogger("APP_revision")
RNN_SAVED_PARAMS = dict(
    rnn_type="lstm", hidden=192, layers=1, dropout=0.3,
    emb_ec=12, emb_age=8, emb_year=32, batch_size=64, lr=0.002,
    weight_decay=0.0001,
)
XGB_SAVED_PARAMS = dict(
    colsample_bytree=0.5611388856023309,
    gamma=1.1673870188120283e-08,
    learning_rate=0.1982844112316131, max_bin=222, max_depth=5,
    min_child_weight=1, n_estimators=985, reg_alpha=0.350895097795572,
    reg_lambda=65.66519852057745, subsample=0.7083713082230609,
)

@dataclass
class Config:
    # Paths: Windows raw strings or forward slashes are supported.
    raw_csv: str = "lifting_npj_raw.CSV"
    output_dir: str = "APP_revision_results"
    encoding: str = "utf-8-sig"
    # Canonical name -> actual column name. Preserve DOB; bornyear is derived.
    columns: dict = field(default_factory=lambda: dict(
        nation="nation", dob="born", bodyweight="bweight", total="total",
        event="event", event_year="eventyear", label="doping",
        snatch="snatch", jerk="jerk"))
    name_col: Optional[str] = None
    existing_id_col: Optional[str] = None
    # Optional CSV: source_row, verified_athlete_id. Must cover EVERY retained row.
    # source_row is a 1-based CSV line number including the header (first data=2).
    linkage_map_csv: Optional[str] = None
    event_date_col: Optional[str] = None
    event_date_format: Optional[str] = None  # e.g. "%Y-%m-%d"
    event_order_col: Optional[str] = None  # verified chronological rank within year
    dob_format: str = "auto"  # auto; "%d.%m.%Y"; "YYMMDD"; "%Y%m%d"
    two_digit_year_pivot: int = 25  # YY<=25 -> 20YY, otherwise 19YY; explicit rule
    require_verified_ids: bool = False
    require_exact_chronology: bool = False
    # Mapping/names are author-supplied; the code cannot establish real identity.
    identity_verification_note: str = "Not independently verified by this workflow."
    label_mode: str = "athlete_any_record"  # or "final_record" (different estimand)
    drop_exact_duplicates: bool = False
    seed: int = 42
    cutoff_year: int = 2019
    min_records: int = 3
    internal_test_fraction: float = 0.20
    calibration_fraction_of_training: float = 0.20
    # Optional exact roles: key, target_source_row, partition; exact matching required.
    split_csv: Optional[str] = None
    n_folds: int = 5
    reference_population: str = "label0"  # or "all"; applies to BOTH RNN and LMM
    rnn_params: dict = field(default_factory=lambda: copy.deepcopy(RNN_SAVED_PARAMS))
    xgb_params: dict = field(default_factory=lambda: copy.deepcopy(XGB_SAVED_PARAMS))
    # Hyperparameters are frozen/transferred, NOT re-tuned by this script.
    max_epochs: int = 100
    early_stopping_patience: int = 8
    early_stopping_fraction: float = 0.10
    standardize_rnn_total: bool = True  # training-only affine scaling; documented change
    refit_rnn_on_all_reference: bool = True
    device: str = "auto"  # PyTorch auto/cpu/cuda/cuda:0
    xgb_device: str = "cpu"  # reliable default; CUDA optional and NOT silently substituted
    n_jobs: int = 2
    lmm_time: str = "competition_order"  # or "elapsed_years"
    lmm_maxiter: int = 500
    lmm_reml: bool = True
    oversample: bool = True
    calibration_C: float = 1e6  # weak L2 penalty, no class weights
    primary_threshold: str = "calibration_f1"  # or calibration_specificity95/fixed_0.5
    bootstrap: int = 1000
    calibration_bins: int = 5
    run_min5: bool = True
    run_cutoff2018: bool = True
    run_strict_year: bool = False
    make_figures: bool = True
    export_shap: bool = True
    shap_max_athletes: int = 1000
    resume: bool = True
    # Reduces epochs/folds/trees only for SOFTWARE tests, never publication results.
    smoke: bool = False

    def validate(self) -> None:
        if self.reference_population not in {"label0", "all"}:
            raise ValueError("reference_population must be label0 or all")
        if self.label_mode not in {"athlete_any_record", "final_record"}:
            raise ValueError("label_mode must be athlete_any_record or final_record")
        if self.lmm_time not in {"competition_order", "elapsed_years"}:
            raise ValueError("lmm_time must be competition_order or elapsed_years")
        if self.primary_threshold not in {"calibration_f1", "calibration_specificity95", "fixed_0.5"}:
            raise ValueError("Unsupported threshold policy")
        if self.min_records < 3 or self.n_folds < 2 or self.max_epochs < 1 or self.bootstrap < 1:
            raise ValueError("min_records>=3, n_folds>=2, max_epochs>=1, bootstrap>=1 required")
        for frac in [self.internal_test_fraction, self.calibration_fraction_of_training,
                     self.early_stopping_fraction]:
            if not 0 < frac < 0.5:
                raise ValueError("Split fractions must be between 0 and 0.5")
        if self.require_verified_ids and not (self.existing_id_col or self.linkage_map_csv):
            raise ValueError("Provide existing_id_col or a completed linkage_map_csv.")
        if self.event_date_col and not self.event_date_format:
            raise ValueError("Set event_date_format explicitly when using event_date_col.")
        if self.event_date_col and self.event_order_col:
            raise ValueError("Use event dates OR chronological order, not both.")
        if self.n_jobs < 1:
            raise ValueError("n_jobs must be >=1")


def jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return jsonable(x.tolist())
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return float(x) if np.isfinite(x) else None
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, Path):
        return str(x)
    return x


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(jsonable(obj), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def save_csv(data: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(path, index=False, encoding="utf-8-sig")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def locate_file(value: str) -> Path:
    p = Path(value).expanduser()
    candidates = [p]
    if not p.is_absolute() and "__file__" in globals():
        candidates.append(Path(__file__).resolve().parent / p)
    for q in candidates:
        if q.is_file():
            return q.resolve()
        if q.parent.is_dir():
            matches = [r for r in q.parent.iterdir() if r.is_file() and r.name.lower() == q.name.lower()]
            if len(matches) == 1:
                return matches[0].resolve()
    raise FileNotFoundError(f"파일을 찾을 수 없습니다: {value}\n현재 폴더: {Path.cwd()}")


def setup_logging(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    LOG.setLevel(logging.INFO)
    for h in list(LOG.handlers):
        h.close(); LOG.removeHandler(h)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S")
    for h in [logging.StreamHandler(sys.stdout), logging.FileHandler(root / "execution.log", encoding="utf-8")]:
        h.setFormatter(fmt); LOG.addHandler(h)
    LOG.propagate = False


def set_seed(seed: int, n_jobs: int = 2) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(n_jobs)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def software_versions() -> dict:
    out = dict(python=sys.version, platform=platform.platform())
    for name in ["numpy", "pandas", "scipy", "scikit-learn", "statsmodels", "torch",
                 "xgboost", "matplotlib", "joblib", "threadpoolctl"]:
        out[name] = importlib.metadata.version(name)
    out["cuda_available"] = torch.cuda.is_available()
    out["torch_cuda_runtime"] = torch.version.cuda
    return out




# ========================================================================
# Data preprocessing and athlete linkage
# ========================================================================

# %% 2. Raw data, full DOB, linkage audit and cohort construction
WORLD_KWS = ["OLYMPIC", "WORLD CHAMPIONSHIP", "WORLD CHAMPIONSHIPS", "WORLD CUP",
             "UNIVERSIADE", "UNIVERSIAD", "UNIVERSITY WORLD CUP", "WORLD GAMES"]
CONTINENTAL_KWS = ["ASIAN", "EUROPEAN", "OCEANIAN", "OCEANIA", "AFRICAN", "PAN-AMERICAN",
                  "PAN AMERICAN", "SOUTH AMERICAN", "CENTRAL AMERICAN", "MEDITERRANEAN",
                  "ARAB", "BALKAN", "COMMONWEALTH", "WEST ASIAN", "EAST ASIAN"]
NATIONAL_KWS = ["NATIONAL", "CHAMPIONSHIP OF", "CHAMPIONSHIPS OF", "CUP OF", "GRAND PRIX OF"]
WEIGHT_LIMITS = [61, 67, 73, 81, 96, 109]
WEIGHT_LABELS = ["<=61", "61-67", "67-73", "73-81", "81-96", "96-109", ">109"]


def normalize_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(value))).strip().casefold()


def parse_dob(value: Any, fmt: str = "auto", pivot: int = 25) -> pd.Timestamp:
    """Never interpret a 4-digit year as a complete DOB. YYMMDD zeros are preserved."""
    if pd.isna(value):
        return pd.NaT
    s = str(value).strip()
    if re.fullmatch(r"\d+\.0", s):
        s = s[:-2]
    if fmt == "YYMMDD" or (fmt == "auto" and re.fullmatch(r"\d{5,6}", s)):
        s = s.zfill(6)
        if not re.fullmatch(r"\d{6}", s):
            return pd.NaT
        yy = int(s[:2]); year = (2000 if yy <= pivot else 1900) + yy
        return pd.to_datetime(f"{year:04d}{s[2:]}", format="%Y%m%d", errors="coerce")
    if fmt != "auto":
        return pd.to_datetime(s, format=fmt, errors="coerce")
    if re.fullmatch(r"\d{8}", s):
        return pd.to_datetime(s, format="%Y%m%d", errors="coerce")
    # No generic ambiguous date parser and no guessing of MM/DD vs DD/MM.
    for f in ["%d.%m.%Y", "%Y-%m-%d", "%d/%m/%Y", "%Y/%m/%d"]:
        date = pd.to_datetime(s, format=f, errors="coerce")
        if pd.notna(date):
            return date
    return pd.NaT


def event_category(name: str) -> str:
    u = str(name).upper()
    if any(k in u for k in WORLD_KWS):
        return "world"
    if any(k in u for k in CONTINENTAL_KWS):
        return "continent"
    if any(k in u for k in NATIONAL_KWS):
        return "nation"
    return "etc"


def fixed_weight_bin(x: float) -> str:
    return WEIGHT_LABELS[int(np.searchsorted(WEIGHT_LIMITS, x, side="left"))]


def read_and_audit(cfg: Config, root: Path) -> tuple[pd.DataFrame, dict]:
    cfg.validate()
    path = locate_file(cfg.raw_csv)
    audit_dir = root / "audit"; audit_dir.mkdir(parents=True, exist_ok=True)
    # dtype=str is essential for 001116 etc.; do not strip leading zeroes.
    raw = pd.read_csv(path, dtype=str, encoding=cfg.encoding)
    required = [cfg.columns[k] for k in ["nation", "dob", "bodyweight", "total", "event", "event_year", "label"]]
    missing = [c for c in required if c not in raw]
    if missing:
        raise ValueError(f"필수 열 누락: {missing}\n실제 열: {list(raw.columns)}\nConfig.columns를 수정하세요.")
    for extra in [cfg.name_col, cfg.existing_id_col, cfg.event_date_col, cfg.event_order_col]:
        if extra and extra not in raw:
            raise ValueError(f"설정에 지정한 열이 없습니다: {extra}")
    raw["source_row"] = np.arange(len(raw)) + 2
    d = pd.DataFrame({"source_row": raw.source_row})
    d["nation"] = raw[cfg.columns["nation"]].map(normalize_text).str.upper()
    dobraw = raw[cfg.columns["dob"]]
    lookup = {v: parse_dob(v, cfg.dob_format, cfg.two_digit_year_pivot) for v in dobraw.dropna().unique()}
    d["dob"] = pd.to_datetime(dobraw.map(lookup))
    d["bornyear"] = d.dob.dt.year
    for k, col in [("total", "total"), ("bweight", "bodyweight"), ("eventyear", "event_year"), ("row_label", "label")]:
        d[k] = pd.to_numeric(raw[cfg.columns[col]], errors="coerce")
    d["event"] = raw[cfg.columns["event"]].fillna("").astype(str)
    d["event_category"] = d.event.map(event_category)
    d["age"] = d.eventyear - d.bornyear
    d["normalized_name"] = raw[cfg.name_col].map(normalize_text) if cfg.name_col else ""
    d["candidate_key"] = d.nation + "|" + d.dob.dt.strftime("%Y-%m-%d").fillna("INVALID_DOB")
    d["key"] = d.candidate_key
    identity_status = "provisional_nationality_full_DOB"
    if cfg.existing_id_col:
        ids = raw[cfg.existing_id_col].fillna("").astype(str).str.strip()
        d["key"] = "ID|" + ids
        d.loc[ids.eq(""), "key"] = ""
        identity_status = "author_supplied_existing_ID_not_independently_verified"
    if cfg.linkage_map_csv:
        m = pd.read_csv(locate_file(cfg.linkage_map_csv), dtype=str, encoding=cfg.encoding)
        if not {"source_row", "verified_athlete_id"}.issubset(m):
            raise ValueError("linkage_map_csv requires source_row and verified_athlete_id")
        m["source_row"] = pd.to_numeric(m.source_row, errors="raise").astype(int)
        if m.source_row.duplicated().any():
            raise ValueError("linkage_map_csv contains duplicated source_row values")
        ids = d.source_row.map(m.set_index("source_row").verified_athlete_id).fillna("").astype(str).str.strip()
        d["key"] = "ID|" + ids
        d.loc[ids.eq(""), "key"] = ""
        identity_status = "author_supplied_linkage_map_not_independently_verified"
    # All valid record labels contribute to athlete status BEFORE performance exclusions.
    identity_ok = d.nation.ne("") & d.dob.notna() & d.key.ne("")
    label_ok = d.row_label.isin([0, 1])
    event_ok = d.eventyear.notna() & (d.eventyear % 1 == 0) & d.eventyear.between(1900, 2100)
    performance_ok = np.isfinite(d.total) & np.isfinite(d.bweight) & (d.total > 0) & (d.bweight > 0)
    missing_key_valid_else = d.key.eq("") & d.nation.ne("") & d.dob.notna() & label_ok & event_ok & performance_ok
    if missing_key_valid_else.any() and (cfg.existing_id_col or cfg.linkage_map_csv):
        save_csv(d.loc[missing_key_valid_else], audit_dir / "UNMAPPED_ROWS.csv")
        raise ValueError("Retainable rows have missing supplied IDs. Complete audit/UNMAPPED_ROWS.csv; no partial-ID fallback is allowed.")
    status_rows = d[identity_ok & label_ok]
    statuses = status_rows.groupby("key").row_label.max()
    discordant = status_rows.groupby("key").row_label.agg(["min", "max", "count"])
    save_csv(discordant[discordant["min"] != discordant["max"]].reset_index(), audit_dir / "discordant_record_labels.csv")
    keep = identity_ok & label_ok & event_ok & performance_ok & (d.age >= 0)
    reason = pd.DataFrame({"source_row": d.source_row, "invalid_identity": ~identity_ok,
                           "invalid_label": ~label_ok, "invalid_year": ~event_ok,
                           "invalid_performance": ~performance_ok, "negative_age": d.age < 0})
    save_csv(reason.loc[~keep], audit_dir / "excluded_rows.csv")
    d = d.loc[keep].copy()
    if d.empty:
        raise ValueError("No valid records after parsing. Check DOB format and column mapping.")
    for col in ["eventyear", "bornyear", "age", "row_label"]:
        d[col] = d[col].astype(int)
    d["athlete_label"] = d.key.map(statuses).astype(int)
    if cfg.event_date_col:
        all_dates = pd.to_datetime(raw[cfg.event_date_col], format=cfg.event_date_format, errors="coerce")
        d["event_date"] = all_dates.loc[d.index]
        if d.event_date.isna().any() or not (d.event_date.dt.year == d.eventyear).all():
            raise ValueError("Invalid/missing event dates or disagreement with eventyear. No silent date fallback.")
        d["_order_value"] = d.event_date.astype("int64") / (86400 * 1e9)
        chronology = "provided_event_dates; ties_resolved_by_source_row"
    elif cfg.event_order_col:
        d["_order_value"] = pd.to_numeric(raw.loc[d.index, cfg.event_order_col], errors="coerce")
        if not np.isfinite(d._order_value).all():
            raise ValueError("event_order_col contains invalid/missing ranks")
        chronology = "provided_within_year_order; ties_resolved_by_source_row"
    else:
        d["_order_value"] = d.source_row
        chronology = "year_then_source_row; within_year_chronology_UNVERIFIED"
    year_ties = d.groupby(["key", "eventyear"]).size().reset_index(name="records_in_year")
    year_ties = year_ties[year_ties.records_in_year > 1]
    save_csv(year_ties, audit_dir / "same_year_multiple_records.csv")
    temporal_ties = d.duplicated(["key", "eventyear", "_order_value"], keep=False)
    if cfg.require_exact_chronology and ((not cfg.event_date_col and not cfg.event_order_col and len(year_ties)) or temporal_ties.any()):
        raise ValueError("Chronology is ambiguous. Supply verified event dates/order or disable require_exact_chronology explicitly.")
    original_cols = [c for c in raw.columns if c != "source_row"]
    dupe_mask = raw.loc[d.index, original_cols].duplicated(keep=False)
    save_csv(d.loc[dupe_mask], audit_dir / "exact_duplicate_candidates.csv")
    duplicate_dropped = 0
    if cfg.drop_exact_duplicates:
        first = ~raw.loc[d.index, original_cols].duplicated(keep="first")
        duplicate_dropped = int((~first).sum()); d = d.loc[first].copy()
    if cfg.columns.get("snatch") in raw and cfg.columns.get("jerk") in raw:
        d["snatch_jerk_mismatch"] = ((d.total - pd.to_numeric(raw.loc[d.index, cfg.columns["snatch"]], errors="coerce")
                                       - pd.to_numeric(raw.loc[d.index, cfg.columns["jerk"]], errors="coerce")).abs() > 0.11)
    else:
        d["snatch_jerk_mismatch"] = False
    d["gross_quality_flag"] = (d.total > 600) | ~d.bweight.between(30, 250) | ~d.age.between(10, 60)
    save_csv(d[d.gross_quality_flag | d.snatch_jerk_mismatch], audit_dir / "quality_flags_NOT_automatically_excluded.csv")
    same_event = d.duplicated(["key", "eventyear", "event"], keep=False)
    save_csv(d[same_event], audit_dir / "same_key_event_candidates.csv")
    if cfg.name_col:
        names = d.groupby("candidate_key").normalized_name.nunique()
        save_csv(d[d.candidate_key.isin(names[names > 1].index)], audit_dir / "name_discrepancies_for_manual_review.csv")
        cross_nations = d[d.normalized_name.ne("")].groupby(["normalized_name", "dob"]).nation.nunique()
        cross_nations = cross_nations[cross_nations > 1].reset_index(name="n_nationalities")
        save_csv(cross_nations, audit_dir / "possible_nationality_changes.csv")
    # Template intentionally leaves verified ID empty. Never pretend checking was done.
    template = d[["source_row", "candidate_key", "key", "nation", "dob", "normalized_name", "eventyear", "event"]].copy()
    template["verified_athlete_id"] = ""; template["review_note"] = ""
    save_csv(template, audit_dir / "linkage_review_template.csv")
    save_csv(d.groupby("key").agg(n_dob=("dob", "nunique"), n_nation=("nation", "nunique"),
                                  n_names=("normalized_name", "nunique"), n_records=("source_row", "size")).reset_index(),
             audit_dir / "key_consistency.csv")
    d = d.sort_values(["key", "eventyear", "_order_value", "source_row"], kind="mergesort").reset_index(drop=True)
    n = d.groupby("key").size(); yy = d.groupby("key").athlete_label.max()
    save_csv(pd.DataFrame([dict(minimum_records=k, n_keys=int((n >= k).sum()),
                               positive_keys=int(yy[n >= k].sum())) for k in [2, 3, 4, 5, 10]]),
             audit_dir / "minimum_record_counts.csv")
    info = dict(raw_file=str(path), raw_sha256=sha256(path), raw_rows=len(raw), retained_rows=len(d),
                retained_keys=d.key.nunique(), invalid_rows=int((~keep).sum()),
                exact_duplicate_rows_dropped=duplicate_dropped, identity_status=identity_status,
                identity_note=cfg.identity_verification_note, name_column=cfg.name_col,
                name_verification_performed_by_code=False, chronology=chronology,
                same_year_key_groups=len(year_ties), unresolved_order_tied_rows=int(temporal_ties.sum()),
                label_mode=cfg.label_mode, label_timing="retrospective; sanction/violation dates not validated",
                athlete_label_definition="maximum valid record label before performance exclusions",
                discordant_label_keys=int((discordant['min'] != discordant['max']).sum()),
                DOB_format=cfg.dob_format, YYMMDD_pivot=cfg.two_digit_year_pivot,
                weight_bins="fixed numeric cut-points, NOT historical Olympic weight classes",
                source_columns=list(raw.columns), workflow_version=VERSION)
    save_json(info, audit_dir / "data_audit.json")
    save_csv(d, audit_dir / "clean_records.csv")
    LOG.info("Data audit: rows %d -> %d; keys=%d; identity=%s", len(raw), len(d), d.key.nunique(), identity_status)
    if "provisional" in identity_status:
        LOG.warning("Full DOB+nationality keys are provisional; names/real identities have NOT been independently verified.")
    if not cfg.event_date_col and not cfg.event_order_col and len(year_ties):
        LOG.warning("%d key-year groups contain multiple records; source-row order is not verified chronology.", len(year_ties))
    return d, info


@dataclass
class AthleteSample:
    key: str
    totals: np.ndarray
    event_categories: list
    ages: list
    years: list
    lmm_history_time: np.ndarray
    lmm_target_time: float
    target: float
    target_source_row: int
    history_source_rows: np.ndarray


def build_cohort(records: pd.DataFrame, cfg: Config, minimum: int,
                 strict_earlier_year: bool = False) -> tuple[pd.DataFrame, list[AthleteSample]]:
    rows, samples = [], []
    for key, g in records.groupby("key", sort=True):
        if len(g) < minimum:
            continue
        target = g.iloc[-1]
        history = g.iloc[:-1]
        if strict_earlier_year:
            history = history[history.eventyear < target.eventyear]
            if len(history) < 2:
                continue
        if cfg.lmm_time == "competition_order":
            times = np.arange(len(history), dtype=float)
            target_time = float(len(history))
        else:
            if "event_date" in g:
                origin = history.event_date.iloc[0]
                times = (history.event_date - origin).dt.total_seconds().to_numpy() / (365.25 * 86400)
                target_time = (target.event_date - origin).total_seconds() / (365.25 * 86400)
            else:
                times = history.eventyear.to_numpy(float) - float(history.eventyear.iloc[0])
                target_time = float(target.eventyear - history.eventyear.iloc[0])
        label = int(target.athlete_label if cfg.label_mode == "athlete_any_record" else target.row_label)
        rows.append(dict(key=key, nation=target.nation, dob=target.dob.strftime("%Y-%m-%d"),
                         bornyear=int(target.bornyear), age=int(target.age), bweight=float(target.bweight),
                         eventyear=int(target.eventyear), total=float(target.total), y=label,
                         final_record_label=int(target.row_label), athlete_any_record_label=int(target.athlete_label),
                         n_records=len(g), n_history=len(history), n_history_years=history.eventyear.nunique(),
                         target_source_row=int(target.source_row), weight_bin=fixed_weight_bin(float(target.bweight)),
                         final_year_records=int((g.eventyear == target.eventyear).sum())))
        samples.append(AthleteSample(key, history.total.to_numpy(np.float32),
                                     history.event_category.astype(str).tolist(), history.age.astype(str).tolist(),
                                     history.eventyear.astype(str).tolist(), np.asarray(times, float), target_time,
                                     float(target.total), int(target.source_row), history.source_row.to_numpy(int)))
    a = pd.DataFrame(rows)
    if a.empty:
        raise ValueError("No eligible athlete samples")
    if not a.key.is_unique:
        raise AssertionError("Duplicate athlete key after cohort construction")
    for i, s in enumerate(samples):
        assert s.key == a.iloc[i].key
        assert s.target_source_row not in set(s.history_source_rows)
        assert len(s.totals) == int(a.iloc[i].n_history)
    return a, samples


def stratified_split(ids: np.ndarray, y: np.ndarray, size: float, seed: int, what: str):
    if len(ids) < 10 or len(np.unique(y[ids])) < 2 or min(np.bincount(y[ids], minlength=2)) < 2:
        raise ValueError(f"{what}: too few athletes/labels for stratified splitting; no unstratified fallback.")
    try:
        return train_test_split(ids, test_size=size, stratify=y[ids], random_state=seed)
    except ValueError as exc:
        raise ValueError(f"{what} split failed: {exc}") from exc


def assign_partitions(a: pd.DataFrame, cfg: Config, cutoff: int,
                      inherited: Optional[pd.DataFrame] = None,
                      use_external_split: bool = False) -> pd.DataFrame:
    a = a.copy(); y = a.y.to_numpy(int)
    if inherited is not None:
        roles = inherited.set_index("key").partition
        a["partition"] = a.key.map(roles)
        if a.partition.isna().any():
            raise ValueError("Sensitivity cohort contains keys not in the parent cohort")
    elif use_external_split and cfg.split_csv:
        s = pd.read_csv(locate_file(cfg.split_csv), dtype={"key": str})
        if not {"key", "partition", "target_source_row"}.issubset(s) or not s.key.is_unique:
            raise ValueError("split_csv requires unique key, partition, target_source_row")
        if set(s.key) != set(a.key):
            raise ValueError("Provided split must match ALL eligible keys exactly; no guessed ID conversion")
        check = a[["key", "target_source_row"]].merge(s, on="key", suffixes=("_new", "_saved"))
        if not (check.target_source_row_new == check.target_source_row_saved).all():
            raise ValueError("Provided split targets differ from newly constructed targets")
        a["partition"] = a.key.map(s.set_index("key").partition)
    else:
        dev = np.flatnonzero(a.eventyear.to_numpy() <= cutoff)
        ext = np.flatnonzero(a.eventyear.to_numpy() > cutoff)
        train, test = stratified_split(dev, y, cfg.internal_test_fraction, cfg.seed, "development/internal")
        fit, cal = stratified_split(train, y, cfg.calibration_fraction_of_training, cfg.seed + 101, "fitting/calibration")
        a["partition"] = ""
        for ids, name in [(fit, "fitting"), (cal, "calibration"), (test, "internal_test"), (ext, "temporal_holdout")]:
            a.loc[ids, "partition"] = name
    valid = {"fitting", "calibration", "internal_test", "temporal_holdout"}
    if not set(a.partition).issubset(valid):
        raise ValueError("Invalid partition name. Use fitting/calibration/internal_test/temporal_holdout.")
    for name in ["fitting", "calibration", "internal_test"]:
        g = a[a.partition == name]
        if g.empty or g.y.nunique() < 2:
            raise ValueError(f"{name}: both labels are required; subgroup too small. No automatic resplit.")
        if (g.eventyear > cutoff).any():
            raise ValueError(f"{name}: contains final years beyond cutoff={cutoff}")
    if (a.loc[a.partition == "temporal_holdout", "eventyear"] <= cutoff).any():
        raise ValueError("Temporal keys violate the cutoff rule")
    return a




# ========================================================================
# RNN longitudinal prediction
# ========================================================================

# %% 3. RNN with training-only encoders, early stopping and cross-fitting support
class SequenceDataset(Dataset):
    def __init__(self, samples: list[AthleteSample], indices: np.ndarray,
                 vocab: list[dict], center: float, scale: float):
        self.samples = samples; self.indices = np.asarray(indices, int)
        self.vocab = vocab; self.center = center; self.scale = scale

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        s = self.samples[int(self.indices[i])]
        total = torch.as_tensor((s.totals - self.center) / self.scale, dtype=torch.float32).reshape(-1, 1)
        fields = [s.event_categories, s.ages, s.years]
        # 0 reserved for both padding/unknown; packed sequences remove padding.
        codes = [torch.tensor([m.get(v, 0) for v in vals], dtype=torch.long)
                 for vals, m in zip(fields, self.vocab)]
        target = torch.tensor((s.target - self.center) / self.scale, dtype=torch.float32)
        return (total, *codes, target)


def sequence_collate(batch):
    lengths = torch.tensor([len(b[0]) for b in batch], dtype=torch.long)
    floats = nn.utils.rnn.pad_sequence([b[0] for b in batch], batch_first=True)
    cats = [nn.utils.rnn.pad_sequence([b[j] for b in batch], batch_first=True) for j in [1, 2, 3]]
    target = torch.stack([b[4] for b in batch])
    return (floats, *cats, lengths, target)


class RNNRegressor(nn.Module):
    def __init__(self, sizes: list[int], params: dict):
        super().__init__()
        self.emb_ec = nn.Embedding(sizes[0] + 1, params["emb_ec"], padding_idx=0)
        self.emb_age = nn.Embedding(sizes[1] + 1, params["emb_age"], padding_idx=0)
        self.emb_year = nn.Embedding(sizes[2] + 1, params["emb_year"], padding_idx=0)
        input_dim = 1 + params["emb_ec"] + params["emb_age"] + params["emb_year"]
        cls = {"lstm": nn.LSTM, "gru": nn.GRU}.get(params["rnn_type"].lower())
        if cls is None:
            raise ValueError("RNN type must be lstm or gru")
        self.kind = params["rnn_type"].lower()
        self.rnn = cls(input_dim, params["hidden"], num_layers=params["layers"], batch_first=True,
                       dropout=params["dropout"] if params["layers"] > 1 else 0.0)
        self.head = nn.Sequential(nn.Linear(params["hidden"], params["hidden"] // 2), nn.ReLU(),
                                  nn.Dropout(params["dropout"]), nn.Linear(params["hidden"] // 2, 1))

    def forward(self, totals, event_categories, ages, years, lengths):
        x = torch.cat([totals, self.emb_ec(event_categories), self.emb_age(ages), self.emb_year(years)], dim=-1)
        # lengths MUST remain on CPU even when tensors/model use CUDA.
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, state = self.rnn(packed)
        hidden = state[0] if self.kind == "lstm" else state
        return self.head(hidden[-1]).squeeze(-1)


def resolve_torch_device(cfg: Config) -> torch.device:
    name = ("cuda" if torch.cuda.is_available() else "cpu") if cfg.device == "auto" else cfg.device
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable. Install CUDA-capable PyTorch or set device='cpu'.")
    return torch.device(name)


def make_vocab_scaler(samples: list[AthleteSample], refs: np.ndarray, cfg: Config):
    vocab = []
    for f in ["event_categories", "ages", "years"]:
        vals = sorted({v for i in refs for v in getattr(samples[int(i)], f)})
        vocab.append({v: j + 1 for j, v in enumerate(vals)})
    values = np.concatenate([np.r_[samples[int(i)].totals, samples[int(i)].target] for i in refs]).astype(float)
    center = float(values.mean()) if cfg.standardize_rnn_total else 0.0
    scale = max(float(values.std()), 1e-8) if cfg.standardize_rnn_total else 1.0
    return vocab, center, scale


def make_loader(samples, ids, vocab, center, scale, cfg, shuffle=False, seed=42):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(SequenceDataset(samples, ids, vocab, center, scale),
                      batch_size=int(cfg.rnn_params["batch_size"]), shuffle=shuffle,
                      num_workers=0, collate_fn=sequence_collate, generator=generator)


@torch.no_grad()
def predict_rnn(model: RNNRegressor, samples, ids, meta, cfg: Config) -> np.ndarray:
    if not len(ids):
        return np.empty(0)
    model.eval(); device = next(model.parameters()).device; ans = []
    loader = make_loader(samples, ids, meta["vocab"], meta["center"], meta["scale"], cfg)
    for total, ec, age, yr, lengths, _target in loader:
        p = model(total.to(device), ec.to(device), age.to(device), yr.to(device), lengths)
        ans.append(p.detach().cpu().numpy().astype(float) * meta["scale"] + meta["center"])
    return np.concatenate(ans)


def train_epoch(model, loader, optimizer, device):
    model.train(); squared_loss = 0.0; n = 0
    for total, ec, age, yr, lengths, target in loader:
        optimizer.zero_grad(set_to_none=True)
        pred = model(total.to(device), ec.to(device), age.to(device), yr.to(device), lengths)
        target = target.to(device)
        loss = torch.mean((pred - target) ** 2)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite RNN loss. Check raw data and hyperparameters.")
        loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        squared_loss += float(loss.detach().cpu()) * len(target); n += len(target)
    return squared_loss / n


def fit_rnn(samples: list[AthleteSample], refs: np.ndarray, cfg: Config, seed: int,
            folder: Path, tag: str) -> tuple[RNNRegressor, dict]:
    folder.mkdir(parents=True, exist_ok=True)
    metadata_file = folder / f"{tag}_rnn.json"; weights_file = folder / f"{tag}_rnn.pt"
    device = resolve_torch_device(cfg)
    wanted_keys = [samples[int(i)].key for i in refs]
    if cfg.resume and metadata_file.exists() and weights_file.exists():
        meta = json.loads(metadata_file.read_text(encoding="utf-8"))
        if meta.get("allowed_reference_keys") != wanted_keys:
            raise ValueError("Cached RNN reference IDs disagree; use a clean output directory.")
        model = RNNRegressor([len(v) for v in meta["vocab"]], cfg.rnn_params).to(device)
        model.load_state_dict(torch.load(weights_file, map_location=device, weights_only=True))
        LOG.info("RNN %s: cached fit loaded", tag)
        return model, meta
    if len(refs) < 12:
        raise ValueError("Too few reference athletes for RNN early stopping")
    start = time.time()
    tr, val = train_test_split(refs, test_size=cfg.early_stopping_fraction, random_state=seed)
    vocab, center, scale = make_vocab_scaler(samples, tr, cfg)
    meta = dict(vocab=vocab, center=center, scale=scale)
    set_seed(seed, cfg.n_jobs)
    model = RNNRegressor([len(v) for v in vocab], cfg.rnn_params).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.rnn_params["lr"], weight_decay=cfg.rnn_params["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    loader = make_loader(samples, tr, vocab, center, scale, cfg, shuffle=True, seed=seed)
    best = math.inf; best_epoch = 0; best_state = None; stale = 0; history = []
    actual_val = np.array([samples[int(i)].target for i in val])
    for epoch in range(1, cfg.max_epochs + 1):
        used_lr = float(optimizer.param_groups[0]["lr"])
        training_mse = train_epoch(model, loader, optimizer, device) * scale**2
        p = predict_rnn(model, samples, val, meta, cfg)
        val_mse = float(np.mean((actual_val - p)**2))
        history.append(dict(phase="early_stopping", epoch=epoch, training_mse_kg2=training_mse,
                            validation_mse_kg2=val_mse, learning_rate=used_lr))
        scheduler.step(val_mse)
        if val_mse < best - 1e-9:
            best = val_mse; best_epoch = epoch; stale = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if epoch == 1 or epoch % 10 == 0:
            LOG.info("RNN %s | epoch %d/%d | validation RMSE %.3f kg", tag, epoch, cfg.max_epochs, np.sqrt(val_mse))
        if stale >= cfg.early_stopping_patience:
            break
    if best_state is None:
        raise RuntimeError("RNN produced no valid checkpoint")
    selected_log = list(history)
    if cfg.refit_rnn_on_all_reference:
        # Refit with the chosen epoch count and development-derived LR schedule.
        # This uses all allowed refs; validation NEVER includes an OOF/test athlete.
        vocab, center, scale = make_vocab_scaler(samples, refs, cfg)
        meta = dict(vocab=vocab, center=center, scale=scale)
        set_seed(seed, cfg.n_jobs)
        model = RNNRegressor([len(v) for v in vocab], cfg.rnn_params).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.rnn_params["lr"], weight_decay=cfg.rnn_params["weight_decay"])
        loader = make_loader(samples, refs, vocab, center, scale, cfg, shuffle=True, seed=seed)
        for epoch in range(1, best_epoch + 1):
            lr = selected_log[epoch - 1]["learning_rate"]
            for group in optimizer.param_groups:
                group["lr"] = lr
            mse = train_epoch(model, loader, optimizer, device) * scale**2
            history.append(dict(phase="refit_all_reference", epoch=epoch, training_mse_kg2=mse,
                                validation_mse_kg2=np.nan, learning_rate=lr))
        used_refs = refs
    else:
        model.load_state_dict(best_state)
        used_refs = tr
    meta.update(params=cfg.rnn_params, seed=seed, device=str(device), best_epoch=best_epoch,
                max_epochs=cfg.max_epochs, early_stopping_epochs=len(selected_log),
                early_stopping_best_rmse=float(np.sqrt(best)),
                allowed_reference_keys=wanted_keys,
                training_reference_keys=[samples[int(i)].key for i in used_refs],
                early_stopping_keys=[samples[int(i)].key for i in val],
                final_refit=cfg.refit_rnn_on_all_reference, hyperparameter_search_performed=False,
                seconds=time.time() - start)
    save_csv(pd.DataFrame(history), folder / f"{tag}_rnn_training.csv")
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, weights_file)
    save_json(meta, metadata_file)
    LOG.info("RNN %s done: refs=%d selected_epoch=%d; %.1f sec", tag, len(used_refs), best_epoch, time.time()-start)
    return model, meta




# ========================================================================
# Linear mixed-effects benchmark
# ========================================================================

# %% 4. Random-intercept + random-slope LMM; conditional effects from PAST only

def lmm_design(times: np.ndarray) -> np.ndarray:
    # Scaling order/year by 10 changes units, not the model family.
    return np.column_stack([np.ones(len(times)), np.asarray(times, float) / 10.0])


def conditional_random_effect(beta: np.ndarray, cov: np.ndarray, sigma2: float,
                              x_history: np.ndarray, y_history: np.ndarray) -> np.ndarray:
    """E[b|past] without inverting cov; stable for nearly singular random effects.
    X and Z are the same two columns (intercept, time) in this simple benchmark.
    """
    z = x_history
    return np.linalg.solve(np.eye(2) + cov @ (z.T @ z) / sigma2,
                           cov @ (z.T @ (np.asarray(y_history) - x_history @ beta)) / sigma2)


def predict_lmm(samples: list[AthleteSample], ids: np.ndarray, meta: dict) -> np.ndarray:
    beta = np.asarray(meta["beta"], float); cov = np.asarray(meta["random_covariance"], float)
    sigma2 = float(meta["residual_variance"])
    ans = []
    for i in ids:
        s = samples[int(i)]
        xh = lmm_design(s.lmm_history_time)
        b = conditional_random_effect(beta, cov, sigma2, xh, s.totals)
        xt = lmm_design(np.array([s.lmm_target_time]))[0]
        # s.target is intentionally NOT used to form this prediction.
        ans.append(float(xt @ beta + xt @ b))
    return np.asarray(ans, float)


def fit_lmm(samples: list[AthleteSample], refs: np.ndarray, cfg: Config,
            folder: Path, tag: str) -> dict:
    file = folder / f"{tag}_lmm.json"
    wanted_keys = [samples[int(i)].key for i in refs]
    if cfg.resume and file.exists():
        meta = json.loads(file.read_text(encoding="utf-8"))
        if meta.get("reference_keys") != wanted_keys:
            raise ValueError("Cached LMM reference keys differ")
        return meta
    start = time.time(); designs=[]; totals=[]; groups=[]
    for i in refs:
        s = samples[int(i)]
        # Population fitting may use the final outcome of TRAINING refs only.
        times = np.r_[s.lmm_history_time, s.lmm_target_time]
        designs.append(lmm_design(times)); totals.append(np.r_[s.totals, s.target])
        groups.extend([s.key] * len(times))
    x = np.vstack(designs); yy = np.concatenate(totals).astype(float); groups = np.asarray(groups)
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        model = sm.MixedLM(yy, x, groups=groups, exog_re=x)
        result = model.fit(reml=cfg.lmm_reml, method=["lbfgs", "bfgs", "cg"],
                           maxiter=cfg.lmm_maxiter, disp=False)
    beta = np.asarray(result.fe_params); cov = np.asarray(result.cov_re); sigma2 = float(result.scale)
    warning_text = [str(w.message) for w in captured]
    diagnostic = dict(converged=bool(result.converged), warnings=warning_text,
                      seconds=time.time()-start, n_reference=len(refs), n_records=len(yy))
    save_json(diagnostic, folder / f"{tag}_lmm_diagnostics.json")
    if not result.converged or not np.isfinite(beta).all() or not np.isfinite(cov).all() or not np.isfinite(sigma2) or sigma2 <= 0:
        raise RuntimeError(f"LMM {tag} did not converge to a usable solution. No simpler model silently substituted. See {folder}.")
    eig = np.linalg.eigvalsh(cov)
    if eig.min() < -1e-7:
        raise RuntimeError("LMM estimated a non-positive-semidefinite random covariance")
    # Check our conditional prediction calculation against fitted statsmodels BLUPs.
    errors=[]
    if eig.min() > 1e-10:
        for key in wanted_keys[:5]:
            ii = groups == key
            estimated = conditional_random_effect(beta, cov, sigma2, x[ii], yy[ii])
            errors.append(float(np.max(np.abs(estimated - np.asarray(result.random_effects[key])))))
        if errors and max(errors) > 1e-5:
            raise AssertionError("Conditional random-effect calculation failed the statsmodels check")
    meta = dict(beta=beta, random_covariance=cov, residual_variance=sigma2,
                time=cfg.lmm_time, time_scale=10.0, reml=cfg.lmm_reml, reference_keys=wanted_keys,
                covariance_eigenvalues=eig, near_singular=bool(eig.min() < max(eig.max(), 1.0) * 1e-6),
                max_blup_check_error=max(errors) if errors else None,
                formula="total ~ time + (time | athlete)", **diagnostic)
    save_json(meta, file)
    LOG.info("LMM %s done: refs=%d; converged=%s; %.1f sec", tag, len(refs), result.converged, time.time()-start)
    return jsonable(meta)


def reference_ids(a: pd.DataFrame, ids: np.ndarray, cfg: Config) -> np.ndarray:
    return ids[a.loc[ids, "y"].to_numpy() == 0] if cfg.reference_population == "label0" else ids


def fit_longitudinal_models(a: pd.DataFrame, samples: list[AthleteSample], cfg: Config,
                            folder: Path) -> pd.DataFrame:
    folder.mkdir(parents=True, exist_ok=True)
    a = a.copy(); y = a.y.to_numpy(int)
    fit = np.flatnonzero(a.partition.to_numpy() == "fitting")
    evaluation = np.flatnonzero(a.partition.to_numpy() != "fitting")
    if np.bincount(y[fit], minlength=2).min() < cfg.n_folds:
        raise ValueError("Too few fitting labels for n_folds. Reduce n_folds explicitly, not automatically.")
    a["rnn_prediction"] = np.nan; a["lmm_prediction"] = np.nan; a["residual_fold"] = -1
    splits = list(StratifiedKFold(cfg.n_folds, shuffle=True, random_state=cfg.seed).split(fit, y[fit]))
    tasks = [(f"fold{k+1}", fit[tr], fit[va], k+1, cfg.seed+1000+k) for k, (tr, va) in enumerate(splits)]
    tasks.append(("final", fit, evaluation, 0, cfg.seed+2000))
    trace=[]; unknown=[]
    for tag, fitting_ids, prediction_ids, fold, seed in tasks:
        refs = reference_ids(a, fitting_ids, cfg)
        assert not set(refs) & set(prediction_ids)
        if any(a.iloc[refs].eventyear > a.loc[a.partition != "temporal_holdout", "eventyear"].max()):
            raise AssertionError("Unexpected future reference")
        model, rmeta = fit_rnn(samples, refs, cfg, seed, folder, tag)
        # Match reference observations exactly when RNN refitting is disabled.
        ref_set = set(rmeta["training_reference_keys"])
        lmm_refs = np.array([i for i in refs if samples[int(i)].key in ref_set], int)
        lmeta = fit_lmm(samples, lmm_refs, cfg, folder, tag)
        held_keys = set(a.iloc[prediction_ids].key)
        assert not held_keys & set(rmeta["allowed_reference_keys"])
        assert not held_keys & set(lmeta["reference_keys"])
        a.loc[prediction_ids, "rnn_prediction"] = predict_rnn(model, samples, prediction_ids, rmeta, cfg)
        a.loc[prediction_ids, "lmm_prediction"] = predict_lmm(samples, prediction_ids, lmeta)
        a.loc[prediction_ids, "residual_fold"] = fold
        for i in prediction_ids:
            trace.append(dict(key=a.iloc[i].key, target_source_row=int(a.iloc[i].target_source_row),
                              partition=a.iloc[i].partition, prediction_model=tag,
                              n_reference_keys=len(refs), no_reference_key_overlap=True))
        if tag == "final":
            for ds in ["calibration", "internal_test", "temporal_holdout"]:
                ii = np.flatnonzero(a.partition.to_numpy() == ds)
                for f, v in zip(["event_categories", "ages", "years"], rmeta["vocab"]):
                    values = [z for i in ii for z in getattr(samples[int(i)], f)]
                    unk = sum(z not in v for z in values)
                    unknown.append(dict(dataset=ds, feature=f, tokens=len(values), unknown=unk,
                                        unknown_fraction=unk/len(values) if values else np.nan))
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if not np.isfinite(a[["rnn_prediction", "lmm_prediction"]].to_numpy()).all():
        raise AssertionError("Missing/non-finite out-of-sample predictions")
    a["rnn_residual"] = a.total - a.rnn_prediction
    a["lmm_residual"] = a.total - a.lmm_prediction
    save_csv(pd.DataFrame(trace), folder.parent / "prediction_provenance.csv")
    save_csv(pd.DataFrame(unknown), folder.parent / "embedding_unknown_rates.csv")
    save_csv(a, folder.parent / "athlete_predictions.csv")
    return a




# ========================================================================
# Metrics and calibration
# ========================================================================

# %% 5. Metrics, calibration diagnostics and conditional uncertainty

def exact_binomial_interval(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return np.nan, np.nan
    low = 0.0 if k == 0 else float(beta_distribution.ppf(0.025, k, n-k+1))
    high = 1.0 if k == n else float(beta_distribution.ppf(0.975, k+1, n-k))
    return low, high


def regression_scores(y, p) -> dict:
    y=np.asarray(y, float); p=np.asarray(p, float)
    if not len(y):
        return dict(r2=np.nan, mae=np.nan, rmse=np.nan)
    mse = float(np.mean((y-p)**2)); variance = float(np.mean((y-y.mean())**2))
    return dict(r2=1.0-mse/variance if variance > 0 else np.nan,
                mae=float(np.mean(np.abs(y-p))), rmse=float(np.sqrt(mse)))


def binary_scores(y, p) -> dict:
    """Tied-score AP/ROC calculations checked against scikit-learn in self_test()."""
    y=np.asarray(y, int); p=np.asarray(p, float); n=len(y)
    if not n:
        return dict(roc_auc=np.nan, average_precision=np.nan, pr_auc_trapezoid=np.nan,
                    brier=np.nan, log_loss=np.nan)
    npos=int(y.sum()); nneg=n-npos
    roc = float((rankdata(p)[y == 1].sum()-npos*(npos+1)/2)/(npos*nneg)) if npos and nneg else np.nan
    if npos:
        order=np.argsort(-p, kind="stable"); ps=p[order]; ys=y[order]
        ends=np.r_[np.flatnonzero(np.diff(ps) != 0), n-1]
        tp=np.cumsum(ys)[ends]; precision=tp/(ends+1); recall=tp/npos
        ap=float(np.sum(np.diff(np.r_[0.,recall])*precision))
        # Trapezoid PR area is reported separately, not mislabeled as AP.
        trap=float(np.sum(np.diff(np.r_[0.,recall]) * (np.r_[1.,precision][1:]+np.r_[1.,precision][:-1])/2))
    else:
        ap=trap=np.nan
    pc=np.clip(p,1e-12,1-1e-12)
    return dict(roc_auc=roc, average_precision=ap, pr_auc_trapezoid=trap,
                brier=float(np.mean((y-p)**2)),
                log_loss=float(-np.mean(y*np.log(pc)+(1-y)*np.log1p(-pc))))


def classification_scores(y, p, threshold: float) -> dict:
    y=np.asarray(y,int); p=np.asarray(p,float)
    pred=p>=threshold
    tp=int(np.sum(pred & (y==1))); fp=int(np.sum(pred & (y==0)))
    fn=int(np.sum(~pred & (y==1))); tn=int(np.sum(~pred & (y==0)))
    n=len(y)
    d=dict(n=n, positive=int(y.sum()), prevalence=float(y.mean()), threshold=float(threshold),
           tn=tn, fp=fp, fn=fn, tp=tp, selected=tp+fp, selected_fraction=(tp+fp)/n,
           mean_prediction=float(p.mean()), observed_expected=float(y.sum()/p.sum()) if p.sum() else np.nan,
           f1=2*tp/(2*tp+fp+fn) if (2*tp+fp+fn) else 0.0, **binary_scores(y,p))
    for name,k,total in [("sensitivity",tp,tp+fn),("specificity",tn,tn+fp),
                         ("ppv",tp,tp+fp),("npv",tn,tn+fn),("fpr",fp,fp+tn)]:
        d[name]=k/total if total else np.nan
        d[name+"_low"],d[name+"_high"]=exact_binomial_interval(k,total)
    return d


def attach_percentiles(row: dict, values: list[dict], names: list[str]) -> dict:
    for name in names:
        v=np.asarray([r[name] for r in values],float); v=v[np.isfinite(v)]
        row[name+"_bootstrap_valid"]=len(v)
        row[name+"_low"],row[name+"_high"]=(np.quantile(v,[.025,.975]) if len(v) else (np.nan,np.nan))
    return row


def regression_interval(y, p, B: int, seed: int) -> dict:
    y=np.asarray(y,float); p=np.asarray(p,float); row=regression_scores(y,p)
    if len(y)<2:
        return attach_percentiles(row,[],["r2","mae","rmse"])
    rng=np.random.default_rng(seed); values=[]
    for _ in range(B):
        ii=rng.integers(0,len(y),len(y)); values.append(regression_scores(y[ii],p[ii]))
    return attach_percentiles(row,values,["r2","mae","rmse"])


def classification_interval(y, p, B: int, seed: int) -> dict:
    y=np.asarray(y,int); p=np.asarray(p,float); rng=np.random.default_rng(seed)
    pos=np.flatnonzero(y==1); neg=np.flatnonzero(y==0); rankrows=[]; probrows=[]
    for _ in range(B):
        if len(pos) and len(neg):
            ii=np.r_[rng.choice(pos,len(pos),replace=True),rng.choice(neg,len(neg),replace=True)]
            rankrows.append(binary_scores(y[ii],p[ii]))
        ii=rng.integers(0,len(y),len(y)); pp=p[ii]; yy=y[ii]
        probrows.append(dict(brier=float(np.mean((yy-pp)**2))))
    out=attach_percentiles({},rankrows,["roc_auc","average_precision","pr_auc_trapezoid"])
    attach_percentiles(out,probrows,["brier"])
    out["rank_ci_method"]="label-stratified athlete bootstrap; conditional on observed prevalence"
    out["brier_ci_method"]="ordinary athlete bootstrap"
    return out


def calibration_diagnostics(y, p) -> dict:
    y=np.asarray(y,int); p=np.asarray(p,float); z=logit(np.clip(p,1e-8,1-1e-8))
    out=dict(n=len(y), positive=int(y.sum()), observed_prevalence=float(y.mean()),
             mean_prediction=float(p.mean()), observed_expected=float(y.sum()/p.sum()) if p.sum() else np.nan,
             low_event_warning=bool(min(int(y.sum()),int((1-y).sum()))<20))
    for key in ["joint_intercept","calibration_slope","calibration_in_the_large"]:
        out[key]=out[key+"_low"]=out[key+"_high"]=np.nan
    if len(np.unique(y))<2 or len(np.unique(z))<2:
        out["status"]="not_estimable_single_class_or_constant_predictions"
        return out
    messages=[]
    for kind in ["joint","offset"]:
        try:
            with warnings.catch_warnings(record=True) as ws:
                warnings.simplefilter("always")
                if kind=="joint":
                    fit=sm.GLM(y, np.column_stack([np.ones(len(y)),z]), family=sm.families.Binomial()).fit(maxiter=200)
                    keys=["joint_intercept","calibration_slope"]
                else:
                    fit=sm.GLM(y,np.ones((len(y),1)),offset=z,family=sm.families.Binomial()).fit(maxiter=200)
                    keys=["calibration_in_the_large"]
            messages.extend(str(w.message) for w in ws)
            params=np.asarray(fit.params); ci=np.asarray(fit.conf_int())
            unstable=(not fit.converged or not np.isfinite(params).all() or not np.isfinite(ci).all()
                      or any("separation" in str(w.message).lower() for w in ws))
            if unstable:
                messages.append(kind+": unstable/separated GLM; values not reported")
                continue
            for j,key in enumerate(keys):
                out[key]=float(params[j]); out[key+"_low"]=float(ci[j,0]); out[key+"_high"]=float(ci[j,1])
        except Exception as exc:
            messages.append(kind+": "+str(exc))
    out["status"]="; ".join(messages) if messages else "estimated; Wald_CIs_may_be_unreliable_with_few_events"
    # joint_intercept != calibration-in-the-large (the latter fixes slope at 1).
    return out


def calibration_bins(y,p,n_bins: int) -> pd.DataFrame:
    y=np.asarray(y,int); p=np.asarray(p,float)
    edges=np.unique(np.quantile(p,np.linspace(0,1,n_bins+1)))
    assigned=np.digitize(p,edges[1:-1],right=True) if len(edges)>1 else np.zeros(len(p),int)
    rows=[]
    for j in np.unique(assigned):
        ii=assigned==j; n=int(ii.sum()); k=int(y[ii].sum()); lo,hi=exact_binomial_interval(k,n)
        rows.append(dict(bin=int(j+1), n=n, positive=k, mean_prediction=float(p[ii].mean()),
                         min_prediction=float(p[ii].min()), max_prediction=float(p[ii].max()),
                         observed=k/n, observed_low=lo, observed_high=hi))
    return pd.DataFrame(rows)


def select_thresholds(y,p) -> dict:
    """Uses calibration outcomes ONLY; a threshold beyond max selects nobody."""
    y=np.asarray(y,int); p=np.asarray(p,float)
    values=np.r_[np.unique(p),np.nextafter(p.max(),np.inf)]
    candidates=[]
    for t in values:
        pred=p>=t; tp=int(np.sum(pred&(y==1))); fp=int(np.sum(pred&(y==0)))
        fn=int(np.sum(~pred&(y==1))); tn=int(np.sum(~pred&(y==0)))
        f1=2*tp/(2*tp+fp+fn) if (2*tp+fp+fn) else 0.
        sens=tp/(tp+fn); spec=tn/(tn+fp)
        candidates.append((float(t),f1,sens,spec))
    # Deterministic tie-break: higher specificity, then higher threshold.
    best=max(candidates,key=lambda x:(x[1],x[3],x[0]))
    highspec=max([x for x in candidates if x[3]>=.95-1e-12],key=lambda x:(x[2],x[3],x[0]))
    return dict(fixed_0_5=.5,calibration_f1=best[0],calibration_specificity95=highspec[0])


def paired_difference(y, p_reference, p_comparison, metric_type: str, B: int, seed: int) -> dict:
    y=np.asarray(y); ref=np.asarray(p_reference,float); comp=np.asarray(p_comparison,float)
    scoring=regression_scores if metric_type=="regression" else binary_scores
    names=["r2","mae","rmse"] if metric_type=="regression" else ["roc_auc","average_precision","brier"]
    r=scoring(y,ref); c=scoring(y,comp); out={k:c[k]-r[k] for k in names}
    rng=np.random.default_rng(seed); values=[]
    pos=np.flatnonzero(y==1); neg=np.flatnonzero(y==0)
    for _ in range(B):
        if metric_type=="classification" and len(pos) and len(neg):
            ii=np.r_[rng.choice(pos,len(pos),replace=True),rng.choice(neg,len(neg),replace=True)]
        else:
            ii=rng.integers(0,len(y),len(y))
        rs=scoring(y[ii],ref[ii]); cs=scoring(y[ii],comp[ii]); d={k:cs[k]-rs[k] for k in names}
        if metric_type=="classification":
            ordinary=rng.integers(0,len(y),len(y))
            d["brier"]=float(np.mean((y[ordinary]-comp[ordinary])**2)-np.mean((y[ordinary]-ref[ordinary])**2))
        values.append(d)
    return attach_percentiles(out,values,names)


def evaluate_regression(a: pd.DataFrame, cfg: Config, folder: Path) -> None:
    rows=[]; paired=[]
    for dataset in ["internal_test","temporal_holdout"]:
        for population in ["all","label0"]:
            g=a[a.partition==dataset]
            if population=="label0":
                g=g[g.y==0]
            if g.empty:
                continue
            for model in ["RNN","LMM"]:
                stats=regression_interval(g.total,g[model.lower()+"_prediction"],cfg.bootstrap,cfg.seed+301)
                rows.append(dict(dataset=dataset,population=population,model=model,n=len(g),positive=int(g.y.sum()),**stats))
            delta=paired_difference(g.total,g.rnn_prediction,g.lmm_prediction,"regression",cfg.bootstrap,cfg.seed+301)
            paired.append(dict(dataset=dataset,population=population,reference="RNN",comparison="LMM",
                               direction="LMM minus RNN",n=len(g),**delta))
    save_csv(pd.DataFrame(rows),folder/"regression_metrics.csv")
    save_csv(pd.DataFrame(paired),folder/"regression_paired_differences.csv")
    lengthrows=[]
    for dataset in ["internal_test","temporal_holdout"]:
        for label,minimum,maximum in [("3-4",3,4),("5-9",5,9),("10+",10,np.inf)]:
            g=a[(a.partition==dataset)&(a.n_records>=minimum)&(a.n_records<=maximum)]
            if g.empty:
                continue
            for model in ["RNN","LMM"]:
                lengthrows.append(dict(dataset=dataset,records_group=label,model=model,n=len(g),
                                       **regression_scores(g.total,g[model.lower()+"_prediction"])))
    save_csv(pd.DataFrame(lengthrows),folder/"regression_by_sequence_length.csv")




# ========================================================================
# XGBoost classification
# ========================================================================

# ============================================================
# 6. XGBoost + PR + Calibration + Held-out evaluation
# Windows / older Python / older sklearn compatible version
# ============================================================

# ------------------------------------------------------------
# Compatibility helper: OneHotEncoder
# ------------------------------------------------------------
def make_onehot_encoder():
    try:
        # sklearn >= 1.2
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        )
    except TypeError:
        # sklearn < 1.2
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse=False
        )


# ------------------------------------------------------------
# Models to evaluate
# ------------------------------------------------------------
def model_specs():
    return [
        ("RNN_full", "rnn_residual", True),
        ("RNN_no_nationality", "rnn_residual", False),
        ("no_residual", None, True),
        ("LMM_residual", "lmm_residual", True),
    ]


# ------------------------------------------------------------
# Fit XGBoost classifier
# ------------------------------------------------------------
def fit_classifier(a, fitting, residual, include_nationality, cfg):

    numeric = [
        "bweight",
        "age",
        "bornyear",
        "eventyear",
        "total"
    ]

    if residual is not None:
        numeric.append(residual)

    categorical = []

    if include_nationality:
        categorical.append("nation")

    categorical.append("weight_bin")

    prep = ColumnTransformer(
        [
            (
                "numeric",
                StandardScaler(),
                numeric
            ),
            (
                "categorical",
                make_onehot_encoder(),
                categorical
            )
        ],
        verbose_feature_names_out=False
    )

    # fitting partition ONLY
    x = np.asarray(
        prep.fit_transform(a.iloc[fitting]),
        dtype=np.float32
    )

    y = a.iloc[fitting].y.to_numpy(dtype=int)

    # --------------------------------------------------------
    # Random oversampling inside fitting partition ONLY
    # --------------------------------------------------------
    indices = np.arange(len(y))

    if cfg.oversample:

        rng = np.random.default_rng(cfg.seed)

        counts = np.bincount(
            y,
            minlength=2
        )

        maximum = int(counts.max())

        additions = []

        for value in [0, 1]:

            pool = np.flatnonzero(y == value)

            if len(pool) == 0:
                raise ValueError(
                    "Classifier fitting requires both labels."
                )

            if len(pool) < maximum:

                sampled = rng.choice(
                    pool,
                    size=maximum - len(pool),
                    replace=True
                )

                additions.append(sampled)

        if len(additions) > 0:
            # Compatible replacement for:
            # np.r_[indices, *additions]
            indices = np.concatenate(
                [indices] + additions
            )

    # --------------------------------------------------------
    # XGBoost
    # --------------------------------------------------------
    model_kwargs = dict(
        objective="binary:logistic",
        tree_method="hist",
        eval_metric="logloss",
        random_state=cfg.seed,
        n_jobs=cfg.n_jobs
    )

    # XGBoost >= 2 supports device=
    # If unavailable, retry without it.
    try:

        model = xgb.XGBClassifier(
            device=cfg.xgb_device,
            **model_kwargs,
            **cfg.xgb_params
        )

        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always")

            model.fit(
                x[indices],
                y[indices]
            )

    except (TypeError, ValueError) as e:

        LOG.warning(
            "XGBoost device argument was not accepted (%s). "
            "Retrying without explicit device argument.",
            str(e)
        )

        model = xgb.XGBClassifier(
            **model_kwargs,
            **cfg.xgb_params
        )

        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always")

            model.fit(
                x[indices],
                y[indices]
            )

    warning_text = [
        str(w.message)
        for w in ws
    ]

    metadata = dict(
        numeric_features=numeric,
        categorical_features=categorical,
        encoded_features=list(
            prep.get_feature_names_out()
        ),
        n_before_resampling=len(y),
        n_after_resampling=len(indices),
        positives_before=int(y.sum()),
        positives_after=int(
            y[indices].sum()
        ),
        hyperparameter_search_performed=False,
        params=cfg.xgb_params,
        warnings=warning_text
    )

    return prep, model, metadata


# ------------------------------------------------------------
# Get raw XGBoost margin
# ------------------------------------------------------------
def classifier_margin(prep, model, a):

    if a.empty:
        return np.empty(0)

    x = np.asarray(
        prep.transform(a),
        dtype=np.float32
    )

    dm = xgb.DMatrix(x)

    margin = model.get_booster().predict(
        dm,
        output_margin=True
    )

    return np.asarray(
        margin,
        dtype=float
    )


# ------------------------------------------------------------
# Sigmoid calibration
# ------------------------------------------------------------
def fit_sigmoid(margin, y, cfg):

    margin = np.asarray(
        margin,
        dtype=float
    )

    y = np.asarray(
        y,
        dtype=int
    )

    if len(np.unique(y)) < 2:
        raise ValueError(
            "Calibration partition needs both labels. "
            "Do not use test labels to fix this."
        )

    cal = LogisticRegression(
        C=cfg.calibration_C,
        solver="lbfgs",
        max_iter=5000,
        tol=1e-10
    )

    with warnings.catch_warnings(record=True) as ws:

        warnings.simplefilter("always")

        cal.fit(
            margin.reshape(-1, 1),
            y
        )

    slope = float(
        cal.coef_[0, 0]
    )

    intercept = float(
        cal.intercept_[0]
    )

    if int(cal.n_iter_[0]) >= 5000:
        raise RuntimeError(
            "Sigmoid calibration did not converge."
        )

    if slope <= 0:
        LOG.warning(
            "Calibration slope <= 0. "
            "Inspect calibration data."
        )

    return dict(
        intercept=intercept,
        slope=slope,
        n=len(y),
        positive=int(y.sum()),
        C=cfg.calibration_C,
        source_partition="calibration",
        input="raw_XGBoost_margin",
        sample_weight="none",
        class_weight="none",
        warnings=[
            str(w.message)
            for w in ws
        ]
    )


def calibrated_probability(margin, cal):

    margin = np.asarray(
        margin,
        dtype=float
    )

    return expit(
        cal["intercept"]
        + cal["slope"] * margin
    )


# ------------------------------------------------------------
# SHAP
# ------------------------------------------------------------
def export_grouped_shap(
    prep,
    model,
    g,
    metadata,
    cfg,
    folder,
    name
):

    if len(g) > cfg.shap_max_athletes:

        g = (
            g.sample(
                cfg.shap_max_athletes,
                random_state=cfg.seed
            )
            .sort_index()
        )

    x = np.asarray(
        prep.transform(g),
        dtype=np.float32
    )

    matrix = xgb.DMatrix(x)

    contributions = np.asarray(
        model.get_booster().predict(
            matrix,
            pred_contribs=True
        ),
        dtype=float
    )

    margins = np.asarray(
        model.get_booster().predict(
            matrix,
            output_margin=True
        ),
        dtype=float
    )

    if (
        contributions.ndim != 2
        or contributions.shape[1] != x.shape[1] + 1
    ):
        raise ValueError(
            "Unexpected TreeSHAP shape."
        )

    error = float(
        np.max(
            np.abs(
                contributions.sum(axis=1)
                - margins
            )
        )
    )

    if error > 1e-3:
        raise AssertionError(
            "TreeSHAP additivity error too high: "
            + str(error)
        )

    numeric = metadata[
        "numeric_features"
    ]

    cats = metadata[
        "categorical_features"
    ]

    groups = {}

    for i, feature in enumerate(numeric):
        groups[feature] = [i]

    offset = len(numeric)

    encoder = prep.named_transformers_[
        "categorical"
    ]

    for feature, levels in zip(
        cats,
        encoder.categories_
    ):

        groups[feature] = list(
            range(
                offset,
                offset + len(levels)
            )
        )

        offset += len(levels)

    if offset != x.shape[1]:

        raise AssertionError(
            "One-hot SHAP grouping index mismatch."
        )

    grouped_dict = {}

    for feature, idx in groups.items():

        grouped_dict[feature] = (
            contributions[:, idx]
            .sum(axis=1)
        )

    grouped = pd.DataFrame(
        grouped_dict
    )

    grouped.insert(
        0,
        "key",
        g.key.to_numpy()
    )

    grouped["bias"] = (
        contributions[:, -1]
    )

    grouped["raw_margin"] = margins

    save_csv(
        grouped,
        folder /
        ("shap_grouped_" + name + ".csv")
    )

    raw = pd.DataFrame(
        contributions[:, :-1],
        columns=metadata[
            "encoded_features"
        ]
    )

    raw.insert(
        0,
        "key",
        g.key.to_numpy()
    )

    save_csv(
        raw,
        folder /
        ("shap_onehot_" + name + ".csv")
    )

    importance_rows = []

    for f in groups:

        importance_rows.append(
            dict(
                feature=f,
                mean_absolute_group_shap=float(
                    np.abs(
                        grouped[f]
                    ).mean()
                )
            )
        )

    importance = pd.DataFrame(
        importance_rows
    )

    importance = (
        importance
        .sort_values(
            "mean_absolute_group_shap",
            ascending=False
        )
    )

    save_csv(
        importance,
        folder /
        ("shap_importance_" + name + ".csv")
    )

    save_json(
        dict(
            method=(
                "XGBoost TreeSHAP pred_contribs; "
                "raw log-odds"
            ),
            signed_dummy_contributions_summed_before_absolute_value=True,
            max_additivity_error=error,
            n=len(g),
            explains=(
                "raw classifier; "
                "NOT calibrated probabilities"
            )
        ),
        folder /
        ("shap_metadata_" + name + ".json")
    )


# ------------------------------------------------------------
# Descriptive nationality diagnostics
# ------------------------------------------------------------
def nationality_diagnostics(
    g,
    p,
    t,
    name,
    version,
    dataset
):

    temp = g[
        ["nation", "y"]
    ].copy()

    temp["probability"] = p

    rows = []

    for nation, s in temp.groupby(
        "nation"
    ):

        stats = classification_scores(
            s.y,
            s.probability,
            t
        )

        row = dict(
            model=name,
            version=version,
            dataset=dataset,
            nation=nation,
            exploratory_only=True,
            small_group=bool(
                len(s) < 20
            )
        )

        row.update(stats)

        rows.append(row)

    return rows


# ------------------------------------------------------------
# Main classifier evaluation
# ------------------------------------------------------------
def evaluate_classifiers(
    a,
    cfg,
    folder
):

    fitting = np.flatnonzero(
        a.partition.to_numpy()
        == "fitting"
    )

    calibration = a[
        a.partition == "calibration"
    ].copy()

    ycal = calibration.y.to_numpy(
        dtype=int
    )

    prevalence_fit = float(
        a.iloc[fitting].y.mean()
    )

    predictions = []
    metrics = []
    calibrations = []
    bins = []
    thresholds = []
    fairness = []

    model_folder = folder / "models"

    model_folder.mkdir(
        exist_ok=True
    )

    # --------------------------------------------------------
    # Each model
    # --------------------------------------------------------
    for (
        name,
        residual,
        include_nationality
    ) in model_specs():

        LOG.info(
            "Classification: %s",
            name
        )

        prep, model, meta = fit_classifier(
            a,
            fitting,
            residual,
            include_nationality,
            cfg
        )

        joblib.dump(
            prep,
            model_folder /
            (name + "_preprocessor.joblib")
        )

        model.save_model(
            model_folder /
            (name + ".json")
        )

        save_json(
            meta,
            model_folder /
            (
                name
                + "_features_and_fit.json"
            )
        )

        # ----------------------------------------------------
        # Calibration partition
        # ----------------------------------------------------
        margins_cal = classifier_margin(
            prep,
            model,
            calibration
        )

        calibrator = fit_sigmoid(
            margins_cal,
            ycal,
            cfg
        )

        save_json(
            calibrator,
            model_folder /
            (name + "_sigmoid.json")
        )

        margins_by_dataset = {}

        for ds in [
            "internal_test",
            "temporal_holdout"
        ]:

            margins_by_dataset[ds] = (
                classifier_margin(
                    prep,
                    model,
                    a[
                        a.partition == ds
                    ]
                )
            )

        # ----------------------------------------------------
        # Raw + calibrated versions
        # ----------------------------------------------------
        for version in [
            "raw",
            "sigmoid"
        ]:

            if version == "raw":

                pcal = expit(
                    margins_cal
                )

            else:

                pcal = calibrated_probability(
                    margins_cal,
                    calibrator
                )

            ts = select_thresholds(
                ycal,
                pcal
            )

            # Compatibility with threshold function
            if (
                "fixed_0_5" in ts
                and "fixed_0.5"
                not in ts
            ):
                ts["fixed_0.5"] = (
                    ts.pop(
                        "fixed_0_5"
                    )
                )

            for policy, t in ts.items():

                thresholds.append(
                    dict(
                        model=name,
                        version=version,
                        policy=policy,
                        threshold=t,
                        calibration_n=len(
                            ycal
                        ),
                        calibration_positive=int(
                            ycal.sum()
                        ),
                        selection_data=(
                            "calibration only; "
                            "no test labels"
                        ),
                        primary=(
                            policy
                            == cfg.primary_threshold
                        )
                    )
                )

            predictions.append(
                pd.DataFrame(
                    dict(
                        key=calibration.key.to_numpy(),
                        target_source_row=(
                            calibration
                            .target_source_row
                            .to_numpy()
                        ),
                        dataset=(
                            "calibration_"
                            "DEVELOPMENT_ONLY"
                        ),
                        model=name,
                        version=version,
                        y=ycal,
                        margin=margins_cal,
                        probability=pcal
                    )
                )
            )

            # ------------------------------------------------
            # Held-out datasets
            # ------------------------------------------------
            for (
                dataset,
                margin
            ) in margins_by_dataset.items():

                g = a[
                    a.partition
                    == dataset
                ].copy()

                if g.empty:
                    continue

                y = g.y.to_numpy(
                    dtype=int
                )

                if version == "raw":

                    p = expit(
                        margin
                    )

                else:

                    p = calibrated_probability(
                        margin,
                        calibrator
                    )

                interval = (
                    classification_interval(
                        y,
                        p,
                        cfg.bootstrap,
                        cfg.seed + 401
                    )
                )

                diag = (
                    calibration_diagnostics(
                        y,
                        p
                    )
                )

                cal_row = dict(
                    dataset=dataset,
                    model=name,
                    version=version
                )

                cal_row.update(
                    diag
                )

                calibrations.append(
                    cal_row
                )

                b = calibration_bins(
                    y,
                    p,
                    cfg.calibration_bins
                )

                b.insert(
                    0,
                    "version",
                    version
                )

                b.insert(
                    0,
                    "model",
                    name
                )

                b.insert(
                    0,
                    "dataset",
                    dataset
                )

                bins.append(b)

                predictions.append(
                    pd.DataFrame(
                        dict(
                            key=g.key.to_numpy(),
                            target_source_row=(
                                g
                                .target_source_row
                                .to_numpy()
                            ),
                            dataset=dataset,
                            model=name,
                            version=version,
                            y=y,
                            margin=margin,
                            probability=p
                        )
                    )
                )

                # --------------------------------------------
                # Threshold-dependent metrics
                # --------------------------------------------
                for policy, t in ts.items():

                    stats = (
                        classification_scores(
                            y,
                            p,
                            t
                        )
                    )

                    null = float(
                        np.mean(
                            (
                                y
                                - prevalence_fit
                            ) ** 2
                        )
                    )

                    row = dict(
                        dataset=dataset,
                        model=name,
                        version=version,
                        policy=policy,
                        is_primary_policy=(
                            policy
                            == cfg.primary_threshold
                        ),
                        brier_null_training_prevalence=null,
                        test_prevalence_only_brier=float(
                            y.mean()
                            * (
                                1
                                - y.mean()
                            )
                        ),
                        low_event_warning=bool(
                            min(
                                y.sum(),
                                len(y)
                                - y.sum()
                            )
                            < 20
                        )
                    )

                    if null > 0:

                        row[
                            "brier_skill_vs_training_prevalence"
                        ] = (
                            1
                            - stats["brier"]
                            / null
                        )

                    else:

                        row[
                            "brier_skill_vs_training_prevalence"
                        ] = np.nan

                    row.update(
                        stats
                    )

                    row.update(
                        interval
                    )

                    metrics.append(
                        row
                    )

                # --------------------------------------------
                # Nationality descriptive diagnostics
                # --------------------------------------------
                if name in [
                    "RNN_full",
                    "RNN_no_nationality"
                ]:

                    if (
                        cfg.primary_threshold
                        in ts
                    ):

                        fairness.extend(
                            nationality_diagnostics(
                                g,
                                p,
                                ts[
                                    cfg.primary_threshold
                                ],
                                name,
                                version,
                                dataset
                            )
                        )

                score = binary_scores(
                    y,
                    p
                )

                LOG.info(
                    (
                        "%s %s %s | "
                        "n=%d positives=%d "
                        "AUC=%.3f AP=%.3f "
                        "Brier=%.4f"
                    ),
                    name,
                    version,
                    dataset,
                    len(y),
                    y.sum(),
                    score["roc_auc"],
                    score[
                        "average_precision"
                    ],
                    score["brier"]
                )

        # ----------------------------------------------------
        # SHAP
        # ----------------------------------------------------
        if (
            cfg.export_shap
            and name in [
                "RNN_full",
                "RNN_no_nationality"
            ]
        ):

            export_grouped_shap(
                prep,
                model,
                a[
                    a.partition
                    == "internal_test"
                ],
                meta,
                cfg,
                folder,
                name
            )

    # --------------------------------------------------------
    # Save all prediction results
    # --------------------------------------------------------
    pred = pd.concat(
        predictions,
        ignore_index=True
    )

    save_csv(
        pred,
        folder /
        "classification_predictions.csv"
    )

    save_csv(
        pd.DataFrame(metrics),
        folder /
        "classification_metrics.csv"
    )

    save_csv(
        pd.DataFrame(calibrations),
        folder /
        "calibration_diagnostics.csv"
    )

    if len(bins) > 0:

        save_csv(
            pd.concat(
                bins,
                ignore_index=True
            ),
            folder /
            "calibration_bins.csv"
        )

    save_csv(
        pd.DataFrame(thresholds),
        folder /
        "thresholds.csv"
    )

    save_csv(
        pd.DataFrame(fairness),
        folder /
        (
            "nationality_diagnostics_"
            "DESCRIPTIVE.csv"
        )
    )

    # --------------------------------------------------------
    # Paired comparisons against RNN_full
    # --------------------------------------------------------
    paired = []

    for dataset in [
        "internal_test",
        "temporal_holdout"
    ]:

        for version in [
            "raw",
            "sigmoid"
        ]:

            ref = pred[
                (pred.dataset == dataset)
                & (pred.version == version)
                & (
                    pred.model
                    == "RNN_full"
                )
            ]

            for name in [
                "RNN_no_nationality",
                "no_residual",
                "LMM_residual"
            ]:

                comp = pred[
                    (pred.dataset == dataset)
                    & (
                        pred.version
                        == version
                    )
                    & (
                        pred.model
                        == name
                    )
                ]

                m = ref.merge(
                    comp,
                    on=[
                        "key",
                        "target_source_row"
                    ],
                    suffixes=(
                        "_ref",
                        "_comp"
                    ),
                    validate="one_to_one"
                )

                if m.empty:
                    continue

                if len(m) != len(ref):
                    raise AssertionError(
                        "Paired comparison "
                        "sample size differs."
                    )

                if not (
                    m.y_ref
                    == m.y_comp
                ).all():

                    raise AssertionError(
                        "Paired comparison "
                        "labels differ."
                    )

                d = paired_difference(
                    m.y_ref,
                    m.probability_ref,
                    m.probability_comp,
                    "classification",
                    cfg.bootstrap,
                    cfg.seed + 501
                )

                row = dict(
                    dataset=dataset,
                    version=version,
                    reference="RNN_full",
                    comparison=name,
                    direction=(
                        "comparison minus "
                        "reference"
                    ),
                    n=len(m),
                    positive=int(
                        m.y_ref.sum()
                    )
                )

                row.update(d)

                paired.append(
                    row
                )

    save_csv(
        pd.DataFrame(paired),
        folder /
        (
            "classification_"
            "paired_differences.csv"
        )
    )


# ------------------------------------------------------------
# FINAL CHECK
# ------------------------------------------------------------



# ========================================================================
# Residual analysis and figures
# ========================================================================

# %% 7. Residual effect sizes and publication-supporting figures

def cliffs_delta(positive: np.ndarray, negative: np.ndarray) -> float:
    if not len(positive) or not len(negative):
        return np.nan
    r=rankdata(np.r_[positive,negative]); n1=len(positive); n0=len(negative)
    u=r[:n1].sum()-n1*(n1+1)/2
    return float(2*u/(n1*n0)-1)


def residual_effects(a: pd.DataFrame,cfg: Config,folder: Path) -> None:
    rows=[]; normalization=[]
    for model in ["rnn","lmm"]:
        col=model+"_residual"
        ref=a[(a.partition=="fitting")&(a.y==0)]
        norms=ref.groupby("weight_bin")[col].agg(["mean","std","count"])
        norms.loc[(norms['count']<5)|(norms['std']<=0),"std"]=np.nan
        normtable=norms.reset_index();normtable.insert(0,"model",model);normalization.append(normtable)
        z=(a[col]-a.weight_bin.map(norms['mean']))/a.weight_bin.map(norms['std'])
        for dataset in ["internal_test","temporal_holdout"]:
            ids=np.flatnonzero(a.partition.to_numpy()==dataset)
            for scale in ["raw_kg","fixed_weight_bin_z"]:
                v=a[col].to_numpy(float) if scale=="raw_kg" else z.to_numpy(float)
                pos=v[ids[a.iloc[ids].y.to_numpy()==1]];neg=v[ids[a.iloc[ids].y.to_numpy()==0]]
                pos=pos[np.isfinite(pos)];neg=neg[np.isfinite(neg)]
                row=dict(dataset=dataset,model=model,scale=scale,n_positive=len(pos),n_negative=len(neg),
                         test="two-sided Mann-Whitney; exploratory; no multiplicity correction")
                if len(pos) and len(neg):
                    row.update(mean_difference=float(pos.mean()-neg.mean()),cliffs_delta=cliffs_delta(pos,neg),
                               mannwhitney_p=float(mannwhitneyu(pos,neg,alternative="two-sided").pvalue))
                    rng=np.random.default_rng(cfg.seed+601); boot=[]
                    for _ in range(cfg.bootstrap):
                        ps=rng.choice(pos,len(pos),replace=True);ns=rng.choice(neg,len(neg),replace=True)
                        boot.append(dict(mean_difference=float(ps.mean()-ns.mean()),cliffs_delta=cliffs_delta(ps,ns)))
                    attach_percentiles(row,boot,["mean_difference","cliffs_delta"])
                rows.append(row)
    save_csv(pd.DataFrame(rows),folder/"residual_effect_sizes.csv")
    save_csv(pd.concat(normalization,ignore_index=True),folder/"residual_training_normalization.csv")


def save_figure(fig,folder: Path,name: str):
    folder.mkdir(parents=True,exist_ok=True)
    fig.tight_layout()
    fig.savefig(folder/f"{name}.png",dpi=220,bbox_inches="tight")
    fig.savefig(folder/f"{name}.svg",bbox_inches="tight")
    plt.close(fig)


def generate_figures(folder: Path,cfg: Config):
    pred=pd.read_csv(folder/"classification_predictions.csv")
    metrics=pd.read_csv(folder/"classification_metrics.csv")
    bins=pd.read_csv(folder/"calibration_bins.csv")
    forecast=pd.read_csv(folder/"athlete_predictions.csv")
    reg=pd.read_csv(folder/"regression_metrics.csv")
    out=folder/"figures"
    labels={"RNN_full":"RNN residual", "LMM_residual":"LMM residual",
            "RNN_no_nationality":"RNN, no nationality", "no_residual":"No residual"}
    stamp=" [SOFTWARE TEST]" if cfg.smoke else ""
    for dataset in ["internal_test","temporal_holdout"]:
        base=pred[(pred.dataset==dataset)&(pred.version=="raw")]
        ref=base[base.model=="RNN_full"]
        if ref.empty:
            continue
        title=f"{dataset.replace('_',' ').title()}: n={len(ref)}, positive labels={int(ref.y.sum())}{stamp}"
        fig,ax=plt.subplots(figsize=(7.4,5.5))
        for name,label in labels.items():
            g=base[base.model==name]
            if g.y.nunique()==2:
                fpr,tpr,_=roc_curve(g.y,g.probability)
                ax.plot(fpr,tpr,label=f"{label} (AUC {roc_auc_score(g.y,g.probability):.3f})")
        ax.plot([0,1],[0,1],linestyle="--",label="Chance")
        ax.set(xlabel="False-positive rate",ylabel="Sensitivity",title=title,xlim=(0,1),ylim=(0,1.02))
        ax.legend(fontsize=8,loc="lower right");save_figure(fig,out,f"ROC_{dataset}")
        fig,ax=plt.subplots(figsize=(7.4,5.5))
        for name,label in labels.items():
            g=base[base.model==name]
            if g.y.sum()>0:
                precision,recall,_=precision_recall_curve(g.y,g.probability)
                ax.step(recall,precision,where="post",label=f"{label} (AP {average_precision_score(g.y,g.probability):.3f})")
        ax.axhline(ref.y.mean(),linestyle="--",label=f"Label prevalence {ref.y.mean():.3f}")
        ax.set(xlabel="Recall",ylabel="Precision (PPV)",title=title,xlim=(0,1),ylim=(0,1.02))
        ax.legend(fontsize=8);save_figure(fig,out,f"PR_{dataset}")
        fig,ax=plt.subplots(figsize=(7.4,5.5))
        for version in ["raw","sigmoid"]:
            b=bins[(bins.dataset==dataset)&(bins.model=="RNN_full")&(bins.version==version)]
            err=np.vstack([b.observed-b.observed_low,b.observed_high-b.observed])
            ax.errorbar(b.mean_prediction,b.observed,yerr=err,marker="o",capsize=3,label=version)
        ax.plot([0,1],[0,1],linestyle="--",label="Ideal")
        ax.set(xlabel="Mean predicted probability",ylabel="Observed positive-label proportion",
               title=title+"\nQuantile bins; exact binomial 95% intervals",xlim=(0,1),ylim=(0,1))
        ax.legend(fontsize=9);save_figure(fig,out,f"calibration_{dataset}")
        for model in ["RNN_full","LMM_residual"]:
            row=metrics[(metrics.dataset==dataset)&(metrics.model==model)&(metrics.version=="sigmoid")
                        &(metrics.policy==cfg.primary_threshold)].iloc[0]
            cm=np.array([[row.tn,row.fp],[row.fn,row.tp]],int)
            fig,ax=plt.subplots(figsize=(5.6,4.8))
            ax.imshow(cm)
            for i in range(2):
                for j in range(2):
                    ax.text(j,i,str(cm[i,j]),ha="center",va="center",fontsize=17,
                            bbox=dict(boxstyle="round",alpha=.6))
            ax.set(xticks=[0,1],yticks=[0,1],xticklabels=["Label 0","Label 1"],yticklabels=["Label 0","Label 1"],
                   xlabel="Predicted",ylabel="Recorded label",
                   title=f"{dataset} | {labels[model]}\nSigmoid, threshold={row.threshold:.4f}{stamp}")
            save_figure(fig,out,f"confusion_{dataset}_{model}")
        for metric in ["mae","rmse"]:
            g=reg[(reg.dataset==dataset)&(reg.population=="all")].set_index("model").loc[["RNN","LMM"]]
            fig,ax=plt.subplots(figsize=(6.5,4.3))
            ax.errorbar(g.index,g[metric],yerr=np.vstack([g[metric]-g[metric+"_low"],g[metric+"_high"]-g[metric]]),
                        fmt="o",capsize=5)
            ax.set(ylabel=metric.upper()+" (kg)",title=title+"\nConditional bootstrap 95% intervals")
            save_figure(fig,out,f"{metric}_{dataset}")
        for model in ["rnn","lmm"]:
            g=forecast[forecast.partition==dataset]
            fig,ax=plt.subplots(figsize=(5.8,5.5))
            ax.scatter(g.total,g[model+"_prediction"],s=13,alpha=.5)
            lim=[float(min(g.total.min(),g[model+"_prediction"].min())),float(max(g.total.max(),g[model+"_prediction"].max()))]
            ax.plot(lim,lim,linestyle="--")
            ax.set(xlabel="Observed total (kg)",ylabel="Predicted total (kg)",title=f"{model.upper()} | {title}")
            save_figure(fig,out,f"forecast_{dataset}_{model}")
    for name in ["RNN_full","RNN_no_nationality"]:
        path=folder/f"shap_importance_{name}.csv"
        if path.exists():
            g=pd.read_csv(path)
            fig,ax=plt.subplots(figsize=(7.4,4.8))
            ax.barh(g.feature,g.mean_absolute_group_shap);ax.invert_yaxis()
            ax.set(xlabel="Mean absolute grouped TreeSHAP (raw log-odds)",
                   title=f"{name}: internal test{stamp}\nSigned one-hot contributions summed within feature")
            save_figure(fig,out,f"SHAP_{name}")




# ========================================================================
# Analysis runner and sensitivity analyses
# ========================================================================

# %% 8. Unified scenario runner, cache fingerprints and results consolidation

def implementation_fingerprint() -> str:
    """Hash function/method bytecode without filenames: identical in .py and notebook."""
    import types, inspect
    def canonical_const(v):
        if isinstance(v,types.CodeType):
            return canonical_code(v)
        if isinstance(v,tuple):
            return [canonical_const(x) for x in v]
        if isinstance(v,(set,frozenset)):
            return sorted([canonical_const(x) for x in v],key=lambda x:json.dumps(x,sort_keys=True))
        if isinstance(v,(str,int,float,bool,type(None))):
            return v
        return repr(v)
    def canonical_code(co):
        return dict(bytecode=co.co_code.hex(),constants=[canonical_const(x) for x in co.co_consts],
                    names=co.co_names,varnames=co.co_varnames)
    content={}
    for name,value in sorted(globals().items()):
        if getattr(value,"__module__",None)!=__name__:
            continue
        if inspect.isfunction(value):
            content[name]=canonical_code(inspect.unwrap(value).__code__)
        elif inspect.isclass(value):
            for method,func in value.__dict__.items():
                if inspect.isfunction(func):
                    content[name+"."+method]=canonical_code(inspect.unwrap(func).__code__)
    content["static_constants"]=[VERSION,WORLD_KWS,CONTINENTAL_KWS,NATIONAL_KWS,WEIGHT_LIMITS,WEIGHT_LABELS]
    return hashlib.sha256(json.dumps(content,sort_keys=True,default=str).encode()).hexdigest()


def effective_config(cfg: Config) -> Config:
    c=copy.deepcopy(cfg)
    if c.smoke:
        c.max_epochs=2; c.early_stopping_patience=2; c.n_folds=2
        c.bootstrap=min(c.bootstrap,25); c.n_jobs=min(c.n_jobs,2)
        c.rnn_params.update(hidden=24,layers=1,emb_ec=4,emb_age=4,emb_year=4,batch_size=64)
        c.xgb_params.update(n_estimators=20,max_depth=2)
        c.shap_max_athletes=min(c.shap_max_athletes,100)
    c.validate()
    return c


def scenario_report(folder: Path,cfg: Config,protocol: dict) -> None:
    classification=pd.read_csv(folder/"classification_metrics.csv")
    regression=pd.read_csv(folder/"regression_metrics.csv")
    rows=classification[(classification.version=="sigmoid")&(classification.policy==cfg.primary_threshold)]
    text=["# APP revision 실행 결과", "",
          "**SOFTWARE TEST ONLY — 논문에 사용 금지**" if cfg.smoke else "**동일 파이프라인의 신규 재분석 결과. 원고 기존 수치의 정확 재현을 뜻하지 않음.**", "",
          f"시나리오: {protocol['scenario']}; cutoff ≤{protocol['cutoff_year']}; 최소 {protocol['minimum_records']}경기.",
          f"선수 식별 상태: {protocol['data_audit']['identity_status']}",
          f"경기 순서: {protocol['data_audit']['chronology']}",
          f"LMM 시간변수: {cfg.lmm_time}; random intercept + random slope.", "",
          "## Forecasting — all held-out keys", "",
          "| Dataset | Model | N | R² | MAE (kg) | RMSE (kg) |", "|---|---|---:|---:|---:|---:|"]
    for r in regression[regression.population=="all"].itertuples():
        text.append(f"| {r.dataset} | {r.model} | {r.n} | {r.r2:.3f} | {r.mae:.3f} | {r.rmse:.3f} |")
    text += ["", "## Classification — sigmoid, development-selected threshold", "",
             "| Dataset | Model | N | Positive | AUC | AP | Brier | Sensitivity | PPV |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows.itertuples():
        text.append(f"| {r.dataset} | {r.model} | {r.n} | {r.positive} | {r.roc_auc:.3f} | {r.average_precision:.3f} | {r.brier:.4f} | {r.sensitivity:.3f} | {r.ppv:.3f} |")
    text += ["", "## 해석·제출 시 필수 확인", "",
             "- 라벨은 제공된 제재 기록의 후향적 분류이며 경기 당시 도핑 여부 또는 향후 제재 발생이 아닙니다.",
             "- 이름 대조·공식 신원 확인을 코드가 대신 수행했다고 작성하면 안 됩니다. audit 파일과 원본 기록을 확인하세요.",
             "- 날짜가 없으면 같은 해의 source-row 순서는 실제 경기 순서가 아닐 수 있습니다.",
             "- label0는 미제재 기록이지 비도핑 입증이 아닙니다. reference_population 설정을 Methods에 기록하세요.",
             "- calibration partition은 보정과 임계값 선택에 함께 쓰였습니다. 이 자료의 성능은 테스트 성능으로 보고하지 않습니다.",
             "- Bootstrap은 저장된 예측과 관찰 라벨에 조건부입니다. 전체 재학습·선수 연결·제재 라벨 오류의 불확실성은 포함하지 않습니다.",
             "- calibration-in-the-large는 기울기를 1로 고정한 intercept이며 joint intercept와 다릅니다.",
             "- 적은 양성 수에서 calibration slope/Wald CI/PR 면적은 불안정합니다. Brier가 작다는 사실만으로 충분한 보정도를 입증하지 않습니다.",
             "- ≥5경기 시나리오는 원래 분할 역할을 유지해 하위집단에서 재학습합니다. 효과 차이를 순수한 기록 길이 효과로 해석하면 안 됩니다.",
             "- 2018 cutoff 시나리오는 2019년 이후를 temporal holdout으로 재정의합니다. 주분석의 holdout과 다릅니다.",
             "- 고정 체중 구간은 과거 시점별 공식 Olympic class harmonization이 아닙니다.",
             "- 국적 제외로 공정성이 증명되는 것은 아니며 국가별 진단은 기술적·탐색적 요약입니다.",
             "- RNN의 raw total/target을 학습 자료 통계로 표준화하는 옵션, early stopping 후 재적합 여부를 Methods에 기재하세요.",
             "- 이 워크플로우는 기존 노트북 저장 hyperparameter를 이식하며 새 randomized search를 시행하지 않습니다.",
             "- 같은 cutoff만 맞췄다고 원래 RNN 결과와 직접 비교할 수 없습니다. 여기서 새로 계산된 동일 target의 RNN/LMM 결과를 함께 사용하세요.",
             "- 모델 간 차이는 개별 CI가 겹치는지가 아니라 paired_differences.csv의 차이 CI로 확인하세요.", ""]
    (folder/"RUN_REPORT_KO.md").write_text("\n".join(text),encoding="utf-8")


def run_scenario(records: pd.DataFrame,data_audit: dict,cfg: Config,root: Path,
                 scenario: str,minimum: int,cutoff: int,
                 inherited: Optional[pd.DataFrame]=None,strict_earlier_year: bool=False) -> tuple[Path,pd.DataFrame]:
    a,samples=build_cohort(records,cfg,minimum,strict_earlier_year)
    a=assign_partitions(a,cfg,cutoff,inherited=inherited,use_external_split=scenario.startswith("main_"))
    context=dict(config=asdict(cfg),raw_sha256=data_audit["raw_sha256"],scenario=scenario,
                 minimum_records=minimum,cutoff_year=cutoff,strict_earlier_year=strict_earlier_year,
                 implementation=implementation_fingerprint(),software=software_versions(),
                 roles_sha256=hashlib.sha256(a[["key","target_source_row","partition","y"]].to_csv(index=False).encode()).hexdigest(),
                 clean_records_sha256=hashlib.sha256(records.to_csv(index=False).encode()).hexdigest())
    # Resume itself is not a scientific setting.
    context["config"].pop("resume",None)
    digest=hashlib.sha256(json.dumps(jsonable(context),sort_keys=True).encode()).hexdigest()
    folder=root/f"{scenario}_{digest[:12]}";folder.mkdir(parents=True,exist_ok=True)
    marker=folder/"COMPLETE.json"
    if cfg.resume and marker.exists():
        old=json.loads(marker.read_text(encoding="utf-8"))
        must=["athlete_predictions.csv","classification_metrics.csv","regression_metrics.csv",
              "calibration_diagnostics.csv","thresholds.csv","classification_paired_differences.csv","RUN_REPORT_KO.md"]
        if old.get("fingerprint")==digest and all((folder/f).exists() for f in must):
            LOG.info("Completed scenario reused: %s",folder.name)
            return folder,pd.read_csv(folder/"athlete_predictions.csv")
    start=time.time()
    protocol=dict(scenario=scenario,minimum_records=minimum,cutoff_year=cutoff,
                  strict_earlier_year=strict_earlier_year,data_audit=data_audit,
                  fingerprint=digest,**{k:v for k,v in context.items() if k not in {"scenario","minimum_records","cutoff_year","strict_earlier_year"}})
    save_json(protocol,folder/"protocol.json")
    save_csv(a[["key","target_source_row","partition","y","eventyear","n_records"]],folder/"splits.csv")
    summary=a.groupby("partition",sort=False).agg(n=("key","size"),positive=("y","sum"),
                                                 minimum_final_year=("eventyear","min"),maximum_final_year=("eventyear","max")).reset_index()
    summary["prevalence"]=summary.positive/summary.n
    save_csv(summary,folder/"split_counts.csv")
    LOG.info("SCENARIO %s\n%s",scenario,summary.to_string(index=False))
    try:
        a=fit_longitudinal_models(a,samples,cfg,folder/"longitudinal_models")
        evaluate_regression(a,cfg,folder)
        evaluate_classifiers(a,cfg,folder)
        residual_effects(a,cfg,folder)
        if cfg.make_figures:
            generate_figures(folder,cfg)
        scenario_report(folder,cfg,protocol)
        save_json(dict(fingerprint=digest,scenario=scenario,seconds=time.time()-start,
                       smoke=cfg.smoke,completed_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
                       hyperparameter_search_performed=False),marker)
        LOG.info("SCENARIO FINISHED: %s (%.1f seconds)",scenario,time.time()-start)
    except Exception:
        (folder/"FAILED.txt").write_text(traceback.format_exc(),encoding="utf-8")
        LOG.exception("Scenario failed; partial models/logs preserved. No result filled in.")
        raise
    return folder,a


def consolidate_results(paths: dict[str,Path],root: Path,cfg: Config) -> dict:
    names=["regression_metrics","classification_metrics","calibration_diagnostics","thresholds",
           "regression_paired_differences","classification_paired_differences","residual_effect_sizes","split_counts"]
    summary_dir=root/"summary";summary_dir.mkdir(exist_ok=True)
    tables={}
    for name in names:
        tables[name]=[]
        for scenario,path in paths.items():
            file=path/(name+".csv")
            if file.exists():
                try:
                    d=pd.read_csv(file)
                except pd.errors.EmptyDataError:
                    continue
                protocol=json.loads((path/"protocol.json").read_text(encoding="utf-8"))
                d.insert(0,"scenario",scenario);d.insert(1,"cutoff_year",protocol["cutoff_year"])
                d.insert(2,"minimum_records",protocol["minimum_records"])
                tables[name].append(d)
        tables[name]=pd.concat(tables[name],ignore_index=True) if tables[name] else pd.DataFrame()
        save_csv(tables[name],summary_dir/(name+"_ALL.csv"))
    c=tables["classification_metrics"]
    primary=c[(c.version=="sigmoid")&(c.policy==cfg.primary_threshold)].copy()
    cols=["scenario","cutoff_year","minimum_records","dataset","model","n","positive","prevalence",
          "roc_auc","roc_auc_low","roc_auc_high","average_precision","average_precision_low","average_precision_high",
          "brier","brier_low","brier_high","sensitivity","specificity","ppv","npv","threshold","tn","fp","fn","tp"]
    save_csv(primary[cols],summary_dir/"KEY_CLASSIFICATION_RESULTS.csv")
    r=tables["regression_metrics"]
    save_csv(r[r.population=="all"],summary_dir/"KEY_RNN_LMM_RESULTS.csv")
    save_json({k:str(v.resolve()) for k,v in paths.items()},root/"scenario_paths.json")
    return dict(root=str(root.resolve()),scenario_paths={k:str(v.resolve()) for k,v in paths.items()},
                classification=primary[cols],regression=r[r.population=="all"],calibration=tables["calibration_diagnostics"])


def audit_only(cfg: Config) -> tuple[pd.DataFrame,dict]:
    cfg=effective_config(cfg);root=Path(cfg.output_dir).expanduser().resolve();setup_logging(root)
    return read_and_audit(cfg,root)


def run_all(cfg: Config) -> dict:
    """Main public entry point used by BOTH notebook and standalone script."""
    cfg=effective_config(cfg)
    root=Path(cfg.output_dir).expanduser().resolve();setup_logging(root)
    set_seed(cfg.seed,cfg.n_jobs)
    LOG.info("APP workflow %s | device=%s | transferred hyperparameters, no search",VERSION,resolve_torch_device(cfg))
    if cfg.smoke:
        LOG.warning("SOFTWARE TEST MODE: reduced epochs/folds/trees; NEVER use these performance estimates in a manuscript.")
    save_json(asdict(cfg),root/"effective_config.json")
    save_json(software_versions(),root/"software_versions.json")
    with threadpool_limits(limits=cfg.n_jobs):
        records,audit=read_and_audit(cfg,root)
        paths={}
        main_name=f"main_{cfg.cutoff_year}"
        paths[main_name],main=run_scenario(records,audit,cfg,root,main_name,cfg.min_records,cfg.cutoff_year)
        if cfg.run_min5:
            paths["minimum5"],_=run_scenario(records,audit,cfg,root,"minimum5",max(5,cfg.min_records),cfg.cutoff_year,inherited=main)
        if cfg.run_cutoff2018 and cfg.cutoff_year!=2018:
            paths["cutoff2018"],_=run_scenario(records,audit,cfg,root,"cutoff2018",cfg.min_records,2018)
        if cfg.run_strict_year:
            paths["strict_earlier_year"],_=run_scenario(records,audit,cfg,root,"strict_earlier_year",cfg.min_records,cfg.cutoff_year,
                                                     inherited=main,strict_earlier_year=True)
        result=consolidate_results(paths,root,cfg)
    LOG.info("ALL REQUESTED SCENARIOS COMPLETE. Key tables: %s",root/"summary")
    print("\n=== Core classification results: sigmoid, development-selected threshold ===")
    print(result["classification"][["scenario","dataset","model","n","positive","roc_auc","average_precision","brier","sensitivity","ppv"]].to_string(index=False))
    print("\n=== RNN vs random-intercept-and-slope LMM: same held-out targets ===")
    print(result["regression"][["scenario","dataset","model","n","r2","mae","rmse"]].to_string(index=False))
    return result




# ========================================================================
# Synthetic-data software tests
# ========================================================================

# %% 9. Software tests (synthetic data only; NOT empirical study results)

def make_synthetic_data(path: Path,n_athletes: int=300,seed: int=812) -> Path:
    rng=np.random.default_rng(seed);rows=[];countries=["AAA","BBB","CCC","DDD"]
    for i in range(n_athletes):
        year=2017 if i%5==0 else (2019 if i%5 in [1,2] else 2021)
        n=6+(i%3)
        birth=pd.Timestamp("1970-01-01")+pd.Timedelta(days=i*21)
        label=int(i%7==0 or i%11==0)
        intercept=250+rng.normal(0,30);slope=5+rng.normal(0,2);weight=80+rng.normal(0,8)
        for j in range(n):
            yy=year-n+1+j
            total=intercept+slope*j+rng.normal(0,10)+(12*label if j==n-1 else 0)
            rows.append(dict(nation=countries[i%4],born=birth.strftime("%d.%m.%Y"),bweight=weight+j*.1,
                             snatch=total*.45,jerk=total*.55,total=total,event="SYNTHETIC WORLD CUP",
                             eventyear=yy,doping=label,name=f"Synthetic athlete {i}"))
    save_csv(pd.DataFrame(rows),path)
    return path


def self_test(output_dir: str="APP_software_test") -> dict:
    root=Path(output_dir).resolve();root.mkdir(parents=True,exist_ok=True)
    assert parse_dob("991116")==pd.Timestamp("1999-11-16")
    assert parse_dob("001116")==pd.Timestamp("2000-11-16")
    assert parse_dob("06.07.1976")==pd.Timestamp("1976-07-06")
    assert pd.isna(parse_dob("1999"))
    rng=np.random.default_rng(31)
    for _ in range(20):
        y=rng.integers(0,2,100);p=np.round(rng.random(100),1)
        m=binary_scores(y,p)
        assert np.isclose(m["roc_auc"],roc_auc_score(y,p))
        assert np.isclose(m["average_precision"],average_precision_score(y,p))
        precision,recall,_=precision_recall_curve(y,p)
        assert np.isclose(m["pr_auc_trapezoid"],auc(recall,precision))
    data=make_synthetic_data(root/"synthetic_INPUT_NOT_REAL.csv")
    cfg=Config(raw_csv=str(data),output_dir=str(root/"test_run"),smoke=True,name_col="name",
               run_min5=True,run_cutoff2018=True,run_strict_year=True,resume=False,bootstrap=20)
    result=run_all(cfg)
    # Changing the HELD-OUT target value cannot change its historical LMM input prediction.
    records,_=read_and_audit(effective_config(cfg),root/"test_run")
    a,samples=build_cohort(records,effective_config(cfg),3)
    path=Path(result["scenario_paths"]["main_2019"])
    lm=json.loads((path/"longitudinal_models/final_lmm.json").read_text(encoding="utf-8"))
    meta=json.loads((path/"longitudinal_models/final_rnn.json").read_text(encoding="utf-8"))
    pp=pd.read_csv(path/"athlete_predictions.csv")
    eval_key=pp.loc[pp.partition=="internal_test","key"].iloc[0]
    ix=int(np.flatnonzero(a.key.to_numpy()==eval_key)[0]);modified=copy.deepcopy(samples)
    modified[ix].target+=9999
    assert np.allclose(predict_lmm(samples,np.array([ix]),lm),predict_lmm(modified,np.array([ix]),lm))
    ecfg=effective_config(cfg)
    network=RNNRegressor([len(v) for v in meta["vocab"]],ecfg.rnn_params)
    network.load_state_dict(torch.load(path/"longitudinal_models/final_rnn.pt",map_location="cpu",weights_only=True))
    assert np.allclose(predict_rnn(network,samples,np.array([ix]),meta,ecfg),predict_rnn(network,modified,np.array([ix]),meta,ecfg))
    proof=dict(passed=True,synthetic_data_only=True,DOB_parser=True,AP_ROC_PR_tie_checks=True,
               heldout_target_invariance_RNN=True,heldout_target_invariance_LMM=True,
               paired_same_target_assertions=True,all_four_scenarios=True,
               versions=software_versions())
    save_json(proof,root/"SELF_TEST_PASSED.json")
    print("\nSOFTWARE SELF TEST PASSED. These are synthetic-data results, not study findings.")
    return proof




# ========================================================================
# Command-line interface
# ========================================================================

# %% 10. Command-line entry point

def main():
    parser=argparse.ArgumentParser(description="APP revision workflow: RNN + random-slope LMM + XGBoost evaluation")
    parser.add_argument("--raw",default=None,help="Path to raw CSV")
    parser.add_argument("--out",default=None,help="Output directory")
    parser.add_argument("--config",help="Optional JSON containing Config fields; CLI overrides documented below")
    parser.add_argument("--device",default=None,help="PyTorch auto/cpu/cuda")
    parser.add_argument("--xgb-device",default=None,help="XGBoost cpu/cuda")
    parser.add_argument("--epochs",type=int,default=None)
    parser.add_argument("--bootstrap",type=int,default=None)
    parser.add_argument("--no-sensitivity",action="store_true")
    parser.add_argument("--audit-only",action="store_true")
    parser.add_argument("--smoke",action="store_true",help="Reduced training for software checks, not final results")
    parser.add_argument("--self-test",action="store_true",help="Run synthetic-data tests, no raw file needed")
    args=parser.parse_args()
    if args.self_test:
        self_test(args.out or "APP_software_test");return
    if args.config:
        cfg=Config(**json.loads(locate_file(args.config).read_text(encoding="utf-8")))
        # Only override path fields if the corresponding switch was explicitly provided.
        if args.raw is not None:
            cfg.raw_csv=args.raw
        if args.out is not None:
            cfg.output_dir=args.out
    else:
        cfg=Config(raw_csv=args.raw or "lifting_npj_raw.CSV",output_dir=args.out or "APP_revision_results")
    if args.device is not None: cfg.device=args.device
    if args.xgb_device is not None: cfg.xgb_device=args.xgb_device
    if args.epochs is not None: cfg.max_epochs=args.epochs
    if args.bootstrap is not None: cfg.bootstrap=args.bootstrap
    if args.no_sensitivity: cfg.run_min5=cfg.run_cutoff2018=cfg.run_strict_year=False
    if args.smoke: cfg.smoke=True
    if args.audit_only:
        _,audit=audit_only(cfg);print(json.dumps(audit,ensure_ascii=False,indent=2))
    else:
        run_all(cfg)




if __name__ == "__main__":
    main()
