# -*- coding: utf-8 -*-
"""
Auto-generated from npj_RNN-Copy2.ipynb
Generated: 2025-10-15T07:12:26
"""

# ---- Cell 0 ----
import os, math, random, json, time, argparse, platform
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ---- Cell 1 ----
SEED = 0
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

torch.backends.cudnn.benchmark = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

IS_WIN = platform.system().lower().startswith("win")
NUM_WORKERS = 0 if IS_WIN else 2
PERSISTENT = False if IS_WIN else True
PIN_MEMORY = torch.cuda.is_available()

# 파일 경로/컬럼명
DATA_PATH = "train_longitudinal_se.csv"   # 사용자 데이터 경로로 변경 가능
athlete_id_col = "athlete_id"
col_total = "total"
col_eventcat = "event_category"
col_age = "age"
col_year = "eventyear"
doping_col = "doping"   # 도핑 라벨 (0/1)

# 기본 하이퍼파라미터 공간
TRAIN_RATIO = 0.8
N_FOLDS = 5
N_TRIALS = 15
EPOCHS = 30
BATCH_SIZE_CHOICES = [32, 64, 128]
LR_CHOICES = [1e-3, 5e-4, 2e-3]
WD_CHOICES = [0.0, 1e-5, 1e-4]
HIDDEN_CHOICES = [64, 96, 128, 160, 192]
LAYERS_CHOICES = [1, 2, 3]
DROPOUT_CHOICES = [0.1, 0.2, 0.3, 0.4]
RNN_TYPES = ["lstm", "gru"]
EMB_CHOICES = [8, 12, 16, 24, 32]
PATIENCE = 5
FINAL_PATIENCE = 8
MAX_GRAD_NORM = 1.0
N_BOOT = 1000
BATCH_LOG_INTERVAL = 20

OUT_DIR = "./rnn_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# ---- Cell 2 ----
# ----------------------- 유틸 -----------------------
def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def set_seed_all(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def factorize_series(s: pd.Series) -> Tuple[np.ndarray, Dict]:
    codes, uniques = pd.factorize(s.astype(str), sort=True)
    mapping = {u: i for i, u in enumerate(uniques)}
    # 0은 유효 카테고리이므로 패딩과 충돌 방지 위해 +1 쉬프트
    codes = codes.astype(np.int64) + 1
    mapping = {k: (v+1) for k, v in mapping.items()}
    return codes, mapping

# ---- Cell 3 ----
# ----------------------- 데이터 구성 -----------------------
@dataclass
class Sample:
    total_seq: np.ndarray  # (L, 1) float32
    ec_seq: np.ndarray     # (L,) int64
    age_seq: np.ndarray    # (L,) int64
    year_seq: np.ndarray   # (L,) int64
    target: float          # float32
    athlete: str

class SeqDataset(Dataset):
    def __init__(self, samples: List[Sample]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            "total_seq": torch.from_numpy(s.total_seq),   # (L,1) float32
            "ec_seq": torch.from_numpy(s.ec_seq),         # (L,) int64
            "age_seq": torch.from_numpy(s.age_seq),       # (L,) int64
            "year_seq": torch.from_numpy(s.year_seq),     # (L,) int64
            "target": torch.tensor(s.target, dtype=torch.float32),  # ()
            "athlete": s.athlete,
        }

def collate_fn(batch):
    lengths = torch.tensor([b["total_seq"].shape[0] for b in batch], dtype=torch.long)
    max_len = int(lengths.max().item())
    B = len(batch)

    def pad_float(seq, feat_dim):
        out = torch.zeros(B, max_len, feat_dim, dtype=torch.float32)
        for i, b in enumerate(batch):
            L = b[seq].shape[0]
            out[i, :L, :] = b[seq]
        return out

    def pad_long(seq):
        out = torch.zeros(B, max_len, dtype=torch.long)
        for i, b in enumerate(batch):
            L = b[seq].shape[0]
            out[i, :L] = b[seq]
        return out

    total_pad = pad_float("total_seq", 1)
    ec_pad = pad_long("ec_seq")
    age_pad = pad_long("age_seq")
    year_pad = pad_long("year_seq")
    targets = torch.stack([b["target"] for b in batch], 0)  # (B,)
    athletes = [b["athlete"] for b in batch]
    return total_pad, ec_pad, age_pad, year_pad, lengths, targets, athletes

def load_and_build_samples(path: str) -> Tuple[List[Sample], Dict[str, int], Dict[str, int], Dict[str, int]]:
    df = pd.read_csv(path)

    needed = [athlete_id_col, col_total, col_eventcat, col_age, col_year, doping_col]
    for c in needed:
        if c not in df.columns:
            raise ValueError(f"CSV에 '{c}' 컬럼이 필요합니다. 현재 컬럼들: {list(df.columns)}")

    df = df.dropna(subset=needed).copy()
    df[col_total] = pd.to_numeric(df[col_total], errors="coerce")
    df[col_age]   = pd.to_numeric(df[col_age], errors="coerce").astype("Int64")
    df[col_year]  = pd.to_numeric(df[col_year], errors="coerce").astype("Int64")
    df[doping_col] = pd.to_numeric(df[doping_col], errors="coerce").fillna(0).astype(int)

    # [핵심] 도핑선수 제외
    doped_athletes = set(df.loc[df[doping_col] == 1, athlete_id_col].astype(str).unique())
    before_n = df[athlete_id_col].nunique()
    df = df[~df[athlete_id_col].astype(str).isin(doped_athletes)].copy()
    after_n = df[athlete_id_col].nunique()
    print(f"[도핑 제외] 전체 선수 {before_n} → {after_n} (제외 {before_n - after_n})")

    # 범주형 코드
    df["_ec_code"], ec_map = factorize_series(df[col_eventcat])
    df["_age_code"], age_map = factorize_series(df[col_age].astype(str))
    df["_year_code"], year_map = factorize_series(df[col_year].astype(str))

    df = df.sort_values([athlete_id_col, col_year, col_total]).reset_index(drop=True)

    samples: List[Sample] = []
    for aid, g in df.groupby(athlete_id_col, sort=False):
        if len(g) < 2:
            continue
        past = g.iloc[:-1]
        last = g.iloc[-1]
        samples.append(Sample(
            total_seq=past[col_total].values.astype(np.float32).reshape(-1, 1),
            ec_seq=past["_ec_code"].values.astype(np.int64),
            age_seq=past["_age_code"].values.astype(np.int64),
            year_seq=past["_year_code"].values.astype(np.int64),
            target=float(last[col_total]),
            athlete=str(aid)
        ))
    print(f"총 선수 샘플 수: {len(samples)} (len>=2 선수만 포함, 도핑선수 제외 후)")
    return samples, ec_map, age_map, year_map

# ---- Cell 4 ----
class RNNRegressor(nn.Module):
    def __init__(self,
                 n_ec: int, n_age: int, n_year: int,
                 emb_ec: int, emb_age: int, emb_year: int,
                 rnn_type: str = "lstm",
                 hidden: int = 128,
                 layers: int = 2,
                 dropout: float = 0.2):
        super().__init__()
        self.emb_ec = nn.Embedding(n_ec + 1, emb_ec, padding_idx=0)
        self.emb_age = nn.Embedding(n_age + 1, emb_age, padding_idx=0)
        self.emb_year = nn.Embedding(n_year + 1, emb_year, padding_idx=0)

        in_dim = 1 + emb_ec + emb_age + emb_year
        self.rnn_type = rnn_type.lower()
        if self.rnn_type == "lstm":
            self.rnn = nn.LSTM(
                input_size=in_dim,
                hidden_size=hidden,
                num_layers=layers,
                batch_first=True,
                dropout=(dropout if layers > 1 else 0.0),
            )
        elif self.rnn_type == "gru":
            self.rnn = nn.GRU(
                input_size=in_dim,
                hidden_size=hidden,
                num_layers=layers,
                batch_first=True,
                dropout=(dropout if layers > 1 else 0.0),
            )
        else:
            raise ValueError("rnn_type must be 'lstm' or 'gru'")

        self.head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1)
        )

    def forward(self, total_seq, ec_seq, age_seq, year_seq, lengths):
        ec = self.emb_ec(ec_seq)
        ag = self.emb_age(age_seq)
        yr = self.emb_year(year_seq)
        x = torch.cat([total_seq, ec, ag, yr], dim=-1)  # (B,L,in_dim)

        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        if self.rnn_type == "lstm":
            _, (h_n, _) = self.rnn(packed)
        else:
            _, h_n = self.rnn(packed)

        h_last = h_n[-1]             # (B, hidden)
        out = self.head(h_last).squeeze(-1)
        return out

# ---- Cell 5 ----
def train_one_epoch(model, loader, opt, scheduler=None, epoch=1, log_interval=BATCH_LOG_INTERVAL):
    model.train()
    loss_fn = nn.MSELoss()
    total_loss = 0.0
    t0 = time.time()
    for bi, (total_seq, ec_seq, age_seq, year_seq, lengths, targets, _) in enumerate(loader, 1):
        total_seq = total_seq.to(DEVICE, non_blocking=True)
        ec_seq = ec_seq.to(DEVICE, non_blocking=True)
        age_seq = age_seq.to(DEVICE, non_blocking=True)
        year_seq = year_seq.to(DEVICE, non_blocking=True)
        lengths = lengths.to(DEVICE, non_blocking=True)
        targets = targets.to(DEVICE, non_blocking=True)

        opt.zero_grad(set_to_none=True)
        preds = model(total_seq, ec_seq, age_seq, year_seq, lengths)
        loss = loss_fn(preds, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        opt.step()
        total_loss += loss.item() * targets.size(0)

        if log_interval and (bi % log_interval == 0):
            print(f"  [epoch {epoch}] batch {bi}/{len(loader)} | loss {loss.item():.4f}")
    avg_loss = total_loss / max(1, len(loader.dataset))
    if scheduler is not None:
        scheduler.step(avg_loss)
    print(f"  [epoch {epoch}] train_loss {avg_loss:.4f} | took {time.time()-t0:.1f}s")
    return avg_loss

# ---- Cell 6 ----
@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    ys, ps = [], []
    for total_seq, ec_seq, age_seq, year_seq, lengths, targets, _ in loader:
        total_seq = total_seq.to(DEVICE, non_blocking=True)
        ec_seq = ec_seq.to(DEVICE, non_blocking=True)
        age_seq = age_seq.to(DEVICE, non_blocking=True)
        year_seq = year_seq.to(DEVICE, non_blocking=True)
        lengths = lengths.to(DEVICE, non_blocking=True)
        preds = model(total_seq, ec_seq, age_seq, year_seq, lengths).cpu().numpy()
        ys.append(targets.numpy()); ps.append(preds)
    y = np.concatenate(ys); p = np.concatenate(ps)
    return {
        "r2": r2_score(y, p),
        "mae": mean_absolute_error(y, p),
        "rmse": rmse(y, p),
        "y": y, "p": p
    }

def build_loaders(train_idx, val_idx, samples, bs):
    train_ds = SeqDataset([samples[i] for i in train_idx])
    val_ds   = SeqDataset([samples[i] for i in val_idx])

    train_loader = DataLoader(
        train_ds, batch_size=bs, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTENT,
        collate_fn=collate_fn, drop_last=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=bs, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTENT,
        collate_fn=collate_fn, drop_last=False
    )
    return train_loader, val_loader

# ---- Cell 7 ----
def randomized_params():
    return {
        "rnn_type": random.choice(RNN_TYPES),
        "hidden": random.choice(HIDDEN_CHOICES),
        "layers": random.choice(LAYERS_CHOICES),
        "dropout": random.choice(DROPOUT_CHOICES),
        "emb_ec": random.choice(EMB_CHOICES),
        "emb_age": random.choice(EMB_CHOICES),
        "emb_year": random.choice(EMB_CHOICES),
        "batch_size": random.choice(BATCH_SIZE_CHOICES),
        "lr": random.choice(LR_CHOICES),
        "weight_decay": random.choice(WD_CHOICES),
    }

def split_train_test(samples, ratio=TRAIN_RATIO):
    athletes = np.array([s.athlete for s in samples])
    uniq = np.unique(athletes)
    rng = np.random.RandomState(SEED)
    rng.shuffle(uniq)
    n_train = int(len(uniq) * ratio)
    train_ids = set(uniq[:n_train])
    train_idx, test_idx = [], []
    for i, s in enumerate(samples):
        if s.athlete in train_ids:
            train_idx.append(i)
        else:
            test_idx.append(i)
    return np.array(train_idx), np.array(test_idx)

def group_indices(indices, samples):
    return np.array([samples[i].athlete for i in indices])

def fit_eval_cv(params, samples, train_idx, n_folds=N_FOLDS, epochs=EPOCHS):
    groups = group_indices(train_idx, samples)
    gkf = GroupKFold(n_splits=n_folds)

    # 전체 범주 수 (임베딩 입력 범위; 패딩 0, 유효코드는 >=1)
    n_ec = int(max([s.ec_seq.max() if len(s.ec_seq)>0 else 0 for s in samples]))
    n_age = int(max([s.age_seq.max() if len(s.age_seq)>0 else 0 for s in samples]))
    n_year = int(max([s.year_seq.max() if len(s.year_seq)>0 else 0 for s in samples]))

    fold_scores = []
    for fold, (tr, va) in enumerate(gkf.split(train_idx, groups=groups)):
        tr_idx = train_idx[tr]; va_idx = train_idx[va]
        train_loader, val_loader = build_loaders(tr_idx, va_idx, samples, params["batch_size"])

        model = RNNRegressor(
            n_ec=n_ec, n_age=n_age, n_year=n_year,
            emb_ec=params["emb_ec"], emb_age=params["emb_age"], emb_year=params["emb_year"],
            rnn_type=params["rnn_type"], hidden=params["hidden"], layers=params["layers"],
            dropout=params["dropout"]
        ).to(DEVICE)

        opt = torch.optim.AdamW(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=2)

        best_r2 = -1e9
        best_state = None
        wait = 0

        for ep in range(1, epochs+1):
            tr_loss = train_one_epoch(model, train_loader, opt, scheduler, epoch=ep)
            val_metrics = evaluate(model, val_loader)

            if val_metrics["r2"] > best_r2:
                best_r2 = val_metrics["r2"]
                best_state = {k: v.cpu() for k, v in model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= PATIENCE:
                    print(f"  [fold {fold+1}] early stop at epoch {ep}")
                    break

        if best_state is not None:
            model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
        val_metrics = evaluate(model, val_loader)
        fold_scores.append(val_metrics["r2"])

        print(f"[Fold {fold+1}/{n_folds}] R^2={val_metrics['r2']:.4f}  MAE={val_metrics['mae']:.4f}  RMSE={val_metrics['rmse']:.4f}")

    return float(np.mean(fold_scores)), float(np.std(fold_scores))

# ---- Cell 8 ----
def _split_inner_val(train_idx, samples, ratio=0.9):
    # 최종 학습에서 early stopping용 내부 검증 (athlete 그룹 유지)
    athletes = np.array([samples[i].athlete for i in train_idx])
    uniq = np.unique(athletes)
    rng = np.random.RandomState(SEED)
    rng.shuffle(uniq)
    n_subtrain = int(len(uniq) * ratio)
    sub_ids = set(uniq[:n_subtrain])
    tr_sub, va_sub = [], []
    for i in train_idx:
        if samples[i].athlete in sub_ids:
            tr_sub.append(i)
        else:
            va_sub.append(i)
    return np.array(tr_sub), np.array(va_sub)

# ---- Cell 9 ----
def train_best_and_test(params, samples, train_idx, test_idx, final_epochs=EPOCHS):
    # 범주수 (패딩 제외 유효코드 max)
    n_ec = int(max([s.ec_seq.max() if len(s.ec_seq)>0 else 0 for s in samples]))
    n_age = int(max([s.age_seq.max() if len(s.age_seq)>0 else 0 for s in samples]))
    n_year = int(max([s.year_seq.max() if len(s.year_seq)>0 else 0 for s in samples]))

    # 최종 학습에서도 내부 검증 분리하여 early stopping
    tr_sub, va_sub = _split_inner_val(train_idx, samples, ratio=0.9)
    train_loader, val_loader = build_loaders(tr_sub, va_sub, samples, params["batch_size"])

    test_ds = SeqDataset([samples[i] for i in test_idx])
    test_loader = DataLoader(test_ds, batch_size=params["batch_size"], shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, persistent_workers=PERSISTENT,
                             collate_fn=collate_fn)

    model = RNNRegressor(
        n_ec=n_ec, n_age=n_age, n_year=n_year,
        emb_ec=params["emb_ec"], emb_age=params["emb_age"], emb_year=params["emb_year"],
        rnn_type=params["rnn_type"], hidden=params["hidden"], layers=params["layers"],
        dropout=params["dropout"]
    ).to(DEVICE)

    opt = torch.optim.AdamW(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=2)

    best_state = None
    best_val_r2 = -1e9
    wait = 0
    for ep in range(1, final_epochs+1):
        _ = train_one_epoch(model, train_loader, opt, scheduler, epoch=ep)
        val_metrics = evaluate(model, val_loader)
        print(f"  [final] epoch {ep} | val R^2={val_metrics['r2']:.4f}  MAE={val_metrics['mae']:.4f}  RMSE={val_metrics['rmse']:.4f}")

        if val_metrics["r2"] > best_val_r2:
            best_val_r2 = val_metrics["r2"]
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= FINAL_PATIENCE:
                print(f"  [final] early stop at epoch {ep}")
                break

    if best_state is not None:
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})

    # Test 평가 및 예측 저장
    ys, ps, aids = [], [], []
    for total_seq, ec_seq, age_seq, year_seq, lengths, targets, athletes in test_loader:
        total_seq = total_seq.to(DEVICE, non_blocking=True)
        ec_seq = ec_seq.to(DEVICE, non_blocking=True)
        age_seq = age_seq.to(DEVICE, non_blocking=True)
        year_seq = year_seq.to(DEVICE, non_blocking=True)
        lengths = lengths.to(DEVICE, non_blocking=True)
        with torch.no_grad():
            pred = model(total_seq, ec_seq, age_seq, year_seq, lengths).cpu().numpy()
        ys.extend(targets.numpy().tolist())
        ps.extend(pred.tolist())
        aids.extend(athletes)

    test_metrics = {
        "r2": r2_score(ys, ps),
        "mae": mean_absolute_error(ys, ps),
        "rmse": rmse(np.array(ys), np.array(ps))
    }

    preds_csv = os.path.join(OUT_DIR, "test_predictions.csv")
    pd.DataFrame({"athlete": aids, "y_true": ys, "y_pred": ps}).to_csv(preds_csv, index=False)
    print(f"Saved test predictions: {preds_csv}")

    return model, test_metrics, pd.DataFrame({"athlete": aids, "y_true": ys, "y_pred": ps})

# ---- Cell 10 ----
def bootstrap_ci(df_pred: pd.DataFrame, n_boot=N_BOOT, seed=SEED):
    rng = np.random.RandomState(seed)
    y = df_pred["y_true"].values
    p = df_pred["y_pred"].values
    n = len(y)
    r2s, maes, rmses = [], [], []
    for _ in range(n_boot):
        idx = rng.choice(np.arange(n), size=n, replace=True)
        yy = y[idx]; pp = p[idx]
        r2s.append(r2_score(yy, pp))
        maes.append(mean_absolute_error(yy, pp))
        rmses.append(rmse(yy, pp))
    def ci(a):
        return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))
    return {
        "r2": {"mean": float(np.mean(r2s)), "ci95": ci(r2s)},
        "mae": {"mean": float(np.mean(maes)), "ci95": ci(maes)},
        "rmse": {"mean": float(np.mean(rmses)), "ci95": ci(rmses)},
    }

def parse_args():
    import argparse, sys
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=N_TRIALS, help="Randomized search trials")
    ap.add_argument("--epochs", type=int, default=EPOCHS, help="Epochs per CV fold and final")
    ap.add_argument("--batch-log-interval", type=int, default=BATCH_LOG_INTERVAL, help="Print every N batches")
    ap.add_argument("--data-path", type=str, default=DATA_PATH)
    # ✅ 노트북이 끼워넣는 알 수 없는 인자(-f ...)를 무시
    args, _ = ap.parse_known_args()
    return args

# ---- Cell 11 ----
def main():
    # Windows 멀티프로세싱 안전
    if IS_WIN:
        try:
            import torch.multiprocessing as mp
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass

    args = parse_args()
    global N_TRIALS, EPOCHS, BATCH_LOG_INTERVAL, DATA_PATH
    N_TRIALS = int(args.trials)
    EPOCHS = int(args.epochs)
    BATCH_LOG_INTERVAL = int(args.batch_log_interval)
    DATA_PATH = args.data_path

    set_seed_all(SEED)

    # 1) 데이터 로드 & 샘플 구성
    samples, ec_map, age_map, year_map = load_and_build_samples(DATA_PATH)
    if len(samples) < 10:
        print("샘플 수가 너무 적습니다. 데이터 확인이 필요합니다.")
        return

    # 2) Train/Test (선수 단위)
    train_idx, test_idx = split_train_test(samples, TRAIN_RATIO)
    print(f"Train samples: {len(train_idx)}, Test samples: {len(test_idx)}")

    # 3) Randomized Search + 5-fold CV (R^2 최대화)
    results = []
    best = None
    for t in range(1, N_TRIALS+1):
        params = randomized_params()
        print(f"\n[Trial {t}/{N_TRIALS}] Params: {params}")
        mean_r2, std_r2 = fit_eval_cv(params, samples, train_idx, n_folds=N_FOLDS, epochs=EPOCHS)
        results.append({"trial": t, "params": params, "mean_r2": mean_r2, "std_r2": std_r2})
        if (best is None) or (mean_r2 > best["mean_r2"]):
            best = {"params": params, "mean_r2": mean_r2, "std_r2": std_r2}
        print(f" -> CV R^2 mean={mean_r2:.4f} (std={std_r2:.4f})")

    cv_path = os.path.join(OUT_DIR, "cv_results.json")
    with open(cv_path, "w", encoding="utf-8") as f:
        json.dump({"results": results, "best": best}, f, ensure_ascii=False, indent=2)
    print(f"Saved CV summary: {cv_path}")
    print(f"Best params: {best['params']} (CV R^2={best['mean_r2']:.4f}±{best['std_r2']:.4f})")

    # 4) 최적 파라미터로 최종 학습 & test 평가 (early stopping 포함)
    model, test_metrics, df_pred = train_best_and_test(best["params"], samples, train_idx, test_idx, final_epochs=EPOCHS)
    print(f"\n[Test set]  R^2={test_metrics['r2']:.4f}  MAE={test_metrics['mae']:.4f}  RMSE={test_metrics['rmse']:.4f}")

    # 5) Bootstrap 1000회 (95% CI)
    ci = bootstrap_ci(df_pred, n_boot=N_BOOT, seed=SEED)
    ci_path = os.path.join(OUT_DIR, "bootstrap_ci.json")
    with open(ci_path, "w", encoding="utf-8") as f:
        json.dump(ci, f, ensure_ascii=False, indent=2)
    print("\n[Bootstrap 95% CI]")
    print(json.dumps(ci, indent=2, ensure_ascii=False))
    print(f"Saved CI: {ci_path}")

if __name__ == "__main__":
    main()

# ---- Cell 12 ----
import os, json
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader

# 1) 데이터 로드(클린 선수만) & 분할
samples, ec_map, age_map, year_map = load_and_build_samples(DATA_PATH)  # 이 함수 안에서 도핑선수 제외됨
train_idx, test_idx = split_train_test(samples, TRAIN_RATIO)

# 2) 저장된 최적 하이퍼파라미터 불러오기
with open(os.path.join(OUT_DIR, "cv_results.json"), "r", encoding="utf-8") as f:
    best_params = json.load(f)["best"]["params"]
print("Best params reloaded:", best_params)

# 3) 최종 학습 및 클린 테스트 평가 (기존 함수 시그니처 그대로, n_vocab 인자 X)
model, test_metrics, df_test = train_best_and_test(
    best_params, samples, train_idx, test_idx, final_epochs=EPOCHS
)
print("[Test(clean)] metrics:", test_metrics)

# 4) 예측 수집 함수 (그대로 사용)
def _collect_preds(model, indices, batch_size, samples_):
    ds = SeqDataset([samples_[i] for i in indices])
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTENT, collate_fn=collate_fn
    )
    model.eval()
    ys, ps, aids = [], [], []
    with torch.no_grad():
        for total_seq, ec_seq, age_seq, year_seq, lengths, targets, athletes in loader:
            total_seq = total_seq.to(DEVICE, non_blocking=True)
            ec_seq = ec_seq.to(DEVICE, non_blocking=True)
            age_seq = age_seq.to(DEVICE, non_blocking=True)
            year_seq = year_seq.to(DEVICE, non_blocking=True)
            lengths = lengths.to(DEVICE, non_blocking=True)
            pred = model(total_seq, ec_seq, age_seq, year_seq, lengths).cpu().numpy().tolist()
            ys.extend(targets.numpy().tolist())
            ps.extend(pred)
            aids.extend(athletes)
    df = pd.DataFrame({"athlete_id": aids, "y_true": ys, "y_pred": ps})
    df["residual"] = df["y_true"] - df["y_pred"]
    df["abs_error"] = df["residual"].abs()
    return df

# 5) (클린) train/test 예측 DataFrame 생성 및 저장 (기존 흐름 유지)
df_train = _collect_preds(model, train_idx, best_params["batch_size"], samples)
df_test  = df_test.rename(columns={"athlete": "athlete_id"})  # train_best_and_test가 만든 df_test 재사용
df_test["residual"] = df_test["y_true"] - df_test["y_pred"]
df_test["abs_error"] = df_test["residual"].abs()
df_train["split"] = "train_clean"
df_test["split"]  = "test_clean"

df_clean = pd.concat([df_train, df_test], ignore_index=True)
clean_csv = os.path.join(OUT_DIR, "train_test_predictions_clean.csv")
df_clean.to_csv(clean_csv, index=False)
print(f"Saved clean train/test predictions: {clean_csv}")

# 6) 원본 CSV의 모든 선수(도핑 포함)로 '마지막 경기 예측' 샘플 만들기
#    — 기존 factorize 코드(클린 데이터에서 얻은 ec_map/age_map/year_map)가 없을 수 있는 카테고리는 0(UNK)로 인코딩
def _encode_with_map_safe(series, mapping):
    # 매핑 없으면 0(UNK)로
    return series.astype(str).map(mapping).fillna(0).astype(np.int64).to_numpy()

full_df = pd.read_csv(DATA_PATH)
# 기본 정리 (원본 컬럼명 사용)
needed = [athlete_id_col, col_total, col_eventcat, col_age, col_year]
full_df = full_df.dropna(subset=needed).copy()
full_df[col_total] = pd.to_numeric(full_df[col_total], errors="coerce")
full_df[col_age]   = pd.to_numeric(full_df[col_age], errors="coerce")
full_df[col_year]  = pd.to_numeric(full_df[col_year], errors="coerce")
full_df = full_df.dropna(subset=[col_total, col_age, col_year])

# 클린에서 만든 맵을 이용해 코드화(미등록 값은 0)
full_df = full_df.sort_values([athlete_id_col, col_year, col_total]).reset_index(drop=True)
full_df["_ec_code"]   = _encode_with_map_safe(full_df[col_eventcat], ec_map)
full_df["_age_code"]  = _encode_with_map_safe(full_df[col_age],      age_map)
full_df["_year_code"] = _encode_with_map_safe(full_df[col_year],     year_map)

# 전체 선수 샘플 구성(도핑 포함, len>=2만)
samples_all = []
for aid, g in full_df.groupby(athlete_id_col, sort=False):
    if len(g) < 2:
        continue
    past, last = g.iloc[:-1], g.iloc[-1]
    samples_all.append(
        Sample(
            total_seq=past[col_total].to_numpy(np.float32).reshape(-1,1),
            ec_seq=past["_ec_code"].to_numpy(np.int64),
            age_seq=past["_age_code"].to_numpy(np.int64),
            year_seq=past["_year_code"].to_numpy(np.int64),
            target=float(last[col_total]),
            athlete=str(aid)
        )
    )

# 7) 전체 선수 대상 예측/오차 저장
all_idx = list(range(len(samples_all)))
df_all = _collect_preds(model, all_idx, best_params["batch_size"], samples_all)
df_all["split"] = "all_players"  # 도핑 포함 전체

all_csv = os.path.join(OUT_DIR, "predictions_all_players.csv")
df_all.to_csv(all_csv, index=False)
print(f"Saved all-players predictions: {all_csv}")

# 8) 확인
try:
    display(df_all.head())
except Exception:
    pass

# ---- Cell 13 ----
import os
import json
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple

# === 안전한 기본값 (현재 작업 디렉토리 기준) ===
DEFAULT_DATA_PATH = "train_longitudinal.csv"   # 입력(롱)
DEFAULT_TEST_FULL = "test_full.csv"            # 입력(통합 테스트)
DEFAULT_ERR_OUT   = "test_errors_only.csv"     # 출력
DEFAULT_OUT_DIR   = "./rnn_outputs"
DEFAULT_EPOCHS    = 30

# 상위 스크립트에서 이미 정의돼 있으면 그 값을 사용
try:
    OUT_DIR
except NameError:
    OUT_DIR = DEFAULT_OUT_DIR
try:
    EPOCHS
except NameError:
    EPOCHS = DEFAULT_EPOCHS

# ---------------------------
# 필요한 유틸/구조 정의 (간소화)
# ---------------------------
def factorize_series(s: pd.Series) -> Tuple[np.ndarray, Dict]:
    """카테고리 → 코드 (패딩=0 충돌 방지 위해 +1 쉬프트)"""
    codes, uniques = pd.factorize(s.astype(str), sort=True)
    codes = codes.astype(np.int64) + 1
    mapping = {u: (i+1) for i, u in enumerate(uniques)}
    return codes, mapping

@dataclass
class Sample:
    total_seq: np.ndarray  # (L,1) float32
    ec_seq: np.ndarray     # (L,) int64
    age_seq: np.ndarray    # (L,) int64
    year_seq: np.ndarray   # (L,) int64
    target: float          # float32
    athlete: str           # athlete_id as str

def load_and_build_samples(path: str):
    """
    train_longitudinal.csv 로부터 RNN 학습용 샘플 리스트 생성.
    - 필요한 컬럼: athlete_id, total, event_category, age, eventyear, doping
    - 도핑 선수(athlete_id) 전체 제외
    - 각 선수 len>=2 인 경우만 포함 (과거 시퀀스 → 마지막 total 예측)
    """
    need = ["athlete_id", "total", "event_category", "age", "eventyear", "doping"]
    df = pd.read_csv(path)
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"{path}에 필요한 컬럼이 없습니다: {missing}")

    df = df.dropna(subset=need).copy()
    df["athlete_id"] = df["athlete_id"].astype(str)
    df["total"] = pd.to_numeric(df["total"], errors="coerce")
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["eventyear"] = pd.to_numeric(df["eventyear"], errors="coerce")
    df["doping"] = pd.to_numeric(df["doping"], errors="coerce").fillna(0).astype(int)
    df = df.dropna(subset=["total","age","eventyear"]).copy()

    # 도핑 선수 제외
    doped_ids = set(df.loc[df["doping"]==1, "athlete_id"].unique())
    before_n = df["athlete_id"].nunique()
    df = df[~df["athlete_id"].isin(doped_ids)].copy()
    after_n = df["athlete_id"].nunique()
    print(f"[도핑 제외] 전체 선수 {before_n} → {after_n} (제외 {before_n - after_n})")

    # 코드화 (패딩 0 회피)
    df["_ec_code"], ec_map   = factorize_series(df["event_category"])
    df["_age_code"], age_map = factorize_series(df["age"])
    df["_year_code"], yr_map = factorize_series(df["eventyear"])

    # 정렬
    df = df.sort_values(["athlete_id", "eventyear", "total"]).reset_index(drop=True)

    # 샘플 생성
    samples: List[Sample] = []
    for aid, g in df.groupby("athlete_id", sort=False):
        if len(g) < 2:
            continue
        past = g.iloc[:-1]
        last = g.iloc[-1]
        samples.append(
            Sample(
                total_seq=past["total"].values.astype(np.float32).reshape(-1,1),
                ec_seq=past["_ec_code"].values.astype(np.int64),
                age_seq=past["_age_code"].values.astype(np.int64),
                year_seq=past["_year_code"].values.astype(np.int64),
                target=float(last["total"]),
                athlete=str(aid),
            )
        )
    print(f"총 선수 샘플 수: {len(samples)} (len>=2, 도핑선수 제외 후)")
    return samples, ec_map, age_map, yr_map

# ---------------------------
# Best 파라미터 적용 + 저장
# ---------------------------
def load_best_params(cv_json_path):
    if not os.path.exists(cv_json_path):
        raise FileNotFoundError(f"Best params JSON not found: {cv_json_path}")
    with open(cv_json_path, "r", encoding="utf-8") as f:
        cv = json.load(f)
    if "best" not in cv or "params" not in cv["best"]:
        raise ValueError("cv_results.json에 best.params가 없습니다.")
    return cv["best"]["params"]

def select_last_rows_per_athlete(df, athlete_col="athlete_id", year_col="eventyear"):
    # 동일 연도 다수행이 있으면 원본 순서상 마지막 선택
    tmp = df.reset_index().rename(columns={"index": "orig_idx"})
    tmp = tmp.sort_values([athlete_col, year_col, "orig_idx"])
    last_idx = tmp.groupby(athlete_col).tail(1).index
    return tmp.loc[last_idx].drop(columns=["orig_idx"]).copy()

def apply_best_to_test_full(
    data_path_longitudinal: str = DEFAULT_DATA_PATH,
    test_full_path: str = DEFAULT_TEST_FULL,
    out_csv_path: str = DEFAULT_ERR_OUT
):
    # 상위 학습 함수가 세션에 있는지 확인
    if "train_best_and_test" not in globals():
        raise NameError(
            "train_best_and_test 함수가 현재 세션에 없습니다. "
            "RNN 학습/평가 코드(모델, DataLoader, train_best_and_test 등)를 먼저 실행하세요."
        )

    # 1) test_full 불러와서 테스트용 선수 집합 확보
    df_testfull = pd.read_csv(test_full_path)
    required_cols = {"athlete_id", "total", "eventyear"}
    if not required_cols.issubset(df_testfull.columns):
        raise ValueError(f"test_full.csv에 필요한 컬럼 {required_cols} 가 모두 있어야 합니다. 현재: {list(df_testfull.columns)}")
    df_testfull["athlete_id"] = df_testfull["athlete_id"].astype(str)
    test_aids = set(df_testfull["athlete_id"].unique())

    # 2) longitudinal → 샘플 구성
    samples, ec_map, age_map, year_map = load_and_build_samples(data_path_longitudinal)

    # 3) best 파라미터 로드
    best_params = load_best_params(os.path.join(OUT_DIR, "cv_results.json"))
    print(f"[BEST PARAMS] {best_params}")

    # 4) custom split (누수 방지: athlete 기준)
    test_idx = np.array([i for i, s in enumerate(samples) if s.athlete in test_aids], dtype=int)
    train_idx = np.array([i for i, s in enumerate(samples) if s.athlete not in test_aids], dtype=int)
    print(f"Custom split -> Train samples: {len(train_idx)}, Test samples: {len(test_idx)}")
    if len(test_idx) == 0:
        raise ValueError("test_full의 선수와 일치하는 샘플이 없습니다. 데이터 정합성을 확인하세요.")
    if len(train_idx) == 0:
        raise ValueError("train에 해당하는 샘플이 0입니다. 분리 기준을 확인하세요.")

    # 5) 최적 파라미터로 최종 학습 및 test 평가 (상위 함수 사용)
    model, test_metrics, df_pred = train_best_and_test(best_params, samples, train_idx, test_idx, final_epochs=EPOCHS)
    print(f"[Custom Test] R^2={test_metrics['r2']:.4f}  MAE={test_metrics['mae']:.4f}  RMSE={test_metrics['rmse']:.4f}")

    # df_pred: ["athlete","y_true","y_pred"]
    df_pred = df_pred.rename(columns={"athlete": "athlete_id"})
    df_pred["athlete_id"] = df_pred["athlete_id"].astype(str)
    df_pred["error"] = df_pred["y_true"] - df_pred["y_pred"]
    df_pred["abs_error"] = df_pred["error"].abs()

    # 6) test_full에서 각 선수의 마지막 기록만 선별
    df_last = select_last_rows_per_athlete(df_testfull, athlete_col="athlete_id", year_col="eventyear")

    # 7) 마지막 기록과 예측결과 매칭
    merged = df_last.merge(df_pred, on="athlete_id", how="inner")

    # 8) error 있는 행만 저장 (보통 전부 존재)
    merged_has_error = merged[merged["error"].notna()].copy()

    # 저장
    dirpath = os.path.dirname(out_csv_path) or "."
    os.makedirs(dirpath, exist_ok=True)
    merged_has_error.to_csv(out_csv_path, index=False)
    print(f"Saved rows with errors only -> {out_csv_path}")
    return merged_has_error

# ===== 실행 예시 (현재 폴더 기준 파일명) =====
errs = apply_best_to_test_full(
    data_path_longitudinal=DEFAULT_DATA_PATH,  # ex) "train_longitudinal.csv"
    test_full_path=DEFAULT_TEST_FULL,          # ex) "test_full.csv"
    out_csv_path=DEFAULT_ERR_OUT               # ex) "test_errors_only.csv"
)
print(errs.head())

# ---- Cell 14 ----
# ==== PATCH: train_longitudinal + test_full 합쳐서 샘플 구성 후 분리 ====
import os, json
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple

DEFAULT_DATA_PATH = "train_longitudinal.csv"
DEFAULT_TEST_FULL = "test_full.csv"
DEFAULT_ERR_OUT   = "test_errors_only.csv"
DEFAULT_OUT_DIR   = "./rnn_outputs"
DEFAULT_EPOCHS    = 30

try:
    OUT_DIR
except NameError:
    OUT_DIR = DEFAULT_OUT_DIR
try:
    EPOCHS
except NameError:
    EPOCHS = DEFAULT_EPOCHS

def factorize_series(s: pd.Series) -> Tuple[np.ndarray, Dict]:
    codes, uniques = pd.factorize(s.astype(str), sort=True)
    codes = codes.astype(np.int64) + 1
    mapping = {u: (i+1) for i, u in enumerate(uniques)}
    return codes, mapping

@dataclass
class Sample:
    total_seq: np.ndarray
    ec_seq: np.ndarray
    age_seq: np.ndarray
    year_seq: np.ndarray
    target: float
    athlete: str

def _prep_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    need = ["athlete_id","total","event_category","age","eventyear","doping"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"필수 컬럼 누락: {missing}")
    df = df.copy()
    df["athlete_id"] = df["athlete_id"].astype(str)
    df["total"] = pd.to_numeric(df["total"], errors="coerce")
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["eventyear"] = pd.to_numeric(df["eventyear"], errors="coerce")
    df["doping"] = pd.to_numeric(df["doping"], errors="coerce").fillna(0).astype(int)
    df = df.dropna(subset=["total","age","eventyear"])
    return df

def load_and_build_samples_from_df(df_all: pd.DataFrame):
    df = _prep_and_clean(df_all)

    # 도핑선수 제외
    before = df["athlete_id"].nunique()
    doped = set(df.loc[df["doping"]==1,"athlete_id"].unique())
    df = df[~df["athlete_id"].isin(doped)].copy()
    after = df["athlete_id"].nunique()
    print(f"[도핑 제외] 전체 선수 {before} → {after} (제외 {before - after})")

    # 코드화
    df["_ec_code"], ec_map   = factorize_series(df["event_category"])
    df["_age_code"], age_map = factorize_series(df["age"])
    df["_year_code"], yr_map = factorize_series(df["eventyear"])

    df = df.sort_values(["athlete_id","eventyear","total"]).reset_index(drop=True)

    samples: List[Sample] = []
    for aid, g in df.groupby("athlete_id", sort=False):
        if len(g) < 2:
            continue
        past = g.iloc[:-1]
        last = g.iloc[-1]
        samples.append(Sample(
            total_seq=past["total"].values.astype(np.float32).reshape(-1,1),
            ec_seq=past["_ec_code"].values.astype(np.int64),
            age_seq=past["_age_code"].values.astype(np.int64),
            year_seq=past["_year_code"].values.astype(np.int64),
            target=float(last["total"]),
            athlete=str(aid)
        ))
    print(f"총 선수 샘플 수: {len(samples)} (len>=2, 도핑 제외)")
    return samples, ec_map, age_map, yr_map

def select_last_rows_per_athlete(df, athlete_col="athlete_id", year_col="eventyear"):
    tmp = df.reset_index().rename(columns={"index":"orig_idx"})
    tmp = tmp.sort_values([athlete_col, year_col, "orig_idx"])
    last_idx = tmp.groupby(athlete_col).tail(1).index
    return tmp.loc[last_idx].drop(columns=["orig_idx"]).copy()

def load_best_params(cv_json_path):
    if not os.path.exists(cv_json_path):
        raise FileNotFoundError(f"Best params JSON not found: {cv_json_path}")
    with open(cv_json_path, "r", encoding="utf-8") as f:
        cv = json.load(f)
    if "best" not in cv or "params" not in cv["best"]:
        raise ValueError("cv_results.json에 best.params가 없습니다.")
    return cv["best"]["params"]

def apply_best_to_test_full_combined(
    train_long_path: str = DEFAULT_DATA_PATH,
    test_full_path: str = DEFAULT_TEST_FULL,
    out_csv_path: str = DEFAULT_ERR_OUT
):
    # 상위 학습 함수 존재 확인
    if "train_best_and_test" not in globals():
        raise NameError("train_best_and_test 함수가 현재 세션에 없습니다. (모델/로더 코드 먼저 실행 필요)")

    # 1) 파일 로드 및 병합
    df_trainlong = pd.read_csv(train_long_path)
    df_testfull  = pd.read_csv(test_full_path)
    df_trainlong = _prep_and_clean(df_trainlong)
    df_testfull  = _prep_and_clean(df_testfull)

    # test용 athlete set
    df_testfull["athlete_id"] = df_testfull["athlete_id"].astype(str)
    test_aids = set(df_testfull["athlete_id"].unique())

    # 2) 합치기(중복행 제거)
    df_all = pd.concat([df_trainlong, df_testfull], axis=0, ignore_index=True)
    df_all = df_all.drop_duplicates(subset=["athlete_id","eventyear","total","event_category","age","doping"])

    # 3) 합쳐진 DF에서 샘플 생성
    samples, ec_map, age_map, yr_map = load_and_build_samples_from_df(df_all)

    # 4) 인덱스 분리 (test_aids 기준)
    test_idx = np.array([i for i, s in enumerate(samples) if s.athlete in test_aids], dtype=int)
    train_idx = np.array([i for i, s in enumerate(samples) if s.athlete not in test_aids], dtype=int)
    print(f"Custom split (combined) -> Train samples: {len(train_idx)}, Test samples: {len(test_idx)}")
    if len(test_idx) == 0:
        raise ValueError("test_full의 선수로 구성된 테스트 샘플이 0개입니다. (len<2일 가능성)")

    # 5) best params 로드 후 최종 학습/평가
    best_params = load_best_params(os.path.join(OUT_DIR, "cv_results.json"))
    print(f"[BEST PARAMS] {best_params}")
    model, test_metrics, df_pred = train_best_and_test(best_params, samples, train_idx, test_idx, final_epochs=EPOCHS)
    print(f"[Custom Test] R^2={test_metrics['r2']:.4f}  MAE={test_metrics['mae']:.4f}  RMSE={test_metrics['rmse']:.4f}")

    # 6) 에러 계산 + 마지막 행만 저장
    df_pred = df_pred.rename(columns={"athlete":"athlete_id"})
    df_pred["athlete_id"] = df_pred["athlete_id"].astype(str)
    df_pred["error"] = df_pred["y_true"] - df_pred["y_pred"]
    df_pred["abs_error"] = df_pred["error"].abs()

    df_last = select_last_rows_per_athlete(df_testfull, athlete_col="athlete_id", year_col="eventyear")
    merged = df_last.merge(df_pred, on="athlete_id", how="inner")
    merged_has_error = merged[merged["error"].notna()].copy()

    os.makedirs(os.path.dirname(out_csv_path) or ".", exist_ok=True)
    merged_has_error.to_csv(out_csv_path, index=False)
    print(f"Saved rows with errors only -> {out_csv_path}")
    return merged_has_error

# === 실행 ===
errs = apply_best_to_test_full_combined(
    train_long_path=DEFAULT_DATA_PATH,
    test_full_path=DEFAULT_TEST_FULL,
    out_csv_path=DEFAULT_ERR_OUT
)
print(errs.head())

# ---- Cell 15 ----
import os
import json
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple

# === 기본 경로(현재 폴더 기준) ===
DEFAULT_DATA_PATH = "train_longitudinal.csv"   # 학습용(롱)
DEFAULT_TEST_FULL = "test_full.csv"            # 통합 테스트 CSV(크로스+롱, 혹은 전체)
DEFAULT_ERR_OUT   = "test_errors_only.csv"     # 출력(선수별 마지막 기록만)
DEFAULT_OUT_DIR   = "./rnn_outputs"
DEFAULT_EPOCHS    = 100

# 상위 스크립트에서 이미 정의돼 있으면 그 값을 사용
try:
    OUT_DIR
except NameError:
    OUT_DIR = DEFAULT_OUT_DIR
try:
    EPOCHS
except NameError:
    EPOCHS = DEFAULT_EPOCHS

# ---------------------------
# 유틸/전처리
# ---------------------------
def factorize_series(s: pd.Series) -> Tuple[np.ndarray, Dict]:
    """카테고리→코드. 패딩 0과 충돌하지 않도록 +1 쉬프트."""
    codes, uniques = pd.factorize(s.astype(str), sort=True)
    codes = codes.astype(np.int64) + 1
    mapping = {u: (i+1) for i, u in enumerate(uniques)}
    return codes, mapping

def _prep_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    """필수 컬럼 보장 및 기본 타입 캐스팅."""
    need = ["athlete_id","total","event_category","age","eventyear","doping"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"필수 컬럼 누락: {missing}")
    df = df.copy()
    df["athlete_id"] = df["athlete_id"].astype(str)
    df["total"] = pd.to_numeric(df["total"], errors="coerce")
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["eventyear"] = pd.to_numeric(df["eventyear"], errors="coerce")
    df["doping"] = pd.to_numeric(df["doping"], errors="coerce").fillna(0).astype(int)
    df = df.dropna(subset=["total","age","eventyear"])
    return df

@dataclass
class Sample:
    total_seq: np.ndarray  # (L,1) float32
    ec_seq: np.ndarray     # (L,) int64
    age_seq: np.ndarray    # (L,) int64
    year_seq: np.ndarray   # (L,) int64
    target: float          # float32 (마지막 total)
    athlete: str           # athlete_id as str

def load_and_build_samples_from_df(df_all: pd.DataFrame):
    """
    합쳐진 DF(train_long + test_full)에서 샘플 생성.
    - 도핑 선수는 학습/평가에서 제외.
    - 각 선수 len>=2만 포함 (과거 시퀀스 → 마지막 total 예측).
    """
    df = _prep_and_clean(df_all)

    # 도핑선수 제외(학습/평가 전용)
    before = df["athlete_id"].nunique()
    doped = set(df.loc[df["doping"]==1,"athlete_id"].unique())
    df = df[~df["athlete_id"].isin(doped)].copy()
    after = df["athlete_id"].nunique()
    print(f"[도핑 제외] 전체 선수 {before} → {after} (제외 {before - after})")

    # 코드화(+1 쉬프트)
    df["_ec_code"], ec_map   = factorize_series(df["event_category"])
    df["_age_code"], age_map = factorize_series(df["age"])
    df["_year_code"], yr_map = factorize_series(df["eventyear"])

    # 정렬
    df = df.sort_values(["athlete_id","eventyear","total"]).reset_index(drop=True)

    # 샘플 생성
    samples: List[Sample] = []
    for aid, g in df.groupby("athlete_id", sort=False):
        if len(g) < 2:
            continue
        past = g.iloc[:-1]
        last = g.iloc[-1]
        samples.append(Sample(
            total_seq=past["total"].values.astype(np.float32).reshape(-1,1),
            ec_seq=past["_ec_code"].values.astype(np.int64),
            age_seq=past["_age_code"].values.astype(np.int64),
            year_seq=past["_year_code"].values.astype(np.int64),
            target=float(last["total"]),
            athlete=str(aid)
        ))
    print(f"총 선수 샘플 수: {len(samples)} (len>=2, 도핑 제외)")
    return samples, ec_map, age_map, yr_map

def select_last_rows_per_athlete(df, athlete_col="athlete_id", year_col="eventyear"):
    """각 선수의 마지막(최종) 기록 1행만 남기기."""
    tmp = df.reset_index().rename(columns={"index":"orig_idx"})
    tmp = tmp.sort_values([athlete_col, year_col, "orig_idx"])
    last_idx = tmp.groupby(athlete_col).tail(1).index
    return tmp.loc[last_idx].drop(columns=["orig_idx"]).copy()

def load_best_params(cv_json_path):
    """rnn_outputs/cv_results.json에서 best params 로드."""
    if not os.path.exists(cv_json_path):
        raise FileNotFoundError(f"Best params JSON not found: {cv_json_path}")
    with open(cv_json_path, "r", encoding="utf-8") as f:
        cv = json.load(f)
    if "best" not in cv or "params" not in cv["best"]:
        raise ValueError("cv_results.json에 best.params가 없습니다.")
    return cv["best"]["params"]

# ---------------------------
# 핵심 실행 함수
# ---------------------------
def apply_best_to_test_full_combined(
    train_long_path: str = DEFAULT_DATA_PATH,
    test_full_path: str = DEFAULT_TEST_FULL,
    out_csv_path: str = DEFAULT_ERR_OUT
):
    # 상위에 모델/로더/학습 함수가 로드되어 있어야 함
    if "train_best_and_test" not in globals():
        raise NameError(
            "train_best_and_test 함수가 현재 세션에 없습니다. "
            "RNN 모델/데이터로더/학습 코드(특히 train_best_and_test)를 먼저 실행하세요."
        )

    # 1) 파일 로드
    df_trainlong = pd.read_csv(train_long_path)
    df_testfull  = pd.read_csv(test_full_path)

    # 저장할 때는 도핑 포함 원본 test_full을 그대로 활용
    df_trainlong = _prep_and_clean(df_trainlong)
    df_testfull  = _prep_and_clean(df_testfull)

    # test로 사용할 선수 집합(athlete_id)
    df_testfull["athlete_id"] = df_testfull["athlete_id"].astype(str)
    test_aids = set(df_testfull["athlete_id"].unique())

    # 2) 학습용 데이터프레임 결합(중복 제거)
    df_all = pd.concat([df_trainlong, df_testfull], axis=0, ignore_index=True)
    df_all = df_all.drop_duplicates(
        subset=["athlete_id","eventyear","total","event_category","age","doping"]
    )

    # 3) 샘플 생성(도핑 제외)
    samples, ec_map, age_map, yr_map = load_and_build_samples_from_df(df_all)

    # 4) 인덱스 분리(누수 방지: athlete 기준)
    test_idx  = np.array([i for i, s in enumerate(samples) if s.athlete in test_aids], dtype=int)
    train_idx = np.array([i for i, s in enumerate(samples) if s.athlete not in test_aids], dtype=int)
    print(f"Custom split (combined) -> Train samples: {len(train_idx)}, Test samples: {len(test_idx)}")
    if len(test_idx) == 0:
        raise ValueError("test_full의 선수로 구성된 테스트 샘플이 0개입니다. (len<2일 가능성)")

    # 5) Best params 로드 후 최종 학습/평가
    best_params = load_best_params(os.path.join(OUT_DIR, "cv_results.json"))
    print(f"[BEST PARAMS] {best_params}")
    model, test_metrics, df_pred = train_best_and_test(
        best_params, samples, train_idx, test_idx, final_epochs=EPOCHS
    )
    print(f"[Custom Test] R^2={test_metrics['r2']:.4f}  MAE={test_metrics['mae']:.4f}  RMSE={test_metrics['rmse']:.4f}")

    # 6) 예측결과에 error 추가
    df_pred = df_pred.rename(columns={"athlete":"athlete_id"})
    df_pred["athlete_id"] = df_pred["athlete_id"].astype(str)
    df_pred["error"] = df_pred["y_true"] - df_pred["y_pred"]
    df_pred["abs_error"] = df_pred["error"].abs()

    # 7) test_full(도핑 포함 원본)에서 각 선수 마지막 기록만 선별
    df_last = select_last_rows_per_athlete(df_testfull, athlete_col="athlete_id", year_col="eventyear")

    # 8) left-join으로 병합 → 도핑 선수도 그대로 남음(예측치 없으면 NaN)
    merged = df_last.merge(df_pred, on="athlete_id", how="left")

    # 9) 저장
    os.makedirs(os.path.dirname(out_csv_path) or ".", exist_ok=True)
    merged.to_csv(out_csv_path, index=False)
    print(f"Saved all test rows (including doping athletes) -> {out_csv_path}")
    return merged

# ===== 실행 예시 =====
if __name__ == "__main__":
    result_df = apply_best_to_test_full_combined(
        train_long_path=DEFAULT_DATA_PATH,  # "train_longitudinal.csv"
        test_full_path=DEFAULT_TEST_FULL,   # "test_full.csv"
        out_csv_path=DEFAULT_ERR_OUT        # "test_errors_only.csv"
    )
    print(result_df.head())

# ---- Cell 16 ----
import os
import json
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple

# === 기본 경로(현재 폴더 기준) ===
DEFAULT_DATA_PATH = "train_longitudinal.csv"   # 학습용(롱)
DEFAULT_TEST_FULL = "test_full.csv"            # 통합 테스트 CSV
DEFAULT_OUT_CSV   = "test_errors_only.csv"     # 출력(선수별 마지막 기록만)
DEFAULT_OUT_DIR   = "./rnn_outputs"
DEFAULT_EPOCHS    = 30

# 상위 스크립트에서 이미 정의돼 있으면 그 값을 사용
try:
    OUT_DIR
except NameError:
    OUT_DIR = DEFAULT_OUT_DIR
try:
    EPOCHS
except NameError:
    EPOCHS = DEFAULT_EPOCHS

# ---------------------------
# 유틸/전처리
# ---------------------------
def factorize_series(s: pd.Series) -> Tuple[np.ndarray, Dict]:
    """카테고리→코드. 패딩 0과 충돌하지 않도록 +1 쉬프트."""
    codes, uniques = pd.factorize(s.astype(str), sort=True)
    codes = codes.astype(np.int64) + 1
    mapping = {u: (i+1) for i, u in enumerate(uniques)}
    return codes, mapping

def _prep_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    """필수 컬럼 보장 및 기본 타입 캐스팅."""
    need = ["athlete_id","total","event_category","age","eventyear","doping"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"필수 컬럼 누락: {missing}")
    df = df.copy()
    df["athlete_id"] = df["athlete_id"].astype(str)
    df["total"] = pd.to_numeric(df["total"], errors="coerce")
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["eventyear"] = pd.to_numeric(df["eventyear"], errors="coerce")
    df["doping"] = pd.to_numeric(df["doping"], errors="coerce").fillna(0).astype(int)
    df = df.dropna(subset=["total","age","eventyear"])
    return df

@dataclass
class Sample:
    total_seq: np.ndarray  # (L,1) float32
    ec_seq: np.ndarray     # (L,) int64
    age_seq: np.ndarray    # (L,) int64
    year_seq: np.ndarray   # (L,) int64
    target: float          # float32 (마지막 total)
    athlete: str           # athlete_id as str

def load_and_build_samples_from_df_including_doping(df_all: pd.DataFrame):
    """
    합쳐진 DF(train_long + test_full)에서 샘플 생성.
    - **도핑 선수 포함**.
    - 각 선수 len>=2만 포함 (과거 시퀀스 → 마지막 total 예측).
    """
    df = _prep_and_clean(df_all)

    # 코드화(+1 쉬프트)
    df["_ec_code"], ec_map   = factorize_series(df["event_category"])
    df["_age_code"], age_map = factorize_series(df["age"])
    df["_year_code"], yr_map = factorize_series(df["eventyear"])

    # 정렬
    df = df.sort_values(["athlete_id","eventyear","total"]).reset_index(drop=True)

    # 샘플 생성
    samples: List[Sample] = []
    skip_singletons = 0
    for aid, g in df.groupby("athlete_id", sort=False):
        if len(g) < 2:
            skip_singletons += 1
            continue
        past = g.iloc[:-1]
        last = g.iloc[-1]
        samples.append(Sample(
            total_seq=past["total"].values.astype(np.float32).reshape(-1,1),
            ec_seq=past["_ec_code"].values.astype(np.int64),
            age_seq=past["_age_code"].values.astype(np.int64),
            year_seq=past["_year_code"].values.astype(np.int64),
            target=float(last["total"]),
            athlete=str(aid)
        ))
    print(f"총 선수 샘플 수: {len(samples)} (len>=2, 도핑 포함) | len<2 제외 선수: {skip_singletons}")
    return samples, ec_map, age_map, yr_map

def select_last_rows_per_athlete(df, athlete_col="athlete_id", year_col="eventyear"):
    """각 선수의 마지막(최종) 기록 1행만 남기기."""
    tmp = df.reset_index().rename(columns={"index":"orig_idx"})
    tmp = tmp.sort_values([athlete_col, year_col, "orig_idx"])
    last_idx = tmp.groupby(athlete_col).tail(1).index
    return tmp.loc[last_idx].drop(columns=["orig_idx"]).copy()

def load_best_params(cv_json_path):
    """rnn_outputs/cv_results.json에서 best params 로드."""
    if not os.path.exists(cv_json_path):
        raise FileNotFoundError(f"Best params JSON not found: {cv_json_path}")
    with open(cv_json_path, "r", encoding="utf-8") as f:
        cv = json.load(f)
    if "best" not in cv or "params" not in cv["best"]:
        raise ValueError("cv_results.json에 best.params가 없습니다.")
    return cv["best"]["params"]

# ---------------------------
# 핵심 실행 함수
# ---------------------------
def apply_best_to_test_full_all_included(
    train_long_path: str = DEFAULT_DATA_PATH,
    test_full_path: str = DEFAULT_TEST_FULL,
    out_csv_path: str = DEFAULT_OUT_CSV
):
    # 상위에 모델/로더/학습 함수가 로드되어 있어야 함
    if "train_best_and_test" not in globals():
        raise NameError(
            "train_best_and_test 함수가 현재 세션에 없습니다. "
            "RNN 모델/데이터로더/학습 코드(특히 train_best_and_test)를 먼저 실행하세요."
        )

    # 1) 파일 로드
    df_trainlong = pd.read_csv(train_long_path)
    df_testfull  = pd.read_csv(test_full_path)

    # 저장/출력은 **도핑 포함 원본 test_full**을 그대로 사용
    df_trainlong = _prep_and_clean(df_trainlong)
    df_testfull  = _prep_and_clean(df_testfull)

    # test로 사용할 선수 집합(athlete_id)
    df_testfull["athlete_id"] = df_testfull["athlete_id"].astype(str)
    test_aids = set(df_testfull["athlete_id"].unique())

    # 2) 학습용 데이터프레임 결합(중복 제거)
    df_all = pd.concat([df_trainlong, df_testfull], axis=0, ignore_index=True)
    df_all = df_all.drop_duplicates(
        subset=["athlete_id","eventyear","total","event_category","age","doping"]
    )

    # 3) 샘플 생성(**도핑 포함**)
    samples, ec_map, age_map, yr_map = load_and_build_samples_from_df_including_doping(df_all)

    # 4) 인덱스 분리(누수 방지: athlete 기준)
    test_idx  = np.array([i for i, s in enumerate(samples) if s.athlete in test_aids], dtype=int)
    train_idx = np.array([i for i, s in enumerate(samples) if s.athlete not in test_aids], dtype=int)
    print(f"Custom split (all included) -> Train samples: {len(train_idx)}, Test samples: {len(test_idx)}")
    # 샘플 수가 0일 수 있음(len<2 제외 때문). 그래도 진행(예: 모델 학습 불가시 에러)

    # 5) Best params 로드 후 최종 학습/평가
    best_params = load_best_params(os.path.join(OUT_DIR, "cv_results.json"))
    print(f"[BEST PARAMS] {best_params}")
    model, test_metrics, df_pred = train_best_and_test(
        best_params, samples, train_idx, test_idx, final_epochs=EPOCHS
    )
    print(f"[Custom Test] R^2={test_metrics['r2']:.4f}  MAE={test_metrics['mae']:.4f}  RMSE={test_metrics['rmse']:.4f}")

    # 6) 예측결과에 error 추가 (모델 예측 성공한 선수들)
    df_pred = df_pred.rename(columns={"athlete":"athlete_id"})
    df_pred["athlete_id"] = df_pred["athlete_id"].astype(str)
    df_pred["y_true"] = pd.to_numeric(df_pred["y_true"], errors="coerce")
    df_pred["y_pred"] = pd.to_numeric(df_pred["y_pred"], errors="coerce")
    df_pred["error"] = df_pred["y_true"] - df_pred["y_pred"]
    df_pred["abs_error"] = df_pred["error"].abs()
    df_pred["prediction_flag"] = "model"

    # 7) test_full(도핑 포함 원본)에서 각 선수 마지막 기록만 선별
    df_last = select_last_rows_per_athlete(df_testfull, athlete_col="athlete_id", year_col="eventyear")

    # 8) left-join → 예측치 없는 선수(대개 len<2)는 NaN
    merged = df_last.merge(df_pred[["athlete_id","y_true","y_pred","error","abs_error","prediction_flag"]],
                           on="athlete_id", how="left")

    # 9) NaN(예측 실패: 단발선수 등) 채우기 → 전체 데이터 저장 보장
    #    대체 규칙: y_pred := total(자기값), y_true := total, error := 0, abs_error := 0
    na_mask = merged["y_pred"].isna()
    if na_mask.any():
        merged.loc[na_mask, "y_true"] = merged.loc[na_mask, "total"]
        merged.loc[na_mask, "y_pred"] = merged.loc[na_mask, "total"]
        merged.loc[na_mask, "error"] = 0.0
        merged.loc[na_mask, "abs_error"] = 0.0
        merged.loc[na_mask, "prediction_flag"] = "fallback_singleton"

    # 10) 저장(도핑 포함, 전체 채워짐)
    os.makedirs(os.path.dirname(out_csv_path) or ".", exist_ok=True)
    merged.to_csv(out_csv_path, index=False)
    print(f"Saved ALL test rows (including doping athletes; filled singletons) -> {out_csv_path}")
    return merged

# ===== 실행 예시 =====
if __name__ == "__main__":
    result_df = apply_best_to_test_full_all_included(
        train_long_path=DEFAULT_DATA_PATH,  # "train_longitudinal.csv"
        test_full_path=DEFAULT_TEST_FULL,   # "test_full.csv"
        out_csv_path=DEFAULT_OUT_CSV        # "test_errors_only.csv"
    )
    print(result_df.head())

# ---- Cell 17 ----
import os, json
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader

# --------- 경로 설정 (필요시 수정) ----------
CS_PATH   = "/mnt/data/train_cross_sectional_se.csv"
LONG_PATH = "/mnt/data/train_longitudinal_se.csv"
OUT_PATH  = os.path.join(OUT_DIR, "train_cross_sectional_with_rnn_se.csv")

# --------- 1) (클린) 샘플/맵 생성 + 최적 하이퍼파라미터 로드 ----------
#    - 도핑선수는 훈련 베이스라인 구축(맵/모델 학습)에서 제외 (기존 함수 로직 유지)
samples_clean, ec_map, age_map, year_map = load_and_build_samples(LONG_PATH)
print(f"[clean] samples: {len(samples_clean)}")

# 동일 재현성을 위해 훈련/검증 분할
train_idx, test_idx = split_train_test(samples_clean, TRAIN_RATIO)
print(f"[clean split] train={len(train_idx)}, test={len(test_idx)}")

# 이전 랜덤서치 결과(best params) 재사용 (없으면 예외 발생)
with open(os.path.join(OUT_DIR, "cv_results.json"), "r", encoding="utf-8") as f:
    best_params = json.load(f)["best"]["params"]
print("Best params reloaded:", best_params)

# --------- 2) 최적 파라미터로 최종 학습 (클린 데이터) ----------
model, test_metrics, _ = train_best_and_test(
    best_params, samples_clean, train_idx, test_idx, final_epochs=EPOCHS
)
print("[Clean Test] metrics:", test_metrics)

# --------- 3) (도핑 포함) '훈련 세트 선수들' 전체에 대해 마지막 경기 예측 샘플 구성 ----------
# cross-sectional에 있는 1339명만 대상으로 예측해 병합하려면, 해당 선수 ID 목록을 가져와 필터링
df_cs = pd.read_csv(CS_PATH)
if athlete_id_col not in df_cs.columns:
    raise ValueError(f"Cross-sectional 파일에 '{athlete_id_col}' 컬럼이 없습니다: {df_cs.columns.tolist()}")

train_athletes = set(df_cs[athlete_id_col].astype(str).unique())
print(f"[CS] unique athletes: {len(train_athletes)}")

# Longitudinal 로드(도핑 포함). 맵은 클린으로부터 얻은 ec_map/age_map/year_map 사용,
# 맵에 없는 값은 0(UNK)로 인코딩.
full_df = pd.read_csv(LONG_PATH)

needed_cols = [athlete_id_col, col_total, col_eventcat, col_age, col_year]
for c in needed_cols:
    if c not in full_df.columns:
        raise ValueError(f"Longitudinal 파일에 '{c}' 컬럼이 필요합니다: {full_df.columns.tolist()}")

full_df = full_df.dropna(subset=needed_cols).copy()
full_df[col_total] = pd.to_numeric(full_df[col_total], errors="coerce")
full_df[col_age]   = pd.to_numeric(full_df[col_age], errors="coerce")
full_df[col_year]  = pd.to_numeric(full_df[col_year], errors="coerce")
full_df = full_df.dropna(subset=[col_total, col_age, col_year])

# cross-sectional의 선수만 남김
full_df[athlete_id_col] = full_df[athlete_id_col].astype(str)
full_df = full_df[full_df[athlete_id_col].isin(train_athletes)].copy()

# 맵 인코딩 (미등록은 0)
full_df = full_df.sort_values([athlete_id_col, col_year, col_total]).reset_index(drop=True)
full_df["_ec_code"]   = _encode_with_map_safe(full_df[col_eventcat], ec_map)
full_df["_age_code"]  = _encode_with_map_safe(full_df[col_age],      age_map)
full_df["_year_code"] = _encode_with_map_safe(full_df[col_year],     year_map)

# 선수별 샘플 구성 (len>=2만)
samples_all_train = []
for aid, g in full_df.groupby(athlete_id_col, sort=False):
    if len(g) < 2:
        # 이 경우는 예측 불가(과거가 없음) → 병합 시 NaN으로 남게 됨
        continue
    past, last = g.iloc[:-1], g.iloc[-1]
    samples_all_train.append(
        Sample(
            total_seq=past[col_total].to_numpy(np.float32).reshape(-1,1),
            ec_seq=past["_ec_code"].to_numpy(np.int64),
            age_seq=past["_age_code"].to_numpy(np.int64),
            year_seq=past["_year_code"].to_numpy(np.int64),
            target=float(last[col_total]),
            athlete=str(aid)
        )
    )
print(f"[all-train] samples (len>=2): {len(samples_all_train)}")

# --------- 4) 예측 수집 ----------
def _collect_preds_simple(model, samples_, batch_size):
    ds = SeqDataset(samples_)
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTENT, collate_fn=collate_fn
    )
    model.eval()
    ys, ps, aids = [], [], []
    with torch.no_grad():
        for total_seq, ec_seq, age_seq, year_seq, lengths, targets, athletes in loader:
            total_seq = total_seq.to(DEVICE, non_blocking=True)
            ec_seq = ec_seq.to(DEVICE, non_blocking=True)
            age_seq = age_seq.to(DEVICE, non_blocking=True)
            year_seq = year_seq.to(DEVICE, non_blocking=True)
            lengths = lengths.to(DEVICE, non_blocking=True)
            pred = model(total_seq, ec_seq, age_seq, year_seq, lengths).cpu().numpy().tolist()
            ys.extend(targets.numpy().tolist())
            ps.extend(pred)
            aids.extend(athletes)
    return pd.DataFrame({"athlete_id": aids, "y_true_long_last": ys, "rnn_pred_total": ps})

df_pred_train = _collect_preds_simple(model, samples_all_train, best_params["batch_size"])
print(df_pred_train.head())

# --------- 5) cross-sectional(1339명)과 병합 ----------
# cross-sectional에서 실제 total이 '마지막 경기 total'인지 여부에 따라 residual 정의가 달라질 수 있음.
# 여기서는 cross-sectional의 'total'을 기준으로 residual 계산.
if col_total not in df_cs.columns:
    raise ValueError(f"Cross-sectional 파일에 '{col_total}' 컬럼이 없습니다: {df_cs.columns.tolist()}")

df_merged = df_cs.copy()
df_merged[athlete_id_col] = df_merged[athlete_id_col].astype(str)

df_merged = df_merged.merge(df_pred_train, on=athlete_id_col, how="left")

# 잔차/절대오차 컬럼 추가:
# - rnn_residual_cs = (CS의 total) - (RNN 예측치)
# - 참고로 y_true_long_last(=longitudinal에서 마지막 경기 total)도 함께 남겨 비교 가능
df_merged["rnn_residual_cs"] = df_merged[col_total] - df_merged["rnn_pred_total"]
df_merged["rnn_abs_error_cs"] = df_merged["rnn_residual_cs"].abs()

# --------- 6) 저장 ----------
df_merged.to_csv(OUT_PATH, index=False)
print(f"\n[Saved] {OUT_PATH}")
print(f"Rows: {len(df_merged)}, Columns: {len(df_merged.columns)}")

# (선택) 결측 선수 수 확인: (len<2)로 시퀀스가 없어 예측 불가했던 케이스
n_missing = df_merged["rnn_pred_total"].isna().sum()
print(f"예측치 결측 선수 수 (len<2로 시퀀스 부재 등): {n_missing}")

# ---- Cell 18 ----
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1) 데이터 로드 (이미 df가 있다면 이 줄은 생략)
df = pd.read_csv("train_cross_sectional_with_rnn.csv")

# 2) 필요한 컬럼 확인
assert {"error", "doping"}.issubset(df.columns), "df에 'error', 'doping' 컬럼이 필요합니다."

# 3) 그룹 분리
neg = df.loc[df["doping"] == 0, "error"].dropna().values
pos = df.loc[df["doping"] == 1, "error"].dropna().values

# 4) 플롯
plt.figure(figsize=(6.5, 4.5))

# 박스플롯(두 그룹), 노치/수염/이상치 숨김 옵션
bp = plt.boxplot(
    [neg, pos],
    notch=True,
    vert=True,
    showfliers=False,
    widths=0.5
)

# 점 산포 jitter (x=1,2 주변으로 작은 난수 오프셋)
rng = np.random.default_rng(0)
x1 = 1 + rng.normal(0, 0.04, size=len(neg))
x2 = 2 + rng.normal(0, 0.04, size=len(pos))

plt.plot(x1, neg, "o", alpha=0.35, markersize=4)
plt.plot(x2, pos, "o", alpha=0.35, markersize=4)

# 0 기준선
plt.axhline(0, linestyle="--", linewidth=1)

# 축/제목/눈금
plt.xticks([1, 2], ["Negative", "Positive"])
plt.xlabel("Doping Status")
plt.ylabel("Prediction Error")
plt.title("Prediction Error by Doping Status")

# y축 범위 제한
plt.ylim(-150, 150)

plt.tight_layout()
plt.show()

# ---- Cell 19 ----
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1) 데이터 로드 (이미 df가 있다면 이 줄은 생략)
df = pd.read_csv("train_cross_sectional_with_rnn.csv")

# 2) 필요한 컬럼 확인
assert {"error", "doping"}.issubset(df.columns), "df에 'error', 'doping' 컬럼이 필요합니다."

# 3) 그룹 분리
neg = df.loc[df["doping"] == 0, "error"].dropna().values
pos = df.loc[df["doping"] == 1, "error"].dropna().values
data = [neg, pos]

# 4) Violin Plot
plt.figure(figsize=(6.5, 4.5))

parts = plt.violinplot(
    data,
    showmeans=True,
    showmedians=True
)

# 색상 지정: Negative(파랑), Positive(주황)
colors = ["#1f77b4", "#ff7f0e"]  # matplotlib 기본 파랑/주황
for i, pc in enumerate(parts['bodies']):
    pc.set_facecolor(colors[i])
    pc.set_edgecolor("black")
    pc.set_alpha(0.7)

# mean, median 선 색상 맞추기
for partname in ('cbars','cmins','cmaxes','cmeans','cmedians'):
    vp = parts[partname]
    vp.set_edgecolor("black")
    vp.set_linewidth(1.2)

# 기준선
plt.axhline(0, linestyle="--", linewidth=1, color="gray")

# 축/제목/범위
plt.xticks([1, 2], ["Negative", "Positive"])
plt.xlabel("Doping Status")
plt.ylabel("Prediction Error")
plt.title("Residuals by Doping Status")

plt.ylim(-150, 150)
plt.tight_layout()
plt.show()

# ---- Cell 20 ----
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1) 데이터 로드
df = pd.read_csv("train_cross_sectional_with_rnn.csv")

# 2) 필요한 컬럼 확인
assert {"error", "doping"}.issubset(df.columns), "df에 'error', 'doping' 컬럼이 필요합니다."

# 3) Doping Status 카테고리화
df["Doping Status"] = df["doping"].map({0: "Negative", 1: "Positive"})

# 4) Violin Plot (seaborn)
plt.figure(figsize=(7, 5))
sns.violinplot(
    data=df,
    x="Doping Status",
    y="error",
    palette={"Negative": "#1f77b4", "Positive": "#ff7f0e"},
    cut=0,                 # 극단값 잘라서 violin이 과도하게 퍼지지 않도록
    inner="quartile",      # 중앙값 + 사분위수만 표시
    linewidth=1.2
)

# 기준선
plt.axhline(0, linestyle="--", linewidth=1, color="gray", alpha=0.8)

# 축/제목
plt.xlabel("Doping Status", fontsize=13, labelpad=10)
plt.ylabel("Residual (kg)", fontsize=13, labelpad=10)
# plt.title("Residuals by Doping Status", fontsize=14, weight="semibold", pad=12)

# y축 범위 제한
plt.ylim(-150, 150)

# 테두리 미니멀
sns.despine()

plt.tight_layout()
# plt.savefig('Residuals.png', dpi =300)
plt.show()

# ---- Cell 21 ----
desc = pd.DataFrame({"Negative": pd.Series(neg), "Positive": pd.Series(pos)}).describe()
print(desc)

# ---- Cell 22 ----
import matplotlib.pyplot as plt
import numpy as np

neg = df.loc[df["doping"] == 0, "error"].dropna().values
pos = df.loc[df["doping"] == 1, "error"].dropna().values

plt.figure(figsize=(7,5))

colors = ["#1f77b4", "#ff7f0e"]  # 파랑/주황

# 박스플롯 색상 입히기
bp = plt.boxplot([neg, pos],
                 notch=True, vert=True, showfliers=False, widths=0.5,
                 patch_artist=True)

for patch, color in zip(bp["boxes"], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.3)  # 반투명
for median in bp["medians"]:
    median.set_color("black")
    median.set_linewidth(2)

# 산점 jitter
rng = np.random.default_rng(0)
x1 = 1 + rng.normal(0, 0.04, size=len(neg))
x2 = 2 + rng.normal(0, 0.04, size=len(pos))
plt.scatter(x1, neg, color=colors[0], alpha=0.4, s=10, label="Negative")
plt.scatter(x2, pos, color=colors[1], alpha=0.5, s=18, label="Positive")

# 기준선, 축, 제목
plt.axhline(0, linestyle="--", color="gray", linewidth=1)
plt.xticks([1,2], ["Negative","Positive"])
plt.xlabel("Doping Status")
plt.ylabel("Prediction Error")
# plt.title("Prediction Error by Doping Status")

plt.grid(axis="y", linestyle=":", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig('prediction_error.png', dpi=300)
plt.show()


if __name__ == '__main__':
    print('This module was auto-converted from a notebook. Import functions as needed.')
