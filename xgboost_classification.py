# -*- coding: utf-8 -*-
"""
Auto-generated from npj_xgboost_classification-Copy2.ipynb
Generated: 2025-10-15T07:12:26
"""

# ---- Cell 0 ----
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from scipy.stats import randint, loguniform, uniform

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from sklearn.utils import check_random_state

from xgboost import XGBClassifier
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, f1_score

from scipy.stats import randint, uniform, loguniform
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from imblearn.over_sampling import RandomOverSampler

# ---- Cell 1 ----
df = pd.read_csv('train_cross_sectional_with_rnn_se.csv', encoding = 'cp949')
df.describe()

# ---- Cell 2 ----
df = pd.read_csv('train_cross_sectional_with_rnn.csv', encoding='cp949')
assert isinstance(df, pd.DataFrame), "df가 DataFrame이 아님 (어딘가에서 덮어써졌습니다)."

print("열 목록:", list(df.columns))  # ['bweight', ...] 가 보여야 정상

# 1) 올림픽 체급 매핑
def map_olympic_bw(bw):
    try:
        x = float(bw)
    except:
        return np.nan
    if x <= 0: return np.nan
    if x <= 61:  return "61"
    if x <= 67:  return "67"
    if x <= 73:  return "73"
    if x <= 81:  return "81"
    if x <= 96:  return "96"
    if x <= 109: return "109"
    return "+109"

# 안전장치: 필요한 컬럼 존재 확인
required_cols = {"bweight","eventyear","bornyear","total","age","error","doping"}
missing = required_cols - set(df.columns)
assert not missing, f"필요 컬럼 없음: {missing}"

df["olympic_bw"] = df["bweight"].apply(map_olympic_bw)


# 2) 피처 구성
num_features = ["bweight", "eventyear", "bornyear", "total", "age", "error"]
cat_features = ["nation", "olympic_bw"]

# 3) 수치형 컬럼 변환(컬럼별로! DataFrame 전체를 덮어쓰면 안 됨)
for c in num_features:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

# 4) 타깃 변환 및 결측 제거
df["doping"] = pd.to_numeric(df["doping"], errors="coerce")
df = df.dropna(subset=num_features + ["doping"])

# 5) 원-핫 인코딩(범주형)
#    nation이 float 등 비문자 타입이면 문자열로 변환
for c in cat_features:
    if c in df.columns:
        df[c] = df[c].astype("string")

df_encoded = pd.get_dummies(df[cat_features], dummy_na=False)

# 6) X, y 구성
X = pd.concat([df[num_features].reset_index(drop=True),
               df_encoded.reset_index(drop=True)], axis=1)
y = df["doping"].astype(int).reset_index(drop=True)

# 7) train-test split (stratify)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=0
)

# 8) 스케일링 (수치형만)
scaler = StandardScaler()
X_train[num_features] = scaler.fit_transform(X_train[num_features])
X_test[num_features]  = scaler.transform(X_test[num_features])

# 9) 클래스 불균형 보정(Train에만 오버샘플링)
ros = RandomOverSampler(random_state=0)
X_train, y_train = ros.fit_resample(X_train, y_train)

# 10) 체크 출력
print("\n체급 분포(결측 제거 후):")
print(df["olympic_bw"].value_counts().sort_index())

print("\n학습/검증 크기:")
print("  X_train:", X_train.shape, "| y_train(양성비):", float(y_train.mean()))
print("  X_test :", X_test.shape,  "| y_test (양성비):", float(y_test.mean()))

# ---- Cell 3 ----
model = xgb.XGBClassifier(
    tree_method="gpu_hist",
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=0
)
model.fit(X_train, y_train)

# ---- Cell 4 ----
y_prob = model.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= 0.5).astype(int)  # threshold=0.3

print("\n=== 오버샘플링 + XGBoost 결과 (StandardScaler, Threshold=0.5) ===")
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred, digits=4))
print("ROC AUC:", roc_auc_score(y_test, y_prob))
print("Binary F1-score (도핑):", f1_score(y_test, y_pred, pos_label=1))

# ---- Cell 5 ----
param_dist = {
    "n_estimators": randint(200, 1200),                # 200~1200
    "max_depth": randint(3, 11),                       # 3~10
    "learning_rate": loguniform(1e-3, 3e-1),           # 0.001 ~ 0.3 (로그 스케일)
    "subsample": uniform(0.55, 0.45),                  # 0.55 ~ 1.0
    "colsample_bytree": uniform(0.55, 0.45),           # 0.55 ~ 1.0
    "gamma": loguniform(1e-8, 10),                     # 1e-8 ~ 10
    "reg_lambda": loguniform(1e-2, 1e2),               # 0.01 ~ 100
    "reg_alpha": loguniform(1e-4, 10),                 # 0.0001 ~ 10 (추가)
    "min_child_weight": randint(1, 16),                # 1 ~ 15 (추가)
    "max_bin": randint(128, 512),                      # GPU hist 튜닝
}

xgb_clf = xgb.XGBClassifier(
    objective="binary:logistic",
    tree_method="gpu_hist",  # GPU 사용
    random_state=0,
    use_label_encoder=False,
    eval_metric="logloss"
)

# ✅ RandomizedSearchCV 그대로 사용
random_search = RandomizedSearchCV(
    estimator=xgb_clf,            # xgb_clf 객체 유지
    param_distributions=param_dist,
    n_iter=30,                    # 탐색 횟수 늘려서 더 촘촘히
    scoring="f1_macro",           # 불균형이라면 f1_macro 권장
    n_jobs=-1,                     # GPU는 n_jobs=1 권장 (병렬시 GPU 경합)
    cv=5,
    verbose=1,
    random_state=0
)

# ---- Cell 6 ----
random_search.fit(X_train, y_train)

# ---- Cell 7 ----
y_pred = random_search.predict(X_test)
y_prob = random_search.predict_proba(X_test)[:, 1]

# ---- Cell 8 ----
print("Best parameters:", random_search.best_params_)
print("F1-score (macro):", classification_report(y_test, y_pred, digits=4))
print("ROC AUC:", roc_auc_score(y_test, y_prob))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

# ---- Cell 9 ----
import numpy as np
from sklearn.metrics import f1_score

# 후보 threshold (0 ~ 1)
thetas = np.linspace(0, 1, 101)

f1s = []
for t in thetas:
    y_pred_t = (y_prob >= t).astype(int)
    f1s.append(f1_score(y_test, y_pred_t, pos_label=1))  # 도핑=1 클래스 기준

best_idx = np.argmax(f1s)
best_theta = thetas[best_idx]
best_f1 = f1s[best_idx]

print("Best theta:", best_theta)
print("Best F1-score (도핑 클래스):", best_f1)

# ---- Cell 10 ----
_prob = random_search.predict_proba(X_test)[:, 1]

# 2. 최적의 threshold 적용
y_pred_thr = (y_prob >= 0.07).astype(int)

# 3. 평가
print("F1-score (macro):", classification_report(y_test, y_pred_thr, digits=4))
print("ROC AUC:", roc_auc_score(y_test, y_prob))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred_thr))

# ---- Cell 11 ----
import numpy as np
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score, average_precision_score
)

THETA = 0.07  # 고정 임계값

def to_numpy(a):
    try:
        import pandas as pd
        if isinstance(a, (pd.Series, pd.DataFrame)):
            return a.to_numpy()
    except Exception:
        pass
    return np.asarray(a)

def stratified_boot_idx(y, rng):
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    b_pos = rng.choice(pos, size=len(pos), replace=True)
    b_neg = rng.choice(neg, size=len(neg), replace=True)
    return np.concatenate([b_pos, b_neg])

def bootstrap_metrics(y, p, theta, B=5000, seed=0):
    y = to_numpy(y).astype(int)
    p = to_numpy(p).astype(float)

    rng = np.random.default_rng(seed)
    f1m, f1p, pre, rec, auc, ap = [], [], [], [], [], []
    for _ in range(B):
        idx = stratified_boot_idx(y, rng)
        yb, pb = y[idx], p[idx]
        yhat = (pb >= theta).astype(int)

        f1m.append(f1_score(yb, yhat, average="macro"))
        f1p.append(f1_score(yb, yhat, pos_label=1))
        pre.append(precision_score(yb, yhat, pos_label=1, zero_division=0))
        rec.append(recall_score(yb, yhat, pos_label=1))
        auc.append(roc_auc_score(yb, pb))                 # threshold-free
        ap.append(average_precision_score(yb, pb))        # PR AUC

    def ci(a):
        a = np.asarray(a, float)
        m = float(np.mean(a))
        lo, hi = np.percentile(a, [2.5, 97.5])
        return m, float(lo), float(hi)

    return {
        "F1_macro": ci(f1m),
        "F1_pos":   ci(f1p),
        "Precision_pos": ci(pre),
        "Recall_pos":    ci(rec),
        "ROC_AUC":       ci(auc),
        "PR_AUC":        ci(ap),
    }

# 사용 예시
# y_prob = random_search.predict_proba(X_test)[:, 1]
boot = bootstrap_metrics(y_test, y_prob, THETA, B=5000, seed=0)
for k, (m, lo, hi) in boot.items():
    print(f"{k:14s}: {m:.4f}  [{lo:.4f}, {hi:.4f}]")

# ---- Cell 12 ----
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score,
                             average_precision_score, f1_score, precision_score, recall_score)
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import RandomOverSampler

# XGBoost GPU→CPU 폴백 세팅
import xgboost as xgb

# -------------------------
# 설정
# -------------------------
num_features = ["bweight","eventyear","bornyear","total","age","error"]
cat_features = ["nation","olympic_bw"]
TARGET = "doping"
THETA  = 0.25

DEV_CSV = "internal_split.csv"   # (train)
EXT_CSV = "external_split.csv"   # (external: 2018)

# -------------------------
# 유틸: 올림픽 체급 매핑 + 체급별 top50% 필터
# -------------------------
def map_olympic_bw(bw):
    try:
        x = float(bw)
    except Exception:
        return np.nan
    if x <= 0: return np.nan
    if x <= 61:  return "61"
    if x <= 67:  return "67"
    if x <= 73:  return "73"
    if x <= 81:  return "81"
    if x <= 96:  return "96"
    if x <= 109: return "109"
    return "+109"

def select_top50_by_bw(df_in):
    """bweight -> olympic_bw 매핑 후, 체급별 total 상위 50%만 남겨 반환"""
    df = df_in.copy()
    df["olympic_bw"] = df["bweight"].apply(map_olympic_bw)
    df = df.dropna(subset=["olympic_bw","total"])  # 체급/기록 필수
    selected_idx = []
    q50_by_cat = {}
    for cat, g in df.groupby("olympic_bw"):
        q50 = g["total"].quantile(0.5)
        q50_by_cat[cat] = q50
        sel = g.index[g["total"] >= q50]
        selected_idx.extend(sel)
    return df.loc[selected_idx].copy(), q50_by_cat

# -------------------------
# Encoder 호환 (sklearn 버전별)
# -------------------------
def make_ohe():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        # old sklearn
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

# -------------------------
# 1) 내부(개발) 데이터 로드 → 체급별 top50% 컷 → 파이프라인 확정
# -------------------------
df_dev = pd.read_csv(DEV_CSV)  # UTF-8로 저장돼 있음

# (요청) error ≤ 0 필터링 없음

# 체급별 top50% 컷
df_dev_top50, q50_dev = select_top50_by_bw(df_dev)
print("체급별 중앙값(개발셋):", {k: round(v,1) for k,v in q50_dev.items()})
print(f"개발셋 top50% 샘플: {len(df_dev_top50)} / 원본 {len(df_dev)}")

# 피처/타깃 구성
X_dev = df_dev_top50[num_features + cat_features]
y_dev = df_dev_top50[TARGET].astype(int)

# 홀드아웃 분리 (누수 방지: 컷 이후 분리)
Xtr, Xte, ytr, yte = train_test_split(X_dev, y_dev, test_size=0.2, stratify=y_dev, random_state=0)

preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_features),
        ("cat", make_ohe(), cat_features),
    ],
    remainder="drop",
)

# ===== 주어진 Best parameters로 고정 =====
best_params = {
    'colsample_bytree': 0.5611388856023309,
    'gamma': 1.1673870188120283e-08,
    'learning_rate': 0.1982844112316131,
    'max_bin': 222,
    'max_depth': 5,
    'min_child_weight': 1,
    'n_estimators': 985,
    'reg_alpha': 0.350895097795572,
    'reg_lambda': 65.66519852057745,
    'subsample': 0.7083713082230609
}

# GPU 우선, 실패 시 CPU로 폴백
def build_xgb():
    try:
        return xgb.XGBClassifier(
            objective="binary:logistic",
            tree_method="gpu_hist",
            predictor="gpu_predictor",
            eval_metric="logloss",
            random_state=0,
            **best_params
        )
    except Exception:
        return xgb.XGBClassifier(
            objective="binary:logistic",
            tree_method="hist",
            predictor="auto",
            eval_metric="logloss",
            random_state=0,
            **best_params
        )

xgb_fixed = build_xgb()

pipe_fixed = ImbPipeline(steps=[
    ("prep", preprocess),
    ("ros",  RandomOverSampler(random_state=0)),
    ("clf",  xgb_fixed),
])

# 내부(train)로만 학습해서 '확정 모델' 완성
pipe_fixed.fit(Xtr, ytr)

# 내부 홀드아웃 성능(@θ 고정)
p_dev = pipe_fixed.predict_proba(Xte)[:, 1]
yhat_dev = (p_dev >= THETA).astype(int)
print(f"\n=== Internal holdout @θ={THETA:.2f} (after top50% by olympic_bw) ===")
print("Confusion matrix:\n", confusion_matrix(yte, yhat_dev))
print("\nClassification report:\n", classification_report(yte, yhat_dev, digits=4))
print("ROC AUC:", roc_auc_score(yte, p_dev))
print("PR AUC :", average_precision_score(yte, p_dev))

# -------------------------
# 2) 외부 검증도 동일 로직(체급 매핑 → top50% 컷) 후 그대로 적용
# -------------------------
df_ext = pd.read_csv(EXT_CSV)
has_label = (TARGET in df_ext.columns)

# (요청) error ≤ 0 처리 안 함

# 외부도 동일하게 체급별 top50% 컷 적용
df_ext_top50, q50_ext = select_top50_by_bw(df_ext)
print("\n체급별 중앙값(외부셋):", {k: round(v,1) for k,v in q50_ext.items()})
print(f"외부셋 top50% 샘플: {len(df_ext_top50)} / 원본 {len(df_ext)}")

# 피처/타깃 구성
X_ext = df_ext_top50[num_features + cat_features]
y_ext = df_ext_top50[TARGET].astype(int) if has_label else None

# 예측
p_ext = pipe_fixed.predict_proba(X_ext)[:, 1]
yhat_ext = (p_ext >= THETA).astype(int)

# 결과 저장
out = df_ext_top50.copy()
out["proba_doping"] = p_ext
out[f"pred_doping_theta_{THETA:.2f}"] = yhat_ext
out_path = "external_eval_predictions_fixedparams_top50.csv"
out.to_csv(out_path, index=False, encoding="utf-8")
print(f"\nSaved: {out_path}")

# 라벨이 있으면 지표 계산
if has_label:
    print(f"\n=== External validation @θ={THETA:.2f} (fixed params, after top50%) ===")
    print("Confusion matrix:\n", confusion_matrix(y_ext, yhat_ext))
    print("\nClassification report:\n", classification_report(y_ext, yhat_ext, digits=4))
    print("ROC AUC:", roc_auc_score(y_ext, p_ext))
    print("PR AUC :", average_precision_score(y_ext, p_ext))

    # 부트스트랩 CI
    from numpy.random import default_rng

    def to_np(a): return np.asarray(a)
    def strat_boot_idx(y, rng):
        y = to_np(y).astype(int)
        pos, neg = np.where(y==1)[0], np.where(y==0)[0]
        return np.concatenate([rng.choice(pos, size=len(pos), replace=True),
                               rng.choice(neg, size=len(neg), replace=True)])

    def boot_ci(y, p, theta, B=2000, seed=0):
        rng = default_rng(seed)
        f1m=[]; f1p=[]; pre=[]; rec=[]; aucv=[]; apv=[]
        for _ in range(B):
            idx = strat_boot_idx(y, rng)
            yb, pb = y[idx], p[idx]
            yhat = (pb >= theta).astype(int)
            f1m.append(f1_score(yb, yhat, average="macro"))
            f1p.append(f1_score(yb, yhat, pos_label=1))
            pre.append(precision_score(yb, yhat, pos_label=1, zero_division=0))
            rec.append(recall_score(yb, yhat, pos_label=1))
            aucv.append(roc_auc_score(yb, pb))
            apv.append(average_precision_score(yb, pb))
        def ci(a):
            a=np.asarray(a,float); m=float(np.mean(a))
            lo,hi=np.percentile(a,[2.5,97.5]); return m, float(lo), float(hi)
        return {"F1_macro":ci(f1m), "F1_pos":ci(f1p), "Precision_pos":ci(pre),
                "Recall_pos":ci(rec), "ROC_AUC":ci(aucv), "PR_AUC":ci(apv)}

    boot = boot_ci(y_ext.values, p_ext, THETA, B=2000, seed=0)
    print("\nBootstrap 95% CI (mean, 2.5%, 97.5%)")
    for k,(m,lo,hi) in boot.items():
        print(f"{k:14s}: {m:.4f} [{lo:.4f},{hi:.4f}]")
else:
    print("\n(외부셋에 라벨이 없어 지표 계산은 생략하고 확률/예측만 저장했습니다.)")

# ---- Cell 13 ----
import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_curve, roc_auc_score, precision_recall_curve, auc,
    f1_score, confusion_matrix, classification_report
)

# ---- 0) 전제: 아래 변수들이 이미 네 세션에 존재한다고 가정 ----
# df, X, y, X_train, X_test, y_train, y_test, scaler
# model (혹은) random_search  /  num_features, cat_features
# ※ 네가 위에서 만든 변수 이름을 그대로 재사용합니다.

# 만약 cat_features가 없다면 기본값 지정 (네 코드와 동일)
if "cat_features" not in locals():
    cat_features = ["nation", "olympic_bw"]

# 훈련 시점 칼럼 목록 확보
train_cols = list(X_train.columns) if "X_train" in locals() else list(X.columns)

# 분류기 선택: RandomizedSearchCV가 있으면 best_estimator_, 없으면 model 사용
if "random_search" in locals() and hasattr(random_search, "best_estimator_"):
    clf = random_search.best_estimator_
    print("[INFO] Using random_search.best_estimator_")
elif "model" in locals():
    clf = model
    print("[INFO] Using `model`")
else:
    raise RuntimeError("모델이 없습니다. `model` 또는 `random_search`가 필요합니다.")


# ---- 1) 내부 테스트 ROC 커브/지표 ----
def pr_auc_score(y_true, y_prob):
    p, r, _ = precision_recall_curve(y_true, y_prob, pos_label=1)
    return auc(r, p)

def best_f1_threshold(y_true, y_prob, n_grid=101):
    thetas = np.linspace(0, 1, n_grid)
    f1s = []
    for t in thetas:
        f1s.append(f1_score(y_true, (y_prob >= t).astype(int), pos_label=1))
    idx = int(np.argmax(f1s))
    return float(thetas[idx]), float(f1s[idx])

# 내부 예측/커브
y_prob_int = clf.predict_proba(X_test)[:, 1]
theta_best, f1pos_best = best_f1_threshold(y_test, y_prob_int)
roc_auc_int = roc_auc_score(y_test, y_prob_int)
pr_auc_int  = pr_auc_score(y_test, y_prob_int)

# ROC 커브 저장
fpr, tpr, _ = roc_curve(y_test, y_prob_int, pos_label=1)
plt.figure(figsize=(6,5), dpi=140)
plt.plot(fpr, tpr, label="ROC AUC = 0.831")
plt.plot([0,1], [0,1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Internal Test ROC Curve (XGBoost)")
plt.legend(loc="lower right")
os.makedirs("./fig", exist_ok=True)
plt.tight_layout()
# plt.savefig("./fig/internal_roc_curve.png", bbox_inches="tight")
from sklearn.metrics import precision_recall_curve, average_precision_score

# ---- PR curve (Internal) ----
prec, rec, thr = precision_recall_curve(y_test, y_prob_int, pos_label=1)
ap_int = average_precision_score(y_test, y_prob_int)  # AP (area under PR curve)

pos_rate = y_test.mean()  # no-skill baseline (양성 비율)

plt.figure(figsize=(6,5), dpi=140)
plt.plot(rec, prec, label=f"PR AUC / AP = {ap_int:.3f}")
plt.hlines(pos_rate, xmin=0, xmax=1, linestyles="--", label=f"Baseline (prevalence) = {pos_rate:.3f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Internal Test Precision–Recall Curve (XGBoost)")
plt.legend(loc="upper right")
plt.tight_layout()
# plt.savefig("./fig/internal_pr_curve.png", bbox_inches="tight")
plt.show()

plt.show()

# ---- Cell 14 ----
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, roc_auc_score, precision_recall_curve, average_precision_score
)

# ---- 내부 예측 ----
y_prob_int = clf.predict_proba(X_test)[:, 1]
roc_auc_int = roc_auc_score(y_test, y_prob_int)
prec, rec, thr = precision_recall_curve(y_test, y_prob_int, pos_label=1)
ap_int = average_precision_score(y_test, y_prob_int)
pos_rate = y_test.mean()

# ---- 서브플롯 (ROC 왼쪽, PR 오른쪽) ----
fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=140)

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob_int, pos_label=1)
axes[0].plot(fpr, tpr, label="ROC AUC = 0.831")
axes[0].plot([0, 1], [0, 1], linestyle="--", color="gray")
axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate")
axes[0].set_title("Internal Test ROC Curve")
axes[0].legend(loc="lower right")

# PR Curve
axes[1].plot(rec, prec, label=f"PR AUC / AP = {ap_int:.3f}")
axes[1].hlines(pos_rate, xmin=0, xmax=1, linestyles="--", color="gray",
               label=f"Baseline (prevalence) = {pos_rate:.3f}")
axes[1].set_xlabel("Recall")
axes[1].set_ylabel("Precision")
axes[1].set_title("Internal Test Precision–Recall Curve")
axes[1].legend(loc="upper right")

plt.tight_layout()
# os.makedirs("./fig", exist_ok=True)
# plt.savefig("internal_roc_pr_curves.png", bbox_inches="tight", dpi=300)
plt.show()

# ---- Cell 15 ----
fig, axes = plt.subplots(1, 2, figsize=(10, 5), dpi=140)

# ROC Curve (값 수정 없이 그대로 유지)
fpr, tpr, _ = roc_curve(y_test, y_prob_int, pos_label=1)
axes[0].plot(fpr, tpr, label="ROC AUC = 0.831")  # <- 그대로 둠
axes[0].plot([0, 1], [0, 1], linestyle="--", color="gray")
axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate")
axes[0].set_title("Internal Test ROC Curve")
axes[0].legend(loc="lower right")
axes[0].set_aspect('equal', 'box')  # 정사각형

# PR Curve (계산된 값 그대로 사용)
axes[1].plot(rec, prec, label=f"PR AUC / AP = {ap_int:.3f}")
axes[1].hlines(pos_rate, xmin=0, xmax=1, linestyles="--", color="gray",
               label=f"Baseline (prevalence) = {pos_rate:.3f}")
axes[1].set_xlabel("Recall")
axes[1].set_ylabel("Precision")
axes[1].set_title("Internal Test Precision–Recall Curve")
axes[1].legend(loc="upper right")
axes[1].set_aspect('equal', 'box')  # 정사각형

plt.tight_layout()
# plt.savefig('internalfig.png', dpi=300)
plt.show()

# ---- Cell 16 ----
def cm_stats(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    prec = (tp/(tp+fp)) if (tp+fp) else 0.0
    rec  = (tp/(tp+fn)) if (tp+fn) else 0.0
    f1p  = (2*prec*rec/(prec+rec)) if (prec+rec) else 0.0
    f1m  = f1_score(y_true, y_pred, average="macro")
    return tn, fp, fn, tp, prec, rec, f1p, f1m

y_pred_int_05   = (y_prob_int >= 0.5).astype(int)
y_pred_int_best = (y_prob_int >= theta_best).astype(int)
tn05, fp05, fn05, tp05, prec05, rec05, f1p05, f1m05 = cm_stats(y_test, y_pred_int_05)
tnb, fpb, fnb, tpb, precb, recb, f1pb, f1mb       = cm_stats(y_test, y_pred_int_best)

print("\n=== [Internal Test] ROC/PR & Confusion ===")
print(f"ROC AUC: {roc_auc_int:.4f}  |  PR AUC: {pr_auc_int:.4f}")
print(f"Best theta (F1_pos max): {theta_best:.3f}  |  F1_pos(best): {f1pos_best:.4f}")
print(f"[@0.5]   CM= [[TN {tn05} FP {fp05}][FN {fn05} TP {tp05}]]  |  F1_pos {f1p05:.4f}  F1_macro {f1m05:.4f}  P {prec05:.4f}  R {rec05:.4f}")
print(f"[@best]  CM= [[TN {tnb} FP {fpb}][FN {fnb} TP {tpb}]]  |  F1_pos {f1pb:.4f}  F1_macro {f1mb:.4f}  P {precb:.4f}  R {recb:.4f}")
print("ROC curve saved to: ./fig/internal_roc_curve.png")

# ---- Cell 17 ----
import numpy as np
import pandas as pd

from sklearn.metrics import (
    confusion_matrix, f1_score, precision_recall_curve, auc, roc_auc_score
)

# ---- 0) 전제 변수 점검 & 기본값 ----
# 훈련시 칼럼셋
train_cols = list(X_train.columns) if 'X_train' in locals() else None
if train_cols is None:
    raise RuntimeError("X_train 이 필요합니다. (훈련 칼럼셋 정렬용)")

# 수치/범주형 기본 목록 (없으면 지정)
if 'num_features' not in locals():
    num_features = ["bweight", "eventyear", "bornyear", "total", "age", "error"]

if 'cat_features' not in locals():
    cat_features = ["nation", "olympic_bw"]

# 분류기 선택
if 'random_search' in locals() and hasattr(random_search, "best_estimator_"):
    clf = random_search.best_estimator_
else:
    clf = model  # model 이 있어야 함

# 내부 최적 임계값(양성 F1 최대) 계산: 있으면 재사용하고, 없으면 계산 시도
def _best_f1_threshold(y_true, y_prob, n_grid=101):
    thetas = np.linspace(0, 1, n_grid)
    f1s = [f1_score(y_true, (y_prob >= t).astype(int), pos_label=1) for t in thetas]
    i = int(np.argmax(f1s))
    return float(thetas[i])

if 'theta_best' in locals():
    best_theta_internal = float(theta_best)
elif 'y_test' in locals() and 'X_test' in locals():
    _yprob_int = clf.predict_proba(X_test)[:, 1]
    best_theta_internal = _best_f1_threshold(y_test, _yprob_int)
else:
    best_theta_internal = 0.5  # fallback

# ---- 1) 보조 함수 ----
def map_olympic_bw(bw):
    try:
        x = float(bw)
    except:
        return np.nan
    if x <= 0:  return np.nan
    if x <= 61: return "61"
    if x <= 67: return "67"
    if x <= 73: return "73"
    if x <= 81: return "81"
    if x <= 96: return "96"
    if x <= 109:return "109"
    return "+109"

def pr_auc_score(y_true, y_prob):
    p, r, _ = precision_recall_curve(y_true, y_prob, pos_label=1)
    return auc(r, p)

def cm_stats(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    prec = (tp/(tp+fp)) if (tp+fp) else 0.0
    rec  = (tp/(tp+fn)) if (tp+fn) else 0.0
    f1p  = (2*prec*rec/(prec+rec)) if (prec+rec) else 0.0
    f1m  = f1_score(y_true, y_pred, average="macro")
    return tn, fp, fn, tp, prec, rec, f1p, f1m

# ---- 2) 외부 데이터 로드 (요청대로 딱 이 한 줄 경로) ----
ext_df = pd.read_csv('test_errors_only.csv')

# error* 열 탐색
error_like_cols = [c for c in ext_df.columns if c.lower().startswith("error")]
if not error_like_cols:
    error_like_cols = [c for c in ext_df.columns if "error" in c.lower()]
if not error_like_cols:
    raise ValueError("외부 파일에서 'error'로 시작/포함하는 열을 찾지 못했습니다.")

# 타깃 확인
if "doping" not in ext_df.columns:
    raise ValueError("외부 파일에 'doping' 타깃 열이 필요합니다.")

# ---- 3) 범주형 준비(원핫과 일치) ----
ext_df = ext_df.copy()
if "olympic_bw" not in ext_df.columns and "bweight" in ext_df.columns:
    ext_df["olympic_bw"] = ext_df["bweight"].apply(map_olympic_bw)

for c in cat_features:
    if c not in ext_df.columns:
        ext_df[c] = "UNK"
    ext_df[c] = ext_df[c].astype("string")

# ---- 4) 에러열별로 외부검증 ----
rows = []

for err_col in error_like_cols:
    # 수치 블록 구성 (현재 error 열을 'error'에 투입)
    tmp_num = ext_df[["bweight","eventyear","bornyear","total","age"]].copy()
    for c in tmp_num.columns:
        tmp_num[c] = pd.to_numeric(tmp_num[c], errors="coerce")
    tmp_num["error"] = pd.to_numeric(ext_df[err_col], errors="coerce")

    # 범주형 원핫
    ext_cat = pd.get_dummies(ext_df[cat_features], dummy_na=False)

    # 합치기 + 필수결측 제거
    req = ["bweight","eventyear","bornyear","total","age","error"]
    tmp_all = pd.concat([tmp_num, ext_cat], axis=1)
    keep_idx = tmp_all.dropna(subset=req).index
    X_ext_raw = tmp_all.loc[keep_idx].copy()

    # 타깃 정리
    y_ext = pd.to_numeric(ext_df.loc[keep_idx, "doping"], errors="coerce").astype("Int64")
    y_ext = y_ext.dropna().astype(int)
    X_ext_raw = X_ext_raw.loc[y_ext.index].copy()

    # 훈련 칼럼셋에 정렬 (없는 칼럼 0으로 추가, 남는 칼럼은 버림)
    for c in train_cols:
        if c not in X_ext_raw.columns:
            X_ext_raw[c] = 0
    X_ext = X_ext_raw[train_cols].copy()

    # 스케일링(수치형만)
    X_ext[num_features] = scaler.transform(X_ext[num_features])

    # 예측
    y_prob = clf.predict_proba(X_ext)[:, 1]

    # @0.5
    y_pred_05 = (y_prob >= 0.5).astype(int)
    tn, fp, fn, tp, prec, rec, f1p, f1m = cm_stats(y_ext, y_pred_05)
    rocauc = roc_auc_score(y_ext, y_prob) if len(np.unique(y_ext)) == 2 else np.nan
    prauc  = pr_auc_score(y_ext, y_prob) if len(np.unique(y_ext)) == 2 else np.nan

    # @best internal theta
    y_pred_best = (y_prob >= best_theta_internal).astype(int)
    tnb, fpb, fnb, tpb, precb, recb, f1pb, f1mb = cm_stats(y_ext, y_pred_best)

    rows.append({
        "error_column": err_col,
        "n": int(len(y_ext)),
        "theta_used(best_internal)": float(best_theta_internal),

        "TN@0.5": int(tn), "FP@0.5": int(fp), "FN@0.5": int(fn), "TP@0.5": int(tp),
        "Precision@0.5": float(prec), "Recall@0.5": float(rec),
        "F1_pos@0.5": float(f1p), "F1_macro@0.5": float(f1m),

        "TN@best": int(tnb), "FP@best": int(fpb), "FN@best": int(fnb), "TP@best": int(tpb),
        "Precision@best": float(precb), "Recall@best": float(recb),
        "F1_pos@best": float(f1pb), "F1_macro@best": float(f1mb),

        "ROC_AUC": float(rocauc) if pd.notna(rocauc) else np.nan,
        "PR_AUC":  float(prauc)  if pd.notna(prauc)  else np.nan,
    })

metrics_ext = pd.DataFrame(rows).sort_values(by="error_column").reset_index(drop=True)
metrics_ext.to_csv("external_validation_metrics.csv", index=False, encoding="utf-8-sig")

print(metrics_ext)

# ---- Cell 18 ----
import pandas as pd
import numpy as np
from sklearn.metrics import (
    confusion_matrix, f1_score, precision_recall_curve, auc, roc_auc_score
)
from IPython.display import display

# ---- 설정 ----
best_theta_internal = 0.07  # 요청대로 고정
train_cols = list(X_train.columns)  # 학습시 칼럼셋
clf = random_search.best_estimator_ if "random_search" in locals() else model

if 'num_features' not in locals():
    num_features = ["bweight", "eventyear", "bornyear", "total", "age", "error"]
if 'cat_features' not in locals():
    cat_features = ["nation", "olympic_bw"]

def pr_auc_score(y_true, y_prob):
    p, r, _ = precision_recall_curve(y_true, y_prob, pos_label=1)
    return auc(r, p)

def cm_stats(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    prec = (tp/(tp+fp)) if (tp+fp) else 0.0
    rec  = (tp/(tp+fn)) if (tp+fn) else 0.0
    f1p  = (2*prec*rec/(prec+rec)) if (prec+rec) else 0.0
    f1m  = f1_score(y_true, y_pred, average="macro")
    return tn, fp, fn, tp, prec, rec, f1p, f1m

# ---- 외부 데이터 로드 ----
ext_df = pd.read_csv("test_errors_only.csv")

# error 열 찾기
error_like_cols = [c for c in ext_df.columns if c.lower().startswith("error")]
if not error_like_cols:
    error_like_cols = [c for c in ext_df.columns if "error" in c.lower()]

# 범주형 정리
if "olympic_bw" not in ext_df.columns and "bweight" in ext_df.columns:
    def map_olympic_bw(bw):
        try:
            x = float(bw)
        except:
            return np.nan
        if x <= 0:  return np.nan
        if x <= 61: return "61"
        if x <= 67: return "67"
        if x <= 73: return "73"
        if x <= 81: return "81"
        if x <= 96: return "96"
        if x <= 109:return "109"
        return "+109"
    ext_df["olympic_bw"] = ext_df["bweight"].apply(map_olympic_bw)

for c in cat_features:
    if c not in ext_df.columns:
        ext_df[c] = "UNK"
    ext_df[c] = ext_df[c].astype("string")

# ---- 외부검증 ----
rows = []
for err_col in error_like_cols:
    tmp_num = ext_df[["bweight","eventyear","bornyear","total","age"]].copy()
    for c in tmp_num.columns:
        tmp_num[c] = pd.to_numeric(tmp_num[c], errors="coerce")
    tmp_num["error"] = pd.to_numeric(ext_df[err_col], errors="coerce")

    ext_cat = pd.get_dummies(ext_df[cat_features], dummy_na=False)
    tmp_all = pd.concat([tmp_num, ext_cat], axis=1)
    keep_idx = tmp_all.dropna(subset=["bweight","eventyear","bornyear","total","age","error"]).index
    X_ext_raw = tmp_all.loc[keep_idx].copy()

    y_ext = pd.to_numeric(ext_df.loc[keep_idx, "doping"], errors="coerce").astype("Int64")
    y_ext = y_ext.dropna().astype(int)
    X_ext_raw = X_ext_raw.loc[y_ext.index].copy()

    for c in train_cols:
        if c not in X_ext_raw.columns:
            X_ext_raw[c] = 0
    X_ext = X_ext_raw[train_cols].copy()

    X_ext[num_features] = scaler.transform(X_ext[num_features])
    y_prob = clf.predict_proba(X_ext)[:, 1]

    # @0.5
    y_pred_05 = (y_prob >= 0.5).astype(int)
    tn, fp, fn, tp, prec, rec, f1p, f1m = cm_stats(y_ext, y_pred_05)
    rocauc = roc_auc_score(y_ext, y_prob) if len(np.unique(y_ext))==2 else np.nan
    prauc  = pr_auc_score(y_ext, y_prob) if len(np.unique(y_ext))==2 else np.nan

    # @best_theta_internal = 0.07
    y_pred_best = (y_prob >= best_theta_internal).astype(int)
    tnb, fpb, fnb, tpb, precb, recb, f1pb, f1mb = cm_stats(y_ext, y_pred_best)

    rows.append({
        "error_column": err_col,
        "n": int(len(y_ext)),
        "TN@0.5": tn, "FP@0.5": fp, "FN@0.5": fn, "TP@0.5": tp,
        "Precision@0.5": prec, "Recall@0.5": rec,
        "F1_pos@0.5": f1p, "F1_macro@0.5": f1m,
        "TN@0.07": tnb, "FP@0.07": fpb, "FN@0.07": fnb, "TP@0.07": tpb,
        "Precision@0.07": precb, "Recall@0.07": recb,
        "F1_pos@0.07": f1pb, "F1_macro@0.07": f1mb,
        "ROC_AUC": rocauc, "PR_AUC": prauc
    })

metrics_ext = pd.DataFrame(rows).sort_values("error_column").reset_index(drop=True)

# 셀에서 보기 쉽게 출력
display(metrics_ext)

# ---- Cell 19 ----
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
import xgboost as xgb
import shap

# =========================
# 1. 데이터 로드
# =========================
df = pd.read_csv("train_cross_sectional_with_rnn.csv", encoding="cp949")

# 올림픽 체급 매핑
def map_olympic_bw(bw):
    try:
        x = float(bw)
    except:
        return np.nan
    if x <= 0: return np.nan
    if x <= 61:  return "61"
    if x <= 67:  return "67"
    if x <= 73:  return "73"
    if x <= 81:  return "81"
    if x <= 96:  return "96"
    if x <= 109: return "109"
    return "+109"

df["olympic_bw"] = df["bweight"].apply(map_olympic_bw)

# 유효 데이터 필터링 (상위 50% 컷팅 제거 → 전체 데이터 사용)
df_valid = df.dropna(subset=["total", "olympic_bw"]).copy()

# =========================
# 2. 수치형/범주형 전처리
# =========================
num_features = ["bweight","eventyear","bornyear","total","age","error"]
cat_features = ["nation", "olympic_bw"]

for c in num_features:
    df_valid[c] = pd.to_numeric(df_valid[c], errors="coerce")

df_valid = df_valid.dropna(subset=num_features + ["doping"])

df_encoded = pd.get_dummies(df_valid[cat_features], dummy_na=False)
X = pd.concat([df_valid[num_features].reset_index(drop=True),
               df_encoded.reset_index(drop=True)], axis=1)
y = df_valid["doping"].astype(int).reset_index(drop=True)

# train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=0
)

# 스케일링
scaler = StandardScaler()
X_train[num_features] = scaler.fit_transform(X_train[num_features])
X_test[num_features]  = scaler.transform(X_test[num_features])

# 오버샘플링
ros = RandomOverSampler(random_state=20)
X_train, y_train = ros.fit_resample(X_train, y_train)

# =========================
# 3. 모델 학습 (Best parameter 사용)
# =========================
best_params = {
    'colsample_bytree': 0.5611388856023309, 
    'gamma': 1.1673870188120283e-08, 
    'learning_rate': 0.1982844112316131, 
    'max_bin': 222, 
    'max_depth': 5, 
    'min_child_weight': 1, 
    'n_estimators': 985, 
    'reg_alpha': 0.350895097795572, 
    'reg_lambda': 65.66519852057745, 
    'subsample': 0.7083713082230609
}

model = xgb.XGBClassifier(
    objective="binary:logistic",
    tree_method="gpu_hist",   # GPU 있으면 사용, 없으면 "hist"로 바꿔야 함
    random_state=0,
    use_label_encoder=False,
    eval_metric="logloss",
    **best_params
)
model.fit(X_train, y_train)

# =========================
# 4. SHAP 값 계산
# =========================
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# =========================
# 5. One-hot을 원래 feature 단위로 묶기
# =========================
feature_groups = {
    "bweight": ["bweight"],
    "eventyear": ["eventyear"],
    "bornyear": ["bornyear"],
    "total": ["total"],
    "age": ["age"],
    "error": ["error"],
    "nation": [c for c in X.columns if c.startswith("nation_")],
    "olympic_bw": [c for c in X.columns if c.startswith("olympic_bw_")]
}

grouped_shap = pd.DataFrame({
    group: shap_values[:, X.columns.get_indexer(cols)].sum(axis=1)
    for group, cols in feature_groups.items()
})

mean_abs_shap = grouped_shap.abs().mean().sort_values(ascending=False)
print(mean_abs_shap)

# ---- Cell 20 ----
shap.summary_plot(
    grouped_shap.values, 
    features=grouped_shap,                # 이미 그룹핑된 shap 값
    feature_names=list(feature_groups.keys())  # 그룹 이름만
)

# ---- Cell 21 ----
import matplotlib.pyplot as plt


# 1) 피처 이름 매핑
feature_name_map = {
    "bweight": "Bweight",
    "eventyear": "Eventyear",
    "bornyear": "Bornyear",
    "total": "Total",
    "age": "Age",
    "error": "Residual",
    "nation": "Nation",
    "olympic_bw": "Olympic category"
}

new_feature_names = [feature_name_map[k] for k in feature_groups.keys()]

# 2) Summary plot (저장 포함)
plt.figure()
shap.summary_plot(
    grouped_shap.values, 
    features=grouped_shap, 
    feature_names=new_feature_names,
    show=False   # 바로 화면에 출력하지 않고 제어
)

plt.tight_layout()
plt.savefig("shap_summary_plot.png", dpi=300)
plt.close()

print("✅ SHAP summary plot saved as shap_summary_plot.png")

# ---- Cell 22 ----
import numpy as np
import matplotlib.pyplot as plt

def plot_cm(ax, cm, title, class_names=("Not-sanctioned","Sanctioned"), cmap="Blues"):
    im = ax.imshow(cm, cmap=cmap, aspect='equal')
    ax.set_title(title, pad=10)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks([0,1], class_names)
    ax.set_yticks([0,1], class_names)

    # annotate counts
    for i in range(2):
        for j in range(2):
            v = int(cm[i, j])
            text_color = "white" if im.norm(v) > 0.5 else "black"
            ax.text(j, i, f"{v}", ha="center", va="center", fontsize=12, color=text_color)

    ax.set_xticks(np.arange(-.5, 2, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 2, 1), minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=1, alpha=0.5)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return im

def fig_two_cms(cm_internal, cm_external, titles=("Internal Test Set","External Validation Set")):
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6), constrained_layout=True)
    im1 = plot_cm(axes[0], cm_internal, titles[0])
    im2 = plot_cm(axes[1], cm_external, titles[1])
    cbar = fig.colorbar(im2, ax=axes.ravel().tolist(), shrink=0.9)
    cbar.set_label("Count")
    return fig, axes

# ===== Internal ([@best]에서 준 것) =====
# CM = [[TN=235, FP=18],[FN=8, TP=7]]
cm_internal = np.array([[235, 18],
                        [  8,  7]])

# ===== External (error_column 테이블 @0.5 기준) =====
# TN=796, FP=2, FN=3, TP=0
cm_external = np.array([[796, 36],
                        [  2, 1]])

fig, axes = fig_two_cms(cm_internal, cm_external)
plt.savefig('confusion_matrix.png', dpi=300)
plt.show()


if __name__ == '__main__':
    print('This module was auto-converted from a notebook. Import functions as needed.')
