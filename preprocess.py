# -*- coding: utf-8 -*-
"""
Auto-generated from npj_preprocess-Copy2.ipynb
Generated: 2025-10-15T07:12:26
"""

# ---- Cell 0 ----
import os
import pandas as pd

# ===== 0) 데이터 로드 =====
df = pd.read_csv('lifting_npj_raw.csv', encoding='utf-8')
print(df.head(3))

# ===== 1) 기본 정리: 파싱/타입 변환 및 age 계산 =====
df["bornyear"]  = pd.to_datetime(df["born"], format="%d.%m.%Y", errors="coerce").dt.year
df["eventyear"] = pd.to_numeric(df["eventyear"], errors="coerce")
df["doping"]    = pd.to_numeric(df["doping"], errors="coerce")
df["age"]       = df["eventyear"] - df["bornyear"]

# ===== 2) 이벤트 카테고리 컬럼 생성 =====
WORLD_KWS = [
    "OLYMPIC", "WORLD CHAMPIONSHIP", "WORLD CHAMPIONSHIPS", "WORLD CUP",
    "UNIVERSIADE", "UNIVERSIAD", "UNIVERSITY WORLD CUP", "WORLD GAMES"
]
CONTINENTAL_KWS = [
    "ASIAN", "EUROPEAN", "OCEANIAN", "OCEANIA", "AFRICAN",
    "PAN-AMERICAN", "PAN AMERICAN", "SOUTH AMERICAN", "CENTRAL AMERICAN",
    "MEDITERRANEAN", "ARAB", "BALKAN", "COMMONWEALTH", "WEST ASIAN", "EAST ASIAN"
]
NATIONAL_KWS = [
    "NATIONAL", "CHAMPIONSHIP OF", "CHAMPIONSHIPS OF", "CUP OF", "GRAND PRIX OF"
]

def categorize_event(name: str) -> str:
    if not isinstance(name, str):
        return "etc"
    u = name.upper()
    if any(kw in u for kw in WORLD_KWS):
        return "world"
    if any(kw in u for kw in CONTINENTAL_KWS):
        return "continent"
    if any(kw in u for kw in NATIONAL_KWS):
        return "nation"
    return "etc"

df["event_category"] = df["event"].apply(categorize_event)

# ===== 3) 결측치 및 total=0 제거 =====
df["total"] = pd.to_numeric(df["total"], errors="coerce")
df = df.dropna(subset=["nation", "bornyear", "eventyear", "doping", "total"])
df = df[df["total"] > 0].copy()

# ===== 4) athlete_id 생성 (nation + bornyear 기준) =====
codes, uniques = pd.factorize(list(zip(df["nation"], df["bornyear"])))
df["athlete_id"] = codes + 1  # 1부터 시작

# ===== 3.5) 3경기 이상 출전 선수만 남기기 (전체 기준) =====
counts = df.groupby("athlete_id").size()
valid_ids = counts[counts >= 3].index
df = df[df["athlete_id"].isin(valid_ids)].copy()

print("3경기 이상 필터 적용 후 기록 수:", len(df))
print("3경기 이상 필터 적용 후 선수 수:", df["athlete_id"].nunique())

# ===== 5) 선수 단위 테이블 생성 (마지막 출전 연도, 도핑 여부) =====
last_comp = df.groupby(["nation", "bornyear"])["eventyear"].max().reset_index()
doping_athletes = df.groupby(["nation", "bornyear"])["doping"].max().reset_index()

athlete_info = pd.merge(last_comp, doping_athletes, on=["nation", "bornyear"], how="left")
athlete_info = athlete_info.merge(
    df[["nation", "bornyear", "athlete_id"]].drop_duplicates(),
    on=["nation", "bornyear"],
    how="left"
)

# ===== 6) Train/Test 분리 (train: ≤2018, test: ≥2019) =====
train_athletes = athlete_info[athlete_info["eventyear"] <= 2018]
test_athletes  = athlete_info[athlete_info["eventyear"] >= 2019]

# ===== 7) 결과 출력 =====
print("전체 선수 수:", athlete_info.shape[0])
print("Train set 선수 수:", train_athletes.shape[0])
print("Test set 선수 수:", test_athletes.shape[0])
print("Train 도핑 선수 수:", (train_athletes["doping"] == 1).sum())
print("Test 도핑 선수 수:", (test_athletes["doping"] == 1).sum())

# ===== 8) Train/Test 행 추출 (선수 키로 조인) =====
train_keys = train_athletes[["nation", "bornyear"]].drop_duplicates()
test_keys  = test_athletes[["nation", "bornyear"]].drop_duplicates()

df_train = df.merge(train_keys, on=["nation", "bornyear"], how="inner")
df_test  = df.merge(test_keys,  on=["nation", "bornyear"], how="inner")

# ===== 9) 각 세트에서 cross-sectional(마지막 1행) / longitudinal(그 전 모든 행) 분리 =====
def split_cross_longitudinal(df_subset):
    # 동일 연도 복수 행이 있으면 원본 행 순서상 '마지막'을 cross로 선택
    tmp = df_subset.reset_index().rename(columns={"index": "orig_idx"})
    tmp = tmp.sort_values(["nation", "bornyear", "eventyear", "orig_idx"])
    last_idx = tmp.groupby(["nation", "bornyear"]).tail(1).index
    cross = tmp.loc[last_idx].copy()
    long  = tmp.drop(last_idx).copy()
    cross.drop(columns=["orig_idx"], inplace=True)
    long.drop(columns=["orig_idx"], inplace=True)
    return cross, long

train_cross, train_long = split_cross_longitudinal(df_train)
test_cross,  test_long  = split_cross_longitudinal(df_test)

# ===== 10) CSV로 저장 =====
os.makedirs("/mnt/data", exist_ok=True)
train_cross_path = "train_cross_sectional_se.csv"
train_long_path  = "train_longitudinal_se.csv"
test_cross_path  = "test_cross_sectional_se.csv"
test_long_path   = "test_longitudinal_se.csv"

train_cross.to_csv(train_cross_path, index=False)
train_long.to_csv(train_long_path, index=False)
test_cross.to_csv(test_cross_path, index=False)
test_long.to_csv(test_long_path, index=False)

# ===== 11) (옵션) 저장한 CSV 재로드하여 요약 확인 =====
train_cross = pd.read_csv(train_cross_path)
train_long  = pd.read_csv(train_long_path)
test_cross  = pd.read_csv(test_cross_path)
test_long   = pd.read_csv(test_long_path)

print("=== Summary ===")
print("Train cross-sectional 행:", train_cross.shape[0])   # 보통 Train 선수 수와 동일
print("Train longitudinal 행:", train_long.shape[0])
print("Test  cross-sectional 행:", test_cross.shape[0])    # 보통 Test 선수 수와 동일
print("Test  longitudinal 행:", test_long.shape[0])

# ===== 12) 간단 무결성 체크 =====
# cross-sectional의 행 수는 athlete 단위 고유 수와 일치해야 함
assert train_cross[["nation", "bornyear"]].drop_duplicates().shape[0] == train_athletes.shape[0], "Train cross-sectional 개수 불일치"
assert test_cross[["nation", "bornyear"]].drop_duplicates().shape[0] == test_athletes.shape[0], "Test cross-sectional 개수 불일치"

print("무결성 검사 통과 ✅")
print("파일 저장 위치:")
print(train_cross_path)
print(train_long_path)
print(test_cross_path)
print(test_long_path)

# ---- Cell 1 ----
test_keys = test_athletes[["nation", "bornyear"]].drop_duplicates()
df_test = df.merge(test_keys, on=["nation", "bornyear"], how="inner")

# ===== CSV 저장 (하나의 파일로만) =====
import os
os.makedirs("/mnt/data", exist_ok=True)
df_test.to_csv("test_full.csv", index=False)

print("Test set 전체 행:", df_test.shape[0])
print("저장 완료 -> /mnt/data/test_full.csv")


if __name__ == '__main__':
    print('This module was auto-converted from a notebook. Import functions as needed.')
