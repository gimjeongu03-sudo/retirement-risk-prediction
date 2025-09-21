import pandas as pd
import numpy as np
import pickle
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

# ===============================
# 0) 데이터 로드
# ===============================
PATH = "퇴직연금_통합데이터.csv"  # <-- 실제 경로로 교체
df = pd.read_csv(PATH, encoding="utf-8-sig")

id_col = "사업자번호"
flow_df = df.copy()

# ===============================
# 1) 설명 함수
# ===============================
def make_explanation(row):
    reasons = []
    if row["적립률_3년변화"] < 0 and row["납입이행률_3년변화"] < 0:
        reasons.append("최근 3년간 적립률과 납입이행률이 동시에 하락했습니다.")
    if row["적립률_3년변화"] <= -0.1:
        reasons.append(f"적립률이 최근 3년간 {row['적립률_3년변화']:.2f}p 하락했습니다.")
    if row["납입이행률_3년변화"] <= -0.1:
        reasons.append(f"납입이행률이 최근 3년간 {row['납입이행률_3년변화']:.2f}p 하락했습니다.")
    if row["적립률_3년변화"] < 0 and row["가입자수_3년변화"] < 0:
        reasons.append("적립률과 가입자수가 동시에 감소했습니다.")
    if row["준수비율_3년변화"] <= -0.1:
        reasons.append(f"준수비율이 최근 3년간 {row['준수비율_3년변화']:.2f}p 하락했습니다.")
    if row["적립률_3년변동폭"] > 0.2:
        reasons.append(f"적립률 변동폭이 {row['적립률_3년변동폭']:.2f}로 큰 편입니다.")
    if row["납입이행률_3년변동폭"] > 0.2:
        reasons.append(f"납입이행률 변동폭이 {row['납입이행률_3년변동폭']:.2f}로 큰 편입니다.")

    if not reasons:
        if row["y_pred_final"] == 0:
            return f"모델이 리스크 확률 {row['p_risk']:.1%}로 판정했습니다."
        else:
            return f"모델이 정상 확률 {row['p_normal']:.1%}로 판정했습니다."
    return " / ".join(reasons)

# ===============================
# 2) 학습 데이터 준비
# ===============================
train_df = flow_df[flow_df["예측연도"] < 2024].copy()
test_df  = flow_df[flow_df["예측연도"] == 2024].copy()

target_col = "재정검증결과_binary"
feature_cols = [c for c in train_df.columns if c not in [id_col,"업체명","예측연도",target_col]]

X_train, X_valid, y_train, y_valid = train_test_split(
    train_df[feature_cols], train_df[target_col],
    test_size=0.2, random_state=42, stratify=train_df[target_col]
)
med_final = X_train.median()

# ===============================
# 3) 모델 학습
# ===============================
model_final = XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    eval_metric="logloss"
)

model_final.fit(X_train.fillna(med_final), y_train,
    eval_set=[(X_valid.fillna(med_final), y_valid)],
    verbose=False
)

print("=== VALID 성능 (2023) ===")
print(classification_report(y_valid, model_final.predict(X_valid.fillna(med_final))))
print("AUC:", roc_auc_score(y_valid, model_final.predict_proba(X_valid.fillna(med_final))[:,1]))

# ===============================
# 4) 모델 & 메타 저장
# ===============================
model_final.save_model("final_model.json")
print("✅ final_model.json 저장 완료")

meta = {"median": med_final, "feature_cols": feature_cols, "rules": "Rule1~6 후처리 적용"}
with open("final_meta.pkl", "wb") as f:
    pickle.dump(meta, f)
print("✅ final_meta.pkl 저장 완료")

# ===============================
# 5) 2024년 예측
# ===============================
X_pred_2024 = test_df[feature_cols].fillna(med_final)
test_df["p_normal"] = model_final.predict_proba(X_pred_2024)[:,1]
test_df["p_risk"]   = 1 - test_df["p_normal"]
test_df["y_pred_model"] = (test_df["p_normal"] >= 0.5).astype(int)

# Rule 보정
combined_mask = (
    ((test_df["적립률_3년변화"] < 0) & (test_df["납입이행률_3년변화"] < 0)) |
    ((test_df["적립률_3년변화"] <= -0.1) | (test_df["납입이행률_3년변화"] <= -0.1)) |
    ((test_df["적립률_3년변화"] < 0) & (test_df["가입자수_3년변화"] < 0)) |
    (test_df["준수비율_3년변화"] <= -0.1) |
    (test_df["적립률_3년변동폭"] > 0.2) |
    (test_df["납입이행률_3년변동폭"] > 0.2)
)
test_df["y_pred_final"] = test_df["y_pred_model"]
test_df.loc[combined_mask, "y_pred_final"] = 0

# 업체명 매핑
if "업체명" in df.columns:
    id_to_name = df[[id_col,"업체명"]].drop_duplicates().set_index(id_col)["업체명"].to_dict()
    test_df["업체명"] = test_df[id_col].map(id_to_name)

# 설명 추가
test_df["explanation"] = test_df.apply(make_explanation, axis=1)

# 저장
cols_to_export = [id_col,"업체명","예측연도","p_normal","p_risk","y_pred_final","explanation"]
test_df[cols_to_export].to_csv("prediction_2024.csv", index=False, encoding="utf-8-sig")
print("✅ prediction_2024.csv 저장 완료")
