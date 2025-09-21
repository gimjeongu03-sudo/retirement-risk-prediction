import os
import numpy as np
import pandas as pd
import streamlit as st
from xgboost import XGBClassifier
import pickle
from auth import init_default_users, is_logged_in, show_login_page, show_logout_button

# ===============================
# 🔧 페이지 기본 설정
# ===============================
st.set_page_config(page_title="퇴직연금 리스크 예측", layout="wide")

# ===============================
# 🔐 인증 확인
# ===============================
init_default_users()

if not is_logged_in():
    show_login_page()
    st.stop()

show_logout_button()

# ===============================
# 🔧 유틸리티 함수
# ===============================
@st.cache_resource
def load_model():
    model = XGBClassifier()
    model.load_model("final_model.json")
    with open("final_meta.pkl", "rb") as f:
        meta = pickle.load(f)
    return model, meta

@st.cache_data
def load_pred_data(path: str, mtime: float):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df

@st.cache_data
def load_raw_data():
    raw = pd.read_csv("퇴직연금_통합_데이터_2014_2024.csv")
    raw.columns = raw.columns.str.strip()
    raw["기준연도"] = raw["기준연도"].astype(str).str[:4].astype(int)
    return raw

def highlight_status(val): return "background-color: #FFCCCC" if val == "리스크" else "background-color: #CCFFCC"
def highlight_source(val): return "background-color: #FFF59D" if val == "Rule" else ""
def highlight_risk_flag(val): return "background-color: #FFD6D6" if val == "⚠️" else ""

def classify_source(row):
    return "Rule" if row.get("y_pred_model") == 1 and row.get("y_pred_final") == 0 else "Model"

def format_currency(val):
    try:
        if pd.isna(val):
            return "-"
        return f"{int(val):,}원"
    except:
        return "-"

def format_rate(x): return f"{x*100:.1f}%" if pd.notna(x) else "-"

def format_percent(x):
    try:
        return f"{x*100:.0f}%" if pd.notna(x) else "-"
    except:
        return "-"

def extract_rule_flags(row):
    rules = []
    if row.get("rule1_flag"): rules.append("Rule1: 적립률+납입 동시 하락")
    if row.get("rule2_flag"): rules.append("Rule2: 적립률 또는 납입이행률 -0.1 이하")
    if row.get("rule3_flag"): rules.append("Rule3: 적립률+가입자수 동시 하락")
    return rules

def peer_text_simple(row):
    if pd.isna(row.get("peer_group")): return "비슷한 규모의 비교 그룹 데이터 없음"
    if pd.isna(row.get("그룹_평균_적립금")): return "비교 그룹 평균 데이터 없음"
    comp = "비슷한"
    if pd.notna(row.get("적립금")) and pd.notna(row.get("그룹_평균_적립금")):
        comp = "낮은" if row["적립금"] < row["그룹_평균_적립금"] else "높은"
    return f"이 기업은 같은 규모 기업 대비 적립금이 {comp} 수준입니다."

# ===============================
# 📈 리밸런싱 & 납부 전략 계산
# ===============================
def simulate_payment_strategy(amount, months, payments, principal_ratio, non_principal_ratio):
    monthly_principal_rate = 0.03 / 12
    monthly_non_principal_rate = 0.10 / 12
    total = 0
    pay_amount = amount / payments
    for m in range(1, months + 1):
        total = total * ((1 + monthly_principal_rate) * principal_ratio +
                         (1 + monthly_non_principal_rate) * non_principal_ratio)
        if m % (months // payments) == 0:
            total += pay_amount
    return total

def dynamic_rebalance_ratio(p_risk):
    if p_risk >= 0.8: return 0.8, 0.2
    elif p_risk >= 0.6: return 0.7, 0.3
    elif p_risk >= 0.4: return 0.6, 0.4
    else: return 0.5, 0.5

def compare_multi_payment_strategies(amount, row):
    try:
        amount = int(str(amount).replace(",", "").replace("원", ""))
    except:
        return None
    base_principal = row.get("원리금보장비중_3년평균")
    base_non = row.get("실적배당비중_3년평균")
    if pd.isna(base_principal): base_principal = 0.5
    if pd.isna(base_non): base_non = 0.5

    p_risk = row.get("p_risk", 0.5)
    rebal_non, rebal_principal = dynamic_rebalance_ratio(p_risk)

    months = 12
    target = amount
    table = []

    for label, n in [("월납(12회)", 12), ("분기납부(3회)", 3), ("반기납부(2회)", 2), ("연말일시납(1회)", 1)]:
        base_amt = simulate_payment_strategy(amount, months, n, base_principal, base_non)
        rebal_amt = simulate_payment_strategy(amount, months, n, rebal_principal, rebal_non)
        diff = int(rebal_amt - base_amt)

        if rebal_amt >= target and base_amt < target:
            faster = f"리밸런싱으로 목표 조기 달성 (+{diff:,}원)"
        elif diff > 0:
            faster = f"리밸런싱으로 더 빠른 회복 예상 (+{diff:,}원)"
        else:
            faster = "-"

        table.append([
            label,
            f"{int(base_amt):,}원 ({base_principal*100:.0f}% 원리금/{base_non*100:.0f}% 비원리금)",
            f"{int(rebal_amt):,}원 ({rebal_principal*100:.0f}% 원리금/{rebal_non*100:.0f}% 비원리금)",
            faster
        ])

    df = pd.DataFrame(table, columns=["납부방식", "기존전략 12개월 예상금액", "리밸런싱 12개월 예상금액", "비고"])
    return df

def payment_plan(amount):
    try:
        amount = int(str(amount).replace(",", "").replace("원", ""))
    except:
        return None
    return pd.DataFrame({
        "납부시점": ["월납(12회)", "분기납부(3회)", "반기납부(2회)", "연말일시납(1회)"],
        "회당납부액": [f"{amount//12:,}원", f"{amount//3:,}원", f"{amount//2:,}원", f"{amount:,}원"]
    })

# ===============================
# 📊 3년 예상 충당금 시나리오 표
# ===============================
def format_currency(val, unit="원"):
    try:
        if pd.isna(val):
            return "-"
        if unit == "억원":
            return f"{val/100000000:.2f}억원"
        return f"{int(val):,}원"
    except:
        return "-"
def create_reserve_scenario_table(row):
    # 2024년 최소 적립금 가져오기
    min_reserve_2024 = row.get("최소적립금(적립기준액)", 0) * 1.1

    if pd.isna(min_reserve_2024) or min_reserve_2024 <= 0:
        st.warning("최소 적립금 데이터가 없어 시나리오 표를 생성할 수 없습니다.")
        return None

    # 시나리오 데이터 계산
    years = [2024, 2025, 2026]
    scenario_data = []

    # 2024년 계산
    reserve_2024 = min_reserve_2024
    actual_reserve_2024 = row.get("적립금", 0)
    ratio_2024 = (actual_reserve_2024 / reserve_2024) if actual_reserve_2024 > 0 else 0
    deficit_ratio_2024 = 1 - ratio_2024
    reserve_ratio_2024 = deficit_ratio_2024 / 3
    expected_reserve_2024 = actual_reserve_2024 * reserve_ratio_2024

    # if reserve_ratio_2024 < 0 :
    #     st.warning("충당할 금액이 없습니다.")
    #     return None

    scenario_data.append({
        "년도": "2024년",
        "최소적립금": format_currency(reserve_2024, "억원"),
        "적립금": format_currency(actual_reserve_2024, "억원"),
        "적립비율": format_percent(ratio_2024),
        "부족비율": format_percent(deficit_ratio_2024),
        "충당비율": format_percent(reserve_ratio_2024),
        "예상충당금": format_currency(expected_reserve_2024, "억원")

    })

    # 2025년 계산 (최소 적립금 10% 증가)
    reserve_2025 = reserve_2024 * 1.1
    # 2025년 충당금을 납부했다고 가정한 적립금
    adjusted_reserve_2025 = actual_reserve_2024 * (1 + reserve_ratio_2024)
    ratio_2025 = ( adjusted_reserve_2025 / reserve_2025) if adjusted_reserve_2025 > 0 else 0
    deficit_ratio_2025 = 1 - ratio_2025
    reserve_ratio_2025 = deficit_ratio_2025 / 3
    expected_reserve_2025 = adjusted_reserve_2025 * reserve_ratio_2025

    scenario_data.append({
        "년도": "2025년",
        "최소적립금": format_currency(reserve_2025, "억원"),
        "적립금": format_currency(adjusted_reserve_2025, "억원"),
        "적립비율": format_percent(ratio_2025),
        "부족비율": format_percent(deficit_ratio_2025),
        "충당비율": format_percent(reserve_ratio_2025),
        "예상충당금": format_currency(expected_reserve_2025, "억원")
    })

    # 2026년 계산 (최소 적립금 10% 증가)
    reserve_2026 = reserve_2025 * 1.1
    # 2026년 충당금을 납부했다고 가정한 적립금
    adjusted_reserve_2026 = adjusted_reserve_2025 * (1 + reserve_ratio_2025)
    ratio_2026 = (adjusted_reserve_2026 / reserve_2026) if adjusted_reserve_2026 > 0 else 0
    deficit_ratio_2026 = 1 - ratio_2026
    reserve_ratio_2026 = deficit_ratio_2026 / 3
    expected_reserve_2026 = adjusted_reserve_2026 * reserve_ratio_2026

    scenario_data.append({
        "년도": "2026년",
        "최소적립금": format_currency(reserve_2026, "억원"),
        "적립금": format_currency(adjusted_reserve_2026, "억원"),
        "적립비율": format_percent(ratio_2026),
        "부족비율": format_percent(deficit_ratio_2026),
        "충당비율": format_percent(reserve_ratio_2026),
        "예상충당금": format_currency(expected_reserve_2026, "억원")
    })

    return pd.DataFrame(scenario_data)
# ===============================
# 📊 데이터 로드
# ===============================
model, meta = load_model()
pred_path = "prediction_2024_all.csv"
df = load_pred_data(pred_path, os.path.getmtime(pred_path))
raw_df = load_raw_data()

# ===============================
# 🔍 검색 기능
# ===============================
st.title("퇴직연금 재정검증 리스크 예측 결과 (2024)")
st.markdown("---")
search_query = st.text_input("🔍 사업자번호 또는 업체명 검색")
if search_query:
    result = df[
        df["사업자번호"].astype(str).str.contains(search_query, na=False) |
        df["업체명"].astype(str).str.contains(search_query, na=False)
    ]
    if result.empty:
        st.warning("검색 결과 없음")
else:
    result = df.copy()

# ===============================
# 🏷️ 컬럼 가공
# ===============================
if "p_risk" in result.columns:
    result["리스크확률"] = result["p_risk"].apply(format_percent)
if "p_normal" in result.columns:
    result["정상확률"] = result["p_normal"].apply(format_percent)

result["최종판정"] = result["y_pred_final"].map({0: "리스크", 1: "정상"})
if "부족액_예상" in result.columns: result["부족액_예상"] = result["부족액_예상"].apply(format_currency)
result["분류출처"] = result.apply(classify_source, axis=1)
risk_flags = result.get("rule1_flag", False) | result.get("rule2_flag", False) | result.get("rule3_flag", False)
result["룰신호"] = risk_flags.map({True: "⚠️", False: ""})

rename_cols = {"적립률_t-1": "1년전_적립률", "적립률_t-2": "2년전_적립률", "적립률_t-3": "3년전_적립률"}
for old, new in rename_cols.items():
    if old in result.columns:
        result.rename(columns={old: new}, inplace=True)
        result[new] = result[new].apply(format_rate)

# ===============================
# 📈 전체 통계
# ===============================
st.markdown("### 전체 통계")
col1, col2, col3 = st.columns(3)
col1.metric("전체 기업 수", len(result))
col2.metric("리스크 기업 수", (result["최종판정"] == "리스크").sum())
col3.metric("정상 기업 수", (result["최종판정"] == "정상").sum())
if st.checkbox("🚨 리스크 기업만 보기"):
    result = result[result["최종판정"] == "리스크"]

# ===============================
# 📋 검색 결과 테이블
# ===============================
st.subheader("기업별 예측 결과")
show_cols = ["사업자번호", "업체명", "리스크확률", "정상확률", "최종판정",
             "룰신호", "분류출처", "3년전_적립률", "2년전_적립률", "1년전_적립률"]
for col in show_cols:
    if col not in result.columns: result[col] = None
styled_df = (result[show_cols]
             .style
             .applymap(highlight_status, subset=["최종판정"])
             .applymap(highlight_source, subset=["분류출처"])
             .applymap(highlight_risk_flag, subset=["룰신호"]))
st.dataframe(styled_df, use_container_width=True)

# ===============================
# 📄 상세보기
# ===============================
if not result.empty:
    row = result.iloc[0]
    st.markdown("---")
    st.subheader(f"🏢 {row.get('업체명','-')} ({row.get('사업자번호','-')})")

    # 확률/판정
    c1, c2, c3 = st.columns([1, 1, 2])
    with c1: st.metric("리스크 확률", f"{row.get('p_risk', 0)*100:.0f}%")
    with c2: st.metric("정상 확률", f"{row.get('p_normal', 0)*100:.0f}%")
    with c3:
        if row.get("최종판정") == "리스크": st.error("🚨 리스크")
        else: st.success("✅ 적정")
        if row.get("룰신호") == "⚠️":
            st.warning("⚠️ 유의가 필요한 업체입니다.")
            for rule in extract_rule_flags(row): st.markdown(f"- {rule}")

    # 최근 3년 적립률
    st.markdown("### 📊 최근 3년 적립률")
    rate_table = pd.DataFrame({"연도": ["3년 전", "2년 전", "1년 전"],
                               "적립률": [row.get("3년전_적립률","-"), row.get("2년전_적립률","-"), row.get("1년전_적립률","-")]})
    st.table(rate_table)

    # 설명
    if row.get("최종판정") == "리스크" and "explanation" in row:
        st.markdown("### 📌 설명 근거")
        for line in str(row["explanation"]).split(" / "):
            if line.strip(): st.markdown(f"- ✔ {line}")

    # 트렌드
    st.markdown("---")
    st.markdown("### 📈 연도별 지표 트렌드")
    try:
        firm_data = raw_df[(raw_df["사업자번호"] == row.get("사업자번호")) | (raw_df["업체명"] == row.get("업체명"))].copy()
        if not firm_data.empty:
            firm_data["적립률"] = firm_data["적립금"] / firm_data["최소적립금(적립기준액)"]
            firm_data["준수비율"] = firm_data["평가적립금합계"] / firm_data["계속기준책임준비금"]
            firm_data["납입이행률"] = firm_data["부담금납입액"] / firm_data["부담금산정액"]
            st.line_chart(firm_data[["기준연도","적립률","준수비율","납입이행률"]].set_index("기준연도"))
        else: st.warning("해당 기업의 원본 연도별 데이터 없음")
    except Exception as e:
        st.warning(f"트렌드 차트 오류: {e}")

    # 🔍 리스크 대응 전략
    if row.get("최종판정") == "리스크" and "부족액_예상" in row:
        st.markdown("---")
        st.markdown("## 🔍 예상부족액 리스크 대응 전략")
        # 🔥 예상부족액 표시 추가
        st.markdown(f"**📌 예상부족액:** {row.get('부족액_예상', '-')}")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 💰 분할납부 계획")
            plan_df = payment_plan(row["부족액_예상"])
            if plan_df is not None: st.table(plan_df)
        with col2:
            st.markdown("### 📈 리밸런싱 전략 (납부방식별 비교)")
            st.caption("원리금: 연3%, 비원리금: 연10% 가정")
            multi_table = compare_multi_payment_strategies(row["부족액_예상"], row)
            if multi_table is not None:
                st.table(multi_table)

    row = result.iloc[0]

    st.write("### 전체 원본 데이터")
    firm_data = raw_df[
        ((raw_df["사업자번호"] == row.get("사업자번호")) | (raw_df["업체명"] == row.get("업체명"))) &
        (raw_df["결산년월"] == "2023-12-01")
        ].copy()
    st.dataframe(firm_data, use_container_width=True)
    st.subheader(f"🏢 {row.get('업체명','-')} ({row.get('사업자번호','-')})")

    # 3년 예상 충당금 시나리오 표 추가
    st.markdown("---")
    st.markdown("### 📊 3년 예상 충당금 시나리오")
    scenario_df = create_reserve_scenario_table(firm_data.iloc[0])
    if scenario_df is not None:
        st.table(scenario_df)

