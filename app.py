"""고객 입력으로 동일 모델의 연체 확률과 SHAP을 확인한다."""

from pathlib import Path
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import shap
import streamlit as st
from finance import FEATURES
from analysis_shap import explain

st.set_page_config(page_title="대출 위험 예측 근거", layout="wide")
st.title("대출 위험 예측 근거")
st.caption("가상 금융 데이터로 학습한 모델의 연체 확률과 변수별 기여도를 확인합니다.")


@st.cache_resource
def classifier():
    """이 미션에 연결한 동일 분류 모델을 불러온다."""
    return joblib.load(Path(__file__).parent / "models/classifier.joblib")


with st.form("customer"):
    left, right = st.columns(2)
    with left:
        age = st.number_input("나이", 20, 100, 35)
        income = st.number_input("연 소득 (만원)", 0, 100000, 5000, 100)
        spending = st.number_input("소비 점수", 1, 100, 50)
    with right:
        debt = st.number_input("소득 대비 부채 비율", 0.0, 1.0, 0.4, 0.01)
        cards = st.number_input("카드 수", 0, 30, 3)
        overdue = st.number_input("최근6개월 연체 횟수", 0, 30, 0)
    submitted = st.form_submit_button("예측 근거 확인")
if submitted:
    raw = pd.DataFrame(
        [[age, income, spending, debt, cards, overdue]], columns=FEATURES
    )
    explanation, probability, residual = explain(classifier(), raw)
    st.metric("예측 연체 확률", f"{probability[0]:.2%}")
    st.write("모형 기준: 거절" if probability[0] >= 0.5 else "모형 기준: 승인")
    st.caption("기준 확률0.5. 오른쪽 기여는 연체 확률을 높이고 왼쪽 기여는 낮춥니다.")
    shap.plots.waterfall(explanation[0], max_display=6, show=False)
    st.pyplot(plt.gcf())
    plt.close("all")
    st.dataframe(
        pd.DataFrame(
            {
                "변수": FEATURES,
                "입력": raw.iloc[0].to_numpy(),
                "확률 기여도": explanation.values[0],
            }
        ),
        hide_index=True,
    )
