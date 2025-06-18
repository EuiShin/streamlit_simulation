import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.optimize import minimize_scalar

# 페이지 설정
st.set_page_config(layout="wide")
st.title("마케팅 효율 시뮬레이션")


# CTR 추정용 데이터 업로드
st.sidebar.header("CTR 추정 데이터 (입력)")
st.sidebar.write('보유 데이터를 입력하세요.')

def get_ctr(impression):
    return np.round(impression ** (-2/7), 5)

# 기본 제공 데이터
table_impressions = np.array([1000, 10000, 100000, 1000000])
df = pd.DataFrame({
    '노출수':table_impressions,
})
df['CTR(%)'] = df['노출수'].apply(get_ctr)*100
df = st.sidebar.data_editor(df, hide_index=True)


# 입력 받기
impressions_max = st.sidebar.number_input("최대 노출수", value=1000000)
cvr = st.sidebar.number_input("전환율 (CVR)", value=0.1)
arpu = st.sidebar.number_input("ARPU (1인당 매출)", value=10000)
sending_cost = st.sidebar.number_input("발송 단가 (1건당 비용)", value=20)

# 데이터 수정시 변경 함수
# 모델 함수 정의: CTR = a * x^b
def ctr_model(x, a, b):
    return a * x**b

params, _ = curve_fit(ctr_model, df['노출수'], df['CTR(%)']/100)
a_fit, b_fit = params

impressions = np.linspace(100, impressions_max, 100)
ctr_func = ctr_model(impressions, a_fit, b_fit)

# 최적화 대상 함수 정의 (이익을 -로 바꿔서 최소화 => 최대화 문제로 전환)
def profit_objective(x):
    ctr = ctr_model(x, a_fit, b_fit)
    clicks = x * ctr
    conversions = clicks * cvr
    revenue = conversions * arpu
    cost = x * sending_cost
    profit = revenue - cost
    return -profit  # 이익 최대화 → 음수 최소화

# 최적화 수행 (연속 변수로 최적 노출 수 찾기)
opt_result = minimize_scalar(profit_objective, bounds=(100, impressions_max+1), method='bounded')
opt_x = int(opt_result.x)
opt_profit = -opt_result.fun
opt_ctr = ctr_model(opt_x, a_fit, b_fit)
opt_roas = (opt_ctr * opt_x * cvr * arpu) / (opt_x * sending_cost) * 100

st.subheader("📈 최대 이익 지점")
col4, col5, col6 = st.columns(3)
col4.metric(label="최적 노출수", value=f"{opt_x:,} 건")
col5.metric(label="최대 이익", value=f"{int(opt_profit):,} 원")
col6.metric(label="ROAS", value=f"{opt_roas:.1f} %")


# 계산
clicks = impressions * ctr_func
conversions = clicks * cvr
costs = impressions * sending_cost
revenues = conversions * arpu
profits = revenues - costs
roas = (revenues / costs) * 100

# 결과 DataFrame
results = pd.DataFrame({
    '노출수': impressions,
    'CTR': ctr_func,
    '클릭수': clicks,
    '전환수': conversions,
    '비용': costs,
    '전환값': revenues,
    '이익': profits,
    'ROAS': roas
}).set_index('노출수')

# 시각화
st.subheader("시각화")
col1, col2, col3 = st.columns(3)
col1.markdown('**CTR**')
col1.line_chart(results['CTR'], height=250, use_container_width=True)
col2.markdown('**클릭수**')
col2.line_chart(results['클릭수'], height=250, use_container_width=True)
col3.markdown('**전환수**')
col3.line_chart(results['전환수'], height=250, use_container_width=True)

col1, col2, col3 = st.columns(3)
col1.markdown('**수익, 비용**')
col1.line_chart(results[['전환값', '비용']], height=250, use_container_width=True)
col2.markdown('**이익**')
col2.line_chart(results['이익'], height=250, use_container_width=True)
col3.markdown('**ROAS**')
col3.line_chart(results['ROAS'], height=250, use_container_width=True)


