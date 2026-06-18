import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from groq import Groq

# 페이지 설정
st.set_page_config(
    page_title="119 신고 긴급도 AI 분류 시스템",
    page_icon="🚨",
    layout="wide"
)

# Groq 클라이언트
client = Groq(api_key=st.secrets["GROQ_API_KEY"])

# 긴급도 분류 함수
def classify_urgency(text):
    prompt = f"""당신은 119 신고 접수 전문가입니다.
아래는 실제 119 신고 전화에서 신고자가 한 말입니다.
신고자의 말만 읽고 긴급도를 판단하세요.

신고 내용: {text}

반드시 아래 형식으로만 답하세요 (다른 말 금지):
긴급도: [상/중/하 중 하나만]
근거: [15자 이내로 핵심 이유만]
권장출동: [즉시출동/일반출동/비긴급처리 중 하나만]"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=100
    )
    return response.choices[0].message.content.strip()

def parse_result(result):
    urgency, reason, action = "알 수 없음", "", ""
    for line in result.split('\n'):
        if '긴급도:' in line:
            for v in ['상', '중', '하']:
                if v in line:
                    urgency = v
        elif '근거:' in line:
            reason = line.replace('근거:', '').strip()
        elif '권장출동:' in line:
            action = line.replace('권장출동:', '').strip()
    return urgency, reason, action

# 메인 헤더
st.title("🚨 119 신고 긴급도 AI 자동 분류 시스템")
st.markdown("##### 제주 도민의 골든타임을 지키는 AI")
st.divider()

tab1, tab2, tab3, tab4 = st.tabs([
    "🔴 실시간 신고 분류기",
    "📊 119 신고 데이터 분석",
    "🤖 AI 성능 분석",
    "💡 탐구 결론"
])

# ── 탭 1: 실시간 분류기 ──
with tab1:
    st.subheader("신고 내용을 입력하면 AI가 즉시 긴급도를 판단합니다")

    if 'history' not in st.session_state:
        st.session_state.history = []

    text_input = st.text_area("신고 텍스트 입력", height=120,
                               placeholder="예: 할아버지가 갑자기 쓰러지셨어요. 숨을 안 쉬는 것 같아요.")

    if st.button("🔍 분류하기", type="primary"):
        if text_input.strip():
            with st.spinner("AI 분석 중..."):
                result = classify_urgency(text_input)
                urgency, reason, action = parse_result(result)

            color = {"상": "🔴", "중": "🟡", "하": "🟢"}.get(urgency, "⚪")
            bg = {"상": "#ff4b4b", "중": "#ffa500", "하": "#00cc44"}.get(urgency, "#888")

            st.markdown(f"""
            <div style='background:{bg};padding:20px;border-radius:12px;color:white;margin:10px 0'>
                <h2>{color} 긴급도: {urgency}</h2>
                <p>📋 판단 근거: {reason}</p>
                <p>🚒 권장 출동: {action}</p>
            </div>
            """, unsafe_allow_html=True)

            st.session_state.history.append({
                "신고 내용": text_input[:50] + "...",
                "긴급도": urgency,
                "근거": reason,
                "권장출동": action
            })
        else:
            st.warning("신고 내용을 입력해주세요.")

    if st.session_state.history:
        st.subheader("📋 분류 이력")
        st.dataframe(pd.DataFrame(st.session_state.history), use_container_width=True)

# ── 탭 2: 데이터 분석 ──
with tab2:
    st.subheader("143,046건 실제 119 신고 데이터 분석")

    col1, col2 = st.columns(2)

    with col1:
        urgency_data = pd.DataFrame({
            '긴급도': ['하 (비긴급)', '중 (일반)', '상 (즉시출동)'],
            '건수': [50656, 48069, 44321]
        })
        fig1 = px.pie(urgency_data, values='건수', names='긴급도',
                      title='긴급도별 신고 분포',
                      color_discrete_sequence=['#00cc44', '#ffa500', '#ff4b4b'])
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        category_data = pd.DataFrame({
            '대분류': ['구급', '구조', '화재', '기타'],
            '건수': [108055, 20478, 9705, 4808]
        })
        fig2 = px.bar(category_data, x='대분류', y='건수',
                      title='신고 유형별 분포',
                      color='건수',
                      color_continuous_scale='Reds')
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        sentiment_data = pd.DataFrame({
            '감정 상태': ['불안/걱정', '당황/난처', '중립', '기타부정'],
            '건수': [76695, 53763, 9355, 3233]
        })
        fig3 = px.bar(sentiment_data, x='감정 상태', y='건수',
                      title='신고자 감정 상태 분포',
                      color='건수',
                      color_continuous_scale='Blues')
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        st.metric("총 신고 건수", "143,046건")
        st.metric("구급 신고 비율", "75.5%")
        st.metric("긴급(상) 신고 비율", "31.0%")
        st.metric("불안/당황 신고자 비율", "90.8%")

# ── 탭 3: AI 성능 분석 ──
with tab3:
    st.subheader("300건 테스트 기반 AI 성능 분석")

    col1, col2, col3 = st.columns(3)
    col1.metric("전체 정확도", "43.3%")
    col2.metric("위험 오분류(상→하)", "4건")
    col3.metric("유효 분류", "300/300건")

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**긴급도별 정확도**")
        perf_data = pd.DataFrame({
            '긴급도': ['상', '중', '하'],
            '정확도': [0.45, 0.38, 0.68],
            'F1-score': [0.54, 0.43, 0.22]
        })
        st.dataframe(perf_data, use_container_width=True)

        cat_data = pd.DataFrame({
            '대분류': ['화재', '구급', '구조', '기타'],
            '정확도': [0.381, 0.419, 0.513, 0.545]
        })
        fig4 = px.bar(cat_data, x='정확도', y='대분류', orientation='h',
                      title='대분류별 AI 정확도',
                      color='정확도', color_continuous_scale='RdYlGn')
        fig4.add_vline(x=0.433, line_dash="dash", line_color="red",
                       annotation_text="전체 평균")
        st.plotly_chart(fig4, use_container_width=True)

    with col2:
        sent_data = pd.DataFrame({
            '감정 상태': ['불안/걱정', '당황/난처', '중립', '기타부정'],
            '정확도': [0.402, 0.458, 0.500, 0.667]
        })
        fig5 = px.bar(sent_data, x='정확도', y='감정 상태', orientation='h',
                      title='감정 상태별 AI 정확도',
                      color='정확도', color_continuous_scale='RdYlGn')
        fig5.add_vline(x=0.433, line_dash="dash", line_color="red",
                       annotation_text="전체 평균")
        st.plotly_chart(fig5, use_container_width=True)

    st.divider()
    st.subheader("🚨 위험 오분류 사례 (실제 긴급 → AI가 비긴급 판단)")
    danger_cases = [
        {
            "신고 내용": "수신기가 건물 전체 올리기 시작했는데요. 저 서울 서대문구 남가좌동 세입자예요.",
            "실제 긴급도": "상",
            "AI 판단": "하",
            "AI 근거": "화재 언급 없음"
        },
        {
            "신고 내용": "화재 경보기가 울리는데요. 저희 상가에서 오작동해서 지금 울리는 거 같거든요.",
            "실제 긴급도": "상",
            "AI 판단": "하",
            "AI 근거": "오작동"
        },
        {
            "신고 내용": "소화기 찾았는데 이거 어떻게 할지 몰라가지고. 안전핀 뽑았는데.",
            "실제 긴급도": "상",
            "AI 판단": "하",
            "AI 근거": "소화기 사용 완료"
        }
    ]
    st.dataframe(pd.DataFrame(danger_cases), use_container_width=True)

# ── 탭 4: 결론 ──
with tab4:
    st.subheader("탐구 배경 및 결론")

    st.markdown("""
    ### 📌 탐구 배경
    119 신고 접수 현장에서는 신고자의 말만으로 상황의 긴급도를 빠르게 판단해야 한다.
    그러나 신고 내용이 불명확하거나 신고자가 당황한 상태일 경우 접수자의 경험과 직관에만
    의존하는 방식은 한계가 있다. 골든타임을 놓치면 인명 피해로 이어질 수 있다.

    ### 🔍 핵심 발견 3가지
    """)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**발견 1**\n\nAI 전체 정확도 43.3%로, 단순 무작위(33%)보다 높지만 실용화엔 한계 존재")
    with col2:
        st.warning("**발견 2**\n\n당황한 신고자(40.2%)가 침착한 신고자(50.0%)보다 AI 정확도 낮음 → 위기 상황에서 AI도 어려움")
    with col3:
        st.error("**발견 3**\n\n화재 신고 정확도 38.1%로 가장 낮음 → 오작동·간접 표현이 AI 판단을 혼란시킴")

    st.markdown("""
    ### ⚠️ AI 분류의 한계점
    - 신고자가 상황을 축소 표현할 경우 과소평가 위험
    - 화재경보 오작동처럼 신고자 스스로 비긴급이라 판단한 경우 AI도 동일하게 오분류
    - 텍스트만으로는 신고자의 목소리 톤·호흡 등 비언어적 정보 반영 불가

    ### 🌊 제주 적용 가능성 및 제언
    - 제주 특화 신고 유형(한라산 고립, 해상 사고, 올레길 조난 등) 추가 학습 필요
    - 현장 접수원 보조 도구로 활용 시 골든타임 단축 가능성 있음
    - 향후 음성 감정 분석과 결합하면 정확도 향상 기대
    """)
