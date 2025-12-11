#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
세종시 버스정류장 최적화 전문가 대시보드
======================================
경영진 보고용 고품질 시각화 및 인터랙티브 대시보드
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from streamlit_folium import folium_static
from datetime import datetime
import json

# ============================================================================
# 페이지 설정
# ============================================================================
st.set_page_config(
    page_title="세종시 버스정류장 최적화 대시보드",
    page_icon="🚌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS 스타일
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A8A;
        text-align: center;
        padding: 1rem 0;
        border-bottom: 3px solid #3B82F6;
        margin-bottom: 2rem;
    }
    .kpi-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .kpi-value {
        font-size: 2rem;
        font-weight: 700;
    }
    .kpi-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #1F2937;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #E5E7EB;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border-left: 4px solid #3B82F6;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# 데이터 로드
# ============================================================================
@st.cache_data
def load_data():
    """분석 결과 데이터 로드"""
    try:
        stations = pd.read_csv('분석결과_정류장별수요.csv')
        grid = pd.read_csv('분석결과_수요그리드.csv')
        new_stations = pd.read_csv('분석결과_신규정류장.csv')

        with open('분석결과_보고서.json', 'r', encoding='utf-8') as f:
            report = json.load(f)

        return stations, grid, new_stations, report
    except FileNotFoundError as e:
        st.error(f"데이터 파일을 찾을 수 없습니다: {e}")
        st.info("먼저 '정밀_정류장_분석.py'를 실행하세요.")
        st.stop()

# ============================================================================
# 메인 대시보드
# ============================================================================
def main():
    # 헤더
    st.markdown('<div class="main-header">🚌 세종시 버스정류장 최적화 분석</div>',
                unsafe_allow_html=True)
    st.markdown('<p style="text-align:center; color:#6B7280; margin-bottom:2rem;">정수계획법 기반 신규 정류장 위치 최적화 - 경영진 보고용</p>',
                unsafe_allow_html=True)

    # 데이터 로드
    stations, grid, new_stations, report = load_data()

    # ========================================================================
    # 사이드바
    # ========================================================================
    with st.sidebar:
        st.image("https://via.placeholder.com/300x100/667eea/ffffff?text=Sejong+City", use_container_width=True)

        st.markdown("## 📊 분석 정보")
        st.info(f"""
        **분석 기간**
        {report['분석기간']['시작']} ~ {report['분석기간']['종료']}

        **분석 일시**
        {report['분석일시']}
        """)

        st.markdown("---")

        st.markdown("## ⚙️ 최적화 설정")
        st.metric("커버리지 반경", f"{report['설정']['커버리지반경_km']} km")
        st.metric("최소 정류장 간격", f"{report['설정']['최소정류장간거리_km']} km")
        st.metric("그리드 해상도", f"~{int(report['설정']['그리드크기']*100)} km")

        st.markdown("---")

        st.markdown("## 📄 보고서 다운로드")
        if st.button("📥 CSV 다운로드", use_container_width=True):
            st.download_button(
                label="신규 정류장 목록",
                data=new_stations.to_csv(index=False, encoding='utf-8-sig'),
                file_name="신규정류장_추천목록.csv",
                mime="text/csv"
            )

    # ========================================================================
    # 주요 지표 (KPI)
    # ========================================================================
    st.markdown('<div class="section-header">📈 주요 성과 지표 (KPI)</div>',
                unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="kpi-container">
            <div class="kpi-label">기존 정류장</div>
            <div class="kpi-value">{report['기존정류장']['총개수']:,}</div>
            <div class="kpi-label">개</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        coverage = report['커버리지']['커버율']
        st.markdown(f"""
        <div class="kpi-container" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">
            <div class="kpi-label">커버리지</div>
            <div class="kpi-value">{coverage:.1f}%</div>
            <div class="kpi-label">현재 커버율</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="kpi-container" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);">
            <div class="kpi-label">신규 정류장</div>
            <div class="kpi-value">{report['최적화결과']['신규정류장개수']}</div>
            <div class="kpi-label">개 추천</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        demand = report['최적화결과']['예상커버수요']
        st.markdown(f"""
        <div class="kpi-container" style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);">
            <div class="kpi-label">예상 커버 수요</div>
            <div class="kpi-value">{demand:,.0f}</div>
            <div class="kpi-label">명</div>
        </div>
        """, unsafe_allow_html=True)

    # ========================================================================
    # 지도 시각화
    # ========================================================================
    st.markdown('<div class="section-header">🗺️ 정류장 배치 지도</div>',
                unsafe_allow_html=True)

    col_map1, col_map2 = st.columns([3, 1])

    with col_map1:
        # Folium 지도 생성
        center_lat = stations['위도'].mean()
        center_lon = stations['경도'].mean()

        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=11,
            tiles='CartoDB positron'
        )

        # 기존 정류장 (수요에 따라 크기 조정)
        max_demand = stations['할당_총수요'].max()
        for _, row in stations[stations['할당_총수요'] > 0].iterrows():
            demand_ratio = row['할당_총수요'] / max_demand
            radius = 3 + demand_ratio * 7

            folium.CircleMarker(
                location=[row['위도'], row['경도']],
                radius=radius,
                color='#9C27B0',
                fill=True,
                fillColor='#9C27B0',
                fillOpacity=0.6,
                popup=f"""
                <b>{row['정류소명']}</b><br>
                할당 수요: {row['할당_총수요']:,.0f}명<br>
                승차: {row['할당_승차']:,.0f}명<br>
                하차: {row['할당_하차']:,.0f}명<br>
                환승: {row['할당_환승']:,.0f}명
                """,
                tooltip=f"{row['정류소명']} ({row['할당_총수요']:,.0f}명)"
            ).add_to(m)

        # 신규 추천 정류장
        for idx, row in new_stations.iterrows():
            folium.Marker(
                location=[row['위도'], row['경도']],
                popup=f"""
                <div style="width:200px;">
                <h4 style="margin:0;">신규 정류장 #{row['우선순위']}</h4>
                <hr style="margin:5px 0;">
                <b>예상 수요:</b> {row['수요']:,.0f}명<br>
                <b>환승:</b> {row['환승']:,.0f}명<br>
                <b>커버 수요:</b> {row['커버_수요']:,.0f}명<br>
                <b>평균 거리:</b> {row['평균거리']:.2f} km<br>
                <b>위치:</b> ({row['위도']:.6f}, {row['경도']:.6f})
                </div>
                """,
                icon=folium.Icon(color='red', icon='star', prefix='fa'),
                tooltip=f"신규 #{row['우선순위']} (수요: {row['수요']:,.0f}명)"
            ).add_to(m)

            # 커버리지 원
            folium.Circle(
                location=[row['위도'], row['경도']],
                radius=500,
                color='#FF5722',
                fill=True,
                fillColor='#FF5722',
                fillOpacity=0.1,
                weight=2
            ).add_to(m)

        # 범례
        legend_html = """
        <div style="position: fixed;
                    bottom: 50px; left: 50px; width: 220px;
                    background-color: white; border:2px solid grey; z-index:9999;
                    font-size:14px; padding: 15px; border-radius: 5px;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
        <h4 style="margin-top:0;">범례</h4>
        <p style="margin: 5px 0;"><span style="color:#9C27B0; font-size: 20px;">●</span> 기존 정류장 (크기=수요)</p>
        <p style="margin: 5px 0;"><span style="color:#FF5722; font-size: 20px;">★</span> 신규 추천 정류장</p>
        <p style="margin: 5px 0;"><span style="color:#FF5722;">○</span> 커버리지 (0.5km)</p>
        </div>
        """
        m.get_root().html.add_child(folium.Element(legend_html))

        folium_static(m, width=None, height=600)

    with col_map2:
        st.markdown("### 📍 신규 정류장 목록")

        for idx, row in new_stations.head(10).iterrows():
            st.markdown(f"""
            <div style="background:#F3F4F6; padding:10px; margin:5px 0; border-radius:5px; border-left:4px solid #EF4444;">
                <b style="color:#EF4444;">#{row['우선순위']}</b><br>
                <b>수요:</b> {row['수요']:,.0f}명<br>
                <small>커버: {row['커버_수요']:,.0f}명</small>
            </div>
            """, unsafe_allow_html=True)

    # ========================================================================
    # 수요 분석
    # ========================================================================
    st.markdown('<div class="section-header">📊 수요 밀도 분석</div>',
                unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["수요 히트맵", "정류장 수요 분포", "커버리지 분석"])

    with tab1:
        st.subheader("수요 밀도 히트맵")

        # 그리드 데이터 히트맵
        fig = px.density_mapbox(
            grid[grid['수요'] > 0],
            lat='위도',
            lon='경도',
            z='수요',
            radius=15,
            zoom=10,
            mapbox_style="open-street-map",
            color_continuous_scale="YlOrRd",
            title="세종시 버스 수요 밀도 분포"
        )

        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)

        # 통계
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("총 그리드 셀", f"{len(grid):,}개")
        with col2:
            st.metric("평균 셀 수요", f"{grid['수요'].mean():.1f}명")
        with col3:
            st.metric("최대 셀 수요", f"{grid['수요'].max():.0f}명")

    with tab2:
        st.subheader("정류장별 수요 분포")

        # 상위 20개 정류장
        top_stations = stations.nlargest(20, '할당_총수요')

        fig = go.Figure()

        fig.add_trace(go.Bar(
            y=top_stations['정류소명'],
            x=top_stations['할당_승차'],
            name='승차',
            orientation='h',
            marker=dict(color='#3B82F6')
        ))

        fig.add_trace(go.Bar(
            y=top_stations['정류소명'],
            x=top_stations['할당_하차'],
            name='하차',
            orientation='h',
            marker=dict(color='#10B981')
        ))

        fig.update_layout(
            title="상위 20개 정류장 승하차 수요",
            xaxis_title="이용객 수",
            yaxis_title="정류장",
            barmode='stack',
            height=600,
            hovermode='y unified'
        )

        st.plotly_chart(fig, use_container_width=True)

        # 데이터 테이블
        st.markdown("#### 상세 데이터")
        st.dataframe(
            top_stations[['정류소명', '할당_승차', '할당_하차', '할당_환승', '할당_총수요']].style.format({
                '할당_승차': '{:,.0f}',
                '할당_하차': '{:,.0f}',
                '할당_환승': '{:,.0f}',
                '할당_총수요': '{:,.0f}'
            }),
            height=400,
            use_container_width=True
        )

    with tab3:
        st.subheader("커버리지 분석")

        # 커버리지 통계
        col1, col2 = st.columns(2)

        with col1:
            # 파이 차트
            coverage_data = pd.DataFrame({
                '구분': ['커버됨', '미커버'],
                '셀수': [grid['커버여부'].sum(), (~grid['커버여부']).sum()]
            })

            fig = px.pie(
                coverage_data,
                values='셀수',
                names='구분',
                title='그리드 커버리지 현황',
                color_discrete_sequence=['#10B981', '#EF4444']
            )

            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(height=400)

            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # 거리 분포
            fig = go.Figure()

            fig.add_trace(go.Histogram(
                x=grid['최단정류장거리_km'],
                nbinsx=30,
                marker=dict(color='#3B82F6'),
                name='거리 분포'
            ))

            fig.add_vline(
                x=report['설정']['커버리지반경_km'],
                line_dash="dash",
                line_color="red",
                annotation_text="커버리지 기준"
            )

            fig.update_layout(
                title='정류장까지 최단 거리 분포',
                xaxis_title='거리 (km)',
                yaxis_title='셀 개수',
                height=400
            )

            st.plotly_chart(fig, use_container_width=True)

        # 개선 효과
        st.markdown("#### 최적화 개선 효과")

        improvement_data = {
            '구분': ['최적화 전', '최적화 후 (예상)'],
            '커버율': [
                report['커버리지']['커버율'],
                min(100, report['커버리지']['커버율'] +
                    (report['최적화결과']['예상커버수요'] / report['수요밀도']['총수요'] * 100))
            ]
        }

        fig = go.Figure(data=[
            go.Bar(
                x=improvement_data['구분'],
                y=improvement_data['커버율'],
                marker=dict(
                    color=improvement_data['커버율'],
                    colorscale='RdYlGn',
                    showscale=False
                ),
                text=[f"{v:.1f}%" for v in improvement_data['커버율']],
                textposition='auto'
            )
        ])

        fig.update_layout(
            title='커버리지 개선 효과',
            yaxis_title='커버율 (%)',
            height=300
        )

        st.plotly_chart(fig, use_container_width=True)

    # ========================================================================
    # 최적화 결과 상세
    # ========================================================================
    st.markdown('<div class="section-header">🎯 최적화 결과 상세</div>',
                unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### 신규 정류장 추천 목록")

        # 전체 목록
        display_df = new_stations.copy()
        display_df = display_df[['우선순위', '위도', '경도', '수요', '환승', '커버_수요', '평균거리']]

        st.dataframe(
            display_df.style.format({
                '위도': '{:.6f}',
                '경도': '{:.6f}',
                '수요': '{:,.0f}',
                '환승': '{:,.0f}',
                '커버_수요': '{:,.0f}',
                '평균거리': '{:.2f}'
            }).background_gradient(subset=['수요'], cmap='YlOrRd'),
            height=500,
            use_container_width=True
        )

    with col2:
        st.markdown("### 핵심 인사이트")

        st.info(f"""
        **🎯 분석 요약**

        총 **{report['기존정류장']['총개수']:,}개** 기존 정류장 중
        **{report['기존정류장']['수요있는정류장']}개**만 실제 수요가 있습니다.

        **{len(grid):,}개** 그리드 셀 분석 결과,
        현재 커버리지는 **{report['커버리지']['커버율']:.1f}%**입니다.

        **{report['최적화결과']['신규정류장개수']}개** 신규 정류장으로
        **{report['최적화결과']['예상커버수요']:,.0f}명**의
        추가 수요를 커버할 수 있습니다.
        """)

        st.success(f"""
        **💡 권장사항**

        1. 우선순위 1-5번 정류장을 우선 설치
        2. 지역별 균형을 고려한 단계적 설치
        3. 환승 수요가 높은 지역 우선 검토
        4. 실제 도로 및 토지 여건 추가 검토 필요
        """)

    # ========================================================================
    # 푸터
    # ========================================================================
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #6B7280; padding: 2rem 0;'>
        <p style='font-size: 0.9rem; margin: 0;'>
            <b>세종시 버스정류장 최적화 분석 시스템</b><br>
            정수계획법 기반 데이터 기반 의사결정 지원<br>
            © 2025 Advanced Traffic Analysis
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
