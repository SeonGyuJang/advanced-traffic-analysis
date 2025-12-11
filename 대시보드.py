#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
세종시 버스정류장 최적화 대시보드
================================
정수계획법 기반 신규 정류장 추천 및 수요 시각화
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from streamlit_folium import folium_static
from datetime import datetime, timedelta
from math import radians, cos, sin, asin, sqrt
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

# ============================================================================
# 유틸리티 함수
# ============================================================================
def haversine_distance(lat1, lon1, lat2, lon2):
    """Haversine 공식으로 실제 거리 계산 (km)"""
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    return 6371 * 2 * asin(sqrt(a))

@st.cache_data
def load_data():
    """데이터 로드"""
    df_stations = pd.read_csv('data/세종도시교통공사_버스정류장_시설현황_20210924.csv')
    df_passenger = pd.read_csv('data/지역별승하차_통합데이터.csv')
    df_passenger['날짜'] = pd.to_datetime(df_passenger['날짜'])
    df_region_coords = pd.read_csv('data/행정구역_중심좌표.csv')

    return df_stations, df_passenger, df_region_coords

def load_optimization_results():
    """최적화 결과 로드"""
    try:
        df_new_stations = pd.read_csv('최적화_신규정류장.csv')
        df_underserved = pd.read_csv('서비스부족지역.csv')
        with open('최적화_분석_보고서.json', 'r', encoding='utf-8') as f:
            report = json.load(f)
        return df_new_stations, df_underserved, report
    except FileNotFoundError:
        return None, None, None

# ============================================================================
# 메인 대시보드
# ============================================================================
def main():
    # 헤더
    st.title("🚌 세종시 버스정류장 최적화 대시보드")
    st.markdown("### 정수계획법 기반 신규 정류장 최적 위치 추천")
    st.markdown("---")

    # 데이터 로드
    df_stations, df_passenger, df_region_coords = load_data()
    df_new_stations, df_underserved, report = load_optimization_results()

    # ========================================================================
    # 사이드바 - 설정 및 필터
    # ========================================================================
    st.sidebar.header("⚙️ 설정")

    # 기간 선택
    st.sidebar.subheader("📅 분석 기간")
    min_date = df_passenger['날짜'].min().date()
    max_date = df_passenger['날짜'].max().date()

    start_date = st.sidebar.date_input(
        "시작일",
        value=datetime(2024, 1, 1).date(),
        min_value=min_date,
        max_value=max_date
    )

    end_date = st.sidebar.date_input(
        "종료일",
        value=max_date,
        min_value=min_date,
        max_value=max_date
    )

    # 파라미터 설정
    st.sidebar.subheader("🎯 최적화 파라미터")
    coverage_radius = st.sidebar.slider(
        "커버리지 반경 (km)",
        min_value=0.1,
        max_value=2.0,
        value=0.5,
        step=0.1
    )

    max_new_stations = st.sidebar.slider(
        "최대 신규 정류장 개수",
        min_value=1,
        max_value=20,
        value=10,
        step=1
    )

    # 재분석 버튼
    if st.sidebar.button("🔄 재분석 실행", type="primary"):
        st.sidebar.info("분석을 실행하려면 터미널에서 스크립트를 실행하세요.")

    # ========================================================================
    # 메인 콘텐츠
    # ========================================================================

    # 기간 필터링
    df_filtered = df_passenger[
        (df_passenger['날짜'] >= pd.to_datetime(start_date)) &
        (df_passenger['날짜'] <= pd.to_datetime(end_date))
    ]

    # 통계 계산
    total_boarding = df_filtered['승차'].sum()
    total_alighting = df_filtered['하차'].sum()
    total_transfer = df_filtered['환승'].sum()
    total_passengers = total_boarding + total_alighting

    # ========================================================================
    # 1. 주요 지표 (KPI)
    # ========================================================================
    st.header("📊 주요 지표")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "기존 정류장",
            f"{len(df_stations):,}개",
            delta=None
        )

    with col2:
        st.metric(
            "총 이용객 (기간 내)",
            f"{total_passengers:,.0f}명",
            delta=None
        )

    with col3:
        if df_new_stations is not None:
            st.metric(
                "추천 신규 정류장",
                f"{len(df_new_stations)}개",
                delta=f"+{len(df_new_stations)}개"
            )
        else:
            st.metric("추천 신규 정류장", "분석 필요", delta=None)

    with col4:
        if df_underserved is not None:
            st.metric(
                "서비스 부족 지역",
                f"{len(df_underserved)}개",
                delta=None
            )
        else:
            st.metric("서비스 부족 지역", "분석 필요", delta=None)

    st.markdown("---")

    # ========================================================================
    # 2. 지도 시각화
    # ========================================================================
    st.header("🗺️ 정류장 분포 지도")

    col_map1, col_map2 = st.columns([2, 1])

    with col_map1:
        # Folium 지도 생성
        center_lat = df_stations['위도'].mean()
        center_lon = df_stations['경도'].mean()

        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=11,
            tiles='OpenStreetMap'
        )

        # 기존 정류장
        for _, row in df_stations.iterrows():
            folium.CircleMarker(
                location=[row['위도'], row['경도']],
                radius=3,
                color='#9C27B0',
                fill=True,
                fillColor='#9C27B0',
                fillOpacity=0.6,
                popup=f"<b>{row['정류소명']}</b><br>기존 정류장"
            ).add_to(m)

        # 신규 추천 정류장
        if df_new_stations is not None and len(df_new_stations) > 0:
            for _, row in df_new_stations.iterrows():
                folium.Marker(
                    location=[row['위도'], row['경도']],
                    popup=f"""
                    <b>{row['행정구역']}</b><br>
                    <b>신규 추천 정류장</b><br>
                    우선순위: {row['우선순위']}<br>
                    예상 수요: {row['총_이용객']:,.0f}명<br>
                    커버 수요: {row['커버_수요']:,.0f}명
                    """,
                    icon=folium.Icon(color='red', icon='plus', prefix='fa')
                ).add_to(m)

                # 커버리지 원
                folium.Circle(
                    location=[row['위도'], row['경도']],
                    radius=coverage_radius * 1000,
                    color='#FF5722',
                    fill=True,
                    fillColor='#FF5722',
                    fillOpacity=0.1,
                    weight=2,
                    popup=f"커버리지 반경 {coverage_radius}km"
                ).add_to(m)

        # 범례 추가
        legend_html = """
        <div style="position: fixed;
                    bottom: 50px; left: 50px; width: 200px; height: auto;
                    background-color: white; border:2px solid grey; z-index:9999;
                    font-size:14px; padding: 10px">
        <p><b>범례</b></p>
        <p><span style="color:#9C27B0;">●</span> 기존 정류장</p>
        <p><span style="color:#FF5722;">📍</span> 신규 추천 정류장</p>
        </div>
        """
        m.get_root().html.add_child(folium.Element(legend_html))

        folium_static(m, width=800, height=600)

    with col_map2:
        st.subheader("📌 신규 정류장 추천")

        if df_new_stations is not None and len(df_new_stations) > 0:
            st.dataframe(
                df_new_stations[['우선순위', '행정구역', '총_이용객', '커버_수요']].style.format({
                    '총_이용객': '{:,.0f}',
                    '커버_수요': '{:,.0f}'
                }),
                height=400
            )
        else:
            st.info("재분석을 실행하여 신규 정류장을 추천받으세요.")

    st.markdown("---")

    # ========================================================================
    # 3. 수요 시각화
    # ========================================================================
    st.header("📈 수요 분석")

    tab1, tab2, tab3 = st.tabs(["시계열 분석", "지역별 분석", "환승 분석"])

    with tab1:
        st.subheader("일별 이용객 추이")

        # 일별 집계
        daily_stats = df_filtered.groupby('날짜').agg({
            '승차': 'sum',
            '하차': 'sum',
            '환승': 'sum'
        }).reset_index()

        daily_stats['총_이용객'] = daily_stats['승차'] + daily_stats['하차']

        # 시계열 그래프
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=daily_stats['날짜'],
            y=daily_stats['승차'],
            name='승차',
            mode='lines',
            line=dict(color='#2E4057', width=2)
        ))

        fig.add_trace(go.Scatter(
            x=daily_stats['날짜'],
            y=daily_stats['하차'],
            name='하차',
            mode='lines',
            line=dict(color='#048A81', width=2)
        ))

        fig.add_trace(go.Scatter(
            x=daily_stats['날짜'],
            y=daily_stats['환승'],
            name='환승',
            mode='lines',
            line=dict(color='#F26419', width=2)
        ))

        fig.update_layout(
            title="일별 승하차 및 환승 추이",
            xaxis_title="날짜",
            yaxis_title="이용객 수",
            hovermode='x unified',
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("행정구역별 수요")

        # 지역별 집계
        region_stats = df_filtered.groupby('행정구역').agg({
            '승차': 'sum',
            '하차': 'sum',
            '환승': 'sum'
        }).reset_index()

        region_stats['총_이용객'] = region_stats['승차'] + region_stats['하차']
        region_stats = region_stats.sort_values('총_이용객', ascending=False)

        # 바차트
        fig = px.bar(
            region_stats.head(15),
            x='행정구역',
            y='총_이용객',
            title='상위 15개 행정구역별 총 이용객',
            color='총_이용객',
            color_continuous_scale='Reds'
        )

        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

        # 상세 테이블
        st.dataframe(
            region_stats.style.format({
                '승차': '{:,.0f}',
                '하차': '{:,.0f}',
                '환승': '{:,.0f}',
                '총_이용객': '{:,.0f}'
            }),
            height=300
        )

    with tab3:
        st.subheader("환승 패턴 분석")

        # 환승 비율 계산
        region_stats['환승률'] = (region_stats['환승'] / region_stats['총_이용객'] * 100).round(2)
        region_stats = region_stats.sort_values('환승률', ascending=False)

        # 환승률 바차트
        fig = px.bar(
            region_stats.head(15),
            x='행정구역',
            y='환승률',
            title='상위 15개 행정구역별 환승률 (%)',
            color='환승률',
            color_continuous_scale='Blues'
        )

        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

        # 환승 vs 총 이용객 산점도
        fig2 = px.scatter(
            region_stats,
            x='총_이용객',
            y='환승',
            size='환승률',
            color='환승률',
            hover_name='행정구역',
            title='총 이용객 vs 환승',
            color_continuous_scale='Viridis'
        )

        fig2.update_layout(height=400)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")

    # ========================================================================
    # 4. 최적화 결과 상세
    # ========================================================================
    if report is not None:
        st.header("🎯 최적화 결과")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("분석 정보")
            st.json(report['분석기간'])
            st.json(report['기존_정류장'])

        with col2:
            st.subheader("최적화 성과")
            st.json(report['서비스_부족_지역'])
            st.json(report['최적화_결과'])

        # 커버리지 개선 효과
        if df_underserved is not None and len(df_underserved) > 0:
            st.subheader("📊 커버리지 개선 효과")

            before_coverage = (len(df_region_coords) - len(df_underserved)) / len(df_region_coords) * 100
            after_coverage = 100.0

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("최적화 전 커버리지", f"{before_coverage:.1f}%")

            with col2:
                st.metric("최적화 후 커버리지", f"{after_coverage:.1f}%")

            with col3:
                st.metric("개선도", f"+{after_coverage - before_coverage:.1f}%p")

    # ========================================================================
    # 푸터
    # ========================================================================
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p>세종시 버스정류장 최적화 대시보드 | 정수계획법 기반 분석</p>
        <p>© 2025 Advanced Traffic Analysis</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
