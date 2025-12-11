#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
세종시 버스정류장 최적화 - 프리미엄 대시보드
경영진 발표용 고품질 인터랙티브 대시보드
"""

import streamlit as st
import pandas as pd
import folium
from folium import plugins
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# 페이지 설정
st.set_page_config(
    page_title="세종시 버스정류장 최적화 분석",
    page_icon="🚌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 커스텀 CSS
st.markdown("""
<style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    .stMetric label {
        font-size: 16px !important;
        font-weight: 600 !important;
    }
    .stMetric [data-testid="stMetricValue"] {
        font-size: 32px !important;
    }
    h1 {
        color: #1f77b4;
        padding-bottom: 10px;
        border-bottom: 3px solid #1f77b4;
    }
    h2 {
        color: #2c3e50;
        margin-top: 20px;
    }
    .highlight {
        background-color: #fff3cd;
        padding: 10px;
        border-left: 4px solid #ffc107;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """데이터 로드"""
    stations = pd.read_csv('분석결과_정류장별수요.csv')
    new_stations = pd.read_csv('분석결과_신규정류장.csv')
    return stations, new_stations

def create_premium_map(stations_df, new_stations_df, coverage_radius, show_existing, show_new, show_coverage, show_heatmap):
    """프리미엄 인터랙티브 지도 생성"""

    # 세종시 중심
    center_lat = 36.48
    center_lon = 127.26

    # 지도 생성
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=12,
        tiles='OpenStreetMap',
        prefer_canvas=True
    )

    # 기존 정류장 표시
    if show_existing:
        existing_group = folium.FeatureGroup(name='기존 정류장')

        for _, row in stations_df.iterrows():
            if row['할당_총수요'] > 0:
                # 수요에 따른 마커 크기
                radius = min(3 + row['할당_총수요'] / 100000, 15)

                # 커버리지 원 (옵션)
                if show_coverage:
                    folium.Circle(
                        location=[row['위도'], row['경도']],
                        radius=coverage_radius * 1000,  # km to m
                        color='#3186cc',
                        fill=True,
                        fillColor='#3186cc',
                        fillOpacity=0.1,
                        weight=1,
                        opacity=0.3
                    ).add_to(existing_group)

                # 정류장 마커
                folium.CircleMarker(
                    location=[row['위도'], row['경도']],
                    radius=radius,
                    popup=folium.Popup(
                        f"""<div style='width: 200px'>
                        <h4>📍 {row['정류소명']}</h4>
                        <hr>
                        <b>총 수요:</b> {row['할당_총수요']:,}명<br>
                        <b>승차:</b> {row['할당_승차']:,}명<br>
                        <b>하차:</b> {row['할당_하차']:,}명<br>
                        <b>환승:</b> {row['할당_환승']:,}명
                        </div>""",
                        max_width=250
                    ),
                    color='#1f77b4',
                    fillColor='#1f77b4',
                    fillOpacity=0.7,
                    weight=2
                ).add_to(existing_group)

        existing_group.add_to(m)

    # 신규 정류장 표시
    if show_new:
        new_group = folium.FeatureGroup(name='신규 추천 정류장 ⭐')

        for _, row in new_stations_df.iterrows():
            # 우선순위에 따른 색상
            if row['우선순위'] <= 5:
                color = '#ff0000'  # 빨강 (최우선)
                icon_color = 'red'
            elif row['우선순위'] <= 10:
                color = '#ff7f0e'  # 주황
                icon_color = 'orange'
            else:
                color = '#ffd700'  # 노랑
                icon_color = 'yellow'

            # 커버리지 원 (옵션)
            if show_coverage:
                folium.Circle(
                    location=[row['위도'], row['경도']],
                    radius=coverage_radius * 1000,
                    color=color,
                    fill=True,
                    fillColor=color,
                    fillOpacity=0.2,
                    weight=2,
                    opacity=0.6
                ).add_to(new_group)

            # 신규 정류장 마커 (별 아이콘)
            folium.Marker(
                location=[row['위도'], row['경도']],
                popup=folium.Popup(
                    f"""<div style='width: 220px'>
                    <h4>⭐ 신규 정류장 추천</h4>
                    <hr>
                    <b>우선순위:</b> {row['우선순위']}위<br>
                    <b>예상 수요:</b> {row['수요']:,}명<br>
                    <b>환승:</b> {row['환승']:,}명<br>
                    <b>커버 수요:</b> {row['커버_수요']:,}명<br>
                    <b>위치:</b> {row['위도']:.4f}, {row['경도']:.4f}
                    </div>""",
                    max_width=250
                ),
                icon=folium.Icon(color=icon_color, icon='star', prefix='fa'),
                tooltip=f"우선순위 {row['우선순위']}위"
            ).add_to(new_group)

        new_group.add_to(m)

    # 수요 히트맵 (옵션)
    if show_heatmap and show_existing:
        heat_data = []
        for _, row in stations_df.iterrows():
            if row['할당_총수요'] > 0:
                heat_data.append([
                    row['위도'],
                    row['경도'],
                    row['할당_총수요'] / 100000  # 정규화
                ])

        if heat_data:
            plugins.HeatMap(
                heat_data,
                name='수요 히트맵',
                min_opacity=0.3,
                max_zoom=13,
                radius=25,
                blur=35,
                gradient={
                    0.0: 'blue',
                    0.5: 'lime',
                    0.7: 'yellow',
                    1.0: 'red'
                }
            ).add_to(m)

    # 레이어 컨트롤
    folium.LayerControl(collapsed=False).add_to(m)

    # 범례 추가
    legend_html = '''
    <div style="position: fixed;
                bottom: 50px; right: 50px; width: 220px;
                background-color: white; z-index:9999;
                border:2px solid grey; border-radius: 5px;
                padding: 10px; font-size: 14px;">
    <h4 style="margin-top: 0;">범례</h4>
    <p><i class="fa fa-circle" style="color:#1f77b4"></i> 기존 정류장</p>
    <p><i class="fa fa-star" style="color:red"></i> 신규 정류장 (우선순위 1-5)</p>
    <p><i class="fa fa-star" style="color:orange"></i> 신규 정류장 (우선순위 6-10)</p>
    <p><i class="fa fa-star" style="color:gold"></i> 신규 정류장 (우선순위 11-15)</p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))

    return m

def create_summary_charts(stations_df, new_stations_df):
    """요약 차트 생성"""

    # 2개 컬럼 차트
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('상위 10개 기존 정류장 수요', '신규 정류장 우선순위별 수요'),
        specs=[[{"type": "bar"}, {"type": "bar"}]]
    )

    # 기존 정류장 상위 10개
    top_existing = stations_df.nlargest(10, '할당_총수요').sort_values('할당_총수요', ascending=True)
    fig.add_trace(
        go.Bar(
            y=top_existing['정류소명'],
            x=top_existing['할당_총수요'],
            orientation='h',
            marker=dict(color='#1f77b4'),
            text=top_existing['할당_총수요'],
            texttemplate='%{text:,.0f}명',
            textposition='outside',
            name='기존 정류장'
        ),
        row=1, col=1
    )

    # 신규 정류장 상위 10개
    top_new = new_stations_df.nsmallest(10, '우선순위').sort_values('우선순위', ascending=False)
    colors = ['#ff0000' if p <= 5 else '#ff7f0e' if p <= 10 else '#ffd700'
              for p in top_new['우선순위']]

    fig.add_trace(
        go.Bar(
            y=[f"우선순위 {p}" for p in top_new['우선순위']],
            x=top_new['수요'],
            orientation='h',
            marker=dict(color=colors),
            text=top_new['수요'],
            texttemplate='%{text:,.0f}명',
            textposition='outside',
            name='신규 정류장'
        ),
        row=1, col=2
    )

    fig.update_layout(
        height=400,
        showlegend=False,
        font=dict(family="NanumGothic, sans-serif", size=12),
        margin=dict(l=20, r=20, t=40, b=20)
    )

    fig.update_xaxes(title_text="수요 (명)", row=1, col=1)
    fig.update_xaxes(title_text="예상 수요 (명)", row=1, col=2)

    return fig

# 메인 앱
def main():
    # 헤더
    st.title("🚌 세종시 버스정류장 최적화 분석")
    st.markdown("### 정수계획법 기반 신규 정류장 위치 선정 시스템")

    # 데이터 로드
    try:
        stations, new_stations = load_data()
    except Exception as e:
        st.error(f"❌ 데이터 로드 실패: {e}")
        st.info("💡 먼저 '정밀_정류장_분석.py'를 실행하여 분석 결과를 생성하세요.")
        return

    # 사이드바 설정
    st.sidebar.header("⚙️ 대시보드 설정")

    st.sidebar.markdown("---")
    st.sidebar.subheader("🎨 표시 옵션")
    show_existing = st.sidebar.checkbox("기존 정류장 표시", value=True)
    show_new = st.sidebar.checkbox("신규 정류장 표시", value=True)
    show_coverage = st.sidebar.checkbox("커버리지 영역 표시", value=True)
    show_heatmap = st.sidebar.checkbox("수요 히트맵 표시", value=True)

    st.sidebar.markdown("---")
    st.sidebar.subheader("📏 분석 파라미터")
    coverage_radius = st.sidebar.slider(
        "커버리지 반경 (km)",
        min_value=0.3,
        max_value=1.0,
        value=0.5,
        step=0.1,
        help="정류장이 커버하는 범위"
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("🔍 필터")
    min_demand = st.sidebar.slider(
        "최소 수요 (명)",
        min_value=0,
        max_value=int(stations['할당_총수요'].max()),
        value=0,
        step=1000,
        help="표시할 기존 정류장의 최소 수요"
    )

    # 필터 적용
    filtered_stations = stations[stations['할당_총수요'] >= min_demand]

    # KPI 섹션
    st.markdown("## 📊 핵심 성과 지표 (KPI)")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric(
            label="기존 정류장",
            value=f"{len(stations):,}개",
            delta=f"{len(filtered_stations[filtered_stations['할당_총수요'] > 0])}개 활성"
        )

    with col2:
        total_demand = stations['할당_총수요'].sum()
        st.metric(
            label="총 수요",
            value=f"{total_demand:,.0f}명"
        )

    with col3:
        st.metric(
            label="신규 정류장",
            value=f"{len(new_stations)}개",
            delta="추천"
        )

    with col4:
        new_demand = new_stations['수요'].sum()
        st.metric(
            label="신규 예상 수요",
            value=f"{new_demand:,.0f}명"
        )

    with col5:
        coverage = (new_demand / total_demand * 100) if total_demand > 0 else 0
        st.metric(
            label="수요 커버율",
            value=f"{coverage:.1f}%",
            delta="증가 예상"
        )

    # 인사이트
    st.markdown("""
    <div class="highlight">
    <b>💡 핵심 인사이트:</b> 신규 정류장 {count}개를 설치하면 약 <b>{demand:,}명</b>의 추가 수요를 커버할 수 있으며,
    우선순위 1-5위 정류장이 전체 신규 수요의 약 <b>{pct:.1f}%</b>를 차지합니다.
    </div>
    """.format(
        count=len(new_stations),
        demand=int(new_demand),
        pct=(new_stations.nsmallest(5, '우선순위')['수요'].sum() / new_demand * 100) if new_demand > 0 else 0
    ), unsafe_allow_html=True)

    st.markdown("---")

    # 지도 섹션
    st.markdown("## 🗺️ 인터랙티브 지도 분석")
    st.markdown("기존 정류장(파란색)과 신규 추천 정류장(별표)의 위치 및 커버리지 영역을 확인하세요.")

    # 지도 생성 및 표시
    premium_map = create_premium_map(
        filtered_stations,
        new_stations,
        coverage_radius,
        show_existing,
        show_new,
        show_coverage,
        show_heatmap
    )

    st_folium(premium_map, width=None, height=600)

    st.markdown("---")

    # 차트 섹션
    st.markdown("## 📈 수요 분석 차트")

    summary_fig = create_summary_charts(stations, new_stations)
    st.plotly_chart(summary_fig, use_container_width=True)

    st.markdown("---")

    # 상세 데이터 테이블
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("### 📋 기존 정류장 상위 10개")
        top_existing = stations.nlargest(10, '할당_총수요')[
            ['정류소명', '할당_총수요', '할당_승차', '할당_하차', '할당_환승']
        ].reset_index(drop=True)
        top_existing.columns = ['정류소명', '총수요', '승차', '하차', '환승']
        top_existing.index = top_existing.index + 1
        st.dataframe(top_existing, use_container_width=True)

    with col_right:
        st.markdown("### ⭐ 신규 정류장 추천 목록")
        new_display = new_stations.nsmallest(10, '우선순위')[
            ['우선순위', '수요', '환승', '커버_수요', '위도', '경도']
        ].reset_index(drop=True)
        new_display.index = new_display.index + 1
        st.dataframe(new_display, use_container_width=True)

    # 푸터
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 12px;'>
    <p>세종시 버스정류장 최적화 분석 시스템 | 정수계획법 기반 | 버전 3.0.0</p>
    <p>© 2025 Advanced Traffic Analysis Team</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
