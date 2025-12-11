#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
세종시 버스정류장 최적화 고도화 대시보드
=======================================
정류장 하나하나를 상세히 분석할 수 있는 전문가급 인터랙티브 대시보드
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from folium import plugins
from streamlit_folium import st_folium
from datetime import datetime
import json
from math import radians, cos, sin, asin, sqrt

# ============================================================================
# 페이지 설정
# ============================================================================
st.set_page_config(
    page_title="세종시 버스정류장 최적화 고도화 대시보드",
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
    .station-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 0.5rem;
        border-left: 4px solid #3B82F6;
    }
    .station-card:hover {
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        transform: translateY(-2px);
        transition: all 0.3s;
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #1E3A8A;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #6B7280;
    }
    .highlight-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# 유틸리티 함수
# ============================================================================
def haversine_distance(lat1, lon1, lat2, lon2):
    """Haversine 거리 계산 (km)"""
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    return 6371 * 2 * asin(sqrt(a))

def find_nearest_stations(target_lat, target_lon, stations_df, n=5):
    """가장 가까운 n개 정류장 찾기"""
    distances = []
    for _, station in stations_df.iterrows():
        dist = haversine_distance(target_lat, target_lon, station['위도'], station['경도'])
        distances.append({
            '정류소명': station['정류소명'],
            '정류소ID': station['정류소ID'],
            '거리_km': dist,
            '수요': station.get('할당_총수요', 0)
        })

    return pd.DataFrame(distances).nsmallest(n, '거리_km')

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
    st.markdown('<div class="main-header">🚌 세종시 버스정류장 최적화 고도화 대시보드</div>',
                unsafe_allow_html=True)

    # 데이터 로드
    stations, grid, new_stations, report = load_data()

    # 세션 상태 초기화
    if 'selected_station' not in st.session_state:
        st.session_state.selected_station = None
    if 'map_center' not in st.session_state:
        st.session_state.map_center = [stations['위도'].mean(), stations['경도'].mean()]

    # ========================================================================
    # 사이드바 - 필터 및 검색
    # ========================================================================
    with st.sidebar:
        st.markdown("## 🔍 검색 및 필터")

        # 정류장 검색
        st.markdown("### 정류장 검색")
        search_query = st.text_input("정류장명으로 검색", placeholder="예: 세종시청")

        if search_query:
            filtered_stations = stations[
                stations['정류소명'].str.contains(search_query, case=False, na=False)
            ]

            if len(filtered_stations) > 0:
                st.success(f"✓ {len(filtered_stations)}개 정류장 발견")

                for idx, row in filtered_stations.head(10).iterrows():
                    if st.button(f"📍 {row['정류소명']}", key=f"search_{idx}"):
                        st.session_state.selected_station = row
                        st.session_state.map_center = [row['위도'], row['경도']]
                        st.rerun()
            else:
                st.warning("검색 결과가 없습니다.")

        st.markdown("---")

        # 수요 필터
        st.markdown("### 수요 필터")
        demand_range = st.slider(
            "수요 범위 (명)",
            min_value=0,
            max_value=int(stations['할당_총수요'].max()),
            value=(0, int(stations['할당_총수요'].max())),
            step=1000
        )

        # 필터 적용
        filtered_by_demand = stations[
            (stations['할당_총수요'] >= demand_range[0]) &
            (stations['할당_총수요'] <= demand_range[1])
        ]

        st.info(f"필터 결과: {len(filtered_by_demand):,}개 정류장")

        st.markdown("---")

        # 정류장 유형 필터
        st.markdown("### 정류장 유형")
        show_existing = st.checkbox("기존 정류장", value=True)
        show_new = st.checkbox("신규 추천", value=True)
        show_grid = st.checkbox("수요 그리드", value=False)

        st.markdown("---")

        # 분석 정보
        st.markdown("### 📊 분석 정보")
        st.json({
            "분석 기간": f"{report['분석기간']['시작']} ~ {report['분석기간']['종료']}",
            "커버리지 반경": f"{report['설정']['커버리지반경_km']} km",
            "그리드 크기": f"~{int(report['설정']['그리드크기']*100)} km"
        })

    # ========================================================================
    # 메인 콘텐츠
    # ========================================================================

    # 탭 구성
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🗺️ 정밀 지도",
        "📍 정류장 상세",
        "📊 통계 분석",
        "🔗 관계 분석",
        "📈 비교 분석"
    ])

    # ========================================================================
    # 탭 1: 정밀 지도
    # ========================================================================
    with tab1:
        st.markdown("## 🗺️ 정밀 인터랙티브 지도")

        col_map, col_info = st.columns([3, 1])

        with col_map:
            # 지도 생성
            center_lat, center_lon = st.session_state.map_center

            m = folium.Map(
                location=[center_lat, center_lon],
                zoom_start=12 if st.session_state.selected_station is None else 15,
                tiles='CartoDB positron'
            )

            # 레이어 그룹
            existing_group = folium.FeatureGroup(name='기존 정류장', show=show_existing)
            new_group = folium.FeatureGroup(name='신규 추천', show=show_new)
            grid_group = folium.FeatureGroup(name='수요 그리드', show=show_grid)

            # 기존 정류장
            if show_existing:
                max_demand = stations['할당_총수요'].max()

                for _, row in filtered_by_demand.iterrows():
                    demand_ratio = row['할당_총수요'] / max_demand if max_demand > 0 else 0
                    radius = 4 + demand_ratio * 8

                    # 수요에 따른 색상
                    if row['할당_총수요'] > 1000000:
                        color = '#D32F2F'
                    elif row['할당_총수요'] > 100000:
                        color = '#F57C00'
                    elif row['할당_총수요'] > 10000:
                        color = '#FBC02D'
                    else:
                        color = '#9C27B0'

                    # 마커
                    folium.CircleMarker(
                        location=[row['위도'], row['경도']],
                        radius=radius,
                        color=color,
                        fill=True,
                        fillColor=color,
                        fillOpacity=0.7,
                        popup=folium.Popup(f"""
                        <div style="width:250px;">
                        <h4 style="margin:0; color:#1E3A8A;">{row['정류소명']}</h4>
                        <hr style="margin:5px 0;">
                        <table style="width:100%; font-size:12px;">
                        <tr><td><b>정류소ID</b></td><td>{row['정류소ID']}</td></tr>
                        <tr><td><b>위치</b></td><td>({row['위도']:.6f}, {row['경도']:.6f})</td></tr>
                        <tr><td colspan="2" style="padding-top:10px;"><b>수요 정보</b></td></tr>
                        <tr><td>총 수요</td><td style="text-align:right; color:#D32F2F; font-weight:bold;">{row['할당_총수요']:,.0f}명</td></tr>
                        <tr><td>승차</td><td style="text-align:right;">{row['할당_승차']:,.0f}명</td></tr>
                        <tr><td>하차</td><td style="text-align:right;">{row['할당_하차']:,.0f}명</td></tr>
                        <tr><td>환승</td><td style="text-align:right;">{row['할당_환승']:,.0f}명</td></tr>
                        </table>
                        </div>
                        """, max_width=300),
                        tooltip=f"{row['정류소명']} ({row['할당_총수요']:,.0f}명)"
                    ).add_to(existing_group)

            # 신규 추천 정류장
            if show_new:
                for _, row in new_stations.iterrows():
                    folium.Marker(
                        location=[row['위도'], row['경도']],
                        popup=folium.Popup(f"""
                        <div style="width:280px;">
                        <h3 style="margin:0; color:#EF4444;">신규 정류장 #{row['우선순위']}</h3>
                        <hr style="margin:10px 0;">
                        <table style="width:100%; font-size:13px;">
                        <tr><td><b>위치</b></td><td>({row['위도']:.6f}, {row['경도']:.6f})</td></tr>
                        <tr><td colspan="2" style="padding-top:10px;"><b>수요 예측</b></td></tr>
                        <tr><td>예상 수요</td><td style="text-align:right; color:#EF4444; font-weight:bold;">{row['수요']:,.0f}명</td></tr>
                        <tr><td>환승</td><td style="text-align:right;">{row['환승']:,.0f}명</td></tr>
                        <tr><td>커버 수요</td><td style="text-align:right; color:#10B981; font-weight:bold;">{row['커버_수요']:,.0f}명</td></tr>
                        <tr><td colspan="2" style="padding-top:10px;"><b>기타 정보</b></td></tr>
                        <tr><td>평균 거리</td><td style="text-align:right;">{row['평균거리']:.2f} km</td></tr>
                        <tr><td>셀 개수</td><td style="text-align:right;">{row['셀개수']}개</td></tr>
                        </table>
                        </div>
                        """, max_width=320),
                        icon=folium.Icon(color='red', icon='star', prefix='fa'),
                        tooltip=f"🌟 우선순위 #{row['우선순위']}"
                    ).add_to(new_group)

                    # 커버리지 원
                    folium.Circle(
                        location=[row['위도'], row['경도']],
                        radius=500,
                        color='#EF4444',
                        fill=True,
                        fillColor='#EF4444',
                        fillOpacity=0.1,
                        weight=2
                    ).add_to(new_group)

            # 수요 그리드
            if show_grid and len(grid) > 0:
                # 상위 수요 그리드만 표시 (성능 고려)
                top_grid = grid.nlargest(100, '수요')

                for _, row in top_grid.iterrows():
                    folium.CircleMarker(
                        location=[row['위도'], row['경도']],
                        radius=3,
                        color='#06A77D',
                        fill=True,
                        fillColor='#06A77D',
                        fillOpacity=0.4,
                        popup=f"수요: {row['수요']:.0f}명<br>거리: {row['최단정류장거리_km']:.2f}km",
                        tooltip=f"{row['수요']:.0f}명"
                    ).add_to(grid_group)

            # 레이어 추가
            existing_group.add_to(m)
            new_group.add_to(m)
            grid_group.add_to(m)

            # 레이어 컨트롤
            folium.LayerControl(collapsed=False).add_to(m)

            # 범례
            legend_html = """
            <div style="position: fixed;
                        bottom: 50px; left: 50px; width: 260px;
                        background-color: white; border:2px solid #1E3A8A; z-index:9999;
                        font-size:13px; padding: 15px; border-radius: 8px;
                        box-shadow: 0 4px 6px rgba(0,0,0,0.2);">
            <h4 style="margin-top:0; color:#1E3A8A; border-bottom: 2px solid #3B82F6; padding-bottom: 8px;">범례</h4>
            <p style="margin: 6px 0;"><span style="color:#D32F2F; font-size: 18px;">●</span> 초고수요 (100만+)</p>
            <p style="margin: 6px 0;"><span style="color:#F57C00; font-size: 18px;">●</span> 고수요 (10만+)</p>
            <p style="margin: 6px 0;"><span style="color:#FBC02D; font-size: 18px;">●</span> 중수요 (1만+)</p>
            <p style="margin: 6px 0;"><span style="color:#9C27B0; font-size: 18px;">●</span> 저수요</p>
            <p style="margin: 6px 0;"><span style="color:#EF4444; font-size: 20px;">★</span> 신규 추천</p>
            <p style="margin: 6px 0;"><span style="color:#06A77D; font-size: 18px;">●</span> 수요 그리드</p>
            </div>
            """
            m.get_root().html.add_child(folium.Element(legend_html))

            # 지도 표시
            map_data = st_folium(m, width=None, height=600, returned_objects=["last_clicked"])

            # 클릭 이벤트 처리
            if map_data and map_data.get("last_clicked"):
                clicked_lat = map_data["last_clicked"]["lat"]
                clicked_lon = map_data["last_clicked"]["lng"]

                # 가장 가까운 정류장 찾기
                nearest = find_nearest_stations(clicked_lat, clicked_lon, stations, n=1)
                if len(nearest) > 0:
                    nearest_id = nearest.iloc[0]['정류소ID']
                    st.session_state.selected_station = stations[stations['정류소ID'] == nearest_id].iloc[0]

        with col_info:
            st.markdown("### 📍 선택된 정류장")

            if st.session_state.selected_station is not None:
                selected = st.session_state.selected_station

                st.markdown(f"""
                <div class="highlight-box">
                    <h3 style="margin:0;">{selected['정류소명']}</h3>
                    <small>ID: {selected['정류소ID']}</small>
                </div>
                """, unsafe_allow_html=True)

                st.markdown(f"""
                **📍 위치**
                - 위도: `{selected['위도']:.6f}`
                - 경도: `{selected['경도']:.6f}`

                **📊 수요 정보**
                - 총 수요: **{selected['할당_총수요']:,.0f}명**
                - 승차: {selected['할당_승차']:,.0f}명
                - 하차: {selected['할당_하차']:,.0f}명
                - 환승: {selected['할당_환승']:,.0f}명
                """)

                # 주변 정류장
                st.markdown("**🔍 가까운 정류장 (5개)**")
                nearest = find_nearest_stations(
                    selected['위도'],
                    selected['경도'],
                    stations[stations['정류소ID'] != selected['정류소ID']],
                    n=5
                )

                for idx, row in nearest.iterrows():
                    st.markdown(f"""
                    <div class="station-card">
                        <b>{row['정류소명']}</b><br>
                        <small>거리: {row['거리_km']:.2f}km | 수요: {row['수요']:,.0f}명</small>
                    </div>
                    """, unsafe_allow_html=True)

                if st.button("🗑️ 선택 해제"):
                    st.session_state.selected_station = None
                    st.rerun()
            else:
                st.info("지도에서 정류장을 클릭하거나 검색하세요.")

    # ========================================================================
    # 탭 2: 정류장 상세
    # ========================================================================
    with tab2:
        st.markdown("## 📍 정류장 상세 정보")

        # 정류장 선택
        station_names = ['선택하세요'] + sorted(stations['정류소명'].unique().tolist())
        selected_name = st.selectbox(
            "정류장 선택",
            station_names,
            index=0
        )

        if selected_name != '선택하세요':
            station_data = stations[stations['정류소명'] == selected_name].iloc[0]

            col1, col2 = st.columns([2, 1])

            with col1:
                st.markdown(f"### {station_data['정류소명']}")

                # 기본 정보
                st.markdown("#### 📋 기본 정보")
                info_df = pd.DataFrame({
                    '항목': ['정류소ID', '위도', '경도'],
                    '값': [
                        station_data['정류소ID'],
                        f"{station_data['위도']:.6f}",
                        f"{station_data['경도']:.6f}"
                    ]
                })
                st.dataframe(info_df, hide_index=True, use_container_width=True)

                # 수요 정보
                st.markdown("#### 📊 수요 정보")
                demand_df = pd.DataFrame({
                    '구분': ['총 수요', '승차', '하차', '환승'],
                    '인원 (명)': [
                        f"{station_data['할당_총수요']:,.0f}",
                        f"{station_data['할당_승차']:,.0f}",
                        f"{station_data['할당_하차']:,.0f}",
                        f"{station_data['할당_환승']:,.0f}"
                    ]
                })
                st.dataframe(demand_df, hide_index=True, use_container_width=True)

                # 수요 차트
                fig = go.Figure(data=[
                    go.Bar(
                        x=['승차', '하차', '환승'],
                        y=[
                            station_data['할당_승차'],
                            station_data['할당_하차'],
                            station_data['할당_환승']
                        ],
                        marker=dict(
                            color=['#3B82F6', '#10B981', '#F59E0B']
                        )
                    )
                ])

                fig.update_layout(
                    title="승하차 및 환승 비교",
                    yaxis_title="인원 (명)",
                    height=300
                )

                st.plotly_chart(fig, use_container_width=True)

            with col2:
                # 주변 정류장
                st.markdown("#### 🔍 주변 정류장")

                nearest = find_nearest_stations(
                    station_data['위도'],
                    station_data['경도'],
                    stations[stations['정류소ID'] != station_data['정류소ID']],
                    n=10
                )

                st.dataframe(
                    nearest[['정류소명', '거리_km', '수요']].style.format({
                        '거리_km': '{:.2f}',
                        '수요': '{:,.0f}'
                    }),
                    hide_index=True,
                    use_container_width=True,
                    height=400
                )

                # 거리 분포 차트
                fig2 = go.Figure(data=[
                    go.Bar(
                        x=nearest['정류소명'],
                        y=nearest['거리_km'],
                        marker=dict(color='#6366F1')
                    )
                ])

                fig2.update_layout(
                    title="주변 정류장 거리",
                    yaxis_title="거리 (km)",
                    height=300,
                    xaxis_tickangle=-45
                )

                st.plotly_chart(fig2, use_container_width=True)

    # ========================================================================
    # 탭 3: 통계 분석
    # ========================================================================
    with tab3:
        st.markdown("## 📊 통계 분석")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "전체 정류장",
                f"{len(stations):,}개",
                f"+{len(new_stations)}개 추천"
            )

        with col2:
            avg_demand = stations['할당_총수요'].mean()
            st.metric(
                "평균 수요",
                f"{avg_demand:,.0f}명",
                f"최대: {stations['할당_총수요'].max():,.0f}명"
            )

        with col3:
            active_stations = (stations['할당_총수요'] > 0).sum()
            st.metric(
                "활성 정류장",
                f"{active_stations}개",
                f"{active_stations/len(stations)*100:.1f}%"
            )

        # 수요 분포
        st.markdown("### 수요 분포 분석")

        col_chart1, col_chart2 = st.columns(2)

        with col_chart1:
            # 히스토그램
            fig = px.histogram(
                stations[stations['할당_총수요'] > 0],
                x='할당_총수요',
                nbins=30,
                title='정류장 수요 분포',
                labels={'할당_총수요': '수요 (명)'}
            )

            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

        with col_chart2:
            # 박스플롯
            fig2 = go.Figure()

            fig2.add_trace(go.Box(
                y=stations[stations['할당_총수요'] > 0]['할당_총수요'],
                name='총 수요',
                marker_color='#3B82F6'
            ))

            fig2.update_layout(
                title='수요 분포 박스플롯',
                yaxis_title='수요 (명)',
                height=400
            )

            st.plotly_chart(fig2, use_container_width=True)

        # 상위/하위 정류장
        st.markdown("### 수요 상위/하위 정류장")

        col_top, col_bottom = st.columns(2)

        with col_top:
            st.markdown("#### 🔝 상위 10개")
            top10 = stations.nlargest(10, '할당_총수요')

            fig3 = go.Figure(data=[
                go.Bar(
                    y=top10['정류소명'],
                    x=top10['할당_총수요'],
                    orientation='h',
                    marker=dict(
                        color=top10['할당_총수요'],
                        colorscale='Reds',
                        showscale=False
                    )
                )
            ])

            fig3.update_layout(height=400, yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig3, use_container_width=True)

        with col_bottom:
            st.markdown("#### 🔻 활성 정류장 중 하위 10개")
            bottom10 = stations[stations['할당_총수요'] > 0].nsmallest(10, '할당_총수요')

            fig4 = go.Figure(data=[
                go.Bar(
                    y=bottom10['정류소명'],
                    x=bottom10['할당_총수요'],
                    orientation='h',
                    marker=dict(
                        color=bottom10['할당_총수요'],
                        colorscale='Blues',
                        showscale=False
                    )
                )
            ])

            fig4.update_layout(height=400, yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig4, use_container_width=True)

    # ========================================================================
    # 탭 4: 관계 분석
    # ========================================================================
    with tab4:
        st.markdown("## 🔗 정류장 간 관계 분석")

        # 정류장 선택
        col_sel1, col_sel2 = st.columns(2)

        with col_sel1:
            station1 = st.selectbox(
                "정류장 1 선택",
                ['선택하세요'] + sorted(stations[stations['할당_총수요'] > 0]['정류소명'].unique().tolist()),
                key='station1'
            )

        with col_sel2:
            station2 = st.selectbox(
                "정류장 2 선택",
                ['선택하세요'] + sorted(stations[stations['할당_총수요'] > 0]['정류소명'].unique().tolist()),
                key='station2'
            )

        if station1 != '선택하세요' and station2 != '선택하세요' and station1 != station2:
            data1 = stations[stations['정류소명'] == station1].iloc[0]
            data2 = stations[stations['정류소명'] == station2].iloc[0]

            # 거리 계산
            distance = haversine_distance(
                data1['위도'], data1['경도'],
                data2['위도'], data2['경도']
            )

            # 비교 표시
            col_comp1, col_comp2, col_comp3 = st.columns(3)

            with col_comp1:
                st.markdown(f"""
                <div class="highlight-box">
                    <h4 style="margin:0;">{station1}</h4>
                    <hr style="margin:10px 0; border-color:rgba(255,255,255,0.3);">
                    <p>총 수요: {data1['할당_총수요']:,.0f}명</p>
                    <p>승차: {data1['할당_승차']:,.0f}명</p>
                    <p>하차: {data1['할당_하차']:,.0f}명</p>
                    <p>환승: {data1['할당_환승']:,.0f}명</p>
                </div>
                """, unsafe_allow_html=True)

            with col_comp2:
                st.metric(
                    "거리",
                    f"{distance:.2f} km",
                    f"{distance*1000:.0f}m"
                )

                # 도보 시간 (시속 5km)
                walk_time = distance / 5 * 60
                st.info(f"도보 시간: 약 {walk_time:.0f}분")

            with col_comp3:
                st.markdown(f"""
                <div class="highlight-box">
                    <h4 style="margin:0;">{station2}</h4>
                    <hr style="margin:10px 0; border-color:rgba(255,255,255,0.3);">
                    <p>총 수요: {data2['할당_총수요']:,.0f}명</p>
                    <p>승차: {data2['할당_승차']:,.0f}명</p>
                    <p>하차: {data2['할당_하차']:,.0f}명</p>
                    <p>환승: {data2['할당_환승']:,.0f}명</p>
                </div>
                """, unsafe_allow_html=True)

            # 비교 차트
            st.markdown("### 수요 비교")

            comparison_df = pd.DataFrame({
                '구분': ['승차', '하차', '환승'],
                station1: [data1['할당_승차'], data1['할당_하차'], data1['할당_환승']],
                station2: [data2['할당_승차'], data2['할당_하차'], data2['할당_환승']]
            })

            fig = go.Figure(data=[
                go.Bar(name=station1, x=comparison_df['구분'], y=comparison_df[station1]),
                go.Bar(name=station2, x=comparison_df['구분'], y=comparison_df[station2])
            ])

            fig.update_layout(
                barmode='group',
                title='승하차 및 환승 비교',
                yaxis_title='인원 (명)',
                height=400
            )

            st.plotly_chart(fig, use_container_width=True)

    # ========================================================================
    # 탭 5: 비교 분석
    # ========================================================================
    with tab5:
        st.markdown("## 📈 기존 vs 신규 비교 분석")

        col_compare1, col_compare2 = st.columns(2)

        with col_compare1:
            st.markdown("### 기존 정류장 통계")

            existing_stats = {
                '총 개수': len(stations),
                '활성 정류장': (stations['할당_총수요'] > 0).sum(),
                '평균 수요': stations['할당_총수요'].mean(),
                '총 수요': stations['할당_총수요'].sum()
            }

            for key, value in existing_stats.items():
                if '개수' in key or '활성' in key:
                    st.metric(key, f"{value:,}개")
                else:
                    st.metric(key, f"{value:,.0f}명")

        with col_compare2:
            st.markdown("### 신규 정류장 예측")

            new_stats = {
                '추천 개수': len(new_stations),
                '예상 총 수요': new_stations['수요'].sum(),
                '평균 수요': new_stations['수요'].mean(),
                '총 커버 수요': new_stations['커버_수요'].sum()
            }

            for key, value in new_stats.items():
                if '개수' in key:
                    st.metric(key, f"{value:,}개")
                else:
                    st.metric(key, f"{value:,.0f}명")

        # 수요 분포 비교
        st.markdown("### 수요 분포 비교")

        fig = go.Figure()

        fig.add_trace(go.Box(
            y=stations[stations['할당_총수요'] > 0]['할당_총수요'],
            name='기존 정류장',
            marker_color='#3B82F6'
        ))

        fig.add_trace(go.Box(
            y=new_stations['수요'],
            name='신규 정류장',
            marker_color='#EF4444'
        ))

        fig.update_layout(
            title='기존 vs 신규 정류장 수요 분포',
            yaxis_title='수요 (명)',
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

    # ========================================================================
    # 푸터
    # ========================================================================
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #6B7280; padding: 1rem 0;'>
        <p>세종시 버스정류장 최적화 고도화 대시보드 | 정밀 분석 시스템</p>
        <p>© 2025 Advanced Traffic Analysis</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
