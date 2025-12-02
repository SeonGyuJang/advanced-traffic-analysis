#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
세종시 버스 정류장 정밀 분석 - 기존 정류장 + 최적화 추천
========================================================
실제 버스정류장 데이터를 활용한 커버리지 분석 및 추가 정류장 제안
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns
from pulp import *
import folium
from folium import plugins
from math import radians, cos, sin, asin, sqrt
import warnings
from datetime import datetime
import json

warnings.filterwarnings('ignore')

# 전문가급 색상 팔레트
COLORS = {
    'primary': '#2E4057',
    'secondary': '#048A81',
    'accent': '#F26419',
    'success': '#06A77D',
    'warning': '#F4B41A',
    'danger': '#D64933',
    'info': '#5C7CFA',
    'light': '#F8F9FA',
    'dark': '#212529',
    'existing': '#9C27B0',  # 기존 정류장
    'new': '#FF5722',        # 신규 추천
}

# ============================================================================
# 한글 폰트 설정
# ============================================================================
def setup_korean_font():
    """한글 폰트 설정"""
    for font in ['NanumGothic', 'NanumBarunGothic', 'Malgun Gothic']:
        if font in [f.name for f in fm.fontManager.ttflist]:
            plt.rcParams['font.family'] = font
            plt.rcParams['axes.unicode_minus'] = False
            return True
    plt.rcParams['font.family'] = 'DejaVu Sans'
    return False

# ============================================================================
# GPS 거리 계산
# ============================================================================
def haversine_distance(lat1, lon1, lat2, lon2):
    """Haversine 공식으로 실제 거리 계산 (km)"""
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    return 6371 * 2 * asin(sqrt(a))

def calculate_coverage(stations_df, regions_df, radius_km):
    """정류장의 지역 커버리지 계산"""
    coverage = []
    for idx, region in regions_df.iterrows():
        min_dist = float('inf')
        nearest_station = None

        for sidx, station in stations_df.iterrows():
            dist = haversine_distance(
                region['위도'], region['경도'],
                station['위도'], station['경도']
            )
            if dist < min_dist:
                min_dist = dist
                nearest_station = station['정류소명']

        coverage.append({
            '지역': region['행정구역'],
            '최단거리_km': min_dist,
            '커버여부': 1 if min_dist <= radius_km else 0,
            '최인접정류장': nearest_station
        })

    return pd.DataFrame(coverage)

def find_optimal_new_stations(existing_stations, demand_regions, uncovered_regions,
                              max_new_stations, coverage_radius):
    """커버되지 않은 지역을 위한 최적 신규 정류장 위치 찾기"""

    if len(uncovered_regions) == 0:
        return []

    # 커버되지 않은 지역만 대상으로 최적화
    n = len(uncovered_regions)

    # 거리 행렬 계산
    distance_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            dist = haversine_distance(
                uncovered_regions.iloc[i]['위도'], uncovered_regions.iloc[i]['경도'],
                uncovered_regions.iloc[j]['위도'], uncovered_regions.iloc[j]['경도']
            )
            distance_matrix[i, j] = distance_matrix[j, i] = dist

    # 수요 가중치
    demand = uncovered_regions['총_이용객'].values
    if demand.max() > demand.min():
        demand_norm = (demand - demand.min()) / (demand.max() - demand.min())
    else:
        demand_norm = np.ones(len(demand))

    building = uncovered_regions['건물수'].values
    if building.max() > building.min():
        building_norm = (building - building.min()) / (building.max() - building.min())
    else:
        building_norm = np.ones(len(building))

    transfer = uncovered_regions['환승'].values
    if transfer.max() > transfer.min():
        transfer_norm = (transfer - transfer.min()) / (transfer.max() - transfer.min())
    else:
        transfer_norm = np.ones(len(transfer))

    weight = 0.60 * demand_norm + 0.25 * building_norm + 0.15 * transfer_norm

    # 커버리지 행렬
    coverage = (distance_matrix <= coverage_radius).astype(int)
    np.fill_diagonal(coverage, 1)

    # 정수계획법 모델
    prob = LpProblem("New_Station_Optimization", LpMaximize)
    x = LpVariable.dicts("station", range(n), cat='Binary')
    y = LpVariable.dicts("covered", range(n), cat='Binary')

    # 목적함수: 가중 수요 최대화
    prob += lpSum([demand[i] * weight[i] * y[i] for i in range(n)])

    # 제약조건: 신규 정류장 수
    prob += lpSum([x[i] for i in range(n)]) <= max_new_stations

    # 제약조건: 커버리지
    for i in range(n):
        prob += y[i] <= lpSum([coverage[i][j] * x[j] for j in range(n)])

    # 최적화 실행
    prob.solve(PULP_CBC_CMD(msg=0))

    # 결과 추출
    selected_indices = [i for i in range(n) if x[i].varValue == 1]
    selected_stations = uncovered_regions.iloc[selected_indices].copy()

    return selected_stations

# ============================================================================
# 메인 분석
# ============================================================================
print("="*100)
print("세종시 버스 정류장 정밀 커버리지 분석 + 최적화 추천".center(100))
print("="*100)
print(f"⏰ 분석 시작: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

use_korean = setup_korean_font()
print(f"{'✓' if use_korean else '⚠'} 한글 폰트: {'적용됨' if use_korean else '미적용'}\n")

# ============================================================================
# 1. 데이터 로드
# ============================================================================
print("[1단계] 데이터 로드")
print("-"*100)

# 기존 버스 정류장 데이터
df_existing_stations = pd.read_csv('data/세종도시교통공사_버스정류장_시설현황_20210924.csv')
# 중복 제거 (정류소ID 기준)
df_existing_stations = df_existing_stations.drop_duplicates(subset=['정류소ID'])

# 승하차 데이터
df_passenger = pd.read_csv('data/지역별승하차_통합데이터.csv')
df_coords = pd.read_csv('data/행정구역_중심좌표.csv')

print(f"✓ 기존 버스 정류장: {len(df_existing_stations):,}개")
print(f"✓ 승하차 데이터: {df_passenger.shape[0]:,}건")
print(f"✓ 행정구역: {len(df_coords)}개")

# ============================================================================
# 2. 수요 데이터 전처리
# ============================================================================
print("\n[2단계] 수요 데이터 전처리")
print("-"*100)

df_passenger['날짜'] = pd.to_datetime(df_passenger['날짜'])
df_passenger['총_이용객'] = df_passenger['승차'] + df_passenger['하차']

region_stats = df_passenger.groupby('행정구역').agg({
    '승차': 'sum',
    '하차': 'sum',
    '환승': 'sum',
    '총_이용객': 'sum'
}).reset_index()

df_demand = pd.merge(region_stats, df_coords, on='행정구역')

total_passengers = df_demand['총_이용객'].sum()
print(f"✓ 총 이용객: {total_passengers:,.0f}명")
print(f"✓ 분석 지역: {len(df_demand)}개")

# ============================================================================
# 3. 기존 정류장의 커버리지 분석
# ============================================================================
print("\n[3단계] 기존 정류장 커버리지 분석")
print("-"*100)

COVERAGE_RADIUS = 0.5  # 500m (도보 5-7분 거리)
print(f"⚙️  커버리지 반경: {COVERAGE_RADIUS}km (도보권)")

# 커버리지 계산
coverage_df = calculate_coverage(df_existing_stations, df_demand, COVERAGE_RADIUS)
df_demand_with_coverage = pd.merge(
    df_demand,
    coverage_df[['지역', '최단거리_km', '커버여부', '최인접정류장']],
    left_on='행정구역',
    right_on='지역'
).drop('지역', axis=1)

# 통계
covered_regions = df_demand_with_coverage[df_demand_with_coverage['커버여부'] == 1]
uncovered_regions = df_demand_with_coverage[df_demand_with_coverage['커버여부'] == 0]

covered_demand = covered_regions['총_이용객'].sum()
uncovered_demand = uncovered_regions['총_이용객'].sum()
coverage_rate = (len(covered_regions) / len(df_demand)) * 100
demand_coverage_rate = (covered_demand / total_passengers) * 100

print(f"\n✅ 현재 커버리지 현황:")
print(f"  • 커버 지역: {len(covered_regions)}/{len(df_demand)}개 ({coverage_rate:.2f}%)")
print(f"  • 커버 수요: {covered_demand:,.0f}명 ({demand_coverage_rate:.2f}%)")
print(f"  • 미커버 지역: {len(uncovered_regions)}개")
print(f"  • 미커버 수요: {uncovered_demand:,.0f}명 ({(uncovered_demand/total_passengers*100):.2f}%)")

if len(uncovered_regions) > 0:
    print(f"\n⚠️  미커버 지역 Top 5:")
    top_uncovered = uncovered_regions.nlargest(5, '총_이용객')
    for idx, row in top_uncovered.iterrows():
        print(f"  • {row['행정구역']:15s} | 이용객: {row['총_이용객']:>10,.0f}명 | 최단거리: {row['최단거리_km']:.2f}km")

# ============================================================================
# 4. 최적 신규 정류장 제안
# ============================================================================
print("\n[4단계] 최적 신규 정류장 위치 제안")
print("-"*100)

MAX_NEW_STATIONS = 10
print(f"⚙️  최대 신규 정류장 수: {MAX_NEW_STATIONS}개")

if len(uncovered_regions) > 0:
    print("🚀 최적화 실행 중...")

    new_stations = find_optimal_new_stations(
        df_existing_stations,
        df_demand,
        uncovered_regions,
        MAX_NEW_STATIONS,
        COVERAGE_RADIUS
    )

    if len(new_stations) > 0:
        print(f"\n✅ 신규 정류장 {len(new_stations)}개 제안:")
        for rank, (idx, row) in enumerate(new_stations.iterrows(), 1):
            print(f"  {rank}. {row['행정구역']:15s} | 이용객: {row['총_이용객']:>10,.0f}명 | "
                  f"건물: {row['건물수']:>6,.0f}개 | GPS: ({row['위도']:.4f}, {row['경도']:.4f})")

        # 신규 정류장 추가 후 커버리지 재계산
        combined_stations = pd.concat([
            df_existing_stations[['정류소명', '위도', '경도']],
            new_stations[['행정구역', '위도', '경도']].rename(columns={'행정구역': '정류소명'})
        ], ignore_index=True)

        new_coverage_df = calculate_coverage(combined_stations, df_demand, COVERAGE_RADIUS)
        new_covered = new_coverage_df[new_coverage_df['커버여부'] == 1]
        new_coverage_rate = (len(new_covered) / len(df_demand)) * 100

        df_demand_final = pd.merge(
            df_demand,
            new_coverage_df[['지역', '최단거리_km', '커버여부']],
            left_on='행정구역',
            right_on='지역',
            suffixes=('', '_신규')
        )

        new_covered_demand = df_demand_final[df_demand_final['커버여부'] == 1]['총_이용객'].sum()
        new_demand_coverage_rate = (new_covered_demand / total_passengers) * 100

        print(f"\n📊 신규 정류장 추가 후 예상 효과:")
        print(f"  • 지역 커버리지: {coverage_rate:.2f}% → {new_coverage_rate:.2f}% (+{new_coverage_rate-coverage_rate:.2f}%p)")
        print(f"  • 수요 커버리지: {demand_coverage_rate:.2f}% → {new_demand_coverage_rate:.2f}% (+{new_demand_coverage_rate-demand_coverage_rate:.2f}%p)")
        print(f"  • 추가 커버 수요: {new_covered_demand - covered_demand:,.0f}명")
    else:
        print("⚠️  추천할 신규 정류장 없음")
        new_stations = pd.DataFrame()
else:
    print("✅ 모든 지역이 이미 커버되어 있습니다!")
    new_stations = pd.DataFrame()
    df_demand_final = df_demand_with_coverage.copy()

# ============================================================================
# 5. 인터랙티브 지도 생성
# ============================================================================
print("\n[5단계] 인터랙티브 지도 생성")
print("-"*100)

center_lat = df_demand['위도'].mean()
center_lon = df_demand['경도'].mean()

# 베이스 맵
m = folium.Map(
    location=[center_lat, center_lon],
    zoom_start=11,
    tiles='CartoDB positron',
    control_scale=True
)

# 타일 레이어
folium.TileLayer('CartoDB dark_matter', name='Dark Mode').add_to(m)
folium.TileLayer('OpenStreetMap', name='기본 지도').add_to(m)

# 피처 그룹
fg_existing = folium.FeatureGroup(name='🚏 기존 버스정류장 (실제)', show=True)
fg_new = folium.FeatureGroup(name='⭐ 신규 정류장 (추천)', show=True)
fg_coverage_existing = folium.FeatureGroup(name='📍 기존 정류장 커버리지', show=False)
fg_coverage_new = folium.FeatureGroup(name='🎯 신규 정류장 커버리지', show=False)
fg_regions = folium.FeatureGroup(name='🗺️ 수요 지역', show=True)

# 1. 기존 버스 정류장 표시 (작은 마커, 클러스터링)
marker_cluster = plugins.MarkerCluster(name='기존 정류장 클러스터').add_to(fg_existing)

for idx, station in df_existing_stations.iterrows():
    folium.CircleMarker(
        location=[station['위도'], station['경도']],
        radius=3,
        color=COLORS['existing'],
        fillColor=COLORS['existing'],
        fillOpacity=0.6,
        weight=1,
        popup=folium.Popup(f"<b>{station['정류소명']}</b><br>노선: {station['노선번호']}", max_width=200),
        tooltip=station['정류소명']
    ).add_to(marker_cluster)

# 기존 정류장 커버리지 (샘플링해서 표시)
sample_stations = df_existing_stations.sample(min(50, len(df_existing_stations)))
for idx, station in sample_stations.iterrows():
    folium.Circle(
        location=[station['위도'], station['경도']],
        radius=COVERAGE_RADIUS * 1000,
        color=COLORS['existing'],
        fill=True,
        fillColor=COLORS['existing'],
        fillOpacity=0.05,
        opacity=0.2,
        weight=1
    ).add_to(fg_coverage_existing)

# 2. 신규 추천 정류장
if len(new_stations) > 0:
    for rank, (idx, station) in enumerate(new_stations.iterrows(), 1):
        # 커버리지 원
        folium.Circle(
            location=[station['위도'], station['경도']],
            radius=COVERAGE_RADIUS * 1000,
            color=COLORS['new'],
            fill=True,
            fillColor=COLORS['new'],
            fillOpacity=0.15,
            opacity=0.5,
            weight=2
        ).add_to(fg_coverage_new)

        # 마커
        icon_html = f'''
        <div style="
            background: linear-gradient(135deg, #FF5722, #F44336);
            width: 45px;
            height: 45px;
            border-radius: 50%;
            border: 4px solid white;
            box-shadow: 0 4px 12px rgba(0,0,0,0.4);
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            color: white;
            font-size: 18px;
        ">{rank}</div>
        '''

        popup_html = f'''
        <div style="font-family: 'Malgun Gothic', sans-serif; min-width: 300px;">
            <h3 style="margin:0 0 10px 0; color: #FF5722; border-bottom: 2px solid #FF5722; padding-bottom: 5px;">
                ⭐ 신규 정류장 추천 #{rank}
            </h3>
            <table style="width: 100%; font-size: 13px;">
                <tr><td><b>위치</b></td><td><b>{station['행정구역']}</b></td></tr>
                <tr><td>총 이용객</td><td>{station['총_이용객']:,.0f}명</td></tr>
                <tr><td>승차</td><td>{station['승차']:,.0f}명</td></tr>
                <tr><td>하차</td><td>{station['하차']:,.0f}명</td></tr>
                <tr><td>환승</td><td>{station['환승']:,.0f}명</td></tr>
                <tr><td>건물 수</td><td>{station['건물수']:,.0f}개</td></tr>
                <tr><td>GPS</td><td>{station['위도']:.6f}, {station['경도']:.6f}</td></tr>
            </table>
            <div style="margin-top: 12px; padding: 10px; background: #FFF3E0; border-radius: 5px; font-size: 12px;">
                <b>💡 추천 이유:</b><br>
                • 기존 정류장 미커버 지역<br>
                • 높은 수요 및 건물 밀집도<br>
                • 최적 위치 알고리즘 선정
            </div>
        </div>
        '''

        folium.Marker(
            location=[station['위도'], station['경도']],
            icon=folium.DivIcon(html=icon_html),
            popup=folium.Popup(popup_html, max_width=320),
            tooltip=f"<b>신규 정류장 #{rank}</b><br>{station['행정구역']}"
        ).add_to(fg_new)

# 3. 수요 지역 표시
for idx, region in df_demand_with_coverage.iterrows():
    if region['커버여부'] == 1:
        color = COLORS['success']
        icon = '✅'
        status = '커버됨'
    else:
        color = COLORS['danger']
        icon = '⚠️'
        status = '미커버'

    popup_html = f'''
    <div style="font-family: 'Malgun Gothic', sans-serif; min-width: 260px;">
        <h4 style="margin:0 0 8px 0; color: {color};">{icon} {status}</h4>
        <table style="width: 100%; font-size: 12px;">
            <tr><td><b>지역</b></td><td><b>{region['행정구역']}</b></td></tr>
            <tr><td>이용객</td><td>{region['총_이용객']:,.0f}명</td></tr>
            <tr><td>최인접정류장</td><td>{region['최인접정류장']}</td></tr>
            <tr><td>거리</td><td>{region['최단거리_km']:.2f} km</td></tr>
        </table>
    </div>
    '''

    folium.CircleMarker(
        location=[region['위도'], region['경도']],
        radius=max(5, min(15, region['총_이용객'] / 100000)),
        color=color,
        fillColor=color,
        fillOpacity=0.6,
        weight=2,
        popup=folium.Popup(popup_html, max_width=280),
        tooltip=f"{region['행정구역']} ({status})"
    ).add_to(fg_regions)

# 레이어 추가
fg_coverage_existing.add_to(m)
fg_coverage_new.add_to(m)
fg_existing.add_to(m)
fg_regions.add_to(m)
fg_new.add_to(m)

# 플러그인
minimap = plugins.MiniMap(toggle_display=True, width=150, height=150)
m.add_child(minimap)

plugins.Fullscreen(
    position='topright',
    title='전체화면',
    title_cancel='전체화면 해제',
    force_separate_button=True
).add_to(m)

plugins.MeasureControl(
    position='topleft',
    primary_length_unit='kilometers',
    secondary_length_unit='meters'
).add_to(m)

folium.LayerControl(position='topright', collapsed=False).add_to(m)

# 범례
# 개선 효과 계산
coverage_improvement = new_coverage_rate - coverage_rate if len(new_stations) > 0 else 0
demand_improvement = new_demand_coverage_rate - demand_coverage_rate if len(new_stations) > 0 else 0

# 조건부 HTML 생성
new_station_info = ''
if len(new_stations) > 0:
    new_station_info = f'''<div style="background: #FFF3E0; padding: 10px; border-radius: 6px; font-size: 12px;">
        <div style="font-weight: bold; margin-bottom: 5px; color: {COLORS['accent']};">신규 정류장 추가 시</div>
        <table style="width: 100%; line-height: 1.6;">
            <tr><td>지역 커버율:</td><td style="text-align: right;"><b>{new_coverage_rate:.2f}%</b> (+{coverage_improvement:.1f}%p)</td></tr>
            <tr><td>수요 커버율:</td><td style="text-align: right;"><b style="color: {COLORS['success']};">{new_demand_coverage_rate:.2f}%</b> (+{demand_improvement:.1f}%p)</td></tr>
        </table>
    </div>'''

legend_html = f'''
<div style="
    position: fixed;
    bottom: 50px;
    right: 50px;
    width: 340px;
    background: white;
    border-radius: 10px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    z-index: 9999;
    font-family: 'Malgun Gothic', sans-serif;
    padding: 15px;
">
    <h4 style="margin: 0 0 12px 0; color: #2E4057; border-bottom: 2px solid #048A81; padding-bottom: 8px;">
        📊 세종시 버스 정류장 정밀 분석
    </h4>

    <div style="margin-bottom: 12px;">
        <div style="display: flex; align-items: center; margin-bottom: 6px;">
            <div style="width: 12px; height: 12px; background: {COLORS['existing']}; border-radius: 50%; margin-right: 10px;"></div>
            <span><b>기존 버스정류장</b> ({len(df_existing_stations):,}개)</span>
        </div>
        <div style="display: flex; align-items: center; margin-bottom: 6px;">
            <div style="width: 35px; height: 35px; background: linear-gradient(135deg, #FF5722, #F44336);
                        border-radius: 50%; border: 3px solid white; margin-right: 10px;
                        display: flex; align-items: center; justify-content: center;
                        color: white; font-weight: bold; font-size: 14px;">N</div>
            <span><b>신규 추천 정류장</b> ({len(new_stations)}개)</span>
        </div>
        <div style="display: flex; align-items: center; margin-bottom: 6px;">
            <div style="width: 16px; height: 16px; background: {COLORS['success']}; border-radius: 50%; margin-right: 10px;"></div>
            <span>커버되는 수요 지역</span>
        </div>
        <div style="display: flex; align-items: center;">
            <div style="width: 16px; height: 16px; background: {COLORS['danger']}; border-radius: 50%; margin-right: 10px;"></div>
            <span>미커버 수요 지역</span>
        </div>
    </div>

    <div style="background: #F8F9FA; padding: 10px; border-radius: 6px; font-size: 12px; margin-bottom: 10px;">
        <div style="font-weight: bold; margin-bottom: 5px; color: {COLORS['primary']};">현재 커버리지</div>
        <table style="width: 100%; line-height: 1.6;">
            <tr><td>지역 커버율:</td><td style="text-align: right;"><b>{coverage_rate:.2f}%</b></td></tr>
            <tr><td>수요 커버율:</td><td style="text-align: right;"><b style="color: {COLORS['info']};">{demand_coverage_rate:.2f}%</b></td></tr>
            <tr><td>커버 수요:</td><td style="text-align: right;">{covered_demand/1e6:.2f}M명</td></tr>
        </table>
    </div>

    {new_station_info}

    <div style="margin-top: 10px; font-size: 11px; color: #666; text-align: center;">
        📱 마커 클릭 시 상세정보 | 커버리지: {COVERAGE_RADIUS*1000}m
    </div>
</div>
'''
m.get_root().html.add_child(folium.Element(legend_html))

# 통계 패널
stats_html = f'''
<div style="
    position: fixed;
    top: 80px;
    left: 50px;
    width: 260px;
    background: rgba(255, 255, 255, 0.95);
    border-radius: 10px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    z-index: 9999;
    font-family: 'Malgun Gothic', sans-serif;
    padding: 15px;
">
    <h4 style="margin: 0 0 10px 0; color: #2E4057;">📈 핵심 지표</h4>
    <div style="font-size: 13px; line-height: 1.8;">
        <div style="margin-bottom: 8px; padding-bottom: 8px; border-bottom: 1px solid #E0E0E0;">
            <div style="color: #666;">기존 버스정류장</div>
            <div style="font-size: 22px; font-weight: bold; color: {COLORS['existing']};">{len(df_existing_stations):,}개</div>
        </div>
        <div style="margin-bottom: 8px; padding-bottom: 8px; border-bottom: 1px solid #E0E0E0;">
            <div style="color: #666;">총 이용객</div>
            <div style="font-size: 20px; font-weight: bold; color: #2E4057;">{total_passengers/1e6:.2f}M</div>
        </div>
        <div style="margin-bottom: 8px; padding-bottom: 8px; border-bottom: 1px solid #E0E0E0;">
            <div style="color: #666;">분석 지역</div>
            <div style="font-size: 18px; font-weight: bold; color: #048A81;">{len(df_demand)}개</div>
        </div>
        <div>
            <div style="color: #666;">커버리지 반경</div>
            <div style="font-size: 14px; color: #5C7CFA;">{COVERAGE_RADIUS*1000}m (도보권)</div>
        </div>
    </div>
</div>
'''
m.get_root().html.add_child(folium.Element(stats_html))

# 지도 저장
m.save('정밀분석_인터랙티브_지도.html')
print("✓ 저장: 정밀분석_인터랙티브_지도.html")

# ============================================================================
# 6. 고급 시각화
# ============================================================================
print("\n[6단계] 고급 시각화 생성")
print("-"*100)

fig = plt.figure(figsize=(22, 14))
gs = GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.3)

fig.patch.set_facecolor('white')
title_text = '세종시 버스 정류장 정밀 커버리지 분석 - 종합 리포트'
fig.suptitle(title_text, fontsize=24, fontweight='bold', y=0.98, color=COLORS['dark'])

subtitle = f"기존 정류장: {len(df_existing_stations)}개 | 신규 추천: {len(new_stations)}개 | 분석일: {datetime.now().strftime('%Y.%m.%d')}"
fig.text(0.5, 0.955, subtitle, ha='center', fontsize=12, color=COLORS['primary'], alpha=0.8)

# 1. 커버리지 현황 비교 (좌상단, 2칸)
ax1 = fig.add_subplot(gs[0, :2])
categories = ['현재', '신규 추가 후']
coverage_rates = [coverage_rate, new_coverage_rate if len(new_stations) > 0 else coverage_rate]
demand_coverage_rates = [demand_coverage_rate, new_demand_coverage_rate if len(new_stations) > 0 else demand_coverage_rate]

x = np.arange(len(categories))
width = 0.35

bars1 = ax1.bar(x - width/2, coverage_rates, width, label='지역 커버율',
                color=COLORS['info'], edgecolor=COLORS['dark'], linewidth=2, alpha=0.85)
bars2 = ax1.bar(x + width/2, demand_coverage_rates, width, label='수요 커버율',
                color=COLORS['success'], edgecolor=COLORS['dark'], linewidth=2, alpha=0.85)

ax1.set_ylabel('커버리지 (%)', fontsize=13, fontweight='bold')
ax1.set_title('📊 커버리지 개선 효과', fontsize=16, fontweight='bold', pad=15)
ax1.set_xticks(x)
ax1.set_xticklabels(categories, fontsize=12)
ax1.legend(loc='upper left', fontsize=11, framealpha=0.9)
ax1.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
ax1.set_ylim([0, 105])
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# 값 레이블
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

# 2. 수요 Top 10 (우상단)
ax2 = fig.add_subplot(gs[0:2, 2:])
top10 = df_demand.nlargest(10, '총_이용객')
y_pos = np.arange(len(top10))

# 색상: 기존 커버 / 미커버 / 신규로 커버 예정
colors_bar = []
for _, row in top10.iterrows():
    region_name = row['행정구역']
    current_coverage = df_demand_with_coverage[df_demand_with_coverage['행정구역'] == region_name]['커버여부'].values[0]

    if current_coverage == 1:
        colors_bar.append(COLORS['success'])  # 이미 커버됨
    elif len(new_stations) > 0 and region_name in new_stations['행정구역'].values:
        colors_bar.append(COLORS['new'])  # 신규 정류장 위치
    else:
        colors_bar.append(COLORS['danger'])  # 미커버

bars = ax2.barh(y_pos, top10['총_이용객'].values/1e6, color=colors_bar,
                edgecolor=COLORS['dark'], linewidth=1.5, alpha=0.85)
ax2.set_yticks(y_pos)
ax2.set_yticklabels(top10['행정구역'].values, fontsize=11)
ax2.set_xlabel('이용객 수 (백만명)', fontsize=12, fontweight='bold')
ax2.set_title('🏆 상위 10개 수요 지역', fontsize=16, fontweight='bold', pad=15)
ax2.invert_yaxis()
ax2.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.8)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

for i, (idx, row) in enumerate(top10.iterrows()):
    ax2.text(row['총_이용객']/1e6 + 0.1, i, f"{row['총_이용객']/1e6:.2f}M",
             va='center', fontsize=10, fontweight='bold')

legend_elements = [
    mpatches.Patch(facecolor=COLORS['success'], edgecolor=COLORS['dark'], label='현재 커버됨'),
    mpatches.Patch(facecolor=COLORS['new'], edgecolor=COLORS['dark'], label='신규 정류장 위치'),
    mpatches.Patch(facecolor=COLORS['danger'], edgecolor=COLORS['dark'], label='미커버')
]
ax2.legend(handles=legend_elements, loc='lower right', framealpha=0.9, fontsize=10)

# 3. 거리 분포 (좌중단)
ax3 = fig.add_subplot(gs[1, 0])
distances = df_demand_with_coverage['최단거리_km'].values
bins = [0, 0.5, 1.0, 1.5, 2.0, 5.0, 100]
labels = ['<0.5km', '0.5-1km', '1-1.5km', '1.5-2km', '2-5km', '>5km']
dist_counts = pd.cut(distances, bins=bins, labels=labels).value_counts().sort_index()

colors_dist = [COLORS['success'], COLORS['success'], COLORS['warning'],
               COLORS['warning'], COLORS['danger'], COLORS['danger']]
ax3.bar(range(len(dist_counts)), dist_counts.values, color=colors_dist,
        edgecolor=COLORS['dark'], linewidth=1.5, alpha=0.85)
ax3.set_xticks(range(len(dist_counts)))
ax3.set_xticklabels(labels, rotation=30, ha='right', fontsize=10)
ax3.set_ylabel('지역 수', fontsize=11, fontweight='bold')
ax3.set_title('📏 최인접 정류장 거리 분포', fontsize=14, fontweight='bold', pad=15)
ax3.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)

for i, val in enumerate(dist_counts.values):
    ax3.text(i, val + 0.5, str(val), ha='center', fontsize=10, fontweight='bold')

# 4. 버스 정류장 노선별 분포 (좌하단)
ax4 = fig.add_subplot(gs[1, 1])
route_counts = df_existing_stations['노선번호'].value_counts().head(10)
ax4.bar(range(len(route_counts)), route_counts.values, color=COLORS['existing'],
        edgecolor=COLORS['dark'], linewidth=1.5, alpha=0.85)
ax4.set_xticks(range(len(route_counts)))
ax4.set_xticklabels(route_counts.index, rotation=45, ha='right', fontsize=9)
ax4.set_ylabel('정류장 수', fontsize=11, fontweight='bold')
ax4.set_title('🚌 노선별 정류장 수 Top 10', fontsize=14, fontweight='bold', pad=15)
ax4.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)

# 5. GPS 지도 - 전체 (하단 왼쪽 2칸)
ax5 = fig.add_subplot(gs[2, :2])

# 기존 정류장 (작게, 많이)
ax5.scatter(df_existing_stations['경도'], df_existing_stations['위도'],
           s=8, c=COLORS['existing'], alpha=0.3, label=f'기존 정류장 ({len(df_existing_stations)}개)')

# 수요 지역
scatter_colors = [COLORS['success'] if row['커버여부']==1 else COLORS['danger']
                  for _, row in df_demand_with_coverage.iterrows()]
scatter_sizes = [max(50, min(300, row['총_이용객']/10000)) for _, row in df_demand_with_coverage.iterrows()]
ax5.scatter(df_demand_with_coverage['경도'], df_demand_with_coverage['위도'],
           s=scatter_sizes, c=scatter_colors, alpha=0.6, edgecolors='white', linewidth=1.5, zorder=3)

# 신규 정류장
if len(new_stations) > 0:
    ax5.scatter(new_stations['경도'], new_stations['위도'],
               s=500, c=COLORS['new'], marker='*', edgecolors='white', linewidth=2,
               label=f'신규 추천 ({len(new_stations)}개)', zorder=5, alpha=0.9)

    for rank, (idx, row) in enumerate(new_stations.iterrows(), 1):
        ax5.text(row['경도'], row['위도'], str(rank), ha='center', va='center',
                fontsize=10, fontweight='bold', color='white', zorder=6)

ax5.set_xlabel('경도', fontsize=12, fontweight='bold')
ax5.set_ylabel('위도', fontsize=12, fontweight='bold')
ax5.set_title('🗺️ 전체 정류장 및 수요 분포', fontsize=16, fontweight='bold', pad=15)
ax5.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
ax5.set_facecolor('#F8F9FA')
ax5.legend(loc='upper left', fontsize=10, framealpha=0.9)

# 6. 신규 정류장 상세 테이블 (하단 우측 2칸)
ax6 = fig.add_subplot(gs[2, 2:])
ax6.axis('off')

if len(new_stations) > 0:
    table_data = []
    for rank, (idx, station) in enumerate(new_stations.iterrows(), 1):
        table_data.append([
            f"#{rank}",
            station['행정구역'],
            f"{station['총_이용객']/1e6:.2f}M",
            f"{station['건물수']:,.0f}",
            f"({station['위도']:.4f}, {station['경도']:.4f})"
        ])

    table = ax6.table(cellText=table_data,
                     colLabels=['순위', '위치', '이용객', '건물수', 'GPS 좌표'],
                     cellLoc='center',
                     loc='center',
                     colWidths=[0.12, 0.25, 0.18, 0.15, 0.30])

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.2)

    # 헤더
    for i in range(5):
        table[(0, i)].set_facecolor(COLORS['new'])
        table[(0, i)].set_text_props(weight='bold', color='white')

    # 행
    for i in range(1, len(table_data)+1):
        for j in range(5):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#FFF3E0')
            else:
                table[(i, j)].set_facecolor('white')
            table[(i, j)].set_edgecolor(COLORS['light'])

    ax6.set_title('⭐ 신규 정류장 추천 상세', fontsize=16, fontweight='bold', pad=20, loc='left')
else:
    ax6.text(0.5, 0.5, '모든 지역이 이미 충분히 커버되어 있습니다!\n추가 정류장이 필요하지 않습니다.',
             ha='center', va='center', fontsize=14, color=COLORS['success'], weight='bold')
    ax6.set_title('⭐ 신규 정류장 추천', fontsize=16, fontweight='bold', pad=20, loc='left')

plt.savefig('정밀분석_종합시각화.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✓ 저장: 정밀분석_종합시각화.png")
plt.close()

# ============================================================================
# 7. 결과 저장
# ============================================================================
print("\n[7단계] 결과 저장")
print("-"*100)

# 커버리지 분석 결과
df_demand_with_coverage.to_csv('정밀분석_커버리지_결과.csv', index=False, encoding='utf-8-sig')
print("✓ 저장: 정밀분석_커버리지_결과.csv")

# 신규 정류장 추천
if len(new_stations) > 0:
    new_stations.to_csv('정밀분석_신규정류장_추천.csv', index=False, encoding='utf-8-sig')
    print("✓ 저장: 정밀분석_신규정류장_추천.csv")

# 통합 보고서 생성
report = {
    '분석_일시': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    '기존_버스정류장_수': len(df_existing_stations),
    '분석_지역_수': len(df_demand),
    '총_이용객': int(total_passengers),
    '커버리지_반경_km': COVERAGE_RADIUS,
    '현재_지역_커버율_%': round(coverage_rate, 2),
    '현재_수요_커버율_%': round(demand_coverage_rate, 2),
    '커버_지역_수': len(covered_regions),
    '미커버_지역_수': len(uncovered_regions),
    '커버_수요': int(covered_demand),
    '미커버_수요': int(uncovered_demand),
    '신규_정류장_추천_수': len(new_stations),
}

if len(new_stations) > 0:
    report['개선후_지역_커버율_%'] = round(new_coverage_rate, 2)
    report['개선후_수요_커버율_%'] = round(new_demand_coverage_rate, 2)
    report['개선효과_지역_%p'] = round(new_coverage_rate - coverage_rate, 2)
    report['개선효과_수요_%p'] = round(new_demand_coverage_rate - demand_coverage_rate, 2)

with open('정밀분석_통합보고서.json', 'w', encoding='utf-8') as f:
    json.dump(report, f, ensure_ascii=False, indent=2)
print("✓ 저장: 정밀분석_통합보고서.json")

# ============================================================================
# 완료
# ============================================================================
print("\n" + "="*100)
print("✅ 정밀 분석 완료!".center(100))
print("="*100)
print("\n📁 생성된 파일:")
print("  1. 정밀분석_인터랙티브_지도.html - 기존+신규 정류장 통합 지도")
print("  2. 정밀분석_종합시각화.png - 종합 분석 차트")
print("  3. 정밀분석_커버리지_결과.csv - 지역별 커버리지 상세")
print("  4. 정밀분석_신규정류장_추천.csv - 신규 정류장 추천 목록")
print("  5. 정밀분석_통합보고서.json - 분석 결과 요약")
print(f"\n⏰ 분석 종료: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*100)
print("\n💡 주요 인사이트:")
print(f"  • 현재 {len(df_existing_stations):,}개 버스 정류장으로 {coverage_rate:.1f}%의 지역을 커버")
print(f"  • {len(uncovered_regions)}개 지역이 커버리지 밖에 위치 (도보 {COVERAGE_RADIUS}km 초과)")
print(f"  • {len(new_stations)}개 신규 정류장 추가 시 커버리지 {new_coverage_rate:.1f}%로 개선 가능" if len(new_stations) > 0
      else f"  • 현재 커버리지가 우수하여 추가 정류장 불필요")
print("="*100)
