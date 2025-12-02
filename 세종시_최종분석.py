#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
세종시 버스 정류장 최적화 - 전문가급 정밀 분석
==============================================
GPS 기반 정수계획법 최적화 + 고급 시각화
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
    'gradient1': ['#667eea', '#764ba2'],
    'gradient2': ['#f093fb', '#f5576c'],
    'gradient3': ['#4facfe', '#00f2fe'],
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

def create_distance_matrix(df):
    """거리 행렬 생성"""
    n = len(df)
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            dist = haversine_distance(
                df.iloc[i]['위도'], df.iloc[i]['경도'],
                df.iloc[j]['위도'], df.iloc[j]['경도']
            )
            matrix[i, j] = matrix[j, i] = dist
    return matrix

# ============================================================================
# 메인 분석
# ============================================================================
print("="*100)
print("세종시 버스 정류장 최적화 - 전문가급 정밀 분석".center(100))
print("="*100)
print(f"⏰ 분석 시작: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

use_korean = setup_korean_font()
print(f"{'✓' if use_korean else '⚠'} 한글 폰트: {'적용됨' if use_korean else '미적용'}\n")

# ============================================================================
# 1. 데이터 로드
# ============================================================================
print("[1단계] 데이터 로드")
print("-"*100)

df_passenger = pd.read_csv('data/지역별승하차_통합데이터.csv')
df_coords = pd.read_csv('data/행정구역_중심좌표.csv')

print(f"✓ 승하차: {df_passenger.shape[0]:,}건")
print(f"✓ GPS 좌표: {df_coords.shape[0]}개 지역")

# ============================================================================
# 2. 데이터 전처리
# ============================================================================
print("\n[2단계] 데이터 전처리")
print("-"*100)

df_passenger['날짜'] = pd.to_datetime(df_passenger['날짜'])
df_passenger['총_이용객'] = df_passenger['승차'] + df_passenger['하차']

region_stats = df_passenger.groupby('행정구역').agg({
    '승차': 'sum',
    '하차': 'sum',
    '환승': 'sum',
    '총_이용객': 'sum'
}).reset_index().sort_values('총_이용객', ascending=False)

df_analysis = pd.merge(region_stats, df_coords, on='행정구역')

total_passengers = df_analysis['총_이용객'].sum()
print(f"✓ 총 이용객: {total_passengers:,.0f}명")
print(f"✓ 분석 지역: {len(df_analysis)}개")

# ============================================================================
# 3. GPS 거리 행렬
# ============================================================================
print("\n[3단계] GPS 거리 계산")
print("-"*100)

distance_matrix = create_distance_matrix(df_analysis)
print(f"✓ 거리 범위: {distance_matrix[distance_matrix > 0].min():.2f} ~ {distance_matrix.max():.2f} km")

# ============================================================================
# 4. 정수계획법 최적화
# ============================================================================
print("\n[4단계] 정수계획법 최적화")
print("-"*100)

MAX_STATIONS = 5
COVERAGE_RADIUS = 5.0

print(f"⚙️  파라미터: {MAX_STATIONS}개 정류장, {COVERAGE_RADIUS}km 반경")

# 가중치 계산
demand = df_analysis['총_이용객'].values
demand_norm = (demand - demand.min()) / (demand.max() - demand.min())

building = df_analysis['건물수'].values
building_norm = (building - building.min()) / (building.max() - building.min() + 1)

transfer = df_analysis['환승'].values
transfer_norm = (transfer - transfer.min()) / (transfer.max() - transfer.min() + 1)

weight = 0.60 * demand_norm + 0.25 * building_norm + 0.15 * transfer_norm

# 커버리지 행렬
n = len(df_analysis)
coverage = (distance_matrix <= COVERAGE_RADIUS).astype(int)
np.fill_diagonal(coverage, 1)

# IP 모델
prob = LpProblem("Bus_Station_Optimization", LpMaximize)
x = LpVariable.dicts("station", range(n), cat='Binary')
y = LpVariable.dicts("covered", range(n), cat='Binary')

prob += lpSum([demand[i] * weight[i] * y[i] for i in range(n)])
prob += lpSum([x[i] for i in range(n)]) <= MAX_STATIONS

for i in range(n):
    prob += y[i] <= lpSum([coverage[i][j] * x[j] for j in range(n)])

print("🚀 최적화 실행 중...")
prob.solve(PULP_CBC_CMD(msg=0))

# 결과
selected = [i for i in range(n) if x[i].varValue == 1]
covered = [i for i in range(n) if y[i].varValue == 1]
not_covered = [i for i in range(n) if i not in covered]

covered_demand = sum([demand[i] for i in covered])
coverage_rate = (covered_demand / demand.sum()) * 100

print(f"\n✅ 최적화 완료")
print(f"  • 선정: {len(selected)}개")
print(f"  • 커버율: {coverage_rate:.2f}% ({len(covered)}/{n}개 지역)")

print(f"\n🎯 선정된 정류장:")
for rank, idx in enumerate(selected, 1):
    row = df_analysis.iloc[idx]
    print(f"  {rank}. {row['행정구역']:10s} | {row['총_이용객']:>10,.0f}명")

# 결과 저장
df_result = df_analysis.copy()
df_result['신규정류장'] = [1 if i in selected else 0 for i in range(n)]
df_result['커버여부'] = [1 if i in covered else 0 for i in range(n)]

min_dist = []
for i in range(n):
    if i in selected:
        min_dist.append(0.0)
    else:
        min_dist.append(min([distance_matrix[i][j] for j in selected]) if selected else 999)
df_result['최단거리_km'] = min_dist

# ============================================================================
# 5. 전문가급 인터랙티브 지도
# ============================================================================
print("\n[5단계] 전문가급 인터랙티브 지도 생성")
print("-"*100)

center_lat = df_analysis['위도'].mean()
center_lon = df_analysis['경도'].mean()

# 베이스 맵 생성 (고품질 타일)
m = folium.Map(
    location=[center_lat, center_lon],
    zoom_start=11,
    tiles='CartoDB positron',
    control_scale=True
)

# 타일 레이어 추가
folium.TileLayer('CartoDB dark_matter', name='Dark Mode').add_to(m)
folium.TileLayer('OpenStreetMap', name='기본 지도').add_to(m)

# 피처 그룹 생성
fg_coverage = folium.FeatureGroup(name='📍 커버리지 범위', show=True)
fg_stations = folium.FeatureGroup(name='🚏 버스 정류장', show=True)
fg_regions = folium.FeatureGroup(name='🗺️ 행정구역', show=True)
fg_connections = folium.FeatureGroup(name='🔗 네트워크 연결', show=False)

# 1. 커버리지 원 (반투명)
for idx in selected:
    row = df_result.iloc[idx]
    folium.Circle(
        location=[row['위도'], row['경도']],
        radius=COVERAGE_RADIUS * 1000,  # km to m
        color='#048A81',
        fill=True,
        fillColor='#048A81',
        fillOpacity=0.1,
        opacity=0.3,
        weight=2,
        popup=f"커버리지: {COVERAGE_RADIUS}km",
    ).add_to(fg_coverage)

# 2. 정류장 간 네트워크 연결선
for i, idx1 in enumerate(selected):
    for idx2 in selected[i+1:]:
        row1 = df_result.iloc[idx1]
        row2 = df_result.iloc[idx2]
        folium.PolyLine(
            locations=[[row1['위도'], row1['경도']], [row2['위도'], row2['경도']]],
            color='#5C7CFA',
            weight=1.5,
            opacity=0.4,
            dash_array='5, 10'
        ).add_to(fg_connections)

# 3. 신규 정류장 마커 (대형, 눈에 띄는)
for rank, idx in enumerate(selected, 1):
    row = df_result.iloc[idx]

    # 커스텀 아이콘 HTML
    icon_html = f'''
    <div style="
        background: linear-gradient(135deg, #F26419, #D64933);
        width: 40px;
        height: 40px;
        border-radius: 50%;
        border: 4px solid white;
        box-shadow: 0 4px 8px rgba(0,0,0,0.3);
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        color: white;
        font-size: 16px;
    ">{rank}</div>
    '''

    # 상세 팝업
    popup_html = f'''
    <div style="font-family: 'Malgun Gothic', sans-serif; min-width: 280px;">
        <h3 style="margin:0 0 10px 0; color: #F26419; border-bottom: 2px solid #F26419; padding-bottom: 5px;">
            🏆 신규 정류장 #{rank}
        </h3>
        <table style="width: 100%; font-size: 13px;">
            <tr><td><b>지역</b></td><td><b>{row['행정구역']}</b></td></tr>
            <tr><td>총 이용객</td><td>{row['총_이용객']:,.0f}명</td></tr>
            <tr><td>승차</td><td>{row['승차']:,.0f}명</td></tr>
            <tr><td>하차</td><td>{row['하차']:,.0f}명</td></tr>
            <tr><td>환승</td><td>{row['환승']:,.0f}명</td></tr>
            <tr><td>건물 수</td><td>{row['건물수']:,.0f}개</td></tr>
            <tr><td>GPS</td><td>{row['위도']:.4f}, {row['경도']:.4f}</td></tr>
        </table>
        <div style="margin-top: 10px; padding: 8px; background: #FFF3E0; border-radius: 5px; font-size: 12px;">
            <b>💡 우선순위:</b> {rank}순위<br>
            <b>📊 비중:</b> 전체의 {(row['총_이용객']/total_passengers*100):.1f}%
        </div>
    </div>
    '''

    folium.Marker(
        location=[row['위도'], row['경도']],
        icon=folium.DivIcon(html=icon_html),
        popup=folium.Popup(popup_html, max_width=300),
        tooltip=f"<b>{row['행정구역']}</b> (신규 정류장 #{rank})"
    ).add_to(fg_stations)

# 4. 커버되는 지역 마커
for idx in covered:
    if idx not in selected:
        row = df_result.iloc[idx]

        popup_html = f'''
        <div style="font-family: 'Malgun Gothic', sans-serif; min-width: 250px;">
            <h4 style="margin:0 0 8px 0; color: #06A77D;">✅ 커버되는 지역</h4>
            <table style="width: 100%; font-size: 12px;">
                <tr><td><b>지역</b></td><td><b>{row['행정구역']}</b></td></tr>
                <tr><td>이용객</td><td>{row['총_이용객']:,.0f}명</td></tr>
                <tr><td>최단 정류장</td><td>{row['최단거리_km']:.2f} km</td></tr>
            </table>
        </div>
        '''

        folium.CircleMarker(
            location=[row['위도'], row['경도']],
            radius=8,
            color='#06A77D',
            fillColor='#06A77D',
            fillOpacity=0.7,
            weight=2,
            popup=folium.Popup(popup_html, max_width=280),
            tooltip=f"{row['행정구역']} (커버됨)"
        ).add_to(fg_regions)

# 5. 커버되지 않는 지역
for idx in not_covered:
    row = df_result.iloc[idx]

    popup_html = f'''
    <div style="font-family: 'Malgun Gothic', sans-serif; min-width: 250px;">
        <h4 style="margin:0 0 8px 0; color: #D64933;">⚠️ 미커버 지역</h4>
        <table style="width: 100%; font-size: 12px;">
            <tr><td><b>지역</b></td><td><b>{row['행정구역']}</b></td></tr>
            <tr><td>이용객</td><td>{row['총_이용객']:,.0f}명</td></tr>
            <tr><td>최단 정류장</td><td>{row['최단거리_km']:.2f} km</td></tr>
        </table>
        <div style="margin-top: 8px; padding: 6px; background: #FFEBEE; border-radius: 4px; font-size: 11px;">
            추가 정류장 필요 또는<br>대체 교통수단 고려
        </div>
    </div>
    '''

    folium.CircleMarker(
        location=[row['위도'], row['경도']],
        radius=6,
        color='#D64933',
        fillColor='#FFF',
        fillOpacity=0.8,
        weight=2,
        popup=folium.Popup(popup_html, max_width=280),
        tooltip=f"{row['행정구역']} (미커버)"
    ).add_to(fg_regions)

# 레이어 추가
fg_coverage.add_to(m)
fg_connections.add_to(m)
fg_regions.add_to(m)
fg_stations.add_to(m)  # 정류장을 마지막에 추가하여 최상단 표시

# 미니맵
minimap = plugins.MiniMap(toggle_display=True, width=150, height=150)
m.add_child(minimap)

# 풀스크린
plugins.Fullscreen(
    position='topright',
    title='전체화면',
    title_cancel='전체화면 해제',
    force_separate_button=True
).add_to(m)

# 측정 도구
plugins.MeasureControl(
    position='topleft',
    primary_length_unit='kilometers',
    secondary_length_unit='meters',
    primary_area_unit='sqkilometers',
    secondary_area_unit='sqmeters'
).add_to(m)

# 레이어 컨트롤
folium.LayerControl(position='topright', collapsed=False).add_to(m)

# 고급 범례
legend_html = f'''
<div style="
    position: fixed;
    bottom: 50px;
    right: 50px;
    width: 320px;
    background: white;
    border-radius: 10px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    z-index: 9999;
    font-family: 'Malgun Gothic', sans-serif;
    padding: 15px;
">
    <h4 style="margin: 0 0 12px 0; color: #2E4057; border-bottom: 2px solid #048A81; padding-bottom: 8px;">
        📊 세종시 버스 정류장 최적화
    </h4>

    <div style="margin-bottom: 12px;">
        <div style="display: flex; align-items: center; margin-bottom: 6px;">
            <div style="width: 30px; height: 30px; background: linear-gradient(135deg, #F26419, #D64933);
                        border-radius: 50%; border: 3px solid white; margin-right: 10px;
                        display: flex; align-items: center; justify-content: center;
                        color: white; font-weight: bold; font-size: 12px;">1</div>
            <span><b>신규 정류장</b> (추천 위치)</span>
        </div>
        <div style="display: flex; align-items: center; margin-bottom: 6px;">
            <div style="width: 16px; height: 16px; background: #06A77D; border-radius: 50%;
                        margin-left: 7px; margin-right: 17px;"></div>
            <span>커버되는 지역</span>
        </div>
        <div style="display: flex; align-items: center; margin-bottom: 6px;">
            <div style="width: 12px; height: 12px; background: white; border: 2px solid #D64933;
                        border-radius: 50%; margin-left: 9px; margin-right: 19px;"></div>
            <span>미커버 지역</span>
        </div>
        <div style="display: flex; align-items: center;">
            <div style="width: 20px; height: 20px; background: rgba(4, 138, 129, 0.1);
                        border: 2px solid rgba(4, 138, 129, 0.3); border-radius: 50%;
                        margin-left: 5px; margin-right: 15px;"></div>
            <span>커버리지 범위 ({COVERAGE_RADIUS}km)</span>
        </div>
    </div>

    <div style="background: #F8F9FA; padding: 10px; border-radius: 6px; font-size: 12px;">
        <table style="width: 100%; line-height: 1.6;">
            <tr><td><b>신규 정류장:</b></td><td style="text-align: right;">{len(selected)}개</td></tr>
            <tr><td><b>커버 지역:</b></td><td style="text-align: right;">{len(covered)}/{n}개</td></tr>
            <tr><td><b>커버율:</b></td><td style="text-align: right;"><b style="color: #06A77D;">{coverage_rate:.2f}%</b></td></tr>
            <tr><td><b>커버 수요:</b></td><td style="text-align: right;">{covered_demand/1e6:.1f}M명</td></tr>
        </table>
    </div>

    <div style="margin-top: 10px; font-size: 11px; color: #666; text-align: center;">
        📱 마커 클릭 시 상세정보 확인
    </div>
</div>
'''
m.get_root().html.add_child(folium.Element(legend_html))

# 통계 패널 (좌측 상단)
stats_html = f'''
<div style="
    position: fixed;
    top: 80px;
    left: 50px;
    width: 250px;
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
            <div style="color: #666;">총 이용객</div>
            <div style="font-size: 20px; font-weight: bold; color: #2E4057;">{total_passengers/1e6:.1f}M</div>
        </div>
        <div style="margin-bottom: 8px; padding-bottom: 8px; border-bottom: 1px solid #E0E0E0;">
            <div style="color: #666;">분석 지역</div>
            <div style="font-size: 18px; font-weight: bold; color: #048A81;">{n}개</div>
        </div>
        <div>
            <div style="color: #666;">최적화 방법</div>
            <div style="font-size: 12px; color: #5C7CFA;">정수계획법 (IP)</div>
        </div>
    </div>
</div>
'''
m.get_root().html.add_child(folium.Element(stats_html))

# 지도 저장
m.save('인터랙티브_지도_최종.html')
print("✓ 저장: 인터랙티브_지도_최종.html")

# ============================================================================
# 6. 전문가급 시각화
# ============================================================================
print("\n[6단계] 전문가급 시각화")
print("-"*100)

# 한 페이지에 모든 차트 (전문가 레이아웃)
fig = plt.figure(figsize=(20, 14))
gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

fig.patch.set_facecolor('white')
title_text = '세종시 버스 정류장 최적화 분석 - 종합 리포트' if use_korean else 'Sejong Bus Station Optimization - Comprehensive Report'
fig.suptitle(title_text, fontsize=22, fontweight='bold', y=0.98, color=COLORS['dark'])

# 서브타이틀
subtitle = f"GPS 기반 정수계획법 최적화  |  커버리지: {coverage_rate:.2f}%  |  분석일: {datetime.now().strftime('%Y.%m.%d')}"
fig.text(0.5, 0.955, subtitle, ha='center', fontsize=11, color=COLORS['primary'], alpha=0.8)

# 1. 수요 Top 10 (좌상단, 큰 영역)
ax1 = fig.add_subplot(gs[0:2, 0])
top10 = df_result.nlargest(10, '총_이용객')
colors_bar = [COLORS['accent'] if row['신규정류장'] == 1 else COLORS['info'] for _, row in top10.iterrows()]
y_pos = np.arange(len(top10))

bars = ax1.barh(y_pos, top10['총_이용객'].values/1e6, color=colors_bar, edgecolor=COLORS['dark'], linewidth=1.5, alpha=0.85)
ax1.set_yticks(y_pos)
ax1.set_yticklabels(top10['행정구역'].values if use_korean else [f'R{i+1}' for i in range(len(top10))], fontsize=11)
ax1.set_xlabel('이용객 수 (백만명)' if use_korean else 'Passengers (Million)', fontsize=12, fontweight='bold')
ax1.set_title('🏆 상위 10개 수요 지역' if use_korean else 'Top 10 Demand Regions', fontsize=14, fontweight='bold', pad=15)
ax1.invert_yaxis()
ax1.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.8)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# 값 레이블
for i, (idx, row) in enumerate(top10.iterrows()):
    ax1.text(row['총_이용객']/1e6 + 0.1, i, f"{row['총_이용객']/1e6:.1f}M",
             va='center', fontsize=10, fontweight='bold')

# 범례
legend_elements = [
    mpatches.Patch(facecolor=COLORS['accent'], edgecolor=COLORS['dark'], label='신규 정류장'),
    mpatches.Patch(facecolor=COLORS['info'], edgecolor=COLORS['dark'], label='일반 지역')
]
ax1.legend(handles=legend_elements, loc='lower right', framealpha=0.9)

# 2. 커버리지 현황 (우상단)
ax2 = fig.add_subplot(gs[0, 1])
sizes = [len(covered), len(not_covered)]
labels = ['커버됨', '미커버'] if use_korean else ['Covered', 'Not Covered']
colors_pie = [COLORS['success'], COLORS['danger']]
explode = (0.05, 0.05)

wedges, texts, autotexts = ax2.pie(sizes, explode=explode, labels=labels, colors=colors_pie,
                                     autopct='%1.1f%%', startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'},
                                     wedgeprops={'edgecolor': 'white', 'linewidth': 2})

for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontsize(14)

ax2.set_title('📊 지역 커버리지' if use_korean else 'Regional Coverage', fontsize=14, fontweight='bold', pad=15)

# 중앙 텍스트
centre_circle = plt.Circle((0, 0), 0.70, fc='white', linewidth=0)
ax2.add_artist(centre_circle)
ax2.text(0, 0, f'{coverage_rate:.1f}%', ha='center', va='center', fontsize=24, fontweight='bold', color=COLORS['success'])

# 3. 수요 커버리지 (우중단)
ax3 = fig.add_subplot(gs[1, 1])
demand_sizes = [covered_demand, total_passengers - covered_demand]
demand_labels = ['커버 수요', '미커버 수요'] if use_korean else ['Covered Demand', 'Uncovered Demand']

wedges, texts, autotexts = ax3.pie(demand_sizes, explode=explode, labels=demand_labels, colors=colors_pie,
                                     autopct=lambda pct: f'{pct:.1f}%\n({pct/100*total_passengers/1e6:.1f}M)',
                                     startangle=90, textprops={'fontsize': 10, 'fontweight': 'bold'},
                                     wedgeprops={'edgecolor': 'white', 'linewidth': 2})

for autotext in autotexts:
    autotext.set_color('white')

ax3.set_title('💰 수요 커버리지' if use_korean else 'Demand Coverage', fontsize=14, fontweight='bold', pad=15)

# 4. GPS 지도 (우하단)
ax4 = fig.add_subplot(gs[2, 1])
scatter_colors = [COLORS['accent'] if row['신규정류장']==1 else (COLORS['success'] if row['커버여부']==1 else COLORS['danger'])
                  for _, row in df_result.iterrows()]
scatter_sizes = [(500 if row['신규정류장']==1 else 150) for _, row in df_result.iterrows()]

ax4.scatter(df_result['경도'], df_result['위도'], s=scatter_sizes, c=scatter_colors,
           alpha=0.7, edgecolors='white', linewidth=2, zorder=3)

# 신규 정류장에 번호 표시
for rank, idx in enumerate(selected, 1):
    row = df_result.iloc[idx]
    ax4.text(row['경도'], row['위도'], str(rank), ha='center', va='center',
            fontsize=12, fontweight='bold', color='white', zorder=4)

ax4.set_xlabel('경도 (Longitude)', fontsize=11, fontweight='bold')
ax4.set_ylabel('위도 (Latitude)', fontsize=11, fontweight='bold')
ax4.set_title('🗺️ GPS 위치 및 정류장 배치' if use_korean else 'GPS Location & Station Layout', fontsize=14, fontweight='bold', pad=15)
ax4.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
ax4.set_facecolor('#F8F9FA')

# 범례
legend_elements = [
    plt.scatter([], [], s=200, c=COLORS['accent'], edgecolors='white', linewidth=2, label='신규 정류장'),
    plt.scatter([], [], s=100, c=COLORS['success'], edgecolors='white', linewidth=2, label='커버 지역'),
    plt.scatter([], [], s=100, c=COLORS['danger'], edgecolors='white', linewidth=2, label='미커버 지역')
]
ax4.legend(handles=legend_elements, loc='upper left', framealpha=0.9, fontsize=9)

# 5. 선정된 정류장 상세 (하단 전체 폭)
ax5 = fig.add_subplot(gs[2, 0])
ax5.axis('off')

table_data = []
for rank, idx in enumerate(selected, 1):
    row = df_result.iloc[idx]
    table_data.append([
        f"#{rank}",
        row['행정구역'],
        f"{row['총_이용객']/1e6:.2f}M",
        f"{row['건물수']:,.0f}",
        f"{(row['총_이용객']/total_passengers*100):.1f}%"
    ])

table = ax5.table(cellText=table_data,
                 colLabels=['순위', '지역', '이용객', '건물수', '비중'],
                 cellLoc='center',
                 loc='center',
                 colWidths=[0.15, 0.25, 0.2, 0.2, 0.2])

table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2.5)

# 헤더 스타일
for i in range(5):
    table[(0, i)].set_facecolor(COLORS['primary'])
    table[(0, i)].set_text_props(weight='bold', color='white')

# 행 스타일 (교차 색상)
for i in range(1, len(table_data)+1):
    for j in range(5):
        if i % 2 == 0:
            table[(i, j)].set_facecolor('#F8F9FA')
        else:
            table[(i, j)].set_facecolor('white')
        table[(i, j)].set_edgecolor(COLORS['light'])

ax5.set_title('🎯 선정된 신규 정류장 상세' if use_korean else 'Selected New Station Details',
             fontsize=14, fontweight='bold', pad=20, loc='left')

# 6. 월별 추이 (중앙 상단)
ax6 = fig.add_subplot(gs[0, 2])
monthly = df_passenger.groupby('월')['총_이용객'].sum() / 1e6
line = ax6.plot(monthly.index, monthly.values, marker='o', linewidth=3, markersize=8,
               color=COLORS['secondary'], markeredgecolor='white', markeredgewidth=2, label='월별 이용객')
ax6.fill_between(monthly.index, monthly.values, alpha=0.2, color=COLORS['secondary'])
ax6.set_xlabel('월' if use_korean else 'Month', fontsize=11, fontweight='bold')
ax6.set_ylabel('이용객 (백만명)' if use_korean else 'Passengers (M)', fontsize=11, fontweight='bold')
ax6.set_title('📈 월별 이용객 추이' if use_korean else 'Monthly Trend', fontsize=14, fontweight='bold', pad=15)
ax6.set_xticks(range(1, 13))
ax6.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
ax6.spines['top'].set_visible(False)
ax6.spines['right'].set_visible(False)

# 평균선
mean_val = monthly.mean()
ax6.axhline(mean_val, color=COLORS['danger'], linestyle='--', linewidth=2, alpha=0.7, label=f'평균: {mean_val:.1f}M')
ax6.legend(loc='upper right', framealpha=0.9)

# 7. 환승 수요 Top 5 (중앙 중단)
ax7 = fig.add_subplot(gs[1, 2])
top5_transfer = df_result.nlargest(5, '환승')
ax7.bar(range(len(top5_transfer)), top5_transfer['환승'].values/1e3,
       color=COLORS['info'], edgecolor=COLORS['dark'], linewidth=1.5, alpha=0.85, width=0.7)
ax7.set_xticks(range(len(top5_transfer)))
ax7.set_xticklabels(top5_transfer['행정구역'].values if use_korean else [f'R{i+1}' for i in range(len(top5_transfer))],
                    rotation=30, ha='right', fontsize=10)
ax7.set_ylabel('환승 인원 (천명)' if use_korean else 'Transfer (K)', fontsize=11, fontweight='bold')
ax7.set_title('🔄 환승 수요 Top 5' if use_korean else 'Top 5 Transfer Demand', fontsize=14, fontweight='bold', pad=15)
ax7.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
ax7.spines['top'].set_visible(False)
ax7.spines['right'].set_visible(False)

# 값 레이블
for i, val in enumerate(top5_transfer['환승'].values):
    ax7.text(i, val/1e3 + 10, f'{val/1e3:.0f}K', ha='center', fontsize=10, fontweight='bold')

# 8. 건물 수 vs 수요 (중앙 하단)
ax8 = fig.add_subplot(gs[2, 2])
scatter = ax8.scatter(df_result['건물수'], df_result['총_이용객']/1e6,
                     s=120, c=df_result['총_이용객']/1e6, cmap='YlOrRd',
                     alpha=0.7, edgecolors=COLORS['dark'], linewidth=1.5)

# 추세선
z = np.polyfit(df_result['건물수'], df_result['총_이용객']/1e6, 1)
p = np.poly1d(z)
ax8.plot(df_result['건물수'], p(df_result['건물수']),
        color=COLORS['danger'], linestyle='--', linewidth=2, alpha=0.8, label='추세선')

ax8.set_xlabel('건물 수' if use_korean else 'Buildings', fontsize=11, fontweight='bold')
ax8.set_ylabel('이용객 (백만명)' if use_korean else 'Passengers (M)', fontsize=11, fontweight='bold')
ax8.set_title('🏢 건물 수 vs 수요' if use_korean else 'Buildings vs Demand', fontsize=14, fontweight='bold', pad=15)
ax8.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
ax8.legend(loc='upper left', framealpha=0.9)
ax8.spines['top'].set_visible(False)
ax8.spines['right'].set_visible(False)

cbar = plt.colorbar(scatter, ax=ax8)
cbar.set_label('이용객 (M)' if use_korean else 'Passengers (M)', fontsize=9)

plt.savefig('종합_시각화_최종.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✓ 저장: 종합_시각화_최종.png")
plt.close()

# ============================================================================
# 7. 결과 저장
# ============================================================================
print("\n[7단계] 결과 저장")
print("-"*100)

df_result.to_csv('최적화_결과_최종.csv', index=False, encoding='utf-8-sig')
print("✓ 저장: 최적화_결과_최종.csv")

# ============================================================================
# 완료
# ============================================================================
print("\n" + "="*100)
print("✅ 전문가급 분석 완료!".center(100))
print("="*100)
print("\n📁 생성된 파일:")
print("  1. 인터랙티브_지도_최종.html - 고급 인터랙티브 지도")
print("  2. 종합_시각화_최종.png - 전문가급 시각화")
print("  3. 최적화_결과_최종.csv - 상세 결과 데이터")
print(f"\n⏰ 분석 종료: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*100)
