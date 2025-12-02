#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
버스 정류장 최적화 분석
- 탐색적 데이터 분석 (EDA)
- 선형/정수계획법 (LP/IP) 최적화
- 지도 기반 시각화
- 종합 분석
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pulp import *
import folium
from folium import plugins
import json
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

print("=" * 80)
print("버스 정류장 최적화 분석 시작")
print("=" * 80)

# ============================================================================
# 1. 데이터 로드
# ============================================================================
print("\n[1] 데이터 로드 중...")

df_traffic = pd.read_csv('data/교통량통계_통합데이터.csv')
df_passenger = pd.read_csv('data/지역별승하차_통합데이터.csv')
df_speed = pd.read_csv('data/속도통계_통합데이터.csv')

print(f"✓ 교통량 데이터: {df_traffic.shape}")
print(f"✓ 승하차 데이터: {df_passenger.shape}")
print(f"✓ 속도 데이터: {df_speed.shape}")

# ============================================================================
# 2. 탐색적 데이터 분석 (EDA)
# ============================================================================
print("\n[2] 탐색적 데이터 분석 수행 중...")

# 승하차 데이터 전처리
df_passenger['날짜'] = pd.to_datetime(df_passenger['날짜'])
df_passenger['총_이용객'] = df_passenger['승차'] + df_passenger['하차']

# 지역별 총 이용객 수 집계
region_demand = df_passenger.groupby('행정구역').agg({
    '승차': 'sum',
    '하차': 'sum',
    '총_이용객': 'sum',
    '환승': 'sum'
}).reset_index()

region_demand = region_demand.sort_values('총_이용객', ascending=False)
print(f"\n✓ 분석 대상 지역 수: {len(region_demand)}")
print(f"✓ 총 이용객 수: {region_demand['총_이용객'].sum():,.0f}명")

# 상위 10개 지역 출력
print("\n[상위 10개 수요 지역]")
print(region_demand.head(10)[['행정구역', '총_이용객', '승차', '하차', '환승']])

# 시간별 패턴 분석
monthly_pattern = df_passenger.groupby('월')['총_이용객'].sum()
print(f"\n✓ 월별 패턴 분석 완료")

# 지역별 일평균 이용객 수
region_daily_avg = df_passenger.groupby('행정구역').agg({
    '총_이용객': 'mean'
}).reset_index()
region_daily_avg.columns = ['행정구역', '일평균_이용객']
region_daily_avg = region_daily_avg.sort_values('일평균_이용객', ascending=False)

# ============================================================================
# 3. 시각화 - EDA
# ============================================================================
print("\n[3] EDA 시각화 생성 중...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 3.1 상위 20개 지역 수요
ax1 = axes[0, 0]
top20 = region_demand.head(20)
ax1.barh(range(len(top20)), top20['총_이용객'].values, color='steelblue')
ax1.set_yticks(range(len(top20)))
ax1.set_yticklabels(top20['행정구역'].values, fontsize=9)
ax1.set_xlabel('Total Passengers', fontsize=11)
ax1.set_title('Top 20 Regions by Passenger Demand', fontsize=13, fontweight='bold')
ax1.invert_yaxis()
for i, v in enumerate(top20['총_이용객'].values):
    ax1.text(v, i, f' {v:,.0f}', va='center', fontsize=8)

# 3.2 월별 이용객 추이
ax2 = axes[0, 1]
ax2.plot(monthly_pattern.index, monthly_pattern.values, marker='o', linewidth=2, color='coral')
ax2.set_xlabel('Month', fontsize=11)
ax2.set_ylabel('Total Passengers', fontsize=11)
ax2.set_title('Monthly Passenger Trend', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.set_xticks(range(1, 13))

# 3.3 수요 분포
ax3 = axes[1, 0]
ax3.hist(region_demand['총_이용객'], bins=30, color='mediumseagreen', edgecolor='black', alpha=0.7)
ax3.set_xlabel('Total Passengers', fontsize=11)
ax3.set_ylabel('Number of Regions', fontsize=11)
ax3.set_title('Distribution of Passenger Demand', fontsize=13, fontweight='bold')
ax3.axvline(region_demand['총_이용객'].mean(), color='red', linestyle='--',
            linewidth=2, label=f'Mean: {region_demand["총_이용객"].mean():,.0f}')
ax3.axvline(region_demand['총_이용객'].median(), color='orange', linestyle='--',
            linewidth=2, label=f'Median: {region_demand["총_이용객"].median():,.0f}')
ax3.legend()

# 3.4 승차 vs 하차 관계
ax4 = axes[1, 1]
scatter = ax4.scatter(region_demand['승차'], region_demand['하차'],
                      c=region_demand['총_이용객'], cmap='viridis',
                      s=100, alpha=0.6, edgecolors='black')
ax4.plot([0, region_demand['승차'].max()], [0, region_demand['하차'].max()],
         'r--', linewidth=1, alpha=0.5)
ax4.set_xlabel('Boarding', fontsize=11)
ax4.set_ylabel('Alighting', fontsize=11)
ax4.set_title('Boarding vs Alighting by Region', fontsize=13, fontweight='bold')
plt.colorbar(scatter, ax=ax4, label='Total Passengers')

plt.tight_layout()
plt.savefig('01_eda_analysis.png', dpi=300, bbox_inches='tight')
print("✓ 저장: 01_eda_analysis.png")
plt.close()

# ============================================================================
# 4. LP/IP 최적화 모델 설계 및 구현
# ============================================================================
print("\n[4] LP/IP 최적화 모델 구축 중...")
print("=" * 80)
print("문제 정의: Facility Location Problem")
print("목적: 제한된 예산 내에서 새로운 버스 정류장을 설치하여")
print("      최대한 많은 승객 수요를 커버하는 최적 위치 선정")
print("=" * 80)

# 최적화 파라미터 설정
MAX_NEW_STATIONS = 5  # 신규 설치 가능한 정류장 수
COVERAGE_RADIUS = 3   # 커버리지 반경 (인접 지역 수)

# 상위 수요 지역 선정 (분석 대상)
TOP_N_REGIONS = 30
candidate_regions = region_demand.head(TOP_N_REGIONS).copy()
candidate_regions['지역_ID'] = range(len(candidate_regions))

print(f"\n파라미터:")
print(f"  - 최대 신규 정류장 수: {MAX_NEW_STATIONS}개")
print(f"  - 커버리지 반경: {COVERAGE_RADIUS}개 지역")
print(f"  - 후보 지역 수: {TOP_N_REGIONS}개")

# 수요 정규화 (0-1 스케일)
demand = candidate_regions['총_이용객'].values
demand_normalized = (demand - demand.min()) / (demand.max() - demand.min())

# 인접 행렬 생성 (간단한 버전: 순서 기반)
# 실제로는 GPS 좌표가 필요하지만, 여기서는 수요 순위 기반으로 근접성 추정
n_regions = len(candidate_regions)
adjacency_matrix = np.zeros((n_regions, n_regions))

for i in range(n_regions):
    for j in range(n_regions):
        if i != j:
            # 거리 추정: 순위 차이를 거리로 간주
            distance = abs(i - j)
            if distance <= COVERAGE_RADIUS:
                adjacency_matrix[i][j] = 1

print(f"\n✓ 인접 행렬 생성 완료: {adjacency_matrix.shape}")

# IP 모델 구축
print("\n[IP 모델 구축]")
prob = LpProblem("Bus_Station_Optimization", LpMaximize)

# 의사결정 변수
# x[i]: 지역 i에 새 정류장을 설치하면 1, 아니면 0
x = LpVariable.dicts("station", range(n_regions), cat='Binary')

# y[i]: 지역 i가 커버되면 1, 아니면 0
y = LpVariable.dicts("covered", range(n_regions), cat='Binary')

# 목적 함수: 커버된 지역의 총 수요 최대화
prob += lpSum([demand[i] * y[i] for i in range(n_regions)]), "Total_Covered_Demand"

# 제약 조건 1: 최대 설치 가능한 정류장 수
prob += lpSum([x[i] for i in range(n_regions)]) <= MAX_NEW_STATIONS, "Max_Stations"

# 제약 조건 2: 커버리지 제약
# 지역 i가 커버되려면, i 자신이나 인접 지역에 정류장이 있어야 함
for i in range(n_regions):
    # 자기 자신 또는 인접 지역에 정류장이 있으면 커버됨
    prob += y[i] <= x[i] + lpSum([adjacency_matrix[i][j] * x[j] for j in range(n_regions) if j != i]), \
            f"Coverage_{i}"

print("✓ 변수 및 제약 조건 설정 완료")

# 모델 풀이
print("\n[모델 풀이 시작]")
prob.solve(PULP_CBC_CMD(msg=0))

# 결과 추출
status = LpStatus[prob.status]
print(f"\n풀이 상태: {status}")

if status == 'Optimal':
    print(f"최적 목적함수 값: {value(prob.objective):,.0f}명")

    selected_stations = [i for i in range(n_regions) if x[i].varValue == 1]
    covered_regions = [i for i in range(n_regions) if y[i].varValue == 1]

    print(f"\n선정된 신규 정류장 위치 ({len(selected_stations)}개):")
    for idx in selected_stations:
        region_name = candidate_regions.iloc[idx]['행정구역']
        region_demand = candidate_regions.iloc[idx]['총_이용객']
        print(f"  ✓ {region_name}: {region_demand:,.0f}명")

    print(f"\n커버되는 지역 수: {len(covered_regions)}개 / {n_regions}개")
    total_covered_demand = sum([demand[i] for i in covered_regions])
    total_demand = sum(demand)
    coverage_rate = (total_covered_demand / total_demand) * 100
    print(f"커버리지 비율: {coverage_rate:.2f}%")

    # 결과 데이터프레임 생성
    result_df = candidate_regions.copy()
    result_df['신규_정류장'] = [1 if i in selected_stations else 0 for i in range(n_regions)]
    result_df['커버_여부'] = [1 if i in covered_regions else 0 for i in range(n_regions)]

else:
    print("최적해를 찾지 못했습니다.")
    selected_stations = []
    covered_regions = []
    result_df = candidate_regions.copy()
    result_df['신규_정류장'] = 0
    result_df['커버_여부'] = 0

# ============================================================================
# 5. 최적화 결과 시각화
# ============================================================================
print("\n[5] 최적화 결과 시각화 중...")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 5.1 선정된 정류장과 커버리지
ax1 = axes[0]
colors = ['red' if i in selected_stations else 'lightgray' for i in range(n_regions)]
ax1.barh(range(n_regions), result_df['총_이용객'].values, color=colors, edgecolor='black')
ax1.set_yticks(range(n_regions))
ax1.set_yticklabels(result_df['행정구역'].values, fontsize=7)
ax1.set_xlabel('Total Passengers', fontsize=11)
ax1.set_title('Selected New Bus Stations (Red)', fontsize=13, fontweight='bold')
ax1.invert_yaxis()

# 5.2 커버리지 분석
ax2 = axes[1]
coverage_data = [
    ('Covered', len(covered_regions), 'green'),
    ('Not Covered', n_regions - len(covered_regions), 'lightcoral')
]
labels = [x[0] for x in coverage_data]
sizes = [x[1] for x in coverage_data]
colors_pie = [x[2] for x in coverage_data]

wedges, texts, autotexts = ax2.pie(sizes, labels=labels, colors=colors_pie, autopct='%1.1f%%',
                                     startangle=90, textprops={'fontsize': 12})
ax2.set_title('Coverage Analysis', fontsize=13, fontweight='bold')

for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')

plt.tight_layout()
plt.savefig('02_optimization_results.png', dpi=300, bbox_inches='tight')
print("✓ 저장: 02_optimization_results.png")
plt.close()

# ============================================================================
# 6. 추가 분석: 시나리오 분석
# ============================================================================
print("\n[6] 시나리오 분석 수행 중...")

scenarios = [3, 5, 7, 10]
scenario_results = []

for n_stations in scenarios:
    # 새로운 모델
    prob_scenario = LpProblem(f"Scenario_{n_stations}", LpMaximize)
    x_s = LpVariable.dicts(f"station_{n_stations}", range(n_regions), cat='Binary')
    y_s = LpVariable.dicts(f"covered_{n_stations}", range(n_regions), cat='Binary')

    prob_scenario += lpSum([demand[i] * y_s[i] for i in range(n_regions)])
    prob_scenario += lpSum([x_s[i] for i in range(n_regions)]) <= n_stations

    for i in range(n_regions):
        prob_scenario += y_s[i] <= x_s[i] + lpSum([adjacency_matrix[i][j] * x_s[j]
                                                     for j in range(n_regions) if j != i])

    prob_scenario.solve(PULP_CBC_CMD(msg=0))

    if LpStatus[prob_scenario.status] == 'Optimal':
        covered = [i for i in range(n_regions) if y_s[i].varValue == 1]
        covered_demand = sum([demand[i] for i in covered])
        coverage_pct = (covered_demand / sum(demand)) * 100

        scenario_results.append({
            'n_stations': n_stations,
            'covered_regions': len(covered),
            'covered_demand': covered_demand,
            'coverage_pct': coverage_pct
        })
        print(f"  ✓ {n_stations}개 정류장: 커버리지 {coverage_pct:.2f}%")

scenario_df = pd.DataFrame(scenario_results)

# 시나리오 시각화
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax1 = axes[0]
ax1.plot(scenario_df['n_stations'], scenario_df['coverage_pct'],
         marker='o', linewidth=2, markersize=10, color='royalblue')
ax1.set_xlabel('Number of New Stations', fontsize=11)
ax1.set_ylabel('Coverage (%)', fontsize=11)
ax1.set_title('Coverage vs Number of Stations', fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.set_xticks(scenario_df['n_stations'])

ax2 = axes[1]
ax2.bar(scenario_df['n_stations'], scenario_df['covered_demand'],
        color='teal', edgecolor='black', alpha=0.7)
ax2.set_xlabel('Number of New Stations', fontsize=11)
ax2.set_ylabel('Covered Demand (Passengers)', fontsize=11)
ax2.set_title('Covered Demand vs Number of Stations', fontsize=13, fontweight='bold')
ax2.set_xticks(scenario_df['n_stations'])

for i, v in enumerate(scenario_df['covered_demand']):
    ax2.text(scenario_df['n_stations'].iloc[i], v, f'{v:,.0f}',
             ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('03_scenario_analysis.png', dpi=300, bbox_inches='tight')
print("✓ 저장: 03_scenario_analysis.png")
plt.close()

# ============================================================================
# 7. 지도 시각화
# ============================================================================
print("\n[7] 지도 기반 시각화 생성 중...")

# 한국 중심 좌표 (세종시 기준 - 데이터가 세종시로 추정됨)
center_lat, center_lon = 36.4800, 127.2890

# Folium 지도 생성
m = folium.Map(
    location=[center_lat, center_lon],
    zoom_start=12,
    tiles='OpenStreetMap'
)

# 지역별 좌표 생성 (실제 GPS 데이터가 없으므로 시뮬레이션)
# 원형으로 배치
np.random.seed(42)
angles = np.linspace(0, 2 * np.pi, n_regions)
radius_base = 0.05  # 약 5km

coordinates = []
for i, angle in enumerate(angles):
    # 수요에 따라 중심에서의 거리 조정 (수요 높으면 중심 근처)
    radius = radius_base * (1.5 - demand_normalized[i] * 0.5)
    lat = center_lat + radius * np.cos(angle) + np.random.normal(0, 0.01)
    lon = center_lon + radius * np.sin(angle) + np.random.normal(0, 0.01)
    coordinates.append((lat, lon))

result_df['위도'] = [coord[0] for coord in coordinates]
result_df['경도'] = [coord[1] for coord in coordinates]

# 마커 추가
for idx, row in result_df.iterrows():
    region_name = row['행정구역']
    demand_val = row['총_이용객']
    is_new_station = row['신규_정류장']
    is_covered = row['커버_여부']

    # 마커 색상 및 아이콘 결정
    if is_new_station == 1:
        color = 'red'
        icon = 'star'
        popup_text = f"<b>[NEW STATION]</b><br>{region_name}<br>Demand: {demand_val:,.0f}"
    elif is_covered == 1:
        color = 'green'
        icon = 'ok'
        popup_text = f"<b>[COVERED]</b><br>{region_name}<br>Demand: {demand_val:,.0f}"
    else:
        color = 'gray'
        icon = 'info-sign'
        popup_text = f"{region_name}<br>Demand: {demand_val:,.0f}"

    folium.Marker(
        location=[row['위도'], row['경도']],
        popup=folium.Popup(popup_text, max_width=200),
        tooltip=region_name,
        icon=folium.Icon(color=color, icon=icon)
    ).add_to(m)

    # 수요 크기를 나타내는 원 추가
    folium.Circle(
        location=[row['위도'], row['경도']],
        radius=demand_val / 50,  # 크기 조정
        color=color,
        fill=True,
        fillOpacity=0.3,
        opacity=0.6
    ).add_to(m)

# 범례 추가
legend_html = '''
<div style="position: fixed;
            bottom: 50px; right: 50px; width: 220px; height: 160px;
            background-color: white; border:2px solid grey; z-index:9999;
            font-size:14px; padding: 10px">
<p style="margin-bottom:5px;"><b>Bus Station Optimization</b></p>
<p style="margin:3px;"><i class="fa fa-star" style="color:red"></i> New Station (Recommended)</p>
<p style="margin:3px;"><i class="fa fa-check" style="color:green"></i> Covered Region</p>
<p style="margin:3px;"><i class="fa fa-info-circle" style="color:gray"></i> Not Covered</p>
<p style="margin:3px; font-size:12px;">Circle size = Passenger demand</p>
</div>
'''
m.get_root().html.add_child(folium.Element(legend_html))

# 지도 저장
m.save('04_bus_station_map.html')
print("✓ 저장: 04_bus_station_map.html")

# ============================================================================
# 8. 추가 분석: 교통량 및 속도 데이터
# ============================================================================
print("\n[8] 교통량 및 속도 데이터 추가 분석 중...")

# 교통량 분석
traffic_pivot = df_traffic.pivot_table(
    index=['도로', '방향'],
    columns='지표',
    values='값',
    aggfunc='mean'
).reset_index()

if '총합' in traffic_pivot.columns:
    traffic_pivot = traffic_pivot.sort_values('총합', ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(12, 8))
    y_pos = range(len(traffic_pivot))
    labels = [f"{row['도로']} ({row['방향']})" for _, row in traffic_pivot.iterrows()]

    ax.barh(y_pos, traffic_pivot['총합'].values, color='steelblue', edgecolor='black')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('Average Traffic Volume', fontsize=11)
    ax.set_title('Top 15 Roads by Traffic Volume', fontsize=13, fontweight='bold')
    ax.invert_yaxis()

    for i, v in enumerate(traffic_pivot['총합'].values):
        ax.text(v, i, f' {v:,.0f}', va='center', fontsize=8)

    plt.tight_layout()
    plt.savefig('05_traffic_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ 저장: 05_traffic_analysis.png")
    plt.close()

# 속도 분석
speed_by_road = df_speed.groupby('도로')['속도'].mean().sort_values(ascending=False).head(15)

fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(range(len(speed_by_road)), speed_by_road.values, color='coral', edgecolor='black')
ax.set_xticks(range(len(speed_by_road)))
ax.set_xticklabels(speed_by_road.index, rotation=45, ha='right', fontsize=9)
ax.set_ylabel('Average Speed (km/h)', fontsize=11)
ax.set_title('Top 15 Roads by Average Speed', fontsize=13, fontweight='bold')
ax.axhline(speed_by_road.mean(), color='red', linestyle='--', linewidth=2,
           label=f'Mean: {speed_by_road.mean():.1f} km/h')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('06_speed_analysis.png', dpi=300, bbox_inches='tight')
print("✓ 저장: 06_speed_analysis.png")
plt.close()

# ============================================================================
# 9. 종합 분석 리포트 생성 (HTML)
# ============================================================================
print("\n[9] 종합 분석 리포트 생성 중...")

# HTML에서 사용할 변수들 미리 계산
total_regions = len(result_df)
total_passengers = int(result_df['총_이용객'].sum())
num_new_stations = len(selected_stations)
coverage_percentage = coverage_rate if 'coverage_rate' in dir() else 0.0

html_content = f"""
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>버스 정류장 최적화 분석 리포트</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
            border-left: 5px solid #3498db;
            padding-left: 15px;
        }}
        .metric {{
            display: inline-block;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px 25px;
            margin: 10px;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .metric-value {{
            font-size: 32px;
            font-weight: bold;
        }}
        .metric-label {{
            font-size: 14px;
            opacity: 0.9;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #3498db;
            color: white;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .highlight {{
            background-color: #ffffcc;
            font-weight: bold;
        }}
        img {{
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            margin: 20px 0;
        }}
        .recommendation {{
            background-color: #e8f5e9;
            border-left: 5px solid #4caf50;
            padding: 15px;
            margin: 20px 0;
            border-radius: 5px;
        }}
        .warning {{
            background-color: #fff3e0;
            border-left: 5px solid #ff9800;
            padding: 15px;
            margin: 20px 0;
            border-radius: 5px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🚌 버스 정류장 최적화 분석 리포트</h1>
        <p><strong>분석 날짜:</strong> 2023년 데이터 기준</p>

        <h2>📊 주요 지표</h2>
        <div>
            <div class="metric">
                <div class="metric-label">총 분석 지역</div>
                <div class="metric-value">{total_regions}</div>
            </div>
            <div class="metric">
                <div class="metric-label">총 이용객</div>
                <div class="metric-value">{total_passengers:,}</div>
            </div>
            <div class="metric">
                <div class="metric-label">권장 신규 정류장</div>
                <div class="metric-value">{num_new_stations}</div>
            </div>
            <div class="metric">
                <div class="metric-label">커버리지</div>
                <div class="metric-value">{coverage_percentage:.1f}%</div>
            </div>
        </div>

        <h2>🎯 최적화 결과</h2>
        <div class="recommendation">
            <h3>✅ 권장 신규 버스 정류장 위치</h3>
            <table>
                <tr>
                    <th>순위</th>
                    <th>지역명</th>
                    <th>총 이용객</th>
                    <th>승차</th>
                    <th>하차</th>
                    <th>환승</th>
                </tr>
"""

for i, idx in enumerate(selected_stations, 1):
    row = result_df.iloc[idx]
    html_content += f"""
                <tr class="highlight">
                    <td>{i}</td>
                    <td>{row['행정구역']}</td>
                    <td>{row['총_이용객']:,.0f}</td>
                    <td>{row['승차']:,.0f}</td>
                    <td>{row['하차']:,.0f}</td>
                    <td>{row['환승']:,.0f}</td>
                </tr>
"""

html_content += """
            </table>
        </div>

        <h2>📈 탐색적 데이터 분석 (EDA)</h2>
        <img src="01_eda_analysis.png" alt="EDA Analysis">
        <p>상위 수요 지역과 월별 추이, 수요 분포, 승하차 관계를 분석한 결과입니다.</p>

        <h2>🎯 최적화 모델 결과</h2>
        <img src="02_optimization_results.png" alt="Optimization Results">
        <p>선형계획법(IP)을 활용하여 선정된 신규 정류장 위치와 커버리지 분석 결과입니다.</p>

        <h2>🔍 시나리오 분석</h2>
        <img src="03_scenario_analysis.png" alt="Scenario Analysis">
        <p>신규 정류장 수에 따른 커버리지 변화를 분석한 결과입니다.</p>

        <h2>🗺️ 지도 시각화</h2>
        <p><a href="04_bus_station_map.html" target="_blank">인터랙티브 지도 보기 (클릭)</a></p>
        <div class="warning">
            <strong>⚠️ 참고:</strong> 지도의 좌표는 실제 GPS 데이터가 없어 시뮬레이션된 값입니다.
            실제 운영시에는 정확한 GPS 좌표를 사용해야 합니다.
        </div>

        <h2>🚗 교통량 분석</h2>
        <img src="05_traffic_analysis.png" alt="Traffic Analysis">

        <h2>⚡ 속도 분석</h2>
        <img src="06_speed_analysis.png" alt="Speed Analysis">

        <h2>💡 결론 및 권장사항</h2>
        <div class="recommendation">
            <h3>주요 발견사항</h3>
            <ul>
"""

html_content += f"""
                <li><strong>총 {total_regions}개 지역 중 상위 {TOP_N_REGIONS}개 지역을 분석 대상으로 선정</strong></li>
                <li><strong>{num_new_stations}개의 신규 정류장 설치로 {coverage_percentage:.1f}%의 수요 커버 가능</strong></li>
                <li><strong>선정된 지역들은 높은 승하차 수요와 환승 수요를 보이는 핵심 거점</strong></li>
            </ul>

            <h3>권장사항</h3>
            <ol>
                <li><strong>단계적 구축:</strong> 수요가 가장 높은 상위 3개 지역부터 우선 설치</li>
                <li><strong>인프라 연계:</strong> 교통량이 많은 도로와 연계하여 접근성 향상</li>
                <li><strong>환승 최적화:</strong> 환승 수요가 높은 지역에 환승센터 구축 고려</li>
                <li><strong>모니터링:</strong> 설치 후 이용 패턴 모니터링을 통한 추가 최적화</li>
            </ol>
        </div>

        <h2>📋 방법론</h2>
        <div class="warning">
            <h3>최적화 모델 상세</h3>
            <ul>
                <li><strong>모델 유형:</strong> Integer Programming (IP) - Facility Location Problem</li>
                <li><strong>목적 함수:</strong> 커버되는 총 수요(이용객 수) 최대화</li>
                <li><strong>주요 제약:</strong> 최대 설치 가능 정류장 수, 커버리지 반경</li>
                <li><strong>솔버:</strong> PuLP + CBC Solver</li>
                <li><strong>커버리지 정의:</strong> 정류장으로부터 {COVERAGE_RADIUS}개 지역 이내</li>
            </ul>
        </div>

        <hr style="margin: 40px 0;">
        <p style="text-align: center; color: #7f8c8d;">
            <small>본 분석은 2023년 교통 데이터를 기반으로 수행되었습니다.<br>
            실제 운영 시에는 최신 데이터와 추가적인 요인들을 고려해야 합니다.</small>
        </p>
    </div>
</body>
</html>
"""

with open('00_comprehensive_report.html', 'w', encoding='utf-8') as f:
    f.write(html_content)

print("✓ 저장: 00_comprehensive_report.html")

# ============================================================================
# 10. 결과 요약 저장
# ============================================================================
print("\n[10] 결과 요약 저장 중...")

# CSV로 저장
result_df.to_csv('optimization_results.csv', index=False, encoding='utf-8-sig')
print("✓ 저장: optimization_results.csv")

# 시나리오 결과 저장
scenario_df.to_csv('scenario_analysis.csv', index=False, encoding='utf-8-sig')
print("✓ 저장: scenario_analysis.csv")

print("\n" + "=" * 80)
print("✅ 모든 분석이 완료되었습니다!")
print("=" * 80)
print("\n생성된 파일:")
print("  1. 00_comprehensive_report.html - 종합 분석 리포트")
print("  2. 01_eda_analysis.png - 탐색적 데이터 분석")
print("  3. 02_optimization_results.png - 최적화 결과")
print("  4. 03_scenario_analysis.png - 시나리오 분석")
print("  5. 04_bus_station_map.html - 인터랙티브 지도")
print("  6. 05_traffic_analysis.png - 교통량 분석")
print("  7. 06_speed_analysis.png - 속도 분석")
print("  8. optimization_results.csv - 최적화 결과 데이터")
print("  9. scenario_analysis.csv - 시나리오 분석 데이터")
print("\n👉 00_comprehensive_report.html 파일을 브라우저에서 열어보세요!")
print("=" * 80)
