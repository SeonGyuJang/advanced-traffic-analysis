#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
세종시 버스 정류장 최적 입지 선정 - GPS 기반 정밀 분석
=====================================================
작성일: 2023
목적: 실제 GPS 좌표를 활용한 정밀한 버스 정류장 최적 위치 선정

주요 기능:
- 실제 GPS 거리 기반 분석 (Haversine 공식)
- 고급 정수계획법 (Integer Programming) 최적화
- 다목적 최적화 (수요 + 건물밀도 + 환승)
- 시나리오 분석
- 한국어 폰트 적용 시각화
- 상세 HTML 리포트 생성
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from pulp import *
import folium
from math import radians, cos, sin, asin, sqrt
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

# ============================================================================
# 한글 폰트 설정
# ============================================================================
def setup_korean_font():
    """한글 폰트 자동 설정"""
    korean_fonts = ['NanumGothic', 'NanumBarunGothic', 'Malgun Gothic']
    available_fonts = [f.name for f in fm.fontManager.ttflist]

    for font in korean_fonts:
        if font in available_fonts:
            plt.rcParams['font.family'] = font
            plt.rcParams['axes.unicode_minus'] = False
            print(f"✓ 한글 폰트 설정: {font}")
            return True

    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['axes.unicode_minus'] = False
    print("⚠ 한글 폰트 없음. 영문으로 표시됩니다.")
    return False

# ============================================================================
# GPS 거리 계산
# ============================================================================
def haversine_distance(lat1, lon1, lat2, lon2):
    """두 GPS 좌표 간 실제 거리 계산 (km)"""
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    return 6371 * c  # 지구 반지름

def create_distance_matrix(df):
    """거리 행렬 생성"""
    n = len(df)
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                matrix[i, j] = haversine_distance(
                    df.iloc[i]['위도'], df.iloc[i]['경도'],
                    df.iloc[j]['위도'], df.iloc[j]['경도']
                )
    return matrix

# ============================================================================
# 메인 실행
# ============================================================================
print("="*100)
print("세종시 버스 정류장 최적 입지 선정 - GPS 기반 정밀 분석".center(100))
print("="*100)
print(f"분석 시작: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

use_korean = setup_korean_font()

# ============================================================================
# 1. 데이터 로드
# ============================================================================
print("\n[1단계] 데이터 로드")
print("-"*100)

df_passenger = pd.read_csv('data/지역별승하차_통합데이터.csv')
df_coords = pd.read_csv('data/행정구역_중심좌표.csv')
df_speed = pd.read_csv('data/속도통계_통합데이터.csv')
df_traffic = pd.read_csv('data/교통량통계_통합데이터.csv')

print(f"✓ 승하차 데이터: {df_passenger.shape[0]:,}행 x {df_passenger.shape[1]}열")
print(f"✓ 행정구역 GPS 좌표: {df_coords.shape[0]}개 지역")
print(f"✓ 속도 통계: {df_speed.shape[0]:,}행")
print(f"✓ 교통량 통계: {df_traffic.shape[0]:,}행")

# ============================================================================
# 2. 데이터 전처리
# ============================================================================
print("\n[2단계] 데이터 전처리 및 분석")
print("-"*100)

df_passenger['날짜'] = pd.to_datetime(df_passenger['날짜'])
df_passenger['총_이용객'] = df_passenger['승차'] + df_passenger['하차']

# 지역별 집계
region_stats = df_passenger.groupby('행정구역').agg({
    '승차': 'sum',
    '하차': 'sum',
    '환승': 'sum',
    '총_이용객': 'sum'
}).reset_index().sort_values('총_이용객', ascending=False)

# GPS 좌표와 병합
df_analysis = pd.merge(region_stats, df_coords, on='행정구역')

print(f"\n✓ 분석 대상 지역: {len(df_analysis)}개")
print(f"✓ 총 이용객: {df_analysis['총_이용객'].sum():,.0f}명")
print(f"\n상위 5개 수요 지역:")
for idx, row in df_analysis.head(5).iterrows():
    print(f"  • {row['행정구역']:10s}: {row['총_이용객']:>12,.0f}명")

# ============================================================================
# 3. GPS 거리 행렬 계산
# ============================================================================
print("\n[3단계] GPS 거리 행렬 계산")
print("-"*100)

distance_matrix = create_distance_matrix(df_analysis)
print(f"✓ 거리 행렬 크기: {distance_matrix.shape}")
print(f"✓ 최단 거리: {distance_matrix[distance_matrix > 0].min():.2f} km")
print(f"✓ 최장 거리: {distance_matrix.max():.2f} km")
print(f"✓ 평균 거리: {distance_matrix[distance_matrix > 0].mean():.2f} km")

# 저장
pd.DataFrame(distance_matrix,
             index=df_analysis['행정구역'],
             columns=df_analysis['행정구역']).to_csv('GPS_거리행렬.csv', encoding='utf-8-sig')

# ============================================================================
# 4. 탐색적 데이터 분석 시각화
# ============================================================================
print("\n[4단계] 탐색적 데이터 분석 시각화")
print("-"*100)

fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle('세종시 대중교통 수요 분석' if use_korean else 'Sejong City Transportation Analysis',
             fontsize=18, fontweight='bold', y=0.995)

# 4.1 지역별 수요 (Top 15)
ax1 = axes[0, 0]
top15 = df_analysis.head(15)
colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top15)))
ax1.barh(range(len(top15)), top15['총_이용객'].values, color=colors, edgecolor='black')
ax1.set_yticks(range(len(top15)))
ax1.set_yticklabels(top15['행정구역'].values if use_korean else [f'R{i+1}' for i in range(len(top15))])
ax1.set_xlabel('총 이용객 수 (명)' if use_korean else 'Total Passengers')
ax1.set_title('행정구역별 총 이용객 (Top 15)' if use_korean else 'Passengers by Region (Top 15)', fontweight='bold')
ax1.invert_yaxis()
ax1.grid(axis='x', alpha=0.3)

# 4.2 월별 추이
ax2 = axes[0, 1]
monthly = df_passenger.groupby('월')['총_이용객'].sum() / 1e6
ax2.plot(monthly.index, monthly.values, marker='o', linewidth=2.5, markersize=8, color='#2E86AB')
ax2.fill_between(monthly.index, monthly.values, alpha=0.3)
ax2.set_xlabel('월' if use_korean else 'Month')
ax2.set_ylabel('이용객 (백만명)' if use_korean else 'Passengers (Million)')
ax2.set_title('월별 이용객 추이' if use_korean else 'Monthly Trend', fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.set_xticks(range(1, 13))

# 4.3 승차 vs 하차
ax3 = axes[0, 2]
scatter = ax3.scatter(df_analysis['승차'], df_analysis['하차'],
                     s=df_analysis['건물수']/20, c=df_analysis['총_이용객'],
                     cmap='YlOrRd', alpha=0.7, edgecolors='black')
ax3.plot([0, df_analysis['승차'].max()], [0, df_analysis['하차'].max()], 'k--', alpha=0.5)
ax3.set_xlabel('승차' if use_korean else 'Boarding')
ax3.set_ylabel('하차' if use_korean else 'Alighting')
ax3.set_title('승차 vs 하차 관계' if use_korean else 'Boarding vs Alighting', fontweight='bold')
plt.colorbar(scatter, ax=ax3, label='총 이용객' if use_korean else 'Total')

# 4.4 수요 분포
ax4 = axes[1, 0]
ax4.hist(df_analysis['총_이용객'], bins=12, color='#06D6A0', edgecolor='black', alpha=0.7)
ax4.axvline(df_analysis['총_이용객'].mean(), color='red', linestyle='--', linewidth=2,
           label=f"평균: {df_analysis['총_이용객'].mean():,.0f}" if use_korean else f"Mean: {df_analysis['총_이용객'].mean():,.0f}")
ax4.set_xlabel('총 이용객' if use_korean else 'Total Passengers')
ax4.set_ylabel('지역 수' if use_korean else 'Number of Regions')
ax4.set_title('수요 분포' if use_korean else 'Demand Distribution', fontweight='bold')
ax4.legend()
ax4.grid(axis='y', alpha=0.3)

# 4.5 GPS 지도 (위경도)
ax5 = axes[1, 1]
scatter = ax5.scatter(df_analysis['경도'], df_analysis['위도'],
                     s=df_analysis['총_이용객']/5000, c=df_analysis['총_이용객'],
                     cmap='RdYlGn_r', alpha=0.7, edgecolors='black', linewidth=1.5)
ax5.set_xlabel('경도 (Longitude)')
ax5.set_ylabel('위도 (Latitude)')
ax5.set_title('GPS 위치 및 수요' if use_korean else 'GPS Location & Demand', fontweight='bold')
ax5.grid(True, alpha=0.3)
plt.colorbar(scatter, ax=ax5, label='이용객' if use_korean else 'Passengers')

# 4.6 건물수 vs 수요
ax6 = axes[1, 2]
ax6.scatter(df_analysis['건물수'], df_analysis['총_이용객'], s=100, alpha=0.6,
           c=df_analysis['총_이용객'], cmap='plasma', edgecolors='black')
z = np.polyfit(df_analysis['건물수'], df_analysis['총_이용객'], 1)
p = np.poly1d(z)
ax6.plot(df_analysis['건물수'], p(df_analysis['건물수']), "r--", linewidth=2)
ax6.set_xlabel('건물 수' if use_korean else 'Buildings')
ax6.set_ylabel('총 이용객' if use_korean else 'Passengers')
ax6.set_title('건물 수 vs 수요' if use_korean else 'Buildings vs Demand', fontweight='bold')
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('01_탐색적_분석.png', dpi=300, bbox_inches='tight')
print("✓ 저장: 01_탐색적_분석.png")
plt.close()

# ============================================================================
# 5. 정수계획법 최적화
# ============================================================================
print("\n[5단계] 정수계획법 최적화 모델")
print("-"*100)

# 파라미터
MAX_STATIONS = 5
COVERAGE_RADIUS_KM = 5.0

print(f"⚙️  최적화 파라미터:")
print(f"  • 신규 정류장: {MAX_STATIONS}개")
print(f"  • 커버리지 반경: {COVERAGE_RADIUS_KM} km")

# 정규화된 가중치
demand = df_analysis['총_이용객'].values
demand_norm = (demand - demand.min()) / (demand.max() - demand.min())

building = df_analysis['건물수'].values
building_norm = (building - building.min()) / (building.max() - building.min() + 1)

transfer = df_analysis['환승'].values
transfer_norm = (transfer - transfer.min()) / (transfer.max() - transfer.min() + 1)

# 복합 가중치 (수요 60%, 건물 25%, 환승 15%)
weight = 0.60 * demand_norm + 0.25 * building_norm + 0.15 * transfer_norm

# 커버리지 행렬
n = len(df_analysis)
coverage = (distance_matrix <= COVERAGE_RADIUS_KM).astype(int)
np.fill_diagonal(coverage, 1)

# IP 모델
print("\n🔧 최적화 모델 구축 중...")
prob = LpProblem("Bus_Station_Optimization", LpMaximize)

# 변수
x = LpVariable.dicts("station", range(n), cat='Binary')
y = LpVariable.dicts("covered", range(n), cat='Binary')

# 목적함수: 가중 수요 최대화
prob += lpSum([demand[i] * weight[i] * y[i] for i in range(n)])

# 제약조건
prob += lpSum([x[i] for i in range(n)]) <= MAX_STATIONS
for i in range(n):
    prob += y[i] <= lpSum([coverage[i][j] * x[j] for j in range(n)])

# 풀이
print("🚀 최적화 실행 중...")
prob.solve(PULP_CBC_CMD(msg=0))

status = LpStatus[prob.status]
print(f"\n✅ 최적화 완료: {status}")

if status == 'Optimal':
    selected = [i for i in range(n) if x[i].varValue == 1]
    covered = [i for i in range(n) if y[i].varValue == 1]

    covered_demand = sum([demand[i] for i in covered])
    total_demand = demand.sum()
    coverage_rate = (covered_demand / total_demand) * 100

    print(f"\n📊 최적화 결과:")
    print(f"  • 선정 정류장: {len(selected)}개")
    print(f"  • 커버 지역: {len(covered)}/{n}개")
    print(f"  • 커버 수요: {covered_demand:,.0f}/{total_demand:,.0f}명")
    print(f"  • 커버리지: {coverage_rate:.2f}%")

    print(f"\n🎯 선정된 정류장:")
    for rank, idx in enumerate(selected, 1):
        row = df_analysis.iloc[idx]
        print(f"  {rank}. {row['행정구역']:10s} | 이용객: {row['총_이용객']:>10,.0f}명 | "
              f"건물: {row['건물수']:>5,.0f}개 | GPS: ({row['위도']:.4f}, {row['경도']:.4f})")

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
    df_result['최단정류장거리_km'] = min_dist

    df_result.to_csv('최적화_결과.csv', index=False, encoding='utf-8-sig')
    print(f"\n✓ 저장: 최적화_결과.csv")

else:
    print(f"❌ 최적화 실패: {status}")
    selected = []
    covered = []
    df_result = df_analysis.copy()

# ============================================================================
# 6. 시나리오 분석
# ============================================================================
print("\n[6단계] 시나리오 분석")
print("-"*100)

scenarios = []
for k in [3, 5, 7, 10]:
    prob_s = LpProblem(f"Scenario_{k}", LpMaximize)
    x_s = LpVariable.dicts(f"st_{k}", range(n), cat='Binary')
    y_s = LpVariable.dicts(f"cv_{k}", range(n), cat='Binary')

    prob_s += lpSum([demand[i] * weight[i] * y_s[i] for i in range(n)])
    prob_s += lpSum([x_s[i] for i in range(n)]) <= k
    for i in range(n):
        prob_s += y_s[i] <= lpSum([coverage[i][j] * x_s[j] for j in range(n)])

    prob_s.solve(PULP_CBC_CMD(msg=0))

    if LpStatus[prob_s.status] == 'Optimal':
        cov = [i for i in range(n) if y_s[i].varValue == 1]
        cov_demand = sum([demand[i] for i in cov])
        cov_pct = (cov_demand / demand.sum()) * 100
        scenarios.append({'정류장수': k, '커버지역': len(cov), '커버수요': cov_demand, '커버율': cov_pct})
        print(f"  ✓ {k}개 정류장: {cov_pct:.2f}% 커버")

df_scenarios = pd.DataFrame(scenarios)
df_scenarios.to_csv('시나리오_분석.csv', index=False, encoding='utf-8-sig')

# 시각화
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax1 = axes[0]
ax1.plot(df_scenarios['정류장수'], df_scenarios['커버율'], marker='o', linewidth=3, markersize=10, color='#FF6B6B')
ax1.set_xlabel('정류장 수' if use_korean else 'Stations')
ax1.set_ylabel('커버율 (%)' if use_korean else 'Coverage (%)')
ax1.set_title('정류장 수별 커버율' if use_korean else 'Coverage by Stations', fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.set_xticks(df_scenarios['정류장수'])

ax2 = axes[1]
ax2.bar(df_scenarios['정류장수'], df_scenarios['커버수요']/1e6, color='#06D6A0', edgecolor='black', width=1.0)
ax2.set_xlabel('정류장 수' if use_korean else 'Stations')
ax2.set_ylabel('커버 수요 (백만명)' if use_korean else 'Covered Demand (M)')
ax2.set_title('정류장 수별 커버 수요' if use_korean else 'Demand by Stations', fontweight='bold')
ax2.set_xticks(df_scenarios['정류장수'])
for i, row in df_scenarios.iterrows():
    ax2.text(row['정류장수'], row['커버수요']/1e6, f"{row['커버수요']/1e6:.2f}M",
             ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('02_시나리오_분석.png', dpi=300, bbox_inches='tight')
print("\n✓ 저장: 02_시나리오_분석.png")
plt.close()

# ============================================================================
# 7. 인터랙티브 지도
# ============================================================================
print("\n[7단계] 인터랙티브 지도 생성")
print("-"*100)

center_lat = df_analysis['위도'].mean()
center_lon = df_analysis['경도'].mean()

m = folium.Map(location=[center_lat, center_lon], zoom_start=11, tiles='OpenStreetMap')

# 마커 추가
for idx, row in df_result.iterrows():
    if row['신규정류장'] == 1:
        color = 'red'
        icon = 'star'
        popup = f"<b>🆕 신규 정류장</b><br>{row['행정구역']}<br>이용객: {row['총_이용객']:,.0f}명"
    elif row['커버여부'] == 1:
        color = 'green'
        icon = 'ok'
        popup = f"<b>✓ 커버됨</b><br>{row['행정구역']}<br>이용객: {row['총_이용객']:,.0f}명"
    else:
        color = 'gray'
        icon = 'info-sign'
        popup = f"{row['행정구역']}<br>이용객: {row['총_이용객']:,.0f}명"

    folium.Marker(
        location=[row['위도'], row['경도']],
        popup=folium.Popup(popup, max_width=250),
        tooltip=row['행정구역'],
        icon=folium.Icon(color=color, icon=icon)
    ).add_to(m)

    # 수요 원
    folium.Circle(
        location=[row['위도'], row['경도']],
        radius=row['총_이용객']/50,
        color=color,
        fill=True,
        fillOpacity=0.3
    ).add_to(m)

m.save('03_인터랙티브_지도.html')
print("✓ 저장: 03_인터랙티브_지도.html")

# ============================================================================
# 8. 종합 리포트
# ============================================================================
print("\n[8단계] 종합 리포트 생성")
print("-"*100)

html = f"""
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <title>세종시 버스 정류장 최적화 분석 리포트</title>
    <style>
        body {{font-family: 'Malgun Gothic', sans-serif; margin: 20px; background: #f5f5f5;}}
        .container {{max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px;}}
        h1 {{color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px;}}
        h2 {{color: #34495e; margin-top: 30px; border-left: 5px solid #3498db; padding-left: 15px;}}
        .metric {{display: inline-block; background: linear-gradient(135deg, #667eea, #764ba2);
                 color: white; padding: 20px; margin: 10px; border-radius: 8px; min-width: 150px;}}
        .metric-value {{font-size: 36px; font-weight: bold;}}
        .metric-label {{font-size: 14px; opacity: 0.9;}}
        table {{width: 100%; border-collapse: collapse; margin: 20px 0;}}
        th, td {{padding: 12px; text-align: left; border-bottom: 1px solid #ddd;}}
        th {{background: #3498db; color: white;}}
        tr:hover {{background: #f5f5f5;}}
        .highlight {{background: #ffffcc; font-weight: bold;}}
        img {{max-width: 100%; border-radius: 8px; margin: 20px 0; box-shadow: 0 4px 6px rgba(0,0,0,0.1);}}
        .success {{background: #d4edda; border-left: 5px solid #28a745; padding: 15px; margin: 20px 0;}}
        .info {{background: #d1ecf1; border-left: 5px solid #17a2b8; padding: 15px; margin: 20px 0;}}
    </style>
</head>
<body>
    <div class="container">
        <h1>🚌 세종시 버스 정류장 최적화 분석 리포트</h1>
        <p><strong>분석 일시:</strong> {datetime.now().strftime('%Y년 %m월 %d일 %H:%M')}</p>
        <p><strong>분석 방법:</strong> GPS 기반 정밀 분석 + 정수계획법 최적화</p>

        <h2>📊 핵심 지표</h2>
        <div>
            <div class="metric">
                <div class="metric-label">분석 지역</div>
                <div class="metric-value">{len(df_analysis)}</div>
            </div>
            <div class="metric">
                <div class="metric-label">총 이용객</div>
                <div class="metric-value">{df_analysis['총_이용객'].sum()/1e6:.1f}M</div>
            </div>
            <div class="metric">
                <div class="metric-label">신규 정류장</div>
                <div class="metric-value">{len(selected)}</div>
            </div>
            <div class="metric">
                <div class="metric-label">커버리지</div>
                <div class="metric-value">{coverage_rate:.1f}%</div>
            </div>
        </div>

        <h2>🎯 최적화 결과</h2>
        <div class="success">
            <h3>선정된 신규 정류장 위치</h3>
            <table>
                <tr><th>순위</th><th>행정구역</th><th>이용객</th><th>건물수</th><th>GPS 좌표</th></tr>
"""

for rank, idx in enumerate(selected, 1):
    row = df_analysis.iloc[idx]
    html += f"""
                <tr class="highlight">
                    <td>{rank}</td>
                    <td>{row['행정구역']}</td>
                    <td>{row['총_이용객']:,.0f}명</td>
                    <td>{row['건물수']:,.0f}개</td>
                    <td>({row['위도']:.4f}, {row['경도']:.4f})</td>
                </tr>
"""

html += f"""
            </table>
        </div>

        <h2>📈 분석 결과</h2>
        <img src="01_탐색적_분석.png" alt="탐색적 분석">
        <img src="02_시나리오_분석.png" alt="시나리오 분석">

        <h2>🗺️ 인터랙티브 지도</h2>
        <p><a href="03_인터랙티브_지도.html" target="_blank" style="color: #3498db; font-size: 18px; font-weight: bold;">📍 지도 보기 (클릭)</a></p>

        <h2>💡 결론 및 권장사항</h2>
        <div class="info">
            <h3>주요 발견사항</h3>
            <ul>
                <li><strong>{len(selected)}개 정류장</strong>으로 <strong>{coverage_rate:.1f}%</strong> 수요 커버 가능</li>
                <li>실제 GPS 거리 기반 정밀 분석 수행</li>
                <li>수요, 건물밀도, 환승 패턴을 종합 고려한 최적화</li>
            </ul>
            <h3>권장사항</h3>
            <ol>
                <li>상위 3개 지역부터 우선 설치 권장</li>
                <li>기존 정류장과의 중복 확인 필요</li>
                <li>도로 접근성 및 토지 이용 가능성 현장 조사 필요</li>
            </ol>
        </div>

        <hr style="margin: 40px 0;">
        <p style="text-align: center; color: #7f8c8d;">
            <small>본 분석은 2023년 데이터 기반으로 수행되었습니다.</small>
        </p>
    </div>
</body>
</html>
"""

with open('00_종합_리포트.html', 'w', encoding='utf-8') as f:
    f.write(html)
print("✓ 저장: 00_종합_리포트.html")

# ============================================================================
# 완료
# ============================================================================
print("\n" + "="*100)
print("✅ 모든 분석 완료!".center(100))
print("="*100)
print("\n생성된 파일:")
print("  1. 00_종합_리포트.html - 종합 분석 리포트")
print("  2. 01_탐색적_분석.png - 탐색적 데이터 분석")
print("  3. 02_시나리오_분석.png - 시나리오 분석")
print("  4. 03_인터랙티브_지도.html - 인터랙티브 지도")
print("  5. 최적화_결과.csv - 상세 결과 데이터")
print("  6. 시나리오_분석.csv - 시나리오 결과")
print("  7. GPS_거리행렬.csv - GPS 거리 행렬")
print("\n👉 00_종합_리포트.html 파일을 브라우저에서 열어보세요!")
print(f"\n분석 종료: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*100)
