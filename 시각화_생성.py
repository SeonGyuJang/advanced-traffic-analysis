#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
정적 시각화 생성
===============
대시보드를 위한 정적 이미지 및 HTML 파일 생성
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import folium
from folium import plugins
from math import radians, cos, sin, asin, sqrt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json

# ============================================================================
# 설정
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

def haversine_distance(lat1, lon1, lat2, lon2):
    """Haversine 공식으로 실제 거리 계산 (km)"""
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    return 6371 * 2 * asin(sqrt(a))

# ============================================================================
# 데이터 로드
# ============================================================================
setup_korean_font()

print("데이터 로드 중...")
df_stations = pd.read_csv('data/세종도시교통공사_버스정류장_시설현황_20210924.csv')
df_stations = df_stations.drop_duplicates(subset=['정류소ID'])

df_passenger = pd.read_csv('data/지역별승하차_통합데이터.csv')
df_passenger['날짜'] = pd.to_datetime(df_passenger['날짜'])

df_region_coords = pd.read_csv('data/행정구역_중심좌표.csv')

# 최적화 결과
try:
    df_new_stations = pd.read_csv('최적화_신규정류장.csv')
    df_underserved = pd.read_csv('서비스부족지역.csv')
    with open('최적화_분석_보고서.json', 'r', encoding='utf-8') as f:
        report = json.load(f)
    print("✓ 최적화 결과 로드 완료")
except FileNotFoundError:
    print("⚠ 최적화 결과를 찾을 수 없습니다. 먼저 버스정류장_최적화_분석.py를 실행하세요.")
    exit(1)

# ============================================================================
# 1. 인터랙티브 지도 생성
# ============================================================================
print("\n인터랙티브 지도 생성 중...")

center_lat = df_stations['위도'].mean()
center_lon = df_stations['경도'].mean()

m = folium.Map(
    location=[center_lat, center_lon],
    zoom_start=11,
    tiles='OpenStreetMap'
)

# 기존 정류장 레이어
existing_layer = folium.FeatureGroup(name='기존 정류장')
for _, row in df_stations.iterrows():
    folium.CircleMarker(
        location=[row['위도'], row['경도']],
        radius=3,
        color='#9C27B0',
        fill=True,
        fillColor='#9C27B0',
        fillOpacity=0.6,
        popup=f"<b>{row['정류소명']}</b><br>기존 정류장"
    ).add_to(existing_layer)
existing_layer.add_to(m)

# 신규 추천 정류장 레이어
new_layer = folium.FeatureGroup(name='신규 추천 정류장')
for _, row in df_new_stations.iterrows():
    # 마커
    folium.Marker(
        location=[row['위도'], row['경도']],
        popup=f"""
        <div style="width: 200px">
        <h4>{row['행정구역']}</h4>
        <b>신규 추천 정류장 #{row['우선순위']}</b><br><br>
        <b>수요 정보:</b><br>
        - 총 이용객: {row['총_이용객']:,.0f}명<br>
        - 환승: {row['환승']:,.0f}명<br>
        - 커버 수요: {row['커버_수요']:,.0f}명<br><br>
        <b>위치:</b><br>
        - 위도: {row['위도']:.6f}<br>
        - 경도: {row['경도']:.6f}
        </div>
        """,
        icon=folium.Icon(color='red', icon='plus', prefix='fa'),
        tooltip=f"신규 #{row['우선순위']}: {row['행정구역']}"
    ).add_to(new_layer)

    # 커버리지 원
    folium.Circle(
        location=[row['위도'], row['경도']],
        radius=500,  # 0.5km
        color='#FF5722',
        fill=True,
        fillColor='#FF5722',
        fillOpacity=0.1,
        weight=2,
        popup=f"커버리지 반경 0.5km"
    ).add_to(new_layer)
new_layer.add_to(m)

# 서비스 부족 지역 레이어
underserved_layer = folium.FeatureGroup(name='서비스 부족 지역')
for _, row in df_underserved.iterrows():
    folium.CircleMarker(
        location=[row['위도'], row['경도']],
        radius=8,
        color='#D64933',
        fill=True,
        fillColor='#D64933',
        fillOpacity=0.5,
        popup=f"""
        <b>{row['행정구역']}</b><br>
        서비스 부족 지역<br>
        최단거리: {row['최단거리_km']:.2f}km<br>
        총 이용객: {row['총_이용객']:,.0f}명
        """,
        tooltip=f"{row['행정구역']} (미커버)"
    ).add_to(underserved_layer)
underserved_layer.add_to(m)

# 레이어 컨트롤 추가
folium.LayerControl().add_to(m)

# 범례 추가
legend_html = """
<div style="position: fixed;
            bottom: 50px; left: 50px; width: 250px; height: auto;
            background-color: white; border:2px solid grey; z-index:9999;
            font-size:14px; padding: 15px; box-shadow: 2px 2px 6px rgba(0,0,0,0.3);">
<h4 style="margin-top:0;">범례</h4>
<p style="margin: 5px 0;"><span style="color:#9C27B0; font-size: 20px;">●</span> 기존 정류장</p>
<p style="margin: 5px 0;"><span style="color:#FF5722; font-size: 20px;">📍</span> 신규 추천 정류장</p>
<p style="margin: 5px 0;"><span style="color:#D64933; font-size: 20px;">●</span> 서비스 부족 지역</p>
<p style="margin: 5px 0;"><span style="color:#FF5722;">○</span> 커버리지 반경 (0.5km)</p>
</div>
"""
m.get_root().html.add_child(folium.Element(legend_html))

# 지도 저장
m.save('최적화_인터랙티브_지도.html')
print("✓ 인터랙티브 지도 저장: 최적화_인터랙티브_지도.html")

# ============================================================================
# 2. 종합 시각화 (정적)
# ============================================================================
print("\n종합 시각화 생성 중...")

fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# 2-1. 일별 승하차 추이
ax1 = fig.add_subplot(gs[0, :2])
daily_stats = df_passenger.groupby('날짜').agg({
    '승차': 'sum',
    '하차': 'sum',
    '환승': 'sum'
}).reset_index()

ax1.plot(daily_stats['날짜'], daily_stats['승차'], label='승차', linewidth=2, color='#2E4057')
ax1.plot(daily_stats['날짜'], daily_stats['하차'], label='하차', linewidth=2, color='#048A81')
ax1.plot(daily_stats['날짜'], daily_stats['환승'], label='환승', linewidth=2, color='#F26419')
ax1.set_title('일별 승하차 및 환승 추이', fontsize=16, fontweight='bold')
ax1.set_xlabel('날짜', fontsize=12)
ax1.set_ylabel('이용객 수', fontsize=12)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# 2-2. 주요 통계
ax2 = fig.add_subplot(gs[0, 2])
ax2.axis('off')
stats_text = f"""
주요 통계

기존 정류장: {len(df_stations):,}개

신규 추천: {len(df_new_stations)}개

서비스 부족: {len(df_underserved)}개

총 데이터: {len(df_passenger):,}건

분석 기간:
{df_passenger['날짜'].min().strftime('%Y-%m-%d')}
~
{df_passenger['날짜'].max().strftime('%Y-%m-%d')}
"""
ax2.text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# 2-3. 지역별 총 이용객 (상위 15개)
ax3 = fig.add_subplot(gs[1, :])
region_stats = df_passenger.groupby('행정구역').agg({
    '승차': 'sum',
    '하차': 'sum'
}).reset_index()
region_stats['총_이용객'] = region_stats['승차'] + region_stats['하차']
region_stats = region_stats.sort_values('총_이용객', ascending=False).head(15)

bars = ax3.barh(region_stats['행정구역'], region_stats['총_이용객'], color='#048A81')
ax3.set_title('행정구역별 총 이용객 (상위 15개)', fontsize=16, fontweight='bold')
ax3.set_xlabel('총 이용객', fontsize=12)
ax3.grid(True, alpha=0.3, axis='x')

# 값 표시
for i, (idx, row) in enumerate(region_stats.iterrows()):
    ax3.text(row['총_이용객'], i, f" {row['총_이용객']:,.0f}",
             va='center', fontsize=9)

# 2-4. 신규 정류장 우선순위
ax4 = fig.add_subplot(gs[2, 0])
ax4.barh(df_new_stations['행정구역'], df_new_stations['총_이용객'], color='#FF5722')
ax4.set_title('신규 정류장 추천 (수요 기준)', fontsize=14, fontweight='bold')
ax4.set_xlabel('예상 수요', fontsize=10)
ax4.grid(True, alpha=0.3, axis='x')
ax4.tick_params(labelsize=9)

# 2-5. 커버 수요
ax5 = fig.add_subplot(gs[2, 1])
ax5.barh(df_new_stations['행정구역'], df_new_stations['커버_수요'], color='#06A77D')
ax5.set_title('신규 정류장 커버 수요', fontsize=14, fontweight='bold')
ax5.set_xlabel('커버 수요', fontsize=10)
ax5.grid(True, alpha=0.3, axis='x')
ax5.tick_params(labelsize=9)

# 2-6. 환승률
ax6 = fig.add_subplot(gs[2, 2])
region_stats_full = df_passenger.groupby('행정구역').agg({
    '승차': 'sum',
    '하차': 'sum',
    '환승': 'sum'
}).reset_index()
region_stats_full['총_이용객'] = region_stats_full['승차'] + region_stats_full['하차']
region_stats_full['환승률'] = (region_stats_full['환승'] / region_stats_full['총_이용객'] * 100).round(2)
region_stats_full = region_stats_full.sort_values('환승률', ascending=False).head(10)

ax6.barh(region_stats_full['행정구역'], region_stats_full['환승률'], color='#F4B41A')
ax6.set_title('환승률 상위 10개 지역 (%)', fontsize=14, fontweight='bold')
ax6.set_xlabel('환승률 (%)', fontsize=10)
ax6.grid(True, alpha=0.3, axis='x')
ax6.tick_params(labelsize=9)

plt.suptitle('세종시 버스정류장 최적화 종합 분석', fontsize=20, fontweight='bold', y=0.98)

plt.savefig('최적화_종합_시각화.png', dpi=300, bbox_inches='tight')
print("✓ 종합 시각화 저장: 최적화_종합_시각화.png")

# ============================================================================
# 3. Plotly 인터랙티브 차트
# ============================================================================
print("\nPlotly 인터랙티브 차트 생성 중...")

# 3-1. 시계열 차트
fig_time = go.Figure()

fig_time.add_trace(go.Scatter(
    x=daily_stats['날짜'],
    y=daily_stats['승차'],
    name='승차',
    mode='lines',
    line=dict(color='#2E4057', width=2)
))

fig_time.add_trace(go.Scatter(
    x=daily_stats['날짜'],
    y=daily_stats['하차'],
    name='하차',
    mode='lines',
    line=dict(color='#048A81', width=2)
))

fig_time.add_trace(go.Scatter(
    x=daily_stats['날짜'],
    y=daily_stats['환승'],
    name='환승',
    mode='lines',
    line=dict(color='#F26419', width=2)
))

fig_time.update_layout(
    title='일별 승하차 및 환승 추이',
    xaxis_title='날짜',
    yaxis_title='이용객 수',
    hovermode='x unified',
    height=500
)

fig_time.write_html('시계열_차트.html')
print("✓ 시계열 차트 저장: 시계열_차트.html")

# 3-2. 지역별 차트
region_stats_sorted = region_stats.sort_values('총_이용객', ascending=True)

fig_region = go.Figure(go.Bar(
    x=region_stats_sorted['총_이용객'],
    y=region_stats_sorted['행정구역'],
    orientation='h',
    marker=dict(
        color=region_stats_sorted['총_이용객'],
        colorscale='Reds',
        showscale=True,
        colorbar=dict(title="이용객 수")
    ),
    text=region_stats_sorted['총_이용객'],
    texttemplate='%{text:,.0f}',
    textposition='outside'
))

fig_region.update_layout(
    title='행정구역별 총 이용객 (상위 15개)',
    xaxis_title='총 이용객',
    yaxis_title='행정구역',
    height=500
)

fig_region.write_html('지역별_차트.html')
print("✓ 지역별 차트 저장: 지역별_차트.html")

print("\n" + "="*80)
print("모든 시각화 생성 완료!")
print("="*80)
print("\n생성된 파일:")
print("  - 최적화_인터랙티브_지도.html")
print("  - 최적화_종합_시각화.png")
print("  - 시계열_차트.html")
print("  - 지역별_차트.html")
print("\n대시보드 실행: streamlit run 대시보드.py")
