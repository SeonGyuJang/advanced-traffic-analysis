#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
고품질 시각화 생성
=================
경영진 보고용 고해상도 시각화 및 인터랙티브 지도
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import folium
from folium import plugins
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from math import radians, cos, sin, asin, sqrt

# ============================================================================
# 한국어 폰트 설정
# ============================================================================
def setup_korean_font():
    """나눔 폰트 설정"""
    font_list = [f.name for f in fm.fontManager.ttflist]

    korean_fonts = ['NanumGothic', 'NanumBarunGothic', 'NanumSquare', 'Malgun Gothic']

    for font in korean_fonts:
        if font in font_list:
            plt.rcParams['font.family'] = font
            plt.rcParams['axes.unicode_minus'] = False
            print(f"✓ 한국어 폰트 설정: {font}")
            return font

    # 폰트를 찾지 못한 경우
    print("⚠ 한국어 폰트를 찾을 수 없습니다. 기본 폰트 사용")
    plt.rcParams['axes.unicode_minus'] = False
    return 'DejaVu Sans'

# ============================================================================
# 데이터 로드
# ============================================================================
print("="*100)
print("고품질 시각화 생성 중...".center(100))
print("="*100)

setup_korean_font()

print("\n데이터 로드 중...")
stations = pd.read_csv('분석결과_정류장별수요.csv')
grid = pd.read_csv('분석결과_수요그리드.csv')
new_stations = pd.read_csv('분석결과_신규정류장.csv')

with open('분석결과_보고서.json', 'r', encoding='utf-8') as f:
    report = json.load(f)

print("✓ 모든 데이터 로드 완료")

# ============================================================================
# 색상 팔레트
# ============================================================================
COLORS = {
    'primary': '#1E3A8A',
    'secondary': '#3B82F6',
    'success': '#10B981',
    'warning': '#F59E0B',
    'danger': '#EF4444',
    'purple': '#9C27B0',
    'orange': '#FF5722',
    'teal': '#14B8A6'
}

# ============================================================================
# 1. 종합 대시보드 (Executive Summary)
# ============================================================================
print("\n[1/5] 종합 대시보드 생성 중...")

fig = plt.figure(figsize=(24, 16))
gs = fig.add_gridspec(4, 3, hspace=0.35, wspace=0.3)

# 전체 제목
fig.suptitle('세종시 버스정류장 최적화 분석 종합 보고서',
             fontsize=28, fontweight='bold', y=0.98)

# 1-1. 주요 지표
ax1 = fig.add_subplot(gs[0, :])
ax1.axis('off')

kpis = [
    ('기존 정류장', f"{report['기존정류장']['총개수']:,}개", COLORS['primary']),
    ('수요 있는 정류장', f"{report['기존정류장']['수요있는정류장']}개", COLORS['secondary']),
    ('현재 커버율', f"{report['커버리지']['커버율']:.1f}%", COLORS['warning']),
    ('신규 정류장', f"{report['최적화결과']['신규정류장개수']}개", COLORS['success']),
    ('예상 커버 수요', f"{report['최적화결과']['예상커버수요']:,.0f}명", COLORS['danger'])
]

for i, (label, value, color) in enumerate(kpis):
    x = 0.1 + i * 0.18
    ax1.add_patch(plt.Rectangle((x, 0.2), 0.15, 0.6, facecolor=color, alpha=0.2, edgecolor=color, linewidth=2))
    ax1.text(x + 0.075, 0.65, value, ha='center', va='center', fontsize=22, fontweight='bold', color=color)
    ax1.text(x + 0.075, 0.3, label, ha='center', va='center', fontsize=14, color='gray')

ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)

# 1-2. 정류장 수요 분포 (상위 15개)
ax2 = fig.add_subplot(gs[1, :])
top_stations = stations.nlargest(15, '할당_총수요')

bars = ax2.barh(range(len(top_stations)), top_stations['할당_총수요'], color=COLORS['secondary'])
ax2.set_yticks(range(len(top_stations)))
ax2.set_yticklabels(top_stations['정류소명'].values, fontsize=11)
ax2.set_xlabel('할당 총수요 (명)', fontsize=13, fontweight='bold')
ax2.set_title('상위 15개 정류장 수요 분포', fontsize=16, fontweight='bold', pad=15)
ax2.grid(True, alpha=0.3, axis='x')

# 값 표시
for i, (idx, row) in enumerate(top_stations.iterrows()):
    ax2.text(row['할당_총수요'], i, f"  {row['할당_총수요']:,.0f}",
             va='center', fontsize=10, color=COLORS['primary'], fontweight='bold')

# 1-3. 커버리지 파이 차트
ax3 = fig.add_subplot(gs[2, 0])
covered = grid['커버여부'].sum()
uncovered = (~grid['커버여부']).sum()

colors = [COLORS['success'], COLORS['danger']]
explode = (0, 0.05)
wedges, texts, autotexts = ax3.pie(
    [covered, uncovered],
    labels=['커버됨', '미커버'],
    autopct='%1.1f%%',
    colors=colors,
    explode=explode,
    startangle=90,
    textprops={'fontsize': 12, 'fontweight': 'bold'}
)

ax3.set_title('그리드 커버리지 현황', fontsize=14, fontweight='bold', pad=15)

# 1-4. 신규 정류장 우선순위
ax4 = fig.add_subplot(gs[2, 1])
top_new = new_stations.head(10)

bars = ax4.barh(range(len(top_new)), top_new['수요'], color=COLORS['orange'])
ax4.set_yticks(range(len(top_new)))
ax4.set_yticklabels([f"#{i+1}" for i in range(len(top_new))], fontsize=11)
ax4.set_xlabel('예상 수요 (명)', fontsize=12, fontweight='bold')
ax4.set_title('신규 정류장 우선순위 (상위 10개)', fontsize=14, fontweight='bold', pad=15)
ax4.grid(True, alpha=0.3, axis='x')

# 1-5. 거리 분포 히스토그램
ax5 = fig.add_subplot(gs[2, 2])
ax5.hist(grid['최단정류장거리_km'], bins=30, color=COLORS['teal'], alpha=0.7, edgecolor='black')
ax5.axvline(report['설정']['커버리지반경_km'], color=COLORS['danger'], linestyle='--', linewidth=2, label='커버리지 기준')
ax5.set_xlabel('최단 정류장 거리 (km)', fontsize=12, fontweight='bold')
ax5.set_ylabel('셀 개수', fontsize=12, fontweight='bold')
ax5.set_title('정류장까지 거리 분포', fontsize=14, fontweight='bold', pad=15)
ax5.legend(fontsize=10)
ax5.grid(True, alpha=0.3)

# 1-6. 분석 정보
ax6 = fig.add_subplot(gs[3, :])
ax6.axis('off')

info_text = f"""
【분석 개요】

분석 기간: {report['분석기간']['시작']} ~ {report['분석기간']['종료']}
분석 일시: {report['분석일시']}

【설정】
• 커버리지 반경: {report['설정']['커버리지반경_km']} km
• 최소 정류장 간격: {report['설정']['최소정류장간거리_km']} km
• 그리드 해상도: {report['설정']['그리드크기']} (약 {int(report['설정']['그리드크기']*100)} km)

【핵심 인사이트】
• 총 {report['기존정류장']['총개수']:,}개 정류장 중 {report['기존정류장']['수요있는정류장']}개만 실제 수요 보유
• 현재 커버리지 {report['커버리지']['커버율']:.1f}%로 개선 여지 큼
• {report['최적화결과']['신규정류장개수']}개 신규 정류장으로 {report['최적화결과']['예상커버수요']:,.0f}명 추가 커버 가능
• 정수계획법 기반 최적화로 데이터 기반 의사결정 지원
"""

ax6.text(0.05, 0.5, info_text, fontsize=12, verticalalignment='center',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3, pad=15),
         family='monospace')

plt.savefig('보고서_종합대시보드.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✓ 종합 대시보드 저장: 보고서_종합대시보드.png")

plt.close()

# ============================================================================
# 2. 수요 분석 상세
# ============================================================================
print("[2/5] 수요 분석 시각화 생성 중...")

fig, axes = plt.subplots(2, 2, figsize=(20, 16))
fig.suptitle('정류장별 수요 분석 상세', fontsize=24, fontweight='bold', y=0.98)

# 2-1. 승하차 비교 (상위 20개)
ax = axes[0, 0]
top20 = stations.nlargest(20, '할당_총수요')

x = np.arange(len(top20))
width = 0.35

bars1 = ax.barh(x + width/2, top20['할당_승차'], width, label='승차', color=COLORS['primary'])
bars2 = ax.barh(x - width/2, top20['할당_하차'], width, label='하차', color=COLORS['secondary'])

ax.set_yticks(x)
ax.set_yticklabels(top20['정류소명'].values, fontsize=10)
ax.set_xlabel('이용객 수 (명)', fontsize=12, fontweight='bold')
ax.set_title('상위 20개 정류장 승하차 비교', fontsize=16, fontweight='bold', pad=15)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='x')

# 2-2. 환승 수요 (상위 15개)
ax = axes[0, 1]
top_transfer = stations.nlargest(15, '할당_환승')

bars = ax.bar(range(len(top_transfer)), top_transfer['할당_환승'], color=COLORS['warning'])
ax.set_xticks(range(len(top_transfer)))
ax.set_xticklabels(top_transfer['정류소명'].values, rotation=45, ha='right', fontsize=10)
ax.set_ylabel('환승 인원 (명)', fontsize=12, fontweight='bold')
ax.set_title('상위 15개 정류장 환승 수요', fontsize=16, fontweight='bold', pad=15)
ax.grid(True, alpha=0.3, axis='y')

# 값 표시
for i, v in enumerate(top_transfer['할당_환승']):
    ax.text(i, v, f'{v:,.0f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# 2-3. 수요 분포 박스플롯
ax = axes[1, 0]
demand_data = stations[stations['할당_총수요'] > 0]['할당_총수요']

bp = ax.boxplot([demand_data], vert=False, patch_artist=True,
                boxprops=dict(facecolor=COLORS['teal'], alpha=0.7),
                medianprops=dict(color='red', linewidth=2),
                flierprops=dict(marker='o', markerfacecolor=COLORS['danger'], markersize=8))

ax.set_yticklabels(['총수요'])
ax.set_xlabel('수요 (명)', fontsize=12, fontweight='bold')
ax.set_title('정류장 수요 분포 (Box Plot)', fontsize=16, fontweight='bold', pad=15)
ax.grid(True, alpha=0.3, axis='x')

# 통계 정보
stats_text = f"평균: {demand_data.mean():,.0f}명\n중앙값: {demand_data.median():,.0f}명\n최대: {demand_data.max():,.0f}명"
ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
        ha='right', va='top', fontsize=11,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# 2-4. 신규 정류장 수요 vs 커버 수요
ax = axes[1, 1]

x = np.arange(len(new_stations))
width = 0.35

bars1 = ax.bar(x - width/2, new_stations['수요'], width, label='예상 수요', color=COLORS['orange'])
bars2 = ax.bar(x + width/2, new_stations['커버_수요'], width, label='커버 수요', color=COLORS['success'])

ax.set_xlabel('신규 정류장 우선순위', fontsize=12, fontweight='bold')
ax.set_ylabel('수요 (명)', fontsize=12, fontweight='bold')
ax.set_title('신규 정류장 수요 분석', fontsize=16, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels([f"#{i+1}" for i in range(len(new_stations))], fontsize=10)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('보고서_수요분석.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✓ 수요 분석 저장: 보고서_수요분석.png")

plt.close()

# ============================================================================
# 3. 인터랙티브 지도
# ============================================================================
print("[3/5] 인터랙티브 지도 생성 중...")

center_lat = stations['위도'].mean()
center_lon = stations['경도'].mean()

m = folium.Map(
    location=[center_lat, center_lon],
    zoom_start=11,
    tiles='CartoDB positron'
)

# 수요 히트맵 레이어
heat_data = [[row['위도'], row['경도'], row['수요']] for _, row in grid[grid['수요'] > 0].iterrows()]
plugins.HeatMap(heat_data, radius=15, blur=25, gradient={0.4: 'blue', 0.65: 'lime', 0.8: 'yellow', 1.0: 'red'}).add_to(m)

# 기존 정류장
max_demand = stations['할당_총수요'].max()
for _, row in stations[stations['할당_총수요'] > 0].head(100).iterrows():  # 상위 100개만
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
        <div style="width:200px;">
        <h4 style="margin:0;">{row['정류소명']}</h4>
        <hr style="margin:5px 0;">
        <b>할당 수요:</b> {row['할당_총수요']:,.0f}명<br>
        <b>승차:</b> {row['할당_승차']:,.0f}명<br>
        <b>하차:</b> {row['할당_하차']:,.0f}명<br>
        <b>환승:</b> {row['할당_환승']:,.0f}명
        </div>
        """,
        tooltip=f"{row['정류소명']}"
    ).add_to(m)

# 신규 정류장
for idx, row in new_stations.iterrows():
    folium.Marker(
        location=[row['위도'], row['경도']],
        popup=f"""
        <div style="width:250px;">
        <h3 style="margin:0; color:#EF4444;">신규 정류장 #{row['우선순위']}</h3>
        <hr style="margin:10px 0;">
        <table style="width:100%; font-size:12px;">
        <tr><td><b>예상 수요</b></td><td align="right">{row['수요']:,.0f}명</td></tr>
        <tr><td><b>환승</b></td><td align="right">{row['환승']:,.0f}명</td></tr>
        <tr><td><b>커버 수요</b></td><td align="right">{row['커버_수요']:,.0f}명</td></tr>
        <tr><td><b>평균 거리</b></td><td align="right">{row['평균거리']:.2f} km</td></tr>
        <tr><td><b>셀 개수</b></td><td align="right">{row['셀개수']}개</td></tr>
        </table>
        <hr style="margin:10px 0;">
        <small>위치: ({row['위도']:.6f}, {row['경도']:.6f})</small>
        </div>
        """,
        icon=folium.Icon(color='red', icon='star', prefix='fa'),
        tooltip=f"🌟 우선순위 #{row['우선순위']} (수요: {row['수요']:,.0f}명)"
    ).add_to(m)

    # 커버리지 원
    folium.Circle(
        location=[row['위도'], row['경도']],
        radius=500,
        color='#FF5722',
        fill=True,
        fillColor='#FF5722',
        fillOpacity=0.15,
        weight=2,
        popup=f"커버리지 반경 0.5km"
    ).add_to(m)

# 범례
legend_html = """
<div style="position: fixed;
            bottom: 50px; left: 50px; width: 280px;
            background-color: white; border:2px solid #1E3A8A; z-index:9999;
            font-size:14px; padding: 20px; border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.2);">
<h3 style="margin-top:0; color:#1E3A8A; border-bottom: 2px solid #3B82F6; padding-bottom: 10px;">범례</h3>
<p style="margin: 8px 0;"><span style="color:#9C27B0; font-size: 24px;">●</span> <b>기존 정류장</b> (크기 = 수요)</p>
<p style="margin: 8px 0;"><span style="color:#EF4444; font-size: 24px;">★</span> <b>신규 추천 정류장</b></p>
<p style="margin: 8px 0;"><span style="color:#FF5722; font-size: 20px;">○</span> 커버리지 범위 (0.5km)</p>
<p style="margin: 8px 0;">🔥 <b>히트맵</b>: 수요 밀도</p>
<hr style="margin: 10px 0;">
<p style="margin: 0; font-size: 11px; color: gray;">
총 {report['최적화결과']['신규정류장개수']}개 신규 정류장 추천<br>
예상 커버 수요: {report['최적화결과']['예상커버수요']:,.0f}명
</p>
</div>
"""
m.get_root().html.add_child(folium.Element(legend_html))

# 저장
m.save('보고서_인터랙티브지도.html')
print("✓ 인터랙티브 지도 저장: 보고서_인터랙티브지도.html")

# ============================================================================
# 4. Plotly 인터랙티브 차트
# ============================================================================
print("[4/5] Plotly 차트 생성 중...")

# 3D Scatter Plot
fig = go.Figure()

# 기존 정류장
fig.add_trace(go.Scatter3d(
    x=stations[stations['할당_총수요'] > 0]['위도'],
    y=stations[stations['할당_총수요'] > 0]['경도'],
    z=stations[stations['할당_총수요'] > 0]['할당_총수요'],
    mode='markers',
    name='기존 정류장',
    marker=dict(
        size=5,
        color=stations[stations['할당_총수요'] > 0]['할당_총수요'],
        colorscale='Purples',
        showscale=True,
        colorbar=dict(title="수요")
    ),
    text=stations[stations['할당_총수요'] > 0]['정류소명'],
    hovertemplate='<b>%{text}</b><br>수요: %{z:,.0f}명<extra></extra>'
))

# 신규 정류장
fig.add_trace(go.Scatter3d(
    x=new_stations['위도'],
    y=new_stations['경도'],
    z=new_stations['수요'],
    mode='markers',
    name='신규 정류장',
    marker=dict(
        size=10,
        color='red',
        symbol='diamond',
        line=dict(color='darkred', width=2)
    ),
    text=[f"신규 #{i+1}" for i in range(len(new_stations))],
    hovertemplate='<b>%{text}</b><br>예상 수요: %{z:,.0f}명<extra></extra>'
))

fig.update_layout(
    title='정류장 3D 분포 (위도 × 경도 × 수요)',
    scene=dict(
        xaxis_title='위도',
        yaxis_title='경도',
        zaxis_title='수요 (명)'
    ),
    height=700
)

fig.write_html('보고서_3D분포.html')
print("✓ 3D 분포 차트 저장: 보고서_3D분포.html")

# ============================================================================
# 5. 요약 보고서 텍스트
# ============================================================================
print("[5/5] 요약 보고서 생성 중...")

summary_md = f"""
# 세종시 버스정류장 최적화 분석 보고서

## 📊 Executive Summary

### 분석 개요
- **분석 기간**: {report['분석기간']['시작']} ~ {report['분석기간']['종료']}
- **분석 일시**: {report['분석일시']}
- **분석 방법**: 정수계획법 (Integer Programming) 기반 최적화

### 주요 발견사항

#### 1. 현황 분석
- **총 정류장**: {report['기존정류장']['총개수']:,}개
- **수요 있는 정류장**: {report['기존정류장']['수요있는정류장']}개 (전체의 {report['기존정류장']['수요있는정류장']/report['기존정류장']['총개수']*100:.1f}%)
- **평균 수요**: {report['기존정류장']['평균수요']:,.0f}명/정류장
- **최대 수요**: {report['기존정류장']['최대수요']:,.0f}명

#### 2. 커버리지 분석
- **그리드 분석**: {report['수요밀도']['그리드셀개수']:,}개 셀 (500m × 500m)
- **현재 커버율**: {report['커버리지']['커버율']:.1f}%
- **미커버 고수요 지역**: {report['커버리지']['미커버고수요지역']}개
- **미커버 수요**: {report['커버리지']['미커버수요']:,.0f}명

#### 3. 최적화 결과
- **신규 정류장 추천**: {report['최적화결과']['신규정류장개수']}개
- **예상 커버 수요**: {report['최적화결과']['예상커버수요']:,.0f}명
- **예상 커버율 개선**: {report['커버리지']['커버율']:.1f}% → {min(100, report['커버리지']['커버율'] + report['최적화결과']['예상커버수요']/report['수요밀도']['총수요']*100):.1f}%

## 📍 신규 정류장 추천 목록

| 우선순위 | 위도 | 경도 | 예상 수요 | 환승 | 커버 수요 |
|---------|------|------|----------|------|----------|
"""

for _, row in new_stations.head(10).iterrows():
    summary_md += f"| {row['우선순위']} | {row['위도']:.6f} | {row['경도']:.6f} | {row['수요']:,.0f}명 | {row['환승']:,.0f}명 | {row['커버_수요']:,.0f}명 |\n"

summary_md += f"""

## 🎯 권장 사항

### 단기 (1-3개월)
1. **우선순위 1-5번 정류장 현장 조사**
   - 실제 도로 상황 및 토지 여건 확인
   - 주변 주요 시설 및 유동 인구 조사

2. **타당성 검토**
   - 예산 및 공사 난이도 평가
   - 지역 주민 의견 수렴

### 중기 (3-6개월)
1. **1단계 설치 (우선순위 1-3번)**
   - 수요가 가장 높은 3개 지점 우선 설치
   - 설치 후 이용 현황 모니터링

2. **효과 분석**
   - 신규 정류장 이용률 분석
   - 주변 정류장 수요 변화 모니터링

### 장기 (6-12개월)
1. **2단계 설치 (우선순위 4-10번)**
   - 1단계 결과를 반영한 추가 설치
   - 지역 균형을 고려한 단계적 확대

2. **지속적 모니터링 및 최적화**
   - 분기별 수요 패턴 분석
   - 정류장 위치 미세 조정

## 📈 기대 효과

1. **서비스 접근성 향상**
   - 도보 7분 이내 정류장 접근 인구 증가
   - 대중교통 이용 편의성 제고

2. **수요 충족**
   - {report['최적화결과']['예상커버수요']:,.0f}명 추가 수요 커버
   - 미충족 수요 해소를 통한 주민 만족도 향상

3. **지역 균형 발전**
   - 상대적 소외 지역 교통 인프라 개선
   - 지역 간 교통 형평성 제고

## ⚠️ 유의사항

1. **현장 조사 필수**
   - 본 분석은 수요 데이터 기반 최적 위치 제시
   - 실제 도로, 토지 여건 등 현장 상황 반드시 확인 필요

2. **단계적 접근**
   - 한 번에 모든 정류장 설치보다는 단계적 설치 권장
   - 각 단계마다 효과 분석 후 다음 단계 진행

3. **지속적 모니터링**
   - 설치 후 이용 현황 지속 모니터링
   - 필요시 위치 조정 또는 추가 설치 검토

---

**문의**: Advanced Traffic Analysis Team
**분석 도구**: Python, PuLP (정수계획법), Folium, Plotly
**데이터 출처**: 세종시 버스 승하차 데이터 (2024-2025)
"""

with open('보고서_요약.md', 'w', encoding='utf-8') as f:
    f.write(summary_md)

print("✓ 요약 보고서 저장: 보고서_요약.md")

# ============================================================================
# 완료
# ============================================================================
print("\n" + "="*100)
print("모든 시각화 생성 완료!".center(100))
print("="*100)
print("\n생성된 파일:")
print("  1. 보고서_종합대시보드.png - 경영진 요약 대시보드")
print("  2. 보고서_수요분석.png - 상세 수요 분석")
print("  3. 보고서_인터랙티브지도.html - 인터랙티브 지도 (웹브라우저에서 열기)")
print("  4. 보고서_3D분포.html - 3D 분포 차트")
print("  5. 보고서_요약.md - 종합 보고서 (마크다운)")
print("\n대시보드 실행:")
print("  streamlit run 전문가_대시보드.py")
