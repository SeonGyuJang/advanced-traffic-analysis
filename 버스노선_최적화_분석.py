#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
세종시 버스 노선 최적화 분석
============================
버스 노선별 수요 분석 및 최적화 제안
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns
import folium
from folium import plugins
from math import radians, cos, sin, asin, sqrt
import warnings
from datetime import datetime
import json
import re

warnings.filterwarnings('ignore')

# 색상 팔레트
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

def parse_interval(interval_str):
    """배차간격 문자열을 분 단위 숫자로 변환"""
    if pd.isna(interval_str) or interval_str == '':
        return None

    # "8~12분", "10~20분" 형태 처리
    match = re.search(r'(\d+)(?:~(\d+))?', str(interval_str))
    if match:
        min_val = int(match.group(1))
        max_val = int(match.group(2)) if match.group(2) else min_val
        return (min_val + max_val) / 2
    return None

def parse_frequency(freq_str):
    """운행횟수 문자열을 숫자로 변환"""
    if pd.isna(freq_str) or freq_str == '':
        return 0

    # "112", "62회" 형태 처리
    match = re.search(r'(\d+\.?\d*)', str(freq_str))
    if match:
        return float(match.group(1))
    return 0

# ============================================================================
# 메인 분석
# ============================================================================
print("="*100)
print("세종시 버스 노선 최적화 분석".center(100))
print("="*100)
print(f"⏰ 분석 시작: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

use_korean = setup_korean_font()
print(f"{'✓' if use_korean else '⚠'} 한글 폰트: {'적용됨' if use_korean else '미적용'}\n")

# ============================================================================
# 1. 데이터 로드
# ============================================================================
print("[1단계] 데이터 로드")
print("-"*100)

# 버스 운행 현황
df_bus = pd.read_csv('data/세종특별자치시_ 버스운행 현황_20250527.csv', encoding='euc-kr')
print(f"✓ 버스 노선: {df_bus.shape[0]}개 노선")

# 버스 정류장
df_stops = pd.read_csv('data/세종도시교통공사_버스정류장_시설현황_20210924.csv')
print(f"✓ 버스 정류장: {df_stops.shape[0]}개 정류장")

# 승하차 데이터
df_passenger = pd.read_csv('data/지역별승하차_통합데이터.csv')
print(f"✓ 승하차 데이터: {df_passenger.shape[0]:,}건")

# GPS 좌표
df_coords = pd.read_csv('data/행정구역_중심좌표.csv')
print(f"✓ GPS 좌표: {df_coords.shape[0]}개 지역")

# ============================================================================
# 2. 데이터 전처리
# ============================================================================
print("\n[2단계] 데이터 전처리")
print("-"*100)

# 버스 데이터 정리
df_bus['배차간격_분'] = df_bus['배차간격'].apply(parse_interval)
df_bus['운행횟수_일'] = df_bus['운행횟수'].apply(parse_frequency)
df_bus['일평균_운행시간'] = df_bus['배차간격_분'] * df_bus['운행횟수_일'] / 60  # 시간

print(f"✓ 버스 노선 유형:")
print(df_bus['구분'].value_counts())

# 승하차 데이터 집계
df_passenger['날짜'] = pd.to_datetime(df_passenger['날짜'])
df_passenger['총_이용객'] = df_passenger['승차'] + df_passenger['하차']

region_stats = df_passenger.groupby('행정구역').agg({
    '승차': 'sum',
    '하차': 'sum',
    '환승': 'sum',
    '총_이용객': 'sum'
}).reset_index()

# GPS 좌표 결합
df_regions = pd.merge(region_stats, df_coords, on='행정구역')

print(f"\n✓ 총 이용객: {region_stats['총_이용객'].sum():,.0f}명")
print(f"✓ 분석 지역: {len(df_regions)}개")

# ============================================================================
# 3. 버스 정류장 위치 추출
# ============================================================================
print("\n[3단계] 버스 정류장 위치 분석")
print("-"*100)

# 정류장 좌표 확인
if 'X좌표' in df_stops.columns and 'Y좌표' in df_stops.columns:
    df_stops['경도'] = df_stops['X좌표']
    df_stops['위도'] = df_stops['Y좌표']
    print(f"✓ 정류장 좌표 {len(df_stops)}개")
elif '경도' in df_stops.columns and '위도' in df_stops.columns:
    print(f"✓ 정류장 좌표 {len(df_stops)}개")
else:
    print("⚠ 정류장 GPS 좌표가 없습니다.")

# ============================================================================
# 4. 노선별 지역 커버리지 분석
# ============================================================================
print("\n[4단계] 노선별 지역 커버리지 분석")
print("-"*100)

def extract_regions_from_route(row):
    """노선의 기점지, 경유지, 종점지에서 행정구역 추출"""
    regions = set()

    for field in ['기점지', '경유지', '종점지']:
        if pd.notna(row[field]):
            text = str(row[field])
            # 행정구역 이름 매칭
            for region in df_regions['행정구역'].values:
                if region in text:
                    regions.add(region)

    return list(regions)

# 각 노선이 커버하는 지역
df_bus['커버지역'] = df_bus.apply(extract_regions_from_route, axis=1)
df_bus['커버지역수'] = df_bus['커버지역'].apply(len)

# 각 노선의 수요 계산
def calculate_route_demand(regions):
    """노선이 커버하는 지역의 총 수요"""
    if not regions:
        return 0
    demand_sum = df_regions[df_regions['행정구역'].isin(regions)]['총_이용객'].sum()
    return demand_sum

df_bus['추정수요'] = df_bus['커버지역'].apply(calculate_route_demand)

print(f"✓ 노선당 평균 커버 지역: {df_bus['커버지역수'].mean():.1f}개")
print(f"✓ 총 추정 수요: {df_bus['추정수요'].sum()/1e6:.1f}M명")

# ============================================================================
# 5. 노선별 수요-공급 분석
# ============================================================================
print("\n[5단계] 노선별 수요-공급 매칭 분석")
print("-"*100)

# 수요 대비 공급 비율 계산
df_bus['수요대비운행비율'] = df_bus.apply(
    lambda x: (x['운행횟수_일'] / (x['추정수요'] / 100000)) if x['추정수요'] > 0 else 0,
    axis=1
)

# 배차간격 적정성 평가
def evaluate_interval(row):
    """배차간격 적정성 평가"""
    if pd.isna(row['배차간격_분']):
        return '정보없음'

    demand = row['추정수요']
    interval = row['배차간격_분']

    if demand > 500000:  # 50만 이상 수요
        if interval <= 10:
            return '적정'
        elif interval <= 20:
            return '부족'
        else:
            return '심각부족'
    elif demand > 200000:  # 20만 이상
        if interval <= 15:
            return '적정'
        elif interval <= 30:
            return '부족'
        else:
            return '심각부족'
    else:  # 저수요
        if interval <= 30:
            return '적정'
        elif interval <= 60:
            return '부족'
        else:
            return '심각부족'

df_bus['배차적정성'] = df_bus.apply(evaluate_interval, axis=1)

print("\n📊 배차간격 적정성 평가:")
print(df_bus['배차적정성'].value_counts())

# ============================================================================
# 6. 노선 최적화 제안 생성
# ============================================================================
print("\n[6단계] 노선 최적화 제안 생성")
print("-"*100)

def generate_optimization_proposal(row):
    """각 노선별 최적화 제안"""
    proposals = []

    demand = row['추정수요']
    interval = row['배차간격_분']
    frequency = row['운행횟수_일']
    coverage = row['커버지역수']
    adequacy = row['배차적정성']

    # 1. 배차간격 조정 제안
    if adequacy == '심각부족':
        if demand > 500000:
            proposals.append(f"배차간격 {interval:.0f}분 → 8-10분으로 단축 (고수요 노선)")
        elif demand > 200000:
            proposals.append(f"배차간격 {interval:.0f}분 → 15분으로 단축")
        else:
            proposals.append(f"배차간격 {interval:.0f}분 → 20-25분으로 조정")
    elif adequacy == '부족':
        if demand > 500000:
            proposals.append(f"배차간격 {interval:.0f}분 → 12분으로 소폭 단축")
        elif demand > 200000:
            proposals.append(f"배차간격 {interval:.0f}분 → 15-18분으로 조정")

    # 2. 운행횟수 조정
    if demand > 500000 and frequency < 80:
        proposals.append(f"일 운행횟수 {frequency:.0f}회 → 100회 이상으로 증편")
    elif demand > 300000 and frequency < 60:
        proposals.append(f"일 운행횟수 {frequency:.0f}회 → 80회로 증편")

    # 3. 커버리지 확대
    if coverage < 2 and demand > 100000:
        proposals.append("경유지 추가로 커버리지 확대 필요")

    # 4. 수요 대비 과잉 운행
    ratio = row['수요대비운행비율']
    if ratio > 50 and demand < 50000:
        proposals.append(f"저수요 노선 - 배차간격 조정으로 효율성 개선")

    # 제안이 없으면
    if not proposals:
        if adequacy == '적정':
            proposals.append("현재 배차 상태 양호 - 유지")
        else:
            proposals.append("추가 분석 필요")

    return ' | '.join(proposals)

df_bus['최적화제안'] = df_bus.apply(generate_optimization_proposal, axis=1)

# 우선순위 점수 계산
df_bus['최적화우선순위'] = (
    (df_bus['추정수요'] / df_bus['추정수요'].max()) * 0.5 +  # 수요 50%
    (df_bus['배차적정성'] == '심각부족').astype(int) * 0.3 +  # 심각부족 30%
    (df_bus['배차적정성'] == '부족').astype(int) * 0.2  # 부족 20%
)

df_bus_sorted = df_bus.sort_values('최적화우선순위', ascending=False)

print("\n🎯 최적화 우선순위 Top 10:")
for idx, row in df_bus_sorted.head(10).iterrows():
    print(f"\n{row['구분']} {row['노선번호']} ({row['기점지']} ↔ {row['종점지']})")
    print(f"  추정수요: {row['추정수요']/1e3:.0f}K명 | 배차: {row['배차간격_분']:.0f}분 | 평가: {row['배차적정성']}")
    print(f"  💡 제안: {row['최적화제안']}")

# ============================================================================
# 7. 인터랙티브 지도 생성
# ============================================================================
print("\n[7단계] 인터랙티브 지도 생성")
print("-"*100)

# 세종시 중심
center_lat = 36.5
center_lon = 127.25

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
fg_high_priority = folium.FeatureGroup(name='🔴 최적화 최우선 노선', show=True)
fg_medium_priority = folium.FeatureGroup(name='🟡 최적화 필요 노선', show=True)
fg_good = folium.FeatureGroup(name='🟢 양호한 노선', show=True)
fg_regions = folium.FeatureGroup(name='📍 수요 지역', show=True)

# 수요 지역 표시
max_demand = df_regions['총_이용객'].max()
for _, row in df_regions.iterrows():
    demand_ratio = row['총_이용객'] / max_demand
    radius = 5 + demand_ratio * 15

    color = (COLORS['danger'] if demand_ratio > 0.7 else
             COLORS['warning'] if demand_ratio > 0.3 else
             COLORS['info'])

    popup_html = f'''
    <div style="font-family: 'Malgun Gothic', sans-serif; min-width: 250px;">
        <h4 style="margin:0 0 10px 0; color: {color};">📍 {row['행정구역']}</h4>
        <table style="width: 100%; font-size: 12px;">
            <tr><td><b>총 이용객</b></td><td><b>{row['총_이용객']:,.0f}명</b></td></tr>
            <tr><td>승차</td><td>{row['승차']:,.0f}명</td></tr>
            <tr><td>하차</td><td>{row['하차']:,.0f}명</td></tr>
            <tr><td>환승</td><td>{row['환승']:,.0f}명</td></tr>
        </table>
    </div>
    '''

    folium.CircleMarker(
        location=[row['위도'], row['경도']],
        radius=radius,
        color=color,
        fillColor=color,
        fillOpacity=0.4,
        weight=2,
        popup=folium.Popup(popup_html, max_width=300),
        tooltip=f"{row['행정구역']}: {row['총_이용객']/1e3:.0f}K명"
    ).add_to(fg_regions)

# 노선별 마커 (최적화 우선순위별)
for rank, (idx, row) in enumerate(df_bus_sorted.iterrows(), 1):
    if row['커버지역수'] == 0:
        continue

    priority_score = row['최적화우선순위']

    # 우선순위에 따른 색상 및 그룹
    if priority_score > 0.5:
        color = COLORS['danger']
        icon_color = 'red'
        fg = fg_high_priority
        priority_text = '최우선'
    elif priority_score > 0.2:
        color = COLORS['warning']
        icon_color = 'orange'
        fg = fg_medium_priority
        priority_text = '필요'
    else:
        color = COLORS['success']
        icon_color = 'green'
        fg = fg_good
        priority_text = '양호'

    # 커버 지역의 중심 계산
    covered = df_regions[df_regions['행정구역'].isin(row['커버지역'])]
    if len(covered) > 0:
        center_lat = covered['위도'].mean()
        center_lon = covered['경도'].mean()

        # 상세 팝업
        covered_regions_str = ', '.join(row['커버지역'][:5])
        if len(row['커버지역']) > 5:
            covered_regions_str += f" 외 {len(row['커버지역'])-5}개"

        popup_html = f'''
        <div style="font-family: 'Malgun Gothic', sans-serif; min-width: 320px;">
            <h3 style="margin:0 0 10px 0; color: {color}; border-bottom: 2px solid {color}; padding-bottom: 5px;">
                {row['구분']} {row['노선번호']}
            </h3>
            <table style="width: 100%; font-size: 13px; line-height: 1.8;">
                <tr><td><b>기점지</b></td><td>{row['기점지']}</td></tr>
                <tr><td><b>종점지</b></td><td>{row['종점지']}</td></tr>
                <tr><td><b>경유지</b></td><td>{row['경유지'] if pd.notna(row['경유지']) else '-'}</td></tr>
                <tr><td>배차간격</td><td><b>{row['배차간격_분']:.0f}분</b></td></tr>
                <tr><td>운행횟수</td><td>{row['운행횟수_일']:.0f}회/일</td></tr>
                <tr><td>커버 지역</td><td>{row['커버지역수']}개</td></tr>
                <tr><td>추정 수요</td><td><b>{row['추정수요']/1e3:.0f}K명</b></td></tr>
                <tr><td>배차 평가</td><td><b style="color: {color};">{row['배차적정성']}</b></td></tr>
            </table>
            <div style="margin-top: 10px; padding: 10px; background: #FFF3E0; border-radius: 5px; border-left: 4px solid {color};">
                <b>💡 최적화 제안:</b><br>
                <span style="font-size: 12px;">{row['최적화제안']}</span>
            </div>
            <div style="margin-top: 8px; padding: 8px; background: #E3F2FD; border-radius: 5px; font-size: 11px;">
                <b>커버 지역:</b> {covered_regions_str}
            </div>
            <div style="margin-top: 5px; text-align: right; font-size: 11px; color: #666;">
                우선순위: {rank}위 / 점수: {priority_score:.2f}
            </div>
        </div>
        '''

        folium.Marker(
            location=[center_lat, center_lon],
            icon=folium.Icon(color=icon_color, icon='bus', prefix='fa'),
            popup=folium.Popup(popup_html, max_width=350),
            tooltip=f"<b>{row['구분']} {row['노선번호']}</b> ({priority_text})"
        ).add_to(fg)

# 레이어 추가
fg_regions.add_to(m)
fg_high_priority.add_to(m)
fg_medium_priority.add_to(m)
fg_good.add_to(m)

# 미니맵
minimap = plugins.MiniMap(toggle_display=True)
m.add_child(minimap)

# 풀스크린
plugins.Fullscreen(
    position='topright',
    title='전체화면',
    title_cancel='전체화면 해제'
).add_to(m)

# 레이어 컨트롤
folium.LayerControl(position='topright', collapsed=False).add_to(m)

# 범례
total_routes = len(df_bus)
high_priority_count = len(df_bus[df_bus['최적화우선순위'] > 0.5])
medium_priority_count = len(df_bus[(df_bus['최적화우선순위'] > 0.2) & (df_bus['최적화우선순위'] <= 0.5)])
good_count = len(df_bus[df_bus['최적화우선순위'] <= 0.2])

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
        🚌 세종시 버스 노선 최적화 분석
    </h4>

    <div style="margin-bottom: 12px;">
        <div style="display: flex; align-items: center; margin-bottom: 8px;">
            <div style="width: 20px; height: 20px; background: {COLORS['danger']}; border-radius: 50%; margin-right: 10px;"></div>
            <span><b>최적화 최우선</b> ({high_priority_count}개 노선)</span>
        </div>
        <div style="display: flex; align-items: center; margin-bottom: 8px;">
            <div style="width: 20px; height: 20px; background: {COLORS['warning']}; border-radius: 50%; margin-right: 10px;"></div>
            <span><b>최적화 필요</b> ({medium_priority_count}개 노선)</span>
        </div>
        <div style="display: flex; align-items: center; margin-bottom: 8px;">
            <div style="width: 20px; height: 20px; background: {COLORS['success']}; border-radius: 50%; margin-right: 10px;"></div>
            <span>현재 상태 양호 ({good_count}개 노선)</span>
        </div>
    </div>

    <div style="background: #F8F9FA; padding: 10px; border-radius: 6px; font-size: 12px; margin-bottom: 10px;">
        <table style="width: 100%; line-height: 1.8;">
            <tr><td><b>총 노선:</b></td><td style="text-align: right;"><b>{total_routes}개</b></td></tr>
            <tr><td><b>분석 지역:</b></td><td style="text-align: right;">{len(df_regions)}개</td></tr>
            <tr><td><b>총 이용객:</b></td><td style="text-align: right;"><b>{region_stats['총_이용객'].sum()/1e6:.1f}M명</b></td></tr>
        </table>
    </div>

    <div style="background: #E3F2FD; padding: 10px; border-radius: 6px; font-size: 11px;">
        <b>📊 배차 적정성 평가</b><br>
        • 적정: {len(df_bus[df_bus['배차적정성']=='적정'])}개<br>
        • 부족: {len(df_bus[df_bus['배차적정성']=='부족'])}개<br>
        • 심각부족: {len(df_bus[df_bus['배차적정성']=='심각부족'])}개
    </div>

    <div style="margin-top: 10px; font-size: 11px; color: #666; text-align: center;">
        📱 버스 마커 클릭 시 상세정보 및 최적화 제안 확인
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
    width: 280px;
    background: rgba(255, 255, 255, 0.95);
    border-radius: 10px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    z-index: 9999;
    font-family: 'Malgun Gothic', sans-serif;
    padding: 15px;
">
    <h4 style="margin: 0 0 10px 0; color: #2E4057;">📈 핵심 지표</h4>
    <div style="font-size: 13px; line-height: 1.8;">
        <div style="margin-bottom: 10px; padding-bottom: 10px; border-bottom: 1px solid #E0E0E0;">
            <div style="color: #666; font-size: 11px;">평균 배차간격</div>
            <div style="font-size: 20px; font-weight: bold; color: {COLORS['primary']};">
                {df_bus['배차간격_분'].mean():.1f}분
            </div>
        </div>
        <div style="margin-bottom: 10px; padding-bottom: 10px; border-bottom: 1px solid #E0E0E0;">
            <div style="color: #666; font-size: 11px;">평균 운행횟수</div>
            <div style="font-size: 20px; font-weight: bold; color: {COLORS['secondary']};">
                {df_bus['운행횟수_일'].mean():.1f}회/일
            </div>
        </div>
        <div style="margin-bottom: 10px; padding-bottom: 10px; border-bottom: 1px solid #E0E0E0;">
            <div style="color: #666; font-size: 11px;">최적화 필요 노선</div>
            <div style="font-size: 20px; font-weight: bold; color: {COLORS['danger']};">
                {high_priority_count + medium_priority_count}개
            </div>
        </div>
        <div>
            <div style="color: #666; font-size: 11px;">개선 가능 수요</div>
            <div style="font-size: 18px; font-weight: bold; color: {COLORS['info']};">
                {df_bus[df_bus['최적화우선순위'] > 0.2]['추정수요'].sum()/1e6:.1f}M명
            </div>
        </div>
    </div>
</div>
'''
m.get_root().html.add_child(folium.Element(stats_html))

# 지도 저장
m.save('버스노선_최적화_지도.html')
print("✓ 저장: 버스노선_최적화_지도.html")

# ============================================================================
# 8. 시각화 생성
# ============================================================================
print("\n[8단계] 시각화 생성")
print("-"*100)

fig = plt.figure(figsize=(20, 14))
gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

fig.patch.set_facecolor('white')
fig.suptitle('세종시 버스 노선 최적화 분석 리포트', fontsize=22, fontweight='bold', y=0.98, color=COLORS['dark'])

subtitle = f"수요-공급 매칭 분석  |  총 {total_routes}개 노선  |  분석일: {datetime.now().strftime('%Y.%m.%d')}"
fig.text(0.5, 0.955, subtitle, ha='center', fontsize=11, color=COLORS['primary'], alpha=0.8)

# 1. 노선 유형별 분포
ax1 = fig.add_subplot(gs[0, 0])
route_types = df_bus['구분'].value_counts()
colors_pie = [COLORS['accent'], COLORS['info'], COLORS['success'], COLORS['warning']][:len(route_types)]
wedges, texts, autotexts = ax1.pie(route_types.values, labels=route_types.index, colors=colors_pie,
                                     autopct='%1.1f%%', startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'},
                                     wedgeprops={'edgecolor': 'white', 'linewidth': 2})
for autotext in autotexts:
    autotext.set_color('white')
ax1.set_title('🚌 노선 유형별 분포', fontsize=14, fontweight='bold', pad=15)

# 2. 배차 적정성 평가
ax2 = fig.add_subplot(gs[0, 1])
adequacy = df_bus['배차적정성'].value_counts()
colors_bar = [COLORS['success'] if '적정' in x else COLORS['warning'] if '부족' in x and '심각' not in x else COLORS['danger']
              for x in adequacy.index]
bars = ax2.bar(range(len(adequacy)), adequacy.values, color=colors_bar, edgecolor=COLORS['dark'], linewidth=1.5, alpha=0.85)
ax2.set_xticks(range(len(adequacy)))
ax2.set_xticklabels(adequacy.index, rotation=30, ha='right', fontsize=10)
ax2.set_ylabel('노선 수', fontsize=11, fontweight='bold')
ax2.set_title('📊 배차 적정성 평가', fontsize=14, fontweight='bold', pad=15)
ax2.grid(axis='y', alpha=0.3, linestyle='--')
for i, val in enumerate(adequacy.values):
    ax2.text(i, val + 1, str(val), ha='center', fontsize=11, fontweight='bold')

# 3. 최적화 우선순위 Top 10
ax3 = fig.add_subplot(gs[0:2, 2])
top10 = df_bus_sorted.head(10)
y_pos = np.arange(len(top10))
colors_priority = [COLORS['danger'] if x > 0.5 else COLORS['warning'] if x > 0.2 else COLORS['success']
                   for x in top10['최적화우선순위'].values]

bars = ax3.barh(y_pos, top10['최적화우선순위'].values, color=colors_priority,
                edgecolor=COLORS['dark'], linewidth=1.5, alpha=0.85)
ax3.set_yticks(y_pos)
ax3.set_yticklabels([f"{row['구분']} {row['노선번호']}" for _, row in top10.iterrows()], fontsize=10)
ax3.set_xlabel('최적화 우선순위 점수', fontsize=11, fontweight='bold')
ax3.set_title('🎯 최적화 우선순위 Top 10', fontsize=14, fontweight='bold', pad=15)
ax3.invert_yaxis()
ax3.grid(axis='x', alpha=0.3, linestyle='--')

# 4. 수요 Top 10 노선
ax4 = fig.add_subplot(gs[1, 0])
top10_demand = df_bus.nlargest(10, '추정수요')
ax4.barh(range(len(top10_demand)), top10_demand['추정수요'].values/1e3,
         color=COLORS['info'], edgecolor=COLORS['dark'], linewidth=1.5, alpha=0.85)
ax4.set_yticks(range(len(top10_demand)))
ax4.set_yticklabels([f"{row['구분']} {row['노선번호']}" for _, row in top10_demand.iterrows()], fontsize=9)
ax4.set_xlabel('추정 수요 (천명)', fontsize=11, fontweight='bold')
ax4.set_title('📈 수요 Top 10 노선', fontsize=14, fontweight='bold', pad=15)
ax4.invert_yaxis()
ax4.grid(axis='x', alpha=0.3, linestyle='--')

# 5. 배차간격 분포
ax5 = fig.add_subplot(gs[1, 1])
intervals = df_bus[df_bus['배차간격_분'].notna()]['배차간격_분']
ax5.hist(intervals, bins=20, color=COLORS['secondary'], edgecolor=COLORS['dark'],
         linewidth=1.5, alpha=0.8)
ax5.axvline(intervals.mean(), color=COLORS['danger'], linestyle='--', linewidth=2,
            label=f'평균: {intervals.mean():.1f}분')
ax5.set_xlabel('배차간격 (분)', fontsize=11, fontweight='bold')
ax5.set_ylabel('노선 수', fontsize=11, fontweight='bold')
ax5.set_title('⏱️ 배차간격 분포', fontsize=14, fontweight='bold', pad=15)
ax5.legend()
ax5.grid(axis='y', alpha=0.3, linestyle='--')

# 6. 수요 vs 운행횟수
ax6 = fig.add_subplot(gs[2, 0])
scatter_colors = [COLORS['danger'] if x > 0.5 else COLORS['warning'] if x > 0.2 else COLORS['success']
                  for x in df_bus['최적화우선순위'].values]
ax6.scatter(df_bus['추정수요']/1e3, df_bus['운행횟수_일'],
           s=120, c=scatter_colors, alpha=0.7, edgecolors=COLORS['dark'], linewidth=1.5)
ax6.set_xlabel('추정 수요 (천명)', fontsize=11, fontweight='bold')
ax6.set_ylabel('운행횟수 (회/일)', fontsize=11, fontweight='bold')
ax6.set_title('📊 수요 vs 운행횟수', fontsize=14, fontweight='bold', pad=15)
ax6.grid(True, alpha=0.3, linestyle='--')

# 범례
legend_elements = [
    plt.scatter([], [], s=100, c=COLORS['danger'], edgecolors=COLORS['dark'], label='최우선'),
    plt.scatter([], [], s=100, c=COLORS['warning'], edgecolors=COLORS['dark'], label='필요'),
    plt.scatter([], [], s=100, c=COLORS['success'], edgecolors=COLORS['dark'], label='양호')
]
ax6.legend(handles=legend_elements, loc='upper left', title='최적화 우선순위')

# 7. 커버 지역 수 분포
ax7 = fig.add_subplot(gs[2, 1])
coverage = df_bus['커버지역수'].value_counts().sort_index()
ax7.bar(coverage.index, coverage.values, color=COLORS['accent'],
        edgecolor=COLORS['dark'], linewidth=1.5, alpha=0.85, width=0.6)
ax7.set_xlabel('커버 지역 수', fontsize=11, fontweight='bold')
ax7.set_ylabel('노선 수', fontsize=11, fontweight='bold')
ax7.set_title('🗺️ 노선별 커버 지역 수', fontsize=14, fontweight='bold', pad=15)
ax7.grid(axis='y', alpha=0.3, linestyle='--')
for i, val in enumerate(coverage.values):
    ax7.text(coverage.index[i], val + 0.5, str(val), ha='center', fontsize=10, fontweight='bold')

# 8. 노선 유형별 평균 지표
ax8 = fig.add_subplot(gs[2, 2])
route_metrics = df_bus.groupby('구분').agg({
    '배차간격_분': 'mean',
    '운행횟수_일': 'mean',
    '추정수요': 'mean'
}).reset_index()

x = np.arange(len(route_metrics))
width = 0.25

bars1 = ax8.bar(x - width, route_metrics['배차간격_분'], width, label='배차간격(분)',
                color=COLORS['info'], edgecolor=COLORS['dark'], linewidth=1)
bars2 = ax8.bar(x, route_metrics['운행횟수_일']/2, width, label='운행횟수(/2)',
                color=COLORS['secondary'], edgecolor=COLORS['dark'], linewidth=1)
bars3 = ax8.bar(x + width, route_metrics['추정수요']/1e4, width, label='수요(/1만)',
                color=COLORS['accent'], edgecolor=COLORS['dark'], linewidth=1)

ax8.set_xticks(x)
ax8.set_xticklabels(route_metrics['구분'], fontsize=10)
ax8.set_ylabel('값 (정규화)', fontsize=11, fontweight='bold')
ax8.set_title('📊 노선 유형별 평균 지표', fontsize=14, fontweight='bold', pad=15)
ax8.legend(fontsize=9)
ax8.grid(axis='y', alpha=0.3, linestyle='--')

plt.savefig('버스노선_최적화_시각화.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✓ 저장: 버스노선_최적화_시각화.png")
plt.close()

# ============================================================================
# 9. 결과 저장
# ============================================================================
print("\n[9단계] 결과 저장")
print("-"*100)

# CSV 저장
df_bus_sorted.to_csv('버스노선_최적화_결과.csv', index=False, encoding='utf-8-sig')
print("✓ 저장: 버스노선_최적화_결과.csv")

# 최우선 노선만 별도 저장
df_priority = df_bus_sorted[df_bus_sorted['최적화우선순위'] > 0.5][
    ['구분', '노선번호', '기점지', '경유지', '종점지', '배차간격_분', '운행횟수_일',
     '커버지역수', '추정수요', '배차적정성', '최적화제안', '최적화우선순위']
]
df_priority.to_csv('버스노선_최우선_최적화.csv', index=False, encoding='utf-8-sig')
print("✓ 저장: 버스노선_최우선_최적화.csv")

# JSON 리포트
report = {
    '분석일시': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    '총_노선수': int(total_routes),
    '분석_지역수': int(len(df_regions)),
    '총_이용객': int(region_stats['총_이용객'].sum()),
    '평균_배차간격_분': float(df_bus['배차간격_분'].mean()),
    '평균_운행횟수_일': float(df_bus['운행횟수_일'].mean()),
    '최적화_최우선_노선수': int(high_priority_count),
    '최적화_필요_노선수': int(medium_priority_count),
    '양호_노선수': int(good_count),
    '배차적정성_평가': {
        '적정': int(len(df_bus[df_bus['배차적정성']=='적정'])),
        '부족': int(len(df_bus[df_bus['배차적정성']=='부족'])),
        '심각부족': int(len(df_bus[df_bus['배차적정성']=='심각부족'])),
        '정보없음': int(len(df_bus[df_bus['배차적정성']=='정보없음']))
    },
    '개선가능_수요': int(df_bus[df_bus['최적화우선순위'] > 0.2]['추정수요'].sum())
}

with open('버스노선_최적화_리포트.json', 'w', encoding='utf-8') as f:
    json.dump(report, f, ensure_ascii=False, indent=2)
print("✓ 저장: 버스노선_최적화_리포트.json")

# ============================================================================
# 완료
# ============================================================================
print("\n" + "="*100)
print("✅ 버스 노선 최적화 분석 완료!".center(100))
print("="*100)
print("\n📁 생성된 파일:")
print("  1. 버스노선_최적화_지도.html - 인터랙티브 지도")
print("  2. 버스노선_최적화_시각화.png - 종합 시각화")
print("  3. 버스노선_최적화_결과.csv - 전체 노선 분석 결과")
print("  4. 버스노선_최우선_최적화.csv - 최우선 최적화 노선")
print("  5. 버스노선_최적화_리포트.json - 종합 리포트")
print(f"\n⏰ 분석 종료: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*100)
