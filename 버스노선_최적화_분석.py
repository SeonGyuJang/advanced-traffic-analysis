#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
세종시 버스 노선 최적화 종합 분석
================================================
버스 노선의 수요 대응력, 효율성, 최적화 방안을 분석합니다.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.gridspec import GridSpec
import seaborn as sns
import folium
from folium import plugins
from math import radians, cos, sin, asin, sqrt
import warnings
from datetime import datetime
import json
from collections import defaultdict

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
    'excellent': '#06A77D',
    'good': '#48C774',
    'fair': '#F4B41A',
    'poor': '#FF9800',
    'critical': '#D64933',
}

# ============================================================================
# 한글 폰트 설정
# ============================================================================
def setup_korean_font():
    """한글 폰트 설정"""
    for font in ['NanumGothic', 'NanumBarunGothic', 'Malgun Gothic', 'DejaVu Sans']:
        if font in [f.name for f in fm.fontManager.ttflist]:
            plt.rcParams['font.family'] = font
            plt.rcParams['axes.unicode_minus'] = False
            return True
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

def calculate_route_length(route_df):
    """노선 총 길이 계산 (km)"""
    if len(route_df) < 2:
        return 0

    total_length = 0
    for i in range(len(route_df) - 1):
        dist = haversine_distance(
            route_df.iloc[i]['위도'], route_df.iloc[i]['경도'],
            route_df.iloc[i+1]['위도'], route_df.iloc[i+1]['경도']
        )
        total_length += dist
    return total_length

# ============================================================================
# 데이터 로드 및 전처리
# ============================================================================
def load_data():
    """데이터 로드"""
    print("📊 데이터 로딩 중...")

    bus_stops = pd.read_csv('data/세종도시교통공사_버스정류장_시설현황_20210924.csv')
    demand = pd.read_csv('data/지역별승하차_통합데이터.csv')
    regions = pd.read_csv('data/행정구역_중심좌표.csv')

    # 수요 데이터 집계
    demand_summary = demand.groupby('행정구역').agg({
        '승차': 'sum',
        '하차': 'sum'
    }).reset_index()
    demand_summary['총수요'] = demand_summary['승차'] + demand_summary['하차']

    # 지역 데이터와 병합
    regions = regions.merge(demand_summary, on='행정구역', how='left')
    regions['총수요'] = regions['총수요'].fillna(0)

    print(f"✅ 버스 노선: {bus_stops['노선번호'].nunique()}개")
    print(f"✅ 정류장: {bus_stops['정류소ID'].nunique()}개")
    print(f"✅ 행정구역: {len(regions)}개")

    return bus_stops, regions, demand_summary

# ============================================================================
# 노선별 상세 분석
# ============================================================================
def analyze_route_details(bus_stops, regions):
    """노선별 상세 분석"""
    print("\n📈 노선별 상세 분석 중...")

    route_analysis = []

    for route_num in sorted(bus_stops['노선번호'].unique()):
        route_data = bus_stops[bus_stops['노선번호'] == route_num].copy()
        route_data = route_data.sort_values('연번')

        # 기본 정보
        num_stops = len(route_data)
        unique_stops = route_data['정류소ID'].nunique()
        route_length = calculate_route_length(route_data)

        # 커버하는 지역 계산 (정류장으로부터 500m 이내)
        covered_regions = set()
        region_coverage = {}

        for _, region in regions.iterrows():
            min_dist = float('inf')
            for _, stop in route_data.iterrows():
                dist = haversine_distance(
                    region['위도'], region['경도'],
                    stop['위도'], stop['경도']
                )
                min_dist = min(min_dist, dist)

            if min_dist <= 0.5:  # 500m 이내
                covered_regions.add(region['행정구역'])
                region_coverage[region['행정구역']] = {
                    'distance': min_dist,
                    'demand': region['총수요']
                }

        # 커버 지역의 총 수요
        total_covered_demand = sum(r['demand'] for r in region_coverage.values())

        # 효율성 지표
        stops_per_km = num_stops / route_length if route_length > 0 else 0
        demand_per_km = total_covered_demand / route_length if route_length > 0 else 0
        demand_per_stop = total_covered_demand / num_stops if num_stops > 0 else 0

        route_analysis.append({
            '노선번호': route_num,
            '정류장수': num_stops,
            '고유정류장수': unique_stops,
            '노선길이_km': round(route_length, 2),
            '커버지역수': len(covered_regions),
            '커버지역': ', '.join(sorted(covered_regions)),
            '총커버수요': int(total_covered_demand),
            '정류장밀도_per_km': round(stops_per_km, 2),
            '수요밀도_per_km': int(demand_per_km),
            '정류장당수요': int(demand_per_stop),
        })

    route_df = pd.DataFrame(route_analysis)

    # 효율성 등급 부여
    route_df['효율성등급'] = pd.cut(
        route_df['수요밀도_per_km'],
        bins=[0, 50000, 100000, 200000, float('inf')],
        labels=['낮음', '보통', '높음', '매우높음']
    )

    return route_df

# ============================================================================
# 지역별 노선 커버리지 분석
# ============================================================================
def analyze_region_coverage(bus_stops, regions):
    """지역별 노선 커버리지 분석"""
    print("\n🗺️  지역별 커버리지 분석 중...")

    region_analysis = []

    for _, region in regions.iterrows():
        region_name = region['행정구역']
        region_demand = region['총수요']

        # 이 지역을 지나는 노선 찾기 (500m 이내)
        serving_routes = set()
        min_distance = float('inf')
        nearest_stop = None

        for _, stop in bus_stops.iterrows():
            dist = haversine_distance(
                region['위도'], region['경도'],
                stop['위도'], stop['경도']
            )

            if dist < min_distance:
                min_distance = dist
                nearest_stop = stop['정류소명']

            if dist <= 0.5:  # 500m 이내
                serving_routes.add(stop['노선번호'])

        # 노선 수와 수요의 균형
        num_routes = len(serving_routes)
        demand_per_route = region_demand / num_routes if num_routes > 0 else region_demand

        # 서비스 수준 평가
        if num_routes == 0:
            service_level = '미커버'
        elif num_routes < 3:
            service_level = '부족'
        elif num_routes < 6:
            service_level = '적정'
        else:
            service_level = '과잉'

        region_analysis.append({
            '행정구역': region_name,
            '총수요': int(region_demand),
            '노선수': num_routes,
            '노선목록': ', '.join(sorted([str(r) for r in serving_routes])) if serving_routes else '없음',
            '수요_per_노선': int(demand_per_route),
            '최단거리_km': round(min_distance, 2),
            '최인접정류장': nearest_stop,
            '서비스수준': service_level
        })

    region_df = pd.DataFrame(region_analysis)
    region_df = region_df.sort_values('총수요', ascending=False)

    return region_df

# ============================================================================
# 노선 중복도 분석
# ============================================================================
def analyze_route_overlap(bus_stops):
    """노선 간 중복도 분석"""
    print("\n🔄 노선 중복도 분석 중...")

    # 각 정류장을 지나는 노선들
    stop_routes = bus_stops.groupby('정류소ID')['노선번호'].apply(set).to_dict()

    # 노선 쌍별 중복 정류장 수 계산
    route_list = sorted(bus_stops['노선번호'].unique())
    overlap_matrix = []

    for i, route1 in enumerate(route_list):
        row = []
        route1_stops = set(bus_stops[bus_stops['노선번호'] == route1]['정류소ID'])

        for route2 in route_list:
            route2_stops = set(bus_stops[bus_stops['노선번호'] == route2]['정류소ID'])
            overlap = len(route1_stops & route2_stops)
            row.append(overlap)

        overlap_matrix.append(row)

    overlap_df = pd.DataFrame(overlap_matrix, index=route_list, columns=route_list)

    # 높은 중복도 쌍 찾기 (자기 자신 제외)
    high_overlap_pairs = []
    for i, route1 in enumerate(route_list):
        for j, route2 in enumerate(route_list):
            if i < j:  # 중복 방지
                overlap = overlap_df.loc[route1, route2]
                if overlap >= 5:  # 5개 이상 공유
                    route1_total = len(bus_stops[bus_stops['노선번호'] == route1])
                    route2_total = len(bus_stops[bus_stops['노선번호'] == route2])
                    overlap_pct = overlap / min(route1_total, route2_total) * 100

                    high_overlap_pairs.append({
                        '노선1': route1,
                        '노선2': route2,
                        '공유정류장수': overlap,
                        '중복비율_%': round(overlap_pct, 1)
                    })

    overlap_pairs_df = pd.DataFrame(high_overlap_pairs)
    overlap_pairs_df = overlap_pairs_df.sort_values('공유정류장수', ascending=False)

    return overlap_df, overlap_pairs_df

# ============================================================================
# 최적화 제안 생성
# ============================================================================
def generate_optimization_recommendations(route_df, region_df):
    """노선 최적화 제안 생성"""
    print("\n💡 최적화 제안 생성 중...")

    recommendations = []

    # 1. 미커버 또는 서비스 부족 지역에 대한 제안
    underserved = region_df[region_df['서비스수준'].isin(['미커버', '부족'])]
    for _, region in underserved.iterrows():
        if region['총수요'] > 100000:  # 수요가 높은 지역만
            recommendations.append({
                '우선순위': 1,
                '유형': '노선추가',
                '대상': region['행정구역'],
                '현재상태': f"노선 {region['노선수']}개, 수요 {region['총수요']:,}",
                '제안사항': f"{region['행정구역']}에 추가 노선 배치 필요 (고수요 지역)",
                '예상효과': '미커버 지역 해소, 주민 접근성 향상'
            })

    # 2. 효율성이 낮은 노선에 대한 제안
    low_efficiency = route_df[
        (route_df['수요밀도_per_km'] < 50000) &
        (route_df['총커버수요'] < 500000)
    ]
    for _, route in low_efficiency.iterrows():
        recommendations.append({
            '우선순위': 2,
            '유형': '노선조정',
            '대상': f"노선 {route['노선번호']}",
            '현재상태': f"수요밀도 {route['수요밀도_per_km']:,}/km",
            '제안사항': f"경로 재조정 또는 고수요 지역 경유 추가",
            '예상효과': '노선 효율성 향상, 운영 비용 절감'
        })

    # 3. 과잉 서비스 지역에 대한 제안
    overserved = region_df[
        (region_df['서비스수준'] == '과잉') &
        (region_df['총수요'] < 1000000)
    ]
    for _, region in overserved.iterrows():
        recommendations.append({
            '우선순위': 3,
            '유형': '노선통합',
            '대상': region['행정구역'],
            '현재상태': f"노선 {region['노선수']}개, 수요 {region['총수요']:,}",
            '제안사항': f"일부 노선 통합 또는 배차 간격 조정",
            '예상효과': '중복 운행 감소, 운영 효율 향상'
        })

    # 4. 정류장이 과도하게 많은 노선
    dense_routes = route_df[route_df['정류장밀도_per_km'] > 20]
    for _, route in dense_routes.iterrows():
        recommendations.append({
            '우선순위': 4,
            '유형': '정류장최적화',
            '대상': f"노선 {route['노선번호']}",
            '현재상태': f"정류장 밀도 {route['정류장밀도_per_km']:.1f}/km",
            '제안사항': f"일부 정류장 통폐합 검토",
            '예상효과': '운행 시간 단축, 표정속도 향상'
        })

    recommendations_df = pd.DataFrame(recommendations)
    recommendations_df = recommendations_df.sort_values('우선순위')

    return recommendations_df

# ============================================================================
# 인터랙티브 지도 생성
# ============================================================================
def create_interactive_map(bus_stops, regions, route_df, region_df):
    """인터랙티브 HTML 지도 생성"""
    print("\n🗺️  인터랙티브 지도 생성 중...")

    # 지도 중심 (세종시)
    center_lat = regions['위도'].mean()
    center_lon = regions['경도'].mean()

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=11,
        tiles='OpenStreetMap'
    )

    # 노선별 색상 매핑
    route_colors = {}
    color_palette = [
        '#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231',
        '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe',
        '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000',
        '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080'
    ]

    for i, route in enumerate(sorted(bus_stops['노선번호'].unique())):
        route_colors[route] = color_palette[i % len(color_palette)]

    # 노선별 그룹 생성
    route_groups = {}
    for route_num in sorted(bus_stops['노선번호'].unique()):
        route_info = route_df[route_df['노선번호'] == route_num].iloc[0]
        group_name = f"노선 {route_num} ({route_info['정류장수']}개, {route_info['노선길이_km']}km)"
        route_groups[route_num] = folium.FeatureGroup(name=group_name, show=False)

    # 노선별 정류장과 경로 추가
    for route_num in sorted(bus_stops['노선번호'].unique()):
        route_data = bus_stops[bus_stops['노선번호'] == route_num].copy()
        route_data = route_data.sort_values('연번')
        route_info = route_df[route_df['노선번호'] == route_num].iloc[0]

        color = route_colors[route_num]

        # 경로선 그리기
        coordinates = [[row['위도'], row['경도']] for _, row in route_data.iterrows()]
        folium.PolyLine(
            coordinates,
            color=color,
            weight=3,
            opacity=0.7,
            popup=f"""
            <b>노선 {route_num}</b><br>
            정류장: {route_info['정류장수']}개<br>
            길이: {route_info['노선길이_km']}km<br>
            커버 지역: {route_info['커버지역수']}개<br>
            총 수요: {route_info['총커버수요']:,}<br>
            효율성: {route_info['효율성등급']}
            """
        ).add_to(route_groups[route_num])

        # 정류장 마커 (첫 정류장과 마지막 정류장만 표시)
        first_stop = route_data.iloc[0]
        last_stop = route_data.iloc[-1]

        folium.CircleMarker(
            location=[first_stop['위도'], first_stop['경도']],
            radius=6,
            color=color,
            fill=True,
            fillColor=color,
            fillOpacity=0.8,
            popup=f"<b>{first_stop['정류소명']}</b><br>노선 {route_num} 시점"
        ).add_to(route_groups[route_num])

        folium.CircleMarker(
            location=[last_stop['위도'], last_stop['경도']],
            radius=6,
            color=color,
            fill=True,
            fillColor='white',
            fillOpacity=0.8,
            popup=f"<b>{last_stop['정류소명']}</b><br>노선 {route_num} 종점"
        ).add_to(route_groups[route_num])

    # 지역 중심점 및 수요 표시
    demand_group = folium.FeatureGroup(name="지역별 수요", show=True)

    for _, region in regions.iterrows():
        region_info = region_df[region_df['행정구역'] == region['행정구역']].iloc[0]

        # 서비스 수준에 따른 색상
        service_colors = {
            '미커버': '#D64933',
            '부족': '#F4B41A',
            '적정': '#06A77D',
            '과잉': '#5C7CFA'
        }
        color = service_colors.get(region_info['서비스수준'], '#808080')

        # 수요에 따른 크기
        radius = min(max(region['총수요'] / 100000, 3), 20)

        folium.CircleMarker(
            location=[region['위도'], region['경도']],
            radius=radius,
            color=color,
            fill=True,
            fillColor=color,
            fillOpacity=0.4,
            popup=f"""
            <b>{region['행정구역']}</b><br>
            총 수요: {region['총수요']:,}<br>
            노선 수: {region_info['노선수']}개<br>
            노선: {region_info['노선목록']}<br>
            서비스 수준: {region_info['서비스수준']}<br>
            최단 거리: {region_info['최단거리_km']}km
            """
        ).add_to(demand_group)

        # 지역명 라벨
        folium.Marker(
            location=[region['위도'], region['경도']],
            icon=folium.DivIcon(html=f"""
                <div style="font-size: 10pt; color: black; font-weight: bold;
                            text-shadow: 1px 1px 2px white;">
                    {region['행정구역']}
                </div>
            """)
        ).add_to(demand_group)

    # 그룹들을 지도에 추가
    demand_group.add_to(m)
    for group in route_groups.values():
        group.add_to(m)

    # 레이어 컨트롤 추가
    folium.LayerControl(collapsed=False).add_to(m)

    # 범례 추가
    legend_html = '''
    <div style="position: fixed;
                bottom: 50px; right: 50px; width: 250px;
                border:2px solid grey; z-index:9999;
                background-color:white; opacity: 0.9;
                padding: 10px; font-size: 12px;
                border-radius: 5px;">
    <p style="margin: 0; font-weight: bold; font-size: 14px;">범례</p>
    <p style="margin: 5px 0;"><span style="color: #D64933;">●</span> 미커버 지역</p>
    <p style="margin: 5px 0;"><span style="color: #F4B41A;">●</span> 부족 (노선 &lt;3개)</p>
    <p style="margin: 5px 0;"><span style="color: #06A77D;">●</span> 적정 (노선 3-5개)</p>
    <p style="margin: 5px 0;"><span style="color: #5C7CFA;">●</span> 과잉 (노선 6개+)</p>
    <p style="margin: 10px 0 5px 0; font-size: 11px;">
        원의 크기: 지역별 총 수요<br>
        노선을 선택하면 경로와 정류장을 볼 수 있습니다
    </p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))

    # 저장
    output_file = '버스노선_최적화_인터랙티브맵.html'
    m.save(output_file)
    print(f"✅ 인터랙티브 지도 저장: {output_file}")

    return m

# ============================================================================
# 시각화
# ============================================================================
def create_visualizations(route_df, region_df, overlap_pairs_df):
    """종합 시각화 생성"""
    print("\n📊 시각화 생성 중...")

    setup_korean_font()

    fig = plt.figure(figsize=(20, 24))
    gs = GridSpec(6, 2, figure=fig, hspace=0.3, wspace=0.3)

    # 1. 노선별 효율성 (수요밀도)
    ax1 = fig.add_subplot(gs[0, :])
    top_routes = route_df.nlargest(15, '수요밀도_per_km')
    colors_eff = [COLORS['excellent'] if x > 200000 else
                  COLORS['good'] if x > 100000 else
                  COLORS['fair'] if x > 50000 else
                  COLORS['poor'] for x in top_routes['수요밀도_per_km']]

    ax1.barh(top_routes['노선번호'].astype(str), top_routes['수요밀도_per_km'], color=colors_eff)
    ax1.set_xlabel('수요 밀도 (명/km)', fontsize=12, fontweight='bold')
    ax1.set_title('노선별 효율성: 수요 밀도 Top 15', fontsize=14, fontweight='bold', pad=20)
    ax1.grid(axis='x', alpha=0.3)
    for i, v in enumerate(top_routes['수요밀도_per_km']):
        ax1.text(v, i, f' {v:,.0f}', va='center', fontsize=10)

    # 2. 지역별 서비스 수준
    ax2 = fig.add_subplot(gs[1, 0])
    service_counts = region_df['서비스수준'].value_counts()
    service_colors_map = {
        '미커버': COLORS['danger'],
        '부족': COLORS['warning'],
        '적정': COLORS['success'],
        '과잉': COLORS['info']
    }
    colors_service = [service_colors_map.get(x, '#808080') for x in service_counts.index]

    ax2.pie(service_counts.values, labels=service_counts.index, autopct='%1.1f%%',
            colors=colors_service, startangle=90)
    ax2.set_title('지역별 서비스 수준 분포', fontsize=13, fontweight='bold', pad=15)

    # 3. 노선 길이 vs 수요
    ax3 = fig.add_subplot(gs[1, 1])
    scatter = ax3.scatter(route_df['노선길이_km'], route_df['총커버수요']/1000,
                         c=route_df['효율성등급'].cat.codes, cmap='RdYlGn',
                         s=100, alpha=0.6, edgecolors='black', linewidth=0.5)
    ax3.set_xlabel('노선 길이 (km)', fontsize=11, fontweight='bold')
    ax3.set_ylabel('총 커버 수요 (천명)', fontsize=11, fontweight='bold')
    ax3.set_title('노선 길이 vs 총 수요', fontsize=13, fontweight='bold', pad=15)
    ax3.grid(alpha=0.3)

    # 상위 5개 노선 라벨링
    top5 = route_df.nlargest(5, '총커버수요')
    for _, route in top5.iterrows():
        ax3.annotate(route['노선번호'],
                    (route['노선길이_km'], route['총커버수요']/1000),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=9, fontweight='bold')

    # 4. 지역별 수요 Top 15
    ax4 = fig.add_subplot(gs[2, :])
    top_demand = region_df.nlargest(15, '총수요')
    colors_demand = [COLORS['danger'] if x == '미커버' else
                    COLORS['warning'] if x == '부족' else
                    COLORS['success'] if x == '적정' else
                    COLORS['info'] for x in top_demand['서비스수준']]

    ax4.barh(top_demand['행정구역'], top_demand['총수요']/1000, color=colors_demand)
    ax4.set_xlabel('총 수요 (천명)', fontsize=12, fontweight='bold')
    ax4.set_title('지역별 총 수요 Top 15 (색상: 서비스 수준)', fontsize=14, fontweight='bold', pad=20)
    ax4.grid(axis='x', alpha=0.3)

    for i, (demand, routes) in enumerate(zip(top_demand['총수요']/1000, top_demand['노선수'])):
        ax4.text(demand, i, f' {demand:.0f}천 ({routes}개 노선)', va='center', fontsize=9)

    # 5. 노선당 평균 수요 (지역별)
    ax5 = fig.add_subplot(gs[3, 0])
    top_ratio = region_df[region_df['노선수'] > 0].nlargest(12, '수요_per_노선')
    ax5.barh(top_ratio['행정구역'], top_ratio['수요_per_노선']/1000, color=COLORS['secondary'])
    ax5.set_xlabel('노선당 수요 (천명)', fontsize=11, fontweight='bold')
    ax5.set_title('노선당 평균 수요 Top 12', fontsize=13, fontweight='bold', pad=15)
    ax5.grid(axis='x', alpha=0.3)

    # 6. 노선 중복도 Top 10
    ax6 = fig.add_subplot(gs[3, 1])
    if len(overlap_pairs_df) > 0:
        top_overlap = overlap_pairs_df.head(10)
        labels = [f"{r['노선1']}-{r['노선2']}" for _, r in top_overlap.iterrows()]
        ax6.barh(labels, top_overlap['공유정류장수'], color=COLORS['accent'])
        ax6.set_xlabel('공유 정류장 수', fontsize=11, fontweight='bold')
        ax6.set_title('노선 중복도 Top 10', fontsize=13, fontweight='bold', pad=15)
        ax6.grid(axis='x', alpha=0.3)

        for i, (stops, pct) in enumerate(zip(top_overlap['공유정류장수'], top_overlap['중복비율_%'])):
            ax6.text(stops, i, f' {stops}개 ({pct}%)', va='center', fontsize=9)

    # 7. 정류장 밀도 분포
    ax7 = fig.add_subplot(gs[4, 0])
    ax7.hist(route_df['정류장밀도_per_km'], bins=20, color=COLORS['info'], alpha=0.7, edgecolor='black')
    ax7.axvline(route_df['정류장밀도_per_km'].median(), color=COLORS['danger'],
               linestyle='--', linewidth=2, label=f"중앙값: {route_df['정류장밀도_per_km'].median():.1f}")
    ax7.set_xlabel('정류장 밀도 (개/km)', fontsize=11, fontweight='bold')
    ax7.set_ylabel('노선 수', fontsize=11, fontweight='bold')
    ax7.set_title('노선별 정류장 밀도 분포', fontsize=13, fontweight='bold', pad=15)
    ax7.legend()
    ax7.grid(alpha=0.3)

    # 8. 커버 지역 수 분포
    ax8 = fig.add_subplot(gs[4, 1])
    coverage_dist = route_df['커버지역수'].value_counts().sort_index()
    ax8.bar(coverage_dist.index, coverage_dist.values, color=COLORS['success'], alpha=0.7, edgecolor='black')
    ax8.set_xlabel('커버 지역 수', fontsize=11, fontweight='bold')
    ax8.set_ylabel('노선 수', fontsize=11, fontweight='bold')
    ax8.set_title('노선별 커버 지역 수 분포', fontsize=13, fontweight='bold', pad=15)
    ax8.grid(axis='y', alpha=0.3)

    # 9. 효율성 등급별 노선 수
    ax9 = fig.add_subplot(gs[5, 0])
    efficiency_counts = route_df['효율성등급'].value_counts()
    colors_efficiency = [COLORS['excellent'], COLORS['good'], COLORS['fair'], COLORS['poor']][:len(efficiency_counts)]
    ax9.bar(efficiency_counts.index, efficiency_counts.values, color=colors_efficiency, alpha=0.7, edgecolor='black')
    ax9.set_ylabel('노선 수', fontsize=11, fontweight='bold')
    ax9.set_title('효율성 등급별 노선 분포', fontsize=13, fontweight='bold', pad=15)
    ax9.grid(axis='y', alpha=0.3)

    # 10. 핵심 통계 요약
    ax10 = fig.add_subplot(gs[5, 1])
    ax10.axis('off')

    summary_text = f"""
    ━━━━━━━━━━━━━━━━━━━━━━━
    📊 핵심 통계 요약
    ━━━━━━━━━━━━━━━━━━━━━━━

    🚌 노선 현황
      • 총 노선 수: {len(route_df)}개
      • 평균 노선 길이: {route_df['노선길이_km'].mean():.1f}km
      • 평균 정류장 수: {route_df['정류장수'].mean():.1f}개

    🗺️ 지역 커버리지
      • 미커버 지역: {len(region_df[region_df['서비스수준']=='미커버'])}개
      • 부족 지역: {len(region_df[region_df['서비스수준']=='부족'])}개
      • 적정 지역: {len(region_df[region_df['서비스수준']=='적정'])}개
      • 과잉 지역: {len(region_df[region_df['서비스수준']=='과잉'])}개

    📈 효율성
      • 평균 수요밀도: {route_df['수요밀도_per_km'].mean():,.0f}명/km
      • 최고 효율 노선: {route_df.loc[route_df['수요밀도_per_km'].idxmax(), '노선번호']}
      • 개선 필요 노선: {len(route_df[route_df['효율성등급']=='낮음'])}개
    """

    ax10.text(0.1, 0.5, summary_text, fontsize=11, verticalalignment='center',
             fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.suptitle('세종시 버스 노선 최적화 종합 분석',
                fontsize=18, fontweight='bold', y=0.995)

    plt.savefig('버스노선_최적화_종합분석.png', dpi=300, bbox_inches='tight')
    print("✅ 시각화 저장: 버스노선_최적화_종합분석.png")
    plt.close()

# ============================================================================
# 메인 실행
# ============================================================================
def main():
    print("="*80)
    print("🚍 세종시 버스 노선 최적화 종합 분석 시스템")
    print("="*80)

    # 데이터 로드
    bus_stops, regions, demand_summary = load_data()

    # 노선별 분석
    route_df = analyze_route_details(bus_stops, regions)
    print("\n" + "="*80)
    print("📊 노선별 효율성 Top 10")
    print("="*80)
    print(route_df.nlargest(10, '수요밀도_per_km')[
        ['노선번호', '정류장수', '노선길이_km', '커버지역수',
         '총커버수요', '수요밀도_per_km', '효율성등급']
    ].to_string(index=False))

    # 지역별 분석
    region_df = analyze_region_coverage(bus_stops, regions)
    print("\n" + "="*80)
    print("🗺️ 지역별 서비스 수준 (고수요 지역 우선)")
    print("="*80)
    print(region_df.head(15)[
        ['행정구역', '총수요', '노선수', '서비스수준', '최단거리_km']
    ].to_string(index=False))

    # 노선 중복도 분석
    overlap_matrix, overlap_pairs_df = analyze_route_overlap(bus_stops)
    if len(overlap_pairs_df) > 0:
        print("\n" + "="*80)
        print("🔄 노선 중복도 Top 10")
        print("="*80)
        print(overlap_pairs_df.head(10).to_string(index=False))

    # 최적화 제안
    recommendations_df = generate_optimization_recommendations(route_df, region_df)
    print("\n" + "="*80)
    print("💡 주요 최적화 제안")
    print("="*80)
    print(recommendations_df.head(15)[
        ['우선순위', '유형', '대상', '제안사항']
    ].to_string(index=False))

    # 결과 저장
    print("\n" + "="*80)
    print("💾 결과 저장 중...")
    print("="*80)

    route_df.to_csv('버스노선_분석결과.csv', index=False, encoding='utf-8-sig')
    print("✅ 노선 분석 결과: 버스노선_분석결과.csv")

    region_df.to_csv('지역별_서비스수준.csv', index=False, encoding='utf-8-sig')
    print("✅ 지역 분석 결과: 지역별_서비스수준.csv")

    recommendations_df.to_csv('노선_최적화_제안.csv', index=False, encoding='utf-8-sig')
    print("✅ 최적화 제안: 노선_최적화_제안.csv")

    if len(overlap_pairs_df) > 0:
        overlap_pairs_df.to_csv('노선_중복도_분석.csv', index=False, encoding='utf-8-sig')
        print("✅ 중복도 분석: 노선_중복도_분석.csv")

    # 종합 보고서
    report = {
        'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'total_routes': len(route_df),
        'total_regions': len(region_df),
        'underserved_regions': len(region_df[region_df['서비스수준'].isin(['미커버', '부족'])]),
        'optimization_recommendations': len(recommendations_df),
        'high_efficiency_routes': len(route_df[route_df['효율성등급'].isin(['높음', '매우높음'])]),
        'low_efficiency_routes': len(route_df[route_df['효율성등급'] == '낮음']),
    }

    with open('버스노선_최적화_보고서.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("✅ 종합 보고서: 버스노선_최적화_보고서.json")

    # 인터랙티브 지도 생성
    create_interactive_map(bus_stops, regions, route_df, region_df)

    # 시각화 생성
    create_visualizations(route_df, region_df, overlap_pairs_df)

    print("\n" + "="*80)
    print("✅ 분석 완료!")
    print("="*80)
    print("\n📁 생성된 파일:")
    print("  • 버스노선_분석결과.csv")
    print("  • 지역별_서비스수준.csv")
    print("  • 노선_최적화_제안.csv")
    print("  • 노선_중복도_분석.csv")
    print("  • 버스노선_최적화_보고서.json")
    print("  • 버스노선_최적화_인터랙티브맵.html")
    print("  • 버스노선_최적화_종합분석.png")

if __name__ == '__main__':
    main()
