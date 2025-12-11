#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
세종시 버스정류장 최적화 분석 (정수계획법)
=========================================
정류장별 수요 데이터 기반 최적 신규 정류장 위치 선정
기간별 분석 지원 및 고도화된 대시보드 제공
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from pulp import *
from math import radians, cos, sin, asin, sqrt
from datetime import datetime
import warnings
import json

warnings.filterwarnings('ignore')

# ============================================================================
# 설정 및 상수
# ============================================================================
COLORS = {
    'primary': '#2E4057',
    'secondary': '#048A81',
    'accent': '#F26419',
    'success': '#06A77D',
    'warning': '#F4B41A',
    'danger': '#D64933',
    'existing': '#9C27B0',
    'new': '#FF5722',
    'demand_high': '#D32F2F',
    'demand_mid': '#FFA726',
    'demand_low': '#66BB6A',
}

COVERAGE_RADIUS = 0.5  # km
MAX_NEW_STATIONS = 10

# ============================================================================
# 유틸리티 함수
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

def get_region_from_coords(lat, lon, region_coords):
    """좌표로부터 가장 가까운 행정구역 찾기"""
    min_dist = float('inf')
    nearest_region = None

    for _, region in region_coords.iterrows():
        dist = haversine_distance(lat, lon, region['위도'], region['경도'])
        if dist < min_dist:
            min_dist = dist
            nearest_region = region['행정구역']

    return nearest_region

# ============================================================================
# 데이터 로드 및 전처리
# ============================================================================
class BusStationOptimizer:
    def __init__(self, start_date=None, end_date=None):
        """
        초기화

        Parameters:
        -----------
        start_date : str, optional
            분석 시작일 (YYYY-MM-DD)
        end_date : str, optional
            분석 종료일 (YYYY-MM-DD)
        """
        self.start_date = start_date
        self.end_date = end_date
        self.load_data()

    def load_data(self):
        """데이터 로드"""
        print("="*100)
        print("데이터 로드 중...".center(100))
        print("="*100)

        # 기존 정류장 데이터
        self.df_stations = pd.read_csv('data/세종도시교통공사_버스정류장_시설현황_20210924.csv')
        self.df_stations = self.df_stations.drop_duplicates(subset=['정류소ID'])

        # 승하차 데이터
        self.df_passenger = pd.read_csv('data/지역별승하차_통합데이터.csv')
        self.df_passenger['날짜'] = pd.to_datetime(self.df_passenger['날짜'])

        # 행정구역 좌표
        self.df_region_coords = pd.read_csv('data/행정구역_중심좌표.csv')

        # 기간 필터링
        if self.start_date:
            self.df_passenger = self.df_passenger[
                self.df_passenger['날짜'] >= pd.to_datetime(self.start_date)
            ]
        if self.end_date:
            self.df_passenger = self.df_passenger[
                self.df_passenger['날짜'] <= pd.to_datetime(self.end_date)
            ]

        print(f"✓ 기존 정류장: {len(self.df_stations):,}개")
        print(f"✓ 승하차 데이터: {len(self.df_passenger):,}건")
        if self.start_date or self.end_date:
            print(f"✓ 분석 기간: {self.start_date or '시작'} ~ {self.end_date or '현재'}")
        print()

    def create_station_demand_data(self):
        """정류장별 수요 데이터 생성"""
        print("\n정류장별 수요 데이터 생성 중...")
        print("-"*100)

        # 정류장에 행정구역 매핑
        self.df_stations['행정구역'] = self.df_stations.apply(
            lambda row: get_region_from_coords(
                row['위도'], row['경도'], self.df_region_coords
            ),
            axis=1
        )

        # 행정구역별 총 수요 계산
        region_demand = self.df_passenger.groupby('행정구역').agg({
            '승차': 'sum',
            '하차': 'sum',
            '환승': 'sum'
        }).reset_index()

        region_demand['총_이용객'] = region_demand['승차'] + region_demand['하차']

        # 행정구역별 정류장 수 계산
        stations_per_region = self.df_stations.groupby('행정구역').size().reset_index(name='정류장수')

        # 병합
        region_demand = region_demand.merge(stations_per_region, on='행정구역', how='left')
        region_demand['정류장수'] = region_demand['정류장수'].fillna(1)

        # 정류장별 수요 배분 (행정구역 수요를 정류장 수로 나눔)
        region_demand['정류장당_승차'] = region_demand['승차'] / region_demand['정류장수']
        region_demand['정류장당_하차'] = region_demand['하차'] / region_demand['정류장수']
        region_demand['정류장당_환승'] = region_demand['환승'] / region_demand['정류장수']
        region_demand['정류장당_총이용객'] = region_demand['총_이용객'] / region_demand['정류장수']

        # 정류장 데이터에 수요 매핑
        self.df_stations = self.df_stations.merge(
            region_demand[['행정구역', '정류장당_승차', '정류장당_하차', '정류장당_환승', '정류장당_총이용객']],
            on='행정구역',
            how='left'
        )

        # 결측치 처리
        self.df_stations['정류장당_승차'] = self.df_stations['정류장당_승차'].fillna(0)
        self.df_stations['정류장당_하차'] = self.df_stations['정류장당_하차'].fillna(0)
        self.df_stations['정류장당_환승'] = self.df_stations['정류장당_환승'].fillna(0)
        self.df_stations['정류장당_총이용객'] = self.df_stations['정류장당_총이용객'].fillna(0)

        print(f"✓ 정류장별 수요 데이터 생성 완료")
        print(f"  - 평균 정류장당 일일 이용객: {self.df_stations['정류장당_총이용객'].mean():.1f}명")
        print(f"  - 최대 정류장당 일일 이용객: {self.df_stations['정류장당_총이용객'].max():.1f}명")
        print()

        return self.df_stations

    def identify_underserved_areas(self, coverage_radius=COVERAGE_RADIUS):
        """서비스가 부족한 지역 식별"""
        print("\n서비스 부족 지역 식별 중...")
        print("-"*100)

        # 행정구역별 수요 및 좌표
        region_demand = self.df_passenger.groupby('행정구역').agg({
            '승차': 'sum',
            '하차': 'sum',
            '환승': 'sum'
        }).reset_index()

        region_demand['총_이용객'] = region_demand['승차'] + region_demand['하차']

        # 좌표 병합
        region_demand = region_demand.merge(self.df_region_coords, on='행정구역', how='left')

        # 각 행정구역의 최근접 정류장까지 거리 계산
        region_demand['최단거리_km'] = region_demand.apply(
            lambda row: min([
                haversine_distance(row['위도'], row['경도'], st['위도'], st['경도'])
                for _, st in self.df_stations.iterrows()
            ]) if len(self.df_stations) > 0 else float('inf'),
            axis=1
        )

        # 서비스 부족 지역 판별
        region_demand['서비스부족'] = region_demand['최단거리_km'] > coverage_radius

        underserved = region_demand[region_demand['서비스부족']].copy()

        print(f"✓ 서비스 부족 지역: {len(underserved)}개")
        if len(underserved) > 0:
            print(f"  - 총 미커버 수요: {underserved['총_이용객'].sum():,.0f}명")
            print(f"  - 평균 거리: {underserved['최단거리_km'].mean():.2f}km")
        print()

        self.underserved_regions = underserved
        return underserved

    def optimize_new_stations(self, max_new_stations=MAX_NEW_STATIONS,
                             coverage_radius=COVERAGE_RADIUS):
        """정수계획법을 통한 최적 신규 정류장 위치 선정"""
        print("\n정수계획법 최적화 실행 중...")
        print("-"*100)

        if not hasattr(self, 'underserved_regions') or len(self.underserved_regions) == 0:
            print("⚠ 서비스 부족 지역이 없습니다.")
            return pd.DataFrame()

        candidates = self.underserved_regions.copy()
        n = len(candidates)

        if n == 0:
            return pd.DataFrame()

        # 거리 행렬 계산
        print("  - 거리 행렬 계산 중...")
        distance_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                dist = haversine_distance(
                    candidates.iloc[i]['위도'], candidates.iloc[i]['경도'],
                    candidates.iloc[j]['위도'], candidates.iloc[j]['경도']
                )
                distance_matrix[i, j] = distance_matrix[j, i] = dist

        # 수요 가중치 계산
        print("  - 수요 가중치 계산 중...")
        demand = candidates['총_이용객'].values
        demand_norm = (demand - demand.min()) / (demand.max() - demand.min() + 1e-10)

        transfer = candidates['환승'].values
        transfer_norm = (transfer - transfer.min()) / (transfer.max() - transfer.min() + 1e-10)

        distance = candidates['최단거리_km'].values
        distance_norm = (distance - distance.min()) / (distance.max() - distance.min() + 1e-10)

        # 종합 가중치: 수요(60%) + 환승(20%) + 거리(20%)
        weights = 0.60 * demand_norm + 0.20 * transfer_norm + 0.20 * distance_norm

        # 커버리지 행렬
        coverage_matrix = (distance_matrix <= coverage_radius).astype(int)
        np.fill_diagonal(coverage_matrix, 1)

        # 정수계획법 모델 구성
        print("  - 정수계획법 모델 생성 중...")
        prob = LpProblem("Bus_Station_Optimization", LpMaximize)

        # 결정 변수
        x = LpVariable.dicts("station", range(n), cat='Binary')  # 정류장 선택 여부
        y = LpVariable.dicts("covered", range(n), cat='Binary')  # 지역 커버 여부

        # 목적함수: 가중 수요 최대화
        prob += lpSum([demand[i] * weights[i] * y[i] for i in range(n)]), "Weighted_Demand"

        # 제약조건 1: 신규 정류장 개수 제한
        prob += lpSum([x[i] for i in range(n)]) <= max_new_stations, "Max_Stations"

        # 제약조건 2: 커버리지 (지역이 커버되려면 근처에 정류장이 있어야 함)
        for i in range(n):
            prob += y[i] <= lpSum([coverage_matrix[i][j] * x[j] for j in range(n)]), f"Coverage_{i}"

        # 최적화 실행
        print("  - 최적화 실행 중...")
        prob.solve(PULP_CBC_CMD(msg=0))

        # 결과 추출
        if LpStatus[prob.status] == 'Optimal':
            selected_indices = [i for i in range(n) if x[i].varValue == 1]
            optimized_stations = candidates.iloc[selected_indices].copy()

            # 각 신규 정류장이 커버하는 수요 계산
            optimized_stations['커버_수요'] = [
                sum([demand[j] for j in range(n) if coverage_matrix[i][j] == 1])
                for i in selected_indices
            ]

            optimized_stations['우선순위'] = range(1, len(optimized_stations) + 1)

            print(f"\n✓ 최적화 완료!")
            print(f"  - 선정된 신규 정류장: {len(optimized_stations)}개")
            print(f"  - 총 커버 수요: {optimized_stations['총_이용객'].sum():,.0f}명")
            print(f"  - 목적함수 값: {value(prob.objective):,.2f}")
            print()

            self.optimized_stations = optimized_stations
            return optimized_stations
        else:
            print(f"⚠ 최적화 실패: {LpStatus[prob.status]}")
            return pd.DataFrame()

    def generate_report(self):
        """분석 보고서 생성"""
        report = {
            '분석일시': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            '분석기간': {
                '시작': self.start_date or '전체',
                '종료': self.end_date or '현재'
            },
            '기존_정류장': {
                '개수': int(len(self.df_stations)),
                '총_수요': int(self.df_stations['정류장당_총이용객'].sum()),
                '평균_수요': float(self.df_stations['정류장당_총이용객'].mean())
            },
            '서비스_부족_지역': {
                '개수': int(len(self.underserved_regions)) if hasattr(self, 'underserved_regions') else 0,
                '총_수요': int(self.underserved_regions['총_이용객'].sum()) if hasattr(self, 'underserved_regions') else 0
            },
            '최적화_결과': {
                '신규_정류장_개수': int(len(self.optimized_stations)) if hasattr(self, 'optimized_stations') else 0,
                '커버_수요': int(self.optimized_stations['총_이용객'].sum()) if hasattr(self, 'optimized_stations') else 0
            }
        }

        with open('최적화_분석_보고서.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        print("✓ 분석 보고서 생성 완료: 최적화_분석_보고서.json")
        return report

    def save_results(self):
        """결과 저장"""
        if hasattr(self, 'optimized_stations'):
            self.optimized_stations.to_csv('최적화_신규정류장.csv', index=False, encoding='utf-8-sig')
            print("✓ 신규 정류장 결과 저장: 최적화_신규정류장.csv")

        if hasattr(self, 'underserved_regions'):
            self.underserved_regions.to_csv('서비스부족지역.csv', index=False, encoding='utf-8-sig')
            print("✓ 서비스 부족 지역 저장: 서비스부족지역.csv")

# ============================================================================
# 메인 실행
# ============================================================================
if __name__ == "__main__":
    setup_korean_font()

    print("\n" + "="*100)
    print("세종시 버스정류장 최적화 분석 (정수계획법)".center(100))
    print("="*100)

    # 옵티마이저 초기화
    # 기간 설정: 2024년 ~ 2025년 데이터 사용
    optimizer = BusStationOptimizer(start_date='2024-01-01', end_date='2025-12-05')

    # 1. 정류장별 수요 데이터 생성
    optimizer.create_station_demand_data()

    # 2. 서비스 부족 지역 식별
    optimizer.identify_underserved_areas(coverage_radius=COVERAGE_RADIUS)

    # 3. 최적화 실행
    optimizer.optimize_new_stations(max_new_stations=MAX_NEW_STATIONS,
                                   coverage_radius=COVERAGE_RADIUS)

    # 4. 보고서 생성
    optimizer.generate_report()

    # 5. 결과 저장
    optimizer.save_results()

    print("\n" + "="*100)
    print("분석 완료!".center(100))
    print("="*100)
