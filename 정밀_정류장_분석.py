#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
세종시 버스정류장 정밀 최적화 분석
=================================
전문가 관점: 정류장별 수요 밀도 기반 신규 정류장 최적 배치
"""

import pandas as pd
import numpy as np
from math import radians, cos, sin, asin, sqrt
from scipy.spatial import Voronoi, voronoi_plot_2d
from sklearn.cluster import DBSCAN
import warnings
import json
from datetime import datetime

warnings.filterwarnings('ignore')

# ============================================================================
# 설정
# ============================================================================
class Config:
    """분석 설정"""
    # 분석 기간
    START_DATE = '2024-01-01'
    END_DATE = '2025-12-05'

    # 정류장 파라미터
    COVERAGE_RADIUS = 0.5  # km - 정류장 커버리지 반경 (도보 7분)
    MIN_STATION_DISTANCE = 0.3  # km - 신규 정류장 간 최소 거리

    # 그리드 분석
    GRID_SIZE = 0.005  # 약 500m x 500m (계산 효율성)

    # 최적화
    MAX_NEW_STATIONS = 15
    MIN_DEMAND_THRESHOLD = 5000  # 최소 수요 기준

    # 가중치
    WEIGHT_DEMAND = 0.50  # 수요
    WEIGHT_TRANSFER = 0.25  # 환승
    WEIGHT_COVERAGE_GAP = 0.25  # 커버리지 갭

# ============================================================================
# 유틸리티 함수
# ============================================================================
def haversine_distance(lat1, lon1, lat2, lon2):
    """Haversine 공식으로 실제 거리 계산 (km)"""
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    return 6371 * 2 * asin(sqrt(a))

def calculate_distance_matrix(coords1, coords2):
    """두 좌표 집합 간 거리 행렬 계산"""
    n1, n2 = len(coords1), len(coords2)
    dist_matrix = np.zeros((n1, n2))

    for i in range(n1):
        for j in range(n2):
            dist_matrix[i, j] = haversine_distance(
                coords1[i][0], coords1[i][1],
                coords2[j][0], coords2[j][1]
            )

    return dist_matrix

# ============================================================================
# 데이터 로더
# ============================================================================
class DataLoader:
    """데이터 로드 및 전처리"""

    def __init__(self, start_date=None, end_date=None):
        self.start_date = start_date or Config.START_DATE
        self.end_date = end_date or Config.END_DATE
        self.load_all_data()

    def load_all_data(self):
        """모든 데이터 로드"""
        print("="*100)
        print("데이터 로드 중...".center(100))
        print("="*100)

        # 기존 정류장
        self.stations = pd.read_csv('data/세종도시교통공사_버스정류장_시설현황_20210924.csv')
        self.stations = self.stations.drop_duplicates(subset=['정류소ID'])
        self.stations = self.stations[['정류소ID', '정류소명', '위도', '경도']].dropna()

        # 승하차 데이터
        self.passenger = pd.read_csv('data/지역별승하차_통합데이터.csv')
        self.passenger['날짜'] = pd.to_datetime(self.passenger['날짜'])

        # 기간 필터링
        self.passenger = self.passenger[
            (self.passenger['날짜'] >= pd.to_datetime(self.start_date)) &
            (self.passenger['날짜'] <= pd.to_datetime(self.end_date))
        ]

        # 행정구역 좌표
        self.regions = pd.read_csv('data/행정구역_중심좌표.csv')

        print(f"✓ 기존 정류장: {len(self.stations):,}개")
        print(f"✓ 승하차 데이터: {len(self.passenger):,}건")
        print(f"✓ 분석 기간: {self.start_date} ~ {self.end_date}")
        print(f"✓ 행정구역: {len(self.regions)}개")
        print()

# ============================================================================
# 정류장 수요 분석기
# ============================================================================
class StationDemandAnalyzer:
    """정류장별 수요 분석"""

    def __init__(self, data_loader):
        self.data = data_loader
        self.stations = data_loader.stations.copy()
        self.passenger = data_loader.passenger.copy()
        self.regions = data_loader.regions.copy()

    def assign_demand_to_stations(self):
        """각 행정구역 수요를 가장 가까운 정류장에 할당"""
        print("\n정류장별 수요 할당 중...")
        print("-"*100)

        # 행정구역별 총 수요 계산
        region_demand = self.passenger.groupby('행정구역').agg({
            '승차': 'sum',
            '하차': 'sum',
            '환승': 'sum'
        }).reset_index()

        # 좌표 병합
        region_demand = region_demand.merge(self.regions, on='행정구역', how='left')
        region_demand = region_demand.dropna(subset=['위도', '경도'])

        # 각 행정구역에서 가장 가까운 정류장 찾기
        region_coords = region_demand[['위도', '경도']].values
        station_coords = self.stations[['위도', '경도']].values

        dist_matrix = calculate_distance_matrix(region_coords, station_coords)
        nearest_stations = np.argmin(dist_matrix, axis=1)
        region_demand['가장가까운정류장ID'] = self.stations.iloc[nearest_stations]['정류소ID'].values
        region_demand['최단거리_km'] = dist_matrix[np.arange(len(region_coords)), nearest_stations]

        # 정류장별 수요 집계
        station_demand = region_demand.groupby('가장가까운정류장ID').agg({
            '승차': 'sum',
            '하차': 'sum',
            '환승': 'sum',
            '최단거리_km': 'mean'
        }).reset_index()

        station_demand.columns = ['정류소ID', '할당_승차', '할당_하차', '할당_환승', '평균거리_km']
        station_demand['할당_총수요'] = station_demand['할당_승차'] + station_demand['할당_하차']

        # 정류장 데이터에 병합
        self.stations = self.stations.merge(station_demand, on='정류소ID', how='left')
        self.stations = self.stations.fillna({
            '할당_승차': 0,
            '할당_하차': 0,
            '할당_환승': 0,
            '할당_총수요': 0,
            '평균거리_km': 0
        })

        print(f"✓ 정류장별 수요 할당 완료")
        print(f"  - 평균 정류장당 수요: {self.stations['할당_총수요'].mean():,.0f}명")
        print(f"  - 최대 정류장 수요: {self.stations['할당_총수요'].max():,.0f}명")
        print(f"  - 수요가 있는 정류장: {(self.stations['할당_총수요'] > 0).sum()}개")
        print()

        return self.stations

# ============================================================================
# 수요 밀도 분석기
# ============================================================================
class DemandDensityAnalyzer:
    """그리드 기반 수요 밀도 분석"""

    def __init__(self, stations, regions, passenger_data):
        self.stations = stations
        self.regions = regions
        self.passenger = passenger_data

    def create_demand_grid(self):
        """수요 밀도 그리드 생성"""
        print("\n수요 밀도 그리드 생성 중...")
        print("-"*100)

        # 세종시 경계 설정
        lat_min = self.stations['위도'].min() - 0.05
        lat_max = self.stations['위도'].max() + 0.05
        lon_min = self.stations['경도'].min() - 0.05
        lon_max = self.stations['경도'].max() + 0.05

        # 그리드 생성
        grid_size = Config.GRID_SIZE
        lat_bins = np.arange(lat_min, lat_max, grid_size)
        lon_bins = np.arange(lon_min, lon_max, grid_size)

        # 행정구역별 수요 계산
        region_demand = self.passenger.groupby('행정구역').agg({
            '승차': 'sum',
            '하차': 'sum',
            '환승': 'sum'
        }).reset_index()

        region_demand = region_demand.merge(self.regions, on='행정구역', how='left')
        region_demand = region_demand.dropna(subset=['위도', '경도'])
        region_demand['총수요'] = region_demand['승차'] + region_demand['하차']

        # 각 그리드 셀의 수요 계산 (간단히 가장 가까운 행정구역의 수요 할당)
        grid_cells = []
        for i, lat in enumerate(lat_bins[:-1]):
            for j, lon in enumerate(lon_bins[:-1]):
                cell_lat = lat + grid_size / 2
                cell_lon = lon + grid_size / 2

                # 가장 가까운 행정구역 찾기
                distances = region_demand.apply(
                    lambda row: haversine_distance(cell_lat, cell_lon, row['위도'], row['경도']),
                    axis=1
                )

                if len(distances) > 0:
                    nearest_idx = distances.idxmin()
                    nearest_region = region_demand.loc[nearest_idx]

                    # 거리에 따른 수요 감쇠 (가까울수록 높은 수요)
                    dist = distances[nearest_idx]
                    demand_factor = max(0, 1 - (dist / 2.0))  # 2km 이내만 수요 할당

                    grid_cells.append({
                        '위도': cell_lat,
                        '경도': cell_lon,
                        '수요': nearest_region['총수요'] * demand_factor / len(lat_bins) / len(lon_bins),
                        '환승': nearest_region['환승'] * demand_factor / len(lat_bins) / len(lon_bins),
                        '행정구역': nearest_region['행정구역']
                    })

        self.grid = pd.DataFrame(grid_cells)

        print(f"✓ 그리드 생성 완료")
        print(f"  - 그리드 크기: {len(lat_bins)} x {len(lon_bins)}")
        print(f"  - 총 셀 개수: {len(self.grid):,}개")
        print(f"  - 평균 셀 수요: {self.grid['수요'].mean():.2f}명")
        print()

        return self.grid

# ============================================================================
# 커버리지 분석기
# ============================================================================
class CoverageAnalyzer:
    """커버리지 갭 분석"""

    def __init__(self, stations, demand_grid):
        self.stations = stations
        self.grid = demand_grid

    def find_coverage_gaps(self):
        """커버되지 않은 고수요 지역 찾기"""
        print("\n커버리지 갭 분석 중...")
        print("-"*100)

        # 벡터화된 거리 계산 (배치 처리)
        grid_coords = self.grid[['위도', '경도']].values
        station_coords = self.stations[['위도', '경도']].values

        print(f"  - 거리 계산 중 ({len(grid_coords):,}개 셀 x {len(station_coords):,}개 정류장)...")

        # 배치 처리로 메모리 효율성 개선
        batch_size = 5000
        min_distances = np.zeros(len(grid_coords))

        for i in range(0, len(grid_coords), batch_size):
            end_i = min(i + batch_size, len(grid_coords))
            batch_coords = grid_coords[i:end_i]

            # 각 배치에 대한 거리 계산
            batch_dist = calculate_distance_matrix(batch_coords, station_coords)
            min_distances[i:end_i] = np.min(batch_dist, axis=1)

            if (i // batch_size) % 5 == 0:
                print(f"    진행: {end_i:,}/{len(grid_coords):,} ({end_i/len(grid_coords)*100:.1f}%)")

        self.grid['최단정류장거리_km'] = min_distances
        self.grid['커버여부'] = min_distances <= Config.COVERAGE_RADIUS

        # 커버되지 않은 고수요 지역
        uncovered = self.grid[
            (~self.grid['커버여부']) &
            (self.grid['수요'] > Config.MIN_DEMAND_THRESHOLD / 1000)
        ].copy()

        uncovered = uncovered.sort_values('수요', ascending=False)

        print(f"\n✓ 커버리지 분석 완료")
        print(f"  - 전체 그리드 셀: {len(self.grid):,}개")
        print(f"  - 커버된 셀: {self.grid['커버여부'].sum():,}개 ({self.grid['커버여부'].mean()*100:.1f}%)")
        print(f"  - 미커버 고수요 셀: {len(uncovered):,}개")
        print(f"  - 미커버 총 수요: {uncovered['수요'].sum():,.0f}명")
        print()

        self.uncovered_areas = uncovered
        return uncovered

# ============================================================================
# 정수계획법 최적화
# ============================================================================
class StationOptimizer:
    """정수계획법 기반 신규 정류장 위치 최적화"""

    def __init__(self, uncovered_areas):
        self.candidates = uncovered_areas.copy()

    def optimize_locations(self, max_stations=Config.MAX_NEW_STATIONS):
        """최적 신규 정류장 위치 선정"""
        print("\n정수계획법 최적화 실행 중...")
        print("-"*100)

        if len(self.candidates) == 0:
            print("⚠ 최적화할 후보 지역이 없습니다.")
            return pd.DataFrame()

        from pulp import LpProblem, LpMaximize, LpVariable, lpSum, LpStatus, PULP_CBC_CMD, value

        # 후보 지역 클러스터링 (너무 가까운 셀들 통합)
        coords = self.candidates[['위도', '경도']].values

        # DBSCAN으로 클러스터링
        clustering = DBSCAN(eps=0.005, min_samples=1).fit(coords)
        self.candidates['클러스터'] = clustering.labels_

        # 각 클러스터의 대표 위치 선정 (수요 가중 평균)
        clusters = []
        for cluster_id in self.candidates['클러스터'].unique():
            cluster_data = self.candidates[self.candidates['클러스터'] == cluster_id]

            total_demand = cluster_data['수요'].sum()
            weighted_lat = (cluster_data['위도'] * cluster_data['수요']).sum() / total_demand
            weighted_lon = (cluster_data['경도'] * cluster_data['수요']).sum() / total_demand

            clusters.append({
                '위도': weighted_lat,
                '경도': weighted_lon,
                '수요': total_demand,
                '환승': cluster_data['환승'].sum(),
                '평균거리': cluster_data['최단정류장거리_km'].mean(),
                '셀개수': len(cluster_data)
            })

        candidates_df = pd.DataFrame(clusters)
        n = len(candidates_df)

        if n == 0:
            return pd.DataFrame()

        print(f"  - 클러스터링 완료: {len(self.candidates)}개 셀 → {n}개 후보 지역")

        # 거리 행렬
        coords = candidates_df[['위도', '경도']].values
        dist_matrix = calculate_distance_matrix(coords, coords)

        # 가중치 계산
        demand = candidates_df['수요'].values
        demand_norm = (demand - demand.min()) / (demand.max() - demand.min() + 1e-10)

        transfer = candidates_df['환승'].values
        transfer_norm = (transfer - transfer.min()) / (transfer.max() - transfer.min() + 1e-10)

        gap = candidates_df['평균거리'].values
        gap_norm = (gap - gap.min()) / (gap.max() - gap.min() + 1e-10)

        weights = (Config.WEIGHT_DEMAND * demand_norm +
                  Config.WEIGHT_TRANSFER * transfer_norm +
                  Config.WEIGHT_COVERAGE_GAP * gap_norm)

        # 커버리지 행렬
        coverage_matrix = (dist_matrix <= Config.COVERAGE_RADIUS).astype(int)
        np.fill_diagonal(coverage_matrix, 1)

        # 정수계획법 모델
        prob = LpProblem("Station_Optimization", LpMaximize)

        # 결정 변수
        x = LpVariable.dicts("station", range(n), cat='Binary')
        y = LpVariable.dicts("covered", range(n), cat='Binary')

        # 목적함수
        prob += lpSum([demand[i] * weights[i] * y[i] for i in range(n)]), "Weighted_Demand"

        # 제약조건
        prob += lpSum([x[i] for i in range(n)]) <= max_stations, "Max_Stations"

        for i in range(n):
            prob += y[i] <= lpSum([coverage_matrix[i][j] * x[j] for j in range(n)]), f"Coverage_{i}"

        # 정류장 간 최소 거리 제약
        for i in range(n):
            for j in range(i+1, n):
                if dist_matrix[i, j] < Config.MIN_STATION_DISTANCE:
                    prob += x[i] + x[j] <= 1, f"MinDist_{i}_{j}"

        # 최적화 실행
        prob.solve(PULP_CBC_CMD(msg=0))

        if LpStatus[prob.status] == 'Optimal':
            selected_indices = [i for i in range(n) if x[i].varValue == 1]
            optimized = candidates_df.iloc[selected_indices].copy()

            # 커버 수요 계산
            optimized['커버_수요'] = [
                sum([demand[j] for j in range(n) if coverage_matrix[i][j] == 1])
                for i in selected_indices
            ]

            optimized['우선순위'] = range(1, len(optimized) + 1)
            optimized = optimized.sort_values('수요', ascending=False).reset_index(drop=True)
            optimized['우선순위'] = range(1, len(optimized) + 1)

            print(f"\n✓ 최적화 완료!")
            print(f"  - 선정된 신규 정류장: {len(optimized)}개")
            print(f"  - 총 예상 수요: {optimized['수요'].sum():,.0f}명")
            print(f"  - 목적함수 값: {value(prob.objective):,.2f}")
            print()

            self.optimized_stations = optimized
            return optimized
        else:
            print(f"⚠ 최적화 실패: {LpStatus[prob.status]}")
            return pd.DataFrame()

# ============================================================================
# 메인 실행
# ============================================================================
def main():
    print("\n" + "="*100)
    print("세종시 버스정류장 정밀 최적화 분석".center(100))
    print("="*100)
    print()

    # 1. 데이터 로드
    data_loader = DataLoader(
        start_date=Config.START_DATE,
        end_date=Config.END_DATE
    )

    # 2. 정류장별 수요 분석
    demand_analyzer = StationDemandAnalyzer(data_loader)
    stations_with_demand = demand_analyzer.assign_demand_to_stations()

    # 3. 수요 밀도 그리드 생성
    density_analyzer = DemandDensityAnalyzer(
        stations_with_demand,
        data_loader.regions,
        data_loader.passenger
    )
    demand_grid = density_analyzer.create_demand_grid()

    # 4. 커버리지 갭 분석
    coverage_analyzer = CoverageAnalyzer(stations_with_demand, demand_grid)
    uncovered_areas = coverage_analyzer.find_coverage_gaps()

    # 5. 최적화
    optimizer = StationOptimizer(uncovered_areas)
    new_stations = optimizer.optimize_locations(max_stations=Config.MAX_NEW_STATIONS)

    # 6. 결과 저장
    print("\n결과 저장 중...")
    print("-"*100)

    # 정류장 수요 데이터
    stations_with_demand.to_csv('분석결과_정류장별수요.csv', index=False, encoding='utf-8-sig')
    print("✓ 정류장별 수요: 분석결과_정류장별수요.csv")

    # 수요 그리드
    demand_grid.to_csv('분석결과_수요그리드.csv', index=False, encoding='utf-8-sig')
    print("✓ 수요 그리드: 분석결과_수요그리드.csv")

    # 신규 정류장
    if len(new_stations) > 0:
        new_stations.to_csv('분석결과_신규정류장.csv', index=False, encoding='utf-8-sig')
        print("✓ 신규 정류장: 분석결과_신규정류장.csv")

    # 보고서
    report = {
        '분석일시': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        '분석기간': {
            '시작': Config.START_DATE,
            '종료': Config.END_DATE
        },
        '기존정류장': {
            '총개수': int(len(stations_with_demand)),
            '수요있는정류장': int((stations_with_demand['할당_총수요'] > 0).sum()),
            '평균수요': float(stations_with_demand['할당_총수요'].mean()),
            '최대수요': float(stations_with_demand['할당_총수요'].max())
        },
        '수요밀도': {
            '그리드셀개수': int(len(demand_grid)),
            '평균셀수요': float(demand_grid['수요'].mean()),
            '총수요': float(demand_grid['수요'].sum())
        },
        '커버리지': {
            '커버율': float(demand_grid['커버여부'].mean() * 100),
            '미커버고수요지역': int(len(uncovered_areas)),
            '미커버수요': float(uncovered_areas['수요'].sum() if len(uncovered_areas) > 0 else 0)
        },
        '최적화결과': {
            '신규정류장개수': int(len(new_stations)) if len(new_stations) > 0 else 0,
            '예상커버수요': float(new_stations['수요'].sum()) if len(new_stations) > 0 else 0
        },
        '설정': {
            '커버리지반경_km': Config.COVERAGE_RADIUS,
            '최소정류장간거리_km': Config.MIN_STATION_DISTANCE,
            '그리드크기': Config.GRID_SIZE,
            '최대신규정류장': Config.MAX_NEW_STATIONS
        }
    }

    with open('분석결과_보고서.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("✓ 분석 보고서: 분석결과_보고서.json")

    print("\n" + "="*100)
    print("분석 완료!".center(100))
    print("="*100)

    return {
        'stations': stations_with_demand,
        'grid': demand_grid,
        'new_stations': new_stations,
        'report': report
    }

if __name__ == "__main__":
    results = main()
