#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
고급 시각화 도구
===============
정류장 간 관계, 네트워크, Voronoi 다이어그램 등 고급 시각화
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
import seaborn as sns
from scipy.spatial import Voronoi, voronoi_plot_2d
import networkx as nx
from math import radians, cos, sin, asin, sqrt
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# 한국어 폰트 설정
# ============================================================================
def setup_korean_font():
    """나눔 폰트 설정"""
    font_list = [f.name for f in fm.fontManager.ttflist]

    korean_fonts = ['NanumGothic', 'NanumBarunGothic', 'NanumSquare']

    for font in korean_fonts:
        if font in font_list:
            plt.rcParams['font.family'] = font
            plt.rcParams['axes.unicode_minus'] = False
            print(f"✓ 한국어 폰트 설정: {font}")
            return font

    plt.rcParams['axes.unicode_minus'] = False
    return 'DejaVu Sans'

# ============================================================================
# 유틸리티 함수
# ============================================================================
def haversine_distance(lat1, lon1, lat2, lon2):
    """Haversine 거리 계산 (km)"""
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    return 6371 * 2 * asin(sqrt(a))

# ============================================================================
# 데이터 로드
# ============================================================================
print("="*100)
print("고급 시각화 도구 실행 중...".center(100))
print("="*100)

setup_korean_font()

print("\n데이터 로드 중...")
stations = pd.read_csv('분석결과_정류장별수요.csv')
grid = pd.read_csv('분석결과_수요그리드.csv')
new_stations = pd.read_csv('분석결과_신규정류장.csv')

print(f"✓ 정류장: {len(stations):,}개")
print(f"✓ 그리드: {len(grid):,}개")
print(f"✓ 신규: {len(new_stations)}개")

# ============================================================================
# 1. 정류장 네트워크 시각화
# ============================================================================
print("\n[1/5] 정류장 네트워크 생성 중...")

# 활성 정류장만 (수요 있는)
active_stations = stations[stations['할당_총수요'] > 0].copy()

# 거리 기반 네트워크 구축
G = nx.Graph()

# 노드 추가
for idx, row in active_stations.iterrows():
    G.add_node(
        row['정류소ID'],
        pos=(row['경도'], row['위도']),
        demand=row['할당_총수요'],
        name=row['정류소명']
    )

# 가까운 정류장 간 엣지 추가 (1km 이내)
print("  - 엣지 계산 중...")
for i, row1 in active_stations.iterrows():
    for j, row2 in active_stations.iterrows():
        if i < j:
            dist = haversine_distance(
                row1['위도'], row1['경도'],
                row2['위도'], row2['경도']
            )

            if dist <= 1.0:  # 1km 이내만
                G.add_edge(
                    row1['정류소ID'],
                    row2['정류소ID'],
                    weight=dist
                )

print(f"  - 노드: {G.number_of_nodes()}개")
print(f"  - 엣지: {G.number_of_edges()}개")

# 네트워크 시각화
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))

# 왼쪽: 지리적 배치
pos = nx.get_node_attributes(G, 'pos')
demand = nx.get_node_attributes(G, 'demand')

# 노드 크기 (수요에 비례)
node_sizes = [np.log10(demand[node] + 1) * 100 for node in G.nodes()]

# 노드 색상 (수요에 따라)
node_colors = [demand[node] for node in G.nodes()]

nx.draw_networkx_edges(G, pos, ax=ax1, alpha=0.2, width=0.5)
nodes = nx.draw_networkx_nodes(
    G, pos, ax=ax1,
    node_size=node_sizes,
    node_color=node_colors,
    cmap='YlOrRd',
    alpha=0.7
)

ax1.set_title('정류장 네트워크 (지리적 배치)', fontsize=16, fontweight='bold', pad=20)
ax1.set_xlabel('경도', fontsize=12)
ax1.set_ylabel('위도', fontsize=12)
ax1.grid(True, alpha=0.3)

# 컬러바
cbar1 = plt.colorbar(nodes, ax=ax1)
cbar1.set_label('수요 (명)', fontsize=11)

# 오른쪽: 중심성 분석
degree_centrality = nx.degree_centrality(G)
betweenness_centrality = nx.betweenness_centrality(G, weight='weight')

# 중심성 높은 노드 크기 증가
centrality_sizes = [betweenness_centrality[node] * 5000 + 50 for node in G.nodes()]
centrality_colors = [degree_centrality[node] for node in G.nodes()]

nx.draw_networkx_edges(G, pos, ax=ax2, alpha=0.2, width=0.5)
nodes2 = nx.draw_networkx_nodes(
    G, pos, ax=ax2,
    node_size=centrality_sizes,
    node_color=centrality_colors,
    cmap='viridis',
    alpha=0.7
)

ax2.set_title('정류장 중심성 분석 (Betweenness Centrality)', fontsize=16, fontweight='bold', pad=20)
ax2.set_xlabel('경도', fontsize=12)
ax2.set_ylabel('위도', fontsize=12)
ax2.grid(True, alpha=0.3)

# 컬러바
cbar2 = plt.colorbar(nodes2, ax=ax2)
cbar2.set_label('Degree Centrality', fontsize=11)

plt.tight_layout()
plt.savefig('고급시각화_네트워크.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✓ 네트워크 시각화 저장: 고급시각화_네트워크.png")
plt.close()

# ============================================================================
# 2. Voronoi 다이어그램
# ============================================================================
print("\n[2/5] Voronoi 다이어그램 생성 중...")

# 상위 수요 정류장만 (시각화 명확성)
top_stations = active_stations.nlargest(50, '할당_총수요')

# Voronoi 다이어그램 계산
points = top_stations[['경도', '위도']].values

try:
    vor = Voronoi(points)

    fig, ax = plt.subplots(figsize=(16, 12))

    # Voronoi 다이어그램 그리기
    voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_colors='blue',
                    line_width=1, line_alpha=0.6, point_size=0)

    # 정류장 표시
    scatter = ax.scatter(
        top_stations['경도'],
        top_stations['위도'],
        s=top_stations['할당_총수요'] / 10000,
        c=top_stations['할당_총수요'],
        cmap='YlOrRd',
        alpha=0.7,
        edgecolors='black',
        linewidth=1.5,
        zorder=5
    )

    # 상위 10개 정류장 이름 표시
    for _, row in top_stations.head(10).iterrows():
        ax.annotate(
            row['정류소명'],
            (row['경도'], row['위도']),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=9,
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7)
        )

    ax.set_title('정류장 Voronoi 다이어그램 (상위 50개)', fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel('경도', fontsize=13)
    ax.set_ylabel('위도', fontsize=13)
    ax.grid(True, alpha=0.3)

    # 컬러바
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('수요 (명)', fontsize=12)

    plt.tight_layout()
    plt.savefig('고급시각화_Voronoi.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("✓ Voronoi 다이어그램 저장: 고급시각화_Voronoi.png")
    plt.close()

except Exception as e:
    print(f"⚠ Voronoi 생성 실패: {e}")

# ============================================================================
# 3. 정류장 간 거리 행렬 히트맵
# ============================================================================
print("\n[3/5] 거리 행렬 히트맵 생성 중...")

# 상위 20개 정류장
top20 = active_stations.nlargest(20, '할당_총수요')

# 거리 행렬 계산
n = len(top20)
dist_matrix = np.zeros((n, n))

for i, (idx1, row1) in enumerate(top20.iterrows()):
    for j, (idx2, row2) in enumerate(top20.iterrows()):
        if i != j:
            dist_matrix[i, j] = haversine_distance(
                row1['위도'], row1['경도'],
                row2['위도'], row2['경도']
            )

# 히트맵
fig, ax = plt.subplots(figsize=(16, 14))

im = ax.imshow(dist_matrix, cmap='RdYlGn_r', aspect='auto')

# 축 설정
ax.set_xticks(range(n))
ax.set_yticks(range(n))
ax.set_xticklabels(top20['정류소명'].values, rotation=45, ha='right', fontsize=10)
ax.set_yticklabels(top20['정류소명'].values, fontsize=10)

# 값 표시
for i in range(n):
    for j in range(n):
        if i != j:
            text = ax.text(j, i, f'{dist_matrix[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=7)

ax.set_title('정류장 간 거리 행렬 (상위 20개, km)', fontsize=18, fontweight='bold', pad=20)

# 컬러바
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('거리 (km)', fontsize=12)

plt.tight_layout()
plt.savefig('고급시각화_거리행렬.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✓ 거리 행렬 저장: 고급시각화_거리행렬.png")
plt.close()

# ============================================================================
# 4. 수요 밀도 3D 표면
# ============================================================================
print("\n[4/5] 3D 수요 밀도 표면 생성 중...")

from mpl_toolkits.mplot3d import Axes3D

# 그리드 데이터로 3D 표면 생성
# 그리드를 2D 배열로 재구성
lat_unique = sorted(grid['위도'].unique())
lon_unique = sorted(grid['경도'].unique())

# 간격 줄이기 (성능)
lat_sample = lat_unique[::5]
lon_sample = lon_unique[::5]

# 메시그리드 생성
X, Y = np.meshgrid(lon_sample, lat_sample)
Z = np.zeros_like(X)

# 수요 매핑
for i, lat in enumerate(lat_sample):
    for j, lon in enumerate(lon_sample):
        matching = grid[(grid['위도'] == lat) & (grid['경도'] == lon)]
        if len(matching) > 0:
            Z[i, j] = matching.iloc[0]['수요']

# 3D 플롯
fig = plt.figure(figsize=(16, 12))
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(X, Y, Z, cmap='YlOrRd', alpha=0.8, edgecolor='none')

# 정류장 표시
ax.scatter(
    active_stations['경도'],
    active_stations['위도'],
    active_stations['할당_총수요'] / 1000,
    c='blue',
    marker='o',
    s=50,
    alpha=0.6,
    label='기존 정류장'
)

# 신규 정류장
ax.scatter(
    new_stations['경도'],
    new_stations['위도'],
    new_stations['수요'],
    c='red',
    marker='*',
    s=200,
    label='신규 정류장'
)

ax.set_xlabel('경도', fontsize=12)
ax.set_ylabel('위도', fontsize=12)
ax.set_zlabel('수요 (명)', fontsize=12)
ax.set_title('수요 밀도 3D 표면', fontsize=18, fontweight='bold', pad=20)

# 컬러바
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='그리드 수요 (명)')

ax.legend(fontsize=10)
ax.view_init(elev=30, azim=45)

plt.savefig('고급시각화_3D표면.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✓ 3D 표면 저장: 고급시각화_3D표면.png")
plt.close()

# ============================================================================
# 5. 정류장 클러스터 분석
# ============================================================================
print("\n[5/5] 정류장 클러스터 분석 중...")

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 특성 준비
features = active_stations[['위도', '경도', '할당_총수요', '할당_환승']].copy()
features['할당_총수요_log'] = np.log10(features['할당_총수요'] + 1)
features['할당_환승_log'] = np.log10(features['할당_환승'] + 1)

# 스케일링
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features[['위도', '경도', '할당_총수요_log', '할당_환승_log']])

# KMeans 클러스터링
n_clusters = 5
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
clusters = kmeans.fit_predict(features_scaled)

active_stations['클러스터'] = clusters

# 시각화
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))

# 왼쪽: 지리적 분포
scatter1 = ax1.scatter(
    active_stations['경도'],
    active_stations['위도'],
    c=clusters,
    s=active_stations['할당_총수요'] / 10000,
    cmap='tab10',
    alpha=0.6,
    edgecolors='black',
    linewidth=1
)

# 클러스터 중심 표시
centers = scaler.inverse_transform(kmeans.cluster_centers_)
ax1.scatter(
    centers[:, 1],  # 경도
    centers[:, 0],  # 위도
    c='red',
    marker='X',
    s=500,
    edgecolors='black',
    linewidth=2,
    label='클러스터 중심'
)

ax1.set_title('정류장 클러스터 분포 (지리)', fontsize=16, fontweight='bold', pad=20)
ax1.set_xlabel('경도', fontsize=12)
ax1.set_ylabel('위도', fontsize=12)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# 오른쪽: 수요 vs 환승
scatter2 = ax2.scatter(
    active_stations['할당_총수요'],
    active_stations['할당_환승'],
    c=clusters,
    s=100,
    cmap='tab10',
    alpha=0.6,
    edgecolors='black',
    linewidth=1
)

ax2.set_title('정류장 클러스터 분포 (수요 vs 환승)', fontsize=16, fontweight='bold', pad=20)
ax2.set_xlabel('총 수요 (명)', fontsize=12)
ax2.set_ylabel('환승 (명)', fontsize=12)
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('고급시각화_클러스터.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✓ 클러스터 분석 저장: 고급시각화_클러스터.png")
plt.close()

# 클러스터별 통계
print("\n클러스터별 통계:")
for i in range(n_clusters):
    cluster_data = active_stations[active_stations['클러스터'] == i]
    print(f"\n클러스터 {i+1}:")
    print(f"  - 정류장 수: {len(cluster_data)}개")
    print(f"  - 평균 수요: {cluster_data['할당_총수요'].mean():,.0f}명")
    print(f"  - 평균 환승: {cluster_data['할당_환승'].mean():,.0f}명")

# ============================================================================
# 완료
# ============================================================================
print("\n" + "="*100)
print("모든 고급 시각화 완료!".center(100))
print("="*100)
print("\n생성된 파일:")
print("  1. 고급시각화_네트워크.png - 정류장 네트워크 및 중심성 분석")
print("  2. 고급시각화_Voronoi.png - Voronoi 다이어그램")
print("  3. 고급시각화_거리행렬.png - 정류장 간 거리 히트맵")
print("  4. 고급시각화_3D표면.png - 수요 밀도 3D 표면")
print("  5. 고급시각화_클러스터.png - 정류장 클러스터 분석")
