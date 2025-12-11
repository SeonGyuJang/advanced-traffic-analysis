#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PPT 발표용 고품질 정적 이미지 생성
한 페이지에 핵심 정보 모두 포함
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False

def haversine_distance(lat1, lon1, lat2, lon2):
    """Haversine 거리 계산 (km)"""
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

def create_presentation_slide():
    """PPT 발표용 종합 슬라이드 생성"""

    print("📊 데이터 로딩 중...")
    stations = pd.read_csv('분석결과_정류장별수요.csv')
    new_stations = pd.read_csv('분석결과_신규정류장.csv')

    # 활성 정류장만 필터링
    active_stations = stations[stations['할당_총수요'] > 0].copy()

    print("🎨 발표 슬라이드 생성 중...")

    # 초고해상도 설정
    fig = plt.figure(figsize=(20, 11), dpi=300)
    fig.patch.set_facecolor('white')

    # GridSpec으로 레이아웃 구성
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3,
                  left=0.05, right=0.95, top=0.92, bottom=0.08)

    # 제목
    fig.suptitle('세종시 버스정류장 최적화 분석 결과',
                 fontsize=28, fontweight='bold', y=0.97)

    # ==================== 1. 지도 (왼쪽 대형) ====================
    ax_map = fig.add_subplot(gs[:, 0:2])

    # 지도 범위 설정
    lat_min = min(stations['위도'].min(), new_stations['위도'].min()) - 0.02
    lat_max = max(stations['위도'].max(), new_stations['위도'].max()) + 0.02
    lon_min = min(stations['경도'].min(), new_stations['경도'].min()) - 0.02
    lon_max = max(stations['경도'].max(), new_stations['경도'].max()) + 0.02

    ax_map.set_xlim(lon_min, lon_max)
    ax_map.set_ylim(lat_min, lat_max)
    ax_map.set_aspect('equal')

    # 배경 그리드
    ax_map.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)

    # 기존 정류장 커버리지 (투명한 원)
    coverage_radius_deg = 0.5 / 111  # 500m to degrees
    for _, row in active_stations.iterrows():
        circle = plt.Circle((row['경도'], row['위도']),
                           coverage_radius_deg,
                           color='#3186cc', alpha=0.08, zorder=1)
        ax_map.add_patch(circle)

    # 기존 정류장 표시 (크기 = 수요)
    sizes = active_stations['할당_총수요'] / 50000
    sizes = np.clip(sizes, 20, 300)

    scatter_existing = ax_map.scatter(
        active_stations['경도'],
        active_stations['위도'],
        s=sizes,
        c='#1f77b4',
        alpha=0.6,
        edgecolors='white',
        linewidth=1.5,
        zorder=3,
        label='기존 정류장'
    )

    # 신규 정류장 커버리지 (강조된 원)
    for _, row in new_stations.iterrows():
        if row['우선순위'] <= 5:
            color = '#ff0000'
            alpha = 0.15
        elif row['우선순위'] <= 10:
            color = '#ff7f0e'
            alpha = 0.12
        else:
            color = '#ffd700'
            alpha = 0.1

        circle = plt.Circle((row['경도'], row['위도']),
                           coverage_radius_deg,
                           color=color, alpha=alpha, zorder=2)
        ax_map.add_patch(circle)

    # 신규 정류장 표시 (별 마커)
    colors = []
    for _, row in new_stations.iterrows():
        if row['우선순위'] <= 5:
            colors.append('#ff0000')
        elif row['우선순위'] <= 10:
            colors.append('#ff7f0e')
        else:
            colors.append('#ffd700')

    ax_map.scatter(
        new_stations['경도'],
        new_stations['위도'],
        s=400,
        c=colors,
        marker='*',
        edgecolors='darkred',
        linewidth=2,
        zorder=5,
        label='신규 정류장 ⭐'
    )

    # 상위 5개 신규 정류장 번호 표시
    for _, row in new_stations.nsmallest(5, '우선순위').iterrows():
        ax_map.annotate(
            f"{row['우선순위']}",
            xy=(row['경도'], row['위도']),
            xytext=(8, 8),
            textcoords='offset points',
            fontsize=11,
            fontweight='bold',
            color='white',
            bbox=dict(boxstyle='circle,pad=0.3', facecolor='red',
                     edgecolor='darkred', linewidth=2),
            zorder=6
        )

    ax_map.set_xlabel('경도', fontsize=12, fontweight='bold')
    ax_map.set_ylabel('위도', fontsize=12, fontweight='bold')
    ax_map.set_title('정류장 위치 및 커버리지 영역',
                     fontsize=16, fontweight='bold', pad=15)

    # 범례
    legend_elements = [
        mpatches.Patch(facecolor='#1f77b4', alpha=0.6, edgecolor='white',
                      linewidth=1.5, label='기존 정류장'),
        mpatches.Patch(facecolor='#ff0000', alpha=0.6,
                      label='신규 우선순위 1-5'),
        mpatches.Patch(facecolor='#ff7f0e', alpha=0.6,
                      label='신규 우선순위 6-10'),
        mpatches.Patch(facecolor='#ffd700', alpha=0.6,
                      label='신규 우선순위 11-15'),
        mpatches.Patch(facecolor='#3186cc', alpha=0.15,
                      label='커버리지 영역 (500m)')
    ]
    ax_map.legend(handles=legend_elements, loc='upper left',
                 fontsize=10, framealpha=0.9)

    # ==================== 2. KPI 카드 (우측 상단) ====================
    ax_kpi = fig.add_subplot(gs[0, 2])
    ax_kpi.axis('off')

    total_demand = stations['할당_총수요'].sum()
    new_demand = new_stations['수요'].sum()
    active_count = len(active_stations)

    kpi_text = f"""
    ┏━━━━━━━━━━━━━━━━━━━━┓
    ┃   핵심 성과 지표   ┃
    ┗━━━━━━━━━━━━━━━━━━━━┛

    📍 기존 정류장
       {len(stations):,}개 (활성 {active_count}개)

    👥 총 수요
       {total_demand:,.0f}명

    ⭐ 신규 정류장
       {len(new_stations)}개 추천

    📈 신규 예상 수요
       {new_demand:,.0f}명

    🎯 수요 커버율 증가
       +{(new_demand/total_demand*100):.1f}%
    """

    ax_kpi.text(0.5, 0.5, kpi_text,
               ha='center', va='center',
               fontsize=11,
               fontfamily='monospace',
               bbox=dict(boxstyle='round,pad=1',
                        facecolor='#f0f8ff',
                        edgecolor='#1f77b4',
                        linewidth=3))

    # ==================== 3. 기존 정류장 차트 (우측 중간) ====================
    ax_existing = fig.add_subplot(gs[1, 2])

    top_existing = active_stations.nlargest(8, '할당_총수요').sort_values('할당_총수요')

    bars1 = ax_existing.barh(
        range(len(top_existing)),
        top_existing['할당_총수요'],
        color='#1f77b4',
        alpha=0.7,
        edgecolor='navy',
        linewidth=1.5
    )

    # 값 표시
    for i, (idx, row) in enumerate(top_existing.iterrows()):
        ax_existing.text(
            row['할당_총수요'],
            i,
            f" {row['할당_총수요']:,.0f}명",
            va='center',
            ha='left',
            fontsize=9,
            fontweight='bold'
        )

    ax_existing.set_yticks(range(len(top_existing)))
    ax_existing.set_yticklabels([name[:10] + '...' if len(name) > 10 else name
                                 for name in top_existing['정류소명']],
                                fontsize=9)
    ax_existing.set_xlabel('수요 (명)', fontsize=10, fontweight='bold')
    ax_existing.set_title('기존 정류장 수요 Top 8',
                         fontsize=12, fontweight='bold', pad=10)
    ax_existing.grid(axis='x', alpha=0.3, linestyle='--')
    ax_existing.spines['top'].set_visible(False)
    ax_existing.spines['right'].set_visible(False)

    # ==================== 4. 신규 정류장 차트 (우측 하단) ====================
    ax_new = fig.add_subplot(gs[2, 2])

    top_new = new_stations.nsmallest(8, '우선순위')

    colors_new = ['#ff0000' if p <= 5 else '#ff7f0e'
                  for p in top_new['우선순위']]

    bars2 = ax_new.barh(
        range(len(top_new)),
        top_new['수요'],
        color=colors_new,
        alpha=0.7,
        edgecolor='darkred',
        linewidth=1.5
    )

    # 값 표시
    for i, (idx, row) in enumerate(top_new.iterrows()):
        ax_new.text(
            row['수요'],
            i,
            f" {row['수요']:,.0f}명",
            va='center',
            ha='left',
            fontsize=9,
            fontweight='bold'
        )

    ax_new.set_yticks(range(len(top_new)))
    ax_new.set_yticklabels([f"우선순위 {p}" for p in top_new['우선순위']],
                          fontsize=9)
    ax_new.set_xlabel('예상 수요 (명)', fontsize=10, fontweight='bold')
    ax_new.set_title('신규 정류장 우선순위 Top 8',
                    fontsize=12, fontweight='bold', pad=10)
    ax_new.grid(axis='x', alpha=0.3, linestyle='--')
    ax_new.spines['top'].set_visible(False)
    ax_new.spines['right'].set_visible(False)

    # 푸터
    fig.text(0.5, 0.02,
            '정수계획법 기반 버스정류장 최적화 분석 | 세종시 교통 데이터 (2024-2025) | Advanced Traffic Analysis Team',
            ha='center', fontsize=10, color='#666')

    # 저장
    output_file = '발표자료_종합분석.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"✅ 발표 슬라이드 저장 완료: {output_file}")

    plt.close()

def create_detailed_table():
    """신규 정류장 상세 테이블 이미지 생성"""

    print("📋 상세 테이블 생성 중...")

    new_stations = pd.read_csv('분석결과_신규정류장.csv')

    fig, ax = plt.subplots(figsize=(16, 10), dpi=300)
    fig.patch.set_facecolor('white')
    ax.axis('off')

    # 제목
    fig.suptitle('신규 버스정류장 추천 목록 (상세)',
                 fontsize=24, fontweight='bold', y=0.96)

    # 테이블 데이터 준비
    table_data = []
    table_data.append(['우선\n순위', '위도', '경도', '예상\n수요(명)',
                      '환승(명)', '커버\n수요(명)', '평가'])

    for _, row in new_stations.iterrows():
        if row['우선순위'] <= 5:
            priority_str = f"★ {row['우선순위']}"
        else:
            priority_str = str(row['우선순위'])

        table_data.append([
            priority_str,
            f"{row['위도']:.4f}",
            f"{row['경도']:.4f}",
            f"{row['수요']:,.0f}",
            f"{row['환승']:,.0f}",
            f"{row['커버_수요']:,.0f}",
            '최우선' if row['우선순위'] <= 5 else '우선' if row['우선순위'] <= 10 else '일반'
        ])

    # 색상 설정
    cell_colors = []
    cell_colors.append(['#1f77b4'] * 7)  # 헤더

    for _, row in new_stations.iterrows():
        if row['우선순위'] <= 5:
            row_color = ['#ffcccc'] * 7  # 빨강 계열
        elif row['우선순위'] <= 10:
            row_color = ['#ffe6cc'] * 7  # 주황 계열
        else:
            row_color = ['#ffffcc'] * 7  # 노랑 계열
        cell_colors.append(row_color)

    # 테이블 생성
    table = ax.table(
        cellText=table_data,
        cellColours=cell_colors,
        cellLoc='center',
        loc='center',
        bbox=[0.05, 0.1, 0.9, 0.8]
    )

    # 스타일링
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)

    # 헤더 스타일
    for i in range(7):
        cell = table[(0, i)]
        cell.set_text_props(weight='bold', color='white', fontsize=12)
        cell.set_facecolor('#1f77b4')

    # 테두리 강조
    for key, cell in table.get_celld().items():
        cell.set_edgecolor('#333')
        cell.set_linewidth(1.5)

    # 인사이트 추가
    insight_text = """
    💡 핵심 인사이트:
    • 우선순위 1-5위 정류장은 즉시 설치를 권장합니다 (예상 수요가 높고 커버리지 효율이 우수)
    • 총 15개 신규 정류장으로 약 10,253명의 추가 수요를 커버할 수 있습니다
    • 신규 정류장은 기존 정류장 간 커버리지 공백을 메우는 전략적 위치에 배치됩니다
    """

    fig.text(0.5, 0.04, insight_text,
            ha='center', va='top', fontsize=11,
            bbox=dict(boxstyle='round,pad=0.8',
                     facecolor='#fffacd',
                     edgecolor='#ffa500',
                     linewidth=2))

    # 저장
    output_file = '발표자료_신규정류장목록.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"✅ 상세 테이블 저장 완료: {output_file}")

    plt.close()

def main():
    """메인 실행"""
    print("=" * 60)
    print("PPT 발표 자료 생성 시작")
    print("=" * 60)

    try:
        # 종합 슬라이드 생성
        create_presentation_slide()

        # 상세 테이블 생성
        create_detailed_table()

        print("\n" + "=" * 60)
        print("✅ 모든 발표 자료 생성 완료!")
        print("=" * 60)
        print("\n생성된 파일:")
        print("  1. 발표자료_종합분석.png - 종합 분석 슬라이드")
        print("  2. 발표자료_신규정류장목록.png - 신규 정류장 상세 목록")
        print("\n이 파일들을 PowerPoint에 바로 삽입하여 사용하세요!")

    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
