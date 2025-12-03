#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
세종시 버스 노선 최적화 인터랙티브 대시보드
================================================
분석 결과를 종합적으로 시각화하는 대시보드를 생성합니다.
"""

import pandas as pd
import json
import folium
from folium import plugins
import numpy as np

def create_comprehensive_dashboard():
    """종합 인터랙티브 대시보드 생성"""
    print("🎨 종합 대시보드 생성 중...")

    # 분석 결과 로드
    route_df = pd.read_csv('버스노선_분석결과.csv')
    region_df = pd.read_csv('지역별_서비스수준.csv')
    recommendations_df = pd.read_csv('노선_최적화_제안.csv')
    overlap_df = pd.read_csv('노선_중복도_분석.csv')

    with open('버스노선_최적화_보고서.json', 'r', encoding='utf-8') as f:
        report = json.load(f)

    # HTML 템플릿 생성
    html_content = f"""
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>세종시 버스 노선 최적화 대시보드</title>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.7.1/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.7.1/dist/leaflet.js"></script>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
        }}

        .container {{
            max-width: 1600px;
            margin: 0 auto;
        }}

        .header {{
            background: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            margin-bottom: 30px;
            text-align: center;
        }}

        .header h1 {{
            color: #2E4057;
            font-size: 2.5em;
            margin-bottom: 10px;
        }}

        .header p {{
            color: #666;
            font-size: 1.1em;
        }}

        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}

        .stat-card {{
            background: white;
            padding: 25px;
            border-radius: 15px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            text-align: center;
            transition: transform 0.3s ease;
        }}

        .stat-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 10px 25px rgba(0,0,0,0.15);
        }}

        .stat-card .number {{
            font-size: 2.5em;
            font-weight: bold;
            margin: 10px 0;
        }}

        .stat-card .label {{
            color: #666;
            font-size: 1.1em;
        }}

        .stat-card.excellent {{ border-top: 4px solid #06A77D; }}
        .stat-card.excellent .number {{ color: #06A77D; }}

        .stat-card.warning {{ border-top: 4px solid #F4B41A; }}
        .stat-card.warning .number {{ color: #F4B41A; }}

        .stat-card.danger {{ border-top: 4px solid #D64933; }}
        .stat-card.danger .number {{ color: #D64933; }}

        .stat-card.info {{ border-top: 4px solid #5C7CFA; }}
        .stat-card.info .number {{ color: #5C7CFA; }}

        .section {{
            background: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            margin-bottom: 30px;
        }}

        .section h2 {{
            color: #2E4057;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 3px solid #667eea;
        }}

        #map {{
            height: 600px;
            border-radius: 10px;
            margin-bottom: 20px;
        }}

        .chart-container {{
            margin: 20px 0;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}

        table thead {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }}

        table th, table td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}

        table tbody tr:hover {{
            background-color: #f5f5f5;
        }}

        .badge {{
            padding: 5px 10px;
            border-radius: 20px;
            font-size: 0.9em;
            font-weight: bold;
            display: inline-block;
        }}

        .badge.excellent {{ background: #06A77D; color: white; }}
        .badge.good {{ background: #48C774; color: white; }}
        .badge.fair {{ background: #F4B41A; color: white; }}
        .badge.poor {{ background: #FF9800; color: white; }}
        .badge.critical {{ background: #D64933; color: white; }}

        .tabs {{
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
            border-bottom: 2px solid #ddd;
        }}

        .tab {{
            padding: 12px 24px;
            cursor: pointer;
            border: none;
            background: none;
            font-size: 1.1em;
            color: #666;
            border-bottom: 3px solid transparent;
            transition: all 0.3s ease;
        }}

        .tab:hover {{
            color: #667eea;
        }}

        .tab.active {{
            color: #667eea;
            border-bottom-color: #667eea;
            font-weight: bold;
        }}

        .tab-content {{
            display: none;
        }}

        .tab-content.active {{
            display: block;
        }}

        .recommendation-item {{
            background: #f8f9fa;
            padding: 15px;
            margin: 10px 0;
            border-left: 4px solid #667eea;
            border-radius: 5px;
        }}

        .recommendation-item.priority-1 {{ border-left-color: #D64933; }}
        .recommendation-item.priority-2 {{ border-left-color: #F4B41A; }}
        .recommendation-item.priority-3 {{ border-left-color: #5C7CFA; }}
        .recommendation-item.priority-4 {{ border-left-color: #06A77D; }}

        .recommendation-item h4 {{
            color: #2E4057;
            margin-bottom: 8px;
        }}

        .recommendation-item p {{
            color: #666;
            margin: 5px 0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚍 세종시 버스 노선 최적화 종합 대시보드</h1>
            <p>분석일자: {report['analysis_date']}</p>
        </div>

        <!-- 핵심 통계 카드 -->
        <div class="stats-grid">
            <div class="stat-card info">
                <div class="label">총 노선 수</div>
                <div class="number">{report['total_routes']}</div>
                <div class="label">개 노선</div>
            </div>
            <div class="stat-card excellent">
                <div class="label">고효율 노선</div>
                <div class="number">{report['high_efficiency_routes']}</div>
                <div class="label">높음/매우높음</div>
            </div>
            <div class="stat-card danger">
                <div class="label">서비스 부족 지역</div>
                <div class="number">{report['underserved_regions']}</div>
                <div class="label">미커버/부족</div>
            </div>
            <div class="stat-card warning">
                <div class="label">개선 필요 노선</div>
                <div class="number">{report['low_efficiency_routes']}</div>
                <div class="label">저효율 노선</div>
            </div>
            <div class="stat-card info">
                <div class="label">최적화 제안</div>
                <div class="number">{report['optimization_recommendations']}</div>
                <div class="label">건</div>
            </div>
        </div>

        <!-- 탭 네비게이션 -->
        <div class="section">
            <div class="tabs">
                <button class="tab active" onclick="showTab('overview')">📊 종합 개요</button>
                <button class="tab" onclick="showTab('routes')">🚌 노선 분석</button>
                <button class="tab" onclick="showTab('regions')">🗺️ 지역 분석</button>
                <button class="tab" onclick="showTab('recommendations')">💡 최적화 제안</button>
                <button class="tab" onclick="showTab('overlap')">🔄 노선 중복</button>
            </div>

            <!-- 종합 개요 탭 -->
            <div id="overview" class="tab-content active">
                <h2>📊 종합 개요</h2>

                <div class="chart-container">
                    <div id="efficiency-chart"></div>
                </div>

                <div class="chart-container">
                    <div id="service-level-chart"></div>
                </div>

                <div class="chart-container">
                    <div id="demand-chart"></div>
                </div>
            </div>

            <!-- 노선 분석 탭 -->
            <div id="routes" class="tab-content">
                <h2>🚌 노선별 상세 분석</h2>

                <table id="routes-table">
                    <thead>
                        <tr>
                            <th>노선번호</th>
                            <th>정류장 수</th>
                            <th>노선 길이(km)</th>
                            <th>커버 지역</th>
                            <th>총 커버 수요</th>
                            <th>수요밀도(/km)</th>
                            <th>효율성</th>
                        </tr>
                    </thead>
                    <tbody>
"""

    # 노선 테이블 생성
    for _, route in route_df.nlargest(20, '수요밀도_per_km').iterrows():
        efficiency_badge = {
            '매우높음': 'excellent',
            '높음': 'good',
            '보통': 'fair',
            '낮음': 'poor'
        }.get(route['효율성등급'], 'fair')

        html_content += f"""
                        <tr>
                            <td><strong>{route['노선번호']}</strong></td>
                            <td>{route['정류장수']}</td>
                            <td>{route['노선길이_km']:.2f}</td>
                            <td>{route['커버지역수']}</td>
                            <td>{route['총커버수요']:,.0f}</td>
                            <td>{route['수요밀도_per_km']:,.0f}</td>
                            <td><span class="badge {efficiency_badge}">{route['효율성등급']}</span></td>
                        </tr>
"""

    html_content += """
                    </tbody>
                </table>
            </div>

            <!-- 지역 분석 탭 -->
            <div id="regions" class="tab-content">
                <h2>🗺️ 지역별 서비스 수준</h2>

                <table id="regions-table">
                    <thead>
                        <tr>
                            <th>행정구역</th>
                            <th>총 수요</th>
                            <th>노선 수</th>
                            <th>노선 목록</th>
                            <th>수요/노선</th>
                            <th>서비스 수준</th>
                        </tr>
                    </thead>
                    <tbody>
"""

    # 지역 테이블 생성
    for _, region in region_df.iterrows():
        service_badge = {
            '미커버': 'critical',
            '부족': 'poor',
            '적정': 'excellent',
            '과잉': 'good'
        }.get(region['서비스수준'], 'fair')

        html_content += f"""
                        <tr>
                            <td><strong>{region['행정구역']}</strong></td>
                            <td>{region['총수요']:,.0f}</td>
                            <td>{region['노선수']}</td>
                            <td>{region['노선목록']}</td>
                            <td>{region['수요_per_노선']:,.0f}</td>
                            <td><span class="badge {service_badge}">{region['서비스수준']}</span></td>
                        </tr>
"""

    html_content += """
                    </tbody>
                </table>
            </div>

            <!-- 최적화 제안 탭 -->
            <div id="recommendations" class="tab-content">
                <h2>💡 노선 최적화 제안</h2>
"""

    # 최적화 제안 항목 생성
    for _, rec in recommendations_df.head(20).iterrows():
        html_content += f"""
                <div class="recommendation-item priority-{rec['우선순위']}">
                    <h4>📍 {rec['대상']} - {rec['유형']}</h4>
                    <p><strong>현재 상태:</strong> {rec['현재상태']}</p>
                    <p><strong>제안 사항:</strong> {rec['제안사항']}</p>
                    <p><strong>예상 효과:</strong> {rec['예상효과']}</p>
                </div>
"""

    html_content += """
            </div>

            <!-- 노선 중복 탭 -->
            <div id="overlap" class="tab-content">
                <h2>🔄 노선 중복도 분석</h2>

                <table id="overlap-table">
                    <thead>
                        <tr>
                            <th>노선 1</th>
                            <th>노선 2</th>
                            <th>공유 정류장 수</th>
                            <th>중복 비율 (%)</th>
                        </tr>
                    </thead>
                    <tbody>
"""

    # 중복도 테이블 생성
    for _, overlap in overlap_df.head(15).iterrows():
        html_content += f"""
                        <tr>
                            <td><strong>{overlap['노선1']}</strong></td>
                            <td><strong>{overlap['노선2']}</strong></td>
                            <td>{overlap['공유정류장수']}</td>
                            <td>{overlap['중복비율_%']:.1f}%</td>
                        </tr>
"""

    html_content += """
                    </tbody>
                </table>
            </div>
        </div>
    </div>

    <script>
        // 탭 전환 함수
        function showTab(tabId) {
            // 모든 탭과 컨텐츠 비활성화
            document.querySelectorAll('.tab').forEach(tab => tab.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));

            // 선택된 탭과 컨텐츠 활성화
            event.target.classList.add('active');
            document.getElementById(tabId).classList.add('active');
        }

        // 효율성 차트 데이터
        const routeData = """ + route_df.nlargest(15, '수요밀도_per_km')[['노선번호', '수요밀도_per_km']].to_json(orient='records') + """;

        const efficiencyTrace = {
            x: routeData.map(r => r.수요밀도_per_km),
            y: routeData.map(r => String(r.노선번호)),
            type: 'bar',
            orientation: 'h',
            marker: {
                color: routeData.map(r => {
                    if (r.수요밀도_per_km > 200000) return '#06A77D';
                    if (r.수요밀도_per_km > 100000) return '#48C774';
                    if (r.수요밀도_per_km > 50000) return '#F4B41A';
                    return '#FF9800';
                })
            }
        };

        const efficiencyLayout = {
            title: '노선별 수요 밀도 Top 15',
            xaxis: { title: '수요 밀도 (명/km)' },
            yaxis: { title: '노선번호' },
            height: 500
        };

        Plotly.newPlot('efficiency-chart', [efficiencyTrace], efficiencyLayout);

        // 서비스 수준 차트
        const regionData = """ + region_df['서비스수준'].value_counts().to_json() + """;

        const serviceLevelTrace = {
            labels: Object.keys(regionData),
            values: Object.values(regionData),
            type: 'pie',
            marker: {
                colors: ['#D64933', '#F4B41A', '#06A77D', '#5C7CFA']
            }
        };

        const serviceLevelLayout = {
            title: '지역별 서비스 수준 분포',
            height: 400
        };

        Plotly.newPlot('service-level-chart', [serviceLevelTrace], serviceLevelLayout);

        // 수요 차트
        const demandData = """ + region_df.nlargest(15, '총수요')[['행정구역', '총수요', '노선수']].to_json(orient='records') + """;

        const demandTrace = {
            x: demandData.map(r => r.행정구역),
            y: demandData.map(r => r.총수요),
            type: 'bar',
            marker: {
                color: demandData.map(r => {
                    if (r.노선수 === 0) return '#D64933';
                    if (r.노선수 < 3) return '#F4B41A';
                    if (r.노선수 <= 5) return '#06A77D';
                    return '#5C7CFA';
                })
            }
        };

        const demandLayout = {
            title: '지역별 총 수요 Top 15 (색상: 서비스 수준)',
            xaxis: { title: '행정구역' },
            yaxis: { title: '총 수요 (명)' },
            height: 500
        };

        Plotly.newPlot('demand-chart', [demandTrace], demandLayout);
    </script>
</body>
</html>
"""

    # HTML 파일 저장
    with open('버스노선_최적화_대시보드.html', 'w', encoding='utf-8') as f:
        f.write(html_content)

    print("✅ 종합 대시보드 저장: 버스노선_최적화_대시보드.html")

if __name__ == '__main__':
    create_comprehensive_dashboard()
