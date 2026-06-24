# 🌍 공공기관 녹색구매 위치 찾기 서비스

🔗 **Demo:** https://green-echo-dashboard-demo.streamlit.app/

> The demo may take a few moments to load if inactive.

## 프로젝트 소개

공공기관의 최근 3년간 녹색구매 데이터를 분석하여 기관의 현재 위치를 진단하고, 유사한 구매 패턴을 가진 기관을 탐색할 수 있는 데이터 분석 서비스입니다.

최근 3년 평균 구매비율, 구매규모, 구매 추세 등을 종합적으로 분석하여 기관의 상대적 위치와 특성을 시각화합니다.

---

## 주요 기능

* 기관명 검색 기반 분석
* 최근 3년 녹색구매 실적 분석
* K-Means 기반 구매패턴 군집 분석
* 기관 유형별 상대 위치 진단
* 구매패턴 자동 분류
* 유사 기관 탐색
* 인터랙티브 대시보드 제공
* 기관별 구매 실적 추이 시각화

---

## 주요 분석 지표

* 최근 3년 녹색구매비율 평균
* 최근 3년 녹색구매액 평균
* 최근 3년 총구매액 평균
* 최근 3년 구매비율 변화량
* 기관 유형 평균 대비 격차

---

## 기술 스택

* Python
* Streamlit
* Pandas
* NumPy
* Plotly
* Scikit-learn (K-Means Clustering)

---

## 실행 방법

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 프로젝트 목적

공공기관의 녹색구매 현황을 보다 직관적으로 이해할 수 있도록 지원하고, 유사 기관 비교와 데이터 기반 의사결정을 돕기 위해 개발되었습니다.

---

# 🌍 Green Purchasing Position Finder for Public Institutions

## Project Overview

This project is a data analytics service that helps public institutions analyze their green purchasing performance and identify organizations with similar purchasing patterns.

By analyzing purchasing ratios, purchasing scale, and historical trends over the past three years, the service visualizes the relative position and characteristics of each institution through an interactive dashboard.

---

## Key Features

* Institution search and analysis
* Green purchasing performance analysis
* K-Means clustering-based purchasing pattern classification
* Relative position analysis within the same institution type
* Similar institution recommendation
* Interactive dashboard visualization
* Historical purchasing performance tracking

---

## Key Metrics

* Average green purchasing ratio (last 3 years)
* Average green purchasing amount (last 3 years)
* Average total purchasing amount (last 3 years)
* Change in purchasing ratio over the last 3 years
* Gap from the average of the same institution type

---

## Tech Stack

* Python
* Streamlit
* Pandas
* NumPy
* Plotly
* Scikit-learn (K-Means Clustering)

---

## Getting Started

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## Project Goal

To help public institutions better understand their green purchasing performance, compare themselves with similar organizations, and support data-driven decision-making through intuitive visualization and analysis.
