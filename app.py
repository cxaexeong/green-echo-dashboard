import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler
from sklearn.metrics.pairwise import euclidean_distances

# ===================================================
# 0. 전역 변수 설정
# ===================================================
CURRENT_YEAR = 2025
ANALYSIS_YEARS = list(range(CURRENT_YEAR - 2, CURRENT_YEAR + 1))
FILE_PATH = "integrated_5years_data_clean_v2.csv"

# ===================================================
# 1. 데이터 로드 및 전처리 (캐싱 적용)
# ===================================================
@st.cache_data
def load_data():
    df = pd.read_csv(FILE_PATH)
    
    df["연도"] = pd.to_numeric(df["연도"], errors="coerce").astype("Int64")
    numeric_cols = ["총구매액", "녹색구매액", "구매비율"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    
    df = df.dropna(subset=["기관명", "기관코드", "연도", "총구매액", "녹색구매액", "구매비율"]).copy()
    df["연도"] = df["연도"].astype(int)
    df["기관명"] = df["기관명"].astype(str).str.strip()
    df["기관코드"] = df["기관코드"].astype(str).str.strip()
    df["기관유형"] = df["기관유형"].astype(str).str.strip()

    org_type = df["기관유형"]
    org_name = df["기관명"]
    conditions = [
        org_type.isin(["시장형공기업", "준시장형공기업"]),
        org_type.isin(["기타공공기관", "지방자치단체출연연구원", "지방자치단체출자출연기관", "특수법인기관"]) |
        ((org_type == "지방공사지방공단") & org_name.str.contains("의료원", na=False)),
    ]
    choices = ["공기업", "기타 및 의료기관"]
    df["통합기관유형"] = np.select(conditions, choices, default=org_type)
    
    return df

df = load_data()

# ===================================================
# 2. UI 및 분석 로직 시작
# ===================================================
# 넓은 화면(Wide mode) 설정 (대시보드가 크기 때문에 추천합니다)
st.set_page_config(layout="wide") 

st.title("🌍 공공기관 녹색구매 위치 찾기 서비스")

search_name = st.text_input("검색할 기관명을 입력하세요 (산업연구원, 한국환경산업기술원 등):").strip()

if search_name:
    find_target = df[(df["연도"] == CURRENT_YEAR) & (df["기관명"] == search_name)].copy()
    
    if find_target.empty:
        find_target = df[(df["연도"] == CURRENT_YEAR) & (df["기관명"].str.contains(search_name, case=False, na=False, regex=False))].copy()

    if find_target.empty:
        st.warning(f"'{search_name}' 기관을 찾을 수 없습니다.")
    else:
        target_row_current = find_target.iloc[0]
        SELECTED_TYPE = target_row_current["통합기관유형"]
        
        st.success("[기관 매칭 성공]")
        st.write(f"검색된 기관명: **{target_row_current['기관명']}**")
        st.write(f"자동 판별된 통합기관유형: **{SELECTED_TYPE}**")

        # ===================================================
        # 4. 최근 3개년 기준 분석용 특징 생성
        # ===================================================
        if SELECTED_TYPE:
            df_filtered = df[df["통합기관유형"] == SELECTED_TYPE].copy()
        else:
            df_filtered = df.copy()

        df_filtered = df_filtered.sort_values(["기관코드", "연도"]).copy()

        feature_rows = []
        df_filtered_3y = df_filtered[df_filtered["연도"].isin(ANALYSIS_YEARS)].copy()
        type_ratio_mean = df_filtered_3y.groupby("기관코드")["구매비율"].mean().mean()

        for org_code, group in df_filtered.groupby("기관코드"):
            group = group.sort_values("연도").copy()
            recent_data = group[group["연도"].isin(ANALYSIS_YEARS)].copy()
            if recent_data.empty: continue

            latest_row = recent_data.sort_values("연도").iloc[-1]
            avg_total_amount = recent_data["총구매액"].mean()
            avg_green_amount = recent_data["녹색구매액"].mean()
            avg_ratio = recent_data["구매비율"].mean()

            if len(recent_data) >= 2:
                recent3_ratio_change = recent_data.iloc[-1]["구매비율"] - recent_data.iloc[0]["구매비율"]
            else:
                recent3_ratio_change = 0

            gap_from_mean = avg_ratio - type_ratio_mean
            log_avg_total_amount = np.log1p(max(avg_total_amount, 0))

            feature_rows.append({
                "기관코드": org_code, "기관명": latest_row["기관명"], "기관유형": latest_row["통합기관유형"],
                "최근3년_총구매액_평균": avg_total_amount, "최근3년_녹색구매액_평균": avg_green_amount,
                "최근3년_구매비율_평균": avg_ratio, "유형평균_대비_격차": gap_from_mean,
                "log_최근3년_총구매액_평균": log_avg_total_amount, "최근3년_구매비율_변화량": recent3_ratio_change,
                "사용연도수": len(recent_data),
            })

        df_features = pd.DataFrame(feature_rows)
        df_features["특수분류"] = "일반"

        if SELECTED_TYPE == "기타 및 의료기관":
            zero_mask = ((df_features["최근3년_구매비율_평균"] <= 0) | (df_features["최근3년_총구매액_평균"] <= 0))
            df_features.loc[zero_mask, "특수분류"] = "구매실적 미미/미이행"

        df_features["최근3년_구매비율_변화량_포맷"] = df_features["최근3년_구매비율_변화량"].apply(lambda val: f"{val:+.2f}%p")

        def format_amount_won(amount):
            if amount >= 100000000: return f"{amount/100000000:.1f}억원"
            elif amount >= 10000: return f"{amount/10000:.0f}만원"
            else: return f"{amount:,.0f}원"

        # ===================================================
        # 5. AI K-Means 군집분석
        # ===================================================
        ratio_col = "최근3년_구매비율_평균"
        green_amount_col = "최근3년_녹색구매액_평균"
        total_amount_col = "최근3년_총구매액_평균"
        log_total_amount_col = "log_최근3년_총구매액_평균"
        trend_col = "최근3년_구매비율_변화량"

        cluster_features = [ratio_col, log_total_amount_col]
        df_cluster_base = df_features[df_features["특수분류"] == "일반"].copy()
        X = df_cluster_base[cluster_features].copy()
        clip_bounds = {}

        for col in cluster_features:
            lower_limit = X[col].quantile(0.01)
            upper_limit = X[col].quantile(0.99)
            clip_bounds[col] = (lower_limit, upper_limit)
            X[col] = np.clip(X[col], lower_limit, upper_limit)

        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
        n_samples = len(df_cluster_base)
        n_clusters = min(3, n_samples)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        df_cluster_base["AI_군집번호"] = kmeans.fit_predict(X_scaled)

        df_features = df_features.merge(df_cluster_base[["기관코드", "AI_군집번호"]], on="기관코드", how="left")
        cluster_name_map = {0: "구매패턴 A", 1: "구매패턴 B", 2: "구매패턴 C"}
        df_features["AI_구매패턴"] = df_features["AI_군집번호"].map(cluster_name_map)
        df_features.loc[df_features["특수분류"] == "구매실적 미미/미이행", "AI_구매패턴"] = "구매실적 미미/미이행"

        # ===================================================
        # 구매패턴별 특성 설명 생성
        # ===================================================
        pattern_summary_df = (
            df_features[df_features["특수분류"] == "일반"]
            .groupby("AI_구매패턴")
            .agg(
                기관수=("기관코드", "count"), 평균_구매비율=(ratio_col, "mean"), 평균_녹색구매액=(green_amount_col, "mean"),
                평균_총구매액=(total_amount_col, "mean"), 평균_변화량=(trend_col, "mean"),
            ).reset_index()
        )

        ratio_median = pattern_summary_df["평균_구매비율"].median()
        total_median = pattern_summary_df["평균_총구매액"].median()
        trend_median = pattern_summary_df["평균_변화량"].median()

        def make_pattern_description(row):
            ratio_level = "고비율" if row["평균_구매비율"] >= ratio_median else "저비율"
            scale_level = "대규모" if row["평균_총구매액"] >= total_median else "소규모"
            trend_level = "상승" if row["평균_변화량"] >= trend_median else "정체·감소"
            return f"{ratio_level}·{scale_level}·{trend_level}형 (기관 {int(row['기관수'])}개, 평균 구매비율 {row['평균_구매비율']:.1f}%, 평균 녹색구매액 {row['평균_녹색구매액']:,.0f}원)"

        pattern_summary_df["패턴설명"] = pattern_summary_df.apply(make_pattern_description, axis=1)
        pattern_desc_map = dict(zip(pattern_summary_df["AI_구매패턴"], pattern_summary_df["패턴설명"]))

        df_features["AI_구매패턴_설명"] = df_features["AI_구매패턴"].map(pattern_desc_map)
        df_features.loc[df_features["AI_구매패턴"] == "구매실적 미미/미이행", "AI_구매패턴_설명"] = "최근 3년 구매비율 또는 총구매액이 0에 가까운 별도 관리 대상"

        centroids_original = scaler.inverse_transform(kmeans.cluster_centers_)
        centroid_df = pd.DataFrame(centroids_original, columns=cluster_features)
        centroid_df["AI_군집번호"] = range(n_clusters)

        # ===================================================
        # 구매패턴 A/B/C 그룹별 핵심 지표 요약
        # ===================================================
        pattern_stats = (
            df_features[df_features["특수분류"] == "일반"]
            .groupby("AI_구매패턴")
            .agg(기관수=("기관코드", "count"), 평균_구매비율=(ratio_col, "mean"), 평균_총구매액=(total_amount_col, "mean"), 평균_구매비율_변화량=(trend_col, "mean"))
            .reset_index().sort_values("AI_구매패턴")
        )

        def rank_text(rank, total):
            if rank == 1: return "가장 높은"
            elif rank == total: return "가장 낮은"
            else: return "중간 수준의"

        def make_all_pattern_summary_text(stats_df):
            stats_df = stats_df.copy()
            n_patterns = len(stats_df)
            stats_df["구매비율_순위"] = stats_df["평균_구매비율"].rank(ascending=False, method="min").astype(int)
            stats_df["총구매액_순위"] = stats_df["평균_총구매액"].rank(ascending=False, method="min").astype(int)
            stats_df["변화량_순위"] = stats_df["평균_구매비율_변화량"].rank(ascending=False, method="min").astype(int)

            lines = []
            for _, row in stats_df.iterrows():
                pattern = row["AI_구매패턴"]
                ratio_desc = rank_text(row["구매비율_순위"], n_patterns)
                amount_desc = rank_text(row["총구매액_순위"], n_patterns)
                trend_desc = rank_text(row["변화량_순위"], n_patterns)
                lines.append(
                    f"<b>{pattern}</b><br>- 최근 3년 평균 구매비율: {row['평균_구매비율']:.2f}%<br>- 최근 3년 총구매액 평균: {row['평균_총구매액']:,.0f}원<br>"
                    f"- 최근 3년 구매비율 변화량 평균: {row['평균_구매비율_변화량']:+.2f}%p<br>- 추가 설명<br>"
                    f"ㆍ평균 구매비율은 {ratio_desc} 수준입니다.<br>ㆍ{pattern}은 {n_patterns}개 패턴 중 총구매액 규모가 {amount_desc} 기관군입니다.<br>"
                    f"ㆍ구매비율 변화량은 {trend_desc} 수준으로 나타납니다."
                )
            return "<br><br>".join(lines)

        all_pattern_summary_text = make_all_pattern_summary_text(pattern_stats)
        df_features["AI_구매패턴_비교설명"] = all_pattern_summary_text
        df_features.loc[df_features["AI_구매패턴"] == "구매실적 미미/미이행", "AI_구매패턴_비교설명"] = "최근 3년 구매비율 또는 총구매액이 0에 가까워 일반 구매패턴 비교에서는 별도 관리 대상입니다."

        normal_df = df_features[df_features["특수분류"] == "일반"].copy()
        ai_ratio_boundary = normal_df[ratio_col].median()
        ai_amount_boundary = normal_df[green_amount_col].median()

        def get_ai_quadrant_label(ratio, amount, r_bound, a_bound):
            high_ratio, high_amount = ratio >= r_bound, amount >= a_bound
            if high_ratio and high_amount: return "우수형"
            elif high_ratio and not high_amount: return "성과유지형"
            elif not high_ratio and high_amount: return "잠재성장형"
            else: return "개선필요형"

        df_features["AI_사분면"] = df_features.apply(lambda row: get_ai_quadrant_label(row[ratio_col], row[green_amount_col], ai_ratio_boundary, ai_amount_boundary), axis=1)

        # ===================================================
        # 7. 상대 위치 지표 및 유사기관
        # ===================================================
        df_features["구매비율_백분위"] = df_features[ratio_col].rank(pct=True) * 100
        df_features["녹색구매액_백분위"] = df_features[green_amount_col].rank(pct=True) * 100
        df_features["총구매액_백분위"] = df_features["최근3년_총구매액_평균"].rank(pct=True) * 100
        df_features["최근3년변화_백분위"] = df_features[trend_col].rank(pct=True) * 100

        def top_percentile_text(pct): return f"상위 {100 - pct:.1f}%"

        target_df = df_features[df_features["기관명"] == target_row_current["기관명"]].copy()

        # ===================================================
        # 8. 기관별 진단 리포트 생성
        # ===================================================
        if not target_df.empty:
            target_row = target_df.iloc[0]
            ai_status = target_row["AI_사분면"]
            pattern_name = target_row["AI_구매패턴"]
            gap_str = f"{target_row['유형평균_대비_격차']:+.2f}%p"
            recent3_str = f"{target_row[trend_col]:+.2f}%p"
            current_ratio = target_row[ratio_col]
            ratio_rank = top_percentile_text(target_row["구매비율_백분위"])
            recent_rank = top_percentile_text(target_row["최근3년변화_백분위"])
            green_amount_str = f"{target_row[green_amount_col]:,.0f}원"
            green_amount_rank = top_percentile_text(target_row["녹색구매액_백분위"])

            ai_report = {
                "우수형": {"pattern": "고성과·상승형", "analysis": "최근 3년 평균 구매비율과 녹색구매액 평균이 모두 AI 기준선 이상입니다."},
                "성과유지형": {"pattern": "고성과·유지형", "analysis": "최근 3년 평균 구매비율은 높지만 녹색구매액 평균은 상대적으로 낮은 위치입니다."},
                "잠재성장형": {"pattern": "저성과·상승형", "analysis": "최근 3년 평균 구매비율은 낮지만 녹색구매액 평균은 높은 위치입니다."},
                "개선필요형": {"pattern": "저성과·정체형", "analysis": "최근 3년 평균 구매비율과 녹색구매액 평균이 모두 AI 기준선보다 낮은 위치입니다."},
            }
            report_data = ai_report[ai_status]

            txt_report_main = f"현재 위치: <b>{ai_status} ({report_data['pattern']})</b><br><br>{report_data['analysis']}"
            txt_ai_position = (
                f"- 현재 검색 기관: {pattern_name}<br>- 검색기관 구매비율: {current_ratio:.2f}% ({ratio_rank})<br>"
                f"- 최근3년 녹색구매액 평균: {green_amount_str} ({green_amount_rank})<br>- 최근3년 구매비율 변화량: {recent3_str} ({recent_rank})<br>"
                f"- 동일유형 평균대비 격차: {gap_str}"
            )
            txt_ai_criteria = (
                "구매패턴 A/B/C는 최근 3년 구매 실적이 비슷한 기관들을 묶어 구분한 유형입니다.<br><br>구분 기준은 다음 2가지입니다.<br><br>"
                "&nbsp;&nbsp;① 최근 3년 구매비율 평균(%)<br>&nbsp;&nbsp;② 최근 3년 총구매액 평균(억 원)<br><br>"
                "즉, 구매비율 수준과 기관의 구매규모(총구매액)를 함께 고려하여<br>유사한 기관끼리 묶은 결과입니다."
            )
            txt_ai_baseline = "- 가로 기준선: 최근 3년 구매비율 평균 중앙값<br>- 세로 기준선: 최근 3년 녹색구매액 평균 중앙값"
            txt_ai_ref = "- 본 분석은 최근 3개년(2023~2025년) 데이터를 기반으로 합니다.<br>- '구매실적 미미/미이행' 기관은 군집분석에서 제외되었습니다."

        # ===================================================
        # 9. 대시보드 생성 함수 
        # ===================================================
        def make_current_dashboard():
            fig = make_subplots(
                rows=4,
                cols=2,
                column_widths=[0.64, 0.36],
                row_heights=[0.51, 0.08, 0.23, 0.29],
                horizontal_spacing=0.04,
                vertical_spacing=0.02,
                specs=[
                    [{"type": "scatter"}, {"type": "xy", "secondary_y": True}],
                    [None, None],
                    [{"type": "table"}, {"type": "table", "rowspan": 2}],
                    [{"type": "table"}, None],
                ],
            )

            x_col, y_col = green_amount_col, ratio_col

            x_lower_bound = df_features[x_col].quantile(0.05)
            x_upper_bound = df_features[x_col].quantile(0.95)
            x_min_raw, x_max_raw = df_features[x_col].min(), df_features[x_col].max()

            x_min = max(x_min_raw, x_lower_bound - abs(x_lower_bound) * 0.5)
            x_max = min(x_max_raw, x_upper_bound + abs(x_upper_bound) * 0.5)

            y_min, y_max = df_features[y_col].min(), df_features[y_col].max()

            x_pad = (x_max - x_min) * 0.12 if x_max != x_min else 0.5
            y_pad = (y_max - y_min) * 0.12 if y_max != y_min else 1

            x_range = [x_min - x_pad, x_max + x_pad]
            y_range = [y_min - y_pad, y_max + y_pad]

            if not target_df.empty:
                target_x = target_df[x_col].iloc[0]
                target_y = target_df[y_col].iloc[0]

                x_range[0] = min(x_range[0], target_x - abs(target_x) * 0.15 - 0.5)
                x_range[1] = max(x_range[1], target_x + abs(target_x) * 0.15 + 0.5)

                y_range[0] = min(y_range[0], target_y - abs(target_y) * 0.10 - 1)
                y_range[1] = max(y_range[1], target_y + abs(target_y) * 0.10 + 1)

            ai_amount_line = ai_amount_boundary
            ai_ratio_line = ai_ratio_boundary

            x_positive = df_features[x_col][df_features[x_col] > 0]
            log_x_min_axis = np.log10(x_positive.min()) - 0.15
            log_x_max_axis = np.log10(x_positive.max()) + 0.35

            # 좌상단 산점도
            cluster_colors = {
                "구매패턴 A": "#A8D8EA",
                "구매패턴 B": "#AAE3A1",
                "구매패턴 C": "#F7C8E0",
                "구매실적 미미/미이행": "#D3D3D3",
            }

            for pattern_name, color in cluster_colors.items():
                subset = df_features[df_features["AI_구매패턴"] == pattern_name]

                fig.add_trace(
                    go.Scatter(
                        x=subset[x_col],
                        y=subset[y_col],
                        mode="markers",
                        name=pattern_name,
                        showlegend=False,
                        marker=dict(color=color, size=10, opacity=0.85, line=dict(width=0.5, color="gray")),
                        hovertext=subset["기관명"],
                        hovertemplate=(
                            "<b>%{hovertext}</b><br>"
                            "최근 3년 구매비율 평균: %{y:.2f}%<br>"
                            "최근 3년 녹색구매액 평균: %{x:,.0f}원<br>"
                            "최근 3년 구매비율 변화량: %{customdata[0]}<br>"
                            "사용 연도 수: %{customdata[1]}년"
                            "<extra></extra>"
                        ),
                        customdata=subset[["최근3년_구매비율_변화량_포맷", "사용연도수"]].values,
                    ),
                    row=1, col=1,
                )

            if not target_df.empty:
                fig.add_trace(
                    go.Scatter(
                        x=target_df[x_col],
                        y=target_df[y_col],
                        mode="markers+text",
                        name="검색 기관",
                        showlegend=False,
                        marker=dict(size=24, symbol="star", color="yellow", line=dict(width=1.8, color="black")),
                        text=[f"<span style='color:blue; font-weight:bold; font-size:13px;'>{target_df.iloc[0]['기관명']}</span>"],
                        textposition="top center",
                        hovertemplate=(
                            "<b>%{text}</b><br>"
                            "최근 3년 구매비율 평균: %{y:.2f}%<br>"
                            "최근 3년 녹색구매액 평균: %{x:,.0f}원"
                            "<extra></extra>"
                        ),
                    ),
                    row=1, col=1,
                )

            # 하단 텍스트 및 테이블 구성
            if not target_df.empty:
                hdr_bg, bdy_bg = "#eef3fb", "white"
                hdr_color, bdy_color = "#1f3a5f", "#1f2933"
                tbl_font = dict(size=15, family="Malgun Gothic")
                tbl_line, trans_col = "rgba(230,230,230,0.8)", "rgba(0,0,0,0)"
                box_bg, box_border = "#f2f6fc", "#bacce4"

                fig.add_trace(
                    go.Table(
                        header=dict(values=[""], height=0, line_color=trans_col),
                        cells=dict(
                            values=[["<b>[ 분석 종합 리포트 ]</b>", txt_report_main, "", "<b>[ AI 구매패턴 및 상대 위치 ]</b>", txt_ai_position]],
                            fill_color=[[hdr_bg, bdy_bg, trans_col, hdr_bg, box_bg]],
                            line_color=[[tbl_line, tbl_line, trans_col, tbl_line, box_border]],
                            font=dict(color=[[hdr_color, bdy_color, trans_col, hdr_color, bdy_color]], **tbl_font),
                            align="left", height=28,
                        ),
                    ), row=3, col=1,
                )

                fig.add_trace(
                    go.Table(
                        header=dict(values=[""], height=0, line_color=trans_col),
                        cells=dict(
                            values=[["<b>[ AI 군집분석 기준 ]</b>", txt_ai_criteria, "", "<b>[ 분석 기준선(사분면 기준) ]</b>", txt_ai_baseline, "<b>[ 참고 ]</b>", txt_ai_ref, ""]],
                            fill_color=[[hdr_bg, bdy_bg, trans_col, hdr_bg, bdy_bg, hdr_bg, bdy_bg, trans_col]],
                            line_color=[[tbl_line, tbl_line, trans_col, tbl_line, tbl_line, tbl_line, tbl_line, trans_col]],
                            font=dict(color=[[hdr_color, bdy_color, trans_col, hdr_color, bdy_color, hdr_color, bdy_color, trans_col]], **tbl_font),
                            align="left", height=28,
                        ),
                    ), row=3, col=2,
                )

                def get_pattern_html(pattern):
                    row = pattern_stats[pattern_stats["AI_구매패턴"] == pattern]
                    if row.empty: return f"<b>{pattern}</b><br><br>데이터 없음"
                    row = row.iloc[0]
                    tmp = pattern_stats.copy()
                    tmp["구매비율_순위"] = tmp["평균_구매비율"].rank(ascending=False, method="min").astype(int)
                    tmp["총구매액_순위"] = tmp["평균_총구매액"].rank(ascending=False, method="min").astype(int)
                    tmp["변화량_순위"] = tmp["평균_구매비율_변화량"].rank(ascending=False, method="min").astype(int)
                    r_rank = tmp[tmp["AI_구매패턴"] == pattern].iloc[0]
                    n_pat = len(pattern_stats)
                    return (
                        f"<b>{pattern}</b><br><br>- 최근 3년 평균 구매비율: {row['평균_구매비율']:.2f}%<br>"
                        f"- 최근 3년 총구매액 평균: {format_amount_won(row['평균_총구매액'])}<br>"
                        f"- 최근 3년 구매비율 변화량 평균: {row['평균_구매비율_변화량']:+.2f}%p<br><br>- 추가 설명<br>"
                        f"&nbsp;&nbsp;ㆍ평균 구매비율은 {rank_text(r_rank['구매비율_순위'], n_pat)} 수준입니다.<br>"
                        f"&nbsp;&nbsp;ㆍ총구매액 규모는 {rank_text(r_rank['총구매액_순위'], n_pat)} 기관군입니다.<br>"
                        f"&nbsp;&nbsp;ㆍ구매비율 변화량은 {rank_text(r_rank['변화량_순위'], n_pat)} 수준입니다."
                    )

                fig.add_trace(
                    go.Table(
                        columnwidth=[0.33, 0.33, 0.33],
                        header=dict(values=["", "", ""], height=0, line_color=trans_col),
                        cells=dict(
                            values=[[get_pattern_html("구매패턴 A"), ""], [get_pattern_html("구매패턴 B"), ""], [get_pattern_html("구매패턴 C"), ""]],
                            fill_color=[["white", trans_col]] * 3,
                            line_color=[[tbl_line, trans_col]] * 3,
                            font=dict(size=15, color=bdy_color, family="Malgun Gothic"),
                            align="left", height=28,
                        ),
                    ), row=4, col=1,
                )

            # 우상단 최근 3년 실적 추이
            trend_df = pd.DataFrame()
            if not target_df.empty:
                target_org_code = target_df.iloc[0]["기관코드"]
                trend_df = df[(df["기관코드"] == target_org_code) & (df["연도"].isin(ANALYSIS_YEARS))].sort_values("연도").copy()
                trend_df["녹색구매액_억원"] = trend_df["녹색구매액"] / 100_000_000

                fig.add_trace(
                    go.Bar(
                        x=trend_df["연도"].astype(str), y=trend_df["녹색구매액_억원"],
                        name="연도별 녹색구매액", showlegend=False, opacity=0.65, marker=dict(color="#ff8db3"),
                        text=[f"{v:,.2f}억" for v in trend_df["녹색구매액_억원"]], textposition="outside", textfont=dict(size=13, color="black"),
                        hovertemplate="<b>%{x}년</b><br>녹색구매액: %{y:,.2f}억 원<extra></extra>",
                    ), row=1, col=2, secondary_y=False,
                )
                fig.add_trace(
                    go.Scatter(
                        x=trend_df["연도"].astype(str), y=trend_df["구매비율"],
                        mode="lines+markers+text", name="연도별 구매비율", showlegend=False,
                        line=dict(color="#9bd86f", width=2), marker=dict(size=7, color="#9bd86f"),
                        text=[f"{v:.1f}%" for v in trend_df["구매비율"]], textposition="top center",
                        hovertemplate="<b>%{x}년</b><br>구매비율: %{y:.2f}%<extra></extra>",
                    ), row=1, col=2, secondary_y=True,
                )

            title_suffix = f" [{SELECTED_TYPE}] 그룹 진단" if SELECTED_TYPE else " (전체 기관 기준)"
            fig.update_layout(
                title=dict(text=f"<b>{CURRENT_YEAR}년 공공기관 녹색구매 위치찾기{title_suffix}</b>", x=0.5, xanchor="center"),
                template="plotly_white", width=1500, height=1600, margin=dict(l=60, r=40, t=80, b=60),
            )

            tick_vals = [10_000_000, 50_000_000, 100_000_000, 500_000_000, 1_000_000_000, 10_000_000_000, 100_000_000_000]
            tick_text = ["1천만", "5천만", "1억", "5억", "10억", "100억", "1000억"]

            fig.update_xaxes(title_text="최근 3년 녹색제품 총구매액 평균", type="log", range=[log_x_min_axis, log_x_max_axis], tickmode="array", tickvals=tick_vals, ticktext=tick_text, row=1, col=1)
            fig.update_yaxes(title_text="최근 3년 녹색제품 구매비율 평균(%)", range=y_range, row=1, col=1)
            fig.update_xaxes(title_text="연도", row=1, col=2)
            fig.update_yaxes(title_text="녹색제품 구매액(억 원)", row=1, col=2, secondary_y=False)

            if not trend_df.empty:
                ratio_max, ratio_min = trend_df["구매비율"].max(), trend_df["구매비율"].min()
                fig.update_yaxes(title_text="녹색제품 구매비율(%)", range=[max(0, ratio_min - 8), min(110, ratio_max + 15)], row=1, col=2, secondary_y=True)
            else:
                fig.update_yaxes(title_text="녹색제품 구매비율(%)", row=1, col=2, secondary_y=True)

            x_axis_min = 10 ** log_x_min_axis
            x_axis_max = 10 ** log_x_max_axis
            horizontal_line_x0 = x_axis_min
            horizontal_line_x1 = x_axis_max / 2.0
            vertical_line_y0 = y_range[0]
            vertical_line_y1 = y_range[1]

            ratio_label = f"최근 3년 구매비율 평균 중앙값: {ai_ratio_line:.1f}%"
            amount_label = f"최근 3년 녹색구매액 평균 중앙값: {format_amount_won(ai_amount_line)}"

            fig.add_shape(type="line", xref="x1", yref="y1", x0=horizontal_line_x0, x1=horizontal_line_x1, y0=ai_ratio_line, y1=ai_ratio_line, line=dict(color="black", width=1.5, dash="dash"))
            fig.add_shape(type="line", xref="x1", yref="y1", x0=ai_amount_line, x1=ai_amount_line, y0=vertical_line_y0, y1=vertical_line_y1, line=dict(color="black", width=1.5, dash="dash"))

            fig.add_annotation(xref="x1", yref="y1", x=np.log10(horizontal_line_x1), y=ai_ratio_line, text=ratio_label, showarrow=False, font=dict(size=11, color="black"), xanchor="right", yanchor="bottom", yshift=5)
            fig.add_annotation(xref="x1", yref="y1", x=np.log10(ai_amount_line), y=vertical_line_y1, text=amount_label, textangle=-90, showarrow=False, font=dict(size=11, color="black"), xanchor="center", xshift=15, yanchor="top")
            #y=vertical_line_y1 - (y_range[1] - y_range[0]) * 0.60
            xb_log = np.log10(ai_amount_line)
            x_boundary_pos = (xb_log - log_x_min_axis) / (log_x_max_axis - log_x_min_axis)
            y_boundary_pos = (ai_ratio_line - y_range[0]) / (y_range[1] - y_range[0])

            left_x, right_x = x_boundary_pos / 2, (1 + x_boundary_pos) / 2
            bottom_y, top_y = y_boundary_pos / 2, (1 + y_boundary_pos) / 2

            fig.add_annotation(xref="x domain", yref="y domain", x=right_x, y=top_y, text="<b>우수형</b><br><span style='font-size:16px; color:black'>(구매율 높음, 구매규모 큼)</span>", showarrow=False, align="center", font=dict(size=16, color="rgba(46, 204, 113, 0.85)"))
            fig.add_annotation(xref="x domain", yref="y domain", x=left_x, y=top_y, text="<b>성과유지형</b><br><span style='font-size:16px; color:black'>(구매율 높음, 구매규모 작음)</span>", showarrow=False, align="center", font=dict(size=16, color="rgba(212, 172, 13, 0.90)"))
            fig.add_annotation(xref="x domain", yref="y domain", x=left_x, y=bottom_y, text="<b>개선필요형</b><br><span style='font-size:16px; color:black'>(구매율 낮음, 구매규모 큼)</span>", showarrow=False, align="center", font=dict(size=16, color="rgba(231, 76, 60, 0.85)"))
            fig.add_annotation(xref="x domain", yref="y domain", x=right_x, y=bottom_y, text="<b>잠재성장형</b><br><span style='font-size:16px; color:black'>(구매율 낮음, 구매규모 작)</span>", showarrow=False, align="center", font=dict(size=16, color="rgba(52, 152, 219, 0.85)"))

            fig.add_annotation(xref="x domain", yref="paper", x=0.5, y=0.44, text=("<span style='color:#A8D8EA'>●</span> 구매패턴 A&nbsp;&nbsp;&nbsp;<span style='color:#AAE3A1'>●</span> 구매패턴 B&nbsp;&nbsp;&nbsp;<span style='color:#F7C8E0'>●</span> 구매패턴 C&nbsp;&nbsp;&nbsp;<span style='color:#000000'>★</span> 검색 기관"), showarrow=False, xanchor="center", yanchor="middle", font=dict(size=12), bgcolor="white", bordercolor="#d9d9d9", borderwidth=1, borderpad=6)

            if not target_df.empty:
                fig.add_annotation(xref="x2 domain", yref="paper", x=0.5, y=1.01, text=f"<b>📊 {target_df.iloc[0]['기관명']} 최근 3년 실적 추이 📊</b>", showarrow=False, xanchor="center", yanchor="bottom", font=dict(size=16, color="#1f3a5f"))
                fig.add_annotation(xref="x2 domain", yref="paper", x=0.5, y=0.44, text=("<span style='color:#ff8db3'>■</span> 연도별 녹색구매액&nbsp;&nbsp;&nbsp;<span style='color:#9bd86f'>━●</span> 연도별 구매비율"), showarrow=False, xanchor="center", yanchor="middle", font=dict(size=12), bgcolor="white", bordercolor="#d9d9d9", borderwidth=1, borderpad=6)

            return fig

        # ===================================================
        # 10. 생성된 대시보드 화면에 출력
        # ===================================================
        with st.spinner('대시보드를 생성하는 중입니다...'):
            dashboard_fig = make_current_dashboard()
            # Streamlit 전용 차트 출력 함수 사용
            st.plotly_chart(dashboard_fig, use_container_width=True)
