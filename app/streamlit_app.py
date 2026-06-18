from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st
import plotly.express as px

# Allow imports when running from project root.
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from crop_profiles import load_crop_profiles
from utils import class_label, score_to_rgb

TRANSLATIONS = {
    "ja": {
        "title": "日本作物適性評価PoC (千葉県匝瑳市飯塚)",
        "caption": "画面 1: 衛星、土壌、気候、地形で評価した、匝瑳市飯塚におけるレモン、リンゴ、オリーブ、大豆の適性評価ヒートマップ。",
        "data_header": "データ",
        "upload_label": "予測CSVをアップロード",
        "loaded_upload": "アップロードされたCSVを読み込みました",
        "loaded_default": "デフォルトのデータを読み込みました",
        "no_predictions": "予測データが見つかりません。まず src/make_demo_data.py と src/train.py を実行してください。",
        "map_settings": "マップ設定",
        "crop_multiselect": "作物",
        "crop_help": "1つの作物を選んでその適性を表示するか、複数の作物を選んで各地点での最適な作物を表示します。",
        "use_model": "学習済みモデルのスコアを使用する",
        "min_score": "最低スコア",
        "point_radius": "点の半径",
        "color_by_label": "点の配色基準",
        "color_by_help": "「適性スコア」は緑/赤のヒートマップを使用します。「作物種別」は作物ごとに異なる色を使用します。",
        "color_by_options": {
            "Crop Type": "作物種別",
            "Suitability Score": "適性スコア",
        },
        "map_style_label": "地図スタイル",
        "map_style_help": "実際の衛星写真か道路地図を選択します。",
        "map_style_options": {
            "Satellite Map": "衛星写真",
            "Road Map (Light)": "道路地図（ライト）",
            "Road Map (Dark)": "道路地図（ダーク）",
        },
        "select_at_least_one": "マップ設定で少なくとも1つの作物を選んでください。",
        "columns_not_found": "カラムが見つかりません",
        "check_pipeline": "予測パイプラインの出力を確認してください。",
        "selected_crop_map": "選択された作物の適性マップ",
        "crop_label": "作物",
        "product_label": "主な製品",
        "sun_label": "必要な日照",
        "shade_label": "耐陰性",
        "map_mode_label": "マップモード",
        "best_crop_per_loc": "各地点で最も適した作物",
        "selected_crops_label": "選択された作物",
        "best_crop_info": "各地点は、選択された作物のうち最も高い適性スコアに基づいて配色されます。ツールチップには選択された各作物の個別のスコアが表示されます。",
        "map_color_legend": "マップカラー凡例",
        "map_crop_legend": "作物別凡例",
        "pale_saturated_help": "(淡い = 低適性, 濃い = 高適性)",
        "crop_legend_caption": "色は作物の種類を表します。点の色の彩度は適性スコアを示します（鮮やか = 高適性）。",
        "score_legend_caption": "色は適性スコアを表します。複数の作物が選択されている場合、その地点で選択された作物のうち最高のスコアに基づいて配色されます。",
        "metric_rows": "地点数",
        "metric_mean": "表示スコア平均",
        "metric_priority": "最優先ゾーン (80点以上)",
        "metric_high": "適性高以上のゾーン (65点以上)",
        "score_dist": "スコア分布",
        "score_dist_title": "表示中の適性スコア分布",
        "feature_importance": "特徴量の重要度",
        "top_feature_importance": "上位の特徴量の重要度",
        "importance_label": "重要度",
        "feature_label": "特徴量",
        "top_candidate_locs": "最有力候補地",
        "best_crop_counts": "最適な作物の選定数",
        "locations_label": "地点数",
        "compare_crops": "作物の比較",
        "footer_caption": "本ダッシュボードはスクリーニング用のツールです。最終的な植栽の決定には、農家による検証や土壌・臨床検査が必要です。",
        "no_valid_rows": "有効なマップデータが見つかりません。",
        "could_not_load_rivers": "河川データを読み込めませんでした",
        "tooltip_map_mode": "マップモード",
        "tooltip_shown_crop": "表示作物",
        "tooltip_shown_score": "表示スコア",
        "crops": {
            "lemon": "レモン",
            "apple": "リンゴ",
            "olive": "オリーブ",
            "soybean": "大豆",
        },
        "crop_details": {
            "lemon": {
                "product": "生鮮レモン / 果汁",
                "sun_requirement": "日当たり良好",
                "shade_tolerance": "低い",
                "agrivoltaics_note": "レモンは高照度を好む作物です。ソーラーシェアリングを行うには、高いパネルクリアランスと広い配置間隔（遮光率最大30〜40％など）が必要です。",
            },
            "apple": {
                "product": "生鮮リンゴ",
                "sun_requirement": "日当たり良好",
                "shade_tolerance": "低い",
                "agrivoltaics_note": "リンゴ園は果実の色づきに強い光を必要とします。ソーラーシェアリングには、薄型/両面受光型の垂直パネル、または広い列間隔を使用する必要があります。",
            },
            "olive": {
                "product": "オリーブオイル",
                "sun_requirement": "日当たり良好",
                "shade_tolerance": "低〜中程度",
                "agrivoltaics_note": "パネル間隔を広く空けて60％以上の光透過率を確保すれば、良好な適合性が得られます。",
            },
            "soybean": {
                "product": "大豆 / 豆腐 / 枝豆",
                "sun_requirement": "日当たり良好",
                "shade_tolerance": "中程度",
                "agrivoltaics_note": "大豆は日本でソーラーパネルの下での栽培に成功しています。部分的な日陰（最大30〜40％の減光）を許容し、収量への影響は軽微です。",
            }
        },
        "suitability_levels": {
            "Very suitable": "極めて適している",
            "High suitability": "適性が高い",
            "Moderate suitability": "適度な適性",
            "Low suitability": "適性が低い"
        }
    },
    "en": {
        "title": "Japan Crop Suitability PoC (Sosa Iizuka)",
        "caption": "Screen 1: Lemon, Apple, Olive, and Soybean suitability heatmaps in Iizuka (Sosa, Chiba) from satellite, soil, climate, and terrain features.",
        "data_header": "Data",
        "upload_label": "Upload prediction CSV",
        "loaded_upload": "Loaded uploaded CSV",
        "loaded_default": "Loaded default",
        "no_predictions": "No predictions found. Run src/make_demo_data.py and src/train.py first.",
        "map_settings": "Map settings",
        "crop_multiselect": "Crop",
        "crop_help": "Select one crop to show that crop's suitability, or multiple crops to show the best selected crop at each location.",
        "use_model": "Use trained model score",
        "min_score": "Minimum score",
        "point_radius": "Point radius",
        "color_by_label": "Color points by",
        "color_by_help": "Suitability Score uses a green/red heat scale. Crop Type uses a distinct color per crop.",
        "color_by_options": {
            "Crop Type": "Crop Type",
            "Suitability Score": "Suitability Score",
        },
        "map_style_label": "Map style",
        "map_style_help": "Choose between real satellite imagery or road maps.",
        "map_style_options": {
            "Satellite Map": "Satellite Map",
            "Road Map (Light)": "Road Map (Light)",
            "Road Map (Dark)": "Road Map (Dark)",
        },
        "select_at_least_one": "Select at least one crop in Map settings.",
        "columns_not_found": "Column(s) not found",
        "check_pipeline": "Check prediction pipeline output.",
        "selected_crop_map": "Selected crop suitability map",
        "crop_label": "Crop",
        "product_label": "Product",
        "sun_label": "Sun",
        "shade_label": "Shade tolerance",
        "map_mode_label": "Map mode",
        "best_crop_per_loc": "Best selected crop per location",
        "selected_crops_label": "Selected crops",
        "best_crop_info": "Each point is colored by the highest suitability score among the selected crops. The tooltip shows each selected crop's individual score.",
        "map_color_legend": "Map color legend",
        "map_crop_legend": "Map crop legend",
        "pale_saturated_help": "(pale = low suitability, saturated = high)",
        "crop_legend_caption": "Colors represent the crop type. Point color saturation indicates the suitability score (vibrant = high suitability).",
        "score_legend_caption": "Colors represent suitability score. When multiple crops are selected, the color uses the best score among the selected crops at that location.",
        "metric_rows": "Rows",
        "metric_mean": "Mean shown score",
        "metric_priority": "Priority zones",
        "metric_high": "High+ zones",
        "score_dist": "Score distribution",
        "score_dist_title": "Shown suitability score distribution",
        "feature_importance": "Feature importance",
        "top_feature_importance": "Top feature importance",
        "importance_label": "Importance",
        "feature_label": "Feature",
        "top_candidate_locs": "Top candidate locations",
        "best_crop_counts": "Best selected crop counts",
        "locations_label": "Locations",
        "compare_crops": "Compare crops",
        "footer_caption": "This dashboard is a screening tool. Final planting decisions require farmer validation and soil/lab testing.",
        "no_valid_rows": "No valid map rows found.",
        "could_not_load_rivers": "Could not load rivers data",
        "tooltip_map_mode": "Map mode",
        "tooltip_shown_crop": "Shown crop",
        "tooltip_shown_score": "Shown score",
        "crops": {
            "lemon": "Lemon",
            "apple": "Apple",
            "olive": "Olive",
            "soybean": "Soybean",
        },
        "crop_details": {
            "lemon": {
                "product": "Fresh lemons / juice",
                "sun_requirement": "Full sun",
                "shade_tolerance": "Low",
                "agrivoltaics_note": "Lemon is a high-light crop. Solar sharing requires high panel clearance and wide spacing (e.g., 30-40% shading max).",
            },
            "apple": {
                "product": "Fresh apples",
                "sun_requirement": "Full sun",
                "shade_tolerance": "Low",
                "agrivoltaics_note": "Apple orchards require high light for fruit coloring. Solar sharing should use thin/bifacial vertical panels or high row spacing.",
            },
            "olive": {
                "product": "Olive oil",
                "sun_requirement": "Full sun",
                "shade_tolerance": "Low to moderate",
                "agrivoltaics_note": "Good compatibility if panels are spaced widely to allow 60%+ light transmission.",
            },
            "soybean": {
                "product": "Soybeans / Tofu / Edamame",
                "sun_requirement": "Full sun",
                "shade_tolerance": "Moderate",
                "agrivoltaics_note": "Soybean is successfully grown under solar panels in Japan. It tolerates partial shade (up to 30-40% reduction in light) with minor yield impacts.",
            }
        },
        "suitability_levels": {
            "Very suitable": "Very suitable",
            "High suitability": "High suitability",
            "Moderate suitability": "Moderate suitability",
            "Low suitability": "Low suitability"
        }
    }
}

st.set_page_config(page_title="日本作物適性評価 PoC (Japan Crop Suitability)", layout="wide")

DEFAULT_DATA = ROOT / "data" / "exports" / "jp_crop_suitability_predictions_30.csv"
MODEL_DIR = ROOT / "models"

CROP_GRADIENTS = {
    "lemon": {
        "start": [255, 253, 210],  # Very pale yellow
        "end": [245, 190, 0]      # Bright golden yellow
    },
    "apple": {
        "start": [255, 230, 230],  # Very pale red/pink
        "end": [230, 30, 30]       # Deep crimson red
    },
    "olive": {
        "start": [230, 248, 230],  # Very pale green
        "end": [34, 139, 34]       # Rich forest green
    },
    "soybean": {
        "start": [255, 240, 220],  # Very pale orange/amber
        "end": [200, 100, 10]      # Rich amber/orange-brown
    }
}


@st.cache_data
def load_data(path_or_buffer) -> pd.DataFrame:
    df = pd.read_csv(path_or_buffer)
    df.columns = [str(c).strip() for c in df.columns]
    return df


@st.cache_data
def load_profiles() -> dict:
    return load_crop_profiles(ROOT / "config" / "crop_profiles.yaml")


def crop_display_name(crop_key: str, profiles: dict, lang: str = "ja") -> str:
    return TRANSLATIONS[lang]["crops"].get(crop_key, profiles.get(crop_key, {}).get("display_name", crop_key))


def get_crop_score_column(crop_key: str, use_model: bool) -> str:
    model_col = f"{crop_key}_model_score"
    rule_col = f"{crop_key}_rule_score"
    return model_col if use_model else rule_col


def add_color_column(
    df: pd.DataFrame,
    score_col: str,
    color_col: str = "color",
) -> pd.DataFrame:
    out = df.copy()
    colors = out[score_col].apply(
        lambda x: score_to_rgb(float(x) if pd.notnull(x) else np.nan)
    )
    out[color_col] = colors
    return out


def score_legend_html_1onldi(lang: str = "ja") -> str:
    """Legend uses the same score_to_rgb() helper used by the map."""
    bins = [
        ("Very suitable", "80–100", 90),
        ("High suitability", "65–79", 72),
        ("Moderate suitability", "40–64", 52),
        ("Low suitability", "0–39", 25),
    ]

    rows = []
    for label, value_range, representative_score in bins:
        r, g, b = score_to_rgb(representative_score)
        display_label = TRANSLATIONS[lang]["suitability_levels"].get(label, label)
        rows.append(
            f"""
            <div style="display:flex;align-items:center;gap:0.5rem;margin:0.20rem 0;">
                <span style="width:16px;height:16px;border-radius:3px;background:rgb({r},{g},{b});
                             border:1px solid rgba(0,0,0,0.25);display:inline-block;"></span>
                <span><b>{display_label}</b> <span style="color:#666;">({value_range})</span></span>
            </div>
            """
        )

    title = TRANSLATIONS[lang]["map_color_legend"]
    caption = TRANSLATIONS[lang]["score_legend_caption"]
    return (
        f"""
        <div style="padding:0.75rem 0.9rem;border:1px solid rgba(49,51,63,0.2);
                    border-radius:0.5rem;margin:0.25rem 0 1rem 0;background:rgba(250,250,250,0.65);">
            <div style="font-weight:700;margin-bottom:0.35rem;">{title}</div>
        """
        + "\n".join(rows)
        + f"""
            <div style="font-size:0.85rem;color:#666;margin-top:0.45rem;">
                {caption}
            </div>
        </div>
        """
    )


def render_score_legend_1on() -> None:
    st.markdown("**Map color legend**")

    legend_items = [
        ("Very suitable", "80–100", 90),
        ("High suitability", "65–79", 72),
        ("Moderate suitability", "40–64", 52),
        ("Low suitability", "0–39", 25),
    ]

    for label, value_range, representative_score in legend_items:
        r, g, b = score_to_rgb(representative_score)

        cols = st.columns([0.08, 0.92])
        with cols[0]:
            st.color_picker(
                label=f"{label} color",
                value=f"#{r:02x}{g:02x}{b:02x}",
                label_visibility="collapsed",
                disabled=True,
                key=f"legend_{label}",
            )

        with cols[1]:
            st.markdown(f"**{label}** ({value_range})")

    st.caption(
        "Colors represent suitability score. When multiple crops are selected, "
        "the color uses the best score among the selected crops at that location."
    )

def render_score_legend(lang: str = "ja") -> None:
    st.markdown(f"**{TRANSLATIONS[lang]['map_color_legend']}**")

    legend_items = [
        ("Very suitable", "80–100", 90),
        ("High suitability", "65–79", 72),
        ("Moderate suitability", "40–64", 52),
        ("Low suitability", "0–39", 25),
    ]

    for label, value_range, representative_score in legend_items:
        r, g, b = score_to_rgb(representative_score)
        display_label = TRANSLATIONS[lang]["suitability_levels"].get(label, label)

        st.markdown(
            f"""
            <div style="display:flex;align-items:center;gap:0.5rem;margin:0.25rem 0;">
                <span style="
                    width:16px;
                    height:16px;
                    border-radius:3px;
                    background-color:rgb({r},{g},{b});
                    border:1px solid rgba(0,0,0,0.3);
                    display:inline-block;
                "></span>
                <span><b>{display_label}</b> <span style="color:#666;">({value_range})</span></span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.caption(TRANSLATIONS[lang]["score_legend_caption"])


def render_crop_legend(selected_crop_keys: list[str], profiles: dict, lang: str = "ja") -> None:
    st.markdown(f"**{TRANSLATIONS[lang]['map_crop_legend']}**")

    for crop_key in selected_crop_keys:
        gradient = CROP_GRADIENTS.get(crop_key, {"start": [240, 240, 240], "end": [120, 120, 120]})
        start = gradient["start"]
        end = gradient["end"]
        name = crop_display_name(crop_key, profiles, lang)
        suffix = TRANSLATIONS[lang]["pale_saturated_help"]

        st.markdown(
            f"""
            <div style="display:flex;align-items:center;gap:0.75rem;margin:0.35rem 0;">
                <span style="
                    width:50px;
                    height:16px;
                    border-radius:3px;
                    background: linear-gradient(90deg, rgb({start[0]},{start[1]},{start[2]}) 0%, rgb({end[0]},{end[1]},{end[2]}) 100%);
                    border:1px solid rgba(0,0,0,0.3);
                    display:inline-block;
                "></span>
                <span><b>{name}</b> <span style="color:#666;font-size:0.85rem;">{suffix}</span></span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.caption(TRANSLATIONS[lang]["crop_legend_caption"])


def prepare_selected_crop_view(
    df: pd.DataFrame,
    selected_crop_keys: list[str],
    profiles: dict,
    score_cols: dict[str, str],
    color_by: str = "Suitability Score",
    lang: str = "ja",
) -> tuple[pd.DataFrame, str]:
    """
    Adds the map score/crop columns.

    - One crop selected: map score/color = that crop's score.
    - Multiple crops selected: map score/color = highest score among selected crops,
      and best_selected_crop identifies which crop won at each point.
    """
    out = df.copy()

    # Rounded score columns make PyDeck tooltips easier to read.
    for crop_key in selected_crop_keys:
        score_col = score_cols[crop_key]
        out[f"{crop_key}_display_score"] = out[score_col].round(1)

    if len(selected_crop_keys) == 1:
        crop_key = selected_crop_keys[0]
        score_col = score_cols[crop_key]

        out["map_score"] = out[score_col]
        out["map_score_display"] = out["map_score"].round(1)
        out["map_crop"] = crop_display_name(crop_key, profiles, lang)
        out["map_mode"] = f"{crop_display_name(crop_key, profiles, lang)}のスコア" if lang == "ja" else f"{crop_display_name(crop_key, profiles, lang)} score"
        out["map_crop_key"] = crop_key

    else:
        selected_score_cols = [score_cols[k] for k in selected_crop_keys]

        out["map_score"] = out[selected_score_cols].max(axis=1)
        out["map_score_display"] = out["map_score"].round(1)

        # idxmax returns the score column name; convert it back to the crop key/display name.
        col_to_crop = {score_cols[k]: k for k in selected_crop_keys}
        best_score_col = out[selected_score_cols].idxmax(axis=1)

        out["best_selected_crop_key"] = best_score_col.map(col_to_crop)
        out["map_crop"] = out["best_selected_crop_key"].map(
            lambda k: crop_display_name(k, profiles, lang)
        )
        out["map_mode"] = TRANSLATIONS[lang]["best_crop_per_loc"]
        out["map_crop_key"] = out["best_selected_crop_key"]

    if color_by == "Crop Type":
        def row_to_crop_color(row):
            crop_key = row["map_crop_key"]
            score = row["map_score"]
            if pd.isnull(score):
                return [160, 160, 160, 200]

            # Linear interpolation for the color gradient of this crop
            gradient = CROP_GRADIENTS.get(crop_key, {
                "start": [240, 240, 240],
                "end": [120, 120, 120]
            })
            start = gradient["start"]
            end = gradient["end"]

            t = np.clip(float(score) / 100.0, 0.0, 1.0)
            r = int(start[0] + (end[0] - start[0]) * t)
            g = int(start[1] + (end[1] - start[1]) * t)
            b = int(start[2] + (end[2] - start[2]) * t)

            return [r, g, b, 230] # high constant opacity so it shows clearly on satellite background!

        out["color"] = out.apply(row_to_crop_color, axis=1)
    else:
        out = add_color_column(out, "map_score")

    return out, "map_score"

def get_rivers_geojson(lang: str = "ja") -> dict | None:
    """Load rivers GeoJSON from Natural Earth data via URL."""
    try:
        import urllib.request
        # Alternative: Try to use a simpler approach with a direct GeoJSON URL
        # Natural Earth provides rivers as GeoJSON
        rivers_url = "https://naciscdn.org/naturalearth/10m/physical/ne_10m_rivers_lake_centerlines.geojson"
        
        with urllib.request.urlopen(rivers_url, timeout=10) as response:
            import json as json_module
            rivers_data = json_module.loads(response.read())
            return rivers_data
    except Exception as e:
        st.warning(f"{TRANSLATIONS[lang]['could_not_load_rivers']}: {e}")
        return None



def render_map(
    df: pd.DataFrame,
    selected_crop_keys: list[str],
    profiles: dict,
    score_cols: dict[str, str],
    radius: int = 10,
    map_background: str = "Satellite Map",
    lang: str = "ja",
) -> None:
    required_cols = ["longitude", "latitude", "map_score", "map_crop", "color"]
    map_df = df.dropna(subset=required_cols).copy()

    if len(map_df) == 0:
        st.warning(TRANSLATIONS[lang]["no_valid_rows"])
        return

    mid_lat = float(map_df["latitude"].mean())
    mid_lon = float(map_df["longitude"].mean())

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=map_df,
        get_position="[longitude, latitude]",
        get_fill_color="color",
        get_radius=radius,
        pickable=True,
        opacity=0.85,
    )

    layers = []
    if map_background == "Satellite Map":
        satellite_layer = pdk.Layer(
            "TileLayer",
            data="https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            pickable=False,
        )
        layers.append(satellite_layer)

    layers.append(layer)

    if map_background != "Satellite Map":
        rivers_data = get_rivers_geojson(lang)
        if rivers_data:
            rivers_layer = pdk.Layer(
                "GeoJsonLayer",
                rivers_data,
                stroked=False,
                filled=False,
                extruded=False,
                wireframe=True,
                get_line_color=[0, 100, 200],
                get_line_width=2,
                pickable=False,
            )
            layers.append(rivers_layer)

    crop_score_lines = ""
    score_suffix = "のスコア" if lang == "ja" else " score"
    for crop_key in selected_crop_keys:
        display = crop_display_name(crop_key, profiles, lang)
        crop_score_lines += f"<br/><b>{display}{score_suffix}:</b> {{{crop_key}_display_score}}"

    tooltip = {
        "html": (
            f"<b>{TRANSLATIONS[lang]['tooltip_map_mode']}:</b> {{map_mode}}"
            f"<br/><b>{TRANSLATIONS[lang]['tooltip_shown_crop']}:</b> {{map_crop}}"
            f"<br/><b>{TRANSLATIONS[lang]['tooltip_shown_score']}:</b> {{map_score_display}}"
            + crop_score_lines
            + "<br/><b>NDVI:</b> {NDVI}"
            "<br/><b>NDMI:</b> {NDMI}"
        ),
        "style": {"backgroundColor": "steelblue", "color": "white"},
    }

    style_map = {
        "Satellite Map": None,
        "Road Map (Light)": "light",
        "Road Map (Dark)": "dark",
    }
    selected_style = style_map.get(map_background, None)
    map_provider = None if map_background == "Satellite Map" else "carto"

    deck = pdk.Deck(
        map_style=selected_style,
        map_provider=map_provider,
        initial_view_state=pdk.ViewState(
            latitude=mid_lat,
            longitude=mid_lon,
            zoom=14,
            pitch=0,
        ),
        layers=layers,
        tooltip=tooltip,
    )

    st.pydeck_chart(deck, use_container_width=True)


def show_feature_importance(crop_key: str, lang: str = "ja") -> None:
    path = MODEL_DIR / f"{crop_key}_feature_importance.csv"

    if not path.exists():
        st.info("Feature importance file not found yet. Run training first." if lang == "en" else "特徴量の重要度ファイルがまだ見つかりません。先に学習を実行してください。")
        return

    imp = pd.read_csv(path).head(12)

    fig = px.bar(
        imp.sort_values("importance"),
        x="importance",
        y="feature",
        orientation="h",
        title=TRANSLATIONS[lang]["top_feature_importance"],
        labels={
            "importance": TRANSLATIONS[lang]["importance_label"],
            "feature": TRANSLATIONS[lang]["feature_label"],
        }
    )

    st.plotly_chart(fig, use_container_width=True)


# Language Selection at the top of the sidebar
if "lang" not in st.session_state:
    st.session_state.lang = "ja"

with st.sidebar:
    lang_choice = st.selectbox("Language / 言語", ["日本語", "English"], index=0)
    lang = "ja" if lang_choice == "日本語" else "en"
    st.session_state.lang = lang

st.title(TRANSLATIONS[lang]["title"])
st.caption(TRANSLATIONS[lang]["caption"])

profiles = load_profiles()
crop_keys = list(profiles.keys())

with st.sidebar:
    st.header(TRANSLATIONS[lang]["data_header"])

    uploaded = st.file_uploader(TRANSLATIONS[lang]["upload_label"], type=["csv"])

    if uploaded is not None:
        df = load_data(uploaded)
        st.success(TRANSLATIONS[lang]["loaded_upload"])

    elif DEFAULT_DATA.exists():
        df = load_data(DEFAULT_DATA)
        st.success(f"{TRANSLATIONS[lang]['loaded_default']}: {DEFAULT_DATA.relative_to(ROOT)}")

    else:
        st.error(TRANSLATIONS[lang]["no_predictions"])
        st.stop()

    st.header(TRANSLATIONS[lang]["map_settings"])

    selected_crop_keys = st.multiselect(
        TRANSLATIONS[lang]["crop_multiselect"],
        crop_keys,
        default=[crop_keys[0],crop_keys[1],crop_keys[2],crop_keys[3]] if crop_keys else [],
        format_func=lambda k: crop_display_name(k, profiles, lang),
        help=TRANSLATIONS[lang]["crop_help"],
    )

    use_model = st.toggle(TRANSLATIONS[lang]["use_model"], value=True)
    min_score = st.slider(TRANSLATIONS[lang]["min_score"], 0, 100, 0)
    radius = st.slider(TRANSLATIONS[lang]["point_radius"], 3, 100, 10, step=1)

    color_by = st.selectbox(
        TRANSLATIONS[lang]["color_by_label"],
        ["Crop Type", "Suitability Score"],
        format_func=lambda x: TRANSLATIONS[lang]["color_by_options"].get(x, x),
        help=TRANSLATIONS[lang]["color_by_help"],
    )

    map_background = st.selectbox(
        TRANSLATIONS[lang]["map_style_label"],
        ["Satellite Map", "Road Map (Light)", "Road Map (Dark)"],
        format_func=lambda x: TRANSLATIONS[lang]["map_style_options"].get(x, x),
        help=TRANSLATIONS[lang]["map_style_help"],
    )

if not selected_crop_keys:
    st.warning(TRANSLATIONS[lang]["select_at_least_one"])
    st.stop()

score_cols = {k: get_crop_score_column(k, use_model) for k in selected_crop_keys}

missing_score_cols = [c for c in score_cols.values() if c not in df.columns]

if missing_score_cols:
    st.error(
        f"{TRANSLATIONS[lang]['columns_not_found']}: {', '.join(missing_score_cols)}. "
        f"{TRANSLATIONS[lang]['check_pipeline']}"
    )
    st.stop()

prepared, map_score_col = prepare_selected_crop_view(
    df=df,
    selected_crop_keys=selected_crop_keys,
    profiles=profiles,
    score_cols=score_cols,
    color_by=color_by,
    lang=lang,
)

filtered = prepared[prepared[map_score_col] >= min_score].copy()

st.subheader(TRANSLATIONS[lang]["selected_crop_map"])

if len(selected_crop_keys) == 1:
    crop_key = selected_crop_keys[0]
    crop_profile = profiles[crop_key]

    if lang == "ja":
        details = TRANSLATIONS["ja"]["crop_details"].get(crop_key, {})
        st.write(
            f"**{TRANSLATIONS[lang]['crop_label']}:** {crop_display_name(crop_key, profiles, lang)}  |  "
            f"**{TRANSLATIONS[lang]['product_label']}:** {details.get('product', '')}  |  "
            f"**{TRANSLATIONS[lang]['sun_label']}:** {details.get('sun_requirement', '')}  |  "
            f"**{TRANSLATIONS[lang]['shade_label']}:** {details.get('shade_tolerance', '')}"
        )
        st.info(details.get("agrivoltaics_note", ""))
    else:
        st.write(
            f"**{TRANSLATIONS[lang]['crop_label']}:** {crop_display_name(crop_key, profiles, lang)}  |  "
            f"**{TRANSLATIONS[lang]['product_label']}:** {crop_profile.get('product', '')}  |  "
            f"**{TRANSLATIONS[lang]['sun_label']}:** {crop_profile.get('sun_requirement', '')}  |  "
            f"**{TRANSLATIONS[lang]['shade_label']}:** {crop_profile.get('shade_tolerance', '')}"
        )
        st.info(crop_profile.get("agrivoltaics_note", ""))

else:
    selected_names = [crop_display_name(k, profiles, lang) for k in selected_crop_keys]

    st.write(
        f"**{TRANSLATIONS[lang]['map_mode_label']}:** {TRANSLATIONS[lang]['best_crop_per_loc']}  |  "
        f"**{TRANSLATIONS[lang]['selected_crops_label']}:** {', '.join(selected_names)}"
    )

    st.info(TRANSLATIONS[lang]["best_crop_info"])

if color_by == "Crop Type":
    render_crop_legend(selected_crop_keys, profiles, lang)
else:
    render_score_legend(lang)

c1, c2, c3, c4 = st.columns(4)

c1.metric(TRANSLATIONS[lang]["metric_rows"], f"{len(filtered):,}")
c2.metric(
    TRANSLATIONS[lang]["metric_mean"],
    f"{filtered[map_score_col].mean():.1f}" if len(filtered) else "n/a",
)
c3.metric(TRANSLATIONS[lang]["metric_priority"], f"{(filtered[map_score_col] >= 80).sum():,}")
c4.metric(TRANSLATIONS[lang]["metric_high"], f"{(filtered[map_score_col] >= 65).sum():,}")

render_map(
    df=filtered,
    selected_crop_keys=selected_crop_keys,
    profiles=profiles,
    score_cols=score_cols,
    radius=radius,
    map_background=map_background,
    lang=lang,
)

left, right = st.columns([1, 1])

with left:
    st.subheader(TRANSLATIONS[lang]["score_dist"])

    fig = px.histogram(
        filtered,
        x=map_score_col,
        nbins=30,
        title=TRANSLATIONS[lang]["score_dist_title"],
        labels={map_score_col: TRANSLATIONS[lang]["suitability_levels"].get("Suitability Score", "Suitability Score")}
    )

    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader(TRANSLATIONS[lang]["feature_importance"])

    if len(selected_crop_keys) == 1:
        show_feature_importance(selected_crop_keys[0], lang)

    else:
        tabs = st.tabs([crop_display_name(k, profiles, lang) for k in selected_crop_keys])

        for tab, crop_key in zip(tabs, selected_crop_keys):
            with tab:
                show_feature_importance(crop_key, lang)

st.subheader(TRANSLATIONS[lang]["top_candidate_locs"])

show_cols = ["longitude", "latitude", "map_crop", "map_score_display"]

for crop_key in selected_crop_keys:
    score_col = score_cols[crop_key]
    class_col = score_col.replace("score", "class")

    component_cols = [
        f"{crop_key}_soil_component",
        f"{crop_key}_water_component",
        f"{crop_key}_heat_component",
        f"{crop_key}_vegetation_component",
        f"{crop_key}_terrain_component",
    ]

    show_cols.append(score_col)

    if class_col in filtered.columns:
        show_cols.append(class_col)

    show_cols.extend(component_cols)

show_cols.extend(
    [
        "NDVI",
        "NDMI",
        "BSI",
        "soil_ph",
        "soil_org_carbon",
        "mean_temp_c",
        "annual_rain_mm",
        "slope_deg",
    ]
)

show_cols = [c for c in show_cols if c in filtered.columns]

top = filtered.sort_values(map_score_col, ascending=False).head(50).copy()

for crop_key in selected_crop_keys:
    score_col = score_cols[crop_key]
    class_col = score_col.replace("score", "class")

    if class_col in top.columns:
        top[f"{crop_key}_class_label"] = top[class_col].apply(class_label)
        show_cols.append(f"{crop_key}_class_label")

# Remove duplicates while preserving order.
show_cols = list(dict.fromkeys(show_cols))

st.dataframe(top[show_cols], use_container_width=True)

if len(selected_crop_keys) > 1:
    st.subheader(TRANSLATIONS[lang]["best_crop_counts"])

    counts = (
        filtered["map_crop"]
        .value_counts()
        .rename_axis(TRANSLATIONS[lang]["best_crop_per_loc"])
        .reset_index(name=TRANSLATIONS[lang]["locations_label"])
    )

    st.dataframe(counts, use_container_width=True)

st.subheader(TRANSLATIONS[lang]["compare_crops"])

compare_cols = ["longitude", "latitude"]

for k in crop_keys:
    for suffix in ["model_score", "model_class", "rule_score", "rule_class"]:
        col = f"{k}_{suffix}"

        if col in df.columns:
            compare_cols.append(col)

for col in ["best_crop", "best_crop_score", "best_crop_class"]:
    if col in df.columns:
        compare_cols.append(col)

sort_col = "best_crop_score" if "best_crop_score" in df.columns else compare_cols[-1]

st.dataframe(
    df[compare_cols].sort_values(sort_col, ascending=False).head(100),
    use_container_width=True,
)

st.caption(TRANSLATIONS[lang]["footer_caption"])