# app.py
# pip install streamlit pandas geopandas pydeck openpyxl

import re
import os
import time
import requests
from difflib import get_close_matches

import numpy as np
import pandas as pd
import geopandas as gpd
import streamlit as st
import pydeck as pdk
import plotly.graph_objects as go
import plotly.express as px

# -------------------- Paths --------------------
XLSX_MAIN = "data/Endeksler.xlsx"
XLSX_SUB = "data/Alt Endeksler.xlsx"
GEO_PATH = "data/adana_mersin.geojson"

# -------------------- CSS --------------------
def load_css():
    css_file = "assets/style.css"
    if os.path.exists(css_file):
        with open(css_file) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def ensure_file(local_path, secret_key, force_download=False):
    if os.path.exists(local_path) and not force_download:
        return local_path
    if "data_urls" not in st.secrets or secret_key not in st.secrets["data_urls"]:
        if os.path.exists(local_path):
             return local_path
        st.error(f"Dosya bulunamadı: {local_path}")
        st.stop()
    url = st.secrets["data_urls"][secret_key]
    sep = "&" if "?" in url else "?"
    url = f"{url}{sep}_={int(time.time())}"
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    with st.spinner(f"Veri indiriliyor: {local_path}..."):
        try:
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            with open(local_path, "wb") as f:
                f.write(r.content)
            st.success(f"Güncellendi: {local_path}")
        except Exception as e:
            st.error(f"Veri indirilirken hata: {e}")
            if not os.path.exists(local_path):
                st.stop()
    return local_path

def clean_cols(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df

def normalize_text(s):
    s = str(s).strip().lower()
    s = (s.replace("ç","c").replace("ğ","g").replace("ı","i").replace("ö","o").replace("ş","s").replace("ü","u"))
    s = re.sub(r"\s+", " ", s)
    return s

def minmax01(s):
    s = pd.to_numeric(s, errors="coerce")
    mn, mx = s.min(skipna=True), s.max(skipna=True)
    if pd.isna(mn) or pd.isna(mx) or mn == mx:
        return pd.Series(np.nan, index=s.index)
    return (s - mn) / (mx - mn)

def colors_from_01(values):
    v = np.clip(values.astype(float), 0, 1)
    x_pts = [0.0, 0.25, 0.50, 0.75, 1.0]
    r_pts = [44,  171, 255, 253, 215]
    g_pts = [123, 217, 255, 174, 25]
    b_pts = [182, 233, 191, 97,  28]
    r = np.interp(v, x_pts, r_pts)
    g = np.interp(v, x_pts, g_pts)
    b = np.interp(v, x_pts, b_pts)
    a = np.full_like(r, 180)
    return np.vstack([r, g, b, a]).T.astype(int)

class MetricMetadata:
    def __init__(self, col_name, label=None, group=None):
        self.col_name = col_name
        self.label = label or col_name
        self.group = group or "Genel"

def build_metric_metadata(df):
    meta = {}
    skip = {"MAHALLEKAYITNO", "İLADI", "İLÇEADI", "MAHALLEKÖYADI", "MAHALLEKOYADI"}
    cols = [str(c).strip() for c in df.columns if c not in skip
            and "sira" not in normalize_text(c)]
    for c in cols:
        group = "Diğer"
        cn = normalize_text(c)
        if cn.startswith("cke") or cn.startswith("gke"):
            group = "Ana Endeksler"
        elif any(x in cn for x in ["demografik", "ekonomik", "kronik", "bagimlilik",
                                    "yabanci", "genc", "aile"]):
            group = "Alt Endeksler"
        elif "kentsel" in cn:
            group = "Kentsel Kırılganlık"
        elif "kirsal" in cn:
            group = "Kırsal Kırılganlık"
        label = c.replace("Skor", "").replace("Düzeltilmiş", "").strip()
        meta[c] = MetricMetadata(col_name=c, label=label, group=group)
    return meta

def clean_id(val):
    try:
        return str(int(float(val)))
    except:
        return str(val).strip()

def prepare_metric_data(df, metric_meta):
    col = metric_meta.col_name
    if col not in df.columns:
        return None, None
    series = df[col].astype(str).str.replace(",", ".").str.strip()
    series = pd.to_numeric(series, errors='coerce')
    mn, mx = series.min(), series.max()
    if pd.isna(mn) or pd.isna(mx) or mn == mx:
        norm = pd.Series(np.nan, index=series.index)
    else:
        norm = (series - mn) / (mx - mn)
    return series, norm

@st.cache_data
def load_data_v2(main_path, sub_path):
    try:
        df_main = pd.read_excel(main_path)
        df_sub = pd.read_excel(sub_path)
    except Exception as e:
        st.error(f"Veri dosyaları okunamadı: {e}")
        return pd.DataFrame()
    df_main = clean_cols(df_main)
    df_sub = clean_cols(df_sub)
    if "MAHALLEKAYITNO" in df_main.columns:
        df_main["MAHALLEKAYITNO"] = df_main["MAHALLEKAYITNO"].apply(clean_id)
    if "MAHALLEKAYITNO" in df_sub.columns:
        df_sub["MAHALLEKAYITNO"] = df_sub["MAHALLEKAYITNO"].apply(clean_id)
    df_full = df_main.merge(df_sub, on="MAHALLEKAYITNO", how="outer", suffixes=("", "_sub"))
    return df_full

@st.cache_data
def load_geo(path):
    gdf = gpd.read_file(path)
    gdf = clean_cols(gdf)
    if "MAHALLEKOD" in gdf.columns:
        gdf["MAHALLEKOD"] = gdf["MAHALLEKOD"].apply(clean_id)
    try:
        if gdf.crs is None:
            gdf.set_crs(epsg=4326, inplace=True)
        elif gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs(epsg=4326)
    except Exception:
        pass
    return gdf

def check_password():
    def password_entered():
        if st.session_state["password"] == st.secrets["general"]["password"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False
    if "password_correct" not in st.session_state:
        st.text_input("Password", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        st.text_input("Password", type="password", on_change=password_entered, key="password")
        st.error("Password incorrect")
        return False
    else:
        return True

def main():
    st.set_page_config(page_title="Adana & Mersin Mahalle Kırılganlık Paneli", layout="wide")
    load_css()
    if not check_password():
        st.stop()
    st.title("Adana & Mersin Mahalle Kırılganlık Paneli")

    force_update = st.sidebar.button("🔄 Verileri Güncelle (Drive'dan İndir)")
    if force_update:
        st.cache_data.clear()
        st.rerun()

    xlsx_main_path = ensure_file(XLSX_MAIN, "main_excel", force_download=force_update)
    xlsx_sub_path = ensure_file(XLSX_SUB, "sub_excel", force_download=force_update)
    geo_path = ensure_file(GEO_PATH, "geo_file", force_download=force_update)

    df = load_data_v2(xlsx_main_path, xlsx_sub_path)
    gdf = load_geo(geo_path)

    raw_len = len(df)
    il_col = "İLADI" if "İLADI" in df.columns else None
    ilce_col = "İLÇEADI" if "İLÇEADI" in df.columns else None
    name_col = "MAHALLEKÖYADI" if "MAHALLEKÖYADI" in df.columns else ("MAHALLEKOYADI" if "MAHALLEKOYADI" in df.columns else None)
    kent_col = "KENTKIRSINIFLAMASI" if "KENTKIRSINIFLAMASI" in df.columns else None

    st.sidebar.header("Filtreler")

    # 0. İl filtresi
    if "selected_province" not in st.session_state:
        st.session_state.selected_province = "Tüm İller"
    provinces = ["Tüm İller"] + sorted(df[il_col].dropna().astype(str).unique().tolist()) if il_col else []
    sel_prov = st.sidebar.selectbox("0. İl Seçimi", provinces, key="selected_province")

    df_filtered = df.copy()
    if sel_prov != "Tüm İller" and il_col:
        df_filtered = df_filtered[df_filtered[il_col] == sel_prov]

    # 1. İlçe filtresi
    if "selected_district" not in st.session_state:
        st.session_state.selected_district = "Tüm İlçeler"
    districts = ["Tüm İlçeler"] + sorted(df_filtered[ilce_col].dropna().astype(str).unique().tolist()) if ilce_col else []
    sel_dist = st.sidebar.selectbox("1. İlçe Seçimi", districts, key="selected_district")

    if sel_dist != "Tüm İlçeler" and ilce_col:
        df_filtered = df_filtered[df_filtered[ilce_col] == sel_dist]

    # 2. Kentsellik filtresi
    if "selected_urbanity" not in st.session_state:
        st.session_state.selected_urbanity = "Tümü"
    urban_opts = ["Tümü"] + sorted(df_filtered[kent_col].dropna().astype(str).unique().tolist()) if kent_col else []
    sel_urban = st.sidebar.selectbox("2. Kentsellik Statüsü", urban_opts, key="selected_urbanity")
    if sel_urban != "Tümü" and kent_col:
         df_filtered = df_filtered[df_filtered[kent_col] == sel_urban]

    # 3. Metrik grubu
    meta_map = build_metric_metadata(df)
    available_groups = sorted(list({m.group for m in meta_map.values()}))
    priority = ["Ana Endeksler", "Alt Endeksler", "Kentsel Kırılganlık", "Kırsal Kırılganlık"]
    available_groups.sort(key=lambda x: priority.index(x) if x in priority else 99)

    if "selected_group" not in st.session_state:
        st.session_state.selected_group = available_groups[0] if available_groups else None
    sel_group = st.sidebar.radio("3. Metrik Grubu", available_groups, key="selected_group")

    # 4. Metrik seçimi
    group_metrics = [m for m in meta_map.values() if m.group == sel_group]
    metric_labels = [m.label for m in group_metrics]
    metric_label_map = {m.label: m.col_name for m in group_metrics}

    if "selected_metric_label" not in st.session_state:
         st.session_state.selected_metric_label = metric_labels[0] if metric_labels else None
    if st.session_state.selected_metric_label not in metric_labels:
        st.session_state.selected_metric_label = metric_labels[0]
    sel_label = st.sidebar.selectbox("4. Metrik", metric_labels, key="selected_metric_label")
    selected_metric_col = metric_label_map[sel_label]
    selected_meta = meta_map[selected_metric_col]

    total_rows = len(df_filtered)
    valid_data_count = df_filtered[selected_metric_col].count()
    coverage_pct = (valid_data_count / total_rows * 100) if total_rows > 0 else 0
    st.sidebar.caption(f"Veri Kapsamı: %{coverage_pct:.1f} ({valid_data_count}/{total_rows} mahalle)")

    raw_s, norm_s = prepare_metric_data(df_filtered, selected_meta)
    df_filtered["val_raw"] = raw_s
    df_filtered["score_norm"] = norm_s

    # 5. Gelişmiş filtreler
    with st.sidebar.expander("Gelişmiş Filtreler", expanded=True):
        use_norm_filter = st.checkbox("Normalize Skor (0-1) Kullan", value=True)
        if use_norm_filter:
            min_s, max_s = st.slider("Filtre Aralığı", 0.0, 1.0, (0.0, 1.0), 0.05)
            mask = (df_filtered["score_norm"] >= min_s) & (df_filtered["score_norm"] <= max_s)
        else:
            rmin = float(df_filtered["val_raw"].min())
            rmax = float(df_filtered["val_raw"].max())
            min_s, max_s = st.slider("Ham Değer Aralığı", rmin, rmax, (rmin, rmax))
            mask = (df_filtered["val_raw"] >= min_s) & (df_filtered["val_raw"] <= max_s)

    topn = st.sidebar.slider("Top N (Tablo)", 10, 200, 30, 10)
    df_final = df_filtered[mask].copy()

    # ── Merge ──
    joined = gdf.merge(df_final, left_on="MAHALLEKOD", right_on="MAHALLEKAYITNO", how="left", suffixes=("_geo", ""))
    joined["_has"] = ~joined["score_norm"].isna()
    joined_valid = joined[joined["_has"]].copy()

    if joined_valid.empty:
        st.warning("Seçili filtrelerde haritaya düşen mahalle bulunamadı.")
        st.stop()

    joined_valid["fill_color"] = colors_from_01(joined_valid["score_norm"].to_numpy()).tolist()

    bounds = joined_valid.geometry.total_bounds
    lat_ctr = (bounds[1] + bounds[3]) / 2
    lon_ctr = (bounds[0] + bounds[2]) / 2
    zoom = 9 if sel_prov == "Tüm İller" else 10

    # ── KPI ──
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Mahalle sayısı", f"{len(joined_valid):,}")
    k2.metric("Ortalama", f"{joined_valid['val_raw'].mean():.3f}")
    k3.metric("Maksimum", f"{joined_valid['val_raw'].max():.3f}")
    k4.metric("Minimum", f"{joined_valid['val_raw'].min():.3f}")

    # ── Tabs ──
    tab_map, tab_charts, tab_tables, tab_detail, tab_debug = st.tabs(
        ["Harita", "İstatistikler", "Tablolar", "Mahalle Detay Analizi", "Teknik"]
    )

    with tab_map:
        layer = pdk.Layer(
            "GeoJsonLayer", joined_valid,
            pickable=True, stroked=True, filled=True,
            get_fill_color="fill_color",
            get_line_color=[0, 0, 0, 100],
            line_width_min_pixels=1,
        )
        view_state = pdk.ViewState(latitude=lat_ctr, longitude=lon_ctr, zoom=zoom, pitch=0)
        tooltip = {
            "html": f"<b>Mahalle:</b> {{{name_col}}}<br/><b>İlçe:</b> {{{ilce_col}}}<br/><b>{sel_label}:</b> {{val_raw}}",
            "style": {"backgroundColor": "steelblue", "color": "white"}
        }
        c_map, c_leg = st.columns([3, 1])
        with c_map:
            st.pydeck_chart(pdk.Deck(
                map_style=None,
                initial_view_state=view_state,
                layers=[layer],
                tooltip=tooltip
            ), use_container_width=True, height=750)
        with c_leg:
            st.markdown(f"**{sel_label}**")
            st.markdown(
                "<div style='background:linear-gradient(to right,rgb(44,123,182),rgb(171,217,233),rgb(255,255,191),rgb(253,174,97),rgb(215,25,28));height:20px;width:100%;border-radius:5px'></div>"
                "<div style='display:flex;justify-content:space-between'><span>Düşük</span><span>Yüksek</span></div>",
                unsafe_allow_html=True
            )
            st.info("Normalize (0-1) skala")
            if il_col and il_col in joined_valid.columns:
                st.markdown("---")
                st.markdown("**İl Özeti**")
                summary = joined_valid.groupby(il_col)["val_raw"].agg(["count", "mean"]).round(3)
                summary.columns = ["n", "Ort."]
                st.dataframe(summary, use_container_width=True)

    with tab_charts:
        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(joined_valid, x="val_raw", nbins=30,
                               title=f"{sel_label} Dağılımı",
                               color_discrete_sequence=["#4a8bc2"],
                               labels={"val_raw": sel_label})
            fig.update_layout(bargap=0.1)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            if il_col and il_col in joined_valid.columns:
                fig2 = px.box(joined_valid, x=il_col, y="val_raw", color=il_col,
                              title=f"{sel_label} — İl Bazında",
                              labels={"val_raw": sel_label, il_col: "İl"},
                              color_discrete_map={"ADANA": "#2E75B6", "MERSİN": "#ED7D31"})
                st.plotly_chart(fig2, use_container_width=True)

        if ilce_col and ilce_col in joined_valid.columns:
            ilce_avg = (joined_valid.groupby(ilce_col)["val_raw"]
                        .mean().sort_values(ascending=False).head(20))
            fig3 = px.bar(ilce_avg, title=f"İlçe Ortalamaları (İlk 20) — {sel_label}",
                          labels={"value": "Ort.", "index": "İlçe"},
                          color=ilce_avg.values,
                          color_continuous_scale=["#2c7bb6", "#ffffbf", "#d7191c"])
            fig3.update_layout(showlegend=False, coloraxis_showscale=False)
            st.plotly_chart(fig3, use_container_width=True)

    with tab_tables:
        st.subheader(f"Top {topn} mahalle")
        show_cols = [c for c in [il_col, ilce_col, name_col, kent_col] if c and c in joined_valid.columns]
        display_df = joined_valid.copy()
        if selected_metric_col in display_df.columns:
            display_df = display_df.drop(columns=[selected_metric_col])
        if sel_label in display_df.columns:
             display_df = display_df.drop(columns=[sel_label])
        if f"{sel_label} (Norm)" in display_df.columns:
             display_df = display_df.drop(columns=[f"{sel_label} (Norm)"])
        display_df = display_df.rename(columns={"val_raw": sel_label, "score_norm": f"{sel_label} (Norm)"})
        final_cols = show_cols + [sel_label, f"{sel_label} (Norm)"]
        top = display_df.sort_values(sel_label, ascending=False).head(topn)[final_cols]
        st.dataframe(top, use_container_width=True)
        csv = display_df[final_cols].to_csv(index=False).encode("utf-8-sig")
        st.download_button("CSV İndir", data=csv, file_name="adana_mersin_filtered.csv", mime="text/csv")

    with tab_detail:
        st.header("Mahalle Detay Analizi")
        if name_col and name_col in joined_valid.columns:
            if ilce_col and ilce_col in joined_valid.columns:
                district_list = ["Tümü"] + sorted(joined_valid[ilce_col].astype(str).unique().tolist())
                selected_district_det = st.selectbox("İlçe Filtrele:", district_list)
                det_filtered_df = joined_valid if selected_district_det == "Tümü" else joined_valid[joined_valid[ilce_col] == selected_district_det]
            else:
                det_filtered_df = joined_valid

            det_filtered_df["_display_name"] = det_filtered_df[name_col].astype(str) + (" (" + det_filtered_df[ilce_col].astype(str) + ")" if ilce_col else "")
            selected_display_name = st.selectbox("Mahalle:", sorted(det_filtered_df["_display_name"].unique().tolist()))
            m_data = det_filtered_df[det_filtered_df["_display_name"] == selected_display_name].iloc[0]

            dk1, dk2, dk3 = st.columns(3)
            dk1.metric("Mahalle", str(m_data[name_col]))
            dk2.metric(f"{sel_label} (Ham)", f"{m_data['val_raw']:.3f}")
            dk3.metric("Normalize Skor", f"{m_data['score_norm']:.3f}")
            st.divider()

            # Mahallenin iline göre filtrele (il bazlı karşılaştırma)
            mahalle_il = str(m_data.get(il_col, "")) if il_col else ""
            df_il = df[df[il_col] == mahalle_il] if mahalle_il and il_col else df

            def render_radar(group_name, title, color):
                group_cols = [m.col_name for m in meta_map.values()
                              if m.group == group_name and m.col_name in df.columns]
                # Mahallenin iline göre kolonları filtrele (karşı ilin kolonlarını gösterme)
                if mahalle_il:
                    other_il = "mersin" if "adana" in mahalle_il.lower() else "adana"
                    group_cols = [c for c in group_cols
                                  if other_il not in normalize_text(c)
                                  or mahalle_il.lower()[:4] in normalize_text(c)]
                valid = [c for c in group_cols if pd.to_numeric(df[c], errors="coerce").notna().any()]
                if not valid:
                    return
                labels, m_vals, il_vals = [], [], []
                for c in valid:
                    s_all = pd.to_numeric(df[c], errors="coerce")
                    s_il = pd.to_numeric(df_il[c], errors="coerce") if mahalle_il else s_all
                    mn, mx = s_all.min(), s_all.max()
                    v = float(m_data[c]) if c in m_data.index and pd.notna(m_data[c]) else 0.0
                    il_avg = float(s_il.mean())
                    labels.append(c[:22])
                    if mx > mn:
                        m_vals.append((v-mn)/(mx-mn))
                        il_vals.append((il_avg-mn)/(mx-mn))
                    else:
                        m_vals.append(0); il_vals.append(0)
                il_label = f"{mahalle_il} Ort." if mahalle_il else "Genel Ort."
                fig = go.Figure()
                fig.add_trace(go.Scatterpolar(r=m_vals, theta=labels, fill='toself',
                                              name=str(m_data[name_col]), line_color=color))
                fig.add_trace(go.Scatterpolar(r=il_vals, theta=labels, fill='toself',
                                              name=il_label, line_color="#aaaaaa"))
                fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                                  showlegend=True, height=400, title=title)
                st.plotly_chart(fig, use_container_width=True)

            cr1, cr2 = st.columns(2)
            with cr1:
                render_radar("Ana Endeksler", "Ana Endeksler", "#d7191c")
            with cr2:
                render_radar("Alt Endeksler", "Alt Endeksler", "#2E75B6")

            # Endeks değerleri tablosu (il bazlı)
            all_endeks = [m.col_name for m in meta_map.values()
                          if m.group in ("Ana Endeksler", "Alt Endeksler")
                          and m.col_name in df.columns
                          and pd.to_numeric(df[m.col_name], errors="coerce").notna().any()]
            tbl = []
            for c in all_endeks:
                if c in m_data.index and pd.notna(m_data[c]):
                    try:
                        genel_ort = float(pd.to_numeric(df[c], errors="coerce").mean())
                        il_ort = float(pd.to_numeric(df_il[c], errors="coerce").mean()) if mahalle_il else genel_ort
                        grp = meta_map[c].group
                        tbl.append({"Gösterge": c, "Grup": grp,
                                    "Mahalle": round(float(m_data[c]), 4),
                                    f"{mahalle_il} Ort.": round(il_ort, 4),
                                    "Genel Ort.": round(genel_ort, 4)})
                    except:
                        pass
            if tbl:
                st.divider()
                st.subheader("Tüm Endeks Değerleri")
                st.dataframe(pd.DataFrame(tbl), use_container_width=True)

    with tab_debug:
        st.write(f"**Excel Satır:** {raw_len}")
        st.write(f"**GeoJSON:** {len(gdf)}")
        st.write(f"**Haritada:** {len(joined)}")
        st.write(f"**Geçerli:** {len(joined_valid)}")
        geo_keys = set(gdf["MAHALLEKOD"].dropna())
        excel_keys = set(df["MAHALLEKAYITNO"].dropna())
        st.write(f"**Ortak ID:** {len(geo_keys & excel_keys)}")
        st.write(f"**Sadece GeoJSON:** {len(geo_keys - excel_keys)}")
        st.write(f"**Sadece Excel:** {len(excel_keys - geo_keys)}")
        st.write("**Excel kolonları:**", list(df.columns))

if __name__ == "__main__":
    main()
