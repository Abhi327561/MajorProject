import os
import sys
import streamlit as st
import pandas as pd
import folium
from streamlit_folium import folium_static

# Set up module path so that 'src', 'config' can be resolved
# when running from project root via: streamlit run crop_health_dashboard/app.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import data_loader
from src import ndvi_analysis
from src import report_generator
import config
import plotly.graph_objects as go


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MONTH_NAMES = {
    1: "January", 2: "February", 3: "March", 4: "April",
    5: "May", 6: "June", 7: "July", 8: "August",
    9: "September", 10: "October", 11: "November", 12: "December",
}


def _health_color(status: str) -> str:
    """Return the hex colour associated with a health classification."""
    mapping = {
        "Healthy": config.COLOR_HEALTHY,
        "Moderate": config.COLOR_MODERATE,
        "Poor": config.COLOR_POOR,
    }
    return mapping.get(status, "#808080")


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

def main():
    # ── Page configuration ─────────────────────────────────────────────────
    st.set_page_config(
        page_title="AI-Based Intelligent Cropland Monitoring",
        page_icon="🌱",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("🌱 AI-Based Intelligent Cropland Monitoring")
    st.subheader("Crop Health Dashboard")
    st.markdown("---")

    # ── Load & validate datasets ───────────────────────────────────────────
    try:
        gdf, df_raw, validation_report = data_loader.load_and_validate_all()
    except Exception as exc:
        st.error(f"Critical error loading datasets: {exc}")
        st.info("Please verify the configuration paths in config.py and the dataset file formats.")
        return

    if not validation_report["success"]:
        st.error("Dataset validation failed. Please review:")
        for err in validation_report["errors"]:
            st.markdown(f"- {err}")
        for warn in validation_report.get("warnings", []):
            st.warning(warn)
        return

    if gdf.empty or df_raw.empty:
        st.warning("Loaded datasets are empty. No data available to display.")
        return

    # ── Pre-process NDVI time-series ───────────────────────────────────────
    try:
        df_processed = ndvi_analysis.process_all_fields(df_raw)
    except Exception as exc:
        st.error(f"Failed to process NDVI analysis: {exc}")
        return

    # ======================================================================
    # SIDEBAR
    # ======================================================================
    st.sidebar.header("Dashboard Controls")

    # ── Field selector ─────────────────────────────────────────────────────
    st.sidebar.markdown("### 🌾 Field Selection")
    field_lookup = {
        f"{row['field_name']} ({row['field_id']})": row['field_id']
        for _, row in gdf.iterrows()
    }
    selected_label = st.sidebar.selectbox("Select Field", list(field_lookup.keys()))
    selected_field_id = field_lookup[selected_label]

    field_meta = gdf[gdf["field_id"] == selected_field_id].iloc[0]
    field_ts = (
        df_processed[df_processed["field_id"] == selected_field_id]
        .sort_values("acquisition_date")
        .reset_index(drop=True)
    )

    if field_ts.empty:
        st.warning(
            f"No NDVI observations found for **{field_meta['field_name']}** "
            f"({selected_field_id})."
        )
        return

    # ── Cloud-cover control ────────────────────────────────────────────────
    st.sidebar.markdown("### ☁️ Cloud Cover Handling")
    cloud_mode = st.sidebar.radio(
        "Observation filter",
        [
            f"Exclude cloudy observations (cloud cover > {config.MAX_CLOUD_COVER}%)",
            "Include all observations (show raw data)",
        ],
        index=0,
        help=(
            "When 'Exclude' is selected, observations whose cloud cover exceeds "
            f"{config.MAX_CLOUD_COVER}% are flagged and their NDVI metrics are "
            "treated as unreliable for health assessment."
        ),
    )
    exclude_cloudy = cloud_mode.startswith("Exclude")

    # ── Date selector ──────────────────────────────────────────────────────
    st.sidebar.markdown("### 📅 Acquisition Date")
    date_labels = []
    date_map = {}
    for _, row in field_ts.iterrows():
        d_str = row["acquisition_date"].strftime("%Y-%m-%d")
        suffix = "  ☁️ Cloudy" if row["cloud_cover"] > config.MAX_CLOUD_COVER else ""
        label = f"{d_str}{suffix}"
        date_labels.append(label)
        date_map[label] = row["acquisition_date"]

    sel_date_label = st.sidebar.selectbox("Select Date", date_labels)
    sel_date = date_map[sel_date_label]

    # ── Selected observation row ───────────────────────────────────────────
    obs = field_ts[field_ts["acquisition_date"] == sel_date].iloc[0]
    is_obs_cloudy = obs["cloud_cover"] > config.MAX_CLOUD_COVER

    # ======================================================================
    # 5. CURRENT FIELD STATUS
    # ======================================================================
    st.header("📋 Current Field Status")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"**Field Name:** {field_meta['field_name']}")
        st.markdown(f"**Field ID:** `{selected_field_id}`")
        st.markdown(f"**Area:** {field_meta['area_ha']:.2f} ha")

    with col2:
        st.markdown(f"**Acquisition Date:** {sel_date.strftime('%Y-%m-%d')}")
        st.markdown(f"**Current NDVI (Mean):** {obs['ndvi_mean']:.3f}")
        st.markdown(f"**Cloud Cover:** {obs['cloud_cover']:.1f}%")

    with col3:
        if is_obs_cloudy and exclude_cloudy:
            badge_color = "#808080"
            badge_text = "EXCLUDED (CLOUDY)"
        else:
            badge_text = obs["health_status"].upper()
            badge_color = _health_color(obs["health_status"])

        st.markdown(
            f'<div style="background-color:{badge_color};color:#fff;padding:16px;'
            f'border-radius:8px;text-align:center;font-weight:700;font-size:22px;">'
            f'{badge_text}</div>',
            unsafe_allow_html=True,
        )

    # Scientific disclaimer
    st.info(
        "💡 **Scientific Note:** NDVI is an optical remote-sensing indicator of "
        "green-leaf chlorophyll content and crop canopy density. A low or declining "
        "NDVI value indicates vegetation stress (potentially caused by water deficit, "
        "nutrient deficiency, weed competition, or pest pressure) and should **not** "
        "be interpreted as a definitive disease diagnosis without direct field "
        "verification."
    )

    # ======================================================================
    # MAP & IMAGERY DUAL COLUMN SECTION
    # ======================================================================
    st.markdown("---")
    st.header("🗺️ Spatial View & Satellite Imagery")
    
    col_map, col_img = st.columns(2)
    
    # ── Interactive Folium Map ─────────────────────────────────────────────
    with col_map:
        st.markdown("### Interactive Field Boundary Map")
        try:
            # Centroid calculations
            centroid = field_meta.geometry.centroid
            lat = centroid.y
            lon = centroid.x
            
            m = folium.Map(location=[lat, lon], zoom_start=15)
            
            # Map imagery backdrop and boundaries
            folium.TileLayer(
                tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                attr="Esri World Imagery",
                name="Satellite Imagery",
                overlay=False,
                control=True
            ).add_to(m)
            folium.TileLayer('OpenStreetMap').add_to(m)
            
            poly_color = _health_color(obs["health_status"])
            if is_obs_cloudy and exclude_cloudy:
                poly_color = "#808080"
                
            geojson_style = lambda x: {
                'fillColor': poly_color,
                'color': poly_color,
                'weight': 3,
                'fillOpacity': 0.4
            }
            
            popup_text = f"""
            <b>Field:</b> {field_meta['field_name']}<br>
            <b>ID:</b> {selected_field_id}<br>
            <b>Area:</b> {field_meta['area_ha']:.2f} ha<br>
            <b>Health status:</b> {obs['health_status']}
            """
            
            folium.GeoJson(
                field_meta.geometry,
                style_function=geojson_style,
                tooltip=field_meta['field_name'],
                popup=folium.Popup(popup_text, max_width=250)
            ).add_to(m)
            
            folium.LayerControl().add_to(m)
            
            folium_static(m, width=600, height=400)
        except Exception as e:
            st.warning(f"Unable to render interactive map: {e}")
            
    # ── Satellite Thumbnail Imagery Support ─────────────────────────────────
    with col_img:
        st.markdown("### Field Capture Imagery")
        # Check files under data/imagery/
        img_name = f"{selected_field_id}_{sel_date.strftime('%Y-%m-%d')}.png"
        img_path = os.path.join(config.DATA_DIR, "imagery", img_name)
        
        if os.path.exists(img_path):
            st.image(img_path, caption=f"Satellite view of Field {selected_field_id} on {sel_date.strftime('%Y-%m-%d')}", use_column_width=True)
        else:
            st.info("ℹ️ Satellite imagery thumbnail not available for this observation date.")
            # Standard schematic canvas helper placeholder description
            st.markdown(
                """
                <div style="border: 2px dashed #CCCCCC; padding: 40px; border-radius: 8px; text-align: center; color: #808080;">
                    🌅 Image Feed Missing / Non-Generated
                </div>
                """,
                unsafe_allow_html=True
            )

    # ======================================================================
    # Plotly Trend & Health Status Band
    # ======================================================================
    st.markdown("---")
    st.header("📈 NDVI Time-Series Trend")
    
    try:
        # Build Plotly
        fig = go.Figure()
        
        # Plot valid points
        valid_pts = field_ts[field_ts['data_status'] == 'Valid']
        fig.add_trace(go.Scatter(
            x=valid_pts['acquisition_date'],
            y=valid_pts['ndvi_mean'],
            mode='lines+markers',
            name='Valid NDVI (Mean)',
            line=dict(color='#2CA02C', width=3),
            marker=dict(size=8, symbol='circle'),
            hovertemplate='Date: %{x|%Y-%m-%d}<br>NDVI: %{y:.3f}<br>Cloud Cover: %{text}%<extra></extra>',
            text=valid_pts['cloud_cover'].apply(lambda x: f"{x:.1f}")
        ))
        
        # Plot cloudy observations if they exist
        cloudy_pts = field_ts[field_ts['data_status'] == 'Cloudy']
        if not cloudy_pts.empty:
            fig.add_trace(go.Scatter(
                x=cloudy_pts['acquisition_date'],
                y=cloudy_pts['ndvi_mean'],
                mode='markers',
                name='Cloudy Observations',
                marker=dict(size=10, symbol='x', color='#FF7F0E'),
                hovertemplate='Date: %{x|%Y-%m-%d}<br>NDVI: %{y:.3f}<br>Cloud Cover: %{text}%<extra></extra>',
                text=cloudy_pts['cloud_cover'].apply(lambda x: f"{x:.1f}")
            ))
            
        # Highlight anomalies
        anomaly_pts = field_ts[field_ts['is_anomaly'] == True]
        if not anomaly_pts.empty:
            fig.add_trace(go.Scatter(
                x=anomaly_pts['acquisition_date'],
                y=anomaly_pts['ndvi_mean'],
                mode='markers',
                name='Stress Anomalies',
                marker=dict(size=14, symbol='triangle-up', color=config.COLOR_POOR, line=dict(color='black', width=1)),
                hovertemplate='Date: %{x|%Y-%m-%d}<br>NDVI: %{y:.3f}<br>Severity: %{text}<extra></extra>',
                text=anomaly_pts['anomaly_severity']
            ))

        # Add Horizontal Health Status Bands
        # Poor Band (NDVI < 0.30)
        fig.add_hrect(
            y0=0.0, y1=config.NDVI_POOR_THRESHOLD,
            fillcolor=config.COLOR_POOR, opacity=0.1,
            annotation_text="Poor Crop Health (< 0.30)",
            annotation_position="bottom left",
            annotation_font=dict(color=config.COLOR_POOR, size=10),
            line_width=0
        )
        
        # Moderate Band (0.30 <= NDVI < 0.60)
        fig.add_hrect(
            y0=config.NDVI_POOR_THRESHOLD, y1=config.NDVI_HEALTHY_THRESHOLD,
            fillcolor=config.COLOR_MODERATE, opacity=0.05,
            annotation_text="Moderate Crop Health (0.30 - 0.60)",
            annotation_position="bottom left",
            annotation_font=dict(color=config.COLOR_MODERATE, size=10),
            line_width=0
        )
        
        # Healthy Band (NDVI >= 0.60)
        fig.add_hrect(
            y0=config.NDVI_HEALTHY_THRESHOLD, y1=1.0,
            fillcolor=config.COLOR_HEALTHY, opacity=0.1,
            annotation_text="Healthy Crop (>= 0.60)",
            annotation_position="bottom left",
            annotation_font=dict(color=config.COLOR_HEALTHY, size=10),
            line_width=0
        )
        
        # VLine reference for selected observation
        fig.add_vline(
            x=sel_date,
            line_dash="dash",
            line_color="#4F4F4F",
            annotation_text="Selected Date",
            annotation_position="bottom right"
        )
        
        fig.update_layout(
            yaxis=dict(range=[0, 1.0]),
            hovermode="closest",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=40, r=40, t=40, b=40)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not render NDVI time-series chart: {e}")

    # ======================================================================
    # 6. SELECTED OBSERVATION DETAILS
    # ======================================================================
    st.header("📊 Selected Observation Details")

    if is_obs_cloudy and exclude_cloudy:
        st.warning(
            f"⚠️ **Observation excluded:** Cloud cover for this acquisition is "
            f"**{obs['cloud_cover']:.1f}%**, exceeding the configured threshold of "
            f"**{config.MAX_CLOUD_COVER}%**. NDVI values are shown for reference but "
            f"are not used for health classification. Switch to 'Include all "
            f"observations' in the sidebar to assess raw values."
        )

    col_a, col_b, col_c, col_d = st.columns(4)

    # Column A – NDVI values
    with col_a:
        st.metric("Mean NDVI", f"{obs['ndvi_mean']:.3f}" if pd.notna(obs['ndvi_mean']) else "N/A")
        st.metric("Median NDVI", f"{obs['ndvi_median']:.3f}" if pd.notna(obs['ndvi_median']) else "N/A")

    # Column B – Previous / delta
    with col_b:
        prev_ndvi = obs["prev_valid_ndvi"]
        has_prev = pd.notna(prev_ndvi)

        st.metric("Previous Valid NDVI", f"{prev_ndvi:.3f}" if has_prev else "N/A")

        delta_val = obs["ndvi_delta"]
        pct_val = obs["ndvi_pct_change"]

        if has_prev and pd.notna(delta_val):
            st.metric(
                "NDVI Change",
                f"{delta_val:+.3f}",
                delta=f"{pct_val:+.1f}%" if pd.notna(pct_val) else None,
            )
        else:
            st.markdown("**NDVI Change:** No previous observation available.")

    # Column C – Classification & cloud
    with col_c:
        st.metric("Health Classification", obs["health_status"])
        st.metric("Cloud Cover", f"{obs['cloud_cover']:.1f}%")

    # Column D – Anomaly status
    with col_d:
        if obs["is_anomaly"]:
            drop_val = obs["ndvi_drop"]
            severity = obs["anomaly_severity"]
            st.markdown(
                f'<div style="background:{config.COLOR_POOR}18;border:2px solid '
                f'{config.COLOR_POOR};padding:12px;border-radius:6px;text-align:center;">'
                f'<span style="color:{config.COLOR_POOR};font-weight:700;font-size:16px;">'
                f'⚠️ ANOMALY DETECTED</span><br>'
                f'<span style="font-size:14px;">Severity: <b>{severity}</b></span><br>'
                f'<span style="font-size:14px;">Drop: <b>{drop_val:.3f}</b></span></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div style="background:{config.COLOR_HEALTHY}18;border:2px solid '
                f'{config.COLOR_HEALTHY};padding:12px;border-radius:6px;text-align:center;">'
                f'<span style="color:{config.COLOR_HEALTHY};font-weight:700;font-size:16px;">'
                f'✓ NORMAL STATUS</span><br>'
                f'<span style="font-size:14px;">No significant vegetation-health drop detected.'
                f'</span></div>',
                unsafe_allow_html=True,
            )

    # ======================================================================
    # 8. DOWNLOAD FIELD HEALTH REPORT
    # ======================================================================
    st.markdown("---")
    st.header("📄 Download Farmer Field Report")
    st.markdown("Generate a PDF summarizing cropland health metrics, historical stress drop events, and farmer-focused recommendations.")
    
    # Calculate total historical anomalies for this field
    total_anoms = int(field_ts['is_anomaly'].sum())
    earliest_date_str = field_ts['acquisition_date'].iloc[0].strftime('%Y-%m-%d')
    latest_date_str = field_ts['acquisition_date'].iloc[-1].strftime('%Y-%m-%d')
    
    # Construct input structures for generator
    meta_dict = {
        "field_id": field_meta["field_id"],
        "field_name": field_meta["field_name"],
        "area_ha": field_meta["area_ha"]
    }
    
    obs_dict = {
        "acquisition_date_str": sel_date.strftime('%Y-%m-%d'),
        "ndvi_mean": obs["ndvi_mean"],
        "ndvi_median": obs["ndvi_median"],
        "prev_valid_ndvi": obs["prev_valid_ndvi"],
        "ndvi_delta": obs["ndvi_delta"],
        "ndvi_pct_change": obs["ndvi_pct_change"],
        "cloud_cover": obs["cloud_cover"],
        "health_status": obs["health_status"],
        "is_anomaly": obs["is_anomaly"],
        "anomaly_severity": obs["anomaly_severity"],
        "ndvi_drop": obs["ndvi_drop"]
    }
    
    try:
        pdf_buffer = report_generator.generate_pdf_report(
            meta_dict,
            obs_dict,
            total_anoms,
            earliest_date_str,
            latest_date_str
        )
        
        st.download_button(
            label="📥 Download Field Health Report (PDF)",
            data=pdf_buffer,
            file_name=f"Field_{selected_field_id}_Health_Report_{sel_date.strftime('%Y-%m-%d')}.pdf",
            mime="application/pdf"
        )
    except Exception as e:
        st.error(f"Error compiling PDF report: {e}")

    # ======================================================================
    # 7. DATA QUALITY & COMPLETENESS
    # ======================================================================
    st.markdown("---")
    st.header("📈 Data Quality & Completeness")

    total_obs = len(field_ts)
    valid_count = int((field_ts["data_status"] == "Valid").sum())
    cloudy_count = int((field_ts["data_status"] == "Cloudy").sum())
    excluded_count = int((field_ts["data_status"] == "Excluded").sum())

    missing_months_map = ndvi_analysis.detect_missing_months(field_ts)
    missing_months = missing_months_map.get(selected_field_id, [])

    col_q1, col_q2 = st.columns(2)

    with col_q1:
        st.markdown("### Observation Counts")
        quality_df = pd.DataFrame(
            {
                "Category": [
                    "Total Acquisitions",
                    "Valid Observations",
                    "Cloudy (cloud > " + str(config.MAX_CLOUD_COVER) + "%)",
                    "Excluded (missing NDVI)",
                ],
                "Count": [total_obs, valid_count, cloudy_count, excluded_count],
            }
        )
        st.table(quality_df)

    with col_q2:
        st.markdown("### Missing Months")
        if missing_months:
            st.warning(
                f"**{len(missing_months)}** month(s) have no valid, cloud-free "
                f"observation for this field in 2026."
            )
            names = [MONTH_NAMES.get(m, str(m)) for m in missing_months]
            st.markdown(", ".join(names))
        else:
            st.success(
                "✓ Full temporal coverage — all 12 calendar months have at least "
                "one valid, cloud-free observation."
            )


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
