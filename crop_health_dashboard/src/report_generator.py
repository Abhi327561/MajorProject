import io
import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

def get_health_description(status: str) -> str:
    """Returns a simplified farmer-friendly health description based on status."""
    descriptions = {
        "Healthy": (
            "Vegetation health currently appears healthy based on the NDVI indicator. "
            "The crop canopy is dense and showing optimal chlorophyll activity. "
            "No immediate action is needed. Continue standard management practices."
        ),
        "Moderate": (
            "Vegetation health is currently moderate. Minor stress or variability detected. "
            "Continued monitoring is recommended to ensure the field does not deteriorate. "
            "Check for localized watering or nutrient issues."
        ),
        "Poor": (
            "Vegetation health appears low based on the NDVI indicator. "
            "Further field inspection is recommended to diagnose potential issues such as "
            "pest infestations, water stress, or nutrient deficiencies."
        ),
        "Unknown": (
            "Vegetation health status is unknown (typically due to heavy cloud cover or "
            "missing satellite telemetry observations)."
        )
    }
    return descriptions.get(status, descriptions["Unknown"])

def get_anomaly_description(is_anomaly: bool, severity: str, drop: float) -> str:
    """Returns a farmer-friendly anomaly explanation."""
    if not is_anomaly:
        return "No significant NDVI stress event detected in the available observations."
    
    severity_label = "MODERATE" if severity == "Moderate" else "SEVERE"
    return (
        f"A significant decrease in vegetation index (NDVI drop of {drop:.2f}) was detected "
        f"compared with the previous valid observation. This has been flagged as a {severity_label} "
        f"stress event. Please perform a physical inspection of this field immediately to check for "
        f"pests, diseases, water logging, or drought."
    )

def generate_pdf_report(field_meta: dict, obs: dict, total_anomalies: int, earliest_date: str, latest_date: str) -> io.BytesIO:
    """Generates a farmer-facing PDF health report in-memory and returns the bytes buffer.

    Args:
        field_meta: Dictionary containing field metadata (field_id, field_name, area_ha).
        obs: Dictionary containing the selected observation row.
        total_anomalies: Total number of anomalies detected historically.
        earliest_date: Date string of the first telemetry.
        latest_date: Date string of the latest telemetry.

    Returns:
        BytesIO: In-memory PDF file.
    """
    buffer = io.BytesIO()
    
    # Page setup - 0.75 in margins
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )
    
    story = []
    
    # Styles
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=24,
        leading=28,
        textColor=colors.HexColor('#2CA02C'),
        spaceAfter=15
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#666666'),
        spaceAfter=20
    )
    
    h1_style = ParagraphStyle(
        'SectionH1',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=14,
        leading=18,
        textColor=colors.HexColor('#333333'),
        spaceBefore=12,
        spaceAfter=8,
        keepWithNext=True
    )
    
    body_style = ParagraphStyle(
        'BodyTextCustom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#333333'),
        spaceAfter=10
    )
    
    bold_body_style = ParagraphStyle(
        'BoldBodyCustom',
        parent=body_style,
        fontName='Helvetica-Bold'
    )
    
    callout_style = ParagraphStyle(
        'CalloutText',
        parent=body_style,
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#111111')
    )
    
    # Header Section
    story.append(Paragraph("AI-Based Intelligent Cropland Monitoring & Analysis", title_style))
    story.append(Paragraph(f"Report Generated on: {datetime.date.today().strftime('%B %d, %Y')} | Crop Health Dashboard Module", subtitle_style))
    story.append(Spacer(1, 10))
    
    # Field Details Section
    story.append(Paragraph("1. Field Identification & Metadata", h1_style))
    
    field_data = [
        [Paragraph("<b>Field Name:</b>", body_style), Paragraph(str(field_meta.get('field_name')), body_style),
         Paragraph("<b>Field ID:</b>", body_style), Paragraph(str(field_meta.get('field_id')), body_style)],
        [Paragraph("<b>Area Size:</b>", body_style), Paragraph(f"{field_meta.get('area_ha'):.2f} ha", body_style),
         Paragraph("<b>Date of Observation:</b>", body_style), Paragraph(str(obs.get('acquisition_date_str')), body_style)]
    ]
    
    t_field = Table(field_data, colWidths=[1.2*inch, 2.3*inch, 1.2*inch, 2.3*inch])
    t_field.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('LINEBELOW', (0,-1), (-1,-1), 1, colors.HexColor('#E0E0E0')),
    ]))
    story.append(t_field)
    story.append(Spacer(1, 15))
    
    # Current Observation Details Section
    story.append(Paragraph("2. Vegetation Analysis Results (NDVI)", h1_style))
    
    health_status = obs.get('health_status', 'Unknown')
    health_color_map = {
        "Healthy": '#2CA02C',
        "Moderate": '#FF7F0E',
        "Poor": '#D62728',
        "Unknown": '#808080'
    }
    status_hex = health_color_map.get(health_status, '#808080')
    
    # Format delta
    delta_val = obs.get('ndvi_delta')
    pct_val = obs.get('ndvi_pct_change')
    # Since pandas is needed, let's import it locally inside function to avoid namespace poll
    import pandas as pd
    if delta_val is not None and not pd.isna(delta_val):
        delta_str = f"{delta_val:+.3f} ({pct_val:+.1f}%)"
    else:
        delta_str = "N/A (First observation / baseline)"
        
    prev_ndvi = obs.get('prev_valid_ndvi')
    prev_ndvi_str = f"{prev_ndvi:.3f}" if prev_ndvi is not None and not pd.isna(prev_ndvi) else "N/A"
    
    obs_data = [
        [Paragraph("<b>Metric</b>", bold_body_style), Paragraph("<b>Value</b>", bold_body_style), Paragraph("<b>Description / Context</b>", bold_body_style)],
        [Paragraph("Mean NDVI", body_style), Paragraph(f"{obs.get('ndvi_mean', 0.0):.3f}", body_style), Paragraph("Average crop greenness index across the polygon.", body_style)],
        [Paragraph("Median NDVI", body_style), Paragraph(f"{obs.get('ndvi_median', 0.0):.3f}", body_style), Paragraph("Middle value of greenness, minimizing outlier effects.", body_style)],
        [Paragraph("Previous Valid NDVI", body_style), Paragraph(prev_ndvi_str, body_style), Paragraph("Vegetation health index from the last clear sky image.", body_style)],
        [Paragraph("NDVI Change (Delta)", body_style), Paragraph(delta_str, body_style), Paragraph("Difference compared to previous valid date.", body_style)],
        [Paragraph("Cloud Cover", body_style), Paragraph(f"{obs.get('cloud_cover', 0.0):.1f}%", body_style), Paragraph("Percentage of cloud coverage during capture.", body_style)],
        [Paragraph("Health Classification", body_style), Paragraph(f"<font color='{status_hex}'><b>{health_status.upper()}</b></font>", body_style), Paragraph("Categorized health based on scientific NDVI ranges.", body_style)]
    ]
    
    t_obs = Table(obs_data, colWidths=[1.8*inch, 1.8*inch, 3.4*inch])
    t_obs.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#F2F2F2')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#E0E0E0')),
    ]))
    story.append(t_obs)
    story.append(Spacer(1, 15))
    
    # Anomaly / Crop Stress Details
    story.append(Paragraph("3. Crop Anomaly & Stress Assessment", h1_style))
    is_anomaly = obs.get('is_anomaly', False)
    anomaly_severity = obs.get('anomaly_severity', 'None')
    ndvi_drop = obs.get('ndvi_drop', 0.0)
    
    anomaly_text = get_anomaly_description(is_anomaly, anomaly_severity, ndvi_drop)
    
    if is_anomaly:
        bg_color = colors.HexColor('#FDE8E8') # light crimson
        border_color = colors.HexColor('#F05252') # red border
    else:
        bg_color = colors.HexColor('#EAF8EA') # light green
        border_color = colors.HexColor('#31C48D') # green border
        
    t_anomaly = Table([[Paragraph(anomaly_text, callout_style)]], colWidths=[7.0*inch])
    t_anomaly.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BACKGROUND', (0,0), (-1,-1), bg_color),
        ('BOX', (0,0), (-1,-1), 1.5, border_color),
        ('TOPPADDING', (0,0), (-1,-1), 10),
        ('BOTTOMPADDING', (0,0), (-1,-1), 10),
        ('LEFTPADDING', (0,0), (-1,-1), 12),
        ('RIGHTPADDING', (0,0), (-1,-1), 12),
    ]))
    story.append(t_anomaly)
    story.append(Spacer(1, 15))
    
    # Farmer Agronomic Recommendation Section
    story.append(Paragraph("4. Agronomic Recommendations & Interpretation", h1_style))
    rec_text = get_health_description(health_status)
    story.append(Paragraph(rec_text, body_style))
    story.append(Spacer(1, 10))
    
    # Historical Context Section
    story.append(Paragraph("5. Historical Context", h1_style))
    hist_text = (
        f"This field has been monitored from <b>{earliest_date}</b> to <b>{latest_date}</b>. "
        f"Over this monitoring window, a total of <b>{total_anomalies}</b> sudden vegetation index drop "
        f"stress anomaly event(s) were captured. Regular remote sensing is highly recommended to "
        f"continue tracking changes in plant canopy thickness and chlorophyll absorption."
    )
    story.append(Paragraph(hist_text, body_style))
    story.append(Spacer(1, 20))
    
    # Disclaimer
    story.append(Paragraph("<b>Disclaimer:</b> NDVI is a remote-sensing indicator of cropland greenness and is not a definitive diagnosis of specific plant diseases, pests, or soil problems. Field verification is always recommended before executing treatment applications.", subtitle_style))
    
    # Build Document
    doc.build(story)
    buffer.seek(0)
    return buffer
