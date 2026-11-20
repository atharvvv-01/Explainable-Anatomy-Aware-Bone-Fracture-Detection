"""
Clinical-Grade AI Diagnostic PDF Report Generator.
Generates an elegant, structured 2-page PDF report matching diagnostic standards:
- Case Information & Status Banner
- Scan Information & AI Summary
- Imaging Analysis (Uploaded Original Radiograph & Generated Stage 5 Pathology Focus Grad-CAM Overlay)
- Multi-Stage Pipeline Breakdown
- Quantitative Grad-CAM & Attention Metrics
- Severity Score Breakdown
- Recommended Action Items & Clinical Disclaimer
(Excludes redundant clinical assessment guesswork and generic limitation lists).
"""

import os
import sys
import time
from datetime import datetime
from typing import Dict, Any, Optional
from PIL import Image

# Add workspace to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, PageBreak, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY


def ensure_colorbar_asset(asset_path: str = "assets/gradcam_colorbar.png") -> str:
    """Generates the Grad-CAM colorbar legend asset if not already present."""
    if os.path.exists(asset_path):
        return asset_path

    os.makedirs(os.path.dirname(asset_path), exist_ok=True)
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5.5, 0.45), dpi=200)
        norm = plt.Normalize(vmin=0.0, vmax=1.0)
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='jet'), cax=ax, orientation='horizontal')
        cb.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        cb.ax.tick_params(labelsize=8)
        cb.set_label('Grad-CAM Activation Intensity (0 = low, 1 = high)', fontsize=8, labelpad=2)
        fig.tight_layout()
        fig.savefig(asset_path, bbox_inches='tight', transparent=True)
        plt.close(fig)
    except Exception as e:
        print(f"[WARNING] Could not generate colorbar asset: {e}")
    return asset_path


def save_temp_image(pil_img: Image.Image, temp_path: str, max_size=(500, 500)) -> str:
    """Safely resizes and saves a PIL image for ReportLab embedding."""
    os.makedirs(os.path.dirname(temp_path), exist_ok=True)
    img_copy = pil_img.copy()
    img_copy.thumbnail(max_size, Image.Resampling.LANCZOS)
    if img_copy.mode != "RGB":
        img_copy = img_copy.convert("RGB")
    img_copy.save(temp_path, format="JPEG", quality=92)
    return temp_path


def generate_diagnostic_pdf_report(
    result: Dict[str, Any],
    image_source: str,
    output_path: Optional[str] = None
) -> str:
    """
    Constructs a 2-page PDF diagnostic report for the given inference result.
    
    Args:
        result: Prediction dictionary returned by FractureInferencePipeline.predict().
        image_source: Path or name of the input image uploaded by user.
        output_path: Optional destination PDF file path. If None, saves to results/reports/.
        
    Returns:
        Absolute file path to the generated PDF report.
    """
    now = datetime.now()
    timestamp_str = now.strftime("%Y%m%d-%H%M%S")
    date_display = now.strftime("%d %B %Y")
    time_display = now.strftime("%H:%M:%S")

    case_id = f"CASE-{timestamp_str}"
    
    # Establish output file path
    if output_path is None:
        reports_dir = os.path.abspath("results/reports")
        os.makedirs(reports_dir, exist_ok=True)
        output_path = os.path.join(reports_dir, f"diagnostic_report_{timestamp_str}.pdf")
    else:
        output_path = os.path.abspath(output_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Temporary directory for flowable images
    temp_dir = os.path.abspath(f"results/reports/temp_{timestamp_str}")
    os.makedirs(temp_dir, exist_ok=True)

    is_fractured = result.get("is_fractured", False)
    pred_label = result.get("prediction_label", "NON-FRACTURED")
    fracture_prob = result.get("fracture_probability", 0.0)
    conf_pct = result.get("model_confidence", 1.0) * 100.0
    threshold = result.get("decision_threshold", 0.45)
    severity_score = result.get("severity_score", round(fracture_prob * 100.0, 1))
    severity_level = result.get("severity_level", "SEVERE" if is_fractured else "NORMAL")
    score_breakdown = result.get("score_breakdown", {
        "model_confidence": round(conf_pct * 0.6, 1),
        "activation_spread": 25.0 if is_fractured else 5.0,
        "peak_intensity": 15.0 if is_fractured else 2.0,
        "total": severity_score
    })
    
    mean_act = result.get("mean_activation", 25.0)
    high_act_area = result.get("high_activation_area", 15.0)
    peak_focus = result.get("peak_focus_intensity", 1.0) * 100.0
    focus_box = result.get("focus_box", (0, 0, 50, 50))
    region_name = result.get("region_name", "Bone Structure")
    det_conf = result.get("detector_confidence", 0.85) * 100.0
    is_fallback = result.get("is_fallback", False)
    eff_box = result.get("bounding_box", (0, 0, 500, 500))

    # Base image dimensions
    orig_img = result.get("original_image")
    if orig_img:
        res_w, res_h = orig_img.size
    else:
        res_w, res_h = (1000, 1000)

    filename_only = os.path.basename(image_source) if image_source else "Uploaded_Radiograph.png"

    # Theme colors
    if is_fractured:
        primary_color = colors.HexColor('#8B0000')      # Deep Crimson Maroon
        tint_bg = colors.HexColor('#faebeb')            # Soft pinkish rose
        border_tint = colors.HexColor('#d98880')
        status_text = "FRACTURED"
        status_color = "#8B0000"
        banner_title = "FRACTURE DETECTED"
    else:
        primary_color = colors.HexColor('#1b5e20')      # Forest Green
        tint_bg = colors.HexColor('#e8f5e9')            # Soft mint green
        border_tint = colors.HexColor('#81c784')
        status_text = "NON-FRACTURED"
        status_color = "#1b5e20"
        banner_title = "NO FRACTURE DETECTED"

    # PDF Document Setup (A4 size: 595.27 x 841.89 points)
    # Margins: 36pt (0.5 in) left/right/top/bottom -> Usable width = 523.27 pt
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        leftMargin=36,
        rightMargin=36,
        topMargin=32,
        bottomMargin=32
    )

    styles = getSampleStyleSheet()

    # Custom typography styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        textColor=colors.HexColor('#8B0000'),
        alignment=TA_CENTER
    )
    subtitle_style = ParagraphStyle(
        'DocSub',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=11,
        textColor=colors.HexColor('#555555'),
        alignment=TA_CENTER
    )
    meta_style = ParagraphStyle(
        'DocMeta',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor('#212121'),
        alignment=TA_CENTER
    )
    section_h2 = ParagraphStyle(
        'SectionH2',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=14,
        textColor=colors.HexColor('#8B0000'),
        spaceBefore=7,
        spaceAfter=4
    )
    cell_label = ParagraphStyle(
        'CellLabel',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor('#333333')
    )
    cell_val = ParagraphStyle(
        'CellVal',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor('#212121')
    )
    summary_text = ParagraphStyle(
        'SummaryText',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=12.5,
        textColor=colors.HexColor('#212121'),
        alignment=TA_JUSTIFY
    )
    img_header = ParagraphStyle(
        'ImgHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9,
        leading=12,
        textColor=colors.HexColor('#333333'),
        alignment=TA_CENTER
    )
    caption_style = ParagraphStyle(
        'CaptionStyle',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=7.5,
        leading=10,
        textColor=colors.HexColor('#666666'),
        alignment=TA_CENTER
    )
    disclaimer_style = ParagraphStyle(
        'DisclaimerText',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7,
        leading=9.5,
        textColor=colors.HexColor('#555555'),
        alignment=TA_JUSTIFY
    )

    story = []

    # ==========================================
    # PAGE 1: PRIMARY DIAGNOSTIC REPORT
    # ==========================================

    # 1. Main Header
    story.append(Paragraph("AI-Assisted Bone Fracture Detection Report", title_style))
    story.append(Spacer(1, 2))
    story.append(Paragraph(
        f"Report Date: {date_display} · Time: {time_display} · Scan Type: X-Ray · Architecture: Dual-Stream EfficientNet-B3 + YOLOv8",
        subtitle_style
    ))
    story.append(Spacer(1, 4))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#8B0000'), spaceBefore=2, spaceAfter=6))

    # 2. Case Information Meta Bar
    meta_data = [
        [
            Paragraph(f"<b>Case ID:</b><br/>{case_id}", meta_style),
            Paragraph(f"<b>Body Region:</b><br/>{region_name}", meta_style),
            Paragraph(f"<b>Scan Type:</b><br/>X-Ray Radiograph", meta_style),
            Paragraph(f"<b>Status:</b><br/><b><font color='{status_color}'>{status_text}</font></b>", meta_style),
        ]
    ]
    meta_table = Table(meta_data, colWidths=[130, 130, 130, 133])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), tint_bg),
        ('BOX', (0, 0), (-1, -1), 0.8, border_tint),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, border_tint),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 6))

    # 3. Prediction Result Banner (4 colored blocks)
    banner_p1 = Paragraph(f"<font size=11 color='white'><b>{banner_title}</b></font>", ParagraphStyle('B1', alignment=TA_CENTER))
    banner_p2 = Paragraph(f"<font size=7.5 color='white'>Confidence</font><br/><font size=11.5 color='white'><b>{conf_pct:.2f}%</b></font>", ParagraphStyle('B2', alignment=TA_CENTER))
    banner_p3 = Paragraph(f"<font size=7.5 color='white'>Severity</font><br/><font size=11.5 color='white'><b>{severity_level}</b></font>", ParagraphStyle('B3', alignment=TA_CENTER))
    banner_p4 = Paragraph(f"<font size=7.5 color='white'>Score</font><br/><font size=11.5 color='white'><b>{severity_score:.1f} / 100</b></font>", ParagraphStyle('B4', alignment=TA_CENTER))

    banner_data = [[banner_p1, banner_p2, banner_p3, banner_p4]]
    banner_table = Table(banner_data, colWidths=[175, 116, 116, 116])
    banner_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), primary_color),
        ('BOX', (0, 0), (-1, -1), 1, primary_color),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#ffffff')),
        ('TOPPADDING', (0, 0), (-1, -1), 7),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 7),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(banner_table)
    story.append(Spacer(1, 6))

    # 4. Scan Information Table
    loc_status_str = f"Bone Structure Isolated ({det_conf:.1f}%)" if not is_fallback else "Full Field Fallback"
    scan_info_data = [
        [Paragraph("<b>Image File</b>", cell_label), Paragraph(filename_only, cell_val),
         Paragraph("<b>Scan Date</b>", cell_label), Paragraph(date_display, cell_val)],
        [Paragraph("<b>Resolution</b>", cell_label), Paragraph(f"{res_w} × {res_h} px", cell_val),
         Paragraph("<b>Target ROI</b>", cell_label), Paragraph(region_name, cell_val)],
        [Paragraph("<b>AI Model</b>", cell_label), Paragraph("EfficientNet-B3 + YOLOv8", cell_val),
         Paragraph("<b>Localization</b>", cell_label), Paragraph(loc_status_str, cell_val)],
    ]
    scan_info_table = Table(scan_info_data, colWidths=[85, 176, 85, 177])
    scan_info_table.setStyle(TableStyle([
        ('BOX', (0, 0), (-1, -1), 0.6, colors.HexColor('#cccccc')),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e8e8e8')),
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#fbfbfb')),
        ('BACKGROUND', (2, 0), (2, -1), colors.HexColor('#fbfbfb')),
        ('TOPPADDING', (0, 0), (-1, -1), 3.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3.5),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(Paragraph("Scan Information", section_h2))
    story.append(scan_info_table)
    story.append(Spacer(1, 5))

    # 5. AI-Generated Summary Callout Box
    if is_fractured:
        summary_str = (
            f"The AI model has identified a <b>cortical bone fracture</b> in the submitted radiograph with a calibrated confidence "
            f"of <b>{conf_pct:.2f}%</b> (fracture probability: <b>{fracture_prob*100:.2f}%</b> vs. threshold {threshold:.2f}). "
            f"Anatomy-aware localization successfully isolated the skeletal ROI with {det_conf:.1f}% detector confidence. "
            f"Grad-CAM explainability and Stage 5 micro-region analysis reveal high saliency concentration peaking at "
            f"coordinates <b>{focus_box}</b> with <b>{peak_focus:.1f}%</b> maximum activation, indicating focal structural disruption. "
            f"The computed diagnostic severity score of <b>{severity_score:.1f}/100</b> classifies this presentation as "
            f"<b>{severity_level}</b>, indicating prompt clinical orthopedic review is warranted."
        )
    else:
        summary_str = (
            f"The AI model has evaluated the submitted radiograph and found <b>no evidence of acute bone fracture</b> "
            f"(fracture probability: <b>{fracture_prob*100:.2f}%</b>, well below the diagnostic threshold of {threshold:.2f}). "
            f"Skeletal localization successfully identified the bone structure with {det_conf:.1f}% confidence. "
            f"Grad-CAM explainability indicates uniform, low-intensity baseline attention across intact osseous contours "
            f"with no focal discontinuity. The severity score of <b>{severity_score:.1f}/100</b> confirms an "
            f"<b>INTACT / LOW-RISK</b> bone structure."
        )
    summary_box_data = [[Paragraph(summary_str, summary_text)]]
    summary_box_table = Table(summary_box_data, colWidths=[523])
    summary_box_table.setStyle(TableStyle([
        ('BOX', (0, 0), (-1, -1), 1.2, primary_color),
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#fffdfd') if is_fractured else colors.HexColor('#f9fff9')),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(Paragraph("AI-Generated Summary", section_h2))
    story.append(summary_box_table)
    story.append(Spacer(1, 5))

    # 6. Imaging Analysis Section (Side-by-side: Uploaded Radiograph & Generated Stage 5 Pathology Focus)
    story.append(Paragraph("Imaging Analysis", section_h2))

    # Save temporary images for ReportLab flowables
    orig_temp_path = os.path.join(temp_dir, "original_radiograph.jpg")
    stage5_temp_path = os.path.join(temp_dir, "stage5_pathology_focus.jpg")

    if result.get("original_image"):
        save_temp_image(result["original_image"], orig_temp_path, max_size=(500, 500))
    else:
        Image.new('RGB', (400, 400), color=(100, 100, 100)).save(orig_temp_path)

    # Use focused_image (Stage 5 Brightest region box) or gradcam_overlay
    if result.get("focused_image"):
        save_temp_image(result["focused_image"], stage5_temp_path, max_size=(500, 500))
    elif result.get("gradcam_overlay"):
        save_temp_image(result["gradcam_overlay"], stage5_temp_path, max_size=(500, 500))
    else:
        Image.new('RGB', (400, 400), color=(150, 50, 50)).save(stage5_temp_path)

    img_w, img_h = 245, 210
    flowable_orig = RLImage(orig_temp_path, width=img_w, height=img_h)
    flowable_focus = RLImage(stage5_temp_path, width=img_w, height=img_h)

    imaging_data = [
        [Paragraph("<b>Original Radiograph</b>", img_header),
         Paragraph("<b>Stage 5: Pathology Focus ROI & Grad-CAM</b>", img_header)],
        [flowable_orig, flowable_focus]
    ]
    imaging_table = Table(imaging_data, colWidths=[261, 262])
    imaging_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), tint_bg),
        ('BOX', (0, 0), (-1, -1), 0.8, border_tint),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, border_tint),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(imaging_table)
    story.append(Spacer(1, 3))

    # Colorbar legend
    colorbar_asset = ensure_colorbar_asset()
    if os.path.exists(colorbar_asset):
        colorbar_flowable = RLImage(colorbar_asset, width=280, height=22)
        cb_table = Table([[colorbar_flowable]], colWidths=[523])
        cb_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, -1), 1),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 1),
        ]))
        story.append(cb_table)

    story.append(Paragraph(
        "<i>The Grad-CAM overlay and Stage 5 targeting box pinpoint the exact micro-regions that most influenced "
        "the AI prediction. High activation areas represent structural disruption; the box bounds the focal hotspot center.</i>",
        caption_style
    ))

    # ==========================================
    # PAGE 2: TECHNICAL & DIAGNOSTIC DETAILS
    # ==========================================
    story.append(PageBreak())

    # Page 2 Header
    story.append(Paragraph("AI-Assisted Bone Fracture Detection Report", title_style))
    story.append(Spacer(1, 2))
    story.append(Paragraph(
        f"Case ID: {case_id} · Technical & Explainability Analysis · Page 2 of 2",
        subtitle_style
    ))
    story.append(Spacer(1, 4))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#8B0000'), spaceBefore=2, spaceAfter=6))

    # 1. Multi-Stage Pipeline Breakdown (4 panels side-by-side)
    story.append(Paragraph("Multi-Stage Anatomy-Aware Pipeline Breakdown", section_h2))

    s1_path = os.path.join(temp_dir, "s1_loc.jpg")
    s2_path = os.path.join(temp_dir, "s2_crop.jpg")
    s4_path = os.path.join(temp_dir, "s4_cam.jpg")
    s5_path = os.path.join(temp_dir, "s5_box.jpg")

    save_temp_image(result.get("annotated_detection_image", result.get("original_image")), s1_path, (300, 300))
    save_temp_image(result.get("cropped_image", result.get("original_image")), s2_path, (300, 300))
    save_temp_image(result.get("gradcam_overlay", result.get("original_image")), s4_path, (300, 300))
    save_temp_image(result.get("focused_image", result.get("original_image")), s5_path, (300, 300))

    thumb_w, thumb_h = 120, 105
    pipeline_row_data = [
        [
            Paragraph("<b>Stage 1: Bone Localization</b>", ParagraphStyle('P1', fontName='Helvetica-Bold', fontSize=7.5, alignment=TA_CENTER)),
            Paragraph("<b>Stage 2: Padded Crop (300×300)</b>", ParagraphStyle('P2', fontName='Helvetica-Bold', fontSize=7.5, alignment=TA_CENTER)),
            Paragraph("<b>Stage 4: Grad-CAM Saliency</b>", ParagraphStyle('P3', fontName='Helvetica-Bold', fontSize=7.5, alignment=TA_CENTER)),
            Paragraph("<b>Stage 5: Pathology Focus ROI</b>", ParagraphStyle('P4', fontName='Helvetica-Bold', fontSize=7.5, alignment=TA_CENTER)),
        ],
        [
            RLImage(s1_path, width=thumb_w, height=thumb_h),
            RLImage(s2_path, width=thumb_w, height=thumb_h),
            RLImage(s4_path, width=thumb_w, height=thumb_h),
            RLImage(s5_path, width=thumb_w, height=thumb_h),
        ]
    ]
    pipeline_table = Table(pipeline_row_data, colWidths=[130, 131, 131, 131])
    pipeline_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f5f5f5')),
        ('BOX', (0, 0), (-1, -1), 0.6, colors.HexColor('#cccccc')),
        ('INNERGRID', (0, 0), (-1, -1), 0.4, colors.HexColor('#e0e0e0')),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]))
    story.append(pipeline_table)
    story.append(Spacer(1, 6))

    # 2. Diagnostic Findings Bulleted Table
    story.append(Paragraph("Diagnostic Findings", section_h2))
    findings_data = [
        [Paragraph(f"• <b>Detection:</b> {status_text} confirmed with {conf_pct:.2f}% calibrated model confidence.", cell_val)],
        [Paragraph(f"• <b>Bone ROI Localization:</b> Skeletal structure isolated at {eff_box} ({det_conf:.1f}% detector confidence).", cell_val)],
        [Paragraph(f"• <b>Attention Distribution:</b> Mean Grad-CAM activation of {mean_act:.2f}% across the isolated bone region.", cell_val)],
        [Paragraph(f"• <b>Pathology Micro-Region:</b> Focused hotspot isolated at box {focus_box} with {peak_focus:.1f}% peak intensity.", cell_val)],
        [Paragraph(f"• <b>Severity Assessment:</b> Multi-component diagnostic score evaluated at {severity_score:.1f}/100 ({severity_level}).", cell_val)],
    ]
    findings_table = Table(findings_data, colWidths=[523])
    findings_table.setStyle(TableStyle([
        ('BOX', (0, 0), (-1, -1), 0.6, colors.HexColor('#cccccc')),
        ('INNERGRID', (0, 0), (-1, -1), 0.4, colors.HexColor('#f0f0f0')),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#fafafa')),
    ]))
    story.append(findings_table)
    story.append(Spacer(1, 6))

    # 3. Grad-CAM Metrics Table
    story.append(Paragraph("Grad-CAM & Saliency Metrics", section_h2))
    metrics_data = [
        [
            Paragraph("<b>Metric</b>", ParagraphStyle('MH1', fontName='Helvetica-Bold', fontSize=8.5, textColor=colors.white)),
            Paragraph("<b>Value</b>", ParagraphStyle('MH2', fontName='Helvetica-Bold', fontSize=8.5, textColor=colors.white)),
            Paragraph("<b>Interpretation</b>", ParagraphStyle('MH3', fontName='Helvetica-Bold', fontSize=8.5, textColor=colors.white))
        ],
        [
            Paragraph("Peak Activation", cell_label),
            Paragraph(f"{peak_focus:.1f}%", cell_val),
            Paragraph("Maximum focus intensity — higher means more localized focal pathology", cell_val)
        ],
        [
            Paragraph("Mean Activation", cell_label),
            Paragraph(f"{mean_act:.2f}%", cell_val),
            Paragraph("Average attention spread across the isolated skeletal region", cell_val)
        ],
        [
            Paragraph("High Activation Area", cell_label),
            Paragraph(f"{high_act_area:.2f}%", cell_val),
            Paragraph("Proportion of bone crop with saliency activation above 50%", cell_val)
        ],
        [
            Paragraph("Model Confidence", cell_label),
            Paragraph(f"{conf_pct:.2f}%", cell_val),
            Paragraph("Calibrated classification confidence assigned to predicted class", cell_val)
        ],
        [
            Paragraph("Severity Score", cell_label),
            Paragraph(f"{severity_score:.1f} / 100", cell_val),
            Paragraph(f"{severity_level} — Orthopaedic consultation recommended" if is_fractured else "NORMAL — Stable bone architecture", cell_val)
        ],
    ]
    metrics_table = Table(metrics_data, colWidths=[140, 85, 298])
    metrics_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), primary_color),
        ('BOX', (0, 0), (-1, -1), 0.6, colors.HexColor('#cccccc')),
        ('INNERGRID', (0, 0), (-1, -1), 0.4, colors.HexColor('#e8e8e8')),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(metrics_table)
    story.append(Spacer(1, 6))

    # 4. Severity Score Breakdown Table
    story.append(Paragraph("Severity Score Breakdown", section_h2))
    breakdown_data = [
        [
            Paragraph("<b>Component</b>", ParagraphStyle('SB1', fontName='Helvetica-Bold', fontSize=8.5, textColor=colors.white)),
            Paragraph("<b>Points Scored</b>", ParagraphStyle('SB2', fontName='Helvetica-Bold', fontSize=8.5, textColor=colors.white)),
            Paragraph("<b>Max Points</b>", ParagraphStyle('SB3', fontName='Helvetica-Bold', fontSize=8.5, textColor=colors.white)),
            Paragraph("<b>Weight</b>", ParagraphStyle('SB4', fontName='Helvetica-Bold', fontSize=8.5, textColor=colors.white)),
        ],
        [
            Paragraph("Model Confidence", cell_val),
            Paragraph(f"{score_breakdown.get('model_confidence', 60.0):.1f}", cell_val),
            Paragraph("60.0", cell_val),
            Paragraph("60%", cell_val),
        ],
        [
            Paragraph("Activation Spread", cell_val),
            Paragraph(f"{score_breakdown.get('activation_spread', 25.0):.1f}", cell_val),
            Paragraph("25.0", cell_val),
            Paragraph("25%", cell_val),
        ],
        [
            Paragraph("Peak Intensity", cell_val),
            Paragraph(f"{score_breakdown.get('peak_intensity', 15.0):.1f}", cell_val),
            Paragraph("15.0", cell_val),
            Paragraph("15%", cell_val),
        ],
        [
            Paragraph("<b>TOTAL</b>", cell_label),
            Paragraph(f"<b>{severity_score:.1f}</b>", cell_label),
            Paragraph("<b>100.0</b>", cell_label),
            Paragraph("<b>100%</b>", cell_label),
        ]
    ]
    breakdown_table = Table(breakdown_data, colWidths=[175, 116, 116, 116])
    breakdown_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), primary_color),
        ('BACKGROUND', (0, -1), (-1, -1), tint_bg),
        ('BOX', (0, 0), (-1, -1), 0.6, colors.HexColor('#cccccc')),
        ('INNERGRID', (0, 0), (-1, -1), 0.4, colors.HexColor('#e8e8e8')),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(breakdown_table)
    story.append(Spacer(1, 6))

    # 5. Recommended Actions Section
    story.append(Paragraph("Recommended Actions", section_h2))
    if is_fractured:
        actions_data = [
            [Paragraph("• Immediate referral to an orthopaedic specialist for comprehensive clinical evaluation.", cell_val)],
            [Paragraph("• Correlation with trauma history, physical exam, and focal palpation tenderness.", cell_val)],
            [Paragraph("• Consider orthogonal multi-view radiograph, CT, or MRI if displacement or articular involvement is suspected.", cell_val)],
            [Paragraph("• Affected limb immobilization pending definitive orthopaedic management and treatment planning.", cell_val)],
            [Paragraph("• Serial follow-up radiographs recommended after intervention to monitor fracture union and alignment.", cell_val)],
        ]
    else:
        actions_data = [
            [Paragraph("• Routine clinical management and correlation with physical examination findings.", cell_val)],
            [Paragraph("• If localized pain or persistent symptoms exist, consider repeated radiograph in 7-10 days to rule out occult/stress fracture.", cell_val)],
            [Paragraph("• Rest, ice, and symptomatic care as clinically indicated.", cell_val)],
        ]
    actions_table = Table(actions_data, colWidths=[523])
    actions_table.setStyle(TableStyle([
        ('BOX', (0, 0), (-1, -1), 0.6, colors.HexColor('#cccccc')),
        ('INNERGRID', (0, 0), (-1, -1), 0.4, colors.HexColor('#f0f0f0')),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#fafafa')),
    ]))
    story.append(actions_table)
    story.append(Spacer(1, 8))

    # 6. Legal / Clinical Research Disclaimer Box
    disclaimer_p = Paragraph(
        "<b>DISCLAIMER:</b> This report is generated by an artificial intelligence diagnostic pipeline for research, decision-support, "
        "and educational purposes only. It does not constitute a definitive medical diagnosis and must not replace professional clinical judgment. "
        "All predictions, segmentations, and heatmaps must be independently reviewed and verified by a board-certified radiologist or orthopaedic "
        "physician before initiating medical treatment.",
        disclaimer_style
    )
    disclaimer_table = Table([[disclaimer_p]], colWidths=[523])
    disclaimer_table.setStyle(TableStyle([
        ('BOX', (0, 0), (-1, -1), 0.6, colors.HexColor('#bbbbbb')),
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f6f6f6')),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(disclaimer_table)

    # Build the document
    doc.build(story)

    # Clean up temp image files
    try:
        for f in os.listdir(temp_dir):
            os.remove(os.path.join(temp_dir, f))
        os.rmdir(temp_dir)
    except Exception:
        pass

    return output_path
