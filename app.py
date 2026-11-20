"""
Streamlit Web Application for Anatomy-Guided Bone Fracture Detection.
Features:
- Home / Project Overview
- Real-time Fracture Detection & Explainability
- Model Performance Dashboard (Test Set Metrics, Confusion Matrix, ROC/PR Curves)
- Experiment Comparison Table & Analysis
- Research Documentation, Limitations & Medical Disclaimers
"""

import os
import json
from PIL import Image
import pandas as pd
import numpy as np
import streamlit as st

# Configure Streamlit page layout
st.set_page_config(
    page_title="Bone Fracture AI | Anatomy-Guided Diagnosis",
    page_icon="🦴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark navy medical theme
st.markdown("""
<style>
    .main {
        background-color: #0b111e;
        color: #e2e8f0;
    }
    .stSidebar {
        background-color: #0f172a;
    }
    .stMetric {
        background-color: #1e293b;
        border-radius: 8px;
        padding: 12px;
        border: 1px solid #334155;
    }
    .status-fractured {
        background-color: #7f1d1d;
        color: #fecaca;
        padding: 12px 18px;
        border-radius: 8px;
        font-weight: 800;
        font-size: 20px;
        border: 1px solid #ef4444;
        text-align: center;
    }
    .status-normal {
        background-color: #14532d;
        color: #bbf7d0;
        padding: 12px 18px;
        border-radius: 8px;
        font-weight: 800;
        font-size: 20px;
        border: 1px solid #22c55e;
        text-align: center;
    }
    .warning-box {
        background-color: #78350f;
        color: #fef3c7;
        padding: 10px 14px;
        border-radius: 6px;
        border-left: 5px solid #f59e0b;
        margin-top: 10px;
        margin-bottom: 10px;
    }
    .disclaimer-box {
        background-color: #1e293b;
        color: #94a3b8;
        padding: 14px;
        border-radius: 8px;
        font-size: 13px;
        border: 1px solid #334155;
        margin-top: 25px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_pipeline():
    """Initializes and caches the two-stage inference pipeline."""
    from src.inference.pipeline import FractureInferencePipeline

    ckpt = "models/checkpoints/best_model.pth"
    if not os.path.exists(ckpt):
        # Check for alternative experiment checkpoint
        for alt in ["models/checkpoints/anatomy_guided_best.pth", "models/checkpoints/baseline_full_image_best.pth"]:
            if os.path.exists(alt):
                ckpt = alt
                break

    thresh = "config/best_threshold.json"
    detector_ckpt = "models/detector/yolov8n_anatomy.pt"

    pipeline = FractureInferencePipeline(
        classifier_checkpoint=ckpt,
        detector_path=detector_ckpt,
        threshold_path=thresh,
        conf_threshold=0.35,
        padding_ratio=0.12
    )
    return pipeline


# Sidebar Navigation
st.sidebar.title("🦴 Fracture AI Dashboard")
st.sidebar.markdown("**Two-Stage Anatomy-Guided Radiograph Diagnostics**")
page = st.sidebar.radio(
    "Select Navigation Section:",
    [
        "Home / Overview",
        "Fracture Detection",
        "Model Performance",
        "Experiment Comparison",
        "About & Limitations"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Model Configuration")
st.sidebar.markdown("- **Stage 1 Detector:** YOLOv8n (Bone ROI Localizer)")
st.sidebar.markdown("- **Stage 2 Classifier:** Dual-Stream EfficientNetB3")
st.sidebar.markdown("- **Input Resolution:** 300 × 300")
st.sidebar.markdown("- **Explainability:** Grad-CAM + Pathology Focus Box")


# -------------------------------------------------------------
# PAGE 1: HOME / OVERVIEW
# -------------------------------------------------------------
if page == "Home / Overview":
    st.title("Anatomy-Guided Bone Fracture Detection and Classification")
    st.markdown("### High-Precision Multi-Region X-Ray Analysis via YOLOv8 and EfficientNetB3")

    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown("""
        #### Clinical Background & Challenge
        Missed bone fractures account for **up to 80% of emergency room diagnostic errors**. Radiographs often contain irrelevant background soft tissue, patient posture variations, radiological markers, and high-noise borders.

        Standard deep learning approaches apply a CNN directly to the full raw X-ray. However, global feature maps are easily confused by non-pathological background artifacts, and subtle hairline fractures can be downsampled into oblivion.

        #### Two-Stage Solution Architecture
        This research system introduces an **anatomy-guided two-stage pipeline**:
        1. **Stage 1 (Anatomical Localization):** A lightweight YOLO object detector isolates the region of interest (Arm, Leg, Hand, Shoulder, Spine).
        2. **Symmetric Padded Cropping:** The detected region is extracted with **12% boundary padding** to preserve cortical bone context without cropping out peripheral fracture lines.
        3. **Stage 2 (Pathology Classification):** An ImageNet-pretrained **EfficientNetB3** model processes the focused crop at 300 × 300 resolution with ImageNet normalization.
        4. **Explainability & Safeguards:** Integrated **Grad-CAM** generates class activation heatmaps, coupled with an automated **full-image fallback mechanism** if localization confidence drops below threshold.
        """)

    with col2:
        st.markdown("#### System Pipeline Flowchart")
        st.code("""
INPUT X-RAY RADIOGRAPH
        │
        ▼
Preprocessing & Aspect Padding
        │
        ▼
STAGE 1: YOLO Localization
(Arm, Leg, Hand, Shoulder, Spine)
        │
        ├──> Confidence < 0.35? ──> [FULL-IMAGE FALLBACK]
        │
        ▼ (Confidence >= 0.35)
12% Padded Bounding Box Crop
        │
        ▼
Resizing to 300 × 300
        │
        ▼
STAGE 2: EfficientNetB3 Classifier
        │
        ▼
Optimal Validation Thresholding
        │
        ▼
Grad-CAM Heatmap & Overlay
        │
        ▼
Final Diagnosis & Clinical Warning
        """, language="text")

    st.markdown("---")
    st.subheader("Dataset & Hardware Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Radiographs", "3,156")
    c2.metric("Target Pathologies", "Fractured vs Normal")
    c3.metric("Anatomical Regions", "5 Classes")
    c4.metric("Inference Engine", "PyTorch + CUDA")


# -------------------------------------------------------------
# PAGE 2: FRACTURE DETECTION
# -------------------------------------------------------------
elif page == "Fracture Detection":
    st.title("Interactive Radiograph Fracture Analysis")
    st.markdown("Upload a patient X-ray image (JPG, JPEG, or PNG) to execute the complete two-stage diagnostic pipeline.")

    uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "jpeg", "png"])

    # Example image selector
    use_sample = st.checkbox("Or select an example radiograph from the dataset")
    sample_path = None
    if use_sample:
        catalog_path = "data/splits/dataset_catalog.csv"
        if os.path.exists(catalog_path):
            cat_df = pd.read_csv(catalog_path)
            sample_choices = [
                f"{row['fracture_label']} - {row['filename']}"
                for _, row in cat_df.sample(n=10, random_state=123).iterrows()
            ]
            selected_choice = st.selectbox("Select example:", sample_choices)
            selected_fname = selected_choice.split(" - ")[-1]
            sample_path = cat_df[cat_df["filename"] == selected_fname]["filepath"].values[0]

    input_image_src = uploaded_file or sample_path

    if input_image_src is not None:
        try:
            pipeline = load_pipeline()
            with st.spinner("Processing radiograph through YOLO localization and EfficientNetB3 classifier..."):
                results = pipeline.predict(input_image_src)

            st.markdown("---")
            st.subheader("Diagnostic Assessment")

            # Main Result Banner
            b1, b2, b3 = st.columns([2, 1, 1])
            with b1:
                if results["is_fractured"]:
                    st.markdown(f'<div class="status-fractured">⚠️ FRACTURE DETECTED (Confidence: {results["model_confidence"]*100:.1f}%)</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="status-normal">✅ NO FRACTURE DETECTED (Confidence: {results["model_confidence"]*100:.1f}%)</div>', unsafe_allow_html=True)

            with b2:
                st.metric("Fracture Probability", f"{results['fracture_probability']*100:.1f}%")
            with b3:
                st.metric("Decision Threshold", f"{results['decision_threshold']:.2f}")

            # Warnings & Safeguards
            if results["is_fallback"]:
                st.markdown(f'<div class="warning-box">⚠️ <b>Detector Fallback Triggered:</b> {results["fallback_warning"]} Full image was automatically evaluated.</div>', unsafe_allow_html=True)

            if results["uncertainty_warning"]:
                st.markdown(f'<div class="warning-box">⚡ {results["uncertainty_warning"]}</div>', unsafe_allow_html=True)

            st.markdown("---")
            st.subheader("5-Stage Visual Radiograph Analytics")

            # Visual display: 5 panels
            v1, v2, v3, v4, v5 = st.columns(5)
            with v1:
                st.markdown("**1. Original Radiograph**")
                st.image(results["original_image"], use_container_width=True)

            with v2:
                st.markdown("**2. Bone ROI Localization**")
                region_status = "Bone Structure" if not results["is_fallback"] else "Full Image (Fallback)"
                st.caption(f"Status: **{region_status}** | Conf: **{results['detector_confidence']*100:.1f}%**")
                st.image(results["annotated_detection_image"], use_container_width=True)

            with v3:
                st.markdown("**3. Padded Bone Crop**")
                st.caption("12% boundary padding (300 × 300)")
                st.image(results["cropped_image"], use_container_width=True)

            with v4:
                st.markdown("**4. Grad-CAM Overlay**")
                st.caption("Cortical activation saliency heatmap")
                st.image(results["gradcam_overlay"], use_container_width=True)

            with v5:
                st.markdown("**5. Pathology Focus ROI**")
                focus_box_label = str(results.get("focus_box", "(N/A)"))
                crop_box_label = str(results.get("crop_focus_box", "(N/A)"))
                peak_focus_pct = results.get("peak_focus_intensity", 0.0) * 100
                st.caption(f"Inside Step 2: **{crop_box_label}** ({peak_focus_pct:.0f}%)")
                st.image(results.get("focused_image", results["gradcam_overlay"]), use_container_width=True)

            # Metadata Table
            with st.expander("Detailed Diagnostic Metadata"):
                meta_dict = {
                    "Pathology Prediction": results["prediction_label"],
                    "Fracture Probability": f"{results['fracture_probability']*100:.2f}%",
                    "Non-Fracture Probability": f"{results['non_fracture_probability']*100:.2f}%",
                    "Decision Threshold": results["decision_threshold"],
                    "Bone ROI Localization": region_status,
                    "Detector Confidence": f"{results['detector_confidence']*100:.2f}%",
                    "Step 2 Bone ROI Bounding Box": str(results["bounding_box"]),
                    "Step 5 Focus Box (Inside Step 2 Box)": str(results.get("focus_box", "N/A")),
                    "Step 5 Crop Focus Box": str(results.get("crop_focus_box", "N/A")),
                    "Peak Focus Activation Intensity": f"{results.get('peak_focus_intensity', 0.0)*100:.1f}%",
                    "Diagnostic Severity Score": f"{results.get('severity_score', 0.0):.1f} / 100 ({results.get('severity_level', 'NORMAL')})",
                    "Fallback Safeguard Engaged": results["is_fallback"]
                }
                st.table(pd.DataFrame(list(meta_dict.items()), columns=["Diagnostic Metric", "Value"]))

            # Generate & Download AI Diagnostic PDF Report
            try:
                from src.evaluation.pdf_report_generator import generate_diagnostic_pdf_report
                img_src_name = getattr(uploaded_file, "name", str(sample_path or "Uploaded_Radiograph.png"))
                report_save_path = "results/reports/latest_web_report.pdf"
                pdf_file_path = generate_diagnostic_pdf_report(
                    result=results,
                    image_source=img_src_name,
                    output_path=report_save_path
                )
                with open(pdf_file_path, "rb") as pdf_f:
                    pdf_data = pdf_f.read()

                st.markdown("---")
                st.subheader("Diagnostic Report Export")
                r_col1, r_col2 = st.columns([1, 2])
                with r_col1:
                    st.download_button(
                        label="📄 Download AI Diagnostic Report (PDF)",
                        data=pdf_data,
                        file_name=f"AI_Bone_Fracture_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
                with r_col2:
                    st.info(f"Report compiled. Local copy: `{pdf_file_path}`")
            except Exception as pdf_ex:
                st.warning(f"Could not compile PDF report: {pdf_ex}")

        except Exception as e:
            st.error(f"Error during image processing: {e}")


# -------------------------------------------------------------
# PAGE 3: MODEL PERFORMANCE
# -------------------------------------------------------------
elif page == "Model Performance":
    st.title("Untouched Test Set Evaluation Metrics")
    st.markdown("All diagnostic metrics below are strictly computed on the held-out test set partition (463 images).")

    metrics_file = "results/metrics/test_evaluation_report.json"
    if not os.path.exists(metrics_file):
        metrics_file = "results/metrics/baseline_full_image_metrics.json"

    metrics_data = None
    if os.path.exists(metrics_file):
        with open(metrics_file, 'r', encoding='utf-8') as f:
            metrics_data = json.load(f)

    if metrics_data:
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Accuracy", f"{metrics_data.get('accuracy', 0)*100:.1f}%")
        m2.metric("Fracture Recall", f"{metrics_data.get('fracture_recall_sensitivity', 0)*100:.1f}%")
        m3.metric("Fracture Precision", f"{metrics_data.get('fracture_precision', 0)*100:.1f}%")
        m4.metric("Fracture F1", f"{metrics_data.get('fracture_f1', 0):.3f}")
        m5.metric("ROC-AUC", f"{metrics_data.get('roc_auc', 0):.3f}")
        m6.metric("PR-AUC", f"{metrics_data.get('pr_auc', 0):.3f}")

        st.markdown("---")
        st.subheader("Diagnostic Curves & Confusion Matrix")

        c1, c2, c3 = st.columns(3)
        cm_path = "results/plots/baseline_full_image_confusion_matrix.png"
        roc_path = "results/plots/baseline_full_image_roc_curve.png"
        pr_path = "results/plots/baseline_full_image_pr_curve.png"

        with c1:
            if os.path.exists(cm_path):
                st.markdown("**Confusion Matrix Heatmap**")
                st.image(cm_path, use_container_width=True)
        with c2:
            if os.path.exists(roc_path):
                st.markdown("**ROC Diagnostic Curve**")
                st.image(roc_path, use_container_width=True)
        with c3:
            if os.path.exists(pr_path):
                st.markdown("**Precision-Recall Curve**")
                st.image(pr_path, use_container_width=True)

        st.markdown("---")
        st.subheader("Training Convergence History")
        hist_path = "results/plots/baseline_full_image_history.png"
        if os.path.exists(hist_path):
            st.image(hist_path, use_container_width=True)
    else:
        st.info("No saved test evaluation metrics found. Run the training and evaluation pipeline to generate metrics.")


# -------------------------------------------------------------
# PAGE 4: EXPERIMENT COMPARISON
# -------------------------------------------------------------
elif page == "Experiment Comparison":
    st.title("Ablation Studies & Model Benchmarks")
    st.markdown("Comparative performance of baseline, anatomy-guided cropping, and class-imbalance mitigation strategies.")

    comp_csv = "results/metrics/experiment_comparison.csv"
    if os.path.exists(comp_csv):
        df_comp = pd.read_csv(comp_csv)
        st.dataframe(df_comp, use_container_width=True)

        comp_plot = "results/plots/experiment_comparison.png"
        if os.path.exists(comp_plot):
            st.markdown("---")
            st.subheader("Comparative Benchmark Graph")
            st.image(comp_plot, use_container_width=True)
    else:
        st.info("Comparison records not yet generated. Complete the experimental runs to populate this table.")


# -------------------------------------------------------------
# PAGE 5: ABOUT / LIMITATIONS
# -------------------------------------------------------------
elif page == "About & Limitations":
    st.title("Methodology, Limitations & Medical Disclaimer")

    st.markdown("""
    ### Model Architecture & Technical Details
    - **Stage 1 (Anatomical Region Localization):**
      Employs a lightweight **YOLOv8n** model trained to recognize the five canonical skeletal regions: Arm, Leg, Hand, Shoulder, and Spine. If localization confidence is `< 0.35`, the system falls back to evaluating the entire radiograph to prevent missed fractures.
    - **Stage 2 (Pathology Classification):**
      Utilizes **EfficientNetB3** pretrained on ImageNet. A multi-phase transfer learning schedule freezes lower low-level filter representations while fine-tuning high-level MBConv blocks.
    - **Class Imbalance Mitigation:**
      Given the **3.66:1 class imbalance** between non-fractured and fractured radiographs, standard binary cross-entropy produces severe false-negative bias. The pipeline incorporates **Class-Weighted Cross-Entropy (pos_weight = 3.55)**, **Focal Loss (gamma=2.0)**, and **Optimal Validation-Set Decision Thresholding**.
    - **Explainability (Grad-CAM):**
      Grad-CAM calculates gradient-weighted activation maps from the final convolutional feature layer, illustrating whether the network is inspecting cortical discontinuity versus textual radiological annotations or plate artifacts.

    ### Current System Limitations
    1. **Pediatric Fractures:** Subtle greenstick and buckle (torus) fractures in pediatric cohorts present unique radiographic appearances that require domain-specific pediatric datasets.
    2. **Multi-Fracture Cases:** The classifier outputs an overall binary pathology probability rather than instance-segmenting individual fracture fragments.
    3. **Radiographic Artifacts:** Hardware fixation plates, surgical screws, and high-density lead markers can occasionally induce localized Grad-CAM hotspots.

    """)

    st.markdown("""
    <div class="disclaimer-box">
    <b>⚠️ MANDATORY MEDICAL RESEARCH DISCLAIMER:</b><br>
    This application is an academic research and educational prototype. It is not a clinically validated diagnostic device and must not be used as a substitute for assessment, triage, or clinical diagnosis by a qualified radiologist or medical professional. Never make clinical treatment decisions based solely on the outputs of this software.
    </div>
    """, unsafe_allow_html=True)
