"""
Unit Tests for AI Diagnostic PDF Report Generator.
"""

import os
import unittest
from PIL import Image
import pypdf

from src.evaluation.pdf_report_generator import generate_diagnostic_pdf_report


class TestReportGenerator(unittest.TestCase):
    def setUp(self):
        self.output_pdf = "results/reports/test_unit_report.pdf"
        self.sample_img = Image.new('RGB', (600, 800), color=(120, 120, 120))
        self.sample_result = {
            "prediction_label": "FRACTURED",
            "is_fractured": True,
            "fracture_probability": 0.8523,
            "non_fracture_probability": 0.1477,
            "decision_threshold": 0.45,
            "model_confidence": 0.9850,
            "uncertainty_warning": "",
            "severity_score": 92.4,
            "severity_level": "SEVERE",
            "mean_activation": 24.5,
            "high_activation_area": 18.2,
            "score_breakdown": {
                "model_confidence": 59.1,
                "activation_spread": 18.3,
                "peak_intensity": 15.0,
                "total": 92.4
            },
            "region_name": "Bone Structure",
            "detector_confidence": 0.8850,
            "bounding_box": (100, 100, 450, 700),
            "is_fallback": False,
            "focus_box": (180, 220, 260, 310),
            "peak_focus_intensity": 1.0,
            "original_image": self.sample_img,
            "annotated_detection_image": self.sample_img,
            "cropped_image": Image.new('RGB', (300, 300), color=(140, 140, 140)),
            "gradcam_overlay": Image.new('RGB', (300, 300), color=(180, 50, 50)),
            "focused_image": Image.new('RGB', (300, 300), color=(200, 30, 30))
        }

    def test_pdf_generation_structure_and_pages(self):
        pdf_path = generate_diagnostic_pdf_report(
            result=self.sample_result,
            image_source="patient_wrist_xray.png",
            output_path=self.output_pdf
        )
        self.assertTrue(os.path.exists(pdf_path))
        self.assertGreater(os.path.getsize(pdf_path), 5000)

        # Verify page count is strictly 2 pages
        reader = pypdf.PdfReader(pdf_path)
        self.assertEqual(len(reader.pages), 2)

        # Verify key sections on page 1
        page1_text = reader.pages[0].extract_text()
        self.assertIn("AI-Assisted Bone Fracture Detection Report", page1_text)
        self.assertIn("FRACTURE DETECTED", page1_text)
        self.assertIn("Scan Information", page1_text)
        self.assertIn("AI-Generated Summary", page1_text)
        self.assertIn("Imaging Analysis", page1_text)

        # Verify key sections on page 2
        page2_text = reader.pages[1].extract_text()
        self.assertIn("Diagnostic Findings", page2_text)
        self.assertIn("Grad-CAM & Saliency Metrics", page2_text)
        self.assertIn("Severity Score Breakdown", page2_text)
        self.assertIn("Recommended Actions", page2_text)
        self.assertIn("DISCLAIMER", page2_text)

        # Verify omitted sections
        self.assertNotIn("Clinical Fracture Assessment", page2_text)
        self.assertNotIn("Limitations", page2_text)


if __name__ == '__main__':
    unittest.main()
