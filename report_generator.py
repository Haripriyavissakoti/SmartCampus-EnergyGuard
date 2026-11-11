"""
SmartCampus EnergyGuard - Academic Audit Report Generator
=========================================================
Generates a publication-grade PDF audit summary using ReportLab.
Includes executive KPIs, facility comparison, critical anomaly event log,
cost estimations, model evaluation metrics, and mandatory simulation disclaimers.
"""

import io
import os
import pandas as pd
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.pdfgen import canvas

class NumberedCanvas(canvas.Canvas):
    """Custom canvas that computes total pages dynamically and adds academic footer."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_footer(num_pages)
            super().showPage()
        super().save()

    def draw_footer(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748b"))
        footer_text = (
            f"SmartCampus EnergyGuard Academic Prototype  |  "
            f"Simulated Data Disclaimer: Not actual college measurements  |  "
            f"Page {self._pageNumber} of {page_count}"
        )
        self.drawCentredString(letter[0] / 2.0, 30, footer_text)
        self.restoreState()

def generate_pdf_report(
    df_filtered,
    tariff=8.50,
    selected_bldg="All Buildings",
    date_range_str="All Time",
    metrics_summary=None
):
    """
    Builds and returns a PDF document as bytes.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=48
    )

    styles = getSampleStyleSheet()
    
    # Custom Typography Styles
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#0f172a")
    )
    
    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=11,
        leading=14,
        textColor=colors.HexColor("#334155")
    )
    
    section_h1 = ParagraphStyle(
        "SectionH1",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=17,
        textColor=colors.HexColor("#1e293b"),
        spaceBefore=12,
        spaceAfter=6
    )
    
    body_style = ParagraphStyle(
        "BodyDark",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#334155")
    )

    body_bold = ParagraphStyle(
        "BodyBold",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#0f172a")
    )
    
    table_cell = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=10,
        textColor=colors.HexColor("#1e293b")
    )
    
    table_cell_bold = ParagraphStyle(
        "TableCellBold",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=10,
        textColor=colors.HexColor("#0f172a")
    )
    
    disclaimer_style = ParagraphStyle(
        "DisclaimerText",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#92400e")
    )

    story = []

    # 1. Header Section
    story.append(Paragraph("SmartCampus EnergyGuard", title_style))
    story.append(Paragraph(
        "Campus Electricity Consumption Monitoring & ML Anomaly Detection Audit Report",
        subtitle_style
    ))
    story.append(Spacer(1, 8))
    
    # 2. Mandatory Simulation Disclaimer Banner
    disclaimer_html = (
        "<b>LEGAL & ACADEMIC DISCLAIMER:</b> "
        "Simulated prototype data — not actual college electricity measurements. "
        "This system demonstrates machine learning anomaly detection methodologies for campus facilities "
        "and is not connected to physical live metering hardware."
    )
    disclaimer_table = Table(
        [[Paragraph(disclaimer_html, disclaimer_style)]],
        colWidths=[540]
    )
    disclaimer_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#fef3c7")),
        ('BORDER', (0, 0), (-1, -1), 1, colors.HexColor("#f59e0b")),
        ('PADDING', (0, 0), (-1, -1), 8),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(disclaimer_table)
    story.append(Spacer(1, 10))

    # 3. Audit Parameters Table
    gen_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    param_data = [
        [
            Paragraph("<b>Generated:</b> " + gen_time_str, body_style),
            Paragraph(f"<b>Facility Scope:</b> {selected_bldg}", body_style)
        ],
        [
            Paragraph(f"<b>Date Filter Range:</b> {date_range_str}", body_style),
            Paragraph(f"<b>Configured Tariff:</b> ₹{tariff:.2f} / kWh (Illustrative)", body_style)
        ]
    ]
    param_table = Table(param_data, colWidths=[270, 270])
    param_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#f1f5f9")),
        ('PADDING', (0, 0), (-1, -1), 6),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(param_table)
    story.append(Spacer(1, 12))

    # 4. Executive KPI Summary Cards
    total_records = len(df_filtered)
    total_energy = df_filtered["energy_usage_kwh"].sum() if total_records > 0 else 0.0
    avg_hourly = df_filtered["energy_usage_kwh"].mean() if total_records > 0 else 0.0
    total_anomalies = (df_filtered["status"] == "Anomaly").sum() if total_records > 0 else 0
    high_anomalies = (df_filtered["severity"] == "High").sum() if total_records > 0 else 0
    total_cost = total_energy * tariff
    
    # Potential excess calculation: max(0, actual - expected)
    if "expected_kwh" in df_filtered.columns and total_records > 0:
        excess_kwh = df_filtered.apply(lambda r: max(0.0, r["energy_usage_kwh"] - r["expected_kwh"]), axis=1).sum()
    else:
        excess_kwh = 0.0
    excess_cost = excess_kwh * tariff

    story.append(Paragraph("1. Executive Summary & Key Performance Indicators", section_h1))
    
    kpi_matrix = [
        [
            Paragraph("Total Energy Consumption", table_cell_bold),
            Paragraph("Total Flagged Anomalies", table_cell_bold),
            Paragraph("High Severity Alerts", table_cell_bold)
        ],
        [
            Paragraph(f"<b>{total_energy:,.1f}</b> kWh", ParagraphStyle("KPI1", parent=body_style, fontSize=11, textColor=colors.HexColor("#0284c7"))),
            Paragraph(f"<b>{total_anomalies:,}</b> records ({total_anomalies/max(1, total_records)*100:.1f}%)", ParagraphStyle("KPI2", parent=body_style, fontSize=11, textColor=colors.HexColor("#dc2626"))),
            Paragraph(f"<b>{high_anomalies:,}</b> critical", ParagraphStyle("KPI3", parent=body_style, fontSize=11, textColor=colors.HexColor("#b91c1c")))
        ],
        [
            Paragraph("Average Hourly Power", table_cell_bold),
            Paragraph("Total Estimated Cost", table_cell_bold),
            Paragraph("Potential Excess Cost", table_cell_bold)
        ],
        [
            Paragraph(f"<b>{avg_hourly:.2f}</b> kWh/h", ParagraphStyle("KPI4", parent=body_style, fontSize=11, textColor=colors.HexColor("#0f172a"))),
            Paragraph(f"<b>₹{total_cost:,.2f}</b>", ParagraphStyle("KPI5", parent=body_style, fontSize=11, textColor=colors.HexColor("#15803d"))),
            Paragraph(f"<b>₹{excess_cost:,.2f}</b> ({excess_kwh:,.1f} kWh)", ParagraphStyle("KPI6", parent=body_style, fontSize=11, textColor=colors.HexColor("#d97706")))
        ]
    ]
    kpi_table = Table(kpi_matrix, colWidths=[180, 180, 180])
    kpi_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#ffffff")),
        ('BORDER', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#f8fafc")),
        ('BACKGROUND', (0, 2), (-1, 2), colors.HexColor("#f8fafc")),
        ('PADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(kpi_table)
    story.append(Spacer(1, 12))

    # 5. Facility-Wise Electricity Consumption & Costs Table
    story.append(Paragraph("2. Facility Consumption & Cost Breakdown", section_h1))
    
    if total_records > 0:
        bldg_summary = []
        # Header
        bldg_header = [
            Paragraph("Facility Name", table_cell_bold),
            Paragraph("Total kWh", table_cell_bold),
            Paragraph("Avg kWh/h", table_cell_bold),
            Paragraph("Anomalies", table_cell_bold),
            Paragraph("High Sev.", table_cell_bold),
            Paragraph("Est. Cost (₹)", table_cell_bold),
            Paragraph("Potential Excess", table_cell_bold)
        ]
        bldg_summary.append(bldg_header)
        
        grouped = df_filtered.groupby("building")
        for bldg_name, group in grouped:
            b_total_kwh = group["energy_usage_kwh"].sum()
            b_avg_kwh = group["energy_usage_kwh"].mean()
            b_anom_cnt = (group["status"] == "Anomaly").sum()
            b_high_cnt = (group["severity"] == "High").sum()
            b_cost = b_total_kwh * tariff
            if "expected_kwh" in group.columns:
                b_excess_kwh = group.apply(lambda r: max(0.0, r["energy_usage_kwh"] - r["expected_kwh"]), axis=1).sum()
            else:
                b_excess_kwh = 0.0
                
            bldg_summary.append([
                Paragraph(bldg_name, table_cell),
                Paragraph(f"{b_total_kwh:,.1f}", table_cell),
                Paragraph(f"{b_avg_kwh:.2f}", table_cell),
                Paragraph(f"{b_anom_cnt:,}", table_cell),
                Paragraph(f"{b_high_cnt:,}", table_cell),
                Paragraph(f"₹{b_cost:,.2f}", table_cell),
                Paragraph(f"{b_excess_kwh:,.1f} kWh", table_cell),
            ])
            
        bldg_table = Table(bldg_summary, colWidths=[110, 70, 60, 60, 55, 95, 90])
        bldg_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#e2e8f0")),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ('PADDING', (0, 0), (-1, -1), 5),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
        ]))
        story.append(bldg_table)
    else:
        story.append(Paragraph("No records found matching current active filters.", body_style))
        
    story.append(Spacer(1, 14))

    # 6. Selected Anomaly Details & Recommendations
    story.append(Paragraph("3. Priority Flagged Anomalies & Recommended Actions", section_h1))
    
    anom_df = df_filtered[df_filtered["status"] == "Anomaly"]
    if len(anom_df) > 0:
        # Show top 6 highest severity or score anomalies
        top_anoms = anom_df.sort_values(by=["severity", "anomaly_score"], ascending=[True, False]).head(6)
        
        anom_table_data = [
            [
                Paragraph("Timestamp & Facility", table_cell_bold),
                Paragraph("Usage / Baseline", table_cell_bold),
                Paragraph("Severity", table_cell_bold),
                Paragraph("Rule-Based Diagnosis & Next Steps", table_cell_bold)
            ]
        ]
        
        for _, row in top_anoms.iterrows():
            ts_str = str(row["timestamp"])[:16]
            bldg = row["building"]
            kwh = row["energy_usage_kwh"]
            expected = row.get("expected_kwh", "N/A")
            expected_str = f"{expected:.1f}" if isinstance(expected, (int, float)) else str(expected)
            sev = row["severity"]
            explanation = row.get("explanation", "Unusual pattern detected.")
            action = row.get("action_recommendation", "Review sensor readings.")
            
            diag_text = f"<b>Hypothesis:</b> {explanation}<br/><b>Action:</b> {action}"
            
            anom_table_data.append([
                Paragraph(f"<b>{bldg}</b><br/>{ts_str}", table_cell),
                Paragraph(f"<b>{kwh} kWh</b><br/>(Base: {expected_str})", table_cell),
                Paragraph(f"<b>{sev}</b><br/>Score: {row['anomaly_score']:.3f}", table_cell),
                Paragraph(diag_text, table_cell)
            ])
            
        anom_table = Table(anom_table_data, colWidths=[110, 85, 75, 270])
        anom_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#fee2e2")),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#fca5a5")),
            ('PADDING', (0, 0), (-1, -1), 5),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#fff1f2")]),
        ]))
        story.append(anom_table)
    else:
        story.append(Paragraph("No anomalies detected in the selected filter range.", body_style))
        
    story.append(Spacer(1, 14))

    # 7. Model Evaluation & Statistical Governance
    story.append(Paragraph("4. Machine Learning Model Evaluation & Governance", section_h1))
    
    if metrics_summary and "precision" in metrics_summary:
        m = metrics_summary
        cm = m.get("confusion_matrix", {})
        eval_html = (
            f"<b>Algorithm:</b> Isolation Forest (Unsupervised) | <b>Estimators:</b> {m.get('n_estimators', 150)} | "
            f"<b>Contamination Factor:</b> {m.get('contamination_rate', 0.04):.2%}<br/>"
            f"<b>Evaluation against Injected Synthetic Ground Truth:</b><br/>"
            f"• Precision: <b>{m.get('precision', 0.0):.4f}</b>  |  Recall: <b>{m.get('recall', 0.0):.4f}</b>  |  "
            f"F1-Score: <b>{m.get('f1_score', 0.0):.4f}</b>  |  Accuracy: <b>{m.get('accuracy', 0.0):.4f}</b><br/>"
            f"• Confusion Matrix Counts: True Positives (TP) = {cm.get('true_positive', 0)}, "
            f"False Positives (FP) = {cm.get('false_positive', 0)}, "
            f"False Negatives (FN) = {cm.get('false_negative', 0)}, "
            f"True Negatives (TN) = {cm.get('true_negative', 0)}<br/>"
            f"<i>Academic Note: {m.get('evaluation_disclaimer', '')}</i>"
        )
    else:
        eval_html = (
            "<b>Algorithm:</b> Isolation Forest (Unsupervised Ensemble).<br/>"
            "Isolation Forest isolates anomalies by randomly partitioning feature dimensions. Points requiring fewer splits "
            "are classified as anomalous. Evaluated against synthetic labels for academic validation only."
        )
        
    eval_table = Table([[Paragraph(eval_html, body_style)]], colWidths=[540])
    eval_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
        ('BORDER', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ('PADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(eval_table)
    
    # Build Document
    doc.build(story, canvasmaker=NumberedCanvas)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes
