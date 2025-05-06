import os
from pathlib import Path
from backend.export import ReportExporter
from datetime import datetime

def test_export_functionality():
    # Initialize the exporter
    exporter = ReportExporter()
    
    # Create a realistic research report
    report = {
        "title": "The Impact of Artificial Intelligence on Healthcare",
        "content": """# The Impact of Artificial Intelligence on Healthcare

## Introduction
Artificial Intelligence (AI) has revolutionized numerous industries, with healthcare being one of the most significant beneficiaries. This report explores the current applications, benefits, and challenges of AI in healthcare.

## Current Applications
1. **Medical Imaging Analysis**
   - AI algorithms can detect abnormalities in X-rays, MRIs, and CT scans with high accuracy
   - Reduces diagnostic time and improves early detection rates

2. **Drug Discovery**
   - AI accelerates the drug development process
   - Predicts potential drug interactions and side effects

3. **Personalized Medicine**
   - AI analyzes patient data to create customized treatment plans
   - Improves treatment outcomes and reduces adverse effects

## Benefits
- Improved diagnostic accuracy
- Reduced healthcare costs
- Enhanced patient outcomes
- Streamlined administrative processes

## Challenges
- Data privacy concerns
- Integration with existing systems
- Need for regulatory frameworks
- Ethical considerations

## Conclusion
While AI in healthcare presents numerous opportunities, careful consideration must be given to ethical, privacy, and regulatory aspects to ensure responsible implementation.""",
        "citations": [
            {
                "title": "Artificial Intelligence in Healthcare: Past, Present and Future",
                "authors": "Jiang, F., Jiang, Y., Zhi, H., Dong, Y., Li, H., Ma, S., ... & Wang, Y.",
                "url": "https://doi.org/10.1016/j.smim.2017.04.001"
            },
            {
                "title": "Machine Learning in Medicine",
                "authors": "Deo, R. C.",
                "url": "https://doi.org/10.1056/NEJMra1814259"
            },
            {
                "title": "The Future of AI in Healthcare",
                "authors": "Topol, E. J.",
                "url": "https://doi.org/10.1038/s41591-019-0422-6"
            }
        ]
    }

    # Test PDF export
    print("\nTesting PDF Export...")
    pdf_path = exporter.export_pdf(report["content"], report["citations"])
    print(f"PDF exported to: {pdf_path}")
    
    # Test DOCX export
    print("\nTesting DOCX Export...")
    docx_path = exporter.export_docx(report, "ai_healthcare_report")
    print(f"DOCX exported to: {docx_path}")
    
    # Test HTML export
    print("\nTesting HTML Export...")
    html_path = exporter.export_html(report, "ai_healthcare_report")
    print(f"HTML exported to: {html_path}")
    
    # Verify files exist
    print("\nVerifying exported files...")
    for path in [pdf_path, docx_path, html_path]:
        if os.path.exists(path):
            print(f"✓ {path} exists")
        else:
            print(f"✗ {path} does not exist")

if __name__ == "__main__":
    print("Starting export functionality test...")
    test_export_functionality()
    print("\nTest completed!") 