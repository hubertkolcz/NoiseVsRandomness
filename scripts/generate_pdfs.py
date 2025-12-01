"""Generate PDF files from presentation materials."""
import os
import sys

def generate_speech_pdf():
    """Generate PDF from speech markdown."""
    try:
        import markdown2
    except ImportError:
        print("Installing markdown2...")
        os.system(f"{sys.executable} -m pip install markdown2")
        import markdown2
    
    # Read markdown
    with open('presentations/COMPREHENSIVE_SPEECH_QUESTIS_20MIN.md', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Convert to HTML
    html_content = markdown2.markdown(content, extras=['tables', 'fenced-code-blocks'])
    
    # Add styling
    styled_html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 40px;
            line-height: 1.6;
            max-width: 900px;
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            border-bottom: 2px solid #3498db;
            padding-bottom: 5px;
            margin-top: 30px;
        }}
        h3 {{
            color: #555;
            margin-top: 20px;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }}
        th {{
            background-color: #3498db;
            color: white;
        }}
        code {{
            background-color: #f4f4f4;
            padding: 2px 5px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
        }}
        pre {{
            background-color: #f4f4f4;
            padding: 10px;
            border-radius: 5px;
            overflow-x: auto;
        }}
        ul, ol {{
            margin: 10px 0;
        }}
        li {{
            margin: 5px 0;
        }}
    </style>
</head>
<body>
{html_content}
</body>
</html>"""
    
    # Save temporary HTML
    with open('presentations/temp_speech.html', 'w', encoding='utf-8') as f:
        f.write(styled_html)
    
    print("✓ Speech HTML generated: presentations/temp_speech.html")
    
    # Try to generate PDF
    try:
        import pdfkit
        pdfkit.from_file('presentations/temp_speech.html', 'presentations/COMPREHENSIVE_SPEECH_QUESTIS_20MIN.pdf')
        print("✓ Speech PDF generated: presentations/COMPREHENSIVE_SPEECH_QUESTIS_20MIN.pdf")
        return True
    except ImportError:
        print("⚠ pdfkit not available. Install wkhtmltopdf for PDF generation.")
        print("  You can use the HTML file or print from browser to PDF.")
        return False
    except Exception as e:
        print(f"⚠ PDF generation failed: {e}")
        print("  You can open temp_speech.html in a browser and print to PDF.")
        return False

def generate_presentation_pdf():
    """Generate PDF from presentation HTML."""
    try:
        import pdfkit
        pdfkit.from_file('presentations/presentation_20slides.html', 'presentations/presentation_20slides.pdf')
        print("✓ Presentation PDF generated: presentations/presentation_20slides.pdf")
        return True
    except ImportError:
        print("⚠ pdfkit not available for presentation PDF.")
        print("  Open presentation_20slides.html in browser and print to PDF.")
        return False
    except Exception as e:
        print(f"⚠ Presentation PDF generation failed: {e}")
        print("  Open presentation_20slides.html in browser and print to PDF.")
        return False

if __name__ == "__main__":
    print("\n" + "="*60)
    print("GENERATING PDF FILES FROM PRESENTATION MATERIALS")
    print("="*60 + "\n")
    
    # Generate speech PDF
    print("1. Processing speech markdown...")
    speech_success = generate_speech_pdf()
    
    print("\n2. Processing presentation HTML...")
    pres_success = generate_presentation_pdf()
    
    print("\n" + "="*60)
    print("SUMMARY:")
    print("="*60)
    print(f"Speech PDF: {'✓ Generated' if speech_success else '✗ Use HTML fallback'}")
    print(f"Presentation PDF: {'✓ Generated' if pres_success else '✗ Use browser print'}")
    print("\nAlternative: Open HTML files in browser and use Print > Save as PDF")
    print("="*60 + "\n")
