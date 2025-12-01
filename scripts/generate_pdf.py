"""Convert markdown to PDF with professional styling"""
import markdown2
from weasyprint import HTML, CSS
from pathlib import Path

# Read markdown file
md_file = Path("presentations/COMPREHENSIVE_SPEECH_QUESTIS_20MIN.md")
with open(md_file, 'r', encoding='utf-8') as f:
    md_content = f.read()

# Convert markdown to HTML
html_content = markdown2.markdown(md_content, extras=['tables', 'fenced-code-blocks'])

# Add CSS styling
css_style = """
    <style>
        body {
            font-family: 'Georgia', serif;
            line-height: 1.6;
            max-width: 800px;
            margin: 40px auto;
            padding: 20px;
            color: #333;
        }
        h1 {
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
            margin-top: 40px;
        }
        h2 {
            color: #34495e;
            border-bottom: 2px solid #95a5a6;
            padding-bottom: 8px;
            margin-top: 30px;
        }
        h3 {
            color: #7f8c8d;
            margin-top: 20px;
        }
        blockquote {
            border-left: 4px solid #3498db;
            padding-left: 20px;
            margin: 20px 0;
            font-style: italic;
            color: #555;
            background-color: #f8f9fa;
            padding: 15px 20px;
        }
        code {
            background-color: #f4f4f4;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
        }
        ul, ol {
            margin-left: 30px;
        }
        strong {
            color: #2c3e50;
        }
        hr {
            border: none;
            border-top: 2px solid #ecf0f1;
            margin: 40px 0;
        }
        @page {
            margin: 2.5cm;
            @bottom-right {
                content: "Page " counter(page);
            }
        }
    </style>
"""

# Combine HTML
full_html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    {css_style}
</head>
<body>
    {html_content}
</body>
</html>
"""

# Generate PDF
output_file = "presentations/COMPREHENSIVE_SPEECH_QUESTIS_20MIN.pdf"
HTML(string=full_html).write_pdf(output_file)

print(f"✅ PDF generated successfully: {output_file}")
print(f"📄 File size: {Path(output_file).stat().st_size / 1024:.1f} KB")
