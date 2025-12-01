"""Convert markdown to styled HTML that can be printed to PDF"""
import markdown2
from pathlib import Path

# Read markdown file
md_file = Path("presentations/COMPREHENSIVE_SPEECH_QUESTIS_20MIN.md")
with open(md_file, 'r', encoding='utf-8') as f:
    md_content = f.read()

# Convert markdown to HTML
html_content = markdown2.markdown(md_content, extras=['tables', 'fenced-code-blocks', 'header-ids'])

# Professional CSS styling optimized for PDF printing
css_style = """
<style>
    @page {
        size: A4;
        margin: 2cm;
    }
    
    @media print {
        body {
            font-size: 11pt;
        }
        h1 {
            page-break-before: always;
        }
        h1:first-of-type {
            page-break-before: avoid;
        }
        h2, h3 {
            page-break-after: avoid;
        }
        blockquote {
            page-break-inside: avoid;
        }
    }
    
    body {
        font-family: 'Segoe UI', 'Helvetica Neue', Arial, sans-serif;
        line-height: 1.6;
        max-width: 900px;
        margin: 0 auto;
        padding: 40px;
        color: #2c3e50;
        background: white;
    }
    
    h1 {
        color: #1a1a2e;
        font-size: 28pt;
        border-bottom: 4px solid #0f4c75;
        padding-bottom: 12px;
        margin-top: 50px;
        margin-bottom: 25px;
        font-weight: 600;
    }
    
    h1:first-of-type {
        margin-top: 0;
        font-size: 32pt;
        text-align: center;
        border-bottom: none;
    }
    
    h2 {
        color: #0f4c75;
        font-size: 20pt;
        border-bottom: 2px solid #3282b8;
        padding-bottom: 8px;
        margin-top: 35px;
        margin-bottom: 20px;
        font-weight: 600;
    }
    
    h3 {
        color: #3282b8;
        font-size: 16pt;
        margin-top: 25px;
        margin-bottom: 15px;
        font-weight: 600;
    }
    
    h4 {
        color: #555;
        font-size: 13pt;
        margin-top: 20px;
        margin-bottom: 10px;
        font-weight: 600;
    }
    
    p {
        margin: 12px 0;
        text-align: justify;
    }
    
    blockquote {
        border-left: 5px solid #3282b8;
        padding: 15px 25px;
        margin: 25px 0;
        background: linear-gradient(to right, #f0f7fa, white);
        font-style: italic;
        color: #2c3e50;
        border-radius: 0 4px 4px 0;
    }
    
    blockquote p {
        margin: 8px 0;
    }
    
    strong {
        color: #1a1a2e;
        font-weight: 700;
    }
    
    em {
        color: #555;
    }
    
    code {
        background-color: #f4f4f4;
        padding: 3px 8px;
        border-radius: 4px;
        font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
        font-size: 9.5pt;
        color: #c7254e;
        border: 1px solid #e1e1e1;
    }
    
    pre {
        background-color: #f8f8f8;
        padding: 15px;
        border-radius: 6px;
        border-left: 4px solid #0f4c75;
        overflow-x: auto;
        margin: 20px 0;
    }
    
    pre code {
        background: none;
        padding: 0;
        border: none;
        color: #333;
    }
    
    ul, ol {
        margin: 15px 0;
        padding-left: 40px;
    }
    
    li {
        margin: 8px 0;
        line-height: 1.7;
    }
    
    hr {
        border: none;
        border-top: 2px solid #e1e8ed;
        margin: 50px 0;
    }
    
    table {
        width: 100%;
        border-collapse: collapse;
        margin: 25px 0;
    }
    
    th, td {
        border: 1px solid #ddd;
        padding: 12px;
        text-align: left;
    }
    
    th {
        background-color: #0f4c75;
        color: white;
        font-weight: 600;
    }
    
    tr:nth-child(even) {
        background-color: #f9f9f9;
    }
    
    .metadata {
        text-align: center;
        color: #7f8c8d;
        margin-bottom: 40px;
        font-size: 10pt;
    }
    
    /* Special formatting for checkboxes */
    li:has(input[type="checkbox"]) {
        list-style-type: none;
        margin-left: -20px;
    }
</style>
"""

# Combine HTML with metadata
full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Comprehensive Speech - QUEST-IS 2025</title>
    {css_style}
</head>
<body>
    <div class="metadata">
        Machine Learning-Driven Quantum Hacking of CHSH-Based QKD<br>
        QUEST-IS 2025 Conference | December 2025<br>
        Hubert Kołcz | Warsaw University of Technology
    </div>
    {html_content}
    <hr>
    <div class="metadata">
        Generated: December 1, 2025 | GitHub: hubertkolcz/NoiseVsRandomness
    </div>
</body>
</html>
"""

# Save HTML
output_file = "presentations/COMPREHENSIVE_SPEECH_QUESTIS_20MIN.html"
with open(output_file, 'w', encoding='utf-8') as f:
    f.write(full_html)

print(f"✅ HTML generated successfully: {output_file}")
print(f"📄 File size: {Path(output_file).stat().st_size / 1024:.1f} KB")
print(f"\n📌 To create PDF:")
print(f"   1. Open {output_file} in Chrome/Edge")
print(f"   2. Press Ctrl+P (Print)")
print(f"   3. Select 'Save as PDF'")
print(f"   4. Set margins to 'Default' or 'Minimum'")
print(f"   5. Enable 'Background graphics'")
print(f"   6. Save as COMPREHENSIVE_SPEECH_QUESTIS_20MIN.pdf")
