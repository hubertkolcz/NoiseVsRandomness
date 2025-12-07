"""
Generate PDF with speaking notes - one page per slide
"""
import re
import subprocess
import sys

# First, try to import reportlab, if not available, install it
try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
    from reportlab.lib.enums import TA_LEFT, TA_CENTER
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
except ImportError:
    print("Installing reportlab...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "reportlab"])
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
    from reportlab.lib.enums import TA_LEFT, TA_CENTER
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

def clean_text(text):
    """Clean text and replace special characters that reportlab can't handle."""
    # Replace special Unicode characters with ASCII equivalents
    replacements = {
        '⟨': '<',
        '⟩': '>',
        '√': 'sqrt',
        '≤': '<=',
        '≥': '>=',
        '≠': '!=',
        '≈': '~=',
        '×': 'x',
        '→': '->',
        '←': '<-',
        '↔': '<->',
        '⊕': '(+)',
        '∞': 'infinity',
        '±': '+/-',
        '°': ' degrees',
        '²': '^2',
        '³': '^3',
        '⁻': '^-',
        '⁰': '^0',
        '¹': '^1',
        '⁴': '^4',
        '⁵': '^5',
        '⁶': '^6',
        '⁷': '^7',
        '⁸': '^8',
        '⁹': '^9',
        '₀': '_0',
        '₁': '_1',
        '₂': '_2',
        '₃': '_3',
        '₄': '_4',
        '₅': '_5',
        '₆': '_6',
        '₇': '_7',
        '₈': '_8',
        '₉': '_9',
        'ρ': 'rho',
        'χ²': 'chi^2',
        'χ': 'chi',
        'ł': 'l',
        # Keep common symbols
        '•': '* ',
        '▸': '> ',
        '✓': '[OK]',
        '✅': '[OK]',
        '❌': '[X]',
        '⚠': '[!]',
        '⚠️': '[!]',
        '🎯': '[TARGET]',
        '🔍': '[SEARCH]',
        '💡': '[IDEA]',
        '📊': '[CHART]',
    }
    
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    return text

def parse_speaking_notes(filepath):
    """Parse the markdown speaking notes file and extract slide information."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split by slide markers
    slides = []
    
    # Match slide headers like: ## **SLIDE 1: Title (0:00-0:50) [50s]**
    # Simple pattern that finds slide headers
    slide_pattern = r'##\s+\*\*SLIDE\s+(\d+):\s+([^(]+?)\s*\(([^)]+)\)\s+\[([^\]]+)\](?:\s*-[^\*]+)?\*\*'
    
    # Find all slide headers and their positions
    slide_headers = list(re.finditer(slide_pattern, content))
    
    # Extract content between slide headers
    for i, match in enumerate(slide_headers):
        slide_num = match.group(1)
        title = match.group(2).strip()
        time_range = match.group(3).strip()
        duration = match.group(4).strip()
        
        # Get content from this match to the next match (or end of file)
        start_pos = match.end()
        if i < len(slide_headers) - 1:
            end_pos = slide_headers[i + 1].start()
        else:
            # Last slide - find the backup section or end
            backup_match = re.search(r'\n#\s+Backup', content[start_pos:])
            if backup_match:
                end_pos = start_pos + backup_match.start()
            else:
                end_pos = len(content)
        
        notes_content = content[start_pos:end_pos].strip()
        
        # Clean up bullet points
        notes_lines = []
        for line in notes_content.split('\n'):
            line = line.strip()
            if line and not line.startswith('---'):
                # Remove markdown bold markers
                line = line.replace('**', '')
                notes_lines.append(line)
        
        slides.append({
            'number': slide_num,
            'title': clean_text(title),
            'time_range': time_range,
            'duration': duration,
            'notes': clean_text('\n'.join(notes_lines))
        })
    
    return slides

def create_pdf(slides, output_path):
    """Create PDF with one page per slide."""
    
    # Create document
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        rightMargin=0.5*inch,
        leftMargin=0.5*inch,
        topMargin=0.5*inch,
        bottomMargin=0.5*inch
    )
    
    # Container for the 'Flowable' objects
    elements = []
    
    # Define styles
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=18,
        textColor='#000080',
        spaceAfter=12,
        alignment=TA_CENTER
    )
    
    slide_title_style = ParagraphStyle(
        'SlideTitle',
        parent=styles['Heading2'],
        fontSize=14,
        textColor='#000080',
        spaceAfter=6,
        spaceBefore=0
    )
    
    timing_style = ParagraphStyle(
        'Timing',
        parent=styles['Normal'],
        fontSize=11,
        textColor='#666666',
        spaceAfter=12,
        alignment=TA_CENTER
    )
    
    notes_style = ParagraphStyle(
        'Notes',
        parent=styles['Normal'],
        fontSize=10,
        leading=14,
        leftIndent=0,
        spaceAfter=6
    )
    
    bullet_style = ParagraphStyle(
        'Bullet',
        parent=styles['Normal'],
        fontSize=10,
        leading=14,
        leftIndent=20,
        firstLineIndent=0,
        spaceAfter=4
    )
    
    # Add title page
    elements.append(Spacer(1, 1*inch))
    elements.append(Paragraph("ML-Driven Quantum Hacking of CHSH-Based QKD", title_style))
    elements.append(Spacer(1, 0.2*inch))
    elements.append(Paragraph("Speaking Notes for 20-Minute Presentation", timing_style))
    elements.append(Spacer(1, 0.2*inch))
    elements.append(Paragraph("19 Slides | Target Duration: 19-20 minutes", timing_style))
    elements.append(PageBreak())
    
    # Add each slide on its own page
    for slide in slides:
        # Slide number and title
        slide_header = f"SLIDE {slide['number']}: {slide['title']}"
        elements.append(Paragraph(slide_header, slide_title_style))
        
        # Timing information
        timing_text = f"Time: {slide['time_range']} | Duration: {slide['duration']}"
        elements.append(Paragraph(timing_text, timing_style))
        
        elements.append(Spacer(1, 0.15*inch))
        
        # Add notes
        notes_lines = slide['notes'].split('\n')
        for line in notes_lines:
            if not line.strip():
                continue
            
            # Convert markdown-style bullets to proper formatting
            if line.startswith('- ') or line.startswith('* '):
                # Bullet point
                text = line[2:].strip()
                # Escape special characters for reportlab
                text = text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                elements.append(Paragraph(f"• {text}", bullet_style))
            elif line.startswith('#'):
                # Sub-header
                text = line.lstrip('#').strip()
                text = text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                elements.append(Paragraph(f"<b>{text}</b>", notes_style))
            else:
                # Regular text
                text = line.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                elements.append(Paragraph(text, notes_style))
        
        # Page break after each slide (except the last one)
        elements.append(PageBreak())
    
    # Build PDF
    doc.build(elements)
    print(f"PDF created successfully: {output_path}")

def main():
    # Paths
    notes_file = r'c:\Users\cp\Documents\GitHub\NoiseVsRandomness\presentations\SPEAKING_NOTES_17SLIDES_FINAL.md'
    output_pdf = r'c:\Users\cp\Documents\GitHub\NoiseVsRandomness\presentations\SPEAKING_NOTES_17SLIDES_FINAL.pdf'
    
    print("Parsing speaking notes...")
    slides = parse_speaking_notes(notes_file)
    print(f"Found {len(slides)} slides")
    
    print("Creating PDF...")
    create_pdf(slides, output_pdf)
    print("Done!")

if __name__ == '__main__':
    main()
