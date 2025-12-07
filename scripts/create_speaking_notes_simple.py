import re
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.enums import TA_LEFT

def clean_text(text):
    """Remove special characters that can't be rendered in PDF."""
    replacements = {
        '\u2190': '<-', '\u2192': '->', '\u2194': '<->',
        '\u2264': '<=', '\u2265': '>=', '\u2260': '!=',
        '\u00d7': 'x', '\u00f7': '/',
        '\u221a': 'sqrt', '\u221e': 'inf',
        '\u03c0': 'pi', '\u03b1': 'alpha', '\u03b2': 'beta',
        '\u27e8': '<', '\u27e9': '>',
        '\u2081': '_1', '\u2082': '_2', '\u2083': '_3',
        '\u00b2': '^2', '\u00b3': '^3',
        '\u207b': '^-', '\u2079': '^9',
        '\u03c1': 'rho', '\u03c7': 'chi',
        '⚠': '[WARNING]', '✓': '[CHECK]',
        '🎯': '[TARGET]', '🔍': '[SEARCH]',
        '💡': '[IDEA]', '📊': '[CHART]',
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text

def main():
    # Read the markdown file
    with open(r'presentations\SPEAKING_NOTES_17SLIDES_FINAL.md', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split into slide sections
    slide_sections = content.split('## **SLIDE')[1:]  # Skip header before first slide
    
    # Stop at Backup section
    if '# Backup' in content:
        backup_pos = content.index('# Backup')
        content_before_backup = content[:backup_pos]
        slide_sections = content_before_backup.split('## **SLIDE')[1:]
    
    print(f"Found {len(slide_sections)} slides")
    
    # Parse each slide
    slides = []
    for section in slide_sections:
        # Extract slide number and title from first line
        first_line = section.split('\n')[0]
        # More flexible regex that handles titles with (N=3) and other parentheses
        match = re.match(r'\s*(\d+):\s+(.+?)\s*\((\d+:\d+-\d+:\d+)\)\s+\[([^\]]+)\]', first_line)
        if match:
            slide_num = match.group(1)
            title = match.group(2).strip()
            time_range = match.group(3).strip()
            duration = match.group(4).strip()
            
            # Get the content (everything after first line, before ---)
            content_lines = section.split('\n')[1:]
            notes = []
            for line in content_lines:
                line = line.strip()
                if line and not line.startswith('---'):
                    line = line.replace('**', '')
                    notes.append(line)
            
            slides.append({
                'number': slide_num,
                'title': clean_text(title),
                'time_range': time_range,
                'duration': duration,
                'notes': clean_text('\n'.join(notes))
            })
    
    print(f"Parsed {len(slides)} slides successfully")
    
    # Create PDF
    output_pdf = r'presentations\SPEAKING_NOTES_17SLIDES_FINAL.pdf'
    doc = SimpleDocTemplate(output_pdf, pagesize=letter,
                          rightMargin=0.5*inch, leftMargin=0.5*inch,
                          topMargin=0.5*inch, bottomMargin=0.5*inch)
    
    # Define styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=16,
        textColor='#1E3A8A',
        spaceAfter=12,
        alignment=TA_LEFT
    )
    
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['BodyText'],
        fontSize=11,
        leading=14,
        alignment=TA_LEFT
    )
    
    # Build PDF content
    story = []
    
    # Title page
    story.append(Paragraph("Speaking Notes", ParagraphStyle('TitlePage', 
                          parent=styles['Heading1'], fontSize=24, alignment=1)))
    story.append(Spacer(1, 0.3*inch))
    story.append(Paragraph("Machine Learning-Driven Quantum Hacking of CHSH-Based QKD",
                          ParagraphStyle('Subtitle', parent=styles['Heading2'], 
                          fontSize=14, alignment=1)))
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph(f"{len(slides)} Slides | 19-20 Minutes",
                          ParagraphStyle('Info', parent=styles['Normal'], 
                          fontSize=12, alignment=1)))
    story.append(PageBreak())
    
    # Add each slide
    for slide in slides:
        # Slide header
        header_text = f"Slide {slide['number']}: {slide['title']}"
        story.append(Paragraph(header_text, title_style))
        
        # Time info
        time_text = f"Time: {slide['time_range']} | Duration: {slide['duration']}"
        story.append(Paragraph(time_text, body_style))
        story.append(Spacer(1, 0.15*inch))
        
        # Notes (preserve line breaks)
        for line in slide['notes'].split('\n'):
            if line.strip():
                story.append(Paragraph(line, body_style))
                story.append(Spacer(1, 0.05*inch))
        
        story.append(PageBreak())
    
    # Build PDF
    doc.build(story)
    print(f"PDF created successfully: {output_pdf}")

if __name__ == '__main__':
    main()
