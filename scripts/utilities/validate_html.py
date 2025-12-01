import re

with open('presentations/presentation_20slides.html', encoding='utf-8') as f:
    content = f.read()

# Find all slide sections
slide_pattern = r'(<!-- Slide \d+:.*?-->.*?)(?=<!-- Slide \d+:|<!-- Navigation -->|$)'
slides = re.findall(slide_pattern, content, re.DOTALL)

print(f"Found {len(slides)} slide sections\n")

for i, slide in enumerate(slides, 1):
    # Count divs in this slide
    open_divs = slide.count('<div')
    close_divs = slide.count('</div>')
    
    # Check for data-slide attribute
    data_slide_match = re.search(r'data-slide="(\d+)"', slide)
    data_slide = data_slide_match.group(1) if data_slide_match else "MISSING"
    
    # Check for footer
    has_footer = 'slide-footer' in slide
    
    status = "✓" if (open_divs == close_divs and has_footer) else "✗"
    
    print(f"{status} Slide {i} (data-slide={data_slide}): {open_divs} open, {close_divs} close, footer={has_footer}")
    
    if open_divs != close_divs:
        print(f"   ⚠️ MISMATCH: {open_divs - close_divs} unclosed divs")
