from bs4 import BeautifulSoup
import sys

with open('presentations/presentation_20slides.html', encoding='utf-8') as f:
    content = f.read()

# Try to parse with different parsers
print("Parsing with html.parser...")
soup = BeautifulSoup(content, 'html.parser')

# Check for any parsing warnings
print("\nChecking for structural issues...")

# Find all slides
slides = soup.find_all('div', {'data-slide': True})
print(f"Total slides with data-slide: {len(slides)}")

# Check each slide for proper nesting
for i, slide in enumerate(slides, 1):
    slide_num = slide.get('data-slide')
    
    # Check if slide has required children
    header = slide.find('div', class_='slide-header')
    content_div = slide.find('div', class_='slide-content')
    footer = slide.find('div', class_='slide-footer')
    
    # Check for orphaned closing tags or unclosed tags
    slide_html = str(slide)
    open_divs = slide_html.count('<div')
    close_divs = slide_html.count('</div>')
    
    if open_divs != close_divs:
        print(f"❌ Slide {slide_num}: Unbalanced divs - {open_divs} open, {close_divs} close")
        print(f"   Difference: {abs(open_divs - close_divs)}")
        
        # Try to find where the issue is
        if i < len(slides):
            # Check if the next slide is being captured inside this one
            next_slide_marker = f'data-slide="{int(slide_num)+1}"'
            if next_slide_marker in slide_html:
                print(f"   ⚠️  Next slide appears to be nested inside this slide!")
    
    if not header and slide_num != '1':  # Slide 1 is title slide
        print(f"⚠️  Slide {slide_num}: Missing slide-header")
    if not content_div:
        print(f"⚠️  Slide {slide_num}: Missing slide-content")
    if not footer and slide_num != '1':
        print(f"⚠️  Slide {slide_num}: Missing slide-footer")

print("\n✓ Basic structural check complete")
