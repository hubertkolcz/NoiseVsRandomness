from bs4 import BeautifulSoup

with open('presentations/presentation_20slides.html', encoding='utf-8') as f:
    soup = BeautifulSoup(f, 'html.parser')

slides = soup.find_all('div', class_='slide')
print(f'Total slides found by BeautifulSoup: {len(slides)}')

for i, slide in enumerate(slides, 1):
    data_slide = slide.get('data-slide')
    has_footer = slide.find('div', class_='slide-footer')
    has_header = slide.find('div', class_='slide-header')
    has_content = slide.find('div', class_='slide-content')
    
    print(f"Slide {i}: data-slide={data_slide}, header={'✓' if has_header else '✗'}, content={'✓' if has_content else '✗'}, footer={'✓' if has_footer else '✗'}")
