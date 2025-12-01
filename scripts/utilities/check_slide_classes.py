from bs4 import BeautifulSoup

soup = BeautifulSoup(open('presentations/presentation_20slides.html', encoding='utf-8'), 'html.parser')

# Find all divs with any class containing "slide"
all_divs = soup.find_all('div')
slide_related = []

for div in all_divs:
    classes = div.get('class', [])
    if any('slide' in c for c in classes):
        slide_related.append({
            'classes': classes,
            'data_slide': div.get('data-slide'),
            'has_children': len(list(div.children)) > 0
        })

print(f"Total divs with 'slide' in class name: {len(slide_related)}")

# Count by class type
from collections import Counter
class_counts = Counter()
for item in slide_related:
    for cls in item['classes']:
        class_counts[cls] += 1

print("\nBreakdown by class:")
for cls, count in sorted(class_counts.items()):
    print(f"  {cls}: {count}")

print(f"\nDivs with class='slide' and data-slide attribute: {len([s for s in slide_related if 'slide' in s['classes'] and s['data_slide']])}")
