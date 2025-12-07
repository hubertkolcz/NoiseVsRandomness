from PyPDF2 import PdfReader

reader = PdfReader('presentations/presentation_20slides.pdf')
print(f'Total pages: {len(reader.pages)}\n')

for i in range(len(reader.pages)):
    print(f'\n{"="*80}')
    print(f'PAGE {i+1}')
    print(f'{"="*80}')
    text = reader.pages[i].extract_text()
    print(text)
