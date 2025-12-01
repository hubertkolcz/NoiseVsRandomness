#!/usr/bin/env python3
"""
HTML to PDF Converter for ML-Driven Quantum Hacking Presentation
Converts presentation_20slides.html to presentation_20slides.pdf
Each slide becomes a separate page in the PDF.

Install requirements:
    pip install playwright
    playwright install chromium

Usage:
    python html_to_pdf_converter.py
"""

import asyncio
from playwright.async_api import async_playwright
import os

async def convert_html_to_pdf(html_file='presentations/presentation_20slides.html', output_file='presentations/presentation_20slides.pdf'):
    """
    Convert HTML presentation to PDF with each slide as a separate page.
    Uses Playwright to render the HTML exactly as it appears in browser.
    Renders each slide individually to ensure all content is captured.
    """
    
    if not os.path.exists(html_file):
        print(f"❌ Error: {html_file} not found")
        return
    
    print(f"Converting {html_file} to PDF format...")
    print("Using Playwright for high-fidelity rendering...")
    
    from PyPDF2 import PdfMerger
    import tempfile
    
    async with async_playwright() as p:
        # Launch browser in headless mode
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page(viewport={'width': 1920, 'height': 1080})
        
        # Load the HTML file
        file_path = os.path.abspath(html_file)
        await page.goto(f'file:///{file_path}')
        
        # Wait for page to fully load
        await page.wait_for_load_state('networkidle')
        await page.wait_for_timeout(1000)
        
        # Get total number of slides
        total_slides = await page.evaluate('totalSlides')
        print(f"\nFound {total_slides} slides to convert")
        print("Rendering each slide individually for complete content capture...")
        
        # Create temporary directory for individual slide PDFs
        temp_dir = tempfile.mkdtemp()
        pdf_files = []
        
        # Inject CSS for single slide display
        await page.add_style_tag(content="""
            .navigation { display: none !important; }
            body { margin: 0; padding: 0; }
            .slide { 
                display: none !important;
                padding: 30px 50px !important;
                background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
            }
            .slide.active { 
                display: flex !important;
                flex-direction: column;
            }
            .slide.title-slide {
                background: linear-gradient(135deg, #1e40af 0%, #7c3aed 100%) !important;
            }
            .slide-footer { display: block !important; }
            
            /* Reduced header sizes and margins */
            .slide-header { margin-bottom: 16px !important; }
            .slide-title { font-size: 34px !important; margin-bottom: 10px !important; padding-bottom: 8px !important; }
            .main-title { font-size: 46px !important; }
            .subtitle { font-size: 24px !important; }
            
            /* Compact content */
            img { max-height: 380px !important; max-width: 100% !important; margin: 12px auto !important; }
            
            /* Lists */
            .bullet-list { font-size: 17px !important; line-height: 1.5 !important; }
            .bullet-list li { margin-bottom: 8px !important; padding-left: 30px !important; }
            .bullet-list li:before { font-size: 22px !important; }
            
            .numbered-list { font-size: 17px !important; line-height: 1.5 !important; }
            .numbered-list li { margin-bottom: 8px !important; padding-left: 38px !important; }
            .numbered-list li:before { font-size: 20px !important; }
            
            /* Boxes and sections */
            .highlight-box { font-size: 17px !important; padding: 14px !important; margin: 14px 0 !important; line-height: 1.5 !important; }
            .section-box { padding: 16px !important; margin: 14px 0 !important; }
            .section-title { font-size: 19px !important; margin-bottom: 10px !important; }
            .card { padding: 14px !important; }
            .card-title { font-size: 18px !important; margin-bottom: 8px !important; }
            .card-content { font-size: 16px !important; }
            
            /* Tables */
            table { font-size: 16px !important; margin: 16px 0 !important; }
            th { padding: 12px !important; font-size: 17px !important; }
            td { padding: 10px 12px !important; }
            
            /* Grids */
            .grid-2, .two-column { gap: 18px !important; margin: 16px 0 !important; }
            .grid-3 { gap: 16px !important; margin: 16px 0 !important; }
            .grid-4 { gap: 14px !important; margin: 14px 0 !important; }
            
            /* Equations */
            .equation { font-size: 32px !important; padding: 18px !important; margin: 20px 0 !important; }
            
            /* Column headers */
            .column-header { font-size: 22px !important; margin-bottom: 12px !important; }
            
            /* Flowchart */
            .flowchart { margin: 16px 0 !important; }
            .flowchart-item { padding: 16px !important; font-size: 16px !important; }
            
            /* Footer elements */
            .acknowledgments { font-size: 13px !important; margin-top: 14px !important; line-height: 1.4 !important; }
            .contact-info { font-size: 14px !important; margin-top: 10px !important; }
            .questions { font-size: 34px !important; margin-top: 16px !important; }
            .author-info { font-size: 20px !important; }
            .coauthors { font-size: 17px !important; }
            .conference-name { font-size: 19px !important; }
            
            /* Slide-specific adjustments */
            [data-slide="8"] .highlight-box { font-size: 15px !important; padding: 12px !important; line-height: 1.4 !important; }
            [data-slide="8"] .section-box { padding: 14px !important; }
            [data-slide="8"] .section-title { font-size: 17px !important; }
            [data-slide="8"] .bullet-list { font-size: 15px !important; }
            
            [data-slide="11"] .highlight-box { font-size: 15px !important; padding: 12px !important; line-height: 1.4 !important; }
            [data-slide="11"] .section-box { padding: 14px !important; }
            [data-slide="11"] .section-title { font-size: 17px !important; }
            [data-slide="11"] .bullet-list { font-size: 16px !important; }
            [data-slide="11"] img { max-height: 320px !important; }
            [data-slide="11"] .slide-title { font-size: 30px !important; }
            
            [data-slide="12"] .highlight-box { font-size: 15px !important; padding: 11px !important; line-height: 1.4 !important; }
            [data-slide="12"] .section-box { padding: 13px !important; }
            [data-slide="12"] .section-title { font-size: 16px !important; }
            [data-slide="12"] .bullet-list { font-size: 14px !important; line-height: 1.4 !important; }
            [data-slide="12"] .bullet-list li { margin-bottom: 6px !important; }
            
            [data-slide="18"] img { max-height: 600px !important; }
        """)
        
        # Render each slide individually
        print(f"\n")
        for slide_num in range(1, total_slides + 1):
            print(f"  Rendering slide {slide_num}/{total_slides}...")
            
            # Navigate to specific slide
            await page.evaluate(f'showSlide({slide_num})')
            await page.wait_for_timeout(800)  # Increased wait time
            
            # Verify the correct slide is visible
            visible_slide = await page.evaluate('document.querySelector(".slide.active")?.getAttribute("data-slide")')
            active_count = await page.evaluate('document.querySelectorAll(".slide.active").length')
            if str(visible_slide) != str(slide_num):
                print(f"    ⚠️  Warning: Expected slide {slide_num} but got {visible_slide}")
            if active_count != 1:
                print(f"    ⚠️  Warning: {active_count} active slides (should be 1)")
            
            # Generate PDF for this slide
            slide_pdf = os.path.join(temp_dir, f'slide_{slide_num:02d}.pdf')
            await page.pdf(
                path=slide_pdf,
                width='11.69in',
                height='8.27in',
                print_background=True,
                margin={'top': '0', 'right': '0', 'bottom': '0', 'left': '0'}
            )
            pdf_files.append(slide_pdf)
        
        await browser.close()
        
        print(f"\n\nMerging {total_slides} slides into final PDF...")
        
        # Merge all individual PDFs
        merger = PdfMerger()
        for pdf_file in pdf_files:
            # Check page count of individual PDF
            from PyPDF2 import PdfReader
            temp_reader = PdfReader(pdf_file)
            page_count = len(temp_reader.pages)
            if page_count > 1:
                print(f"  ⚠️  {os.path.basename(pdf_file)} has {page_count} pages!")
            merger.append(pdf_file)
        
        merger.write(output_file)
        merger.close()
        
        # Cleanup temporary files
        import shutil
        shutil.rmtree(temp_dir)
    
    # Get file size
    file_size = os.path.getsize(output_file) / (1024 * 1024)  # Convert to MB
    
    print(f"\n✅ PDF created successfully: {output_file}")
    print(f"   Pages: {total_slides}")
    print(f"   Format: A4 Landscape (11.69\" × 8.27\")")
    print(f"   Size: {file_size:.2f} MB")
    print(f"   Each slide is a separate page")

if __name__ == '__main__':
    asyncio.run(convert_html_to_pdf())
