"""
PDF Page Extractor - Extract specific page ranges from PDF documents

This script extracts specified page ranges from a PDF file while preserving
the original formatting and structure.
"""

import sys
from pathlib import Path
from PyPDF2 import PdfReader, PdfWriter


def extract_pdf_pages(
    input_pdf: str,
    output_pdf: str,
    start_page: int = 1,
    end_page: int = None
) -> None:
    """
    Extract pages from a PDF file and save to a new PDF.
    
    Args:
        input_pdf: Path to the input PDF file
        output_pdf: Path to save the extracted PDF
        start_page: Starting page number (1-indexed, inclusive)
        end_page: Ending page number (1-indexed, inclusive). If None, extract to the last page
    
    Raises:
        FileNotFoundError: If input PDF does not exist
        ValueError: If page range is invalid
    """
    input_path = Path(input_pdf)
    
    # Validate input file
    if not input_path.exists():
        raise FileNotFoundError(f"Input PDF not found: {input_pdf}")
    
    # Read the input PDF
    print(f"Reading PDF: {input_pdf}")
    reader = PdfReader(input_pdf)
    total_pages = len(reader.pages)
    print(f"Total pages in document: {total_pages}")
    
    # Validate and adjust page range
    if start_page < 1:
        raise ValueError(f"Start page must be >= 1, got {start_page}")
    
    if end_page is None:
        end_page = total_pages
    
    if end_page > total_pages:
        print(f"Warning: End page {end_page} exceeds total pages {total_pages}, adjusting to {total_pages}")
        end_page = total_pages
    
    if start_page > end_page:
        raise ValueError(f"Start page ({start_page}) cannot be greater than end page ({end_page})")
    
    # Extract pages
    print(f"Extracting pages {start_page} to {end_page}...")
    writer = PdfWriter()
    
    # PyPDF2 uses 0-indexed pages internally
    for page_num in range(start_page - 1, end_page):
        writer.add_page(reader.pages[page_num])
    
    # Save the output PDF
    output_path = Path(output_pdf)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_pdf, 'wb') as output_file:
        writer.write(output_file)
    
    pages_extracted = end_page - start_page + 1
    print(f"Successfully extracted {pages_extracted} pages to: {output_pdf}")


def main():
    """Main function with example usage"""
    # Example configuration
    input_pdf = "c:/Users/qu179/PycharmProjects/zag-ai/playground-repo/rag/files/large/usda.pdf"
    output_pdf = "c:/Users/qu179/PycharmProjects/zag-ai/playground-repo/rag/files/large/usda_first_100_pages.pdf"
    
    # Extract first 100 pages
    start_page = 1
    end_page = 100
    
    try:
        extract_pdf_pages(
            input_pdf=input_pdf,
            output_pdf=output_pdf,
            start_page=start_page,
            end_page=end_page
        )
        print("\n✓ Extraction completed successfully!")
        
    except Exception as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    # You can modify these parameters as needed
    # Example 1: Extract first 100 pages (default behavior)
    main()
    
    # Example 2: Extract custom range (uncomment to use)
    # extract_pdf_pages(
    #     input_pdf="path/to/input.pdf",
    #     output_pdf="path/to/output.pdf",
    #     start_page=50,
    #     end_page=150
    # )
    
    # Example 3: Extract from page 10 to the end
    # extract_pdf_pages(
    #     input_pdf="path/to/input.pdf",
    #     output_pdf="path/to/output.pdf",
    #     start_page=10,
    #     end_page=None  # None means extract to the last page
    # )
