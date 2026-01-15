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


def split_pdf_with_overlap(
    input_pdf: str,
    output_dir: str,
    pages_per_part: int,
    overlap_pages: int = 0
) -> list[str]:
    """
    Split a PDF into multiple parts with optional page overlap.
    
    Args:
        input_pdf: Path to the input PDF file
        output_dir: Directory to save the split PDF files
        pages_per_part: Number of pages in each part
        overlap_pages: Number of pages to overlap between consecutive parts
    
    Returns:
        List of paths to the generated PDF files
    
    Raises:
        FileNotFoundError: If input PDF does not exist
        ValueError: If parameters are invalid
    
    Example:
        With pages_per_part=10 and overlap_pages=1:
        - Part 1: pages 1-10
        - Part 2: pages 10-19 (page 10 overlaps)
        - Part 3: pages 19-28 (page 19 overlaps)
    """
    input_path = Path(input_pdf)
    
    # Validate input file
    if not input_path.exists():
        raise FileNotFoundError(f"Input PDF not found: {input_pdf}")
    
    # Validate parameters
    if pages_per_part < 1:
        raise ValueError(f"pages_per_part must be >= 1, got {pages_per_part}")
    
    if overlap_pages < 0:
        raise ValueError(f"overlap_pages must be >= 0, got {overlap_pages}")
    
    if overlap_pages >= pages_per_part:
        raise ValueError(f"overlap_pages ({overlap_pages}) must be less than pages_per_part ({pages_per_part})")
    
    # Read the input PDF
    print(f"Reading PDF: {input_pdf}")
    reader = PdfReader(input_pdf)
    total_pages = len(reader.pages)
    print(f"Total pages in document: {total_pages}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Calculate the step size (how many pages to advance for each part)
    step_size = pages_per_part - overlap_pages
    
    # Get base name without extension
    base_name = input_path.stem
    
    # Split the PDF
    output_files = []
    part_num = 1
    start_page = 1
    
    while start_page <= total_pages:
        # Calculate end page for this part
        end_page = min(start_page + pages_per_part - 1, total_pages)
        
        # Generate output filename with page range
        output_filename = f"{base_name}_pages_{start_page}-{end_page}.pdf"
        output_file_path = output_path / output_filename
        
        # Extract this part
        print(f"\nExtracting Part {part_num}: pages {start_page}-{end_page}...")
        writer = PdfWriter()
        
        # PyPDF2 uses 0-indexed pages internally
        for page_num in range(start_page - 1, end_page):
            writer.add_page(reader.pages[page_num])
        
        # Save the part
        with open(output_file_path, 'wb') as output_file:
            writer.write(output_file)
        
        pages_extracted = end_page - start_page + 1
        print(f"Saved {pages_extracted} pages to: {output_file_path}")
        output_files.append(str(output_file_path))
        
        # Move to next part
        # If we've reached the last page, stop
        if end_page >= total_pages:
            break
        
        # Calculate next start page with overlap
        start_page = start_page + step_size
        part_num += 1
    
    print(f"\n✓ Successfully split PDF into {len(output_files)} parts")
    return output_files


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


def main_interactive():
    """Interactive mode for splitting PDF with overlap"""
    print("=" * 60)
    print("PDF Splitter with Overlap")
    print("=" * 60)
    
    # Get input PDF path
    input_pdf = input("\nEnter the path to the PDF file: ").strip().strip('"').strip("'")
    
    if not input_pdf:
        print("Error: PDF path cannot be empty")
        return
    
    # Get output directory
    output_dir = input("Enter the output directory: ").strip().strip('"').strip("'")
    
    if not output_dir:
        print("Error: Output directory cannot be empty")
        return
    
    # Get pages per part
    try:
        pages_per_part = int(input("Enter pages per part (e.g., 10): ").strip())
    except ValueError:
        print("Error: Pages per part must be a valid integer")
        return
    
    # Get overlap pages
    try:
        overlap_pages = int(input("Enter overlap pages (e.g., 1): ").strip())
    except ValueError:
        print("Error: Overlap pages must be a valid integer")
        return
    
    # Perform the split
    print("\n" + "=" * 60)
    try:
        output_files = split_pdf_with_overlap(
            input_pdf=input_pdf,
            output_dir=output_dir,
            pages_per_part=pages_per_part,
            overlap_pages=overlap_pages
        )
        
        print("\n" + "=" * 60)
        print(f"✓ Split completed! Generated {len(output_files)} files:")
        for file_path in output_files:
            print(f"  - {file_path}")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    # Run interactive mode for PDF splitting with overlap
    main_interactive()
    
    # ========== Other Usage Examples ==========
    
    # Example 1: Programmatic usage of split_pdf_with_overlap
    # split_pdf_with_overlap(
    #     input_pdf="path/to/input.pdf",
    #     output_dir="path/to/output_directory",
    #     pages_per_part=10,
    #     overlap_pages=1
    # )
    
    # Example 2: Extract specific page range (original function)
    # extract_pdf_pages(
    #     input_pdf="path/to/input.pdf",
    #     output_pdf="path/to/output.pdf",
    #     start_page=1,
    #     end_page=100
    # )
    
    # Example 3: Extract from page 10 to the end
    # extract_pdf_pages(
    #     input_pdf="path/to/input.pdf",
    #     output_pdf="path/to/output.pdf",
    #     start_page=10,
    #     end_page=None  # None means extract to the last page
    # )
