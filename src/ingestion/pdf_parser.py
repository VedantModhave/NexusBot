"""
PDF Parser Module
Extracts text from PDF files using pdfplumber.
"""

import os
from typing import List, Dict, Any

try:
    import pdfplumber
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False


class PDFParser:
    """Extracts and chunks text from PDF documents."""

    def __init__(self):
        if not PDF_AVAILABLE:
            print("Warning: pdfplumber not installed. PDF parsing unavailable.")

    def extract_text(self, pdf_path: str) -> str:
        """Extract all text from a PDF file."""
        if not PDF_AVAILABLE:
            raise ImportError("pdfplumber is required for PDF parsing. Install with: pip install pdfplumber")

        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        full_text = []
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                text = page.extract_text()
                if text:
                    full_text.append(f"[Page {page_num}]\n{text}")

        return "\n\n".join(full_text)

    def extract_pages(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract text from each page as separate documents."""
        if not PDF_AVAILABLE:
            raise ImportError("pdfplumber is required for PDF parsing.")

        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        pages = []
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                text = page.extract_text()
                if text:
                    pages.append({
                        "page_number": page_num,
                        "text": text.strip(),
                        "source": os.path.basename(pdf_path)
                    })

        return pages

    def parse_directory(self, directory: str) -> List[Dict[str, Any]]:
        """Parse all PDFs in a directory."""
        results = []
        if not os.path.exists(directory):
            return results

        for filename in os.listdir(directory):
            if filename.lower().endswith(".pdf"):
                filepath = os.path.join(directory, filename)
                try:
                    pages = self.extract_pages(filepath)
                    results.extend(pages)
                    print(f"Parsed {filename}: {len(pages)} pages")
                except Exception as e:
                    print(f"Error parsing {filename}: {e}")

        return results


if __name__ == "__main__":
    parser = PDFParser()
    print(f"PDF support available: {PDF_AVAILABLE}")
