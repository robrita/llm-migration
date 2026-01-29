"""
LLM Evaluation Tool - PDF text extraction.

This module provides PDF text extraction using PyMuPDF (fitz).
"""

import logging
import time
from typing import Any

import fitz  # PyMuPDF

# Module-level logger
logger = logging.getLogger(__name__)

# Maximum file size: 50MB
MAX_PDF_SIZE_BYTES = 50 * 1024 * 1024


def extract_text_from_pdf(pdf_bytes: bytes) -> dict[str, Any]:
    """
    Extract text content from a PDF file.

    Args:
        pdf_bytes: Raw PDF file content as bytes

    Returns:
        Success: {"ok": True, "content": str, "page_count": int, "file_size_bytes": int, "processing_time": float}
        Failure: {"ok": False, "error": str}
    """
    start_time = time.time()

    try:
        # Validate file size
        file_size = len(pdf_bytes)
        if file_size > MAX_PDF_SIZE_BYTES:
            error_msg = f"PDF exceeds maximum size limit ({file_size / (1024 * 1024):.1f}MB > 50MB)"
            logger.warning(error_msg)
            return {"ok": False, "error": error_msg}

        if file_size == 0:
            logger.warning("Empty PDF file received")
            return {"ok": False, "error": "PDF file is empty"}

        logger.info(f"Extracting text from PDF ({file_size / 1024:.1f}KB)")

        # Open PDF from bytes
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")

        try:
            page_count = len(doc)
            if page_count == 0:
                logger.warning("PDF has no pages")
                return {"ok": False, "error": "PDF has no pages"}

            # Extract text from all pages (preserve page indices for reference)
            pages: list[str] = []
            for page_num in range(page_count):
                page = doc[page_num]
                page_text = page.get_text("text")
                pages.append(page_text.strip())

            # Join non-empty pages for backward-compat content field
            content = "\n\n".join(p for p in pages if p)

            if not content.strip():
                logger.warning("PDF contains no extractable text")
                return {"ok": False, "error": "No extractable text in PDF (may be image-based)"}

            processing_time = time.time() - start_time
            logger.info(
                f"Extracted {len(content)} chars from {page_count} pages in {processing_time:.2f}s"
            )

            return {
                "ok": True,
                "pages": pages,
                "content": content,
                "page_count": page_count,
                "file_size_bytes": file_size,
                "processing_time": processing_time,
            }

        finally:
            doc.close()

    except fitz.FileDataError as e:
        logger.error(f"Invalid PDF format: {e}")
        return {"ok": False, "error": f"Invalid PDF format: {e}"}
    except Exception as e:
        logger.error(f"PDF extraction failed: {e}", exc_info=True)
        return {"ok": False, "error": f"PDF extraction failed: {e}"}
