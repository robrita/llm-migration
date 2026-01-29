"""
Unit tests for helpers/pdf_extractor.py

Tests cover:
- Text extraction from valid PDFs
- File size validation (50MB limit)
- Empty PDF handling
- Invalid PDF handling
"""

from unittest.mock import MagicMock, patch

import pytest

from helpers.pdf_extractor import MAX_PDF_SIZE_BYTES, extract_text_from_pdf

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_pdf_with_text():
    """Create a mock PDF document with extractable text."""
    mock_page1 = MagicMock()
    mock_page1.get_text.return_value = "Page 1 content: Introduction to testing."

    mock_page2 = MagicMock()
    mock_page2.get_text.return_value = "Page 2 content: More detailed information."

    mock_doc = MagicMock()
    mock_doc.__len__ = MagicMock(return_value=2)
    mock_doc.__getitem__ = MagicMock(side_effect=[mock_page1, mock_page2])
    mock_doc.close = MagicMock()

    return mock_doc


@pytest.fixture
def mock_pdf_empty_text():
    """Create a mock PDF document with no extractable text."""
    mock_page = MagicMock()
    mock_page.get_text.return_value = "   \n   "  # Whitespace only

    mock_doc = MagicMock()
    mock_doc.__len__ = MagicMock(return_value=1)
    mock_doc.__getitem__ = MagicMock(return_value=mock_page)
    mock_doc.close = MagicMock()

    return mock_doc


@pytest.fixture
def mock_pdf_no_pages():
    """Create a mock PDF document with no pages."""
    mock_doc = MagicMock()
    mock_doc.__len__ = MagicMock(return_value=0)
    mock_doc.close = MagicMock()

    return mock_doc


# ============================================================================
# Test Cases
# ============================================================================


@pytest.mark.unit
class TestExtractTextFromPDF:
    """Tests for extract_text_from_pdf function."""

    def test_extract_text_success(self, mock_pdf_with_text):
        """Test successful text extraction from valid PDF."""
        with patch("helpers.pdf_extractor.fitz.open", return_value=mock_pdf_with_text):
            # Create mock PDF bytes (small size)
            pdf_bytes = b"mock pdf content" * 100

            result = extract_text_from_pdf(pdf_bytes)

            assert result["ok"] is True
            assert "content" in result
            assert "Page 1 content" in result["content"]
            assert "Page 2 content" in result["content"]
            assert result["page_count"] == 2
            assert result["file_size_bytes"] == len(pdf_bytes)
            assert "processing_time" in result
            mock_pdf_with_text.close.assert_called_once()

    def test_empty_pdf_bytes(self):
        """Test handling of empty PDF bytes."""
        result = extract_text_from_pdf(b"")

        assert result["ok"] is False
        assert "empty" in result["error"].lower()

    def test_pdf_size_limit_exceeded(self):
        """Test 50MB file size limit enforcement."""
        # Create bytes larger than 50MB
        large_pdf_bytes = b"x" * (MAX_PDF_SIZE_BYTES + 1)

        result = extract_text_from_pdf(large_pdf_bytes)

        assert result["ok"] is False
        assert "50MB" in result["error"]

    def test_pdf_at_size_limit(self, mock_pdf_with_text):
        """Test PDF exactly at 50MB limit should work."""
        with patch("helpers.pdf_extractor.fitz.open", return_value=mock_pdf_with_text):
            # Exactly at limit
            pdf_bytes = b"x" * MAX_PDF_SIZE_BYTES

            result = extract_text_from_pdf(pdf_bytes)

            # Should succeed (at limit, not over)
            assert result["ok"] is True

    def test_pdf_no_extractable_text(self, mock_pdf_empty_text):
        """Test handling of PDF with no extractable text (image-based)."""
        with patch("helpers.pdf_extractor.fitz.open", return_value=mock_pdf_empty_text):
            pdf_bytes = b"mock pdf content"

            result = extract_text_from_pdf(pdf_bytes)

            assert result["ok"] is False
            assert "no extractable text" in result["error"].lower()
            mock_pdf_empty_text.close.assert_called_once()

    def test_pdf_no_pages(self, mock_pdf_no_pages):
        """Test handling of PDF with no pages."""
        with patch("helpers.pdf_extractor.fitz.open", return_value=mock_pdf_no_pages):
            pdf_bytes = b"mock pdf content"

            result = extract_text_from_pdf(pdf_bytes)

            assert result["ok"] is False
            assert "no pages" in result["error"].lower()
            mock_pdf_no_pages.close.assert_called_once()

    def test_invalid_pdf_format(self):
        """Test handling of invalid PDF format."""
        import fitz

        with patch(
            "helpers.pdf_extractor.fitz.open",
            side_effect=fitz.FileDataError("Cannot open broken file"),
        ):
            pdf_bytes = b"not a valid pdf"

            result = extract_text_from_pdf(pdf_bytes)

            assert result["ok"] is False
            assert "invalid pdf format" in result["error"].lower()

    def test_generic_exception_handling(self):
        """Test handling of unexpected exceptions."""
        with patch(
            "helpers.pdf_extractor.fitz.open",
            side_effect=RuntimeError("Unexpected error"),
        ):
            pdf_bytes = b"some pdf bytes"

            result = extract_text_from_pdf(pdf_bytes)

            assert result["ok"] is False
            assert "extraction failed" in result["error"].lower()

    def test_processing_time_recorded(self, mock_pdf_with_text):
        """Test that processing time is recorded."""
        with patch("helpers.pdf_extractor.fitz.open", return_value=mock_pdf_with_text):
            pdf_bytes = b"mock pdf content"

            result = extract_text_from_pdf(pdf_bytes)

            assert result["ok"] is True
            assert result["processing_time"] >= 0
            assert isinstance(result["processing_time"], float)


@pytest.mark.unit
class TestPDFSizeLimit:
    """Tests specifically for PDF size limit."""

    def test_max_size_constant(self):
        """Test MAX_PDF_SIZE_BYTES is 50MB."""
        assert MAX_PDF_SIZE_BYTES == 50 * 1024 * 1024

    def test_size_limit_consistent_with_spec(self):
        """Test size limit matches specification (50MB)."""
        # From spec.md: "Maximum 50MB PDF file size"
        expected_mb = 50
        actual_mb = MAX_PDF_SIZE_BYTES / (1024 * 1024)
        assert actual_mb == expected_mb
