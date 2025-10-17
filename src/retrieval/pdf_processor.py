"""
PDF processor for extracting and chunking academic papers.
Handles text extraction, citation preservation, and metadata extraction.
"""

from typing import List, Dict, Optional
from pathlib import Path
import fitz  # PyMuPDF
from loguru import logger
from src.config import config
import re
import hashlib


class PDFChunk:
    """Represents a chunk of text from a PDF"""
    def __init__(
        self,
        text: str,
        page: int,
        chunk_id: str,
        metadata: Dict
    ):
        self.text = text
        self.page = page
        self.chunk_id = chunk_id
        self.metadata = metadata


class PDFProcessor:
    """Process PDF files for RAG ingestion"""

    def __init__(self):
        self.chunk_size = config.rag.chunk_size
        self.chunk_overlap = config.rag.chunk_overlap
        logger.info(f"PDF processor initialized (chunk_size={self.chunk_size}, overlap={self.chunk_overlap})")

    def extract_text_from_pdf(self, pdf_path: str) -> List[Dict]:
        """
        Extract text from PDF with page information.

        Args:
            pdf_path: Path to PDF file

        Returns:
            List of dicts with 'page' and 'text' keys
        """
        try:
            doc = fitz.open(pdf_path)
            pages = []

            for page_num, page in enumerate(doc, start=1):
                text = page.get_text()
                if text.strip():
                    pages.append({
                        'page': page_num,
                        'text': text
                    })

            doc.close()

            logger.info(f"✅ Extracted {len(pages)} pages from {Path(pdf_path).name}")
            return pages

        except Exception as e:
            logger.error(f"Failed to extract text from {pdf_path}: {e}")
            raise

    def extract_metadata(self, pdf_path: str) -> Dict:
        """
        Extract metadata from PDF.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Dict with metadata (title, author, year, etc.)
        """
        try:
            doc = fitz.open(pdf_path)
            metadata = doc.metadata or {}
            doc.close()

            # Extract filename info
            filename = Path(pdf_path).stem

            # Try to extract year from filename (common pattern: paper_2024.pdf)
            year_match = re.search(r'(19|20)\d{2}', filename)
            year = year_match.group(0) if year_match else None

            result = {
                'title': metadata.get('title') or filename,
                'author': metadata.get('author') or 'Unknown',
                'subject': metadata.get('subject', ''),
                'year': year,
                'filename': Path(pdf_path).name,
                'source': pdf_path
            }

            logger.debug(f"Metadata extracted: {result['title']}")
            return result

        except Exception as e:
            logger.error(f"Failed to extract metadata from {pdf_path}: {e}")
            # Return basic metadata
            return {
                'title': Path(pdf_path).stem,
                'author': 'Unknown',
                'year': None,
                'filename': Path(pdf_path).name,
                'source': pdf_path
            }

    def chunk_text(
        self,
        text: str,
        page: int,
        metadata: Dict
    ) -> List[PDFChunk]:
        """
        Split text into overlapping chunks.

        Args:
            text: Text to chunk
            page: Page number
            metadata: PDF metadata

        Returns:
            List of PDFChunk objects
        """
        # Split by sentences (rough approximation)
        sentences = re.split(r'(?<=[.!?])\s+', text)

        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            sentence_length = len(sentence)

            # If adding this sentence exceeds chunk size, save current chunk
            if current_length + sentence_length > self.chunk_size and current_chunk:
                chunk_text = ' '.join(current_chunk)

                # Create unique chunk ID
                chunk_id = self._generate_chunk_id(chunk_text, page)

                # Create chunk with metadata
                chunk_metadata = {
                    **metadata,
                    'page': page,
                    'chunk_size': len(chunk_text)
                }

                chunks.append(PDFChunk(
                    text=chunk_text,
                    page=page,
                    chunk_id=chunk_id,
                    metadata=chunk_metadata
                ))

                # Start new chunk with overlap
                # Keep last few sentences for overlap
                overlap_text = ' '.join(current_chunk[-3:]) if len(current_chunk) > 3 else ''
                current_chunk = [overlap_text, sentence] if overlap_text else [sentence]
                current_length = len(overlap_text) + sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length

        # Add final chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunk_id = self._generate_chunk_id(chunk_text, page)

            chunk_metadata = {
                **metadata,
                'page': page,
                'chunk_size': len(chunk_text)
            }

            chunks.append(PDFChunk(
                text=chunk_text,
                page=page,
                chunk_id=chunk_id,
                metadata=chunk_metadata
            ))

        return chunks

    def _generate_chunk_id(self, text: str, page: int) -> str:
        """Generate unique ID for a chunk"""
        # Create hash from text + page
        content = f"{text}_{page}"
        hash_obj = hashlib.md5(content.encode())
        return f"chunk_{hash_obj.hexdigest()[:12]}"

    def extract_citations(self, text: str) -> List[str]:
        """
        Extract academic citations from text.

        Args:
            text: Text to extract citations from

        Returns:
            List of citation strings
        """
        # Pattern for citations like "Zhang et al., 2024" or "Smith and Jones (2023)"
        patterns = [
            r'\b([A-Z][a-z]+\s+et\s+al\.\s*,?\s*\(?\d{4}\)?)',
            r'\b([A-Z][a-z]+\s+and\s+[A-Z][a-z]+\s*,?\s*\(?\d{4}\)?)',
            r'\b([A-Z][a-z]+\s+&\s+[A-Z][a-z]+\s*,?\s*\(?\d{4}\)?)'
        ]

        citations = []
        for pattern in patterns:
            matches = re.findall(pattern, text)
            citations.extend(matches)

        # Remove duplicates while preserving order
        seen = set()
        unique_citations = []
        for citation in citations:
            if citation not in seen:
                seen.add(citation)
                unique_citations.append(citation)

        return unique_citations

    def process_pdf(self, pdf_path: str) -> List[PDFChunk]:
        """
        Process entire PDF: extract text, metadata, and create chunks.

        Args:
            pdf_path: Path to PDF file

        Returns:
            List of PDFChunk objects ready for ingestion
        """
        logger.info(f"Processing PDF: {pdf_path}")

        # Extract metadata
        metadata = self.extract_metadata(pdf_path)

        # Extract text by page
        pages = self.extract_text_from_pdf(pdf_path)

        # Create chunks from all pages
        all_chunks = []
        for page_data in pages:
            page_chunks = self.chunk_text(
                text=page_data['text'],
                page=page_data['page'],
                metadata=metadata
            )
            all_chunks.extend(page_chunks)

            # Extract citations from page
            citations = self.extract_citations(page_data['text'])
            if citations:
                logger.debug(f"Page {page_data['page']}: found {len(citations)} citations")

        logger.success(f"✅ Processed {pdf_path}: {len(all_chunks)} chunks created")

        return all_chunks

    def process_directory(self, directory_path: str) -> Dict[str, List[PDFChunk]]:
        """
        Process all PDFs in a directory.

        Args:
            directory_path: Path to directory containing PDFs

        Returns:
            Dict mapping PDF filename to list of chunks
        """
        pdf_dir = Path(directory_path)
        if not pdf_dir.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")

        pdf_files = list(pdf_dir.glob("*.pdf"))
        logger.info(f"Found {len(pdf_files)} PDF files in {directory_path}")

        results = {}
        for pdf_path in pdf_files:
            try:
                chunks = self.process_pdf(str(pdf_path))
                results[pdf_path.name] = chunks
            except Exception as e:
                logger.error(f"Failed to process {pdf_path.name}: {e}")
                continue

        logger.success(f"✅ Processed {len(results)} PDFs successfully")

        return results


# Global PDF processor instance
pdf_processor = PDFProcessor()
