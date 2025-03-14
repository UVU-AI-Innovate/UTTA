#!/usr/bin/env python3
"""
Document Processor for Scientific Educational Books

This module handles the extraction, processing, and chunking of content from
educational books in various formats. It converts raw text into structured
knowledge chunks that can be stored in the vector database.

Supported formats:
- PDF (using PyMuPDF if available, fallback to PyPDF2)
- DOCX (using python-docx)
- TXT (direct reading)
"""

import os
import re
import logging
import json
import csv
from typing import List, Dict, Any, Optional
import importlib.util

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename="document_processing.log"
)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    """
    Process educational books and extract structured knowledge chunks.
    
    This class handles:
    1. Reading various document formats
    2. Extracting text content
    3. Dividing content into meaningful chunks
    4. Adding metadata and categorization
    5. Exporting structured data for vector storage
    """
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        """
        Initialize the document processor.
        
        Args:
            chunk_size: Target size of text chunks in characters
            chunk_overlap: Overlap between adjacent chunks to maintain context
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunks = []
        self.supported_extensions = ['.pdf', '.txt', '.docx']
        self._check_dependencies()
    
    def _check_dependencies(self):
        """Check for available document processing libraries."""
        # Check for PDF processing libraries
        self.has_pymupdf = importlib.util.find_spec("fitz") is not None
        self.has_pypdf2 = importlib.util.find_spec("PyPDF2") is not None
        
        # Check for DOCX processing
        self.has_docx = importlib.util.find_spec("docx") is not None
        
        # Log available processors
        logger.info("Document processors available:")
        logger.info(f"PyMuPDF (premium PDF extraction): {'Available' if self.has_pymupdf else 'Not available'}")
        logger.info(f"PyPDF2 (basic PDF extraction): {'Available' if self.has_pypdf2 else 'Not available'}")
        logger.info(f"python-docx (DOCX extraction): {'Available' if self.has_docx else 'Not available'}")
    
    def process_books(self, books_dir: str) -> None:
        """
        Process all supported books in the specified directory.
        
        Args:
            books_dir: Directory containing educational books
        """
        if not os.path.exists(books_dir):
            logger.error(f"Books directory not found: {books_dir}")
            return
        
        self.chunks = []  # Reset chunks before processing
        
        # Get all files with supported extensions
        book_files = [
            f for f in os.listdir(books_dir) 
            if os.path.isfile(os.path.join(books_dir, f)) and 
            any(f.lower().endswith(ext) for ext in self.supported_extensions)
        ]
        
        if not book_files:
            logger.warning(f"No supported book files found in {books_dir}")
            return
        
        logger.info(f"Found {len(book_files)} books to process")
        
        # Process each book
        for book_file in book_files:
            book_path = os.path.join(books_dir, book_file)
            try:
                logger.info(f"Processing book: {book_file}")
                self._process_book(book_path)
                logger.info(f"Successfully processed: {book_file}")
            except Exception as e:
                logger.error(f"Error processing {book_file}: {str(e)}")
        
        # Categorize chunks after all books are processed
        self._categorize_chunks()
        
        logger.info(f"Completed processing. Generated {len(self.chunks)} knowledge chunks")
    
    def _process_book(self, book_path: str) -> None:
        """
        Process a single book file.
        
        Args:
            book_path: Path to the book file
        """
        file_extension = os.path.splitext(book_path)[1].lower()
        
        # Extract text based on file type
        if file_extension == '.pdf':
            text = self._extract_pdf_text(book_path)
        elif file_extension == '.docx':
            text = self._extract_docx_text(book_path)
        elif file_extension == '.txt':
            text = self._extract_txt_text(book_path)
        else:
            logger.warning(f"Unsupported file format: {file_extension}")
            return
        
        if not text:
            logger.warning(f"No text content extracted from {book_path}")
            return
        
        # Create book metadata
        book_metadata = {
            "source": os.path.basename(book_path),
            "file_path": book_path,
            "file_type": file_extension[1:],  # Remove leading dot
            "file_size_kb": os.path.getsize(book_path) / 1024,
        }
        
        # Create chunks from the text
        self._create_chunks(text, book_metadata)
    
    def _extract_pdf_text(self, pdf_path: str) -> str:
        """
        Extract text from a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text content
        """
        text = ""
        
        # Try PyMuPDF first (better quality)
        if self.has_pymupdf:
            try:
                import fitz  # PyMuPDF
                doc = fitz.open(pdf_path)
                
                # Extract text from each page
                for page_num in range(len(doc)):
                    page = doc.load_page(page_num)
                    text += page.get_text() + "\n\n"
                
                doc.close()
                return text
            except Exception as e:
                logger.error(f"PyMuPDF extraction failed: {str(e)}")
                # Fall back to PyPDF2
        
        # Fall back to PyPDF2
        if self.has_pypdf2:
            try:
                import PyPDF2
                with open(pdf_path, 'rb') as pdf_file:
                    pdf_reader = PyPDF2.PdfReader(pdf_file)
                    for page_num in range(len(pdf_reader.pages)):
                        page = pdf_reader.pages[page_num]
                        text += page.extract_text() + "\n\n"
                return text
            except Exception as e:
                logger.error(f"PyPDF2 extraction failed: {str(e)}")
        
        logger.error(f"Could not extract text from PDF: {pdf_path}")
        return ""
    
    def _extract_docx_text(self, docx_path: str) -> str:
        """
        Extract text from a DOCX file.
        
        Args:
            docx_path: Path to the DOCX file
            
        Returns:
            Extracted text content
        """
        if not self.has_docx:
            logger.error("python-docx not available for DOCX extraction")
            return ""
        
        try:
            import docx
            doc = docx.Document(docx_path)
            text = "\n\n".join([paragraph.text for paragraph in doc.paragraphs])
            return text
        except Exception as e:
            logger.error(f"Error extracting DOCX content: {str(e)}")
            return ""
    
    def _extract_txt_text(self, txt_path: str) -> str:
        """
        Extract text from a plain text file.
        
        Args:
            txt_path: Path to the text file
            
        Returns:
            Extracted text content
        """
        try:
            with open(txt_path, 'r', encoding='utf-8') as file:
                return file.read()
        except UnicodeDecodeError:
            # Try with another encoding if UTF-8 fails
            try:
                with open(txt_path, 'r', encoding='latin-1') as file:
                    return file.read()
            except Exception as e:
                logger.error(f"Error reading text file: {str(e)}")
                return ""
    
    def _create_chunks(self, text: str, metadata: Dict[str, Any]) -> None:
        """
        Divide text into overlapping chunks with metadata.
        
        Args:
            text: The full text to chunk
            metadata: Book metadata to include with each chunk
        """
        # Clean the text
        text = self._clean_text(text)
        
        # Split into paragraphs (better chunks than arbitrary splits)
        paragraphs = [p for p in text.split('\n') if p.strip()]
        
        current_chunk = ""
        for paragraph in paragraphs:
            # If adding this paragraph exceeds chunk size, save current chunk and start new one
            if len(current_chunk) + len(paragraph) > self.chunk_size and current_chunk:
                # Create chunk with metadata
                chunk_data = {
                    "text": current_chunk,
                    "metadata": metadata.copy(),
                    "length": len(current_chunk),
                    "category": "uncategorized"  # Will be categorized later
                }
                self.chunks.append(chunk_data)
                
                # Start new chunk with overlap
                current_chunk = current_chunk[-self.chunk_overlap:] if self.chunk_overlap > 0 else ""
            
            # Add paragraph to current chunk
            if current_chunk:
                current_chunk += "\n\n" + paragraph
            else:
                current_chunk = paragraph
        
        # Add the last chunk if not empty
        if current_chunk:
            chunk_data = {
                "text": current_chunk,
                "metadata": metadata.copy(),
                "length": len(current_chunk),
                "category": "uncategorized"
            }
            self.chunks.append(chunk_data)
    
    def _clean_text(self, text: str) -> str:
        """
        Clean extracted text.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
        """
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page numbers and headers/footers (common in PDFs)
        text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
        
        # Remove ASCII art and non-text elements
        text = re.sub(r'[^\w\s.,;:!?()\[\]{}"\'`-]', '', text)
        
        return text.strip()
    
    def _categorize_chunks(self) -> None:
        """
        Categorize chunks based on content analysis.
        
        This method identifies the educational category of each chunk
        using keyword analysis and pattern matching.
        """
        # Educational categories and their associated keywords
        categories = {
            "classroom_management": [
                "classroom management", "behavior", "discipline", "rules", 
                "routines", "expectations", "consequences", "positive reinforcement"
            ],
            "teaching_methodology": [
                "teaching method", "pedagogy", "instruction", "learning theory",
                "differentiation", "assessment", "curriculum", "lesson plan"
            ],
            "student_development": [
                "development", "cognitive", "social emotional", "psychology",
                "learning style", "motivation", "engagement", "brain development"
            ],
            "subject_content": [
                "mathematics", "reading", "science", "language arts",
                "social studies", "curriculum", "standards", "objectives"
            ],
            "special_education": [
                "special education", "IEP", "accommodation", "modification",
                "learning disability", "inclusion", "intervention", "diverse learners"
            ]
        }
        
        # Categorize each chunk
        for chunk in self.chunks:
            text = chunk["text"].lower()
            scores = {}
            
            # Calculate category scores based on keyword matches
            for category, keywords in categories.items():
                score = sum(1 for keyword in keywords if keyword in text)
                scores[category] = score
            
            # Assign the highest-scoring category
            if any(scores.values()):
                best_category = max(scores.items(), key=lambda x: x[1])[0]
                chunk["category"] = best_category
            else:
                # Keep as uncategorized if no strong match
                chunk["category"] = "general_education"
    
    def get_chunks(self) -> List[Dict[str, Any]]:
        """
        Get all processed chunks.
        
        Returns:
            List of knowledge chunks with metadata
        """
        return self.chunks
    
    def get_chunks_by_category(self, category: str) -> List[Dict[str, Any]]:
        """
        Get chunks filtered by category.
        
        Args:
            category: The category to filter by
            
        Returns:
            Filtered chunks
        """
        return [chunk for chunk in self.chunks if chunk["category"] == category]
    
    def save_chunks_to_file(self, output_path: str) -> None:
        """
        Save all chunks to a JSON file.
        
        Args:
            output_path: Path to save the JSON file
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump({"chunks": self.chunks}, f, indent=2)
            logger.info(f"Saved {len(self.chunks)} chunks to {output_path}")
        except Exception as e:
            logger.error(f"Error saving chunks to file: {str(e)}")
    
    def export_to_csv(self, output_path: str) -> None:
        """
        Export chunks to CSV format.
        
        Args:
            output_path: Path to save the CSV file
        """
        if not self.chunks:
            logger.warning("No chunks to export")
            return
        
        try:
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                # Determine fields from the first chunk
                fields = ["text", "category"]
                fields.extend([f"metadata_{k}" for k in self.chunks[0]["metadata"].keys()])
                
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                
                # Write each chunk as a row
                for chunk in self.chunks:
                    row = {
                        "text": chunk["text"],
                        "category": chunk["category"]
                    }
                    
                    # Flatten metadata
                    for k, v in chunk["metadata"].items():
                        row[f"metadata_{k}"] = v
                    
                    writer.writerow(row)
                
            logger.info(f"Exported {len(self.chunks)} chunks to CSV: {output_path}")
        except Exception as e:
            logger.error(f"Error exporting to CSV: {str(e)}")


if __name__ == "__main__":
    # Example usage
    processor = DocumentProcessor()
    books_dir = os.path.join("knowledge_base", "books")
    
    if os.path.exists(books_dir) and os.listdir(books_dir):
        processor.process_books(books_dir)
        
        # Save processed chunks
        output_dir = os.path.join("knowledge_base", "processed")
        os.makedirs(output_dir, exist_ok=True)
        processor.save_chunks_to_file(os.path.join(output_dir, "knowledge_chunks.json"))
        
        print(f"Processed {len(processor.get_chunks())} chunks from books in {books_dir}")
    else:
        print(f"No books found in {books_dir}") 