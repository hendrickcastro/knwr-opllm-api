import logging
import io
from PyPDF2 import PdfReader
from docx import Document
import chardet

def setup_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

def sanitize_input(input_string: str) -> str:
    # Implementa la lógica de sanitización aquí
    return input_string.strip()

def extract_text_from_document(content: bytes, filename: str) -> str:
    if filename.endswith('.pdf'):
        return extract_text_from_pdf(content)
    elif filename.endswith('.docx'):
        return extract_text_from_docx(content)
    elif filename.endswith(('.md', '.py', '.cs', '.js')):
        # Detectar la codificación del archivo
        encoding = chardet.detect(content)['encoding']
        return content.decode(encoding)
    else:
        # Para otros tipos de archivos, intentamos decodificar como UTF-8
        try:
            return content.decode('utf-8')
        except UnicodeDecodeError:
            # Si falla, usamos la detección de codificación
            encoding = chardet.detect(content)['encoding']
            return content.decode(encoding)

def extract_text_from_pdf(content: bytes) -> str:
    pdf = PdfReader(io.BytesIO(content))
    return " ".join(page.extract_text() for page in pdf.pages)

def extract_text_from_docx(content: bytes) -> str:
    doc = Document(io.BytesIO(content))
    return " ".join(paragraph.text for paragraph in doc.paragraphs)