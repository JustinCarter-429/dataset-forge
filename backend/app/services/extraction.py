import hashlib
import re
from pathlib import Path
from typing import Protocol
from ..domain.extraction_models import CanonicalExtractedDocument, ExtractionElement, ExtractionElementType, ExtractionStatistics, ExtractionValidation, SourceLocation


class ExtractionError(Exception):
    def __init__(self, code: str, message: str): self.code, self.message = code, message; super().__init__(message)


class DocumentExtractor(Protocol):
    def supports(self, extension: str) -> bool: ...
    def extract(self, file_path: Path, source_file_id: str, source_filename: str, mime_type: str) -> CanonicalExtractedDocument: ...


def _element_id(source_file_id: str, order: int, kind: str, text: str, page: int | None) -> str:
    return hashlib.sha256(f"{source_file_id}:{order}:{kind}:{page}:{text}".encode("utf-8")).hexdigest()[:20]


def _build_document(source_file_id: str, source_filename: str, mime_type: str, extractor: str, version: str | None, elements: list[ExtractionElement], page_count: int | None, warnings: list[str] | None = None) -> CanonicalExtractedDocument:
    meaningful = [element for element in elements if element.text.strip() or element.rows]
    if not meaningful: raise ExtractionError("EMPTY_EXTRACTION", "No meaningful content was extracted from this document.")
    character_count = sum(len(element.text) for element in meaningful)
    word_count = sum(len(re.findall(r"\S+", element.text)) for element in meaningful)
    stats = ExtractionStatistics(pageCount=page_count, characterCount=character_count, wordCount=word_count, elementCount=len(elements), headingCount=sum(e.type == ExtractionElementType.HEADING for e in elements), paragraphCount=sum(e.type == ExtractionElementType.PARAGRAPH for e in elements), listItemCount=sum(e.type == ExtractionElementType.LIST_ITEM for e in elements), tableCount=sum(e.type == ExtractionElementType.TABLE for e in elements), tableRowCount=sum(len(e.rows or []) for e in elements), nonEmptyElementCount=len(meaningful))
    quality = "suspicious" if character_count < 20 else "valid"
    all_warnings = list(warnings or [])
    if quality == "suspicious": all_warnings.append("Extracted content is unusually short; review before generation.")
    validation = ExtractionValidation(valid=True, quality=quality, warnings=all_warnings)
    return CanonicalExtractedDocument(documentId=source_file_id, sourceFileId=source_file_id, sourceFilename=source_filename, mimeType=mime_type, extractor=extractor, extractorVersion=version, statistics=stats, blocks=elements, validation=validation)


class PlainTextExtractor:
    def supports(self, extension: str) -> bool: return extension.lower() == "txt"

    def extract(self, file_path: Path, source_file_id: str, source_filename: str, mime_type: str) -> CanonicalExtractedDocument:
        try:
            raw = file_path.read_bytes()
            try: text = raw.decode("utf-8-sig")
            except UnicodeDecodeError: text = raw.decode("cp1252")
        except (OSError, UnicodeDecodeError) as exc: raise ExtractionError("EXTRACTION_FAILED", "The text document could not be read safely.") from exc
        elements: list[ExtractionElement] = []
        section_path: list[str] = []
        for index, line in enumerate(text.splitlines(), 1):
            if not line.strip(): continue
            kind = ExtractionElementType.HEADING if (len(line) <= 100 and (line.isupper() or line.endswith(":"))) else ExtractionElementType.PARAGRAPH
            if kind == ExtractionElementType.HEADING: section_path = [line.strip()]
            elements.append(ExtractionElement(elementId=_element_id(source_file_id, index, kind.value, line, None), type=kind, text=line, order=index, sectionPath=section_path.copy(), sourceLocation=SourceLocation(lineStart=index, lineEnd=index, locator=f"line:{index}")))
        return _build_document(source_file_id, source_filename, mime_type, "plain_text", "1", elements, None)


class DoclingExtractor:
    def __init__(self): self._converter = None
    def supports(self, extension: str) -> bool: return extension.lower() in {"pdf", "docx"}

    def _get_converter(self):
        if self._converter is None:
            try:
                from docling.datamodel.base_models import InputFormat
                from docling.datamodel.pipeline_options import PdfPipelineOptions
                from docling.datamodel.settings import settings as docling_settings
                from docling.document_converter import DocumentConverter, PdfFormatOption
                docling_settings.inference.compile_torch_models = False
                # Keep Phase 2 deterministic/offline. Layout/table models can be enabled in a later deployment profile.
                pdf_options = PdfPipelineOptions(do_ocr=False, do_table_structure=False)
                self._converter = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_options)})
            except Exception as exc: raise ExtractionError("EXTRACTOR_UNAVAILABLE", "Docling is not available in this environment.") from exc
        return self._converter

    def extract(self, file_path: Path, source_file_id: str, source_filename: str, mime_type: str) -> CanonicalExtractedDocument:
        try: result = self._get_converter().convert(str(file_path))
        except ExtractionError: raise
        except Exception as exc: raise ExtractionError("CORRUPT_DOCUMENT", "Docling could not parse this document.") from exc
        document = result.document
        elements: list[ExtractionElement] = []
        seen_objects: set[int] = set()
        section_path: list[str] = []
        for order, pair in enumerate(document.iterate_items(), 1):
            item = pair[0] if isinstance(pair, tuple) else pair
            if id(item) in seen_objects: continue
            seen_objects.add(id(item))
            label = str(getattr(item, "label", "text")).lower()
            page = None
            provenance = getattr(item, "prov", None) or []
            if provenance: page = getattr(provenance[0], "page_no", None)
            if "table" in label:
                rows: list[list[str]] = []
                try:
                    dataframe = item.export_to_dataframe(document)
                    rows = [[str(value) for value in row] for row in dataframe.fillna("").values.tolist()]
                except Exception:
                    text = str(getattr(item, "text", "") or "")
                    rows = [[cell.strip() for cell in line.split("|")] for line in text.splitlines() if line.strip()]
                table_text = "\n".join(" | ".join(row) for row in rows)
                elements.append(ExtractionElement(elementId=_element_id(source_file_id, order, "table", table_text, page), type=ExtractionElementType.TABLE, text=table_text, order=order, pageNumber=page, sectionPath=section_path.copy(), rows=rows, sourceLocation=SourceLocation(pageNumber=page, locator=f"docling:{order}")))
                continue
            text = str(getattr(item, "text", "") or "").strip()
            if not text: continue
            if "section" in label or "heading" in label: kind = ExtractionElementType.HEADING; section_path = [text]
            elif "list" in label: kind = ExtractionElementType.LIST_ITEM
            elif "caption" in label: kind = ExtractionElementType.CAPTION
            elif "code" in label: kind = ExtractionElementType.CODE_BLOCK
            elif "paragraph" in label: kind = ExtractionElementType.PARAGRAPH
            else: kind = ExtractionElementType.TEXT
            elements.append(ExtractionElement(elementId=_element_id(source_file_id, order, kind.value, text, page), type=kind, text=text, order=order, pageNumber=page, sectionPath=section_path.copy(), sourceLocation=SourceLocation(pageNumber=page, locator=f"docling:{order}")))
        page_count = len(getattr(document, "pages", {}) or {}) or None
        version = None
        try:
            import docling
            version = getattr(docling, "__version__", None)
        except Exception: pass
        return _build_document(source_file_id, source_filename, mime_type, "docling", version, elements, page_count)


class FastPdfExtractor:
    """Local text-first PDF extraction for responsive interactive planning.

    Docling's layout pipeline can spend minutes initializing/processing a
    modest PDF even with OCR and table structure disabled.  This bounded
    extractor uses PyMuPDF blocks plus pdfplumber's deterministic table finder,
    retaining page and table structure without model inference.
    """
    def supports(self, extension: str) -> bool: return extension.lower() == "pdf"

    def extract(self, file_path: Path, source_file_id: str, source_filename: str, mime_type: str) -> CanonicalExtractedDocument:
        try:
            import fitz
            import pdfplumber
            pdf = fitz.open(file_path)
            elements: list[ExtractionElement] = []; section_path: list[str] = []; order = 0
            for page_number, page in enumerate(pdf, 1):
                for block in page.get_text("blocks"):
                    text = str(block[4]).strip()
                    if not text: continue
                    order += 1
                    lines = [line.strip() for line in text.splitlines() if line.strip()]
                    # A short, title-cased or all-caps first line is a useful
                    # deterministic heading signal; no sentence decomposition.
                    heading = len(lines) == 1 and len(text) <= 140 and (text.isupper() or not text.endswith((".", "!", "?")))
                    kind = ExtractionElementType.HEADING if heading else ExtractionElementType.PARAGRAPH
                    if heading: section_path = [text]
                    elements.append(ExtractionElement(elementId=_element_id(source_file_id, order, kind.value, text, page_number), type=kind, text=text, order=order, pageNumber=page_number, sectionPath=section_path.copy(), sourceLocation=SourceLocation(pageNumber=page_number, locator=f"pymupdf:{order}")))
            with pdfplumber.open(file_path) as table_pdf:
                for page_number, page in enumerate(table_pdf.pages, 1):
                    for table in page.extract_tables() or []:
                        rows = [[str(cell or "").strip() for cell in row] for row in table]
                        text = "\n".join(" | ".join(row) for row in rows).strip()
                        if not text: continue
                        order += 1
                        elements.append(ExtractionElement(elementId=_element_id(source_file_id, order, "table", text, page_number), type=ExtractionElementType.TABLE, text=text, order=order, pageNumber=page_number, sectionPath=section_path.copy(), rows=rows, sourceLocation=SourceLocation(pageNumber=page_number, locator=f"pdfplumber:{order}")))
            return _build_document(source_file_id, source_filename, mime_type, "pymupdf_pdfplumber", None, elements, len(pdf))
        except Exception as exc:
            raise ExtractionError("CORRUPT_DOCUMENT", "PDF could not be parsed safely.") from exc


class ExtractionService:
    def __init__(self, extractors: list[DocumentExtractor] | None = None): self.extractors = extractors or [PlainTextExtractor(), FastPdfExtractor(), DoclingExtractor()]
    def extract(self, file_path: Path, source_file_id: str, source_filename: str, mime_type: str) -> CanonicalExtractedDocument:
        extension = file_path.suffix.lower().lstrip(".")
        for extractor in self.extractors:
            if extractor.supports(extension): return extractor.extract(file_path, source_file_id, source_filename, mime_type)
        raise ExtractionError("UNSUPPORTED_DOCUMENT", "This document type is not supported for extraction.")
