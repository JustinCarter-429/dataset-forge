from datetime import datetime, timezone
from enum import Enum
from pydantic import BaseModel, Field, field_validator


class ExtractionElementType(str, Enum):
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST_ITEM = "list_item"
    TABLE = "table"
    CAPTION = "caption"
    CODE_BLOCK = "code_block"
    TEXT = "text"


class SourceLocation(BaseModel):
    page_number: int | None = Field(default=None, alias="pageNumber")
    line_start: int | None = Field(default=None, alias="lineStart")
    line_end: int | None = Field(default=None, alias="lineEnd")
    locator: str | None = None
    model_config = {"populate_by_name": True}


class ExtractionElement(BaseModel):
    element_id: str = Field(alias="elementId")
    type: ExtractionElementType
    text: str = ""
    order: int
    page_number: int | None = Field(default=None, alias="pageNumber")
    section_path: list[str] = Field(default_factory=list, alias="sectionPath")
    source_location: SourceLocation | None = Field(default=None, alias="sourceLocation")
    rows: list[list[str]] | None = None
    model_config = {"populate_by_name": True}


class ExtractionStatistics(BaseModel):
    page_count: int | None = Field(default=None, alias="pageCount")
    character_count: int = Field(alias="characterCount")
    word_count: int = Field(alias="wordCount")
    element_count: int = Field(alias="elementCount")
    heading_count: int = Field(alias="headingCount")
    paragraph_count: int = Field(alias="paragraphCount")
    list_item_count: int = Field(alias="listItemCount")
    table_count: int = Field(alias="tableCount")
    table_row_count: int = Field(alias="tableRowCount")
    non_empty_element_count: int = Field(alias="nonEmptyElementCount")
    model_config = {"populate_by_name": True}

    @field_validator("page_count")
    @classmethod
    def valid_page_count(cls, value):
        if value is not None and value < 0: raise ValueError("page_count cannot be negative")
        return value


class ExtractionValidation(BaseModel):
    valid: bool
    quality: str = "valid"
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class CanonicalExtractedDocument(BaseModel):
    document_id: str = Field(alias="documentId")
    source_file_id: str = Field(alias="sourceFileId")
    source_filename: str = Field(alias="sourceFilename")
    mime_type: str = Field(alias="mimeType")
    extractor: str
    extractor_version: str | None = Field(default=None, alias="extractorVersion")
    extracted_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), alias="extractedAt")
    metadata: dict[str, str] = Field(default_factory=dict)
    statistics: ExtractionStatistics
    elements: list[ExtractionElement] = Field(alias="blocks")
    validation: ExtractionValidation
    model_config = {"populate_by_name": True}
