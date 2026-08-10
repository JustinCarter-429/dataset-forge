from ..domain.extraction_models import CanonicalExtractedDocument


def analyze_extraction(document: CanonicalExtractedDocument) -> dict[str, object]:
    headings = [element.text for element in document.elements if element.type.value == "heading"]
    return {"sectionCount": len(headings), "headingCount": document.statistics.heading_count, "tableCount": document.statistics.table_count, "availablePageCount": document.statistics.page_count, "contentVolume": document.statistics.character_count, "structureSummary": "Structured content detected" if document.statistics.table_count or headings else "Linear text content", "warnings": document.validation.warnings}
