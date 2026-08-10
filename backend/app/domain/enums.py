from enum import Enum


class OutputFormat(str, Enum):
    JSON = "json"
    CSV = "csv"


class GenerationStage(str, Enum):
    IDLE = "idle"
    UPLOADING = "uploading"
    VALIDATING_INPUT = "validating_input"
    EXTRACTING = "extracting"
    GENERATING = "generating"
    VALIDATING_DATASET = "validating_dataset"
    EXPORTING = "exporting"
    PACKAGING = "packaging"
    COMPLETE = "complete"
    FAILED = "failed"
