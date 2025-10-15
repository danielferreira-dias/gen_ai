from enum import Enum
from dataclasses import dataclass

class PIIType(Enum):
    EMAIL = "email"
    PHONE = "phone"
    SSN = "ssn"

@dataclass
class PIIEntity:
    """Represents a detected PII entity"""
    text: str
    pii_type: PIIType
    start: int
    end: int
    confidence: float = 1.0