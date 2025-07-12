from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import uuid

@dataclass
class BaseModel:
    """Base class for all data models with common functionality"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary for database storage"""
        return {k: v for k, v in self.__dict__.items()}
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BaseModel':
        """Create model instance from dictionary"""
        return cls(**data)
    

@dataclass
class TimestampTypes(BaseModel):
    """Represents the timestamp type which helps determine decay and is used for other searches"""
    timestamp_type_id: int
    timestamp_type_name: str
    memorytype_id: int
    
@dataclass
class MemoryTypes(BaseModel):
    """Represents the memory type which assists in placement of memories in right area of brain and helps determine decay"""
    memory_type_id: int
    content: str
    created_timestamp_id: int
    updated_timestamp_id: int
    encoded_timestamp_id: int
    decoded_timestamp_id: int
    last_access_timestamp_id: int
    expiration_timestamp_id: int
    last_retrieval_timestamp_id: int
    source: str
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
