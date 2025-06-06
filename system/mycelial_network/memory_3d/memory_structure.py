# --- memory_structure.py - Enhanced 3D memory system data structure ---

import logging
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
import uuid

# Configure logging
logger = logging.getLogger("MemoryStructure")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

class MemoryStructure:
    """
    Enhanced definition of the 4-layered 3D memory system structure.
    Includes detailed table structures for all components.
    """
    
    def __init__(self, brain_grid=None):
        """Initialize the memory structure"""
        self.brain_grid = brain_grid
        self.initialized = False
        self.creation_time = datetime.now().isoformat()
        
        # Initialize the database tables/collections
        self._init_tables()
        
        logger.info("Memory structure initialized")
    
    def _init_tables(self):
        """Initialize the database tables/collections structure"""
        # Create empty tables/collections
        
        # --- Primary Node Table ---
        self.nodes = {}  # Primary node table
        
        # --- Level 1: Categories ---
        self.categories = {}  # Domain categories
        self.subcategories = {}  # Subcategories
        
        # --- Level 2: Time-Space ---
        self.coordinates = {}  # 3D coordinates in brain
        self.temporal_refs = {}  # Temporal reference points
        
        # --- Level 3-4: Concepts ---
        self.concepts = {}  # Core concepts
        self.related_concepts = {}  # Related concepts
        self.concept_relationships = {}  # Relationships between concepts
        
        # --- Supporting Tables ---
        self.memory_types = {}  # Memory types with properties
        self.meta_tags = {}  # Meta tags for semantic search
        self.personal_tags = {}  # Personal tags
        self.frequencies = {}  # Frequency assignments
        
        # --- Glyph Tables ---
        self.glyphs = {}  # Glyph base information
        self.glyph_images = {}  # Actual glyph image data
        self.glyph_encodings = {}  # Encoding methods
        self.exif_data = {}  # Surface EXIF data
        self.steganography_data = {}  # Hidden steganography data
        
        # --- Token Tables ---
        self.tokens = {}  # Base token information
        self.token_vectors = {}  # Vector representations
        self.token_similarities = {}  # Semantic similarity information
        self.token_meta_tags = {}  # Meta tags specifically for tokens
        
        # --- Junction Tables ---
        self.node_meta_tags = {}  # Many-to-many node to meta tags
        self.node_personal_tags = {}  # Many-to-many node to personal tags
        self.node_concepts = {}  # Many-to-many node to concepts
    
    def get_table_structure(self):
        """
        Return the complete database schema
        showing the structure of the memory system.
        """
        schema = {
            "node": {
                "node_id": "UUID (PK)",
                "content": "Text/Blob",
                "creation_timestamp": "Timestamp",
                "last_accessed": "Timestamp",
                "last_modified": "Timestamp",
                "access_count": "Integer",
                "is_active": "Boolean",
                "wbs_level_id": "Foreign Key -> wbs-levels",
                "main_category_id": "Foreign Key -> main_category",
                "domain_id": "Foreign Key -> domain",
                "memory_type_id": "Foreign Key -> memory_types",
                "coordinate_id": "Foreign Key -> coordinates",
                "temporal_ref_id": "Foreign Key -> temporal_refs",
                "concept_id": "Foreign Key -> concepts",
                "related_concept_id": "Foreign Key -> related_concepts",
                "frequency_id": "Foreign Key -> frequencies",
                "glyph_id": "Foreign Key -> glyphs",
                "token_id": "Foreign Key -> tokens",
                "current_strength": "Float",
                "is_node": "Boolean",
                "energy_expenditure": "Float",
            },
              "wbs-levels" : {
                "wbs_level_id": "UUID (PK)",
                "level_name": "String",
                "level_description": "Text",
                "level_type": "String",  
                "parent_level_id": "Foreign Key -> wbs-levels (Self)",
            },
            "level_types" : {
                "level_type_id": "UUID (PK)",
                "type_name": "String",
                "type_description": "Text"
            },
            "main_category": {
                "main_category_id": "Integer (PK)",
                "main_category_name": "String",
                "main_category_description": "Text",
                "wbs_level_id": "Foreign Key -> wbs-levels",
                "domain_id": "Foreign Key -> domain",
            },

            "domain": {
                "domain_id": "Integer (PK)",
                "name": "String",
                "description": "Text",
                "wbs_level_id": "Foreign Key -> wbs-levels",
                "memory_type_id": "Foreign Key -> memory_types",
            },

            "memory_types": {
                "memory_type_id": "Integer (PK)",
                "name": "String",
                "decay_rate": "Float",
                "priority": "Float",
                "preferred_region": "String",
                "storage_duration": "String",
                "coordinate_id": "Foreign Key -> coordinates",
                "wbs_level_id": "Foreign Key -> wbs-levels",
            },            
            "coordinates": {
                "coordinate_id": "UUID (PK)",
                "x": "Float",
                "y": "Float",
                "z": "Float",
                "brain_region_id": "Foreign Key -> brain_regions", 
                "brain_subregion": "Foreign Key -> brain_subregions",
                "previous_coordinate_id": "Foreign Key -> coordinates (Self)",
                "temporal_ref_id": "Foreign Key -> temporal_refs",
            },
            
            "temporal_refs": {
                "temporal_ref_id": "UUID (PK)",
                "creation_time": "Timestamp",
                "content_time": "Timestamp",
                "content_year": "Integer",  # Added specific year field
                "last_modified": "Timestamp",
                "chronological_position": "Integer",
            },
            
            "concepts": {
                "concept_id": "UUID (PK)",
                "name": "String",
                "abstraction_level": "Integer",
                "description": "Text",
                "wbs_level_id": "Foreign Key -> wbs-levels",
                "semantic_vector": "Blob",
                "creation_time": "Timestamp",
                "academic_score": "Float",  # Academic credibility (0-1)
                "logical_score": "Float",   # Logical plausibility (0-1)
                "ethical_score": "Float",   # Ethical considerations (0-1)
                "spiritual_score": "Float", # Spiritual significance (0-1)
                "conceptual_score": "Float",# Innovation/hypothetical nature (0-1)
                "consensus_score": "Float", # General agreement level (0-1)
                "personal_significance": "Float", # Personal importance (0-1)
                "universality": "Float",     # How broadly applicable (0-1)
                "domain_id": "Foreign Key -> domain",
            },
            
            "related_concepts": {
                "related_concept_id": "UUID (PK)",
                "concept_id": "Foreign Key -> concepts",
                "name": "String",
                "description": "Text",
                "wbs_level_id": "Foreign Key -> wbs-levels",
                "semantic_vector": "Blob",
                "creation_time": "Timestamp",
                "academic_score": "Float",  # Academic credibility (0-1)
                "logical_score": "Float",   # Logical plausibility (0-1)
                "ethical_score": "Float",   # Ethical considerations (0-1)
                "spiritual_score": "Float", # Spiritual significance (0-1)
                "conceptual_score": "Float",# Innovation/hypothetical nature (0-1)
                "consensus_score": "Float", # General agreement level (0-1)
                "personal_significance": "Float", # Personal importance (0-1)
                "universality": "Float",     # How broadly applicable (0-1)
                "connection_strength": "Float",  # Strength of relationship to parent
                "concept_id": "Foreign Key -> concepts"
            },
            
            "concept_relationships": {
                "relationship_id": "UUID (PK)",
                "source_concept_id": "Foreign Key -> concepts",
                "target_concept_id": "Foreign Key -> concepts OR related_concepts",
                "relationship_type": "String",  # e.g., 'is_a', 'part_of', 'causes', etc.
                "strength": "Float",
                "bidirectional": "Boolean",
                "created_time": "Timestamp"
            },
            
            "meta_tags": {
                "meta_tag_id": "UUID (PK)",
                "name": "String",
                "category": "String",
                "description": "Text"
            },
            
            "personal_tags": {
                "personal_tag_id": "UUID (PK)",
                "name": "String",
                "importance": "Float",
                "emotional_value": "Float",
                "creation_time": "Timestamp"
            },
            
            "frequencies": {
                "frequency_id": "UUID (PK)",
                "base_frequency_hz": "Float",
                "harmonic_pattern": "String",
                "amplitude": "Float",
                "phase": "Float"
            },
            
            # Glyph-related tables
            "glyphs": {
                "glyph_id": "UUID (PK)",
                "name": "String",
                "description": "Text",
                "glyph_type": "String",  # e.g., 'symbol', 'sigil', 'icon', etc.
                "creation_time": "Timestamp",
                "glyph_image_id": "Foreign Key -> glyph_images",
                "encoding_id": "Foreign Key -> glyph_encodings"
            },
            
            "glyph_images": {
                "glyph_image_id": "UUID (PK)",
                "format_type": "String",  # e.g., 'png', 'svg', 'jpg', etc.
                "image_url": "String",    # URL to image if stored externally
                "image_data": "Blob",     # Binary data if stored in database
                "dimensions": "String",   # e.g., '256x256'
                "color_space": "String",  # e.g., 'RGB', 'CMYK', etc.
                "file_size": "Integer"
            },
            
            "glyph_encodings": {
                "encoding_id": "UUID (PK)",
                "encoding_type": "String",  # e.g., 'exif', 'steganography', etc.
                "encoding_method": "String",
                "decoding_method": "String",
                "encryption_key": "String",  # If applicable
                "exif_data_id": "Foreign Key -> exif_data",
                "steg_data_id": "Foreign Key -> steganography_data"
            },
            
            "exif_data": {
                "exif_data_id": "UUID (PK)",
                "data_type": "String",
                "field_name": "String",
                "field_value": "Text",
                "visible": "Boolean"
            },
            
            "steganography_data": {
                "steg_data_id": "UUID (PK)",
                "data_type": "String",
                "hidden_content": "Blob",
                "encryption_level": "Integer",
                "retrieval_key": "String"
            },


            
            # Token-related tables
            "tokens": {
                "token_id": "UUID (PK)",
                "token_text": "String",
                "token_type": "String",  # e.g., 'word', 'subword', 'phrase', etc.
                "language": "String",
                "token_vector_id": "Foreign Key -> token_vectors",
                "creation_time": "Timestamp"
            },
            
            "token_vectors": {
                "token_vector_id": "UUID (PK)",
                "vector_data": "Blob",    # Binary embedding data
                "vector_dimensions": "Integer",
                "embedding_model": "String",
                "normalization": "String"
            },
            
            "token_similarities": {
                "similarity_id": "UUID (PK)",
                "source_token_id": "Foreign Key -> tokens",
                "target_token_id": "Foreign Key -> tokens",
                "similarity_score": "Float",
                "dissimilarity_score": "Float",
                "context_overlap": "Float"
            },
            
            "token_meta_tags": {
                "token_meta_tag_id": "UUID (PK)",
                "token_id": "Foreign Key -> tokens",
                "meta_tag_id": "Foreign Key -> meta_tags",
                "relevance_score": "Float"
            },

            "brain_regions": {
            "brain_region_id": "UUID (PK)",
            "region_name": "String",
            "region_type": "String",  # 'major' or 'sub'
            "parent_region_id": "Foreign Key -> brain_regions (Self)",
            "hemisphere": "String",  # 'left', 'right', or 'both'
            "description": "Text"
            },
            
            # Junction tables
            "node_meta_tags": {
                "node_meta_tag_id": "UUID (PK)",
                "node_id": "Foreign Key -> nodes",
                "meta_tag_id": "Foreign Key -> meta_tags",
                "relevance_score": "Float"
            },
            
            "node_personal_tags": {
                "node_personal_tag_id": "UUID (PK)",
                "node_id": "Foreign Key -> nodes",
                "personal_tag_id": "Foreign Key -> personal_tags",
                "importance_score": "Float"
            },
            
            "node_concepts": {
                "node_concept_id": "UUID (PK)",
                "node_id": "Foreign Key -> nodes",
                "concept_id": "Foreign Key -> concepts",
                "relevance_score": "Float",
                "relationship_type": "String"
            }
        }
        
        return schema
    
    # API methods would remain largely the same, with additions for new tables...
    
    def create_memory_node(self, content, category_id, memory_type_id, 
                        coordinate=None, temporal_ref=None,
                        tags=None, concepts=None, frequency=None, glyph=None) -> str:
        """Define the API for creating a memory node."""
        node_id = str(uuid.uuid4())
        logger.info(f"Memory node creation API called: node_id={node_id}")
        return node_id
    
    def create_concept(self, name, description, academic_score=0.5, 
                    logical_score=0.5, _ethical_score=0.5, _spiritual_score=0.5,
                    _conceptual_score=0.5) -> str:
        """
        Define the API for creating a concept with various scores.
        Parameters prefixed with underscore are stored but not used directly in this method.
        
        Returns:
            str: Concept ID
        """
        concept_id = str(uuid.uuid4())
        logger.info(f"Concept creation API called: concept_id={concept_id}, name={name}")
        # If you want to eliminate warnings, actually use the parameters
        # For example, store them in a variable or log them
        scores = {
            "academic": academic_score,
            "logical": logical_score,
            "ethical": _ethical_score,
            "spiritual": _spiritual_score,
            "conceptual": _conceptual_score
        }
        logger.debug(f"Concept scores: {scores}")
        return concept_id
    
    def create_related_concept(self, parent_concept_id, name, description,
                            academic_score=0.5, logical_score=0.5,
                            ethical_score=0.5, spiritual_score=0.5,
                            conceptual_score=0.5, connection_strength=0.5) -> str:
        """
        Define the API for creating a related concept.
        
        Returns:
            str: Related concept ID
        """
        related_concept_id = str(uuid.uuid4())
        logger.info(f"Related concept creation API called: related_concept_id={related_concept_id}, parent={parent_concept_id}")
        return related_concept_id
    
    def create_glyph(self, name, description, glyph_type, image_data=None, 
                   image_url=None, format_type="png",
                   encoding_type="exif", encoding_method="standard") -> str:
        """
        Define the API for creating a glyph with associated image and encoding.
        
        Returns:
            str: Glyph ID
        """
        glyph_id = str(uuid.uuid4())
        logger.info(f"Glyph creation API called: glyph_id={glyph_id}, name={name}")
        return glyph_id
    
    def add_exif_data(self, glyph_id, field_name, field_value, visible=True) -> bool:
        """
        Define the API for adding EXIF data to a glyph.
        
        Returns:
            bool: Success status
        """
        logger.info(f"Add EXIF data API called: glyph={glyph_id}, field={field_name}")
        return True
    
    def add_steganography_data(self, glyph_id, hidden_content, encryption_level=1) -> bool:
        """
        Define the API for adding steganography data to a glyph.
        
        Returns:
            bool: Success status
        """
        logger.info(f"Add steganography data API called: glyph={glyph_id}")
        return True
    
    def create_token(self, token_text, token_type, language="en",
                   vector_data=None, embedding_model="default") -> str:
        """
        Define the API for creating a token with vector data.
        
        Returns:
            str: Token ID
        """
        token_id = str(uuid.uuid4())
        logger.info(f"Token creation API called: token_id={token_id}, text={token_text}")
        return token_id
    
    def add_token_similarity(self, source_token_id, target_token_id,
                          similarity_score, dissimilarity_score=None) -> bool:
        """
        Define the API for adding similarity data between tokens.
        
        Returns:
            bool: Success status
        """
        logger.info(f"Add token similarity API called: source={source_token_id}, target={target_token_id}")
        return True
    
    def add_token_meta_tag(self, token_id, meta_tag_name, relevance_score=1.0) -> bool:
        """
        Define the API for adding a meta tag to a token.
        
        Returns:
            bool: Success status
        """
        logger.info(f"Add token meta tag API called: token={token_id}, tag={meta_tag_name}")
        return True
    
    # Additional search methods would be implemented here...
    
    def search_by_concept_scores(self, min_academic=None, min_logical=None,
                              min_ethical=None, min_spiritual=None,
                              min_conceptual=None) -> List[str]:
        """
        Define the API for searching concepts by their various scores.
        
        Returns:
            List[str]: List of matching concept IDs
        """
        logger.info(f"Search by concept scores API called")
        return []
    
    def search_by_glyph_encoding(self, encoding_type, encoding_method=None) -> List[str]:
        """
        Define the API for searching glyphs by encoding type and method.
        
        Returns:
            List[str]: List of matching glyph IDs
        """
        logger.info(f"Search by glyph encoding API called: type={encoding_type}")
        return []
    
    def search_tokens_by_similarity(self, token_id, min_similarity=0.7) -> List[str]:
        """
        Define the API for searching tokens by similarity to a reference token.
        
        Returns:
            List[str]: List of matching token IDs
        """
        logger.info(f"Search tokens by similarity API called: reference={token_id}")
        return []
    
    def get_memory_stats(self):
        """
        Get statistics about the memory structure.
        In a real implementation, this would query the database.
        
        Returns:
            Dict: Memory statistics
        """
        return {
            "schema_version": "1.0",
            "tables": list(self.get_table_structure().keys()),
            "relationships": [
                "categories -> subcategories (1:N)",
                "nodes -> categories (N:1)",
                "nodes -> subcategories (N:1)",
                "nodes -> memory_types (N:1)",
                "nodes -> coordinates (N:1)",
                "nodes -> temporal_refs (N:1)",
                "nodes -> frequencies (N:1)",
                "nodes -> glyphs (N:1)",
                "nodes -> tokens (N:1)",
                "nodes <-> meta_tags (N:M)",
                "nodes <-> personal_tags (N:M)",
                "nodes <-> concepts (N:M)",
                "concepts <-> concepts (N:M)",
                "concepts -> related_concepts (1:N)",
                "glyphs -> glyph_images (1:1)",
                "glyphs -> glyph_encodings (1:1)",
                "glyph_encodings -> exif_data (1:1)",
                "glyph_encodings -> steganography_data (1:1)",
                "tokens -> token_vectors (1:1)",
                "tokens <-> tokens via token_similarities (N:M)",
                "tokens <-> meta_tags via token_meta_tags (N:M)"
            ],
            "creation_time": self.creation_time
        }