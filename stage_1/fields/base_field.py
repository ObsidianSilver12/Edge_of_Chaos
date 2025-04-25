# --- START OF FILE base_field.py ---

"""
Base Field Module

Defines the fundamental field structure that serves as the foundation for all
dimensional fields in the soul development framework.

Author: Soul Development Framework Team
"""

import logging
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
import uuid

# Configure logging
logger = logging.getLogger(__name__)

class BaseField:
    """
    Base class for all fields in the soul development framework.
    Provides the fundamental structure and behaviors for dimensional fields.
    """
    
    def __init__(self, 
                 field_id: Optional[str] = None,
                 name: str = "Base Field",
                 dimensions: Tuple[float, float, float] = (100.0, 100.0, 100.0),
                 base_frequency: float = 432.0,
                 resonance: float = 0.7,
                 stability: float = 0.8,
                 coherence: float = 0.7):
        """
        Initialize a base field with fundamental properties.
        
        Args:
            field_id: Unique identifier for the field
            name: Descriptive name of the field
            dimensions: 3D dimensions of the field (x, y, z)
            base_frequency: The primary frequency of the field in Hz
            resonance: Resonance value (0.0-1.0)
            stability: Stability value (0.0-1.0)
            coherence: Coherence value (0.0-1.0)
            
        Raises:
            ValueError: If any parameters are invalid
            TypeError: If parameters are of incorrect type
        """
        # Validate input parameters
        if dimensions[0] <= 0 or dimensions[1] <= 0 or dimensions[2] <= 0:
            raise ValueError("Field dimensions must be positive values")
        
        if base_frequency <= 0:
            raise ValueError("Base frequency must be positive")
            
        if not 0.0 <= resonance <= 1.0:
            raise ValueError(f"Resonance must be between 0.0 and 1.0, got {resonance}")
            
        if not 0.0 <= stability <= 1.0:
            raise ValueError(f"Stability must be between 0.0 and 1.0, got {stability}")
            
        if not 0.0 <= coherence <= 1.0:
            raise ValueError(f"Coherence must be between 0.0 and 1.0, got {coherence}")
        
        # Core field identifiers
        self.field_id = field_id or str(uuid.uuid4())
        self.name = name
        self.creation_time = datetime.now().isoformat()
        self.field_type = "base"
        
        # Physical properties
        self.dimensions = dimensions
        self.volume = dimensions[0] * dimensions[1] * dimensions[2]
        self.center = (dimensions[0]/2, dimensions[1]/2, dimensions[2]/2)
        self.boundaries = [(0, dimensions[0]), (0, dimensions[1]), (0, dimensions[2])]
        
        # Energy properties
        self.base_frequency = base_frequency
        self.harmonics = self._generate_harmonics(base_frequency)
        self.resonance = resonance
        self.stability = stability
        self.coherence = coherence
        
        # Field state
        self.active = True
        self.entities = []  # List of entities currently in the field
        self.connections = {}  # Connections to other fields
        self.creation_parameters = {
            'field_id': self.field_id,
            'name': self.name,
            'dimensions': self.dimensions,
            'base_frequency': self.base_frequency,
            'resonance': self.resonance,
            'stability': self.stability,
            'coherence': self.coherence
        }
        
        # Field unique properties
        self.field_properties = {}
        
        # Field energy grid (initialized as needed)
        self._energy_grid = None
        
        logger.info(f"Field '{self.name}' (ID: {self.field_id}) initialized with base_frequency={base_frequency}Hz, stability={stability:.2f}")
    
    def _generate_harmonics(self, base_frequency: float) -> List[float]:
        """
        Generate harmonic frequencies based on the field's base frequency.
        
        Args:
            base_frequency: The base frequency to generate harmonics from
            
        Returns:
            List of harmonic frequencies
            
        Raises:
            ValueError: If base_frequency is invalid
        """
        if base_frequency <= 0:
            raise ValueError("Cannot generate harmonics from non-positive frequency")
            
        # Generate first 7 harmonics
        return [base_frequency * (i + 1) for i in range(7)]
    
    def initialize_energy_grid(self, resolution: Tuple[int, int, int] = (10, 10, 10)) -> np.ndarray:
        """
        Initialize the field's energy grid with the specified resolution.
        
        Args:
            resolution: Number of points along each dimension
            
        Returns:
            The initialized energy grid
            
        Raises:
            ValueError: If resolution is invalid
        """
        if any(r <= 0 for r in resolution):
            raise ValueError("Grid resolution must be positive in all dimensions")
            
        # Create a baseline energy grid
        self._energy_grid = np.ones(resolution) * self.stability
        
        # Add coherence patterns
        x, y, z = np.indices(resolution)
        x_norm = x / resolution[0]
        y_norm = y / resolution[1]
        z_norm = z / resolution[2]
        
        # Create a standing wave pattern based on coherence
        pattern = np.sin(2 * np.pi * x_norm) * np.sin(2 * np.pi * y_norm) * np.sin(2 * np.pi * z_norm)
        
        # Apply coherence to the pattern intensity
        self._energy_grid += pattern * self.coherence * 0.3
        
        # Add resonance patterns
        resonance_pattern = np.sin(4 * np.pi * x_norm * self.resonance) * \
                           np.sin(4 * np.pi * y_norm * self.resonance) * \
                           np.sin(4 * np.pi * z_norm * self.resonance)
        self._energy_grid += resonance_pattern * self.resonance * 0.2
        
        # Ensure values stay in valid range
        self._energy_grid = np.clip(self._energy_grid, 0.0, 1.0)
        
        return self._energy_grid
    
    def get_energy_at_point(self, point: Tuple[float, float, float]) -> float:
        """
        Get the energy level at a specific point in the field.
        
        Args:
            point: (x, y, z) coordinates
            
        Returns:
            Energy level at the specified point
            
        Raises:
            ValueError: If point is outside field boundaries
            RuntimeError: If energy grid is not initialized
        """
        # Validate point is within boundaries
        for i, (coord, (min_bound, max_bound)) in enumerate(zip(point, self.boundaries)):
            if coord < min_bound or coord > max_bound:
                raise ValueError(f"Coordinate {coord} at dimension {i} is outside field boundaries ({min_bound}, {max_bound})")
        
        if self._energy_grid is None:
            raise RuntimeError("Energy grid not initialized. Call initialize_energy_grid() first")
            
        # Convert absolute coordinates to grid indices
        grid_shape = self._energy_grid.shape
        x_idx = int(point[0] / self.dimensions[0] * grid_shape[0])
        y_idx = int(point[1] / self.dimensions[1] * grid_shape[1])
        z_idx = int(point[2] / self.dimensions[2] * grid_shape[2])
        
        # Ensure indices are in bounds
        x_idx = min(max(0, x_idx), grid_shape[0] - 1)
        y_idx = min(max(0, y_idx), grid_shape[1] - 1)
        z_idx = min(max(0, z_idx), grid_shape[2] - 1)
        
        return float(self._energy_grid[x_idx, y_idx, z_idx])
    
    def add_entity(self, entity_id: str, position: Tuple[float, float, float]) -> bool:
        """
        Add an entity to the field at the specified position.
        
        Args:
            entity_id: Unique identifier for the entity
            position: (x, y, z) position in the field
            
        Returns:
            True if entity was added successfully
            
        Raises:
            ValueError: If position is invalid or entity already exists
        """
        # Validate position is within field boundaries
        for i, (coord, (min_bound, max_bound)) in enumerate(zip(position, self.boundaries)):
            if coord < min_bound or coord > max_bound:
                raise ValueError(f"Entity position {position} is outside field boundaries")
        
        # Check if entity already exists in this field
        if any(e['id'] == entity_id for e in self.entities):
            raise ValueError(f"Entity {entity_id} already exists in field {self.field_id}")
            
        # Add entity
        self.entities.append({
            'id': entity_id,
            'position': position,
            'entry_time': datetime.now().isoformat()
        })
        
        logger.info(f"Entity {entity_id} added to field {self.name} at position {position}")
        return True
    
    def remove_entity(self, entity_id: str) -> bool:
        """
        Remove an entity from the field.
        
        Args:
            entity_id: ID of entity to remove
            
        Returns:
            True if entity was removed
            
        Raises:
            ValueError: If entity does not exist in this field
        """
        # Find entity
        entity_index = None
        for i, entity in enumerate(self.entities):
            if entity['id'] == entity_id:
                entity_index = i
                break
                
        if entity_index is None:
            raise ValueError(f"Entity {entity_id} not found in field {self.field_id}")
            
        # Remove entity
        removed_entity = self.entities.pop(entity_index)
        logger.info(f"Entity {entity_id} removed from field {self.name}")
        
        return True
    
    def connect_field(self, target_field_id: str, connection_type: str, connection_strength: float) -> bool:
        """
        Establish a connection to another field.
        
        Args:
            target_field_id: ID of field to connect to
            connection_type: Type of connection (e.g., "portal", "overlap")
            connection_strength: Strength of connection (0.0-1.0)
            
        Returns:
            True if connection was established
            
        Raises:
            ValueError: If parameters are invalid
        """
        if not 0.0 <= connection_strength <= 1.0:
            raise ValueError(f"Connection strength must be between 0.0 and 1.0, got {connection_strength}")
            
        # Create the connection
        self.connections[target_field_id] = {
            'type': connection_type,
            'strength': connection_strength,
            'established_time': datetime.now().isoformat()
        }
        
        logger.info(f"Connected field {self.name} to field {target_field_id} with {connection_type} connection (strength: {connection_strength:.2f})")
        return True
    
    def disconnect_field(self, target_field_id: str) -> bool:
        """
        Remove a connection to another field.
        
        Args:
            target_field_id: ID of field to disconnect
            
        Returns:
            True if connection was removed
            
        Raises:
            ValueError: If connection does not exist
        """
        if target_field_id not in self.connections:
            raise ValueError(f"No connection exists to field {target_field_id}")
            
        # Remove the connection
        del self.connections[target_field_id]
        
        logger.info(f"Disconnected field {self.name} from field {target_field_id}")
        return True
    
    def get_field_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive metrics about the field's current state.
        
        Returns:
            Dictionary of field metrics
        """
        # Calculate derived metrics
        entity_count = len(self.entities)
        connection_count = len(self.connections)
        
        # Energy distribution metrics
        energy_metrics = {}
        if self._energy_grid is not None:
            energy_metrics = {
                'mean_energy': float(np.mean(self._energy_grid)),
                'min_energy': float(np.min(self._energy_grid)),
                'max_energy': float(np.max(self._energy_grid)),
                'std_energy': float(np.std(self._energy_grid))
            }
        
        # Create metrics dictionary
        metrics = {
            'field_id': self.field_id,
            'name': self.name,
            'field_type': self.field_type,
            'active': self.active,
            'creation_time': self.creation_time,
            'current_time': datetime.now().isoformat(),
            
            # Physical properties
            'dimensions': self.dimensions,
            'volume': self.volume,
            
            # Energy properties
            'base_frequency': self.base_frequency,
            'resonance': self.resonance,
            'stability': self.stability,
            'coherence': self.coherence,
            'harmonics': self.harmonics,
            
            # Entity and connection metrics
            'entity_count': entity_count,
            'connection_count': connection_count,
            
            # Energy metrics if available
            'energy_metrics': energy_metrics
        }
        
        return metrics
    
    def calculate_field_resonance(self, external_frequency: float) -> float:
        """
        Calculate how strongly the field resonates with an external frequency.
        
        Args:
            external_frequency: Frequency to test resonance against
            
        Returns:
            Resonance value between 0.0 and 1.0
            
        Raises:
            ValueError: If external_frequency is invalid
        """
        if external_frequency <= 0:
            raise ValueError("External frequency must be positive")
        
        # Check resonance against base frequency
        base_resonance = 1.0 - min(1.0, abs(self.base_frequency - external_frequency) / self.base_frequency)
        
        # Check resonance against harmonics
        harmonic_resonances = []
        for harmonic in self.harmonics:
            ratio = min(harmonic / external_frequency, external_frequency / harmonic)
            if ratio > 0.9:  # Close to 1:1 ratio
                harmonic_resonances.append(ratio)
                
        # Calculate overall resonance
        if harmonic_resonances:
            harmonic_resonance = max(harmonic_resonances)
            overall_resonance = 0.7 * base_resonance + 0.3 * harmonic_resonance
        else:
            overall_resonance = base_resonance
            
        # Modify by field's inherent resonance property
        final_resonance = overall_resonance * self.resonance
        
        return float(min(1.0, max(0.0, final_resonance)))
    
    def modify_field_property(self, property_name: str, new_value: Any) -> bool:
        """
        Modify a core field property safely.
        
        Args:
            property_name: Name of property to modify
            new_value: New value for the property
            
        Returns:
            True if property was modified
            
        Raises:
            ValueError: If property_name is invalid or new_value is invalid
            AttributeError: If property doesn't exist
        """
        allowed_properties = ['name', 'resonance', 'stability', 'coherence', 'base_frequency', 'active']
        
        if property_name not in allowed_properties:
            raise ValueError(f"Cannot modify property '{property_name}'. Allowed properties: {allowed_properties}")
        
        # Validate new value
        if property_name in ['resonance', 'stability', 'coherence'] and not (0.0 <= new_value <= 1.0):
            raise ValueError(f"Property '{property_name}' must be between 0.0 and 1.0, got {new_value}")
            
        if property_name == 'base_frequency' and new_value <= 0:
            raise ValueError("Base frequency must be positive")
            
        if property_name == 'active' and not isinstance(new_value, bool):
            raise ValueError("Property 'active' must be a boolean")
            
        # Update property
        old_value = getattr(self, property_name)
        setattr(self, property_name, new_value)
        
        # Update dependent properties if necessary
        if property_name == 'base_frequency':
            self.harmonics = self._generate_harmonics(new_value)
            
        logger.info(f"Field property '{property_name}' changed from {old_value} to {new_value} for field {self.name}")
        return True
    
    def __str__(self) -> str:
        """String representation of the field."""
        return f"BaseField(name={self.name}, id={self.field_id}, freq={self.base_frequency}Hz, stability={self.stability:.2f})"
    
    def __repr__(self) -> str:
        """Detailed representation."""
        return f"<BaseField id='{self.field_id}' name='{self.name}' type='{self.field_type}' freq={self.base_frequency}Hz stability={self.stability:.2f}>"

# --- END OF FILE base_field.py ---