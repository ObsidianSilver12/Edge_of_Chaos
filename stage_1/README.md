# Soul Development Framework - Visualization System

This module provides comprehensive visualization capabilities for the Soul Development Framework, allowing you to monitor and analyze the soul formation process through visual representations of fields, souls, and journey stages.

## Features

- **Integrated with Controllers**: Automatically generate visualizations at key decision points in the soul formation process
- **Field Visualization**: Visualize the VoidField, SephirothFields, and their interactions with souls
- **Soul Visualization**: Visualize soul structure, energy, frequency signatures, and aspects
- **Journey Visualization**: Track the soul's progress through each stage of the formation process
- **Customizable**: Enable/disable visualization for specific stages or components
- **Interactive & Non-interactive**: Display images in interactive environments or save to disk for later analysis

## Directory Structure

```
visualization/
  ├── field_visualization.py       # Field system visualizations
  ├── soul_visualization_enhanced.py # Soul structure visualizations
  ├── soul_journey_visualizer.py   # Journey stage visualizations
  ├── visualization_integration.py # Integration with controllers
  ├── controller_visualization_hooks.py # Hooks for controllers
  └── visualization_main.py        # Standalone visualization tool
```

## Key Visualization Points

The system is designed to visualize the soul at these critical decision points:

1. **After Guff Strengthening**: Check initial energy and stability before proceeding
2. **After Sephiroth Journey**: Verify resonance with the correct Sephirah
3. **After Creator Entanglement**: Confirm connection quality with the Creator
4. **After Harmonic Strengthening**: Check frequency harmony and stabilization
5. **After Life Cord Formation**: Verify cord integrity and field integration
6. **Before Birth**: Review all metrics to ensure readiness for incarnation
7. **After Birth**: Confirm successful incarnation and final soul state

## Integration with Controllers

The visualization system integrates with the existing controllers through non-invasive hooks:

- **FieldController**: Visualize field state and soul-field interactions
- **SoulCompletionController**: Visualize each step of the completion process
- **RootController**: Visualize the entire soul formation flow from start to finish

## Usage

### Automatic Visualization During Processing

```python
# Enable visualization when initializing the controller
field_controller = FieldController(grid_size=GRID_SIZE, enable_visualization=True)

# Or use the visualization system directly
from visualization.visualization_integration import initialize_visualization

# Initialize visualization at the start of the process
viz_manager = initialize_visualization(
    enable=True,       # Master switch to enable/disable visualization
    display=True,      # Whether to display images in interactive environments
    save=True,         # Whether to save images to disk
    stages=["guff", "sephiroth", "birth"]  # Only visualize specific stages
)

# Visualize at specific points
viz_manager.visualize_guff_strengthening(soul, field_controller)
```

### Standalone Visualization of Completed Souls

```bash
# Visualize a single soul
python visualization_main.py --soul output/completed_souls/soul_123.json --stages dashboard journey

# Visualize all souls in a directory
python visualization_main.py --directory output/completed_souls --stages core aspects frequency

# Generate all visualizations
python visualization_main.py --soul output/completed_souls/soul_123.json --stages all
```

## Available Visualizations

### Field Visualizations
- Field energy distribution
- Sephiroth Tree of Life
- Edge of Chaos regions
- Soul-field interactions
- Field frequency spectrum

### Soul Visualizations
- Core structure (3D)
- Energy and harmony (radar)
- Aspect map
- Frequency signature
- Life cord
- Identity attributes

### Journey Visualizations
- Timeline of all journey stages
- Sephiroth journey path
- Creator entanglement
- Birth process
- Mother resonance glyph
- Comprehensive journey dashboard

## Requirements

- Python 3.7+
- Matplotlib
- NumPy
- SciPy

## Installation

1. Ensure the visualization directory is in your Python path
2. Install the required dependencies:

```bash
pip install matplotlib numpy scipy
```

## Additional Notes

- Visualization is optional and can be disabled for production runs or performance reasons
- In non-interactive environments (servers, batch processing), images are saved but not displayed
- Customize output directories for organized storage of visualizations