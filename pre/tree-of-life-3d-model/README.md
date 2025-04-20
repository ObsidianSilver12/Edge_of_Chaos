# Tree of Life 3D Modeling Tool

This tool allows you to visualize and explore the connections between Sephiroth on the Tree of Life and their relationship to Platonic solids.

## Features

- Interactive 3D environment
- All 10 Sephiroth nodes with proper positioning
- All 5 Platonic solids with accurate geometric measurements
- Connect any points to identify geometric patterns
- Measure distances and angles between points
- Move objects freely in 3D space
- Precise positioning with numerical inputs

## Getting Started

1. Make sure you have Node.js installed (version 14+ recommended)
2. Navigate to the project directory
3. Install dependencies:

```
npm install
```

4. Start the development server:

```
npm start
```

The app will open in your browser at http://localhost:3000

## How to Use

### Adding Platonic Solids
Click the buttons in the Platonic Solids section to add each type of solid to the scene.

### Moving Objects
Click on any object (Sephiroth node or Platonic solid) to select it. A transform control will appear that allows you to:
- Drag the object freely in 3D space
- Use the position inputs for precise positioning

### Connecting Points
1. Click the "Connect Points" button to enter connect mode
2. Click on a first object to start the connection
3. Click on a second object to complete the connection
4. A line will be drawn between the two objects

### Measuring
1. Click the "Measure" button to enter measure mode
2. Click on two objects to measure the distance between them
3. The measurement will appear in the Measurements section

### Camera Control
- Left-click and drag to rotate the view
- Right-click and drag to pan
- Scroll to zoom in/out

## Finding Platonic Solid Patterns

To identify which Sephiroth form each Platonic solid:

1. Add a Platonic solid to the scene
2. Move it to align its vertices with the appropriate Sephiroth
3. Use the "Connect Points" tool to draw lines between the Sephiroth
4. Use the "Measure" tool to verify distances and angles

The geometric requirements for each Platonic solid are:

- **Tetrahedron (Fire)**: 4 triangular faces, 4 vertices, with all angles at vertices equal to 180°
- **Cube (Earth)**: 6 square faces, 8 vertices, with all angles at vertices equal to 270°
- **Octahedron (Air)**: 8 triangular faces, 6 vertices, with all angles at vertices equal to 240°
- **Icosahedron (Water)**: 20 triangular faces, 12 vertices, with all angles at vertices equal to 300°
- **Dodecahedron (Aether)**: 12 pentagonal faces, 20 vertices, with all angles at vertices equal to 324°

## Tips for Accurate Alignment

- Start with the Platonic solid that seems most obvious in the Tree of Life structure
- Verify your connections by measuring distances - all edges of a Platonic solid should be equal length
- Verify angles at vertices match the expected values for that Platonic solid
- You may need to adjust the positions of objects slightly to account for visualization inaccuracies
- Remember that in 3D space, some structures may be easier to see from certain angles

## Saving Your Work

Currently, this tool doesn't support saving configurations. If you find a good alignment:
- Take screenshots from multiple angles
- Note the coordinates of key points
- Document the connections you've identified

## Troubleshooting

If the app doesn't run properly:
- Make sure all dependencies are installed (`npm install`)
- Check that you're using a modern browser (Chrome, Firefox, Edge recommended)
- If you encounter performance issues, try closing other applications

## Further Development

This tool was created to help explore the geometric relationships between the Sephiroth and Platonic solids. Future enhancements could include:
- Automatic pattern detection
- Configuration saving and loading
- Additional geometric analysis tools
- Export to 3D file formats
