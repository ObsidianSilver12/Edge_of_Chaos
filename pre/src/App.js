import React, { useState, useEffect, useRef } from 'react';
import * as THREE from 'three';
import * as d3 from 'd3';

function App() {
  const canvasRef = useRef(null);
  const [gridSize, setGridSize] = useState(50);
  const [sephiroth, setSephiroth] = useState([]);
  const [energyNodes, setEnergyNodes] = useState([]);
  const [lines, setLines] = useState([]);
  const [selectedNode, setSelectedNode] = useState(null);
  const [drawMode, setDrawMode] = useState(false);
  const [energyMode, setEnergyMode] = useState(false);
  const [lineStart, setLineStart] = useState(null);
  const [currentColor, setCurrentColor] = useState('#fff');
  const [textEditMode, setTextEditMode] = useState(false);

  useEffect(() => {
    if (sephiroth.length === 0) {
      const BLOCK_HEIGHT = gridSize;
      const gridBlocksWide = Math.floor(window.innerWidth / gridSize);
      const bottomCenterX = Math.floor(gridBlocksWide / 2) * gridSize + 25;
      const bottomY = window.innerHeight - BLOCK_HEIGHT + 25;
      
      setSephiroth([
        {
          x: bottomCenterX,
          y: bottomY - (15 * gridSize),  // 15 blocks up from bottom
          name: 'Kether'
        },
        {
          x: bottomCenterX,
          y: bottomY - (12 * gridSize),  // 12 blocks up
          name: 'Daath'
        },
        {
          x: bottomCenterX,
          y: bottomY - (8 * gridSize),   // 8 blocks up
          name: 'Tiphareth'
        },
        {
          x: bottomCenterX,
          y: bottomY - (2 * gridSize),   // 2 blocks up
          name: 'Yesod'
        },
        {
          x: bottomCenterX,
          y: bottomY,                    // At bottom
          name: 'Malkuth'
        }
      ]);
    }
  }, [gridSize]);
  
  
  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;

    function drawGrid() {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      
      ctx.beginPath();
      ctx.strokeStyle = '#333';
      
      for(let x = 0; x <= canvas.width; x += gridSize) {
        ctx.moveTo(x, 0);
        ctx.lineTo(x, canvas.height);
      }
      
      for(let y = 0; y <= canvas.height; y += gridSize) {
        ctx.moveTo(0, y);
        ctx.lineTo(canvas.width, y);
      }
      ctx.stroke();

      // Draw lines connecting to center of nodes
      lines.forEach(line => {
        ctx.beginPath();
        ctx.strokeStyle = line.color;
        ctx.lineWidth = 2;
        ctx.moveTo(line.start.x, line.start.y);
        ctx.lineTo(line.end.x, line.end.y);
        ctx.stroke();
      });

      // Draw Sephiroth nodes
      sephiroth.forEach(node => {
        ctx.beginPath();
        ctx.fillStyle = node === selectedNode ? '#ff0' : '#fff';
        ctx.arc(node.x, node.y, 10, 0, Math.PI * 2);
        ctx.fill();
        ctx.fillStyle = '#fff';
        ctx.font = '16px Arial';
        ctx.fillText(node.name, node.x + 15, node.y);
      });

      // Draw Energy nodes
      energyNodes.forEach(node => {
        ctx.beginPath();
        ctx.fillStyle = node === selectedNode ? '#ff0' : '#ff69b4';
        ctx.arc(node.x, node.y, 8, 0, Math.PI * 2);
        ctx.fill();
      });
    }

    drawGrid();
  }, [sephiroth, energyNodes, lines, selectedNode, gridSize]);

  const snapToGrid = (x, y) => {
    return {
      x: Math.round(x / gridSize) * gridSize,
      y: Math.round(y / gridSize) * gridSize
    };
  };

  const handleNodeClick = (node) => {
    if (textEditMode) {
      const newLabel = prompt('Enter new name for this Sephirah:', node.name);
      if (newLabel) {
        setSephiroth(sephiroth.map(s => 
          s === node ? {...s, name: newLabel} : s
        ));
      }
    }
  };

  const handleCanvasClick = (e) => {
    const rect = canvasRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    
    const clickedSephirah = sephiroth.find(node => {
      const distance = Math.sqrt(
        Math.pow(node.x - x, 2) + 
        Math.pow(node.y - y, 2)
      );
      return distance < 10;
    });

    const clickedEnergyNode = energyNodes.find(node => {
      const distance = Math.sqrt(
        Math.pow(node.x - x, 2) + 
        Math.pow(node.y - y, 2)
      );
      return distance < 8;
    });

    if (textEditMode && clickedSephirah) {
      handleNodeClick(clickedSephirah);
      return;
    }

    const snapped = snapToGrid(x, y);

    if (!textEditMode) {
      if (drawMode) {
        if (!lineStart) {
          setLineStart(clickedSephirah || clickedEnergyNode || {x, y});
        } else {
          const endNode = clickedSephirah || clickedEnergyNode || {x, y};
          if (endNode !== lineStart) {
            setLines([...lines, {
              start: lineStart,
              end: endNode,
              color: energyMode ? '#ff69b4' : currentColor,
              length: Math.sqrt(
                Math.pow(lineStart.x - endNode.x, 2) + 
                Math.pow(lineStart.y - endNode.y, 2)
              ) / gridSize
            }]);
          }
          setLineStart(null);
        }
      } else if (energyMode) {
        setEnergyNodes([...energyNodes, {
          x: snapped.x,
          y: snapped.y
        }]);
      } else if (sephiroth.length > 2) {
        setSephiroth([...sephiroth, {
          x: snapped.x,
          y: snapped.y,
          name: `Sephirah ${sephiroth.length + 1}`
        }]);
      }
    }
  };

  const handleUndo = () => {
    if (lines.length > 0) {
      setLines(lines.slice(0, -1));
    } else if (energyNodes.length > 0) {
      setEnergyNodes(energyNodes.slice(0, -1));
    } else if (sephiroth.length > 2) {
      setSephiroth(sephiroth.slice(0, -1));
    }
  };

  return (
    <div>
      <canvas 
        ref={canvasRef}
        onClick={handleCanvasClick}
        style={{background: '#000'}}
      />
      <div style={{
        position: 'absolute', 
        top: 10, 
        left: 10, 
        color: 'white',
        fontSize: '16px'
      }}>
        <div>
          <label>Grid Size: </label>
          <input 
            type="number" 
            value={gridSize}
            onChange={(e) => setGridSize(Number(e.target.value))}
            min="10"
            max="100"
            style={{fontSize: '16px'}}
          />
        </div>
        <button 
          onClick={() => {
            setDrawMode(!drawMode);
            setEnergyMode(false);
          }}
          style={{fontSize: '16px', margin: '5px'}}
        >
          {drawMode ? 'Place Nodes' : 'Draw Lines'}
        </button>
        <button 
          onClick={() => {
            setEnergyMode(!energyMode);
            setDrawMode(false);
          }}
          style={{fontSize: '16px', margin: '5px'}}
        >
          {energyMode ? 'Place Sephiroth' : 'Place Energy Nodes'}
        </button>
        <button 
          onClick={() => setTextEditMode(!textEditMode)}
          style={{fontSize: '16px', margin: '5px'}}
        >
          {textEditMode ? 'Exit Text Edit' : 'Edit Text'}
        </button>
        <div>
          <button onClick={() => setCurrentColor('#ffff00')} style={{fontSize: '16px', margin: '5px'}}>Yellow</button>
          <button onClick={() => setCurrentColor('#87CEEB')} style={{fontSize: '16px', margin: '5px'}}>Light Blue</button>
          <button onClick={() => setCurrentColor('#ff0000')} style={{fontSize: '16px', margin: '5px'}}>Red</button>
          <button onClick={() => setCurrentColor('#ffffff')} style={{fontSize: '16px', margin: '5px'}}>White</button>
          <button onClick={() => setCurrentColor('#DEB887')} style={{fontSize: '16px', margin: '5px'}}>Light Brown</button>
          <button onClick={handleUndo} style={{fontSize: '16px', margin: '5px'}}>Undo</button>
        </div>
        <div>Current Color: <div style={{
          display: 'inline-block',
          width: '20px',
          height: '20px',
          backgroundColor: currentColor,
          border: '1px solid white',
          verticalAlign: 'middle'
        }}></div></div>
        <div>
          <h3>Measurements</h3>
          {lines.map((line, i) => (
            <div key={i}>
              Line {i + 1}: {line.length.toFixed(2)} units
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

export default App;






