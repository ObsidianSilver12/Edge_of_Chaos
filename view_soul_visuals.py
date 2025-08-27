#!/usr/bin/env python3
"""
Quick viewer for soul visualizations
Converts HTML files to viewable images and shows file browser
"""

import os
import sys
import webbrowser
from pathlib import Path

def find_latest_soul_visuals():
    """Find the most recent soul visualization directory"""
    visuals_dir = Path("output/visuals")
    
    if not visuals_dir.exists():
        print("❌ No visuals directory found at output/visuals")
        return None
    
    # Find the most recent simulation directory
    sim_dirs = [d for d in visuals_dir.iterdir() if d.is_dir() and d.name.startswith('sim_')]
    
    if not sim_dirs:
        print("❌ No simulation directories found")
        return None
    
    # Sort by modification time, get most recent
    latest_sim = max(sim_dirs, key=lambda d: d.stat().st_mtime)
    
    soul_evolution_dir = latest_sim / "soul_evolution"
    
    if not soul_evolution_dir.exists():
        print(f"❌ No soul_evolution directory found in {latest_sim}")
        return None
    
    return soul_evolution_dir

def list_visualization_files(soul_dir):
    """List all visualization files"""
    print(f"\n📁 Soul Visualizations in: {soul_dir}")
    print("=" * 60)
    
    stages = ['soul_spark', 'creator_entanglement', 'sephiroth_journey', 
              'pre_identity_crystallization', 'post_identity_crystallization']
    
    total_files = 0
    
    for stage in stages:
        stage_dir = soul_dir / stage
        if stage_dir.exists():
            files = list(stage_dir.glob("*.html"))
            png_files = list(stage_dir.glob("*.png"))
            
            print(f"\n🎨 {stage.replace('_', ' ').title()}:")
            if png_files:
                for png_file in png_files:
                    print(f"   📸 {png_file.name}")
                    total_files += 1
            if files:
                for html_file in files:
                    print(f"   🌐 {html_file.name}")
                    total_files += 1
            if not files and not png_files:
                print(f"   ❌ No files found")
    
    # Check for overview files
    overview_files = list(soul_dir.glob("*.html"))
    if overview_files:
        print(f"\n📊 Overview Files:")
        for file in overview_files:
            print(f"   🌐 {file.name}")
            total_files += 1
    
    print(f"\n📈 Total visualization files: {total_files}")
    return total_files

def open_visualizations(soul_dir):
    """Open visualizations in browser"""
    
    # Priority files to open first
    priority_files = [
        "color_evolution.html",
        "evolution_timeline.html", 
        "combined_3d_evolution.html",
        "final_metrics_dashboard.html"
    ]
    
    opened_files = []
    
    # Open priority overview files
    for filename in priority_files:
        file_path = soul_dir / filename
        if file_path.exists():
            print(f"🌐 Opening: {filename}")
            webbrowser.open(f"file://{file_path.absolute()}")
            opened_files.append(filename)
    
    # Open one file from each stage
    stages = ['soul_spark', 'creator_entanglement', 'sephiroth_journey', 
              'pre_identity_crystallization', 'post_identity_crystallization']
    
    for stage in stages:
        stage_dir = soul_dir / stage
        if stage_dir.exists():
            # Prefer 3D model, then frequency spectrum, then overview
            for pattern in ["*3d_soul_model.html", "*frequency_spectrum.html", "*overview.html"]:
                files = list(stage_dir.glob(pattern))
                if files:
                    file_path = files[0]
                    print(f"🎨 Opening: {stage}/{file_path.name}")
                    webbrowser.open(f"file://{file_path.absolute()}")
                    opened_files.append(f"{stage}/{file_path.name}")
                    break
    
    if opened_files:
        print(f"\n✅ Opened {len(opened_files)} visualization files in your browser")
        print("🔍 Check your browser tabs to view the soul evolution visualizations")
    else:
        print("❌ No visualization files found to open")

def view_latest_soul():
    """View the latest soul's visualizations"""
    print("🔍 Looking for latest soul visualizations...")
    
    soul_dir = find_latest_soul_visuals()
    
    if not soul_dir:
        print("\n💡 Tip: Run a soul simulation first to generate visualizations")
        return False
    
    total_files = list_visualization_files(soul_dir)
    
    if total_files == 0:
        print("❌ No visualization files found")
        return False
    
    print(f"\n🎯 Found visualizations in: {soul_dir.name}")
    
    response = input("\n📖 Open visualizations in browser? (y/n): ").lower().strip()
    
    if response in ['y', 'yes', '']:
        open_visualizations(soul_dir)
        return True
    else:
        print("📁 Visualization files are ready to view manually")
        print(f"   Location: {soul_dir}")
        return True

def show_all_souls():
    """Show all available soul visualizations"""
    visuals_dir = Path("output/visuals")
    
    if not visuals_dir.exists():
        print("❌ No visuals directory found")
        return
    
    sim_dirs = [d for d in visuals_dir.iterdir() if d.is_dir() and d.name.startswith('sim_')]
    
    if not sim_dirs:
        print("❌ No soul simulations found")
        return
    
    print(f"\n📚 Available Soul Simulations ({len(sim_dirs)} found):")
    print("=" * 60)
    
    # Sort by modification time (newest first)
    sim_dirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
    
    for i, sim_dir in enumerate(sim_dirs, 1):
        soul_evolution_dir = sim_dir / "soul_evolution"
        
        if soul_evolution_dir.exists():
            files_count = len(list(soul_evolution_dir.rglob("*.html")))
            png_count = len(list(soul_evolution_dir.rglob("*.png")))
            
            print(f"{i:2d}. {sim_dir.name}")
            print(f"    📁 Files: {files_count} HTML, {png_count} PNG")
            print(f"    📅 Modified: {sim_dir.stat().st_mtime}")
            
            # Try to find soul name from metrics
            metrics_file = Path("output/metrics/soul_metrics.json")
            if metrics_file.exists():
                try:
                    import json
                    with open(metrics_file) as f:
                        data = json.load(f)
                        if 'IDENTITY_CRYSTALLIZATION_SUMMARY' in data:
                            soul_data = data['IDENTITY_CRYSTALLIZATION_SUMMARY']
                            if 'final_state' in soul_data:
                                name = soul_data['final_state'].get('name', 'Unknown')
                                print(f"    👤 Soul Name: {name}")
                except:
                    pass
            print()

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='View Soul Visualizations')
    parser.add_argument('--all', action='store_true', help='Show all available souls')
    parser.add_argument('--latest', action='store_true', help='View latest soul (default)')
    
    args = parser.parse_args()
    
    print("👁️  Soul Visualization Viewer")
    print("=" * 40)
    
    if args.all:
        show_all_souls()
    else:
        # Default: view latest
        view_latest_soul()

if __name__ == "__main__":
    main()