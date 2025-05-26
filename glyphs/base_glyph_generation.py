# --- START OF FILE glyphs/base_glyph_generation.py ---
"""
Utility script to generate and save base images for Sacred Geometry
and Platonic Solids using a consistent 'get_base_glyph_elements' method
from each geometry/solid module.
Corrected calls to match defined get_base_glyph_elements signatures.
"""
import logging
import os
import sys
from typing import Tuple, Dict, Any
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PatchCollection # PatchCollection for circles
from matplotlib.patches import Circle, PathPatch # PathPatch for polygons from MplPath
from matplotlib.path import Path as MplPath
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection


# --- Path Setup ---
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_script_dir)
sys.path.insert(0, project_root)
shared_path = os.path.join(project_root, "shared")
if os.path.isdir(shared_path):
    sys.path.insert(0, shared_path)
else:
    # Try to locate 'shared' relative to 'stage_1' if a common structure is used
    stage_1_path = os.path.join(project_root, "stage_1")
    if os.path.isdir(stage_1_path):
        # If 'shared' is a sibling of 'stage_1' (e.g. project_root/shared, project_root/stage_1)
        potential_shared_path = os.path.join(os.path.dirname(stage_1_path), "shared")
        if os.path.isdir(potential_shared_path):
            sys.path.insert(0, potential_shared_path)
        else: # If 'shared' is inside 'stage_1' (e.g. stage_1/shared)
            potential_shared_path_in_stage1 = os.path.join(stage_1_path, "shared")
            if os.path.isdir(potential_shared_path_in_stage1):
                 sys.path.insert(0, potential_shared_path_in_stage1)


try:
    from constants.constants import LOG_LEVEL as ROOT_LOG_LEVEL, LOG_FORMAT as ROOT_LOG_FORMAT
    from glyphs.sigil_constants import (
        SACRED_GEOMETRY_BASES_PATH, PLATONIC_SOLID_BASES_PATH,
        DEFAULT_LINE_COLOR_BASE_IMAGE, TRANSPARENT_BACKGROUND,
        DEFAULT_LINE_WIDTH, DEFAULT_SIGIL_RESOLUTION # For consistency
    )
except ImportError as e:
    print(f"FATAL ERROR: Could not import constants or sigil_constants. Current sys.path: {sys.path}. Details: {e}")
    sys.exit(1)

# --- Import Functions/Classes that provide get_base_glyph_elements ---
try:
    # Sacred Geometry
    from shared.seed_of_life import SeedOfLife
    from shared.flower_of_life import FlowerOfLife
    from shared.fruit_of_life import get_base_glyph_elements as get_fruit_elements
    from shared.tree_of_life import TreeOfLife # Assuming class with the method
    from shared.metatrons_cube import MetatronsCube
    from shared.sri_yantra import SriYantra
    from shared.vesica_piscis import get_base_glyph_elements as get_vesica_elements
    from shared.egg_of_life import get_base_glyph_elements as get_egg_elements
    from shared.germ_of_life import get_base_glyph_elements as get_germ_elements
    from shared.vector_equilibrium import get_base_glyph_elements as get_ve_elements
    from shared.star_tetrahedron import get_base_glyph_elements as get_star_tetra_elements
    from shared.merkaba import Merkaba

    # Platonic Solids
    from platonics.tetrahedron import get_base_glyph_elements as get_tetra_elements
    from platonics.hexahedron import get_base_glyph_elements as get_hexa_elements
    from platonics.octahedron import get_base_glyph_elements as get_octa_elements
    from platonics.dodecahedron import get_base_glyph_elements as get_dodeca_elements
    from platonics.icosahedron import get_base_glyph_elements as get_icosa_elements

except ImportError as e:
    print(f"ERROR: Failed to import one or more geometry/platonic modules for base glyphs. Details: {e}")
    sys.exit(1)


# --- Logger Setup ---
logger = logging.getLogger("BaseGlyphGenerator")
if not logger.handlers:
    logger.setLevel(getattr(logging, ROOT_LOG_LEVEL, logging.INFO))
    formatter = logging.Formatter(ROOT_LOG_FORMAT)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)


def setup_dirs():
    os.makedirs(SACRED_GEOMETRY_BASES_PATH, exist_ok=True)
    os.makedirs(PLATONIC_SOLID_BASES_PATH, exist_ok=True)
    logger.info(f"Asset directories ensured: {SACRED_GEOMETRY_BASES_PATH}, {PLATONIC_SOLID_BASES_PATH}")

def save_plot_as_transparent_glyph(fig, ax, filepath: str, is_3d: bool = False, pad_inches=0.05):
    ax.axis('off')
    if is_3d and hasattr(ax, 'set_zticks'): # Check if it's a 3D axes object
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.xaxis.set_pane_color(TRANSPARENT_BACKGROUND)
        ax.yaxis.set_pane_color(TRANSPARENT_BACKGROUND)
        ax.zaxis.set_pane_color(TRANSPARENT_BACKGROUND)
        ax.grid(False)
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    # Using pad=0 in tight_layout can sometimes crop too much.
    # bbox_inches='tight' with pad_inches in savefig is generally more reliable.
    # plt.tight_layout(pad=0.1) # Adjusted padding slightly
    fig.savefig(filepath, transparent=True, dpi=300, bbox_inches='tight', pad_inches=pad_inches)
    logger.info(f"Saved base glyph: {filepath}")
    plt.close(fig)

def get_line_rgba_color() -> Tuple[float, float, float, float]:
    r, g, b = DEFAULT_LINE_COLOR_BASE_IMAGE # This is (0,0,0)
    return (r/255.0 if r > 1 else r, g/255.0 if g > 1 else g, b/255.0 if b > 1 else b, 1.0)

def draw_elements(ax, elements: Dict[str, Any], is_3d: bool):
    """Draws lines, circles, polygons from the elements dictionary."""
    line_color = get_line_rgba_color()
    line_width = DEFAULT_LINE_WIDTH * 0.75 # Consistent thin lines for base

    if 'lines' in elements and elements['lines']:
        if is_3d:
            line_segments_3d = []
            for p1, p2 in elements['lines']:
                line_segments_3d.append([tuple(p1), tuple(p2)])
            if line_segments_3d:
                lc = Line3DCollection(line_segments_3d, colors=line_color, linewidths=line_width)
                ax.add_collection(lc)
        else: # 2D
            line_segments_2d = []
            for p1, p2 in elements['lines']:
                line_segments_2d.append([tuple(p1), tuple(p2)])
            if line_segments_2d:
                lc = LineCollection(line_segments_2d, colors=line_color, linewidths=line_width)
                ax.add_collection(lc)

    if 'circles' in elements and elements['circles'] and not is_3d:
        patches = []
        for circ_data in elements['circles']:
            patches.append(Circle(circ_data['center'], circ_data['radius'], fill=False,
                                  edgecolor=line_color, linewidth=line_width))
        if patches: ax.add_collection(PatchCollection(patches, match_original=True))

    if 'polygons' in elements and elements['polygons'] and not is_3d:
        patches = []
        for poly_verts in elements['polygons']:
            patches.append(PathPatch(MplPath(poly_verts + [poly_verts[0]]), fill=False, # Close polygon for PathPatch
                                        edgecolor=line_color, linewidth=line_width))
        if patches: ax.add_collection(PatchCollection(patches, match_original=True))
            
    if 'points' in elements and elements['points']: # For bindu points etc.
        points_np = np.array(elements['points'])
        if is_3d and points_np.ndim == 2 and points_np.shape[1] == 3:
            ax.scatter(points_np[:,0], points_np[:,1], points_np[:,2], s=DEFAULT_LINE_WIDTH*5, color=line_color[:3], alpha=0.8)
        elif not is_3d and points_np.ndim == 2 and points_np.shape[1] == 2:
            ax.scatter(points_np[:,0], points_np[:,1], s=DEFAULT_LINE_WIDTH*5, color=line_color[:3], alpha=0.8)


def set_plot_limits(ax, elements: Dict[str, Any], is_3d: bool, default_padding: float = 0.2):
    """Sets plot limits based on bounding_box or content."""
    bbox = elements.get('bounding_box')
    if bbox and all(k in bbox for k in ['xmin','xmax','ymin','ymax']):
        ax.set_xlim([bbox['xmin'], bbox['xmax']])
        ax.set_ylim([bbox['ymin'], bbox['ymax']])
        if is_3d and 'zmin' in bbox and 'zmax' in bbox :
            ax.set_zlim([bbox['zmin'], bbox['zmax']])
            ax.set_box_aspect([bbox['xmax']-bbox['xmin'], bbox['ymax']-bbox['ymin'], bbox['zmax']-bbox['zmin']]) # Attempt aspect ratio
    else:
        ax.autoscale_view() # Fallback to Matplotlib's autoscale
        if is_3d:
            current_lims = [ax.get_xlim(), ax.get_ylim(), ax.get_zlim()]
            ranges = [lim[1]-lim[0] for lim in current_lims]
            max_range = max(ranges) if ranges else 2.0 # Default if no range
            centers = [np.mean(lim) for lim in current_lims]
            ax.set_xlim(centers[0]-max_range/2, centers[0]+max_range/2)
            ax.set_ylim(centers[1]-max_range/2, centers[1]+max_range/2)
            ax.set_zlim(centers[2]-max_range/2, centers[2]+max_range/2)
            ax.set_box_aspect([1,1,1])


    if is_3d: ax.view_init(elev=20, azim=30)


def generate_all_base_images():
    logger.info("Generating all base glyph images...")
    setup_dirs()

    center_3d_param = (0.0, 0.0, 0.0)
    edge_length_param = 1.0
    radius_param = 1.0
    center_2d_param = (0.0, 0.0)
    # This resolution is for the *data generation step* if a geometry module needs it.
    # The actual output image resolution is controlled by savefig's dpi.
    data_gen_resolution_param = 100

    # --- Platonic Solids ---
    platonic_map = {
        "tetrahedron": get_tetra_elements, "hexahedron": get_hexa_elements,
        "octahedron": get_octa_elements, "dodecahedron": get_dodeca_elements,
        "icosahedron": get_icosa_elements,
        "star_tetrahedron": get_star_tetra_elements, # size is edge_length
        "merkaba": Merkaba, # Class
        "vector_equilibrium": get_ve_elements, # radius is dist to vertex
    }
    for name, generator_item in platonic_map.items():
        try:
            logger.debug(f"Generating base Platonic Solid/3D Geom: {name}...")
            elements = None
            if name == "merkaba":
                instance = Merkaba(radius=radius_param) # Merkaba takes radius
                elements = instance.get_base_glyph_elements()
            elif name == "star_tetrahedron":
                 elements = generator_item(center=center_3d_param, size=edge_length_param)
            elif name == "vector_equilibrium":
                 elements = generator_item(center=center_3d_param, radius=radius_param)
            else: # Standard Platonics
                elements = generator_item(center=center_3d_param, edge_length=edge_length_param)

            if not elements: logger.error(f"No elements for {name}."); continue

            is_3d = elements.get('projection_type') == '3d'
            fig = plt.figure(figsize=(5,5))
            ax = fig.add_subplot(111, projection='3d' if is_3d else None)
            if not is_3d: ax.set_aspect('equal')

            draw_elements(ax, elements, is_3d)
            set_plot_limits(ax, elements, is_3d)
            filepath = os.path.join(PLATONIC_SOLID_BASES_PATH, f"{name}_base.png")
            save_plot_as_transparent_glyph(fig, ax, filepath, is_3d=is_3d)
        except Exception as e: logger.error(f"Failed for {name}: {e}", exc_info=True)

    # --- Sacred Geometries (mostly 2D) ---
    sacred_geometry_map = {
        "seed_of_life": SeedOfLife, "flower_of_life": FlowerOfLife,
        "fruit_of_life": get_fruit_elements, "vesica_piscis": get_vesica_elements,
        "egg_of_life": get_egg_elements, "germ_of_life": get_germ_elements,
        "tree_of_life": TreeOfLife, "metatrons_cube": MetatronsCube,
        "sri_yantra": SriYantra
    }
    for name, generator_item in sacred_geometry_map.items():
        try:
            logger.debug(f"Generating base Sacred Geometry: {name}...")
            elements = None
            if isinstance(generator_item, type): # It's a class
                instance = generator_item(radius=radius_param, resolution=data_gen_resolution_param)
                elements = instance.get_base_glyph_elements()
            else: # It's a function
                if name == "vesica_piscis":
                    c1_vp = (-radius_param / 2, 0.0); c2_vp = (radius_param / 2, 0.0)
                    elements = generator_item(center1=c1_vp, center2=c2_vp, radius=radius_param)
                else: # fruit, egg, germ
                    elements = generator_item(center=center_2d_param, radius=radius_param)
            
            if not elements: logger.error(f"No elements for {name}."); continue

            is_3d_sg = elements.get('projection_type') == '3d' # Should be '2d' for these
            fig_sg, ax_sg = plt.subplots(figsize=(5,5))
            if is_3d_sg : ax_sg.remove(); ax_sg = fig_sg.add_subplot(111, projection='3d')
            else: ax_sg.set_aspect('equal')

            draw_elements(ax_sg, elements, is_3d_sg)
            set_plot_limits(ax_sg, elements, is_3d_sg)
            filepath = os.path.join(SACRED_GEOMETRY_BASES_PATH, f"{name}_base.png")
            save_plot_as_transparent_glyph(fig_sg, ax_sg, filepath, is_3d=is_3d_sg)
        except Exception as e: logger.error(f"Failed for {name}: {e}", exc_info=True)

    logger.info("Base glyph image generation process complete.")


if __name__ == "__main__":
    logger.info("=== Starting Base Glyph Image Generation Utility ===")
    generate_all_base_images()
    logger.info("=== Base Glyph Image Generation Finished ===")

# --- END OF FILE glyphs/base_glyph_generation.py ---