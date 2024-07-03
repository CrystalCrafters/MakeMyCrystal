import trimesh
import pyvista as pv
from cif_reader import get_structure_with_cif, bond_by_nearest_neighbors
from geometry_processor import add_supports, rotate_structure, translate_structure
import numpy as np
from stl import mesh

# Example 1 usage
file_path = 'Yb2Si2O7.cif'
stl_file_path = 'Yb2Si2O7.stl'
num_unit_cells = [1, 1, 1]  # 3 unit cells in X, 1 in Y and 1 in Z
new_min, new_max = 0.3, 1.5  # Define the new scale

# Specify radius for atoms and bonds
atomic_radii = {
    "H": 53, "He": 31, "Li": 167, "Be": 112, "B": 87,
    "C": 67, "N": 56, "O": 48, "F": 42, "Ne": 38,
    "Na": 190, "Mg": 145, "Al": 118, "Si": 111, "P": 98,
    "S": 87, "Cl": 79, "Ar": 71, "K": 243, "Ca": 194,
    "Sc": 184, "Ti": 176, "V": 171, "Cr": 166, "Mn": 161,
    "Fe": 156, "Co": 152, "Ni": 149, "Cu": 145, "Zn": 142,
    "Ga": 136, "Ge": 125, "As": 114, "Se": 103, "Br": 94,
    "Kr": 87, "Rb": 265, "Sr": 219, "Y": 212, "Zr": 206,
    "Nb": 198, "Mo": 190, "Tc": 183, "Ru": 178, "Rh": 173,
    "Pd": 169, "Ag": 165, "Cd": 161, "In": 156, "Sn": 145,
    "Sb": 133, "Te": 123, "I": 115, "Xe": 108, "Cs": 298,
    "Ba": 253, "La": None, "Ce": None, "Pr": 247, "Nd": 206,
    "Pm": 205, "Sm": 238, "Eu": 231, "Gd": 233, "Tb": 225,
    "Dy": 228, "Ho": 226, "Er": 226, "Tm": 222, "Yb": 222,
    "Lu": 217, "Hf": 208, "Ta": 200, "W": 193, "Re": 188,
    "Os": 185, "Ir": 180, "Pt": 177, "Au": 174, "Hg": 171,
    "Tl": 156, "Pb": 154, "Bi": 143, "Po": 135, "At": 127,
    "Rn": 120, "Fr": None, "Ra": None, "Ac": None, "Th": None,
    "Pa": None, "U": None, "Np": None, "Pu": None, "Am": None,
    "Cm": None, "Bk": None, "Cf": None, "Es": None, "Fm": None,
    "Md": None, "No": None, "Lr": None, "Rf": None, "Db": None,
    "Sg": None, "Bh": None, "Hs": None, "Mt": None, "Ds": None,
    "Rg": None, "Cn": None, "Nh": None, "Fl": None, "Mc": None,
    "Lv": None, "Ts": None, "Og": None, "": 70,
}

# Filter out None values and find the min and max values
filtered_radii = {k: v for k, v in atomic_radii.items() if v is not None}
min_radii = min(filtered_radii.values())
max_radii = max(filtered_radii.values())

# Scale function
def scale_radius(radius):
    return new_min + (new_max - new_min) * (radius - min_radii) / (max_radii - min_radii)

# Apply the scale transformation
atomic_radii = {atom: scale_radius(radius) if radius is not None else None for atom, radius in atomic_radii.items()}
bond_radius = 0.25

def create_sphere(position, radius=0.5, color=(0, 0, 0)):
    sphere = trimesh.creation.icosphere(radius=radius)
    sphere.apply_translation(position)
    sphere.visual.vertex_colors = np.array([color] * len(sphere.vertices), dtype=np.uint8)
    return sphere

def create_cylinder(start, end, radius=0.1):
    vec = np.array(end) - np.array(start)
    length = np.linalg.norm(vec)
    direction = vec / length
    cylinder = trimesh.creation.cylinder(radius=radius, height=length)
    align_matrix = trimesh.geometry.align_vectors([0, 0, 1], direction)
    cylinder.apply_transform(align_matrix)
    midpoint = (np.array(start) + np.array(end)) / 2
    cylinder.apply_translation(midpoint)
    return cylinder

def create_arrow(start, direction, length=2.0, shaft_radius=0.2, tip_radius=0.4, tip_length=0.8):
    end = np.array(start) + np.array(direction) * length
    arrow_body = create_cylinder(start, end, shaft_radius)
    cone = trimesh.creation.cone(radius=tip_radius, height=tip_length)
    cone.apply_translation(end)
    align_matrix = trimesh.geometry.align_vectors([0, 0, 1], direction)
    cone.apply_transform(align_matrix)
    return arrow_body + cone

def atoms_and_bonds_to_mesh(structure):
    atoms_and_bonds_mesh = trimesh.Trimesh()
    magnetic_spins = []

    for atom in structure:
        atom_radius = atomic_radii[atom['atom_label']]
        atom_sphere = create_sphere(position=atom['cartesian_position'], radius=atom_radius)
        atoms_and_bonds_mesh += atom_sphere

        for connection in atom['connected_atoms']:
            start = atom['cartesian_position']
            end = connection['connected_cartesian_position']
            bond_cylinder = create_cylinder(start, end, radius=bond_radius)
            atoms_and_bonds_mesh += bond_cylinder

        if 'magnetic_spin' in atom and atom['magnetic_spin']['direction'] != 'none':
            direction_map = {'up': [0, 0, 1], 'down': [0, 0, -1]}
            direction = direction_map.get(atom['magnetic_spin']['direction'], [0, 0, 0])
            spin_length = atom_radius * 1.5  # Scale the arrow length according to the atom's radius
            spin_shaft_radius = atom_radius * 0.15  # Scale the shaft radius
            spin_tip_radius = atom_radius * 0.3  # Scale the tip radius
            spin_tip_length = atom_radius * 0.3  # Scale the tip length
            magnetic_spins.append((atom['cartesian_position'], direction, spin_length, spin_shaft_radius, spin_tip_radius, spin_tip_length))

    for pos, direction, length, shaft_radius, tip_radius, tip_length in magnetic_spins:
        arrow = create_arrow(pos, direction, length=length, shaft_radius=shaft_radius, tip_radius=tip_radius, tip_length=tip_length)
        atoms_and_bonds_mesh += arrow

    return atoms_and_bonds_mesh

def export_to_stl(mesh, file_path):
    mesh.export(file_path)


unique_atoms = get_structure_with_cif(
    file_path=file_path,
    num_unit_cells=num_unit_cells,
    magnetic_spin_atoms={"Yb": "up"}
)

# Ensure unique_atoms is not None before proceeding
if unique_atoms is not None:
    unique_atoms = rotate_structure(unique_atoms, [0, 0, 0])
    unique_atoms = translate_structure(unique_atoms, [0, 0, 2])
    unique_atoms = rotate_structure(unique_atoms, [10, 10, 0])
    unique_atoms = bond_by_nearest_neighbors(unique_atoms, tolerance=0.1)
    meshh = atoms_and_bonds_to_mesh(unique_atoms)

    # Add supports
    # meshh = add_supports(meshh, unique_atoms, atomic_radii, base_level=0.0)

    pv_mesh = pv.wrap(meshh)
    plotter = pv.Plotter()
    plotter.add_mesh(pv_mesh, color=None)
    plotter.show_axes()  # Show axes in the plot
    plotter.show()

    export_to_stl(meshh, stl_file_path)
else:
    print("Error: unique_atoms is None.")
