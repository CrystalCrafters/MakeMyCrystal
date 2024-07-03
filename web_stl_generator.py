# web-stl-generator.py

import os
import numpy as np
import trimesh
from ase.io import read
import pyvista as pv
from cif_reader import get_structure_with_cif, bond_by_nearest_neighbors
from geometry_processor import add_supports, rotate_structure, translate_structure  # Import the support functions


def generate_stl_from_params(file_path, num_unit_cells, rotation_angles, translation_vector, base_level):
    stl_file_path = file_path.replace('.cif', '.stl')
    new_min, new_max = 0.3, 1.5

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

    filtered_radii = {k: v for k, v in atomic_radii.items() if v is not None}
    min_radii = min(filtered_radii.values())
    max_radii = max(filtered_radii.values())

    def scale_radius(radius):
        return new_min + (new_max - new_min) * (radius - min_radii) / (max_radii - min_radii)

    atomic_radii = {atom: scale_radius(radius) if radius is not None else None for atom, radius in atomic_radii.items()}
    bond_radius = 0.15

    def create_sphere(position, radius=0.5, color=(0, 0, 0)):
        sphere = trimesh.creation.icosphere(radius=radius)
        sphere.apply_translation(position)
        sphere.visual.vertex_colors = np.array([color] * len(sphere.vertices), dtype=np.uint8)
        return sphere

    def create_cylinder(start, end, radius=0.1):
        vec = np.array(end) - np.array(start)
        length = np.linalg.norm(vec) + 0.125
        direction = vec / length
        cylinder = trimesh.creation.cylinder(radius=radius, height=length)
        align_matrix = trimesh.geometry.align_vectors([0, 0, 1], direction)
        cylinder.apply_transform(align_matrix)
        midpoint = (np.array(start) + np.array(end)) / 2
        cylinder.apply_translation(midpoint)
        return cylinder

    def atoms_and_bonds_to_mesh(structure):
        atoms_and_bonds_mesh = trimesh.Trimesh()
        for atom in structure:
            atom_radius = atomic_radii[atom['atom_label']]
            atom_sphere = create_sphere(position=atom['cartesian_position'], radius=atom_radius)
            atoms_and_bonds_mesh += atom_sphere

            for connection in atom['connected_atoms']:
                start = atom['cartesian_position']
                end = connection['connected_cartesian_position']
                bond_cylinder = create_cylinder(start, end, radius=bond_radius)
                atoms_and_bonds_mesh += bond_cylinder

        return atoms_and_bonds_mesh

    def export_to_stl(mesh, file_path):
        mesh.export(file_path)

    unique_atoms = get_structure_with_cif(file_path=file_path, num_unit_cells=num_unit_cells)
    unique_atoms = rotate_structure(unique_atoms, rotation_angles)
    unique_atoms = translate_structure(unique_atoms, translation_vector)
    unique_atoms = bond_by_nearest_neighbors(unique_atoms)
    mesh = atoms_and_bonds_to_mesh(unique_atoms)
    mesh = add_supports(mesh, unique_atoms, atomic_radii, base_level=base_level)

    export_to_stl(mesh, stl_file_path)

    return stl_file_path
