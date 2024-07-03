import numpy as np
import trimesh


def create_base_cylinder(position, base_radius, height, small_cylinder_height=0.5, small_cylinder_radius=0.5):
    main_height = height - small_cylinder_height
    base = trimesh.creation.cylinder(radius=base_radius, height=main_height, sections=32)
    base_translation = np.array([position[0], position[1], position[2] - main_height / 2 - small_cylinder_height])
    base.apply_translation(base_translation)
    small_cylinder = trimesh.creation.cylinder(radius=small_cylinder_radius, height=small_cylinder_height, sections=32)
    small_translation = np.array([position[0], position[1], position[2] - small_cylinder_height / 2])
    small_cylinder.apply_translation(small_translation)
    combined_cylinder = base + small_cylinder
    return combined_cylinder


def create_small_cylinder(position, height, radius):
    cylinder = trimesh.creation.cylinder(radius=radius, height=height, sections=32)
    cylinder.apply_translation([position[0], position[1], position[2] - height / 2])
    return cylinder


def check_atom_between(atom_position, support_position, atom_radius, support_height):
    # Check if there is any atom in the path of the vertical support
    z1 = support_position[2]
    z2 = support_position[2] - support_height
    return z1 > atom_position[2] > z2 and np.linalg.norm(
        np.array(support_position[:2]) - np.array(atom_position[:2])) < atom_radius


def add_supports(atoms_and_bonds_mesh, structure, atomic_radii, base_level=0.0):
    for atom in structure:
        atom_radius = atomic_radii[atom['atom_label']]
        height_to_base = float(atom['cartesian_position'][2]) - float(base_level)

        if height_to_base > 0:
            support_position = atom['cartesian_position']
            support_height = height_to_base

            # Check if any atom is between the support
            atoms_between = [a for a in structure if
                             a != atom and check_atom_between(a['cartesian_position'], support_position,
                                                              atomic_radii[a['atom_label']], support_height)]

            if atoms_between:
                for between_atom in atoms_between:
                    between_position = between_atom['cartesian_position']
                    between_radius = atomic_radii[between_atom['atom_label']]

                    # Add base cylinder up to just before the between_atom
                    base_cylinder = create_base_cylinder(position=support_position,
                                                         base_radius=atom_radius * 0.2,
                                                         height=support_position[2] - between_position[
                                                             2] - between_radius - 0.05,
                                                         small_cylinder_height=atom_radius + 0.03,
                                                         small_cylinder_radius=atom_radius * 0.2)
                    atoms_and_bonds_mesh += base_cylinder

                    # Add small cylinder on top of the between_atom
                    top_small_cylinder = create_small_cylinder(position=[between_position[0], between_position[1], between_position[2]+between_radius+0.05],
                                                               height=0.7,
                                                               radius=between_radius * 0.2)
                    atoms_and_bonds_mesh += top_small_cylinder

                    # Adjust the support position and height for the remaining support
                    support_position = [between_position[0], between_position[1],
                                        between_position[2] - between_radius - 0.01]
                    support_height = support_position[2] - base_level

                # Finally, add the remaining base cylinder from the last adjusted position
                base_cylinder = create_base_cylinder(position=support_position,
                                                     base_radius=atom_radius * 0.2,
                                                     height=support_height,
                                                     small_cylinder_height=atom_radius + 0.03,
                                                     small_cylinder_radius=atom_radius * 0.2)
                atoms_and_bonds_mesh += base_cylinder
            else:
                base_cylinder = create_base_cylinder(position=support_position,
                                                     base_radius=atom_radius * 0.2,
                                                     height=support_height,
                                                     small_cylinder_height=atom_radius + 0.03,
                                                     small_cylinder_radius=atom_radius * 0.2)
                atoms_and_bonds_mesh += base_cylinder

    return atoms_and_bonds_mesh




def translate_structure(atoms_data, translation):
    translated_atoms = []
    for atom in atoms_data:
        new_atom = atom.copy()
        new_atom['cartesian_position'] = [
            atom['cartesian_position'][0] + translation[0],
            atom['cartesian_position'][1] + translation[1],
            atom['cartesian_position'][2] + translation[2]
        ]
        for connected_atom in new_atom['connected_atoms']:
            connected_atom['connected_cartesian_position'] = [
                connected_atom['connected_cartesian_position'][0] + translation[0],
                connected_atom['connected_cartesian_position'][1] + translation[1],
                connected_atom['connected_cartesian_position'][2] + translation[2]
            ]
        translated_atoms.append(new_atom)
    return translated_atoms


def rotate_x(coord, theta):
    theta = np.radians(theta)
    R = np.array([[1, 0, 0],
                  [0, np.cos(theta), -np.sin(theta)],
                  [0, np.sin(theta), np.cos(theta)]])
    return np.dot(R, coord)


def rotate_y(coord, theta):
    theta = np.radians(theta)
    R = np.array([[np.cos(theta), 0, np.sin(theta)],
                  [0, 1, 0],
                  [-np.sin(theta), 0, np.cos(theta)]])
    return np.dot(R, coord)


def rotate_z(coord, theta):
    theta = np.radians(theta)
    R = np.array([[np.cos(theta), -np.sin(theta), 0],
                  [np.sin(theta), np.cos(theta), 0],
                  [0, 0, 1]])
    return np.dot(R, coord)


def rotate_structure(atoms_data, angles):
    rotated_atoms = []
    for atom in atoms_data:
        new_atom = atom.copy()
        current_position = np.array(atom['cartesian_position'])

        # Apply rotations in the order: x, y, z
        current_position = rotate_x(current_position, angles[0])
        current_position = rotate_y(current_position, angles[1])
        current_position = rotate_z(current_position, angles[2])

        new_atom['cartesian_position'] = current_position.tolist()

        for connected_atom in new_atom['connected_atoms']:
            current_connected_position = np.array(connected_atom['connected_cartesian_position'])

            # Apply rotations in the order: x, y, z
            current_connected_position = rotate_x(current_connected_position, angles[0])
            current_connected_position = rotate_y(current_connected_position, angles[1])
            current_connected_position = rotate_z(current_connected_position, angles[2])

            connected_atom['connected_cartesian_position'] = current_connected_position.tolist()

        rotated_atoms.append(new_atom)

    return rotated_atoms


def process_geometry(atoms, cell_parameters, magnetic_spins):
    # Process the atoms and cell parameters to generate geometry
    geometry = []
    for atom in atoms:
        x = atom['x'] * cell_parameters['a']
        y = atom['y'] * cell_parameters['b']
        z = atom['z'] * cell_parameters['c']
        geometry.append((x, y, z, atom['element']))

    processed_spins = []
    if magnetic_spins:
        for i, spin in enumerate(magnetic_spins):
            x = spin['x']
            y = spin['y']
            z = spin['z']
            # Assuming that the spin direction is normalized or needs normalization
            magnitude = np.linalg.norm([x, y, z])
            direction = (x / magnitude, y / magnitude, z / magnitude)
            processed_spins.append((geometry[i][:3], direction))

    return geometry, processed_spins

