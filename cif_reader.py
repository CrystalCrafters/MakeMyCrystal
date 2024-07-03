from pymatgen.io.cif import CifParser
from pymatgen.analysis.local_env import CrystalNN
from pymatgen.analysis.graphs import StructureGraph
from mp_api.client import MPRester
import numpy as np

def get_structure_with_cif(file_path, num_unit_cells=None, is_primitive=False, target_atoms=None,
                           magnetic_spin_atoms=None, site_index_spin=None):
    # Set default number of unit cells to 1x1x1 if none specified
    if num_unit_cells is None:
        num_unit_cells = [1, 1, 1]

    # Initialize CIF parser and parse the structure, optionally as primitive
    parser = CifParser(file_path)
    structure = parser.parse_structures(primitive=is_primitive)[0]

    # Convert unit cells numbers to integers and prepare for expansion
    x_unit_cell = int(np.ceil(num_unit_cells[0]))
    y_unit_cell = int(np.ceil(num_unit_cells[1]))
    z_unit_cell = int(np.ceil(num_unit_cells[2]))

    # Initialize empty list to store unique atoms' information
    unique_atoms = []
    # Retrieve lattice matrix from the structure for Cartesian transformations
    lattice = structure.lattice.matrix

    # Use CrystalNN to define a strategy for local environment analysis
    nn = CrystalNN()
    # Create a graph of the structure based on local environment
    graph = StructureGraph.from_local_env_strategy(structure, nn)

    # Loop over each unit cell in each direction
    for nx in range(x_unit_cell):
        for ny in range(y_unit_cell):
            for nz in range(z_unit_cell):
                # Loop over each site in the structure
                for idx, site in enumerate(structure):
                    # Calculate new fractional coordinates for the site in the extended unit cell
                    fractional_coords = site.frac_coords + np.array([nx, ny, nz])
                    # Check if the site's fractional coordinates are within the desired range
                    if all(fractional_coords[i] < num_unit_cells[i] for i in range(3)):
                        # Get the element symbol
                        atom_label = site.species_string[:-2]
                        # Only proceed if no target atoms specified or the current atom is a target atom
                        if target_atoms is None or atom_label in target_atoms:
                            # Store detailed information about the atom
                            atom_info = {
                                "atom_label": atom_label,
                                "oxi_atom_label": site.species_string,
                                "fractional_position": fractional_coords.tolist(),
                                "cartesian_position": (site.coords + np.dot([nx, ny, nz], lattice)).tolist(),
                                "connected_atoms": [],
                                "magnetic_spin": {},
                                "site_index": idx
                            }
                            # Get connected sites from the graph
                            connected_sites = graph.get_connected_sites(idx)
                            for connected_site in connected_sites:
                                # Calculate the connected site's fractional and Cartesian coordinates
                                connected_fractional_coords = connected_site.site.frac_coords + np.array([nx, ny, nz])
                                connection = {
                                    "connected_to": connected_site.site.species_string,
                                    "bond_length": connected_site.weight,
                                    "connected_fractional_position": connected_fractional_coords.tolist(),
                                    "connected_cartesian_position": (
                                            connected_site.site.coords + np.dot([nx, ny, nz], lattice)).tolist(),
                                    "site_index": connected_site.index
                                }
                                # Append connection information to the atom info
                                atom_info["connected_atoms"].append(connection)
                            # Add the atom's detailed info to the unique_atoms list
                            unique_atoms.append(atom_info)

    # Add magnetic spin information if provided
    if magnetic_spin_atoms or site_index_spin:
        unique_atoms = add_magnetic_spin_info(unique_atoms, magnetic_spin_atoms, site_index_spin)

    # Return the list of unique atoms with their full structural and connection details
    return unique_atoms

def add_magnetic_spin_info(unique_atoms, magnetic_spin_atoms=None, site_index_spin=None):
    """
    Adds magnetic spin information to the unique_atoms list.

    unique_atoms:
        A list of dictionaries containing atom information.

    magnetic_spin_atoms:
        A dictionary where keys are atom labels and values are the direction of the magnetic spin.
        Example: {"Fe": "up", "Ni": "down"}

    site_index_spin:
        A dictionary where keys are site indices and values are the direction of the magnetic spin.
        Example: {0: "up", 1: "down"}
    """
    for atom in unique_atoms:
        atom_label = atom['atom_label']
        site_index = atom['site_index']
        if site_index_spin and site_index in site_index_spin:
            atom['magnetic_spin'] = {"direction": site_index_spin[site_index]}
        elif magnetic_spin_atoms and atom_label in magnetic_spin_atoms:
            atom['magnetic_spin'] = {"direction": magnetic_spin_atoms[atom_label]}
        else:
            atom['magnetic_spin'] = {"direction": "none"}
    return unique_atoms

# Define an asynchronous function to fetch material summaries from the Materials Project database
async def fetch_materials(**kwargs):
    """
    Asynchronously fetch materials data from the Materials Project database.

    Args:
        **kwargs: Arbitrary keyword arguments passed to the Materials Project search API.

    Returns:
        A list of dictionaries containing materials data if successful, or an error message.

    Raises:
        Exception: If there's an issue with fetching data from the API.
    """
    # Specify your API key here (replace the dummy key with your actual API key)
    api_key = "cSFVj0Awg9nQlro7yWhYacD4TRst78YZ"

    try:
        # Initialize the MPRester client with your API key in an asynchronous context
        with MPRester(api_key) as mpr:
            # Perform a search query on the Materials Project using provided parameters
            results = mpr.summary.search(**kwargs)
            # Check if the search returns any results
            if not results:
                return "No data found with the given search parameters."
            # If results are found, return them
            return results
    except Exception as e:
        # Return a formatted error message if an exception occurs during the fetch operation
        return f"Failed to fetch data: {str(e)}"

# Define an asynchronous function for obtaining structures from the Materials Project API
async def get_structure_with_api(structure, num_unit_cells=None, target_atoms=None):
    """
    Asynchronously retrieves and analyzes a structure by extending it to multiple unit cells
    and optionally filtering by target atoms.

    Args:
        structure: A structure object containing the initial unit cell information.
        num_unit_cells (list, optional): List of integers representing the number of unit cells
                                         in the x, y, and z directions.
        target_atoms (list, optional): List of atomic symbols to specifically include in the results.

    Returns:
        A list containing detailed information about each atom within the specified structure
        and unit cell dimensions, including connections to nearby atoms.
    """
    # Default to a single unit cell in each dimension if not specified
    if num_unit_cells is None:
        num_unit_cells = [1, 1, 1]

    # Calculate the number of unit cells in each direction
    x_unit_cell = int(np.ceil(num_unit_cells[0]))
    y_unit_cell = int(np.ceil(num_unit_cells[1]))
    z_unit_cell = int(np.ceil(num_unit_cells[2]))

    # Initialize an empty list to hold information about each unique atom found
    unique_atoms = []
    # Extract the first structure and its lattice matrix for coordinate calculations
    structure = structure[0].structure
    lattice = structure.lattice.matrix

    # Initialize crystal nearest neighbor finding tool and structure graph for connection analysis
    nn = CrystalNN()
    graph = StructureGraph.from_local_env_strategy(structure, nn)

    # Loop over each possible unit cell defined by the Cartesian product of the x, y, and z ranges
    for nx in range(x_unit_cell):
        for ny in range(y_unit_cell):
            for nz in range(z_unit_cell):
                # Iterate over each atomic site in the structure
                for idx, site in enumerate(structure):
                    # Calculate new fractional coordinates considering the current unit cell offsets
                    fractional_coords = site.frac_coords + np.array([nx, ny, nz])
                    # Check if the site is within the extended structure defined by num_unit_cells
                    if all(fractional_coords[i] < num_unit_cells[i] for i in range(3)):
                        atom_label = site.species_string
                        # Continue only if no target atoms are specified or if the site is one of the target atoms
                        if target_atoms is None or atom_label in target_atoms:
                            atom_info = {
                                "atom_label": atom_label,
                                "fractional_position": fractional_coords.tolist(),
                                "cartesian_position": (site.coords + np.dot([nx, ny, nz], lattice)).tolist(),
                                "connected_atoms": []
                            }
                            # Retrieve connections for this site from the structure graph
                            connected_sites = graph.get_connected_sites(idx)
                            for connected_site in connected_sites:
                                # Calculate fractional and Cartesian coordinates for connected sites
                                connected_fractional_coords = connected_site.site.frac_coords + np.array([nx, ny, nz])
                                connection = {
                                    "connected_to": connected_site.site.species_string,
                                    "bond_length": connected_site.weight,
                                    "connected_fractional_position": connected_fractional_coords.tolist(),
                                    "connected_cartesian_position": (
                                            connected_site.site.coords + np.dot([nx, ny, nz], lattice)).tolist(),
                                    "site_index": connected_site.index
                                }
                                # Add this connection to the atom's information
                                atom_info["connected_atoms"].append(connection)

                            # Append the fully detailed atom information to the list of unique atoms
                            unique_atoms.append(atom_info)

    #return the complete list of unique atoms and their details
    return unique_atoms

def bond_by_nearest_neighbors(data, tolerance=0.1):
    """
    Identifies and records the nearest neighbors for each atom in a structure based on Cartesian coordinates.

    Args:
        data (list of dicts): List containing dictionaries with atom information, including their Cartesian coordinates.
        tolerance (float): Fractional tolerance for considering other atoms as nearest neighbors, relative to the closest distance.

    Returns:
        list of dicts: The input data list, but updated with nearest neighbor connections for each atom.
    """

    # Convert list of atom cartesian positions into a numpy array for distance calculations
    positions = np.array([atom['cartesian_position'] for atom in data])

    # Calculate the Euclidean distance matrix for all pairs of atoms
    distance_matrix = np.linalg.norm(positions[:, np.newaxis] - positions, axis=2)

    # Avoid self-comparison by setting diagonal values to infinity
    np.fill_diagonal(distance_matrix, np.inf)

    # Determine the minimum distance to a neighbor for each atom
    min_distances = np.min(distance_matrix, axis=1)
    # Get the index of the closest neighbor for each atom
    nearest_neighbors_indices = np.argmin(distance_matrix, axis=1)

    # Clear any existing connected atoms data to avoid duplication
    for atom in data:
        atom['connected_atoms'] = []

    # Iterate over each atom to establish its nearest neighbors within the specified tolerance
    for index, atom in enumerate(data):
        # Extract the nearest distance found for the current atom
        nearest_distance = min_distances[index]

        # Identify indices of all atoms that are within the tolerance range of the nearest distance
        close_indices = np.where((distance_matrix[index] <= nearest_distance * (1 + tolerance)) &
                                 (distance_matrix[index] >= nearest_distance * (1 - tolerance)))[0]

        # For each close index, add the atom as a connected neighbor if it's not the atom itself
        for close_index in close_indices:
            if close_index != index:  # Ensure the atom does not connect to itself
                connection = {
                    'nearest_neighbor_index': close_index,
                    'connected_cartesian_position': data[close_index]['cartesian_position']
                }
                atom['connected_atoms'].append(connection)

    # Return the data with updated connections
    return data

# # Run the main function using asyncio
# if __name__ == "__main__":
#     asyncio.run(get_structure_with_api())

# unique_atoms = get_structure_with_cif('Yb2Si2O7.cif', target_atoms=['Yb'])
# connected_atoms = find_nearest_neighbors(unique_atoms)
#
# #
# print(unique_atoms)
# print(connected_atoms)
