"""
Core functions for protein structural analysis and distance matrix calculation.
Supports CA/CB atoms, single-letter AA codes, and robust filtering against Calcium ions.
"""
import numpy as np
from scipy.spatial.distance import cdist
from biopandas.pdb import PandasPdb
import utils

# Standard 3-to-1 amino acid mapping
AA_3_TO_1 = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
}

def get_distance_matrix(file_path, atom_type='CB'):
    """ Calculate pairwise distance matrix. Handles PDB and CIF. """
    if file_path.lower().endswith('.cif'):
        return utils.get_dist_matrix_from_cif(file_path, atom_type)
    
    ppdb = PandasPdb().read_pdb(file_path)
    atom_df = ppdb.df['ATOM']
    
    # Selection with Calcium (Ca) protection: ensure element is Carbon ('C')
    if atom_type == 'CB':
        mask = (
            ((atom_df['atom_name'] == 'CB') & (atom_df['element_symbol'] == 'C')) |
            ((atom_df['residue_name'] == 'GLY') & (atom_df['atom_name'] == 'CA') & (atom_df['element_symbol'] == 'C'))
        )
        atoms = atom_df[mask]
    else:
        atoms = atom_df[(atom_df['atom_name'] == 'CA') & (atom_df['element_symbol'] == 'C')]
        
    coords = atoms[['x_coord', 'y_coord', 'z_coord']].values
    return cdist(coords, coords, metric='euclidean')

def get_contact_map(dist_matrix, threshold):
    """ Binary contact map (1.0 for contact, 0.0 otherwise) """
    return (dist_matrix <= threshold).astype(float)

def get_nearby_residues(file_path, target_idx, expected_aa_1, thresholds=[4, 5, 6, 7, 8], atom_type='CA'):
    """
    Find nearby residues with 1-letter code validation.
    """
    # Standardized info extraction via utils
    res_info = utils.get_residue_info(file_path, atom_type)
    res_names_3 = [item[0] for item in res_info]
    coords = np.array([item[1] for item in res_info])

    if target_idx >= len(res_names_3):
        raise IndexError(f"Index {target_idx} out of range.")

    # Validation
    actual_aa_3 = res_names_3[target_idx]
    actual_aa_1 = AA_3_TO_1.get(actual_aa_3, '?')
    if actual_aa_1 != expected_aa_1.upper():
        raise ValueError(f"Validation Error: Index {target_idx} is {actual_aa_3}({actual_aa_1}), not {expected_aa_1}")
    
    print(f"Validated: {actual_aa_3}({actual_aa_1}) found at index {target_idx}.\n")

    target_coord = coords[target_idx].reshape(1, 3)
    distances = cdist(target_coord, coords, metric='euclidean')[0]
    
    results = {}
    for t in thresholds:
        indices = np.where((distances <= t) & (np.arange(len(distances)) != target_idx))[0]
        results[t] = [{
            "index": int(i), 
            "res_1": AA_3_TO_1.get(res_names_3[i], '?'),
            "dist": round(distances[i], 2)
        } for i in indices]
    return results