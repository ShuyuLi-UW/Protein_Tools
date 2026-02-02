""" Utility functions for file parsing. """
import os
import numpy as np
from scipy.spatial.distance import cdist
from biopandas.pdb import PandasPdb

try:
    from Bio.PDB import MMCIFParser
    HAS_BIOPYTHON = True
except ImportError:
    HAS_BIOPYTHON = False

def get_residue_info(file_path, atom_type='CA'):
    """ Unified extractor for PDB and CIF. """
    if file_path.lower().endswith('.cif'):
        return get_residue_info_from_cif(file_path, atom_type)
    
    ppdb = PandasPdb().read_pdb(file_path)
    df = ppdb.df['ATOM']
    mask = (df['atom_name'] == atom_type) & (df['element_symbol'] == 'C')
    if atom_type == 'CB':
        mask = ((df['atom_name'] == 'CB') & (df['element_symbol'] == 'C')) | \
               ((df['residue_name'] == 'GLY') & (df['atom_name'] == 'CA') & (df['element_symbol'] == 'C'))
    
    filtered = df[mask]
    return list(zip(filtered['residue_name'], filtered[['x_coord', 'y_coord', 'z_coord']].values))

def get_residue_info_from_cif(cif_path, atom_type='CA'):
    if not HAS_BIOPYTHON: raise ImportError("Install biopython.")
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure('protein', cif_path)
    res_info = []
    for residue in structure[0].get_residues():
        if residue.id[0] != ' ': continue # Skip HETATM
        target = atom_type
        if atom_type == 'CB' and 'CB' not in residue: target = 'CA'
        if target in residue and residue[target].element == 'C':
            res_info.append((residue.get_resname(), residue[target].get_coord()))
    return res_info

def get_dist_matrix_from_cif(cif_path, atom_type='CA'):
    info = get_residue_info_from_cif(cif_path, atom_type)
    coords = np.array([i[1] for i in info])
    return cdist(coords, coords, metric='euclidean')