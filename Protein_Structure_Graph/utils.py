""" Utility functions for file parsing. """
import os
import numpy as np
from scipy.spatial.distance import cdist
from biopandas.pdb import PandasPdb
import pandas as pd
import io
from scipy.spatial.distance import cdist

try:
    from Bio.PDB import MMCIFParser
    HAS_BIOPYTHON = True
except ImportError:
    HAS_BIOPYTHON = False
    
def read_cif_atoms(cif_path):
    """ Manually parse the _atom_site loop from a CIF file into a Pandas DataFrame. """
    with open(cif_path, 'r') as f:
        lines = f.readlines()
    
    start_idx = -1
    cols = []
    data_start = -1
    for i, line in enumerate(lines):
        if line.startswith('_atom_site.'):
            cols.append(line.strip().split('.')[1])
            if start_idx == -1: start_idx = i
        elif start_idx != -1 and not line.startswith('_atom_site.'):
            data_start = i
            break
            
    data_lines = []
    for line in lines[data_start:]:
        if line.strip() == '' or line.startswith('#') or line.startswith('loop_'):
            break
        data_lines.append(line.strip())
    
    df = pd.read_csv(io.StringIO('\n'.join(data_lines)), sep='\s+', names=cols, engine='python')
    return df
    

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