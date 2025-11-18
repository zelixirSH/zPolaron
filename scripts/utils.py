import torch
import numpy as np

residues = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'HIS', 'ILE', 'LEU', 
            'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL', 'GLY']

lig_atom_types = ['N.1', 'N.2', 'N.3', 'N.4', 'N.am', 'N.ar', 'N.pl3', 'C.1', 
                  'C.2', 'C.3', 'C.ar', 'C.cat', 'O.co2', 'O.2', 'O.3', 'S.2', 
                  'S.3', 'S.o', 'S.o2', 'H', 'P.3', 'Hal', 'Metal', 'Other']

halogen_list = ['F', 'Cl', 'Br', 'I']

metal_list = ['Na', 'K', 'Ca', 'Fe', 'Be', 'Zn', 'Cu', 'Re', 'V', 
              'Pt', 'Ir', 'Rh', 'Sb', 'Mg', 'Co', 'Os', 'Ru']


pdb_to_sybyl = {
    'ALA': {'N':'N.am', 'CA':'C.3', 'C':'C.2', 'O':'O.2', 'CB':'C.3', 'OXT': 'O.co2'}, 
    'ARG': {'N':'N.am', 'CA':'C.3', 'C':'C.2', 'O':'O.2', 'CB':'C.3', 'CG':'C.3', 'CD':'C.3', 'NE':'N.pl3', 'CZ':'C.cat', 'NH1':'N.pl3', 'NH2':'N.pl3', 'OXT': 'O.co2'}, 
    'ASN': {'N':'N.am', 'CA':'C.3', 'C':'C.2', 'O':'O.2', 'CB':'C.3', 'CG':'C.2', 'OD1':'O.2', 'ND2':'N.am', 'OXT': 'O.co2'}, 
    'ASP': {'N':'N.am', 'CA':'C.3', 'C':'C.2', 'O':'O.2', 'CB':'C.3', 'CG':'C.2', 'OD1':'O.co2', 'OD2':'O.co2', 'OXT': 'O.co2'}, 
    'CYS': {'N':'N.am', 'CA':'C.3', 'C':'C.2', 'O':'O.2', 'CB':'C.3', 'SG':'S.3', 'OXT': 'O.co2'}, 
    'GLN': {'N':'N.am', 'CA':'C.3', 'C':'C.2', 'O':'O.2', 'CB':'C.3', 'CG':'C.3', 'CD':'C.2', 'OE1':'O.2', 'NE2':'N.am', 'OXT': 'O.co2'}, 
    'GLU': {'N':'N.am', 'CA':'C.3', 'C':'C.2', 'O':'O.2', 'CB':'C.3', 'CG':'C.3', 'CD':'C.2', 'OE1':'O.co2', 'OE2':'O.co2', 'OXT': 'O.co2'}, 
    'HIS': {'N':'N.am', 'CA':'C.3', 'C':'C.2', 'O':'O.2', 'CB':'C.3', 'CG':'C.ar', 'ND1':'N.ar', 'CD2':'C.ar', 'CE1':'C.ar', 'NE2':'N.ar', 'OXT': 'O.co2'}, 
    'ILE': {'N':'N.am', 'CA':'C.3', 'C':'C.2', 'O':'O.2', 'CB':'C.3', 'CG1':'C.3', 'CG2':'C.3', 'CD1':'C.3', 'OXT': 'O.co2'}, 
    'LEU': {'N':'N.am', 'CA':'C.3', 'C':'C.2', 'O':'O.2', 'CB':'C.3', 'CG':'C.3', 'CD1':'C.3', 'CD2':'C.3', 'OXT': 'O.co2'}, 
    'LYS': {'N':'N.am', 'CA':'C.3', 'C':'C.2', 'O':'O.2', 'CB':'C.3', 'CG':'C.3', 'CD':'C.3', 'CE':'C.3', 'NZ':'N.4', 'OXT': 'O.co2'}, 
    'MET': {'N':'N.am', 'CA':'C.3', 'C':'C.2', 'O':'O.2', 'CB':'C.3', 'CG':'C.3', 'SD':'S.3', 'CE':'C.3', 'OXT': 'O.co2'}, 
    'PHE': {'N':'N.am', 'CA':'C.3', 'C':'C.2', 'O':'O.2', 'CB':'C.3', 'CG':'C.ar', 'CD1':'C.ar', 'CD2':'C.ar', 'CE1':'C.ar', 'CE2':'C.ar', 'CZ':'C.ar', 'CH2':'C.ar', 'OXT': 'O.co2'}, 
    'PRO': {'N':'N.am', 'CA':'C.3', 'C':'C.2', 'O':'O.2', 'CB':'C.3', 'CG':'C.3', 'CD':'C.3', 'OXT': 'O.co2'}, 
    'SER': {'N':'N.am', 'CA':'C.3', 'C':'C.2', 'O':'O.2', 'CB':'C.3', 'OG':'O.3', 'OXT': 'O.co2'}, 
    'THR': {'N':'N.am', 'CA':'C.3', 'C':'C.2', 'O':'O.2', 'CB':'C.3', 'OG1':'O.3', 'CG2':'C.3', 'OXT': 'O.co2'}, 
    'TRP': {'N':'N.am', 'CA':'C.3', 'C':'C.2', 'O':'O.2', 'CB':'C.3', 'CG':'C.ar', 'CD1':'C.ar', 'CD2':'C.ar', 'NE1':'N.ar', 'CE2':'C.ar', 'CE3':'C.ar', 'CZ2':'C.ar', 'CZ3':'C.ar', 'CH2':'C.ar', 'OXT': 'O.co2'}, 
    'TYR': {'N':'N.am', 'CA':'C.3', 'C':'C.2', 'O':'O.2', 'CB':'C.3', 'CG':'C.ar', 'CD1':'C.ar', 'CD2':'C.ar', 'CE1':'C.ar', 'CE2':'C.ar', 'CZ':'C.ar', 'OH':'O.3','OXT': 'O.co2'}, 
    'VAL': {'N':'N.am', 'CA':'C.3', 'C':'C.2', 'O':'O.2', 'CB':'C.3', 'CG1':'C.3', 'CG2':'C.3', 'OXT': 'O.co2'}, 
    'GLY': {'N':'N.am', 'CA':'C.3', 'C':'C.2', 'O':'O.2', 'OXT': 'O.co2'}
}


restype_3to1 = {
        "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
        "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
        "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
        "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    }


restype_1to3 = {
        "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
        "Q": "GLN", "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE",
        "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
        "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
    }


vdw_radius = {'C': 1.70, 'N':  1.55, 'O':  1.52, 'S': 1.80,
              'F': 1.47, 'Cl': 1.75, 'Br': 1.85, 'I': 1.98}


def calc_confidence(S: torch.Tensor, log_probs: torch.Tensor):
    """
    S.shape:      [L]         (predicted sequence)
    log_probs.shape: [L, 20]  (predicted log probabilities)
    """
    S_one_hot = torch.nn.functional.one_hot(S, 20)  # [L, 20]
    loss_per_pos = -(S_one_hot * log_probs).sum(-1)  # [L]
    valid_loss = loss_per_pos  # [L]
    confidence = valid_loss.sum() / (len(S)+ 1e-8)
    return float(np.exp(-confidence.cpu().numpy()))


def calc_fitness(S: torch.Tensor, probs: torch.Tensor):
    """
    S.shape:      [L]        (true sequence or target sequence)
    probs.shape: [L, 20]     (predicted probabilities)
    """
    true_aa_probs_list = []
    for i, aa_num in enumerate(list(S)):
        true_aa_probs_list.append(probs[i, aa_num].item())
    true_aa_probs = np.array(true_aa_probs_list)
    log = - np.log(true_aa_probs)
    ave_fitness = np.mean(log)
    return float(ave_fitness)


def seq_idx_to_residue(pred_seq_idx):
    seq_list = []
    for idx in pred_seq_idx:
        seq_list.append(restype_3to1[residues[idx]])
    return "".join(seq_list)


def get_per_residue_esm_embedding(esm_model, batch_converter, protein_sequence, device=None):
    if device is None:
        device = torch.device('cpu')
    
    data = [("protein", protein_sequence)]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_tokens = batch_tokens.to(device)
    
    with torch.no_grad():
        results = esm_model(batch_tokens, repr_layers=[esm_model.num_layers], return_contacts=False)
    
    token_representations = results["representations"][esm_model.num_layers][0]
    residue_embeddings = token_representations[1:-1]  # 去掉首尾特殊标记

    if device.type == 'cuda':
        residue_embeddings = residue_embeddings.cpu()
    
    return residue_embeddings
