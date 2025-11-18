from collections import defaultdict
import numpy as np
from openbabel import pybel, openbabel
from scripts.utils import residues, pdb_to_sybyl, restype_3to1



class Protein:
    def __init__(self, file_path, pocket_xyz, cutoff):
        self.pocket_xyz = pocket_xyz
        self.all_residue_1_letter = {}  # {'chain_id': MNPAKL, ...}
        self.all_residue_seq = {}  # {'chain_id': ['1', '2', ...], ...}
        self.atoms = {}  # {atom_id: (atom_name, x, y, z, atom_type, residue_name, residue_seq, residue_idx, chain_id)}
        self.bonds = {}  # {atom_id: [connected_atom_id1, connected_atom_id2, ...]}
        self.amides = []             # amide 
        self.residues = []    # [[atom1, atom2, ...], ...]
        self.reserved_residues_dic = {}  # {'chain_id':{'residue_seq':{'residue_index':0, 'residue_name':'ASP', 'atoms_id':[atom_id1, ...], 'virt_CB_xyz':np.array}}, ...}
        self.reserved_residue_names = [] # ['ARG', 'GLU', 'GLY', ...]
        self.connected_residues = {'peptide_bonded':[], 'spatial_close':[], 'geometrical_edge':[]}  # {'spatial_close':[([i, j], dist),]}
        
        self._pre_process_pdb(file_path)
        self._parse_reserved_pocket(pocket_xyz=pocket_xyz, cutoff=cutoff)
        self._find_connected_residues(distance_threshold=6.0)
        self._classify_atoms()
        self._identify_aromatic_rings()


    def calculate_distance(self, xyz1, xyz2):
        dx = xyz1[0] - xyz2[0]
        dy = xyz1[1] - xyz2[1]
        dz = xyz1[2] - xyz2[2]
        return np.sqrt(dx*dx + dy*dy + dz*dz)
    

    def _pre_process_pdb(self, file_path):
        processed_pdb_lines = []
        all_residue_1_letter = defaultdict(list)
        all_residue_seq = defaultdict(list)
        previous_residue_seq = None
        previous_chain_id = None
        with open(file_path, 'r') as f:
            for line in f:
                if line.startswith('TER'):
                    processed_pdb_lines.append(line)
                elif (line.startswith('ATOM') or line.startswith('HETATM')) and line[17:20] in residues:
                    processed_pdb_lines.append(line)
                    chain_id = line[21]
                    residue_name = line[17:20]
                    residue_seq = line[22:27].strip()
                    if (residue_seq != previous_residue_seq) or (chain_id != previous_chain_id):
                        previous_residue_seq = residue_seq
                        previous_chain_id = chain_id
                        residue_name_1_letter = restype_3to1[residue_name]
                        all_residue_1_letter[chain_id].append(residue_name_1_letter)
                        all_residue_seq[chain_id].append(residue_seq)
        self.all_residue_1_letter = {k: ''.join(v) for k, v in all_residue_1_letter.items()}
        self.all_residue_seq = dict(all_residue_seq)
        self.processed_pdb_lines = processed_pdb_lines


    def _parse_reserved_pocket(self, pocket_xyz, cutoff=None):
        pocket_residues = defaultdict(list)
        reserved_residue_lines = []
        previous_residue_seq = None
        before_CA_lines = []
        after_CA = False
        for line in self.processed_pdb_lines:
            if line.startswith('ATOM') or line.startswith('HETATM'):
                atom_name = line[12:16].strip()
                chain_id = line[21]
                residue_seq = line[22:27].strip()
                
                if residue_seq != previous_residue_seq:
                    previous_residue_seq = residue_seq
                    before_CA_lines = []
                    after_CA = False

                if not after_CA:
                    before_CA_lines.append(line)
                else:
                    reserved_residue_lines.append(line)
                    
                if atom_name == 'CA':
                    xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
                    d = self.calculate_distance(xyz, pocket_xyz)
                    if cutoff is None or d < cutoff:
                        after_CA = True
                        reserved_residue_lines += before_CA_lines
                        
        openbabel.obErrorLog.SetOutputLevel(0)
        pdb_content = "".join(reserved_residue_lines)
        mol = pybel.readstring("pdb", pdb_content)
        reserved_new_content = mol.write("pdb")
        self.reserved_new_lines = reserved_new_content.splitlines() # type: ignore

        residue_idx = -1
        previous_residue_seq = None
        previous_chain_id = None
        bonds = defaultdict(list)
        reserved_residues_dic = defaultdict(dict)
        
        virtual_CB_list = []
        temp_main_chain = {}
        last_residue_check = True

        for line in self.reserved_new_lines:
            if line.startswith('ATOM') or line.startswith('HETATM'):
                atom_id = int(line[6:11])
                atom_name = line[12:16].strip()
                residue_name = line[17:20]
                chain_id = line[21]
                residue_seq = line[22:27].strip()
                element_type = line[76:78].strip()
                x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
                atom_type = pdb_to_sybyl[residue_name][atom_name] if element_type != 'H' else 'H'
                if residue_seq != previous_residue_seq or chain_id != previous_chain_id:
                    if residue_seq != None and previous_chain_id != None:
                        if all(key in temp_main_chain for key in ['N', 'CA', 'C']):
                            coor_CA = np.array(temp_main_chain['CA'])
                            coor_N = np.array(temp_main_chain['N'])
                            coor_C = np.array(temp_main_chain['C'])
                            b = coor_CA - coor_N
                            c = coor_C - coor_CA
                            a = np.cross(b,c)
                            vCB_xyz = -0.58273431*a + 0.56802827*b - 0.54067466*c + coor_CA
                        elif 'CA' in temp_main_chain:
                            vCB_xyz = np.array(temp_main_chain['CA'])
                        elif 'N' in temp_main_chain and 'C' in temp_main_chain:
                            vCB_xyz = 0.5*(np.array(temp_main_chain['N']) + np.array(temp_main_chain['C']))
                        elif 'N' in temp_main_chain:
                            vCB_xyz = np.array(temp_main_chain['N'])
                        elif 'C' in temp_main_chain:
                            vCB_xyz = np.array(temp_main_chain['C'])
                        else:
                            vCB_xyz = np.array([99., 99., 99.])

                        reserved_residues_dic[previous_chain_id][previous_residue_seq]['virt_CB_xyz'] = vCB_xyz
                        previous_residue_idx = reserved_residues_dic[previous_chain_id][previous_residue_seq]['residue_index']
                        if chain_id != previous_chain_id:
                            virtual_CB_list.append([previous_residue_idx, chain_id, residue_seq, vCB_xyz])
                        elif residue_seq != previous_residue_seq:
                            virtual_CB_list.append([previous_residue_idx, previous_chain_id, residue_seq, vCB_xyz])
                        temp_main_chain = {}

                    previous_residue_seq = residue_seq
                    previous_chain_id = chain_id
                    residue_idx += 1
                    reserved_residues_dic[chain_id][residue_seq] = {
                        'residue_index': residue_idx, 
                        'residue_name':residue_name,
                        'atoms_id':[atom_id],
                        'carbons':[],
                        'amide_carbons':[],
                        'alpha_carbons':[],
                        'aromatic_carbons':[],
                        'aliphatic_carbons':[],
                        'weak_hbond_donors':{'ali':[], 'aro':[]},
                        'nitrogens':[],
                        'amide_nitrogens':[],
                        'positively_charged_nitrogens':[],
                        'hbond_donors_NH':[],
                        'hbond_acceptors_N':[],
                        'oxygens':[],
                        'amide_oxygens':[],
                        'negatively_charged_oxygens':[],
                        'hbond_donors_OH':[],
                        'hbond_acceptors_O':[],
                        'sulfurs':[],
                        'aromatic_rings':[],
                        'virt_CB_xyz':None
                        }
                    self.reserved_residue_names.append(residue_name)
                    
                else:
                    #print(chain_id, residue_seq)
                    reserved_residues_dic[chain_id][residue_seq]['atoms_id'].append(atom_id)

                if atom_name in ['N', 'CA', 'C']:
                    temp_main_chain[atom_name] = (x, y, z)
                
                self.atoms[atom_id] = {
                    'atom_name': atom_name,
                    'xyz': np.array((float(x), float(y), float(z))),
                    'atom_type': atom_type,
                    'residue_name': residue_name,
                    'residue_seq': residue_seq,
                    'residue_idx': residue_idx,
                    'chain_id': chain_id
                    }
                #(atom_name, x, y, z, atom_type, residue_name, residue_seq, residue_idx, chain_id)
                bonds[atom_id] = []
            elif line.startswith('CONECT'):
                # last residue
                if last_residue_check:
                    if all(key in temp_main_chain for key in ['N', 'CA', 'C']):
                        coor_CA = np.array(temp_main_chain['CA'])
                        coor_N = np.array(temp_main_chain['N'])
                        coor_C = np.array(temp_main_chain['C'])
                        b = coor_CA - coor_N
                        c = coor_C - coor_CA
                        a = np.cross(b,c)
                        vCB_xyz = -0.58273431*a + 0.56802827*b - 0.54067466*c + coor_CA
                    elif 'CA' in temp_main_chain:
                        vCB_xyz = np.array(temp_main_chain['CA'])
                    elif 'N' in temp_main_chain and 'C' in temp_main_chain:
                        vCB_xyz = 0.5*(np.array(temp_main_chain['N']) + np.array(temp_main_chain['C']))
                    elif 'N' in temp_main_chain:
                        vCB_xyz = np.array(temp_main_chain['N'])
                    elif 'C' in temp_main_chain:
                        vCB_xyz = np.array(temp_main_chain['C'])
                    else:
                        vCB_xyz = np.array([99., 99., 99.])
                    
                    reserved_residues_dic[previous_chain_id][previous_residue_seq]['virt_CB_xyz'] = vCB_xyz
                    previous_residue_idx = reserved_residues_dic[previous_chain_id][previous_residue_seq]['residue_index']
                    if chain_id != previous_chain_id:
                        virtual_CB_list.append([previous_residue_idx, chain_id, residue_seq, vCB_xyz])
                    elif residue_seq != previous_residue_seq:
                        virtual_CB_list.append([previous_residue_idx, previous_chain_id, residue_seq, vCB_xyz])
                    temp_main_chain = {}
                    last_residue_check = False

                atom_ids = [line[i:i+5].strip() for i in range(6, 31, 5)]
                atom_ids = [int(a) for a in atom_ids if a]
                center_id = atom_ids[0]
                bonds[center_id].extend(list(set(atom_ids[1:])))
                
        self.bonds = dict(bonds)
        self.reserved_residues_dic = dict(reserved_residues_dic)
        self.virtual_CB_list = virtual_CB_list

    
    def _classify_atoms(self):
        for atom_id, atom_data in self.atoms.items():
            # (atom_name, x, y, z, atom_type, residue_name, residue_seq, residue_idx, chain_id)
            atom_type = atom_data['atom_type']
            atom_name = atom_data['atom_name']
            chain_id = atom_data['chain_id']
            residue_seq = atom_data['residue_seq']
            residue_name = atom_data['residue_name']
            if atom_type.startswith("C."):
                self.reserved_residues_dic[chain_id][residue_seq]['carbons'].append(atom_id)
                #self.carbons.append(atom_id)    # carbon
                if atom_name == 'C':
                    self.reserved_residues_dic[chain_id][residue_seq]['amide_carbons'].append(atom_id)
                    #self.amide_carbons.append(atom_id)    # amide carbon
                    id_C = atom_id
                    id_O = None
                    id_N = None
                    for neighbor in self.bonds[atom_id]:
                        neighbor_name = self.atoms[neighbor]['atom_name']
                        if neighbor_name == 'N':
                            id_N = neighbor
                        elif neighbor_name == 'O':
                            id_O = neighbor
                        elif neighbor_name == 'CA':
                            id_CA = neighbor
                            self.reserved_residues_dic[chain_id][residue_seq]['alpha_carbons'].append(id_CA)
                            #self.alpha_carbons.append(id_CA)
                    if None not in (id_C, id_O, id_N):
                        self.amides.append([id_C, id_O, id_N])
                elif atom_type == "C.ar":    # aromatic carbon
                    self.reserved_residues_dic[chain_id][residue_seq]['aromatic_carbons'].append(atom_id)
                    #self.aromatic_carbons.append(atom_id)
                    for neighbor in self.bonds[atom_id]:
                        neighbor_name = self.atoms[neighbor]['atom_name']
                        if "H" in neighbor_name:
                            self.reserved_residues_dic[chain_id][residue_seq]['weak_hbond_donors']['aro'].append([atom_id, neighbor]) 
                            #self.weak_hbond_donors['aro'].append([atom_id, neighbor]) 
                elif atom_type in {"C.3", "C.2", "C.1"}:    # aliphatic carbon
                    self.reserved_residues_dic[chain_id][residue_seq]['aliphatic_carbons'].append(atom_id)
                    #self.aliphatic_carbons.append(atom_id)
                    for neighbor in self.bonds[atom_id]:
                        neighbor_name = self.atoms[neighbor]['atom_name']
                        if "H" in neighbor_name:
                            self.reserved_residues_dic[chain_id][residue_seq]['weak_hbond_donors']['ali'].append([atom_id, neighbor]) 
                            #self.weak_hbond_donors['ali'].append([atom_id, neighbor]) 
            
            elif atom_type.startswith("N"):
                self.reserved_residues_dic[chain_id][residue_seq]['nitrogens'].append(atom_id)
                #self.nitrogens.append(atom_id)    # nitrogen
                if atom_name == 'N':
                    self.reserved_residues_dic[chain_id][residue_seq]['amide_nitrogens'].append(atom_id)
                    #self.amide_nitrogens.append(atom_id)    # amide nitrogen
                elif (
                    (atom_name == 'NZ' and residue_name == 'LYS') or \
                    ((atom_name == 'NE' or atom_name.startswith('NH')) and residue_name == 'ARG') or \
                    ((atom_name == 'ND1' or atom_name == 'NE2') and residue_name == 'HIS')
                ):
                    self.reserved_residues_dic[chain_id][residue_seq]['positively_charged_nitrogens'].append(atom_id)
                    #self.positively_charged_nitrogens.append(atom_id)
                    
                # find N or NH
                is_donor = False
                for neighbor in self.bonds[atom_id]:
                    neighbor_name = self.atoms[neighbor]['atom_name']
                    if "H" in neighbor_name:
                        self.reserved_residues_dic[chain_id][residue_seq]['hbond_donors_NH'].append([atom_id, neighbor])
                        #self.hbond_donors_NH.append([atom_id, neighbor])  # N-H Hbond donor 
                        is_donor = True
                if not is_donor:
                    self.reserved_residues_dic[chain_id][residue_seq]['hbond_acceptors_N'].append(atom_id)
                    #self.hbond_acceptors_N.append(atom_id)
                    
            elif atom_type.startswith("O"):
                self.reserved_residues_dic[chain_id][residue_seq]['oxygens'].append(atom_id)
                #self.oxygens.append(atom_id)    # oxygen
                if atom_name == 'O':    # amide oxygen
                    self.reserved_residues_dic[chain_id][residue_seq]['amide_oxygens'].append(atom_id)
                    #self.amide_oxygens.append(atom_id)
                elif (
                    (atom_name.startswith('OE') and residue_name == 'GLU') or \
                    (atom_name.startswith('OD') and residue_name == 'ASP')
                ):
                    self.reserved_residues_dic[chain_id][residue_seq]['negatively_charged_oxygens'].append(atom_id)
                    #self.negatively_charged_oxygens.append(atom_id)
                    
                # find O or OH
                is_donor = False
                for neighbor in self.bonds[atom_id]:
                    neighbor_name = self.atoms[neighbor]['atom_name']
                    if "H" in neighbor_name:
                        self.reserved_residues_dic[chain_id][residue_seq]['hbond_donors_OH'].append([atom_id, neighbor])
                        #self.hbond_donors_OH.append([atom_id, neighbor])  # O-H Hbond donor 
                        is_donor = True
                if not is_donor:
                    self.reserved_residues_dic[chain_id][residue_seq]['hbond_acceptors_O'].append(atom_id)
                    #self.hbond_acceptors_O.append(atom_id)
                    
            elif atom_type.startswith("S."):    # sulfur
                self.reserved_residues_dic[chain_id][residue_seq]['sulfurs'].append(atom_id)
                #self.sulfurs.append(atom_id)
                

    def _identify_aromatic_rings(self):
        visited = set()
        def is_aromatic(atom_id):
            atom_type = self.atoms[atom_id]['atom_type']
            return atom_type in {"C.ar", "N.ar", "O.ar", "S.ar"}
        
        def dfs(atom_id, path, ring_atoms):
            if atom_id in path:
                ring_start_index = path.index(atom_id)
                ring = path[ring_start_index:]
                if 5 <= len(ring) <= 6 and all(is_aromatic(atom) for atom in ring):
                    if ring not in ring_atoms:
                        ring_atoms.append(ring)
                return True
            if atom_id in visited:
                return False
            visited.add(atom_id)
            path.append(atom_id)
            for neighbor in self.bonds[atom_id]:
                if is_aromatic(neighbor):
                    dfs(neighbor, path, ring_atoms)
            path.pop()
            return False

        for chain_id, chain_data in self.reserved_residues_dic.items():
            for residue_seq, residue_data in chain_data.items():
                ring_atoms = []
                for atom_id in residue_data['atoms_id']:
                    if atom_id not in visited and is_aromatic(atom_id):
                        dfs(atom_id, [], ring_atoms)
                        self.reserved_residues_dic[chain_id][residue_seq]['aromatic_rings'] = ring_atoms


    def _find_connected_residues(self, distance_threshold=6.0):
        def get_main_chain_xyzs(chain_id, residue_seq):
            main_chain = ['C', 'O', 'N', 'CA']
            main_chain_atom_xyzs = {}
            for atom_id in self.reserved_residues_dic[chain_id][residue_seq]['atoms_id']:
                atom_name = self.atoms[atom_id]['atom_name']
                if len(main_chain_atom_xyzs) == 4:
                    break
                if atom_name in main_chain:
                    main_chain_atom_xyzs[atom_name] = self.atoms[atom_id]['xyz']
            if len(main_chain_atom_xyzs) < 4:
                current_keys = main_chain_atom_xyzs.keys()
                missing_keys = main_chain - current_keys
                existing_arrays = [main_chain_atom_xyzs[k] for k in main_chain if k in main_chain_atom_xyzs]
                mean_xyz = np.mean(existing_arrays, axis=0)
                for key in missing_keys:
                    main_chain_atom_xyzs[key] = mean_xyz.copy()
            return main_chain_atom_xyzs
        # peptide bonded
        for id_C, id_O, id_N in self.amides:
            residue_idx1 = self.atoms[id_C]['atom_type']
            residue_idx2 = self.atoms[id_N]['atom_type']
            self.connected_residues['peptide_bonded'].append([residue_idx1, residue_idx2])
        # spatial close
        for i in range(len(self.virtual_CB_list)):
            residue_idx1, chain_id1, residue_seq1, vCB_xyz1 = self.virtual_CB_list[i]
            main_chain_atom_xyzs1 = get_main_chain_xyzs(chain_id1, residue_seq1)
            for j in range(i + 1, len(self.virtual_CB_list)):
                residue_idx2, chain_id2, residue_seq2, vCB_xyz2 = self.virtual_CB_list[j]
                #residue_idx1 = self.virtual_CB_list[i][0]
                #coor1 = self.virtual_CB_list[i][3]
                #residue_idx2 = self.virtual_CB_list[j][0]
                #coor2 = self.virtual_CB_list[j][3]
                dist = np.linalg.norm(vCB_xyz1 - vCB_xyz2)
                if dist < distance_threshold:
                    spatial_pair_dist = ([residue_idx1, residue_idx2], dist)
                    self.connected_residues['spatial_close'].append(spatial_pair_dist)

                    main_chain_atom_xyzs2 = get_main_chain_xyzs(chain_id2, residue_seq2)
                    
                    atoms = ['C', 'O', 'N', 'CA']
                    distance_matrix = []
                    for atom_name1 in atoms:
                        row = []
                        for atom_name2 in atoms:
                            delta = main_chain_atom_xyzs1[atom_name1] - main_chain_atom_xyzs2[atom_name2]
                            row.append(np.linalg.norm(delta))
                        distance_matrix.append(row)

                    main_chain_distances_list1 = [d for row in distance_matrix for d in row]
                    main_chain_distances_list2 = [d for row in zip(*distance_matrix) for d in row]
                    self.connected_residues['geometrical_edge'].extend([
                        ([residue_idx1, residue_idx2], main_chain_distances_list1),
                        ([residue_idx2, residue_idx1], main_chain_distances_list2)
                    ])


if __name__ == "__main__":
    import argparse
    from parse_ligand import Ligand

    parser = argparse.ArgumentParser()
    parser.add_argument("-r", type=str, 
                        default='/home/wzh/data/datasets/pdbbind2019/all/10gs/10gs_protein.pdb', 
                        help='receptor file, pdb format')
    
    args = parser.parse_args()
    #rec_pdb_file = args.r
    
    dataset = '/home/wzh/data/datasets/pdbbind2019/all/'
    pdb = '2zc9'
    
    lig_mol2_file = f'{dataset}/{pdb}/{pdb}_ligand.mol2'
    rec_pdb_file = f'{dataset}/{pdb}/{pdb}_protein.pdb'

    ligand = Ligand(lig_mol2_file)
    pocket_xyz = ligand.pocket_center
    molecule = Protein(rec_pdb_file, pocket_xyz=pocket_xyz, cutoff=20)

    #molecule = Protein('/home/wzh/data/datasets/pdbbind2019/all/1a0q/1a0q_protein.pdb', (13.45304348, 20.57547826, 59.11582609), cutoff=20)
    #print(molecule.connected_residues['spatial_close'])
    
    #print(molecule.atoms[230])
    #print(molecule.all_residue_1_letter)
    #print(molecule.all_residue_seq)
    #for chain_id, seq in molecule.all_residue_1_letter.items():
    #    print(chain_id, seq)
    #print(molecule.atoms)
    #print(molecule.bonds)
    #print(molecule.carbons)
    #print(molecule.nitrogens)
    #print(molecule.aromatic_rings)
    #print(molecule.hbond_donors_OH)
    #print(molecule.hbond_donors_NH)
    #print(molecule.hbond_acceptors_O)
    #print(molecule.hbond_acceptors_N)
    #print(molecule.weak_hbond_donors)
    #print(molecule.positively_charged_nitrogens)
    #print(molecule.negatively_charged_oxygens)
    #print(molecule.sulfurs)
    #print(molecule.amides)
    #print(molecule.amide_carbons)
    #print(molecule.amide_oxygens)
    #print(molecule.amide_nitrogens)
    #print(molecule.residues)
    #print(molecule.reserved_residues_dic['A']['5'])
    #print(molecule.reserved_residue_names)
    #print(molecule.connected_residues)
    #print(molecule.reserved_residues_dic)