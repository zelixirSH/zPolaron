import math
from collections import defaultdict
import numpy as np
from openbabel import pybel, openbabel

from scripts.utils import residues, pdb_to_sybyl, restype_3to1


class Protein:
    def __init__(self, file_path, pocket_xyz, cutoff):
        self.pocket_xyz = pocket_xyz

        self.all_residue_1_letter = {}   # {'chain_id': 'MNPAKL', ...}
        self.all_residue_seq = {}        # {'chain_id': ['1', '2', ...], ...}
        self.atoms = {}                  # {atom_id: {...}}
        self.bonds = {}                  # {atom_id: [connected_atom_id1, ...]}
        self.amides = []                 # [[id_C, id_O, id_N], ...]
        self.residues = []               # reserved, kept for compatibility
        self.reserved_residues_dic = {}   # {'chain_id': {'residue_seq': {...}}}
        self.reserved_residue_names = []  # ['ARG', 'GLU', 'GLY', ...]
        self.connected_residues = {
            'peptide_bonded': [],
            'spatial_close': [],
            'geometrical_edge': []
        }
        self.virtual_CB_list = []

        self._pre_process_pdb(file_path)
        self._parse_reserved_pocket(pocket_xyz=pocket_xyz, cutoff=cutoff)
        self._classify_atoms()
        self._identify_aromatic_rings()
        self._find_connected_residues(distance_threshold=8.0)
        self._compute_backbone_dihedrals()

    def calculate_distance(self, xyz1, xyz2):
        dx = xyz1[0] - xyz2[0]
        dy = xyz1[1] - xyz2[1]
        dz = xyz1[2] - xyz2[2]
        return np.sqrt(dx * dx + dy * dy + dz * dz)

    def _pre_process_pdb(self, file_path):
        processed_pdb_lines = []
        all_residue_1_letter = defaultdict(list)
        all_residue_seq = defaultdict(list)

        residue_blocks = defaultdict(list)  # {(chain_id, residue_seq): [lines]}
        residue_order = []                  # [(chain_id, residue_seq, residue_name), ...]

        with open(file_path, 'r') as f:
            for line in f:
                if line.startswith('TER'):
                    continue

                if (line.startswith('ATOM') or line.startswith('HETATM')) and line[17:20] in residues:
                    chain_id = line[21]
                    residue_name = line[17:20]
                    residue_seq = line[22:27].strip()
                    key = (chain_id, residue_seq)

                    if key not in residue_blocks:
                        residue_order.append((chain_id, residue_seq, residue_name))
                    residue_blocks[key].append(line)

        for chain_id, residue_seq, residue_name in residue_order:
            lines = residue_blocks[(chain_id, residue_seq)]

            heavy_atoms = []
            hydrogens = []
            append_heavy = heavy_atoms.append
            append_h = hydrogens.append

            for l in lines:
                if l[76:78].strip() == 'H':
                    append_h(l)
                else:
                    append_heavy(l)

            processed_pdb_lines.extend(heavy_atoms)
            processed_pdb_lines.extend(hydrogens)

            all_residue_1_letter[chain_id].append(restype_3to1.get(residue_name, 'X'))
            all_residue_seq[chain_id].append(residue_seq)

        self.all_residue_1_letter = {k: ''.join(v) for k, v in all_residue_1_letter.items()}
        self.all_residue_seq = dict(all_residue_seq)
        self.processed_pdb_lines = processed_pdb_lines

    def _compute_virtual_cb_xyz(self, temp_main_chain):
        if all(key in temp_main_chain for key in ['N', 'CA', 'C']):
            coor_CA = np.array(temp_main_chain['CA'])
            coor_N = np.array(temp_main_chain['N'])
            coor_C = np.array(temp_main_chain['C'])
            b = coor_CA - coor_N
            c = coor_C - coor_CA
            a = np.cross(b, c)
            vCB_xyz = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + coor_CA
        elif 'CA' in temp_main_chain:
            vCB_xyz = np.array(temp_main_chain['CA'])
        elif 'N' in temp_main_chain and 'C' in temp_main_chain:
            vCB_xyz = 0.5 * (np.array(temp_main_chain['N']) + np.array(temp_main_chain['C']))
        elif 'N' in temp_main_chain:
            vCB_xyz = np.array(temp_main_chain['N'])
        elif 'C' in temp_main_chain:
            vCB_xyz = np.array(temp_main_chain['C'])
        else:
            vCB_xyz = np.array([99.0, 99.0, 99.0])
        return vCB_xyz

    def _finalize_residue(self, reserved_residues_dic, chain_id, residue_seq, residue_idx,
                          temp_main_chain, virtual_CB_list):
        if chain_id is None or residue_seq is None:
            return

        residue_entry = reserved_residues_dic.get(chain_id, {}).get(residue_seq)
        if residue_entry is None:
            return

        vCB_xyz = self._compute_virtual_cb_xyz(temp_main_chain)
        residue_entry['virt_CB_xyz'] = vCB_xyz
        virtual_CB_list.append([residue_idx, chain_id, residue_seq, vCB_xyz])

    def _parse_reserved_pocket(self, pocket_xyz, cutoff=None):
        reserved_residue_lines = []
        previous_residue_seq = None
        previous_chain_id = None
        before_CA_lines = []
        after_CA = False

        for line in self.processed_pdb_lines:
            if line.startswith('ATOM') or line.startswith('HETATM'):
                atom_name = line[12:16].strip()
                residue_seq = line[22:27].strip()
                chain_id = line[21]

                if residue_seq != previous_residue_seq or chain_id != previous_chain_id:
                    previous_residue_seq = residue_seq
                    previous_chain_id = chain_id
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
                        reserved_residue_lines.extend(before_CA_lines)

        if not reserved_residue_lines:
            self.reserved_new_lines = []
            self.bonds = {}
            self.reserved_residues_dic = {}
            self.virtual_CB_list = []
            return

        pdb_content = "".join(reserved_residue_lines)
        openbabel.obErrorLog.SetOutputLevel(0)
        mol = pybel.readstring("pdb", pdb_content)
        reserved_new_content = mol.write("pdb")

        self.reserved_new_lines = reserved_new_content.splitlines()

        residue_idx = -1
        previous_residue_seq = None
        previous_chain_id = None
        current_residue_exists = False
        temp_main_chain = {}

        bonds = defaultdict(list)
        reserved_residues_dic = defaultdict(dict)
        virtual_CB_list = []

        atoms_dict = self.atoms
        sybyl_map = pdb_to_sybyl
        reserved_names_append = self.reserved_residue_names.append

        def start_new_residue(chain_id, residue_seq, residue_name, atom_id):
            nonlocal residue_idx, temp_main_chain, current_residue_exists
            residue_idx += 1
            reserved_residues_dic[chain_id][residue_seq] = {
                'residue_index': residue_idx,
                'residue_name': residue_name,
                'atoms_id': [atom_id],
                'carbons': [],
                'amide_carbons': [],
                'alpha_carbons': [],
                'aromatic_carbons': [],
                'aliphatic_carbons': [],
                'weak_hbond_donors': {'ali': [], 'aro': []},
                'nitrogens': [],
                'amide_nitrogens': [],
                'positively_charged_nitrogens': [],
                'hbond_donors_NH': [],
                'hbond_acceptors_N': [],
                'oxygens': [],
                'amide_oxygens': [],
                'negatively_charged_oxygens': [],
                'hbond_donors_OH': [],
                'hbond_acceptors_O': [],
                'sulfurs': [],
                'aromatic_rings': [],
                'virt_CB_xyz': None
            }
            reserved_names_append(residue_name)
            temp_main_chain = {}
            current_residue_exists = True

        for line in self.reserved_new_lines:
            if line.startswith('ATOM') or line.startswith('HETATM'):
                atom_id = int(line[6:11])
                atom_name = line[12:16].strip()
                residue_name = line[17:20]
                chain_id = line[21]
                residue_seq = line[22:27].strip()
                element_type = line[76:78].strip()
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])

                if residue_seq != previous_residue_seq or chain_id != previous_chain_id:
                    if current_residue_exists:
                        self._finalize_residue(
                            reserved_residues_dic=reserved_residues_dic,
                            chain_id=previous_chain_id,
                            residue_seq=previous_residue_seq,
                            residue_idx=reserved_residues_dic[previous_chain_id][previous_residue_seq]['residue_index'],
                            temp_main_chain=temp_main_chain,
                            virtual_CB_list=virtual_CB_list
                        )

                    start_new_residue(chain_id, residue_seq, residue_name, atom_id)
                    previous_residue_seq = residue_seq
                    previous_chain_id = chain_id
                else:
                    reserved_residues_dic[chain_id][residue_seq]['atoms_id'].append(atom_id)

                if atom_name in ['N', 'CA', 'C']:
                    temp_main_chain[atom_name] = (x, y, z)

                try:
                    atom_type = sybyl_map[residue_name][atom_name] if element_type != 'H' else 'H'
                except KeyError:
                    atom_type = 'X'

                atoms_dict[atom_id] = {
                    'atom_name': atom_name,
                    'xyz': np.array((x, y, z)),
                    'atom_type': atom_type,
                    'residue_name': residue_name,
                    'residue_seq': residue_seq,
                    'residue_idx': residue_idx,
                    'chain_id': chain_id
                }
                bonds[atom_id] = []

            elif line.startswith('CONECT'):
                atom_ids = [line[i:i + 5].strip() for i in range(6, 31, 5)]
                atom_ids = [int(a) for a in atom_ids if a]
                if not atom_ids:
                    continue

                center_id = atom_ids[0]
                neighbors = [a for a in atom_ids[1:] if a != center_id]
                if neighbors:
                    bonds[center_id].extend(neighbors)

        if current_residue_exists:
            self._finalize_residue(
                reserved_residues_dic=reserved_residues_dic,
                chain_id=previous_chain_id,
                residue_seq=previous_residue_seq,
                residue_idx=reserved_residues_dic[previous_chain_id][previous_residue_seq]['residue_index'],
                temp_main_chain=temp_main_chain,
                virtual_CB_list=virtual_CB_list
            )

        dedup_bonds = {}
        for atom_id, neighbor_list in bonds.items():
            seen = set()
            unique_neighbors = []
            for nb in neighbor_list:
                if nb != atom_id and nb not in seen:
                    seen.add(nb)
                    unique_neighbors.append(nb)
            dedup_bonds[atom_id] = unique_neighbors

        self.bonds = dedup_bonds
        self.reserved_residues_dic = {k: dict(v) for k, v in reserved_residues_dic.items()}
        self.virtual_CB_list = virtual_CB_list

    def _classify_atoms(self):
        atoms = self.atoms
        bonds = self.bonds
        reserved = self.reserved_residues_dic

        for atom_id, atom_data in atoms.items():
            atom_type = atom_data['atom_type']
            atom_name = atom_data['atom_name']
            chain_id = atom_data['chain_id']
            residue_seq = atom_data['residue_seq']
            residue_name = atom_data['residue_name']

            residue_entry = reserved.get(chain_id, {}).get(residue_seq)
            if residue_entry is None:
                continue

            if atom_type.startswith("C."):
                residue_entry['carbons'].append(atom_id)

                if atom_name == 'C':
                    residue_entry['amide_carbons'].append(atom_id)
                    id_C = atom_id
                    id_O = None
                    id_N = None
                    for neighbor in bonds.get(atom_id, []):
                        neighbor_name = atoms[neighbor]['atom_name']
                        if neighbor_name == 'N':
                            id_N = neighbor
                        elif neighbor_name == 'O':
                            id_O = neighbor
                        elif neighbor_name == 'CA':
                            residue_entry['alpha_carbons'].append(neighbor)
                    if None not in (id_C, id_O, id_N):
                        self.amides.append([id_C, id_O, id_N])

                elif atom_type == "C.ar":
                    residue_entry['aromatic_carbons'].append(atom_id)
                    for neighbor in bonds.get(atom_id, []):
                        neighbor_name = atoms[neighbor]['atom_name']
                        if "H" in neighbor_name:
                            residue_entry['weak_hbond_donors']['aro'].append([atom_id, neighbor])

                elif atom_type in {"C.3", "C.2", "C.1"}:
                    residue_entry['aliphatic_carbons'].append(atom_id)
                    for neighbor in bonds.get(atom_id, []):
                        neighbor_name = atoms[neighbor]['atom_name']
                        if "H" in neighbor_name:
                            residue_entry['weak_hbond_donors']['ali'].append([atom_id, neighbor])

            elif atom_type.startswith("N"):
                residue_entry['nitrogens'].append(atom_id)

                if atom_name == 'N':
                    residue_entry['amide_nitrogens'].append(atom_id)
                elif (
                    (atom_name == 'NZ' and residue_name == 'LYS') or
                    ((atom_name == 'NE' or atom_name.startswith('NH')) and residue_name == 'ARG') or
                    ((atom_name == 'ND1' or atom_name == 'NE2') and residue_name == 'HIS')
                ):
                    residue_entry['positively_charged_nitrogens'].append(atom_id)

                is_donor = False
                for neighbor in bonds.get(atom_id, []):
                    neighbor_name = atoms[neighbor]['atom_name']
                    if "H" in neighbor_name:
                        residue_entry['hbond_donors_NH'].append([atom_id, neighbor])
                        is_donor = True
                if not is_donor:
                    residue_entry['hbond_acceptors_N'].append(atom_id)

            elif atom_type.startswith("O"):
                residue_entry['oxygens'].append(atom_id)

                if atom_name == 'O':
                    residue_entry['amide_oxygens'].append(atom_id)
                elif (
                    (atom_name.startswith('OE') and residue_name == 'GLU') or
                    (atom_name.startswith('OD') and residue_name == 'ASP')
                ):
                    residue_entry['negatively_charged_oxygens'].append(atom_id)

                is_donor = False
                for neighbor in bonds.get(atom_id, []):
                    neighbor_name = atoms[neighbor]['atom_name']
                    if "H" in neighbor_name:
                        residue_entry['hbond_donors_OH'].append([atom_id, neighbor])
                        is_donor = True
                if not is_donor:
                    residue_entry['hbond_acceptors_O'].append(atom_id)

            elif atom_type.startswith("S."):
                residue_entry['sulfurs'].append(atom_id)

    def _identify_aromatic_rings(self):
        atoms = self.atoms
        bonds = self.bonds

        def is_aromatic(atom_id):
            atom_type = atoms[atom_id]['atom_type']
            return atom_type in {"C.ar", "N.ar", "O.ar", "S.ar"}

        for chain_id, chain_data in self.reserved_residues_dic.items():
            for residue_seq, residue_data in chain_data.items():
                aromatic_atoms = [atom_id for atom_id in residue_data['atoms_id'] if is_aromatic(atom_id)]
                aromatic_set = set(aromatic_atoms)

                if not aromatic_atoms:
                    residue_data['aromatic_rings'] = []
                    continue

                found_signatures = set()
                ring_atoms = []

                def dfs(start_atom, current_atom, path):
                    if len(path) > 6:
                        return

                    for neighbor in bonds.get(current_atom, []):
                        if neighbor == start_atom and len(path) in (5, 6):
                            signature = frozenset(path)
                            if signature not in found_signatures:
                                found_signatures.add(signature)
                                ring_atoms.append(sorted(signature))
                            continue

                        if neighbor in aromatic_set and neighbor not in path and len(path) < 6:
                            dfs(start_atom, neighbor, path + [neighbor])

                for start_atom in aromatic_atoms:
                    dfs(start_atom, start_atom, [start_atom])

                residue_data['aromatic_rings'] = ring_atoms

    def _rbf_expand(self, distances, D_min=0., D_max=20., D_count=16):
        """
        distances: list or np.array, shape [N]

        return:
            shape [N * D_count]
        """
        distances = np.array(distances)

        centers = np.linspace(D_min, D_max, D_count)
        width = (D_max - D_min) / D_count

        rbf = np.exp(
            -((distances[..., None] - centers) ** 2) / (width ** 2)
        )

        return rbf.reshape(-1)

    def _build_local_frame(self, main_chain_xyzs):
        CA = main_chain_xyzs['CA']
        C = main_chain_xyzs['C']
        N = main_chain_xyzs['N']

        x = C - CA
        x = x / (np.linalg.norm(x) + 1e-8)

        y = N - CA
        y = y / (np.linalg.norm(y) + 1e-8)

        z = np.cross(x, y)
        z = z / (np.linalg.norm(z) + 1e-8)

        y = np.cross(z, x)

        return x, y, z

    def _find_connected_residues(self, distance_threshold=8.0):
        def get_main_chain_xyzs(chain_id, residue_seq):
            main_chain = {'C', 'O', 'N', 'CA'}
            main_chain_atom_xyzs = {}
            residue_atoms = self.reserved_residues_dic[chain_id][residue_seq]['atoms_id']

            for atom_id in residue_atoms:
                atom_name = self.atoms[atom_id]['atom_name']
                if atom_name in main_chain:
                    main_chain_atom_xyzs[atom_name] = self.atoms[atom_id]['xyz']
                    if len(main_chain_atom_xyzs) == 4:
                        break

            if not main_chain_atom_xyzs:
                fallback = np.array([0.0, 0.0, 0.0])
                for key in main_chain:
                    main_chain_atom_xyzs[key] = fallback.copy()
                return main_chain_atom_xyzs

            if len(main_chain_atom_xyzs) < 4:
                missing_keys = main_chain - set(main_chain_atom_xyzs.keys())
                existing_arrays = [main_chain_atom_xyzs[k] for k in main_chain_atom_xyzs]
                mean_xyz = np.mean(existing_arrays, axis=0)
                for key in missing_keys:
                    main_chain_atom_xyzs[key] = mean_xyz.copy()

            return main_chain_atom_xyzs

        # peptide bonded
        peptide_seen = set()
        for id_C, id_O, id_N in self.amides:
            residue_idx1 = self.atoms[id_C]['residue_idx']
            residue_idx2 = self.atoms[id_N]['residue_idx']
            pair = (residue_idx1, residue_idx2)
            if pair not in peptide_seen:
                peptide_seen.add(pair)
                self.connected_residues['peptide_bonded'].append([residue_idx1, residue_idx2])
            
        # cache main-chain coords
        main_chain_cache = {}
        for chain_id, chain_data in self.reserved_residues_dic.items():
            for residue_seq in chain_data:
                main_chain_cache[(chain_id, residue_seq)] = get_main_chain_xyzs(chain_id, residue_seq)

        virtual_cb_list = self.virtual_CB_list
        geometrical_edge_extend = self.connected_residues['geometrical_edge'].extend
        spatial_close_append = self.connected_residues['spatial_close'].append
        atoms_order = ['C', 'O', 'N', 'CA']

        for i in range(len(virtual_cb_list)):
            residue_idx1, chain_id1, residue_seq1, vCB_xyz1 = virtual_cb_list[i]
            main_chain_atom_xyzs1 = main_chain_cache[(chain_id1, residue_seq1)]

            for j in range(i + 1, len(virtual_cb_list)):
                residue_idx2, chain_id2, residue_seq2, vCB_xyz2 = virtual_cb_list[j]

                dist = np.linalg.norm(vCB_xyz1 - vCB_xyz2)
                if dist < distance_threshold:
                    spatial_close_append(([residue_idx1, residue_idx2], dist))

                    main_chain_atom_xyzs2 = main_chain_cache[(chain_id2, residue_seq2)]

                    distance_matrix = []
                    for atom_name1 in atoms_order:
                        row = []
                        xyz1 = main_chain_atom_xyzs1[atom_name1]
                        for atom_name2 in atoms_order:
                            delta = xyz1 - main_chain_atom_xyzs2[atom_name2]
                            row.append(np.linalg.norm(delta))
                        distance_matrix.append(row)

                    main_chain_distances_list1 = np.array([d for row in distance_matrix for d in row])
                    main_chain_distances_list2 = np.array([d for row in zip(*distance_matrix) for d in row])
                    rbf_feature1 = self._rbf_expand(main_chain_distances_list1,
                                                    D_min=0.0, D_max=20.0, D_count=16)
                    rbf_feature2 = self._rbf_expand(main_chain_distances_list2,
                                                    D_min=0.0, D_max=20.0, D_count=16)

                    x1, y1, z1 = self._build_local_frame(main_chain_atom_xyzs1)
                    x2, y2, z2 = self._build_local_frame(main_chain_atom_xyzs2)

                    delta_12 = vCB_xyz2 - vCB_xyz1

                    proj_12 = np.array([np.dot(delta_12, x1), np.dot(delta_12, y1), np.dot(delta_12, z1)])
                    proj_21 = np.array([np.dot(-delta_12, x2), np.dot(-delta_12, y2), np.dot(-delta_12, z2)])
                    proj_12 = proj_12 / (dist + 1e-8)
                    proj_21 = proj_21 / (dist + 1e-8)

                    dist_scalar = np.array([dist / distance_threshold])
                    geo_feature1 = np.concatenate([rbf_feature1, proj_12, dist_scalar])
                    geo_feature2 = np.concatenate([rbf_feature2, proj_21, dist_scalar])

                    geometrical_edge_extend([([residue_idx1, residue_idx2], geo_feature1),
                                             ([residue_idx2, residue_idx1], geo_feature2)])


    def _compute_dihedral_angle(self, p1, p2, p3, p4):
        b1 = p2 - p1
        b2 = p3 - p2
        b3 = p4 - p3

        n1 = np.cross(b1, b2)
        n2 = np.cross(b2, b3)

        n1_norm = np.linalg.norm(n1)
        n2_norm = np.linalg.norm(n2)
        b2_norm = np.linalg.norm(b2)

        if n1_norm < 1e-8 or n2_norm < 1e-8 or b2_norm < 1e-8:
            return 0.0

        n1 = n1 / n1_norm
        n2 = n2 / n2_norm
        b2 = b2 / b2_norm

        return math.atan2(np.dot(np.cross(n1, n2), b2), np.dot(n1, n2))

    def _compute_backbone_dihedrals(self):
        self.dihedral_features = {}  # residue_index -> [sinφ, cosφ, sinψ, cosψ, sinω, cosω]

        main_chain_cache = {}
        for chain_id, chain_data in self.reserved_residues_dic.items():
            for residue_seq, residue_data in chain_data.items():
                coords = {}
                for atom_id in residue_data['atoms_id']:
                    atom = self.atoms.get(atom_id)
                    if atom is None:
                        continue
                    if atom['atom_name'] in ('N', 'CA', 'C'):
                        coords[atom['atom_name']] = atom['xyz']
                if all(k in coords for k in ('N', 'CA', 'C')):
                    main_chain_cache[(chain_id, residue_seq)] = coords

        for chain_id, chain_data in self.reserved_residues_dic.items():
            seq_list = self.all_residue_seq[chain_id]
            for residue_seq, residue_data in chain_data.items():
                residue_idx = residue_data['residue_index']

                try:
                    pos = seq_list.index(residue_seq)
                except ValueError:
                    self.dihedral_features[residue_idx] = [0.0] * 6
                    continue

                dihedral = [0.0] * 6

                current_coords = main_chain_cache.get((chain_id, residue_seq))
                if current_coords is None:
                    self.dihedral_features[residue_idx] = dihedral
                    continue

                N_i = current_coords['N']
                CA_i = current_coords['CA']
                C_i = current_coords['C']

                # φ: C_{i-1} - N_i - CA_i - C_i
                if pos > 0:
                    prev_seq = seq_list[pos - 1]
                    prev_coords = main_chain_cache.get((chain_id, prev_seq))
                    if prev_coords is not None and 'C' in prev_coords:
                        phi = self._compute_dihedral_angle(
                            prev_coords['C'], N_i, CA_i, C_i
                        )
                        dihedral[0] = math.sin(phi)
                        dihedral[1] = math.cos(phi)

                # ψ: N_i - CA_i - C_i - N_{i+1}
                if pos < len(seq_list) - 1:
                    next_seq = seq_list[pos + 1]
                    next_coords = main_chain_cache.get((chain_id, next_seq))
                    if next_coords is not None and 'N' in next_coords:
                        psi = self._compute_dihedral_angle(
                            N_i, CA_i, C_i, next_coords['N']
                        )
                        dihedral[2] = math.sin(psi)
                        dihedral[3] = math.cos(psi)

                # ω: CA_{i-1} - C_{i-1} - N_i - CA_i
                if pos > 0:
                    prev_seq = seq_list[pos - 1]
                    prev_coords = main_chain_cache.get((chain_id, prev_seq))
                    if prev_coords is not None and 'CA' in prev_coords and 'C' in prev_coords:
                        omega = self._compute_dihedral_angle(
                            prev_coords['CA'], prev_coords['C'], N_i, CA_i
                        )
                        dihedral[4] = math.sin(omega)
                        dihedral[5] = math.cos(omega)

                self.dihedral_features[residue_idx] = dihedral


if __name__ == "__main__":
    import argparse
    from parse_ligand import Ligand
    from utils import residues, pdb_to_sybyl, restype_3to1
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r",
        type=str,
        default='../examples/protein.pdb',
        help='receptor file, pdb format'
    )
    args = parser.parse_args()

    lig_mol2_file = f'../examples/ligand.mol2'
    rec_pdb_file = f'../examples/protein.pdb'

    ligand = Ligand(lig_mol2_file)
    pocket_xyz = ligand.pocket_center
    molecule = Protein(rec_pdb_file, pocket_xyz=pocket_xyz, cutoff=120)
    print(molecule.atoms[3])
    print(molecule.bonds[3])
    #print(molecule.amides)
    
