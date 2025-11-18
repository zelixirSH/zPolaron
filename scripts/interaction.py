import math
from scripts.parse_ligand import Ligand
from scripts.parse_protein import Protein
import numpy as np
from itertools import combinations
from scripts.utils import vdw_radius


class Interaction_PL:
    def __init__(self, protein: Protein, ligand: Ligand):
        self.ligand = ligand
        self.protein = protein
        '''
        Types of interaction between receptor-ligand:
            Hydrophobic: 
                aliphatic-aromatic, aromatic-aliphatic, aliphatic-aliphatic,
                carbon-halogen, sulfur-aromatic
            Hydrogen bonding:
                O...H-O, O-H...O, O...H-N, N-H...O, N-H...N
            pi stacking:
                edge to face, face to face
            Weak hydrogen bonding:
                O...H-Caro, Caro-H...O, O...H-Cali, Cali-H...O
            Salt bridge:
                Positive_N...Negative_O, Negative_O...Positive_N
            Amide stacking:
                edge to face, face to face
            Cation pi:
                Positive_N...aromatic_C, aromatic_C...Positive_N
            Halogen bonding:
                O...Cl, N...Cl, S...Cl, O...Br, N...Br, S...Br, O...I, N...I, S...I
            Multipolar halogen interaction:
                amide_C...F, amide_C...Cl, amide_N...F, amide_N...Cl
        '''
        self.edge_index_interact_PL = [[], []] # torch.empty((2, 0), dtype=torch.long)
        self.edge_feature_dic = {}  #  {(lig_atom_idx, residue_idx): [feature]}

        self.interac_PL_type = [
        'hydrophobic_ali_aro', 'hydrophobic_aro_ali', 'hydrophobic_ali_ali', 
        'hydrophobic_C_hal', 'hydrophobic_S_aro',
        'hbond_acceptor_res', 'hbond_donor_res',
        'pi_stacking_e2f', 'pi_stacking_f2f',
        'weak_hbond_acceptor_res', 'weak_hbond_donor_res',
        'salt_bridge_N_O', 'salt_bridge_O_N',
        'amide_stacking_e2f', 'amide_stacking_f2f',
        'cation_pi_Npos_Caro', 'cation_pi_Caro_Npos',
        'halogen_bonding',
        'multipolar_halogen']

        self.edge_index_geometric_PL = [[], []]
        self.geometric_edge_feature_dic = {}

        self.get_interaction_feature(distance_threshold=6.0)

        
    def calculate_distance(self, atom1, atom2):
        xyz1 = atom1['xyz']
        xyz2 = atom2['xyz']
        return np.linalg.norm(xyz1 - xyz2)
    
    
    def calculate_angle(self, atom1, atom2, atom3):
        BA = atom1['xyz'] - atom2['xyz']
        BC = atom3['xyz'] - atom2['xyz']
        dot_product = np.dot(BA, BC)
        mag_BA = np.linalg.norm(BA)
        mag_BC = np.linalg.norm(BC)
        cos_theta = dot_product / (mag_BA * mag_BC)
        angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))
        angle_deg = np.degrees(angle)
        return angle_deg

    
    def calculate_normal_vector(self, atom1, atom2, atom3):
        AB = atom2['xyz'] - atom1['xyz']
        AC = atom3['xyz'] - atom1['xyz']
        normal_vector = np.cross(AB, AC)
        normal_vector = normal_vector / np.linalg.norm(normal_vector)
        return normal_vector
    
    
    def calculate_vector_angle(self, vector1, vector2):
        v1 = np.asarray(vector1)
        v2 = np.asarray(vector2)
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        cos_theta = np.clip(dot_product / (norm_v1 * norm_v2), -1.0, 1.0)
        angle_rad = np.arccos(cos_theta)
        angle_deg = np.degrees(angle_rad)
        return min(angle_deg, 180 - angle_deg)
    
    
    def calculate_ring_center(self, atoms):
        coords = np.array([atom['xyz'] for atom in atoms])
        center = np.mean(coords, axis=0)
        return center
        
        
    def _find_hydrophobic(self, residue_data, dis_cutoff=4):
        def add_interact_num(protein_atoms, ligand_atoms, interact_key):                
            for protein_atom_id in protein_atoms:
                protein_atom = self.protein.atoms[protein_atom_id]
                residue_idx = protein_atom['residue_idx']
                for ligand_atom_id in ligand_atoms:
                    ligand_atom = self.ligand.atoms[ligand_atom_id]
                    lig_atom_idx = ligand_atom['atom_index']
                    if (lig_atom_idx, residue_idx) not in self.edge_feature_dic:
                        continue
                    distance = self.calculate_distance(protein_atom, ligand_atom)
                    if distance <= dis_cutoff:
                        interaction_index = self.interac_PL_type.index(interact_key)
                        self.edge_feature_dic[(lig_atom_idx, residue_idx)][interaction_index] += 1
        # aliphatic-aromatic interactions
        add_interact_num(residue_data['aliphatic_carbons'], self.ligand.aromatic_carbons, 'hydrophobic_ali_aro')
        # aliphatic-aliphatic interactions
        add_interact_num(residue_data['aliphatic_carbons'], self.ligand.aliphatic_carbons, 'hydrophobic_ali_ali')
        # aromatic-aliphatic interactions
        add_interact_num(residue_data['aromatic_carbons'], self.ligand.aliphatic_carbons, 'hydrophobic_aro_ali')
        # carbon-halogen interactions
        add_interact_num(residue_data['carbons'], self.ligand.halogens, 'hydrophobic_C_hal')
        # sulfur-aromatic interactions
        add_interact_num(residue_data['sulfurs'], self.ligand.aromatic_carbons, 'hydrophobic_S_aro')
    
    
    def _find_hbond(self, residue_data, dis_cutoff=3.9, ang_cutoff=90):
        def add_interact_num(donor_ids, acceptor_atoms, interact_key, is_donor_protein=True):
            for donor in donor_ids:
                donor_atom = (self.protein.atoms if is_donor_protein else self.ligand.atoms)[donor[0]]
                donor_H = (self.protein.atoms if is_donor_protein else self.ligand.atoms)[donor[1]]
                for acceptor_id in acceptor_atoms:
                    acceptor_atom = (self.ligand.atoms if is_donor_protein else self.protein.atoms)[acceptor_id]
                    residue_idx = donor_atom['residue_idx'] if is_donor_protein else acceptor_atom['residue_idx']
                    lig_atom_idx = acceptor_atom['atom_index'] if is_donor_protein else donor_atom['atom_index']
                    if (lig_atom_idx, residue_idx) not in self.edge_feature_dic:
                        continue
                    distance = self.calculate_distance(donor_atom, acceptor_atom)
                    angle = self.calculate_angle(donor_atom, donor_H, acceptor_atom)
                    if distance <= dis_cutoff and angle >= ang_cutoff:
                        interaction_index = self.interac_PL_type.index(interact_key)
                        self.edge_feature_dic[(lig_atom_idx, residue_idx)][interaction_index] += 1
        # O...H-O
        add_interact_num(self.ligand.hbond_donors_OH, residue_data['hbond_acceptors_O'], 
                         'hbond_acceptor_res', is_donor_protein=False)
        # O...H-N
        add_interact_num(self.ligand.hbond_donors_NH, residue_data['hbond_acceptors_O'], 
                         'hbond_acceptor_res', is_donor_protein=False)
        # O-H...O
        add_interact_num(residue_data['hbond_donors_OH'], self.ligand.hbond_acceptors_O, 
                         'hbond_donor_res', is_donor_protein=True)
        # N-H...O
        add_interact_num(residue_data['hbond_donors_NH'], self.ligand.hbond_acceptors_O, 
                         'hbond_donor_res', is_donor_protein=True)
        # N-H...N
        add_interact_num(residue_data['hbond_donors_NH'], self.ligand.hbond_acceptors_N, 
                         'hbond_donor_res', is_donor_protein=True)
    
    
    def _find_pi_stacking(self, residue_data, dis_cutoff=4.0, theta_e2f=60, theta_f2f=30):
        for protein_aro_ring in residue_data['aromatic_rings']:
            protein_ring_atoms = [self.protein.atoms[atom_id] for atom_id in protein_aro_ring]
            xyz1 = self.calculate_ring_center(protein_ring_atoms)
            residue_idx = protein_ring_atoms[0]['residue_idx']
            protein_ring_normal_vec = self.calculate_normal_vector(protein_ring_atoms[0], 
                                                                   protein_ring_atoms[2], 
                                                                   protein_ring_atoms[4])
            for ligand_aro_ring in self.ligand.aromatic_rings:
                ligand_ring_atoms = [self.ligand.atoms[atom_id] for atom_id in ligand_aro_ring]
                xyz2 = self.calculate_ring_center(ligand_ring_atoms)
                ligand_ring_normal_vec = self.calculate_normal_vector(ligand_ring_atoms[0], 
                                                                      ligand_ring_atoms[2], 
                                                                      ligand_ring_atoms[4])
                distance = np.linalg.norm(xyz1 - xyz2)
                angle = self.calculate_vector_angle(protein_ring_normal_vec, ligand_ring_normal_vec)
                if distance <= (dis_cutoff + 1) and angle >= theta_e2f:
                    # edge to face
                    for ligand_ring_atom in ligand_ring_atoms:
                        lig_atom_idx = ligand_ring_atom['atom_index']
                        if (lig_atom_idx, residue_idx) not in self.edge_feature_dic:
                            continue
                        interaction_index = self.interac_PL_type.index('pi_stacking_e2f')
                        self.edge_feature_dic[(lig_atom_idx, residue_idx)][interaction_index] += 1
                elif distance <= dis_cutoff and angle <= theta_f2f:
                    # face to face
                    for ligand_ring_atom in ligand_ring_atoms:
                        lig_atom_idx = ligand_ring_atom['atom_index']
                        if (lig_atom_idx, residue_idx) not in self.edge_feature_dic:
                            continue
                        interaction_index = self.interac_PL_type.index('pi_stacking_f2f')
                        self.edge_feature_dic[(lig_atom_idx, residue_idx)][interaction_index] += 1
    

    def _find_weak_hbond(self, residue_data, dis_cutoff=3.6, ang_cutoff=130):
        def add_interact_num(donor_ids, acceptor_atoms, interact_key, is_donor_protein=True):
            for donor in donor_ids:
                donor_atom = (self.protein.atoms if is_donor_protein else self.ligand.atoms)[donor[0]]
                donor_H = (self.protein.atoms if is_donor_protein else self.ligand.atoms)[donor[1]]
                for acceptor in acceptor_atoms:
                    acceptor_atom = (self.ligand.atoms if is_donor_protein else self.protein.atoms)[acceptor]
                    residue_idx = donor_atom['residue_idx'] if is_donor_protein else acceptor_atom['residue_idx']
                    lig_atom_idx = acceptor_atom['atom_index'] if is_donor_protein else donor_atom['atom_index']
                    if (lig_atom_idx, residue_idx) not in self.edge_feature_dic:
                        continue
                    distance = self.calculate_distance(donor_atom, acceptor_atom)
                    angle = self.calculate_angle(donor_atom, donor_H, acceptor_atom)
                    if distance <= dis_cutoff and angle >= ang_cutoff:
                        interaction_index = self.interac_PL_type.index(interact_key)
                        self.edge_feature_dic[(lig_atom_idx, residue_idx)][interaction_index] += 1
        # O...H-Caro
        add_interact_num(self.ligand.weak_hbond_donors['aro'], residue_data['hbond_acceptors_O'], 
                         'weak_hbond_acceptor_res', is_donor_protein=False)
        # O...H-Cali
        add_interact_num(self.ligand.weak_hbond_donors['ali'], residue_data['hbond_acceptors_O'], 
                         'weak_hbond_acceptor_res', is_donor_protein=False)
        # Caro-H...O
        add_interact_num(residue_data['weak_hbond_donors']['aro'], self.ligand.hbond_acceptors_O, 
                         'weak_hbond_donor_res', is_donor_protein=True)
        # Cali-H...O
        add_interact_num(residue_data['weak_hbond_donors']['ali'], self.ligand.hbond_acceptors_O, 
                         'weak_hbond_donor_res', is_donor_protein=True)
        
        
    def _find_salt_bridge(self, residue_data, dis_cutoff=4.0):
        def add_interact_num(protein_atoms, ligand_atoms, interact_key):                
            for protein_atom_id in protein_atoms:
                protein_atom = self.protein.atoms[protein_atom_id]
                residue_idx = protein_atom['residue_idx']
                for ligand_atom_id in ligand_atoms:
                    ligand_atom = self.ligand.atoms[ligand_atom_id]
                    lig_atom_idx = ligand_atom['atom_index']
                    if (lig_atom_idx, residue_idx) not in self.edge_feature_dic:
                        continue
                    distance = self.calculate_distance(protein_atom, ligand_atom)
                    if distance <= dis_cutoff:
                        interaction_index = self.interac_PL_type.index(interact_key)
                        self.edge_feature_dic[(lig_atom_idx, residue_idx)][interaction_index] += 1
        # Positive_N...Negative_O
        add_interact_num(residue_data['positively_charged_nitrogens'], 
                         self.ligand.negatively_charged_oxygens, 'salt_bridge_N_O')
        # Negative_O...Positive_N
        add_interact_num(residue_data['positively_charged_nitrogens'], 
                         self.ligand.positively_charged_nitrogens, 'salt_bridge_N_O')
           
    
    def _find_amide_stacking(self, dis_cutoff=4.0, theta_e2f=60, theta_f2f=30):
        for protein_amide in self.protein.amides:
            protein_amide_atoms = [self.protein.atoms[atom_id] for atom_id in protein_amide]
            xyz1 = protein_amide_atoms[0]['xyz']
            residue_idx = protein_amide_atoms[0]['residue_idx']
            protein_amide_normal_vec = self.calculate_normal_vector(protein_amide_atoms[0], 
                                                                    protein_amide_atoms[1], 
                                                                    protein_amide_atoms[2])
            for ligand_aro_ring in self.ligand.aromatic_rings:
                ligand_ring_atoms = [self.ligand.atoms[atom_id] for atom_id in ligand_aro_ring]
                xyz2 = np.array(self.calculate_ring_center(ligand_ring_atoms))
                ligand_ring_normal_vec = self.calculate_normal_vector(ligand_ring_atoms[0], 
                                                                      ligand_ring_atoms[2], 
                                                                      ligand_ring_atoms[4])
                distance = np.linalg.norm(xyz1 - xyz2)
                angle = self.calculate_vector_angle(protein_amide_normal_vec, ligand_ring_normal_vec)
                if distance <= dis_cutoff and angle >= theta_e2f:
                    # edge to face
                    for ligand_ring_atom in ligand_ring_atoms:
                        lig_atom_idx = ligand_ring_atom['atom_index']
                        if (lig_atom_idx, residue_idx) not in self.edge_feature_dic:
                            continue
                        interaction_index = self.interac_PL_type.index('amide_stacking_e2f')
                        self.edge_feature_dic[(lig_atom_idx, residue_idx)][interaction_index] += 1
                elif distance <= dis_cutoff and angle <= theta_f2f:
                    # face to face
                    for ligand_ring_atom in ligand_ring_atoms:
                        lig_atom_idx = ligand_ring_atom['atom_index']
                        if (lig_atom_idx, residue_idx) not in self.edge_feature_dic:
                            continue
                        interaction_index = self.interac_PL_type.index('amide_stacking_f2f')
                        self.edge_feature_dic[(lig_atom_idx, residue_idx)][interaction_index] += 1

    
    def _find_cation_pi(self, residue_data, dis_cutoff=5.0):
        # residue N - ligand ring
        for nitrogen_id in residue_data['positively_charged_nitrogens']:
            nitrogen_atom = self.protein.atoms[nitrogen_id]
            xyz1 = nitrogen_atom['xyz']
            residue_idx = nitrogen_atom['residue_idx']
            for aro_ring in self.ligand.aromatic_rings:
                ring_atoms = [self.ligand.atoms[atom_id] for atom_id in aro_ring]
                xyz2 = np.array(self.calculate_ring_center(ring_atoms))
                distance = np.linalg.norm(xyz1 - xyz2)
                if distance < dis_cutoff:
                    for ligand_ring_atom in ring_atoms:
                        lig_atom_idx = ligand_ring_atom['atom_index']
                        if (lig_atom_idx, residue_idx) not in self.edge_feature_dic:
                            continue
                        interaction_index = self.interac_PL_type.index('cation_pi_Npos_Caro')
                        self.edge_feature_dic[(lig_atom_idx, residue_idx)][interaction_index] += 1
        # ligand N - residue ring
        for nitrogen_id in self.ligand.positively_charged_nitrogens:
            nitrogen_atom = self.ligand.atoms[nitrogen_id]
            xyz1 = nitrogen_atom['xyz']
            lig_atom_idx = nitrogen_atom['atom_index']
            for aro_ring in residue_data['aromatic_rings']:
                ring_atoms = [self.protein.atoms[atom_id] for atom_id in aro_ring]
                xyz2 = np.array(self.calculate_ring_center(ring_atoms))
                residue_idx = ring_atoms[0]['residue_idx']
                if (lig_atom_idx, residue_idx) not in self.edge_feature_dic:
                    continue
                distance = np.linalg.norm(xyz1 - xyz2)
                if distance < dis_cutoff:
                    interaction_index = self.interac_PL_type.index('cation_pi_Caro_Npos')
                    self.edge_feature_dic[(lig_atom_idx, residue_idx)][interaction_index] += 1
    

    def _find_halogen_bonding(self, residue_data, dis_cutoff=0.2):
        def add_interact_num(protein_atoms, halogen_ids):
            for atom_id in protein_atoms:
                atom = self.protein.atoms[atom_id]
                residue_idx = atom['residue_idx']
                protein_element_type = atom['atom_type'].split('.')[0]
                protein_atom_vdw_r = vdw_radius[protein_element_type]
                for halogen_id in halogen_ids:
                    halogen_atom = self.ligand.atoms[halogen_id]
                    lig_atom_idx = halogen_atom['atom_index']
                    halogen_type = halogen_atom['atom_type']
                    halogen_vdw_r = vdw_radius[halogen_type]
                    if (lig_atom_idx, residue_idx) not in self.edge_feature_dic:
                        continue
                    distance = self.calculate_distance(atom, halogen_atom)
                    if distance <= (protein_atom_vdw_r + halogen_vdw_r + dis_cutoff):
                        for atom_bonded_id in self.protein.bonds[atom_id]:
                            bonded_atom = self.protein.atoms[atom_bonded_id]
                            for halogen_bonded_id in self.ligand.bonds[halogen_id]:
                                halogen_bonded_atom = self.ligand.atoms[halogen_bonded_id]
                                alpha1 = self.calculate_angle(halogen_bonded_atom, halogen_atom, atom)
                                alpha2 = self.calculate_angle(bonded_atom, atom, halogen_atom)
                                if 130 <= alpha1 <= 180 and 90 <= alpha2 <= 150:
                                    interaction_index = self.interac_PL_type.index('halogen_bonding')
                                    self.edge_feature_dic[(lig_atom_idx, residue_idx)][interaction_index] += 1
        add_interact_num(residue_data['oxygens'], self.ligand.halogens)
        add_interact_num(residue_data['sulfurs'], self.ligand.halogens)
        add_interact_num(residue_data['nitrogens'], self.ligand.halogens)
        

    def _find_multipolar_halogen(self, residue_data, dis_cutoff=0.2):
        def add_interact_num(protein_atoms, halogen_ids):
            for atom_id in protein_atoms:
                atom = self.protein.atoms[atom_id]
                atom_name = atom['atom_name']
                residue_idx = atom['residue_idx']
                protein_element_type = atom['atom_type'].split('.')[0]
                protein_atom_vdw_r = vdw_radius[protein_element_type]
                for halogen_id in halogen_ids:
                    halogen_atom = self.ligand.atoms[halogen_id]
                    lig_atom_idx = halogen_atom['atom_index']
                    halogen_type = halogen_atom['atom_type']
                    halogen_vdw_r = vdw_radius[halogen_type]
                    if (lig_atom_idx, residue_idx) not in self.edge_feature_dic:
                        continue
                    distance = self.calculate_distance(atom, halogen_atom)
                    if distance <= (protein_atom_vdw_r + halogen_vdw_r + dis_cutoff):
                        for atom_bonded_id in self.protein.bonds[atom_id]:
                            bonded_atom = self.protein.atoms[atom_bonded_id]
                            if ((atom_name == 'C' and bonded_atom['atom_name'] == 'O') or \
                                (atom_name == 'N' and bonded_atom['atom_name'] == 'C')):
                                for halogen_bonded_id in self.ligand.bonds[halogen_id]:
                                    halogen_bonded_atom = self.ligand.atoms[halogen_bonded_id]
                                    theta1 = self.calculate_angle(halogen_bonded_atom, halogen_atom, atom)
                                    theta2 = self.calculate_angle(bonded_atom, atom, halogen_atom)
                                    if theta1 >= 140 and 70 <= theta2 <= 110:
                                        interaction_index = self.interac_PL_type.index('multipolar_halogen')
                                        self.edge_feature_dic[(lig_atom_idx, residue_idx)][interaction_index] += 1
        add_interact_num(residue_data['amide_carbons'],   self.ligand.halogens)
        add_interact_num(residue_data['amide_nitrogens'], self.ligand.halogens)


    def get_interaction_feature(self, distance_threshold=8.0):
        def get_main_chain_xyzs(chain_id, residue_seq):
            main_chain = ['C', 'O', 'N', 'CA']
            main_chain_atom_xyzs = {}
            for atom_id in self.protein.reserved_residues_dic[chain_id][residue_seq]['atoms_id']:
                atom_name = self.protein.atoms[atom_id]['atom_name']
                if len(main_chain_atom_xyzs) == 4:
                    break
                if atom_name in main_chain:
                    main_chain_atom_xyzs[atom_name] = self.protein.atoms[atom_id]['xyz']
            if len(main_chain_atom_xyzs) < 4:
                current_keys = main_chain_atom_xyzs.keys()
                missing_keys = main_chain - current_keys
                existing_arrays = [main_chain_atom_xyzs[k] for k in main_chain if k in main_chain_atom_xyzs]
                mean_xyz = np.mean(existing_arrays, axis=0)
                for key in missing_keys:
                    main_chain_atom_xyzs[key] = mean_xyz.copy()
            return main_chain_atom_xyzs
        
        for chain_id, chain_data in self.protein.reserved_residues_dic.items():
            for residue_seq, residue_data in chain_data.items():
                virt_CB_xyz = residue_data['virt_CB_xyz']
                for atom_id, atom_data in self.ligand.atoms.items():
                    lig_atom_xyz = atom_data['xyz']
                    distance = np.linalg.norm(virt_CB_xyz - lig_atom_xyz)
                    if distance > distance_threshold:
                        continue
                    
                    lig_atom_index = atom_data['atom_index']
                    residue_index = residue_data['residue_index']
                    
                    self.edge_index_interact_PL[0].append(lig_atom_index)
                    self.edge_index_interact_PL[1].append(residue_index)
                    edge_feature_list = [0] * len(self.interac_PL_type)
                    self.edge_feature_dic[(lig_atom_index, residue_index)] = edge_feature_list

                    main_chain_atom_xyzs = get_main_chain_xyzs(chain_id, residue_seq)
                    lig_main_chain_dist = []
                    for main_chain_atom in ['C', 'O', 'N', 'CA']:
                        dist_main_chain = np.linalg.norm(main_chain_atom_xyzs[main_chain_atom] - lig_atom_xyz)
                        lig_main_chain_dist.append(dist_main_chain)
                    lig_main_chain_dist.append(distance)
                    self.edge_index_geometric_PL[0].append(lig_atom_index)
                    self.edge_index_geometric_PL[1].append(residue_index)
                    self.geometric_edge_feature_dic[(lig_atom_index, residue_index)] = lig_main_chain_dist

                self._find_hydrophobic(residue_data, dis_cutoff=4)
                self._find_hbond(residue_data, dis_cutoff=3.9, ang_cutoff=90)
                self._find_pi_stacking(residue_data, dis_cutoff=5.5, theta_e2f=60, theta_f2f=30)
                self._find_weak_hbond(residue_data, dis_cutoff=3.6, ang_cutoff=130)
                self._find_salt_bridge(residue_data, dis_cutoff=4.0)
                self._find_cation_pi(residue_data, dis_cutoff=5.0)
                self._find_halogen_bonding(residue_data, dis_cutoff=0.2)
                self._find_multipolar_halogen(residue_data, dis_cutoff=0.2)
        self._find_amide_stacking(dis_cutoff=5.0, theta_e2f=60, theta_f2f=30)



class Interaction_PP:
    def __init__(self, protein: Protein):
        self.protein = protein
        '''
        Types of interaction between receptor residues:
            Hydrophobic: 
                aliphatic-aromatic, aliphatic-aliphatic, sulfur-aromatic
            Hydrogen bonding:
                O-H...O, N-H...O, N-H...N
            pi stacking:
                edge to face, face to face
            Weak hydrogen bonding:
                Caro-H...O, Cali-H...O
            Salt bridge:
                Positive_N...Negative_O
            Cation pi:
                Positive_N...aromatic_C
            Disulfide bonding:
                S...S
        '''
        self.edge_index_interact_PP = [[], []] #torch.empty((2, 0), dtype=torch.long)
        self.edge_feature_dic = {}  #  {(residue_idx1, residue_idx2): [feature]}

        self.interac_PP_type_full = [
        'hydrophobic_ali_aro', 'hydrophobic_aro_ali', 'hydrophobic_ali_ali', 'hydrophobic_S_aro', 'hydrophobic_aro_S',
        'hbond_OH_O', 'hbond_O_OH', 'hbond_NH_O', 'hbond_O_NH', 'hbond_NH_N', 'hbond_N_NH',
        'pi_stacking_e2f', 'pi_stacking_f2f',
        'weak_hbond_CHaro_O', 'weak_hbond_O_CHaro', 'weak_hbond_CHali_O', 'weak_hbond_O_CHali',
        'salt_bridge_N_O', 'salt_bridge_O_N',
        'cation_pi_Npos_Caro', 'cation_pi_Caro_Npos',
        'disulfide_bonding_S_S']

        self.interac_PP_type = [
        'hydrophobic_ali_aro', 'hydrophobic_ali_ali', 'hydrophobic_S_aro',
        'hbond_OH_O', 'hbond_NH_O', 'hbond_NH_N',
        'pi_stacking_e2f', 'pi_stacking_f2f',
        'weak_hbond_CHaro_O', 'weak_hbond_CHali_O', 
        'salt_bridge_N_O',
        'cation_pi_Npos_Caro',
        'disulfide_bonding_S_S']

        self.get_interaction_feature(distance_threshold=6.0)
    
    
    def calculate_distance(self, atom1, atom2):
        xyz1 = atom1['xyz']
        xyz2 = atom2['xyz']
        return np.linalg.norm(xyz1 - xyz2)
    
    
    def calculate_angle(self, atom1, atom2, atom3):
        BA = atom1['xyz'] - atom2['xyz']
        BC = atom3['xyz'] - atom2['xyz']
        dot_product = np.dot(BA, BC)
        mag_BA = np.linalg.norm(BA)
        mag_BC = np.linalg.norm(BC)
        cos_theta = dot_product / (mag_BA * mag_BC)
        angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))
        angle_deg = np.degrees(angle)
        return angle_deg

    
    def calculate_normal_vector(self, atom1, atom2, atom3):
        AB = atom2['xyz'] - atom1['xyz']
        AC = atom3['xyz'] - atom1['xyz']
        normal_vector = np.cross(AB, AC)
        normal_vector = normal_vector / np.linalg.norm(normal_vector)
        return normal_vector
    
    
    def calculate_vector_angle(self, vector1, vector2):
        v1 = np.asarray(vector1)
        v2 = np.asarray(vector2)
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        cos_theta = np.clip(dot_product / (norm_v1 * norm_v2), -1.0, 1.0)
        angle_rad = np.arccos(cos_theta)
        angle_deg = np.degrees(angle_rad)
        return min(angle_deg, 180 - angle_deg)
    
    
    def calculate_ring_center(self, atoms):
        coords = np.array([atom['xyz'] for atom in atoms])
        center = np.mean(coords, axis=0)
        return center
        

    def calculate_dihedral(self, p1, p2, p3, p4):
        """
        计算四个点(C₁-S₁-S₂-C₂)的二面角（单位：度）
        参数：
            p1, p2, p3, p4: 四个点的坐标，形状为 (3,) 的 numpy 数组
        返回：
            二面角(角度制，范围 [-180°, 180°])
        """
        # 计算向量 b1, b2, b3
        b1 = p2 - p1
        b2 = p3 - p2
        b3 = p4 - p3
        # 计算法向量 n1 和 n2
        n1 = np.cross(b1, b2)
        n2 = np.cross(b2, b3)
        # 归一化法向量和 b2
        n1 /= np.linalg.norm(n1)
        n2 /= np.linalg.norm(n2)
        b2_normalized = b2 / np.linalg.norm(b2)
        # 计算夹角（弧度）
        angle = np.arctan2(np.dot(np.cross(n1, n2), b2_normalized), np.dot(n1, n2))
        # 转换为角度并调整范围
        dihedral_deg = np.degrees(angle)
        return dihedral_deg


    def _find_hydrophobic(self, residue_data1, residue_data2, edge_feature_list: list, dis_cutoff=4):
        def add_interact_num(protein_atoms1, protein_atoms2, interact_key):
            for protein_atom_id1 in protein_atoms1:
                protein_atom1 = self.protein.atoms[protein_atom_id1]
                for protein_atom_id2 in protein_atoms2:
                    protein_atom2 = self.protein.atoms[protein_atom_id2]
                    distance = self.calculate_distance(protein_atom1, protein_atom2)
                    if distance <= dis_cutoff:
                        interaction_index = self.interac_PP_type.index(interact_key)
                        edge_feature_list[interaction_index] += 1
        # aliphatic-aromatic interactions
        add_interact_num(residue_data1['aliphatic_carbons'], 
                         residue_data2['aromatic_carbons'], 'hydrophobic_ali_aro')
        add_interact_num(residue_data1['aromatic_carbons'], 
                         residue_data2['aliphatic_carbons'], 'hydrophobic_ali_aro')
        # aliphatic-aliphatic interactions
        add_interact_num(residue_data1['aliphatic_carbons'], 
                         residue_data2['aliphatic_carbons'], 'hydrophobic_ali_ali')
        # sulfur-aromatic interactions
        add_interact_num(residue_data1['sulfurs'], 
                         residue_data2['aromatic_carbons'], 'hydrophobic_S_aro')
        add_interact_num(residue_data1['aromatic_carbons'], 
                         residue_data2['sulfurs'], 'hydrophobic_S_aro')


    def _find_hbond(self, residue_data1, residue_data2, edge_feature_list: list, 
                    dis_cutoff=3.9, ang_cutoff=140):
        def add_interact_num(donor_ids, acceptor_atoms, interact_key):
            for donor in donor_ids:
                donor_atom = self.protein.atoms[donor[0]]
                donor_H = self.protein.atoms[donor[1]]
                for acceptor in acceptor_atoms:
                    acceptor_atom = self.protein.atoms[acceptor]
                    distance = self.calculate_distance(donor_atom, acceptor_atom)
                    angle = self.calculate_angle(donor_atom, donor_H, acceptor_atom)
                    if distance <= dis_cutoff and angle >= ang_cutoff:
                        interaction_index = self.interac_PP_type.index(interact_key)
                        edge_feature_list[interaction_index] += 1
        # O-H...O
        add_interact_num(residue_data1['hbond_donors_OH'], 
                         residue_data2['hbond_acceptors_O'], 'hbond_OH_O')
        add_interact_num(residue_data2['hbond_donors_OH'], 
                         residue_data1['hbond_acceptors_O'], 'hbond_OH_O')
        # N-H...O
        add_interact_num(residue_data1['hbond_donors_NH'], 
                         residue_data2['hbond_acceptors_O'], 'hbond_NH_O')
        add_interact_num(residue_data2['hbond_donors_NH'], 
                         residue_data1['hbond_acceptors_O'], 'hbond_NH_O')
        # N-H...N
        add_interact_num(residue_data1['hbond_donors_NH'], 
                         residue_data2['hbond_acceptors_N'], 'hbond_NH_N')
        add_interact_num(residue_data2['hbond_donors_NH'], 
                         residue_data1['hbond_acceptors_N'], 'hbond_NH_N')


    def _find_pi_stacking(self, residue_data1, residue_data2, edge_feature_list: list,
                          dis_cutoff=4.5, theta_e2f=60, theta_f2f=30):
        for protein_aro_ring1 in residue_data1['aromatic_rings']:
            protein_ring_atoms1 = [self.protein.atoms[atom_id] for atom_id in protein_aro_ring1]
            xyz1 = np.array(self.calculate_ring_center(protein_ring_atoms1))
            protein_ring_normal_vec1 = self.calculate_normal_vector(protein_ring_atoms1[0], 
                                                                    protein_ring_atoms1[2], 
                                                                    protein_ring_atoms1[4])
            for protein_aro_ring2 in residue_data2['aromatic_rings']:
                protein_ring_atoms2 = [self.protein.atoms[atom_id] for atom_id in protein_aro_ring2]
                xyz2 = np.array(self.calculate_ring_center(protein_ring_atoms2))
                protein_ring_normal_vec2 = self.calculate_normal_vector(protein_ring_atoms2[0], 
                                                                        protein_ring_atoms2[2], 
                                                                        protein_ring_atoms2[4])
                distance = np.linalg.norm(xyz1 - xyz2)
                angle = self.calculate_vector_angle(protein_ring_normal_vec1, protein_ring_normal_vec2)
                if distance <= (dis_cutoff + 1) and angle >= theta_e2f:
                    # edge to face
                    interaction_index = self.interac_PP_type.index('pi_stacking_e2f')
                    edge_feature_list[interaction_index] += 1
                elif distance <= dis_cutoff and angle <= theta_f2f:
                    # face to face
                    interaction_index = self.interac_PP_type.index('pi_stacking_f2f')
                    edge_feature_list[interaction_index] += 1


    def _find_weak_hbond(self, residue_data1, residue_data2, edge_feature_list: list,
                         dis_cutoff=3.6, ang_cutoff=130):
        def add_interact_num(donor_ids, acceptor_atoms, interact_key):
            for donor in donor_ids:
                donor_atom = self.protein.atoms[donor[0]]
                donor_H = self.protein.atoms[donor[1]]
                for acceptor in acceptor_atoms:
                    acceptor_atom = self.protein.atoms[acceptor]
                    distance = self.calculate_distance(donor_atom, acceptor_atom)
                    angle = self.calculate_angle(donor_atom, donor_H, acceptor_atom)
                    if distance <= dis_cutoff and angle >= ang_cutoff:
                        interaction_index = self.interac_PP_type.index(interact_key)
                        edge_feature_list[interaction_index] += 1
        # Caro-H...O
        add_interact_num(residue_data1['weak_hbond_donors']['aro'], 
                         residue_data2['hbond_acceptors_O'], 'weak_hbond_CHaro_O')
        add_interact_num(residue_data2['weak_hbond_donors']['aro'], 
                         residue_data1['hbond_acceptors_O'], 'weak_hbond_CHaro_O')
        # Cali-H...O
        add_interact_num(residue_data1['weak_hbond_donors']['ali'], 
                         residue_data2['hbond_acceptors_O'], 'weak_hbond_CHali_O')        
        add_interact_num(residue_data2['weak_hbond_donors']['ali'], 
                         residue_data1['hbond_acceptors_O'], 'weak_hbond_CHali_O')        
        

    def _find_salt_bridge(self, residue_data1, residue_data2, edge_feature_list: list, dis_cutoff=4.0):
        def add_interact_num(protein_atoms1, protein_atoms2, interact_key):
            for protein_atom_id1 in protein_atoms1:
                protein_atom1 = self.protein.atoms[protein_atom_id1]
                for protein_atom_id2 in protein_atoms2:
                    protein_atom2 = self.protein.atoms[protein_atom_id2]
                    distance = self.calculate_distance(protein_atom1, protein_atom2)
                    if distance <= dis_cutoff:
                        interaction_index = self.interac_PP_type.index(interact_key)
                        edge_feature_list[interaction_index] += 1
        # Positive_N...Negative_O
        add_interact_num(residue_data1['positively_charged_nitrogens'], 
                         residue_data2['negatively_charged_oxygens'], 'salt_bridge_N_O')
        add_interact_num(residue_data1['negatively_charged_oxygens'], 
                         residue_data2['positively_charged_nitrogens'], 'salt_bridge_N_O')


    def _find_cation_pi(self, residue_data1, residue_data2, edge_feature_list: list, dis_cutoff=4.0):
        def add_interact_num(positive_nitrogens, aromatic_rings2, interact_key):
            for protein_nitrogen_id in positive_nitrogens:
                protein_nitrogen_atom = self.protein.atoms[protein_nitrogen_id]
                xyz1 = protein_nitrogen_atom['xyz']
                for protein_aro_ring in aromatic_rings2:
                    protein_ring_atoms = [self.protein.atoms[atom_id] for atom_id in protein_aro_ring]
                    xyz2 = np.array(self.calculate_ring_center(protein_ring_atoms))
                    distance = np.linalg.norm(xyz1 - xyz2)
                    if distance < dis_cutoff:
                        interaction_index = self.interac_PP_type.index(interact_key)
                        edge_feature_list[interaction_index] += 1
        # Positive_N...aromatic_C
        add_interact_num(residue_data1['positively_charged_nitrogens'], 
                         residue_data2['aromatic_rings'], 'cation_pi_Npos_Caro')
        add_interact_num(residue_data2['positively_charged_nitrogens'], 
                         residue_data1['aromatic_rings'], 'cation_pi_Npos_Caro')


    def _find_disulfide_bonding(self, residue_data1, residue_data2, edge_feature_list: list,
                                dis1=1.83, dis2=2.23, dih1=75, dih2=105):
        def get_CB_xyz(sulful_id):
            xyz_CB = None
            sulful_atom = self.protein.atoms[sulful_id]
            chain_id = sulful_atom['chain_id']
            sulful_residue_idx = sulful_atom['residue_idx']
            sulful_residue_seq = sulful_atom['residue_seq']
            sulful_residue_atom_id_list = self.protein.reserved_residues_dic[chain_id][sulful_residue_seq]['atoms_id']
            for atom_id in sulful_residue_atom_id_list:
                atom = self.protein.atoms[atom_id]
                if atom['atom_name'] == 'CB':
                    xyz_CB = atom['xyz']
                    break
            return xyz_CB
        
        for protein_sulfur_id1 in residue_data1['sulfurs']:
            protein_sulfur1 = self.protein.atoms[protein_sulfur_id1]
            if protein_sulfur1['residue_name'] == 'CYS':
                xyz_S1 = protein_sulfur1['xyz']
                xyz_C1 = get_CB_xyz(protein_sulfur_id1)
                for protein_sulfur_id2 in residue_data2['sulfurs']:
                    protein_sulfur2 = self.protein.atoms[protein_sulfur_id2]
                    if protein_sulfur2['residue_name'] == 'CYS':
                        xyz_S2 = protein_sulfur2['xyz']
                        xyz_C2 = get_CB_xyz(protein_sulfur_id2)
                        if xyz_C1 is None or xyz_C2 is None:
                            continue
                        distance = self.calculate_distance(protein_sulfur1, protein_sulfur2)
                        dihedral = self.calculate_dihedral(xyz_C1, xyz_S1, xyz_S2, xyz_C2)
                        if dis1 < distance < dis2 and dih1 < abs(dihedral) < dih2:
                            interaction_index = self.interac_PP_type.index('disulfide_bonding_S_S')
                            edge_feature_list[interaction_index] += 1


    def _process_interaction_pair(self, residue_data1, residue_data2):
        residue_idx1 = residue_data1['residue_index']
        residue_idx2 = residue_data2['residue_index']
        self.edge_index_interact_PP[0].append(residue_idx1)
        self.edge_index_interact_PP[1].append(residue_idx2)
        edge_feature_list = [0] * len(self.interac_PP_type)
        self._find_hydrophobic(residue_data1, residue_data2, edge_feature_list, dis_cutoff=4)
        self._find_hbond(residue_data1, residue_data2, edge_feature_list, dis_cutoff=3.9, ang_cutoff=140)
        self._find_pi_stacking(residue_data1, residue_data2, edge_feature_list, dis_cutoff=4.5, theta_e2f=60, theta_f2f=30)
        self._find_weak_hbond(residue_data1, residue_data2, edge_feature_list, dis_cutoff=3.6, ang_cutoff=130)
        self._find_salt_bridge(residue_data1, residue_data2, edge_feature_list, dis_cutoff=4.0)
        self._find_cation_pi(residue_data1, residue_data2, edge_feature_list, dis_cutoff=4.0)
        self._find_disulfide_bonding(residue_data1, residue_data2, edge_feature_list, dis1=1.83, dis2=2.23, dih1=75, dih2=105)
        self.edge_feature_dic[(residue_idx1, residue_idx2)] = edge_feature_list

        
    def get_interaction_feature(self, distance_threshold=6.0):
        # same chain
        chains = sorted(self.protein.reserved_residues_dic.keys())
        for chain_id in chains:
            residue_seqs = list(self.protein.reserved_residues_dic[chain_id].keys())
            residue_datas = list(self.protein.reserved_residues_dic[chain_id].values())
            for i in range(len(residue_seqs)):
                residue_seq1, residue_data1 = residue_seqs[i], residue_datas[i]
                virt_CB_xyz1 = residue_data1['virt_CB_xyz']
                for j in range(i+1, len(residue_seqs)):
                    residue_seq2, residue_data2 = residue_seqs[j], residue_datas[j]
                    virt_CB_xyz2 = residue_data2['virt_CB_xyz']
                    distance = np.linalg.norm(virt_CB_xyz1 - virt_CB_xyz2)
                    if distance > distance_threshold:
                        continue
                    self._process_interaction_pair(residue_data1, residue_data2)
        # different chains
        for chain1, chain2 in combinations(chains, 2):
            for residue_seq1, residue_data1 in self.protein.reserved_residues_dic[chain1].items():
                virt_CB_xyz1 = residue_data1['virt_CB_xyz']
                for residue_seq2, residue_data2 in self.protein.reserved_residues_dic[chain2].items():
                    virt_CB_xyz2 = residue_data2['virt_CB_xyz']
                    distance = np.linalg.norm(virt_CB_xyz1 - virt_CB_xyz2)
                    if distance > distance_threshold:
                        continue
                    self._process_interaction_pair(residue_data1, residue_data2)

  

if __name__ == "__main__":
    dataset = '/home/wzh/data/datasets/pdbbind2019/all/'
    pdb = '1a1e'
    #pdb = '3i1y'
    lig_mol2_file = f'{dataset}/{pdb}/{pdb}_ligand.mol2'
    rec_pdb_file = f'{dataset}/{pdb}/{pdb}_protein.pdb'

    ligand = Ligand(lig_mol2_file)
    pocket_xyz= ligand.pocket_center
    protein = Protein(rec_pdb_file, pocket_xyz=pocket_xyz, cutoff=20, cutoff_surface=12)

    interpp = Interaction_PP(protein)
    interpl = Interaction_PL(protein, ligand)

    n=0
    n0 = 0
    for pair, feat in interpl.edge_feature_dic.items():
        n += 1
        if feat.count(0) == len(feat):
            n0 += 1
    print(f'PL: total: {n} zero: {n0} none-zero: {n-n0}')

    n=0
    n0 = 0
    for pair, feat in interpp.edge_feature_dic.items():
        n += 1
        if feat.count(0) == len(feat):
            n0 += 1
    print(f'PP: total: {n} zero: {n0} none-zero: {n-n0}')

    #print(len(inter.hydrophobic['ali_aro']))
    #print(protein.aromatic_rings)
    #print(1385, protein.atoms[1385])
    #print(13, ligand.atoms['13'])