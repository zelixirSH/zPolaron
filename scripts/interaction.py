import math
from scripts.parse_ligand import Ligand
from scripts.parse_protein import Protein
import numpy as np
from itertools import combinations
from scripts.utils import vdw_radius
from collections import defaultdict


class Interaction_PL:
    def __init__(self, protein, ligand):
        self.ligand = ligand
        self.protein = protein

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
            'multipolar_halogen'
        ]
        self.interac_index = {name: i for i, name in enumerate(self.interac_PL_type)}

        self.edge_index_interact_PL = [[], []]
        
        self.edge_feature_dic = {}

        self.edge_index_geometric_PL = [[], []]
        self.geometric_edge_feature_dic = {}

        self.main_chain_cache = {}

        self.ligand_atom_ids_all = np.array(list(self.ligand.atoms.keys()), dtype=int)
        if self.ligand_atom_ids_all.size > 0:
            self.ligand_atom_xyzs_all = np.stack(
                [self.ligand.atoms[atom_id]['xyz'] for atom_id in self.ligand_atom_ids_all],
                axis=0
            )
        else:
            self.ligand_atom_xyzs_all = np.empty((0, 3), dtype=float)

        self.ligand_aromatic_atom_set = {
            atom_id for atom_id, atom in self.ligand.atoms.items()
            if atom['atom_type'] in {"C.ar", "N.ar", "O.ar", "S.ar"}
        }
        self.ligand_aromatic_carbons_set = set(self.ligand.aromatic_carbons)
        self.ligand_aliphatic_carbons_set = set(self.ligand.aliphatic_carbons)
        self.ligand_halogens_set = set(self.ligand.halogens)
        self.ligand_hbond_acceptors_O_set = set(self.ligand.hbond_acceptors_O)
        self.ligand_hbond_acceptors_N_set = set(self.ligand.hbond_acceptors_N)
        self.ligand_pos_n_set = set(self.ligand.positively_charged_nitrogens)
        self.ligand_neg_o_set = set(self.ligand.negatively_charged_oxygens)

        self.get_interaction_feature(distance_threshold=8.0)

    def calculate_distance(self, atom1, atom2):
        return np.linalg.norm(atom1['xyz'] - atom2['xyz'])

    def calculate_angle(self, atom1, atom2, atom3):
        ba = atom1['xyz'] - atom2['xyz']
        bc = atom3['xyz'] - atom2['xyz']
        norm_ba = np.linalg.norm(ba)
        norm_bc = np.linalg.norm(bc)
        if norm_ba == 0 or norm_bc == 0:
            return None
        cos_theta = np.dot(ba, bc) / (norm_ba * norm_bc)
        angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))
        return float(np.degrees(angle))

    def calculate_vector_angle(self, vector1, vector2):
        v1 = np.asarray(vector1, dtype=float)
        v2 = np.asarray(vector2, dtype=float)
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 == 0 or n2 == 0:
            return None
        cos_theta = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
        angle_deg = float(np.degrees(np.arccos(cos_theta)))
        return min(angle_deg, 180.0 - angle_deg)

    def calculate_ring_center(self, atoms):
        coords = np.array([atom['xyz'] for atom in atoms], dtype=float)
        if coords.size == 0:
            return np.zeros(3, dtype=float)
        return np.mean(coords, axis=0)

    def calculate_plane_normal(self, atoms):
        coords = np.array([atom['xyz'] for atom in atoms], dtype=float)
        if coords.shape[0] < 3:
            return None

        centered = coords - np.mean(coords, axis=0)
        try:
            _, _, vh = np.linalg.svd(centered, full_matrices=False)
            normal = vh[-1]
        except np.linalg.LinAlgError:
            normal = np.cross(centered[1] - centered[0], centered[2] - centered[0])

        norm = np.linalg.norm(normal)
        if norm == 0:
            return None
        return normal / norm

    def calculate_normal_vector(self, atom1, atom2, atom3):
        return self.calculate_plane_normal([atom1, atom2, atom3])

    def _get_main_chain_xyzs(self, chain_id, residue_seq):
        main_chain = {'C', 'O', 'N', 'CA'}
        main_chain_atom_xyzs = {}
        residue_atoms = self.protein.reserved_residues_dic[chain_id][residue_seq]['atoms_id']

        for atom_id in residue_atoms:
            atom_name = self.protein.atoms[atom_id]['atom_name']
            if atom_name in main_chain:
                main_chain_atom_xyzs[atom_name] = self.protein.atoms[atom_id]['xyz']
                if len(main_chain_atom_xyzs) == 4:
                    break

        if len(main_chain_atom_xyzs) == 0:
            zero = np.zeros(3, dtype=float)
            return {k: zero.copy() for k in ['C', 'O', 'N', 'CA']}

        if len(main_chain_atom_xyzs) < 4:
            missing_keys = main_chain - set(main_chain_atom_xyzs.keys())
            existing_arrays = [main_chain_atom_xyzs[k] for k in main_chain_atom_xyzs]
            if existing_arrays:
                mean_xyz = np.mean(existing_arrays, axis=0)
            else:
                mean_xyz = np.zeros(3, dtype=float)
            for key in missing_keys:
                main_chain_atom_xyzs[key] = mean_xyz.copy()

        return main_chain_atom_xyzs

    def _build_main_chain_cache(self):
        cache = {}
        for chain_id, chain_data in self.protein.reserved_residues_dic.items():
            for residue_seq in chain_data:
                cache[(chain_id, residue_seq)] = self._get_main_chain_xyzs(chain_id, residue_seq)
        return cache

    def _increment_feature_for_ligand_atoms(self, ligand_atom_ids, residue_idx, feature_key):
        feature_idx = self.interac_index[feature_key]
        for lig_atom_id in ligand_atom_ids:
            lig_atom_idx = self.ligand.atoms[lig_atom_id]['atom_index']
            edge = self.edge_feature_dic.get((lig_atom_idx, residue_idx))
            if edge is not None:
                edge[feature_idx] += 1

    def _update_distance_contacts(self, protein_atom_ids, ligand_atom_ids, residue_idx, feature_key, cutoff):
        protein_atom_ids = list(protein_atom_ids)
        ligand_atom_ids = np.asarray(list(ligand_atom_ids), dtype=int)

        if len(protein_atom_ids) == 0 or ligand_atom_ids.size == 0:
            return

        ligand_xyzs = np.stack([self.ligand.atoms[int(aid)]['xyz'] for aid in ligand_atom_ids], axis=0)
        feature_idx = self.interac_index[feature_key]

        for protein_atom_id in protein_atom_ids:
            protein_atom = self.protein.atoms[protein_atom_id]
            dists = np.linalg.norm(ligand_xyzs - protein_atom['xyz'], axis=1)
            mask = dists <= cutoff
            if not np.any(mask):
                continue

            for lig_atom_id in ligand_atom_ids[mask]:
                lig_atom_idx = self.ligand.atoms[int(lig_atom_id)]['atom_index']
                edge = self.edge_feature_dic.get((lig_atom_idx, residue_idx))
                if edge is not None:
                    edge[feature_idx] += 1

    def _update_angle_contacts(
        self,
        donor_pairs,
        acceptor_atom_ids,
        residue_idx,
        feature_key,
        donor_is_protein,
        distance_cutoff,
        angle_cutoff
    ):
        donor_pairs = list(donor_pairs)
        acceptor_atom_ids = np.asarray(list(acceptor_atom_ids), dtype=int)

        if len(donor_pairs) == 0 or acceptor_atom_ids.size == 0:
            return

        donor_dict = self.protein.atoms if donor_is_protein else self.ligand.atoms
        acceptor_dict = self.ligand.atoms if donor_is_protein else self.protein.atoms
        feature_idx = self.interac_index[feature_key]

        acceptor_xyzs = np.stack([acceptor_dict[int(aid)]['xyz'] for aid in acceptor_atom_ids], axis=0)

        for donor in donor_pairs:
            donor_id, donor_h_id = donor
            donor_atom = donor_dict[donor_id]
            donor_h_atom = donor_dict[donor_h_id]

            dists = np.linalg.norm(acceptor_xyzs - donor_atom['xyz'], axis=1)
            mask = dists <= distance_cutoff
            if not np.any(mask):
                continue

            for acceptor_id in acceptor_atom_ids[mask]:
                acceptor_atom = acceptor_dict[int(acceptor_id)]
                angle = self.calculate_angle(donor_atom, donor_h_atom, acceptor_atom)
                if angle is None or angle < angle_cutoff:
                    continue

                lig_atom_idx = acceptor_atom['atom_index'] if donor_is_protein else donor_atom['atom_index']
                res_idx = donor_atom['residue_idx'] if donor_is_protein else acceptor_atom['residue_idx']
                edge = self.edge_feature_dic.get((lig_atom_idx, res_idx))
                if edge is not None:
                    edge[feature_idx] += 1

    def _find_hydrophobic(self, residue_data, candidate_ligand_ids, dis_cutoff=4.0):
        candidate_set = set(candidate_ligand_ids)

        lig_aro = candidate_set & self.ligand_aromatic_carbons_set
        lig_ali = candidate_set & self.ligand_aliphatic_carbons_set
        lig_hal = candidate_set & self.ligand_halogens_set

        self._update_distance_contacts(
            residue_data['aliphatic_carbons'], lig_aro,
            residue_data['residue_index'], 'hydrophobic_ali_aro', dis_cutoff
        )
        self._update_distance_contacts(
            residue_data['aromatic_carbons'], lig_ali,
            residue_data['residue_index'], 'hydrophobic_aro_ali', dis_cutoff
        )
        self._update_distance_contacts(
            residue_data['aliphatic_carbons'], lig_ali,
            residue_data['residue_index'], 'hydrophobic_ali_ali', dis_cutoff
        )
        self._update_distance_contacts(
            residue_data['carbons'], lig_hal,
            residue_data['residue_index'], 'hydrophobic_C_hal', dis_cutoff
        )
        self._update_distance_contacts(
            residue_data['sulfurs'], lig_aro,
            residue_data['residue_index'], 'hydrophobic_S_aro', dis_cutoff
        )

    def _find_hbond(self, residue_data, candidate_ligand_ids, dis_cutoff=3.9, ang_cutoff=90):
        candidate_set = set(candidate_ligand_ids)

        # ligand donor -> protein acceptor
        self._update_angle_contacts(
            self.ligand.hbond_donors_OH,
            candidate_set & set(residue_data['hbond_acceptors_O']),
            residue_data['residue_index'],
            'hbond_acceptor_res',
            donor_is_protein=False,
            distance_cutoff=dis_cutoff,
            angle_cutoff=ang_cutoff
        )
        self._update_angle_contacts(
            self.ligand.hbond_donors_NH,
            candidate_set & set(residue_data['hbond_acceptors_O']),
            residue_data['residue_index'],
            'hbond_acceptor_res',
            donor_is_protein=False,
            distance_cutoff=dis_cutoff,
            angle_cutoff=ang_cutoff
        )
        self._update_angle_contacts(
            self.ligand.hbond_donors_NH,
            candidate_set & set(residue_data['hbond_acceptors_N']),
            residue_data['residue_index'],
            'hbond_acceptor_res',
            donor_is_protein=False,
            distance_cutoff=dis_cutoff,
            angle_cutoff=ang_cutoff
        )

        # protein donor -> ligand acceptor
        self._update_angle_contacts(
            residue_data['hbond_donors_OH'],
            candidate_set & self.ligand_hbond_acceptors_O_set,
            residue_data['residue_index'],
            'hbond_donor_res',
            donor_is_protein=True,
            distance_cutoff=dis_cutoff,
            angle_cutoff=ang_cutoff
        )
        self._update_angle_contacts(
            residue_data['hbond_donors_NH'],
            candidate_set & self.ligand_hbond_acceptors_O_set,
            residue_data['residue_index'],
            'hbond_donor_res',
            donor_is_protein=True,
            distance_cutoff=dis_cutoff,
            angle_cutoff=ang_cutoff
        )
        self._update_angle_contacts(
            residue_data['hbond_donors_NH'],
            candidate_set & self.ligand_hbond_acceptors_N_set,
            residue_data['residue_index'],
            'hbond_donor_res',
            donor_is_protein=True,
            distance_cutoff=dis_cutoff,
            angle_cutoff=ang_cutoff
        )

    def _find_weak_hbond(self, residue_data, candidate_ligand_ids, dis_cutoff=3.6, ang_cutoff=130):
        candidate_set = set(candidate_ligand_ids)

        self._update_angle_contacts(
            self.ligand.weak_hbond_donors['aro'],
            candidate_set & set(residue_data['hbond_acceptors_O']),
            residue_data['residue_index'],
            'weak_hbond_acceptor_res',
            donor_is_protein=False,
            distance_cutoff=dis_cutoff,
            angle_cutoff=ang_cutoff
        )
        self._update_angle_contacts(
            self.ligand.weak_hbond_donors['ali'],
            candidate_set & set(residue_data['hbond_acceptors_O']),
            residue_data['residue_index'],
            'weak_hbond_acceptor_res',
            donor_is_protein=False,
            distance_cutoff=dis_cutoff,
            angle_cutoff=ang_cutoff
        )

        self._update_angle_contacts(
            residue_data['weak_hbond_donors']['aro'],
            candidate_set & self.ligand_hbond_acceptors_O_set,
            residue_data['residue_index'],
            'weak_hbond_donor_res',
            donor_is_protein=True,
            distance_cutoff=dis_cutoff,
            angle_cutoff=ang_cutoff
        )
        self._update_angle_contacts(
            residue_data['weak_hbond_donors']['ali'],
            candidate_set & self.ligand_hbond_acceptors_O_set,
            residue_data['residue_index'],
            'weak_hbond_donor_res',
            donor_is_protein=True,
            distance_cutoff=dis_cutoff,
            angle_cutoff=ang_cutoff
        )

    def _find_salt_bridge(self, residue_data, candidate_ligand_ids, dis_cutoff=4.0):
        candidate_set = set(candidate_ligand_ids)

        # Positive_N (protein) ... Negative_O (ligand)
        self._update_distance_contacts(
            residue_data['positively_charged_nitrogens'],
            candidate_set & self.ligand_neg_o_set,
            residue_data['residue_index'],
            'salt_bridge_N_O',
            dis_cutoff
        )

        # Negative_O (protein) ... Positive_N (ligand)
        self._update_distance_contacts(
            residue_data['negatively_charged_oxygens'],
            candidate_set & self.ligand_pos_n_set,
            residue_data['residue_index'],
            'salt_bridge_O_N',
            dis_cutoff
        )

    def _find_pi_stacking(self, residue_data, candidate_ligand_ids, dis_cutoff=5.5, theta_e2f=60, theta_f2f=30):
        candidate_set = set(candidate_ligand_ids)

        if not residue_data['aromatic_rings']:
            return
        if len(candidate_set & self.ligand_aromatic_atom_set) == 0:
            return

        idx_e2f = self.interac_index['pi_stacking_e2f']
        idx_f2f = self.interac_index['pi_stacking_f2f']

        for protein_aro_ring in residue_data['aromatic_rings']:
            protein_ring_atoms = [self.protein.atoms[atom_id] for atom_id in protein_aro_ring]
            if len(protein_ring_atoms) < 3:
                continue

            protein_center = self.calculate_ring_center(protein_ring_atoms)
            protein_normal = self.calculate_plane_normal(protein_ring_atoms)
            if protein_normal is None:
                continue

            residue_idx = protein_ring_atoms[0]['residue_idx']

            for ligand_aro_ring in self.ligand.aromatic_rings:
                if len(candidate_set.intersection(ligand_aro_ring)) == 0:
                    continue

                ligand_ring_atoms = [self.ligand.atoms[atom_id] for atom_id in ligand_aro_ring]
                if len(ligand_ring_atoms) < 3:
                    continue

                ligand_center = self.calculate_ring_center(ligand_ring_atoms)
                ligand_normal = self.calculate_plane_normal(ligand_ring_atoms)
                if ligand_normal is None:
                    continue

                distance = np.linalg.norm(protein_center - ligand_center)
                angle = self.calculate_vector_angle(protein_normal, ligand_normal)
                if angle is None:
                    continue

                if distance <= (dis_cutoff + 1.0) and angle >= theta_e2f:
                    for lig_atom in ligand_ring_atoms:
                        lig_atom_idx = lig_atom['atom_index']
                        edge = self.edge_feature_dic.get((lig_atom_idx, residue_idx))
                        if edge is not None:
                            edge[idx_e2f] += 1
                elif distance <= dis_cutoff and angle <= theta_f2f:
                    for lig_atom in ligand_ring_atoms:
                        lig_atom_idx = lig_atom['atom_index']
                        edge = self.edge_feature_dic.get((lig_atom_idx, residue_idx))
                        if edge is not None:
                            edge[idx_f2f] += 1

    def _find_cation_pi(self, residue_data, candidate_ligand_ids, dis_cutoff=5.0):
        candidate_set = set(candidate_ligand_ids)

        if len(residue_data['positively_charged_nitrogens']) > 0 and len(candidate_set & self.ligand_aromatic_atom_set) > 0:
            idx_npos_caro = self.interac_index['cation_pi_Npos_Caro']
            for nitrogen_id in residue_data['positively_charged_nitrogens']:
                nitrogen_atom = self.protein.atoms[nitrogen_id]
                xyz1 = nitrogen_atom['xyz']
                residue_idx = nitrogen_atom['residue_idx']

                for aro_ring in self.ligand.aromatic_rings:
                    if len(candidate_set.intersection(aro_ring)) == 0:
                        continue
                    ring_atoms = [self.ligand.atoms[atom_id] for atom_id in aro_ring]
                    xyz2 = self.calculate_ring_center(ring_atoms)
                    distance = np.linalg.norm(xyz1 - xyz2)
                    if distance < dis_cutoff:
                        for lig_atom in ring_atoms:
                            lig_atom_idx = lig_atom['atom_index']
                            edge = self.edge_feature_dic.get((lig_atom_idx, residue_idx))
                            if edge is not None:
                                edge[idx_npos_caro] += 1

        if len(self.ligand.positively_charged_nitrogens) > 0 and len(residue_data['aromatic_rings']) > 0:
            idx_caro_npos = self.interac_index['cation_pi_Caro_Npos']
            for nitrogen_id in candidate_set & self.ligand_pos_n_set:
                nitrogen_atom = self.ligand.atoms[nitrogen_id]
                xyz1 = nitrogen_atom['xyz']
                lig_atom_idx = nitrogen_atom['atom_index']

                for aro_ring in residue_data['aromatic_rings']:
                    ring_atoms = [self.protein.atoms[atom_id] for atom_id in aro_ring]
                    if len(ring_atoms) < 3:
                        continue
                    residue_idx = ring_atoms[0]['residue_idx']
                    if self.edge_feature_dic.get((lig_atom_idx, residue_idx)) is None:
                        continue
                    xyz2 = self.calculate_ring_center(ring_atoms)
                    distance = np.linalg.norm(xyz1 - xyz2)
                    if distance < dis_cutoff:
                        edge = self.edge_feature_dic.get((lig_atom_idx, residue_idx))
                        if edge is not None:
                            edge[idx_caro_npos] += 1

    def _find_halogen_bonding(self, residue_data, candidate_ligand_ids, dis_cutoff=0.2):
        candidate_set = set(candidate_ligand_ids)
        halogen_ids = candidate_set & self.ligand_halogens_set
        if len(halogen_ids) == 0:
            return

        idx_halogen = self.interac_index['halogen_bonding']
        protein_acceptor_atoms = residue_data['oxygens'] + residue_data['sulfurs'] + residue_data['nitrogens']

        for atom_id in protein_acceptor_atoms:
            atom = self.protein.atoms[atom_id]
            residue_idx = atom['residue_idx']
            protein_element_type = atom['atom_type'].split('.')[0]
            protein_vdw_r = vdw_radius.get(protein_element_type)
            if protein_vdw_r is None:
                continue

            for halogen_id in halogen_ids:
                halogen_atom = self.ligand.atoms[halogen_id]
                lig_atom_idx = halogen_atom['atom_index']
                halogen_type = halogen_atom['atom_type'].split('.')[0]
                halogen_vdw_r = vdw_radius.get(halogen_type)
                if halogen_vdw_r is None:
                    continue

                edge = self.edge_feature_dic.get((lig_atom_idx, residue_idx))
                if edge is None:
                    continue

                distance = self.calculate_distance(atom, halogen_atom)
                if distance > (protein_vdw_r + halogen_vdw_r + dis_cutoff):
                    continue

                protein_bonded = self.protein.bonds.get(atom_id, [])
                halogen_bonded = self.ligand.bonds.get(halogen_id, [])

                for bonded_atom_id in protein_bonded:
                    bonded_atom = self.protein.atoms[bonded_atom_id]
                    for halogen_bonded_id in halogen_bonded:
                        halogen_bonded_atom = self.ligand.atoms[halogen_bonded_id]
                        alpha1 = self.calculate_angle(halogen_bonded_atom, halogen_atom, atom)
                        alpha2 = self.calculate_angle(bonded_atom, atom, halogen_atom)
                        if alpha1 is None or alpha2 is None:
                            continue
                        if 130 <= alpha1 <= 180 and 90 <= alpha2 <= 150:
                            edge[idx_halogen] += 1
                            break

    def _find_multipolar_halogen(self, residue_data, candidate_ligand_ids, dis_cutoff=0.2):
        candidate_set = set(candidate_ligand_ids)
        halogen_ids = candidate_set & self.ligand_halogens_set
        if len(halogen_ids) == 0:
            return

        idx_multi = self.interac_index['multipolar_halogen']
        protein_atoms = residue_data['amide_carbons'] + residue_data['amide_nitrogens']

        for atom_id in protein_atoms:
            atom = self.protein.atoms[atom_id]
            atom_name = atom['atom_name']
            residue_idx = atom['residue_idx']
            protein_element_type = atom['atom_type'].split('.')[0]
            protein_vdw_r = vdw_radius.get(protein_element_type)
            if protein_vdw_r is None:
                continue

            for halogen_id in halogen_ids:
                halogen_atom = self.ligand.atoms[halogen_id]
                lig_atom_idx = halogen_atom['atom_index']
                halogen_type = halogen_atom['atom_type'].split('.')[0]
                halogen_vdw_r = vdw_radius.get(halogen_type)
                if halogen_vdw_r is None:
                    continue

                edge = self.edge_feature_dic.get((lig_atom_idx, residue_idx))
                if edge is None:
                    continue

                distance = self.calculate_distance(atom, halogen_atom)
                if distance > (protein_vdw_r + halogen_vdw_r + dis_cutoff):
                    continue

                for bonded_atom_id in self.protein.bonds.get(atom_id, []):
                    bonded_atom = self.protein.atoms[bonded_atom_id]
                    if ((atom_name == 'C' and bonded_atom['atom_name'] == 'O') or
                        (atom_name == 'N' and bonded_atom['atom_name'] == 'C')):
                        for halogen_bonded_id in self.ligand.bonds.get(halogen_id, []):
                            halogen_bonded_atom = self.ligand.atoms[halogen_bonded_id]
                            theta1 = self.calculate_angle(halogen_bonded_atom, halogen_atom, atom)
                            theta2 = self.calculate_angle(bonded_atom, atom, halogen_atom)
                            if theta1 is None or theta2 is None:
                                continue
                            if theta1 >= 140 and 70 <= theta2 <= 110:
                                edge[idx_multi] += 1
                                break

    def _find_amide_stacking(self, candidate_atom_ids_by_residue_idx, dis_cutoff=5.0, theta_e2f=60, theta_f2f=30):
        if not self.protein.amides or not self.ligand.aromatic_rings:
            return

        idx_e2f = self.interac_index['amide_stacking_e2f']
        idx_f2f = self.interac_index['amide_stacking_f2f']

        for protein_amide in self.protein.amides:
            amide_atoms = [self.protein.atoms[atom_id] for atom_id in protein_amide]
            if len(amide_atoms) < 3:
                continue

            residue_idx = amide_atoms[0]['residue_idx']
            candidate_set = candidate_atom_ids_by_residue_idx.get(residue_idx)
            if not candidate_set:
                continue

            amide_center = self.calculate_ring_center(amide_atoms)
            amide_normal = self.calculate_plane_normal(amide_atoms)
            if amide_normal is None:
                continue

            for ligand_ring in self.ligand.aromatic_rings:
                if len(candidate_set.intersection(ligand_ring)) == 0:
                    continue

                ligand_ring_atoms = [self.ligand.atoms[atom_id] for atom_id in ligand_ring]
                if len(ligand_ring_atoms) < 3:
                    continue

                ligand_center = self.calculate_ring_center(ligand_ring_atoms)
                ligand_normal = self.calculate_plane_normal(ligand_ring_atoms)
                if ligand_normal is None:
                    continue

                distance = np.linalg.norm(amide_center - ligand_center)
                angle = self.calculate_vector_angle(amide_normal, ligand_normal)
                if angle is None:
                    continue

                if distance <= dis_cutoff and angle >= theta_e2f:
                    for lig_atom in ligand_ring_atoms:
                        lig_atom_idx = lig_atom['atom_index']
                        edge = self.edge_feature_dic.get((lig_atom_idx, residue_idx))
                        if edge is not None:
                            edge[idx_e2f] += 1
                elif distance <= dis_cutoff and angle <= theta_f2f:
                    for lig_atom in ligand_ring_atoms:
                        lig_atom_idx = lig_atom['atom_index']
                        edge = self.edge_feature_dic.get((lig_atom_idx, residue_idx))
                        if edge is not None:
                            edge[idx_f2f] += 1

    def _rbf_expand(self, distances, D_min=0.0, D_max=8.0, D_count=8):
        D_mu = np.linspace(D_min, D_max, D_count)
        D_sigma = (D_max - D_min) / D_count
        distances = np.asarray(distances)[..., None]

        rbf = np.exp(-((distances - D_mu) / D_sigma) ** 2)

        return rbf.reshape(-1)
    
    def _build_local_frame(self, main_chain_xyzs):
        N = main_chain_xyzs['N']
        CA = main_chain_xyzs['CA']
        C = main_chain_xyzs['C']

        x = C - CA
        x /= np.linalg.norm(x) + 1e-8

        y = N - CA
        y /= np.linalg.norm(y) + 1e-8

        z = np.cross(x, y)
        z /= np.linalg.norm(z) + 1e-8

        y = np.cross(z, x)
        y /= np.linalg.norm(y) + 1e-8

        return x, y, z

    def get_interaction_feature(self, distance_threshold=8.0):
        self.main_chain_cache = self._build_main_chain_cache()

        candidate_atom_ids_by_residue_idx = {}

        if self.ligand_atom_ids_all.size == 0:
            self._find_amide_stacking(candidate_atom_ids_by_residue_idx, dis_cutoff=5.0, theta_e2f=60, theta_f2f=30)
            return

        for chain_id, chain_data in self.protein.reserved_residues_dic.items():
            for residue_seq, residue_data in chain_data.items():
                virt_CB_xyz = residue_data.get('virt_CB_xyz', None)
                if virt_CB_xyz is None:
                    continue
                if np.allclose(virt_CB_xyz, np.array([99.0, 99.0, 99.0])):
                    continue

                distances = np.linalg.norm(self.ligand_atom_xyzs_all - virt_CB_xyz, axis=1)
                candidate_mask = distances <= distance_threshold
                if not np.any(candidate_mask):
                    continue

                residue_idx = residue_data['residue_index']
                candidate_ligand_ids = self.ligand_atom_ids_all[candidate_mask].tolist()
                candidate_atom_ids_by_residue_idx[residue_idx] = set(candidate_ligand_ids)

                main_chain_atom_xyzs = self.main_chain_cache[(chain_id, residue_seq)]
                x_axis, y_axis, z_axis = self._build_local_frame(main_chain_atom_xyzs)

                for lig_atom_id in candidate_ligand_ids:
                    lig_atom_data = self.ligand.atoms[lig_atom_id]
                    lig_atom_index = lig_atom_data['atom_index']
                    lig_xyz = lig_atom_data['xyz']

                    key = (lig_atom_index, residue_idx)

                    if key not in self.geometric_edge_feature_dic:
                        # interaction edge
                        self.edge_index_interact_PL[0].append(lig_atom_index)
                        self.edge_index_interact_PL[1].append(residue_idx)
                        self.edge_feature_dic[key] = np.zeros(len(self.interac_PL_type),
                                                              dtype=np.int32)
                        self.edge_index_geometric_PL[0].append(lig_atom_index)
                        self.edge_index_geometric_PL[1].append(residue_idx)

                        lig_main_chain_dist = []
                        for main_chain_atom in ['C', 'O', 'N', 'CA']:
                            lig_main_chain_dist.append(
                                np.linalg.norm(main_chain_atom_xyzs[main_chain_atom] - lig_xyz)
                            )
                        lig_main_chain_dist.append(np.linalg.norm(virt_CB_xyz - lig_xyz))

                        dist_arr = np.array(lig_main_chain_dist)

                        rbf_feature = self._rbf_expand(dist_arr, D_min=0.0, D_max=8.0, D_count=8)

                        delta = lig_xyz - virt_CB_xyz
                        norm_delta = np.linalg.norm(delta) + 1e-8

                        proj_feature = np.array([np.dot(delta, x_axis),np.dot(delta, y_axis),
                                                 np.dot(delta, z_axis)]) / norm_delta

                        geo_feature = np.concatenate([rbf_feature, proj_feature])

                        self.geometric_edge_feature_dic[key] = geo_feature

                # interaction features
                self._find_hydrophobic(residue_data, candidate_ligand_ids, dis_cutoff=4.0)
                self._find_hbond(residue_data, candidate_ligand_ids, dis_cutoff=3.9, ang_cutoff=90)
                self._find_pi_stacking(residue_data, candidate_ligand_ids, dis_cutoff=5.5, theta_e2f=60, theta_f2f=30)
                self._find_weak_hbond(residue_data, candidate_ligand_ids, dis_cutoff=3.6, ang_cutoff=130)
                self._find_salt_bridge(residue_data, candidate_ligand_ids, dis_cutoff=4.0)
                self._find_cation_pi(residue_data, candidate_ligand_ids, dis_cutoff=5.0)
                self._find_halogen_bonding(residue_data, candidate_ligand_ids, dis_cutoff=0.2)
                self._find_multipolar_halogen(residue_data, candidate_ligand_ids, dis_cutoff=0.2)

        self._find_amide_stacking(candidate_atom_ids_by_residue_idx, dis_cutoff=5.0, theta_e2f=60, theta_f2f=30)



class Interaction_PP:
    def __init__(self, protein):
        self.protein = protein

        self.interac_PP_type_full = [
            'hydrophobic_ali_aro', 'hydrophobic_aro_ali', 'hydrophobic_ali_ali',
            'hydrophobic_S_aro', 'hydrophobic_aro_S',
            'hbond_OH_O', 'hbond_O_OH', 'hbond_NH_O', 'hbond_O_NH', 'hbond_NH_N', 'hbond_N_NH',
            'pi_stacking_e2f', 'pi_stacking_f2f',
            'weak_hbond_CHaro_O', 'weak_hbond_O_CHaro', 'weak_hbond_CHali_O', 'weak_hbond_O_CHali',
            'salt_bridge_N_O', 'salt_bridge_O_N',
            'cation_pi_Npos_Caro', 'cation_pi_Caro_Npos',
            'disulfide_bonding_S_S'
        ]

        self.interac_PP_type = [
            'hydrophobic_ali_aro', 'hydrophobic_ali_ali', 'hydrophobic_S_aro',
            'hbond_OH_O', 'hbond_NH_O', 'hbond_NH_N',
            'pi_stacking_e2f', 'pi_stacking_f2f',
            'weak_hbond_CHaro_O', 'weak_hbond_CHali_O',
            'salt_bridge_N_O',
            'cation_pi_Npos_Caro',
            'disulfide_bonding_S_S'
        ]

        self.interac_map = {name: i for i, name in enumerate(self.interac_PP_type)}

        self.edge_index_interact_PP = [[], []]
        self.edge_feature_dic = {}  # {(residue_idx1, residue_idx2): np.ndarray}

        self.residue_records = self._build_residue_records()
        self.get_interaction_feature(distance_threshold=6.0)

    def calculate_distance(self, atom1, atom2):
        return np.linalg.norm(atom1['xyz'] - atom2['xyz'])

    def _distance_xyz(self, xyz1, xyz2):
        return np.linalg.norm(np.asarray(xyz1, dtype=float) - np.asarray(xyz2, dtype=float))

    def _angle_from_xyz(self, p1, p2, p3):
        ba = np.asarray(p1, dtype=float) - np.asarray(p2, dtype=float)
        bc = np.asarray(p3, dtype=float) - np.asarray(p2, dtype=float)
        n1 = np.linalg.norm(ba)
        n2 = np.linalg.norm(bc)
        if n1 == 0 or n2 == 0:
            return None
        cos_theta = np.dot(ba, bc) / (n1 * n2)
        angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))
        return float(np.degrees(angle))

    def calculate_angle(self, atom1, atom2, atom3):
        return self._angle_from_xyz(atom1['xyz'], atom2['xyz'], atom3['xyz'])

    def calculate_normal_vector(self, atom1, atom2, atom3):
        normal = np.cross(atom2['xyz'] - atom1['xyz'], atom3['xyz'] - atom1['xyz'])
        norm = np.linalg.norm(normal)
        if norm == 0:
            return None
        return normal / norm

    def calculate_vector_angle(self, vector1, vector2):
        v1 = np.asarray(vector1, dtype=float)
        v2 = np.asarray(vector2, dtype=float)
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 == 0 or n2 == 0:
            return None
        cos_theta = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
        angle_deg = float(np.degrees(np.arccos(cos_theta)))
        return min(angle_deg, 180.0 - angle_deg)

    def calculate_ring_center(self, atoms):
        coords = np.array([atom['xyz'] for atom in atoms], dtype=float)
        if coords.size == 0:
            return np.zeros(3, dtype=float)
        return coords.mean(axis=0)

    def calculate_dihedral(self, p1, p2, p3, p4):
        b1 = np.asarray(p2, dtype=float) - np.asarray(p1, dtype=float)
        b2 = np.asarray(p3, dtype=float) - np.asarray(p2, dtype=float)
        b3 = np.asarray(p4, dtype=float) - np.asarray(p3, dtype=float)

        n1 = np.cross(b1, b2)
        n2 = np.cross(b2, b3)

        n1_norm = np.linalg.norm(n1)
        n2_norm = np.linalg.norm(n2)
        b2_norm = np.linalg.norm(b2)

        if n1_norm < 1e-8 or n2_norm < 1e-8 or b2_norm < 1e-8:
            return 0.0

        n1 = n1 / n1_norm
        n2 = n2 / n2_norm
        b2_unit = b2 / b2_norm

        angle = np.arctan2(np.dot(np.cross(n1, n2), b2_unit), np.dot(n1, n2))
        return float(np.degrees(angle))

    def _build_atom_group_cache(self, atom_ids):
        seen = set()
        uniq_ids = []
        xyzs = []

        for atom_id in atom_ids:
            if atom_id in seen:
                continue
            seen.add(atom_id)
            uniq_ids.append(atom_id)
            xyzs.append(self.protein.atoms[atom_id]['xyz'])

        if not uniq_ids:
            return {
                'ids': np.empty(0, dtype=int),
                'xyzs': np.empty((0, 3), dtype=float)
            }

        return {
            'ids': np.array(uniq_ids, dtype=int),
            'xyzs': np.stack(xyzs, axis=0).astype(float)
        }

    def _build_donor_group_cache(self, donor_pairs):
        seen = set()
        uniq_pairs = []
        donor_xyzs = []
        donor_h_xyzs = []

        for donor_id, donor_h_id in donor_pairs:
            key = (donor_id, donor_h_id)
            if key in seen:
                continue
            seen.add(key)
            uniq_pairs.append(key)
            donor_xyzs.append(self.protein.atoms[donor_id]['xyz'])
            donor_h_xyzs.append(self.protein.atoms[donor_h_id]['xyz'])

        if not uniq_pairs:
            return {
                'pairs': [],
                'donor_xyzs': np.empty((0, 3), dtype=float),
                'donor_h_xyzs': np.empty((0, 3), dtype=float)
            }

        return {
            'pairs': uniq_pairs,
            'donor_xyzs': np.stack(donor_xyzs, axis=0).astype(float),
            'donor_h_xyzs': np.stack(donor_h_xyzs, axis=0).astype(float)
        }

    def _get_cb_xyz(self, residue_data):
        for atom_id in residue_data['atoms_id']:
            atom = self.protein.atoms[atom_id]
            if atom['atom_name'] == 'CB':
                return atom['xyz']
        return None

    def _plane_normal_from_atoms(self, atom_ids):
        coords = np.array([self.protein.atoms[atom_id]['xyz'] for atom_id in atom_ids], dtype=float)
        if coords.shape[0] < 3:
            return None

        centered = coords - coords.mean(axis=0)
        try:
            _, _, vh = np.linalg.svd(centered, full_matrices=False)
            normal = vh[-1]
        except np.linalg.LinAlgError:
            normal = np.cross(coords[1] - coords[0], coords[2] - coords[0])

        norm = np.linalg.norm(normal)
        if norm < 1e-8:
            return None
        return normal / norm

    def _build_aromatic_ring_cache(self, residue_data):
        raw_rings = residue_data.get('aromatic_rings', [])
        seen = set()
        uniq_rings = []
        ring_centers = []
        ring_normals = []

        for ring in raw_rings:
            ring_sig = tuple(sorted(ring))
            if ring_sig in seen:
                continue
            seen.add(ring_sig)
            uniq_rings.append(ring_sig)

            ring_atoms = [self.protein.atoms[atom_id] for atom_id in ring_sig]
            ring_centers.append(self.calculate_ring_center(ring_atoms))
            ring_normals.append(self._plane_normal_from_atoms(ring_sig))

        return {
            'rings': uniq_rings,
            'centers': ring_centers,
            'normals': ring_normals
        }

    def _build_residue_records(self):
        records = []
        for chain_id, chain_data in self.protein.reserved_residues_dic.items():
            for residue_seq, residue_data in chain_data.items():
                virt_cb = residue_data.get('virt_CB_xyz', None)
                if virt_cb is None:
                    continue

                record = {
                    'chain_id': chain_id,
                    'residue_seq': residue_seq,
                    'residue_idx': residue_data['residue_index'],
                    'residue_name': residue_data['residue_name'],
                    'virt_CB_xyz': np.asarray(virt_cb, dtype=float),
                    'data': residue_data,
                    'cache': {
                        'aliphatic_carbons': self._build_atom_group_cache(residue_data['aliphatic_carbons']),
                        'aromatic_carbons': self._build_atom_group_cache(residue_data['aromatic_carbons']),
                        'sulfurs': self._build_atom_group_cache(residue_data['sulfurs']),
                        'oxygens': self._build_atom_group_cache(residue_data['oxygens']),
                        'nitrogens': self._build_atom_group_cache(residue_data['nitrogens']),
                        'positively_charged_nitrogens': self._build_atom_group_cache(residue_data['positively_charged_nitrogens']),
                        'negatively_charged_oxygens': self._build_atom_group_cache(residue_data['negatively_charged_oxygens']),
                        'hbond_acceptors_O': self._build_atom_group_cache(residue_data['hbond_acceptors_O']),
                        'hbond_acceptors_N': self._build_atom_group_cache(residue_data['hbond_acceptors_N']),
                        'hbond_donors_OH': self._build_donor_group_cache(residue_data['hbond_donors_OH']),
                        'hbond_donors_NH': self._build_donor_group_cache(residue_data['hbond_donors_NH']),
                        'weak_hbond_donors_aro': self._build_donor_group_cache(residue_data['weak_hbond_donors']['aro']),
                        'weak_hbond_donors_ali': self._build_donor_group_cache(residue_data['weak_hbond_donors']['ali']),
                        'aromatic_rings': self._build_aromatic_ring_cache(residue_data),
                        'cb_xyz': self._get_cb_xyz(residue_data)
                    }
                }
                records.append(record)

        records.sort(key=lambda x: x['residue_idx'])
        return records

    def _pairwise_distances(self, xyzs1, xyzs2):
        if xyzs1.size == 0 or xyzs2.size == 0:
            return None
        return np.linalg.norm(xyzs1[:, None, :] - xyzs2[None, :, :], axis=2)

    def _count_distance_contacts(self, group1, group2, feature_idx, edge_feature_list, cutoff):
        dmat = self._pairwise_distances(group1['xyzs'], group2['xyzs'])
        if dmat is None:
            return
        edge_feature_list[feature_idx] += int(np.count_nonzero(dmat <= cutoff))

    def _count_donor_acceptor_contacts(self, donor_group, acceptor_group, feature_idx, edge_feature_list,
                                       distance_cutoff, angle_cutoff):
        donor_pairs = donor_group['pairs']
        donor_xyzs = donor_group['donor_xyzs']
        donor_h_xyzs = donor_group['donor_h_xyzs']
        acceptor_xyzs = acceptor_group['xyzs']

        if len(donor_pairs) == 0 or acceptor_xyzs.size == 0:
            return

        dmat = self._pairwise_distances(donor_xyzs, acceptor_xyzs)
        if dmat is None:
            return

        hits = np.argwhere(dmat <= distance_cutoff)
        if hits.size == 0:
            return

        for donor_idx, acceptor_idx in hits:
            angle = self._angle_from_xyz(
                donor_xyzs[donor_idx],
                donor_h_xyzs[donor_idx],
                acceptor_xyzs[acceptor_idx]
            )
            if angle is None or angle < angle_cutoff:
                continue
            edge_feature_list[feature_idx] += 1

    def _find_hydrophobic(self, residue_data1, residue_data2, edge_feature_list, dis_cutoff=4.0):
        c1 = residue_data1['cache']
        c2 = residue_data2['cache']

        self._count_distance_contacts(
            c1['aliphatic_carbons'], c2['aromatic_carbons'],
            self.interac_map['hydrophobic_ali_aro'], edge_feature_list, dis_cutoff
        )
        self._count_distance_contacts(
            c1['aromatic_carbons'], c2['aliphatic_carbons'],
            self.interac_map['hydrophobic_ali_aro'], edge_feature_list, dis_cutoff
        )
        self._count_distance_contacts(
            c1['aliphatic_carbons'], c2['aliphatic_carbons'],
            self.interac_map['hydrophobic_ali_ali'], edge_feature_list, dis_cutoff
        )
        self._count_distance_contacts(
            c1['sulfurs'], c2['aromatic_carbons'],
            self.interac_map['hydrophobic_S_aro'], edge_feature_list, dis_cutoff
        )
        self._count_distance_contacts(
            c1['aromatic_carbons'], c2['sulfurs'],
            self.interac_map['hydrophobic_S_aro'], edge_feature_list, dis_cutoff
        )

    def _find_hbond(self, residue_data1, residue_data2, edge_feature_list, dis_cutoff=3.9, ang_cutoff=140):
        c1 = residue_data1['cache']
        c2 = residue_data2['cache']
        idx = self.interac_map['hbond_OH_O']

        self._count_donor_acceptor_contacts(
            c1['hbond_donors_OH'], c2['hbond_acceptors_O'],
            idx, edge_feature_list, dis_cutoff, ang_cutoff
        )
        self._count_donor_acceptor_contacts(
            c2['hbond_donors_OH'], c1['hbond_acceptors_O'],
            idx, edge_feature_list, dis_cutoff, ang_cutoff
        )
        self._count_donor_acceptor_contacts(
            c1['hbond_donors_NH'], c2['hbond_acceptors_O'],
            self.interac_map['hbond_NH_O'], edge_feature_list, dis_cutoff, ang_cutoff
        )
        self._count_donor_acceptor_contacts(
            c2['hbond_donors_NH'], c1['hbond_acceptors_O'],
            self.interac_map['hbond_NH_O'], edge_feature_list, dis_cutoff, ang_cutoff
        )
        self._count_donor_acceptor_contacts(
            c1['hbond_donors_NH'], c2['hbond_acceptors_N'],
            self.interac_map['hbond_NH_N'], edge_feature_list, dis_cutoff, ang_cutoff
        )
        self._count_donor_acceptor_contacts(
            c2['hbond_donors_NH'], c1['hbond_acceptors_N'],
            self.interac_map['hbond_NH_N'], edge_feature_list, dis_cutoff, ang_cutoff
        )

    def _find_weak_hbond(self, residue_data1, residue_data2, edge_feature_list, dis_cutoff=3.6, ang_cutoff=130):
        c1 = residue_data1['cache']
        c2 = residue_data2['cache']

        self._count_donor_acceptor_contacts(
            c1['weak_hbond_donors_aro'], c2['hbond_acceptors_O'],
            self.interac_map['weak_hbond_CHaro_O'], edge_feature_list, dis_cutoff, ang_cutoff
        )
        self._count_donor_acceptor_contacts(
            c2['weak_hbond_donors_aro'], c1['hbond_acceptors_O'],
            self.interac_map['weak_hbond_CHaro_O'], edge_feature_list, dis_cutoff, ang_cutoff
        )
        self._count_donor_acceptor_contacts(
            c1['weak_hbond_donors_ali'], c2['hbond_acceptors_O'],
            self.interac_map['weak_hbond_CHali_O'], edge_feature_list, dis_cutoff, ang_cutoff
        )
        self._count_donor_acceptor_contacts(
            c2['weak_hbond_donors_ali'], c1['hbond_acceptors_O'],
            self.interac_map['weak_hbond_CHali_O'], edge_feature_list, dis_cutoff, ang_cutoff
        )

    def _find_salt_bridge(self, residue_data1, residue_data2, edge_feature_list, dis_cutoff=4.0):
        c1 = residue_data1['cache']
        c2 = residue_data2['cache']
        idx = self.interac_map['salt_bridge_N_O']

        # N+ (res1) ... O- (res2)
        self._count_distance_contacts(
            c1['positively_charged_nitrogens'], c2['negatively_charged_oxygens'],
            idx, edge_feature_list, dis_cutoff
        )
        # N+ (res2) ... O- (res1)
        self._count_distance_contacts(
            c2['positively_charged_nitrogens'], c1['negatively_charged_oxygens'],
            idx, edge_feature_list, dis_cutoff
        )

    def _find_pi_stacking(self, residue_data1, residue_data2, edge_feature_list,
                          dis_cutoff=4.5, theta_e2f=60, theta_f2f=30):
        c1 = residue_data1['cache']
        c2 = residue_data2['cache']

        rings1 = c1['aromatic_rings']['rings']
        rings2 = c2['aromatic_rings']['rings']
        if not rings1 or not rings2:
            return

        centers1 = c1['aromatic_rings']['centers']
        normals1 = c1['aromatic_rings']['normals']
        centers2 = c2['aromatic_rings']['centers']
        normals2 = c2['aromatic_rings']['normals']

        found_e2f = False
        found_f2f = False

        for i, ring1 in enumerate(rings1):
            n1 = normals1[i]
            if n1 is None:
                continue
            c1_xyz = centers1[i]

            for j, ring2 in enumerate(rings2):
                n2 = normals2[j]
                if n2 is None:
                    continue

                dist = np.linalg.norm(c1_xyz - centers2[j])
                if dist > (dis_cutoff + 1.0):
                    continue

                angle = self.calculate_vector_angle(n1, n2)
                if angle is None:
                    continue

                if angle >= theta_e2f:
                    found_e2f = True
                    break
                if dist <= dis_cutoff and angle <= theta_f2f:
                    found_f2f = True

            if found_e2f:
                break

        if found_e2f:
            edge_feature_list[self.interac_map['pi_stacking_e2f']] += 1
        elif found_f2f:
            edge_feature_list[self.interac_map['pi_stacking_f2f']] += 1

    def _find_cation_pi(self, residue_data1, residue_data2, edge_feature_list, dis_cutoff=4.0):
        c1 = residue_data1['cache']
        c2 = residue_data2['cache']
        idx = self.interac_map['cation_pi_Npos_Caro']

        # positive N in res1 with aromatic rings in res2
        if c1['positively_charged_nitrogens']['xyzs'].size > 0 and c2['aromatic_rings']['rings']:
            found = False
            for n_xyz in c1['positively_charged_nitrogens']['xyzs']:
                for r_xyz in c2['aromatic_rings']['centers']:
                    if np.linalg.norm(n_xyz - r_xyz) < dis_cutoff:
                        found = True
                        break
                if found:
                    break
            if found:
                edge_feature_list[idx] += 1

        # positive N in res2 with aromatic rings in res1
        if c2['positively_charged_nitrogens']['xyzs'].size > 0 and c1['aromatic_rings']['rings']:
            found = False
            for n_xyz in c2['positively_charged_nitrogens']['xyzs']:
                for r_xyz in c1['aromatic_rings']['centers']:
                    if np.linalg.norm(n_xyz - r_xyz) < dis_cutoff:
                        found = True
                        break
                if found:
                    break
            if found:
                edge_feature_list[idx] += 1

    def _find_disulfide_bonding(self, residue_data1, residue_data2, edge_feature_list,
                                dis1=1.83, dis2=2.23, dih1=75, dih2=105):
        if residue_data1['residue_name'] != 'CYS' or residue_data2['residue_name'] != 'CYS':
            return

        c1 = residue_data1['cache']
        c2 = residue_data2['cache']

        sulfur_ids1 = c1['sulfurs']['ids']
        sulfur_ids2 = c2['sulfurs']['ids']

        if sulfur_ids1.size == 0 or sulfur_ids2.size == 0:
            return

        cb1 = c1['cb_xyz']
        cb2 = c2['cb_xyz']
        if cb1 is None or cb2 is None:
            return

        for sid1 in sulfur_ids1:
            atom_s1 = self.protein.atoms[int(sid1)]
            xyz_s1 = atom_s1['xyz']

            for sid2 in sulfur_ids2:
                atom_s2 = self.protein.atoms[int(sid2)]
                xyz_s2 = atom_s2['xyz']

                distance = self._distance_xyz(xyz_s1, xyz_s2)
                if not (dis1 < distance < dis2):
                    continue

                dihedral = self.calculate_dihedral(cb1, xyz_s1, xyz_s2, cb2)

                if dih1 < abs(dihedral) < dih2:
                    edge_feature_list[self.interac_map['disulfide_bonding_S_S']] += 1
                    return

    def _process_interaction_pair(self, residue_data1, residue_data2):
        residue_idx1 = residue_data1['residue_idx']
        residue_idx2 = residue_data2['residue_idx']

        pair_key = (residue_idx1, residue_idx2) if residue_idx1 < residue_idx2 else (residue_idx2, residue_idx1)
        if pair_key in self.edge_feature_dic:
            return

        edge_feature_list = np.zeros(len(self.interac_PP_type), dtype=np.int32)

        self._find_hydrophobic(residue_data1, residue_data2, edge_feature_list, dis_cutoff=4.0)
        self._find_hbond(residue_data1, residue_data2, edge_feature_list, dis_cutoff=3.9, ang_cutoff=140)
        self._find_pi_stacking(residue_data1, residue_data2, edge_feature_list, dis_cutoff=4.5, theta_e2f=60, theta_f2f=30)
        self._find_weak_hbond(residue_data1, residue_data2, edge_feature_list, dis_cutoff=3.6, ang_cutoff=130)
        self._find_salt_bridge(residue_data1, residue_data2, edge_feature_list, dis_cutoff=4.0)
        self._find_cation_pi(residue_data1, residue_data2, edge_feature_list, dis_cutoff=4.0)
        self._find_disulfide_bonding(residue_data1, residue_data2, edge_feature_list, dis1=1.83, dis2=2.23, dih1=75, dih2=105)

        self.edge_index_interact_PP[0].append(pair_key[0])
        self.edge_index_interact_PP[1].append(pair_key[1])
        self.edge_feature_dic[pair_key] = edge_feature_list

    def get_interaction_feature(self, distance_threshold=6.0):
        records = self.residue_records
        n = len(records)
        if n < 2:
            return

        for i in range(n):
            res1 = records[i]
            cb1 = res1['virt_CB_xyz']
            if cb1 is None:
                continue

            for j in range(i + 1, n):
                res2 = records[j]
                cb2 = res2['virt_CB_xyz']
                if cb2 is None:
                    continue

                distance = np.linalg.norm(cb1 - cb2)
                if distance > distance_threshold:
                    continue

                self._process_interaction_pair(res1, res2)

  

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
