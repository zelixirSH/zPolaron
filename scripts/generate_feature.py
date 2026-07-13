import random
import torch
from torch_geometric.data import HeteroData
from collections import defaultdict
from scripts.interaction import Interaction_PL, Interaction_PP
from scripts.utils import residues, lig_atom_types, halogen_list, metal_list, get_esm_embedding_with_gap_split
import numpy as np



def mask_edge(mask_chain_res_seq_list, reserved_residues_dic):
    mask_res_idx_list = []
    for mask_chain_res_seq in mask_chain_res_seq_list:
        chain_id = mask_chain_res_seq.split('_')[0]
        res_seq = mask_chain_res_seq.split('_')[1]
        mask_res_idx = reserved_residues_dic[chain_id][res_seq]['residue_index']
        mask_res_idx_list.append(mask_res_idx)
    return mask_res_idx_list


class GraphFeature:
    def __init__(self, interaction_PL: Interaction_PL, interaction_PP: Interaction_PP, esm_bedding_dic: dict, mask=None):
        self.interaction_PL = interaction_PL
        self.interaction_PP = interaction_PP

        self.num_residue = len(interaction_PL.protein.reserved_residue_names)
        self.num_ligand_atoms = len(interaction_PL.ligand.atoms)

        # ESM
        sorted_res = sorted(
            [
                (chain_id, res_seq, data['residue_index'])
                for chain_id, chain_data in interaction_PP.protein.reserved_residues_dic.items()
                for res_seq, data in chain_data.items()
            ],
            key=lambda x: x[2]
        )

        esm_features = []
        for chain_id, res_seq, _ in sorted_res:
            residue_idx = interaction_PP.protein.all_residue_seq[chain_id].index(res_seq)
            esm_features.append(esm_bedding_dic[chain_id][residue_idx])
        self.protein_node_feature = torch.stack(esm_features)

        # dihedral
        dihedral_features = []
        for chain_id, res_seq, idx_in_reserved in sorted_res:
            feat = interaction_PP.protein.dihedral_features.get(idx_in_reserved, [0.0]*6)
            dihedral_features.append(torch.tensor(feat, dtype=torch.float))
        self.dihedral_node_feature = torch.stack(dihedral_features)

        # One-hot
        residue_labels = torch.tensor(
            [residues.index(r) for r in interaction_PL.protein.reserved_residue_names],
            dtype=torch.long
        )
        onehot = torch.zeros(self.num_residue, 21)
        if mask is not None:
            onehot[mask, 0] = 1.0           # masked residue
            unmasked = ~mask
            onehot[unmasked, residue_labels[unmasked] + 1] = 1.0   # unmasked
        else:
            onehot[torch.arange(self.num_residue), residue_labels + 1] = 1.0
        self.identity_onehot = onehot

        # ligand one hot
        l_atom_types_copy = interaction_PL.ligand.atom_types.copy()
        for i in range(len(l_atom_types_copy)):
            at = l_atom_types_copy[i]
            if at in halogen_list: l_atom_types_copy[i] = 'Hal'
            elif at in metal_list: l_atom_types_copy[i] = 'Metal'
            elif not at in lig_atom_types: l_atom_types_copy[i] = 'Other'

        ligand_type_indices = torch.tensor([lig_atom_types.index(at) for at in l_atom_types_copy])
        self.ligand_node_feature = torch.zeros(self.num_ligand_atoms, len(lig_atom_types))
        self.ligand_node_feature.scatter_(1, ligand_type_indices.unsqueeze(1), 1)

    def generate_graph(self):
        data = HeteroData()
        data['residue'].x = torch.cat([
            self.protein_node_feature,      # [N, 640]
            self.dihedral_node_feature,     # [N, 6]
            self.identity_onehot,           # [N, 21]
        ], dim=-1)  # → [N, 667]
        data['ligand'].x = self.ligand_node_feature
        data['residue'].y = torch.tensor(
            [residues.index(r) for r in self.interaction_PL.protein.reserved_residue_names],
            dtype=torch.long
        )

        def set_edge(edge_dict, edge_type, is_pp=False):
            src_l, dst_l, attr_l = [], [], []

            for (u, v), feat in edge_dict.items():
                feat = torch.as_tensor(feat, dtype=torch.float)

                if is_pp:
                    src_l.extend([u, v])
                    dst_l.extend([v, u])
                    attr_l.extend([feat, feat])
                else:
                    src_l.append(u)
                    dst_l.append(v)
                    attr_l.append(feat)

            if len(src_l) > 0:
                edge_index = torch.tensor([src_l, dst_l], dtype=torch.long)
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long)

            data[edge_type].edge_index = edge_index

            if len(attr_l) > 0:
                edge_attr = torch.stack(attr_l, dim=0)
            else:
                if len(edge_dict) > 0:
                    dim = len(next(iter(edge_dict.values())))
                else:
                    dim = 1
                edge_attr = torch.empty((0, dim), dtype=torch.float)

            data[edge_type].edge_attr = edge_attr

        set_edge(self.interaction_PL.edge_feature_dic, ('ligand', 'interacts_with', 'residue'))
        set_edge(self.interaction_PL.geometric_edge_feature_dic, ('ligand', 'geometrical', 'residue'))
        set_edge(self.interaction_PP.edge_feature_dic, ('residue', 'interacts_between', 'residue'), is_pp=True)
        
        # peptide
        p_src, p_dst = [], []
        for (s, d) in self.interaction_PL.protein.connected_residues['peptide_bonded']:
            p_src.extend([s, d]); p_dst.extend([d, s])
        data['residue', 'peptide_bonded', 'residue'].edge_index = torch.tensor([p_src, p_dst], dtype=torch.long)

        # PP geometrical
        geo_pp_src, geo_pp_dst, geo_pp_attr = [], [], []
        for (u, v), feat in self.interaction_PP.protein.connected_residues['geometrical_edge']:
            geo_pp_src.append(u); geo_pp_dst.append(v)
            geo_pp_attr.append(feat)
        data['residue', 'geometrical', 'residue'].edge_index = torch.tensor([geo_pp_src, geo_pp_dst], dtype=torch.long)
        geo_pp_attr_np = np.array(geo_pp_attr)
        data['residue', 'geometrical', 'residue'].edge_attr = torch.tensor(geo_pp_attr_np, dtype=torch.float)

        # ligand covalent bonds
        l_src, l_dst = [], []
        for k_id, b_ids in self.interaction_PL.ligand.bonds.items():
            s = int(k_id) - 1
            for b_id in b_ids:
                d = int(b_id) - 1
                l_src.extend([s, d]); l_dst.extend([d, s])
        data['ligand', 'ligand_bonded', 'ligand'].edge_index = torch.tensor([l_src, l_dst], dtype=torch.long)

        return data


def run_get_feature(ligand_file, protein_file, mask_res_list, esm_model, batch_converter, device, cutoff=40.0):
    ligand = Ligand(ligand_file)
    pocket_xyz = ligand.pocket_center
    protein = Protein(protein_file, pocket_xyz, cutoff)
    
    redesigned_set = set(mask_res_list)
    indeces_in_reserved = []
    
    for res_id in redesigned_set:
        chain = res_id[0]
        res_seq = res_id[1:]
        if chain in protein.reserved_residues_dic and res_seq in protein.reserved_residues_dic[chain]:
            idx = protein.reserved_residues_dic[chain][res_seq]['residue_index']
            indeces_in_reserved.append(idx)
    
    mask_tensor = torch.zeros(len(protein.reserved_residue_names), dtype=torch.bool)
    if indeces_in_reserved:
        mask_tensor[indeces_in_reserved] = True

    inter_pl = Interaction_PL(protein, ligand)
    inter_pp = Interaction_PP(protein)

    # ESM
    esm_embedding_dic = {}
    for chain_id, seq in protein.all_residue_1_letter.items():
        chain_seq_list = list(seq)
        for pos, res_seq in enumerate(protein.all_residue_seq[chain_id]):
            if f"{chain_id}{res_seq}" in redesigned_set:
                chain_seq_list[pos] = '<mask>'
        
        masked_sequence = ''.join(chain_seq_list)
        esm_embedding_dic[chain_id] = get_esm_embedding_with_gap_split(
            esm_model, batch_converter, masked_sequence,
            protein.all_residue_seq[chain_id], device
        )
    
    feat_gen = GraphFeature(inter_pl, inter_pp, esm_embedding_dic, mask_tensor)
    g = feat_gen.generate_graph()
    g['residue'].mask = mask_tensor

    # PL interaction
    pl_edge = g['ligand', 'interacts_with', 'residue']
    if pl_edge.edge_index.numel() > 0:
        e_mask_pl = g['residue'].mask[pl_edge.edge_index[1]]
        pl_edge.edge_y = pl_edge.edge_attr.clone()
        pl_edge.edge_mask = e_mask_pl
        pl_edge.edge_attr[e_mask_pl] = 0

    # PP interaction
    pp_edge = g['residue', 'interacts_between', 'residue']
    if pp_edge.edge_index.numel() > 0:
        src_n, dst_n = pp_edge.edge_index
        e_mask_pp = g['residue'].mask[src_n] | g['residue'].mask[dst_n]
        pp_edge.edge_y = pp_edge.edge_attr.clone()
        pp_edge.edge_mask = e_mask_pp
        pp_edge.edge_attr[e_mask_pp] = 0

    reserved_chain_ids, reserved_pos_in_chain = [], []
    reserved_seq_to_idx = {}
    for chain_id, full_seq_list in protein.all_residue_seq.items():
        if chain_id not in protein.reserved_residues_dic: continue
        reserved_seq_to_idx[chain_id] = {}
        for pos, res_seq in enumerate(full_seq_list):
            if res_seq in protein.reserved_residues_dic[chain_id]:
                reserved_chain_ids.append(chain_id)
                reserved_pos_in_chain.append(pos)
                reserved_seq_to_idx[chain_id][res_seq] = protein.reserved_residues_dic[chain_id][res_seq]['residue_index']
    
    g.reserved_chain_ids = reserved_chain_ids
    g.reserved_pos_in_chain = reserved_pos_in_chain
    g.chain_full_seq = protein.all_residue_1_letter
    g.all_residue_seq = protein.all_residue_seq
    g.reserved_seq_to_idx = reserved_seq_to_idx

    return g


if __name__ == "__main__":
    import esm
    import argparse
    from parse_ligand import Ligand
    from parse_protein import Protein
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-inp", type=str, default="train_pdb.dat",
                        help="Input pdb data.")
    parser.add_argument("-out", type=str, default="graphs_edge.pt",
                        help="Output. Graph data.")
    parser.add_argument("-cutoff_interaction", type=float, default=20,
                        help="Cut off for detecting reserved residues.")
    parser.add_argument("-esm_dim", type=float, default=640,
                        help="Cut off for detecting pocket.")
    args = parser.parse_args()

    #dataset = '/public/home/lzzheng/wzh/dataset/pdbbind2019/all/'
    #dataset = '/srv/nfs4/Mercury/wangzh/dataset/pdbbind2020/'
    dataset = '/sugon_store/zhengliangzhen/wzh/datasets/pdbbind2020'
    if args.esm_dim == 640:
        esm_model, alphabet = esm.pretrained.esm2_t30_150M_UR50D()
        esm_model.eval()
        batch_converter = alphabet.get_batch_converter()
    elif args.esm_dim == 480:
        esm_model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
        esm_model.eval()
        batch_converter = alphabet.get_batch_converter()
    elif args.esm_dim == 1280:
        esm_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        esm_model.eval()
        batch_converter = alphabet.get_batch_converter()
    
    pdb_list = []
    with open(args.inp, 'r') as f:
        for line in f:
            pdb_list.append(line.strip())
    
    output_graph_file = args.out
    graphs = []

    for pdb in pdb_list:
        lig_mol2_file = f'{dataset}/{pdb}/{pdb}_ligand.mol2'
        rec_pdb_file = f'{dataset}/{pdb}/{pdb}_protein.pdb'
        print(f"Producing {pdb}...")
        try:
            ligand = Ligand(lig_mol2_file)
            pocket_xyz = ligand.pocket_center
            protein = Protein(rec_pdb_file, pocket_xyz=pocket_xyz, cutoff_interaction=args.cutoff_interaction, cutoff_pocket=args.cutoff_pocket)
            
            pocket_res_num = 0
            for chain, res_seqs in protein.pocket_residues.items():
                pocket_res_num += len(res_seqs)
            mask_num = int(1.0*pocket_res_num)
            
            mask_chain_res_seq_list = random_mask(mask_num, protein.pocket_residues)

            indeces_in_masked_all_chain = defaultdict(list)
            indeces_in_reserved_residues = []
            for chain_res_seq in mask_chain_res_seq_list:
                chain, res_seq = chain_res_seq.split('_')
                idx_in_reserved_residues = protein.reserved_residues_dic[chain][res_seq]['residue_index']
                idx_in_target_chain = protein.all_residue_seq[chain].index(res_seq)
                name_in_reserved_residues = protein.reserved_residues_dic[chain][res_seq]['residue_name']
                name_in_target_chain = protein.all_residue_1_letter[chain][idx_in_target_chain]
                indeces_in_reserved_residues.append(idx_in_reserved_residues)
                indeces_in_masked_all_chain[chain].append(idx_in_target_chain)
            
            indeces_in_masked_all_chain = dict(indeces_in_masked_all_chain)
            mask = torch.zeros(len(protein.reserved_residue_names), dtype=torch.bool)
            mask[indeces_in_reserved_residues] = True
            

            inter_pl = Interaction_PL(protein, ligand)
            inter_pp = Interaction_PP(protein)

            esm_bedding_dic = {}
            for chain_id, seq in protein.all_residue_1_letter.items():
                chain_seq_list = list(seq)
                if chain_id in indeces_in_masked_all_chain:
                    for idx in indeces_in_masked_all_chain[chain_id]:
                        chain_seq_list[idx] = '<mask>'
                modified_sequence = ''.join(chain_seq_list)
                esm_bedding_dic[chain_id] = get_esm_embedding_with_gap_split(
                    esm_model, batch_converter, modified_sequence,
                    protein.all_residue_seq[chain_id], device=None,
                )
            
            feature = GraphFeature(inter_pl, inter_pp, esm_bedding_dic)
            g = feature.generate_graph()
            g['residue'].mask = mask
            
            # ligand-interacts_with-residue
            if ('ligand', 'interacts_with', 'residue') in g.edge_types:
                edge_index = g['ligand', 'interacts_with', 'residue'].edge_index
                edge_attr = g['ligand', 'interacts_with', 'residue'].edge_attr
                
                dst_nodes = edge_index[1]
                edge_mask = g['residue'].mask[dst_nodes]
                
                g['ligand', 'interacts_with', 'residue'].edge_y = edge_attr.clone()
                g['ligand', 'interacts_with', 'residue'].edge_mask = edge_mask
                
                edge_attr[edge_mask] = 0
                g['ligand', 'interacts_with', 'residue'].edge_attr = edge_attr

            if ('residue', 'interacts_between', 'residue') in g.edge_types:
                edge_index = g['residue', 'interacts_between', 'residue'].edge_index
                edge_attr = g['residue', 'interacts_between', 'residue'].edge_attr
                
                src_nodes = edge_index[0]
                dst_nodes = edge_index[1]
                mask = g['residue'].mask
                edge_mask = mask[src_nodes] | mask[dst_nodes]
                
                g['residue', 'interacts_between', 'residue'].edge_y = edge_attr.clone()
                g['residue', 'interacts_between', 'residue'].edge_mask = edge_mask
                
                edge_attr[edge_mask] = 0
                g['residue', 'interacts_between', 'residue'].edge_attr = edge_attr
            
            graphs.append(g)
            print(f"Accomplishing {pdb}...")
        except Exception as e:
            print(f"Error processing {pdb}: {e}")

    torch.save(graphs, output_graph_file)
    print(f"All graphs have been saved in {output_graph_file}")
    
    
