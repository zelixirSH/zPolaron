import random
import torch
from torch_geometric.data import HeteroData
from collections import defaultdict
from scripts.interaction import Interaction_PL, Interaction_PP
from scripts.utils import residues, lig_atom_types, halogen_list, metal_list


def get_per_residue_esm_embedding(esm_model, batch_converter, protein_sequence, 
                                  indeces_in_masked_all_chain=None):
    if indeces_in_masked_all_chain:
        masked_sequence_list = list(protein_sequence)
        for idx_in_masked_all_chain in indeces_in_masked_all_chain:
            masked_sequence_list[idx_in_masked_all_chain] = '<mask>'
        masked_sequence = ''.join(masked_sequence_list)
    else:
        masked_sequence = protein_sequence
    data = [("protein", masked_sequence)]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    with torch.no_grad():
        results = esm_model(batch_tokens, repr_layers=[esm_model.num_layers], return_contacts=False)
    token_representations = results["representations"][esm_model.num_layers][0]  # shape: (seq_len+2, embedding_dim)
    residue_embeddings = token_representations[1:-1]  # (seq_len, embedding_dim)
    return residue_embeddings


def mask_edge(mask_chain_res_seq_list, reserved_residues_dic):
    mask_res_idx_list = []
    for mask_chain_res_seq in mask_chain_res_seq_list:
        chain_id = mask_chain_res_seq.split('_')[0]
        res_seq = mask_chain_res_seq.split('_')[1]
        mask_res_idx = reserved_residues_dic[chain_id][res_seq]['residue_index']
        mask_res_idx_list.append(mask_res_idx)
    return mask_res_idx_list


class GraphFeature:
    def __init__(self, interaction_PL: Interaction_PL, interaction_PP: Interaction_PP, esm_bedding_dic: dict):
        """
        interaction_PL: Interaction_PL class.
        interaction_PP: Interaction_PP class.
        """
        self.interaction_PL = interaction_PL
        self.interaction_PP = interaction_PP


        self.num_residue = len(interaction_PL.protein.reserved_residue_names)
        self.num_ligand_atoms = len(interaction_PL.ligand.atoms)

        num_residues = len(residues)
        num_lig_atom_types = len(lig_atom_types)

        esm_features = []
        for chain_id, chain_info_dic in interaction_PP.protein.reserved_residues_dic.items():
            for resi_seq, reserved_residue_info in chain_info_dic.items():
                residue_idx = interaction_PP.protein.all_residue_seq[chain_id].index(resi_seq)
                esm_features.append(esm_bedding_dic[chain_id][residue_idx])
        self.protein_node_feature = torch.stack(esm_features)

        protein_residue_indices = torch.tensor(
            [residues.index(residue) for residue in interaction_PL.protein.reserved_residue_names]
                )
        
        for i in range(len(interaction_PL.ligand.atom_types)):
            if interaction_PL.ligand.atom_types[i] in halogen_list:
                interaction_PL.ligand.atom_types[i] = 'Hal'
            elif interaction_PL.ligand.atom_types[i] in metal_list:
                interaction_PL.ligand.atom_types[i] = 'Metal'
            elif not interaction_PL.ligand.atom_types[i] in lig_atom_types:
                interaction_PL.ligand.atom_types[i] = 'Other'

        ligand_atom_type_indices = torch.tensor(
            [lig_atom_types.index(atom_type) for atom_type in interaction_PL.ligand.atom_types]
                )
        
        self.ligand_node_feature = torch.zeros(self.num_ligand_atoms, num_lig_atom_types)  # (num_ligand_nodes, 24)
        self.ligand_node_feature.scatter_(1, ligand_atom_type_indices.unsqueeze(1), 1)

        self.edge_ind_PL_fet_dic = {}
        self.edge_index_interact_PL = [[], []] #torch.empty((2, 0), dtype=torch.long)
        self.edge_feature_interact_PL = [] #torch.empty((0, self.num_interac_PL_type), dtype=torch.float)

        self.edge_ind_PP_fet_dic = {}
        self.edge_index_interact_PP = [[], []] #torch.empty((2, 0), dtype=torch.long)
        self.edge_feature_interact_PP = [] #torch.empty((0, self.num_interac_PL_type), dtype=torch.float)

        self.edge_ind_geo_PP_fet_dic = {}
        self.edge_index_geometric_PP = [[], []]
        self.edge_feature_geometric_PP = []

        self.edge_ind_geo_PL_fet_dic = {}
        self.edge_index_geometric_PL = [[], []]
        self.edge_feature_geometric_PL = []

        self.graph = self.generate_graph()
    

    def generate_graph(self):
        data = HeteroData()
        data['residue'].x = self.protein_node_feature
        data['ligand'].x = self.ligand_node_feature
        
        data['residue'].y = torch.tensor(
            [residues.index(residue) for residue in self.interaction_PL.protein.reserved_residue_names],
             dtype=torch.long
            )
        
        # protein-ligand interaction edge
        for (lig_node_idx, rec_node_idx), edge_feature in self.interaction_PL.edge_feature_dic.items():
            self.edge_index_interact_PL[0].append(lig_node_idx)
            self.edge_index_interact_PL[1].append(rec_node_idx)
            self.edge_feature_interact_PL.append(edge_feature)
        data['ligand', 'interacts_with', 'residue'].edge_index = torch.tensor(self.edge_index_interact_PL, dtype=torch.long)
        data['ligand', 'interacts_with', 'residue'].edge_attr = torch.tensor(self.edge_feature_interact_PL, dtype=torch.float)
        #data['residue', 'interacts_with', 'ligand'].edge_index = torch.tensor([self.edge_index_interact_PL[1], self.edge_index_interact_PL[0]], dtype=torch.long)
        #data['residue', 'interacts_with', 'ligand'].edge_attr = torch.tensor(self.edge_feature_interact_PL, dtype=torch.float)
        
        # protein-ligand geometrical edge
        for (lig_node_idx, rec_node_idx), edge_feature in self.interaction_PL.geometric_edge_feature_dic.items():
            self.edge_index_geometric_PL[0].append(lig_node_idx)
            self.edge_index_geometric_PL[1].append(rec_node_idx)
            self.edge_feature_geometric_PL.append(edge_feature)
        data['ligand', 'geometrical', 'residue'].edge_index = torch.tensor(self.edge_index_geometric_PL, dtype=torch.long)
        data['ligand', 'geometrical', 'residue'].edge_attr = torch.tensor(self.edge_feature_geometric_PL, dtype=torch.float)

        # protein-protein interaction edge
        for (rec_node_idx1, rec_node_idx2), edge_feature in self.interaction_PP.edge_feature_dic.items():
            self.edge_index_interact_PP[0].append(rec_node_idx1)
            self.edge_index_interact_PP[1].append(rec_node_idx2)
            self.edge_feature_interact_PP.append(edge_feature)
        edge_index_interact_PP_expand = [[], []]
        edge_index_interact_PP_expand[0] = self.edge_index_interact_PP[0] + self.edge_index_interact_PP[1]
        edge_index_interact_PP_expand[1] = self.edge_index_interact_PP[1] + self.edge_index_interact_PP[0]
        edge_feature_interact_PP_expand = []
        edge_feature_interact_PP_expand = self.edge_feature_interact_PP + self.edge_feature_interact_PP
        data['residue', 'interacts_between', 'residue'].edge_index = torch.tensor(edge_index_interact_PP_expand, dtype=torch.long)
        data['residue', 'interacts_between', 'residue'].edge_attr = torch.tensor(edge_feature_interact_PP_expand, dtype=torch.float)


        # protein-protein peptide bond edge
        peptide_bonded_edges = self.interaction_PL.protein.connected_residues['peptide_bonded']
        peptide_bonded_src = []
        peptide_bonded_dst = []
        for (src, dst) in peptide_bonded_edges:
            peptide_bonded_src += [src, dst]
            peptide_bonded_dst += [dst, src]
        peptide_bonded_edge_index = torch.tensor([peptide_bonded_src, peptide_bonded_dst], dtype=torch.long)
        num_peptide_bonded_edges = peptide_bonded_edge_index.shape[1]
        data['residue', 'peptide_bonded', 'residue'].edge_index = peptide_bonded_edge_index
        data['residue', 'peptide_bonded', 'residue'].edge_attr = torch.ones(num_peptide_bonded_edges, 1)
        
        # protein-protein geometrical edge
        geometrical_edges = self.interaction_PP.protein.connected_residues['geometrical_edge']
        for (rec_node_idx1, rec_node_idx2), geo_feature in geometrical_edges:
            self.edge_index_geometric_PP[0].append(rec_node_idx1)
            self.edge_index_geometric_PP[1].append(rec_node_idx2)
            self.edge_feature_geometric_PP.append(geo_feature)
        data['residue', 'geometrical', 'residue'].edge_index = torch.tensor(self.edge_index_geometric_PP, dtype=torch.long)
        data['residue', 'geometrical', 'residue'].edge_attr = torch.tensor(self.edge_feature_geometric_PP, dtype=torch.float)


        # ligand-ligand bonded edge
        ligand_edge_src = []
        ligand_edge_dst = []
        src_dst_pairs = []
        for key_atom_id, bonded_atom_ids in self.interaction_PL.ligand.bonds.items():
            src = int(key_atom_id) - 1
            for bonded_atom_id in bonded_atom_ids:
                dst = int(bonded_atom_id) - 1
                if not (src, dst) in src_dst_pairs:
                    src_dst_pairs.append((src, dst))
                    ligand_edge_src += [src, dst]
                    ligand_edge_dst += [dst, src]
        ligand_bonded_edge_index = torch.tensor([ligand_edge_src, ligand_edge_dst], dtype=torch.long)
        num_ligand_bonded_edges = ligand_bonded_edge_index.shape[1]
        data['ligand', 'ligand_bonded', 'ligand'].edge_index = ligand_bonded_edge_index
        data['ligand', 'ligand_bonded', 'ligand'].edge_attr = torch.ones(num_ligand_bonded_edges, 1)

        return data


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
    
    output_graph_file = args.out  # 保存图的文件名
    graphs = []  # 用于存储所有的图

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

            #print(indeces_in_masked_all_chain)
            esm_bedding_dic = {}
            for chain_id, seq in protein.all_residue_1_letter.items():
                per_residue_esm_embedding = torch.zeros(len(seq))
                if chain_id in indeces_in_masked_all_chain:
                    mask_idx_list = indeces_in_masked_all_chain[chain_id]
                    per_residue_esm_embedding = get_per_residue_esm_embedding(esm_model, batch_converter, seq, mask_idx_list)
                else:
                    per_residue_esm_embedding = get_per_residue_esm_embedding(esm_model, batch_converter, seq)
                esm_bedding_dic[chain_id] = per_residue_esm_embedding
            
            feature = GraphFeature(inter_pl, inter_pp, esm_bedding_dic)
            g = feature.generate_graph()
            g['residue'].mask = mask
            
            # 处理 ligand-interacts_with-residue 边
            if ('ligand', 'interacts_with', 'residue') in g.edge_types:
                edge_index = g['ligand', 'interacts_with', 'residue'].edge_index
                edge_attr = g['ligand', 'interacts_with', 'residue'].edge_attr
                
                # 获取需要mask的边索引
                dst_nodes = edge_index[1]  # 目标节点是residue
                edge_mask = g['residue'].mask[dst_nodes]  # 形状: (num_edges,)
                
                # 保留原始多维特征作为标签
                g['ligand', 'interacts_with', 'residue'].edge_y = edge_attr.clone()
                g['ligand', 'interacts_with', 'residue'].edge_mask = edge_mask
                
                # 将mask边的多维特征向量整体置零
                edge_attr[edge_mask] = 0  # 自动广播到所有特征维度
                g['ligand', 'interacts_with', 'residue'].edge_attr = edge_attr

            # 处理 residue-interacts_between-residue 边
            if ('residue', 'interacts_between', 'residue') in g.edge_types:
                edge_index = g['residue', 'interacts_between', 'residue'].edge_index
                edge_attr = g['residue', 'interacts_between', 'residue'].edge_attr
                
                # 获取需要mask的边索引
                src_nodes = edge_index[0]
                dst_nodes = edge_index[1]
                mask = g['residue'].mask
                edge_mask = mask[src_nodes] | mask[dst_nodes]  # 形状: (num_edges,)
                
                # 保留原始多维特征作为标签
                g['residue', 'interacts_between', 'residue'].edge_y = edge_attr.clone()
                g['residue', 'interacts_between', 'residue'].edge_mask = edge_mask
                
                # 将mask边的多维特征向量整体置零
                edge_attr[edge_mask] = 0  # 自动广播到所有特征维度
                g['residue', 'interacts_between', 'residue'].edge_attr = edge_attr
            
            graphs.append(g)
            print(f"Accomplishing {pdb}...")
        except Exception as e:
            print(f"Error processing {pdb}: {e}")

    # 保存所有图到一个文件
    torch.save(graphs, output_graph_file)
    print(f"所有图已保存到 {output_graph_file}")
    
    
