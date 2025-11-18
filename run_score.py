import random
import os
import sys
import esm
import torch
import torch.nn.functional as F
from collections import defaultdict
from scripts.model import MPNN_HeteroGNN
from scripts.parse_ligand import Ligand
from scripts.parse_protein import Protein
from scripts.generate_feature import GraphFeature
from scripts.interaction import Interaction_PL, Interaction_PP
from scripts.utils import *


def get_feature(ligand_mol2_file, receptor_pdb_file, mutations=None, cutoff=40., use_interaction=False):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    ligand = Ligand(ligand_mol2_file)
    pocket_xyz = ligand.pocket_center
    protein = Protein(receptor_pdb_file, pocket_xyz, cutoff)
    
    indeces_in_reserved_residues = []
    mutation_set = set()
    mutation_dict = {}
    if mutations:
        mutation_set = {res.strip() for res in mutations.split(',')}
        for redesigned_res in mutation_set:
            chain = redesigned_res[0]
            origin_res = redesigned_res[2]
            res_seq = redesigned_res[3:-1]
            mutate_res = redesigned_res[-1]
            if chain in protein.reserved_residues_dic and res_seq in protein.reserved_residues_dic[chain]:
                idx = protein.reserved_residues_dic[chain][res_seq]['residue_index']
                indeces_in_reserved_residues.append(idx)
                
                mutation_dict[(chain, res_seq)] = mutate_res
            else:
                print(f"[Error] Mutated residues {redesigned_res} are not included, please increase the cutoff.")
                sys.exit(1)

    mask = torch.zeros(len(protein.reserved_residue_names), dtype=torch.bool)
    if indeces_in_reserved_residues:
        mask[indeces_in_reserved_residues] = True

    inter_pl = Interaction_PL(protein, ligand)
    inter_pp = Interaction_PP(protein)
    
    esm_model, alphabet = esm.pretrained.esm2_t30_150M_UR50D()
    esm_model = esm_model.to(device)
    esm_model.eval()
    batch_converter = alphabet.get_batch_converter()
    
    esm_embedding_dic = {}
    for chain_id, seq in protein.all_residue_1_letter.items():
        chain_seq = list(seq)
        
        if mutation_dict:
            for pos, residue in enumerate(protein.all_residue_seq[chain_id]):
                residue_key = (chain_id, residue)
                if residue_key in mutation_dict:
                    target_aa = mutation_dict[residue_key]
                    chain_seq[pos] = target_aa
        
        mutated_sequence = ''.join(chain_seq)
        
        esm_embedding_dic[chain_id] = get_per_residue_esm_embedding(
            esm_model, batch_converter, mutated_sequence, device
        )
    
    feature = GraphFeature(inter_pl, inter_pp, esm_embedding_dic)
    g = feature.generate_graph()
    g['residue'].mask = mask
    
    if not use_interaction:
        print('[info] Score without using the interaction information of the original residues...')

        # 处理 ligand-interacts_with-residue 边
        edge_index = g[('ligand', 'interacts_with', 'residue')].edge_index
        edge_attr = g[('ligand', 'interacts_with', 'residue')].edge_attr
        dst_nodes = edge_index[1]
        edge_mask = g['residue'].mask[dst_nodes]
        
        g[('ligand', 'interacts_with', 'residue')].edge_y = edge_attr.clone()
        g[('ligand', 'interacts_with', 'residue')].edge_mask = edge_mask
        edge_attr[edge_mask] = 0
        g[('ligand', 'interacts_with', 'residue')].edge_attr = edge_attr

        # 处理 residue-interacts_between-residue 边
        edge_index = g[('residue', 'interacts_between', 'residue')].edge_index
        edge_attr = g[('residue', 'interacts_between', 'residue')].edge_attr
        src_nodes, dst_nodes = edge_index
        mask = g['residue'].mask
        edge_mask = mask[src_nodes] | mask[dst_nodes]
        
        g[('residue', 'interacts_between', 'residue')].edge_y = edge_attr.clone()
        g[('residue', 'interacts_between', 'residue')].edge_mask = edge_mask
        edge_attr[edge_mask] = 0
        g[('residue', 'interacts_between', 'residue')].edge_attr = edge_attr
    
    else:
        print('[info] Score with using the interaction information of the original residues...')
    
    reserved_chain_ids = []
    reserved_pos_in_chain = []
    reserved_seq_to_idx = {}
    for chain_id, full_seq_list in protein.all_residue_seq.items():
        if chain_id not in protein.reserved_residues_dic:
            continue
        reserved_seq_to_idx[chain_id] = {}
        for pos, res_seq in enumerate(full_seq_list):
            if res_seq in protein.reserved_residues_dic[chain_id]:
                reserved_chain_ids.append(chain_id)
                reserved_pos_in_chain.append(pos)
                reserved_seq_to_idx[chain_id][res_seq] = \
                    protein.reserved_residues_dic[chain_id][res_seq]['residue_index']
    g.reserved_chain_ids = reserved_chain_ids
    g.reserved_pos_in_chain = reserved_pos_in_chain
    g.chain_full_seq = protein.all_residue_1_letter
    g.all_residue_seq = protein.all_residue_seq
    g.reserved_seq_to_idx = reserved_seq_to_idx
    
    return g


def score(graph, model_path="best_model.pth", output_file='output.txt', mutations=None):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    graph = graph.to(device)
    
    # 从图数据中获取链相关信息
    reserved_chain_ids = getattr(graph, 'reserved_chain_ids', None)
    reserved_pos_in_chain = getattr(graph, 'reserved_pos_in_chain', None)
    chain_full_seq = getattr(graph, 'chain_full_seq', None)
    all_residue_seq = getattr(graph, 'all_residue_seq', None)
    reserved_seq_to_idx = getattr(graph, 'reserved_seq_to_idx', None)

    # 维度信息
    num_residue_types = graph['residue'].x.shape[1]
    num_lig_atom_types = graph['ligand'].x.shape[1]
    num_interac_PP_type = graph[('residue', 'interacts_between', 'residue')].edge_attr.shape[1]
    num_interac_PL_type = graph[('ligand', 'interacts_with', 'residue')].edge_attr.shape[1]

    # 初始化模型
    model = MPNN_HeteroGNN(
        num_residue_types=num_residue_types,
        num_lig_atom_types=num_lig_atom_types,
        num_interac_PP_type=num_interac_PP_type,
        num_interac_PL_type=num_interac_PL_type,
        num_blocks=4,
        num_heads=4,
        hidden_dim=256,
        dropout_rate=0.0
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 构建输出内容
    output_content = []
    output_content.append("Sequence Scoring Output\n")
    output_content.append("="*50 + "\n\n")
    
    with torch.no_grad():
        # 获取模型预测结果
        out = model(
            graph.x_dict, 
            graph.edge_index_dict, 
            graph.edge_attr_dict, 
            graph
        )

        residue_logits = out['residue']
        residue_probs = F.softmax(residue_logits, dim=1)
        residue_logprobs = F.log_softmax(residue_logits, dim=1)
        labels = graph['residue'].y
        
        total_residues = sum(len(seq) for seq in chain_full_seq.values()) if chain_full_seq else len(graph['residue'].x)
        output_content.append(f"Total residues: {total_residues}\n\n")

        # 初始化变量
        mutated_indices = []
        new_labels = labels.clone()
        
        # 处理突变（如果指定）
        if mutations:
            mutation_set = {res.strip() for res in mutations.split(',')}
            for redesigned_res in mutation_set:
                chain = redesigned_res[0]
                origin_res = redesigned_res[2]
                res_seq = redesigned_res[3:-1]
                mutate_res = redesigned_res[-1]

                target_idx = reserved_seq_to_idx[chain][res_seq]
                
                # 解析突变残基（如"A_W36T" -> T）
                mutated_idx = residues.index(restype_1to3[mutate_res])
                
                # 更新标签并记录突变位置
                new_labels[target_idx] = mutated_idx
                mutated_indices.append(target_idx)
        
        # 计算完整序列指标
        full_confidence = calc_confidence(new_labels, residue_logprobs)
        full_fitness = calc_fitness(new_labels, residue_probs)
        
        # 输出突变位置指标（如果有突变）
        if mutated_indices:
            # 提取突变位置的预测概率和标签
            mut_probs = residue_probs[mutated_indices]
            mut_logprobs = residue_logprobs[mutated_indices]
            mut_labels = new_labels[mutated_indices]
            
            # 计算突变位置指标
            mut_confidence = calc_confidence(mut_labels, mut_logprobs)
            mut_fitness = calc_fitness(mut_labels, mut_probs)
            
            output_content.append(f"Metrics computed on {len(mutated_indices)} mutated residues\n")
            output_content.append("-"*50 + "\n")
            output_content.append(f"  Confidence of mutated residues: {mut_confidence:.6f}\n")
            output_content.append(f"  Fitness of mutated residues:    {mut_fitness:.6f}\n\n")
        
        # 输出完整序列指标
        output_content.append(f"Metrics computed on all {len(new_labels)} residues\n")
        output_content.append("-"*50 + "\n")
        output_content.append(f"  Confidence of full sequence: {full_confidence:.6f}\n")
        output_content.append(f"  Fitness of full sequence:    {full_fitness:.6f}\n\n")
        
        # 添加链级序列信息（如果有）
        if reserved_chain_ids and chain_full_seq:
            output_content.append("\nPer-chain Sequence Details\n")
            output_content.append("="*50 + "\n\n")
            
            # 分离每条链的原始序列
            chain_original_seqs = {}
            chain_mutated_seqs = {}
            for chain_id in set(reserved_chain_ids):
                chain_original_seqs[chain_id] = []
                chain_mutated_seqs[chain_id] = []
            
            # 收集所有位置
            for i, chain_id in enumerate(reserved_chain_ids):
                aa_idx = labels[i].item()
                mutated_aa_idx = new_labels[i].item()
                
                chain_original_seqs[chain_id].append(restype_3to1[residues[aa_idx]])
                chain_mutated_seqs[chain_id].append(restype_3to1[residues[mutated_aa_idx]])
            
            # 输出每条链的信息
            for chain_id in chain_original_seqs.keys():
                original_str = ''.join(chain_original_seqs[chain_id])
                mutated_str = ''.join(chain_mutated_seqs[chain_id])
                
                output_content.append(f"Chain {chain_id}:\n")
                output_content.append(f"  Original sequence: {original_str}\n")
                output_content.append(f"  Mutated sequence:  {mutated_str}\n\n")
                
                # 标记突变位置
                if mutated_indices and original_str != mutated_str:
                    # 找出当前链中的突变位置
                    chain_mut_indices = []
                    for i, c_id in enumerate(reserved_chain_ids):
                        if c_id == chain_id and i in mutated_indices:
                            chain_mut_indices.append(i)
                    
                    if chain_mut_indices:
                        output_content.append("  Mutation positions:\n")
                        for idx in chain_mut_indices:
                            # 获取残基位置
                            pos_in_chain = reserved_pos_in_chain[idx]
                            res_num = all_residue_seq[chain_id][pos_in_chain]
                            
                            # 获取原始和突变后的氨基酸
                            orig_aa = restype_3to1[residues[labels[idx].item()]]
                            mut_aa = restype_3to1[residues[new_labels[idx].item()]]
                            
                            output_content.append(f"    {res_num}: {orig_aa} → {mut_aa}\n")
                        output_content.append("\n")

        # 将输出内容写入文件并打印到控制台
        full_output = "".join(output_content)
        #print(full_output)
        
        with open(output_file, 'w') as f:
            f.write(full_output)



if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--receptor", type=str, required=True,
                        help="Input receptor PDB file.")
    parser.add_argument("-l", "--ligand", type=str, required=True,
                        help="Input ligand MOL2 file.")
    parser.add_argument("-m", "--model", type=str, 
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts/best_model.pth"),
                        help="Model file.")
    parser.add_argument("-o", "--output", type=str, default="output.txt",
                        help="Output file path (default: output.txt).")
    parser.add_argument("-c", "--cutoff", type=float, default=40.0,
                        help="Cutoff distance for detecting reserved residues (default: 40Å).")
    parser.add_argument("-u", "--use_interaction", action="store_true", default=False,
                        help=("Score the sequence with interactional information. "
                              "Use when the scored sequence matches the sequence of PDB "
                              "Use if you want to introduce the interaction information "
                              "of the original residue while mutating."))
    parser.add_argument("-t", "--mutations", type=str,
                        help=("Specify mutations in format: Chain_mutation. "
                              "Example: A_W36T,A_Q109P,A_V221D."))

    args = parser.parse_args()

    graph = get_feature(
        ligand_mol2_file=args.ligand, 
        receptor_pdb_file=args.receptor, 
        mutations=args.mutations, 
        cutoff=args.cutoff, 
        use_interaction=args.use_interaction        
    )
    
    score(
        graph, 
        model_path=args.model, 
        output_file=args.output, 
        mutations=args.mutations
    )