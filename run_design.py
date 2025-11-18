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
from scripts.utils import calc_confidence, calc_fitness, seq_idx_to_residue, get_per_residue_esm_embedding


def get_feature(ligand_mol2_file, receptor_pdb_file, redesigned_residues=None, cutoff=40., use_interaction=False):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # 配体和受体处理
    ligand = Ligand(ligand_mol2_file)
    pocket_xyz = ligand.pocket_center
    protein = Protein(receptor_pdb_file, pocket_xyz, cutoff)
    
    # 掩码残基处理
    indeces_in_reserved_residues = []
    redesigned_set = set()
    if redesigned_residues:
        redesigned_set = {res.strip() for res in redesigned_residues.split(',')}
        for redesigned_res in redesigned_set:
            chain = redesigned_res[0]
            res_seq = redesigned_res[1:]
            if chain in protein.reserved_residues_dic and res_seq in protein.reserved_residues_dic[chain]:
                idx = protein.reserved_residues_dic[chain][res_seq]['residue_index']
                indeces_in_reserved_residues.append(idx)
            else:
                print(f"[Error] Redesigned residues {redesigned_res} are not included, please increase the cutoff.")
                sys.exit(1)

    mask = torch.zeros(len(protein.reserved_residue_names), dtype=torch.bool)
    if indeces_in_reserved_residues:
        mask[indeces_in_reserved_residues] = True

    # 相互作用计算
    inter_pl = Interaction_PL(protein, ligand)
    inter_pp = Interaction_PP(protein)

    # ESM模型加载
    esm_model, alphabet = esm.pretrained.esm2_t30_150M_UR50D()
    esm_model = esm_model.to(device)
    esm_model.eval()
    batch_converter = alphabet.get_batch_converter()
    
    # 掩码序列生成与嵌入计算
    esm_embedding_dic = {}
    for chain_id, seq in protein.all_residue_1_letter.items():
        chain_seq = list(seq)  # 准备链序列列表
        # 应用掩码：将重设计残基替换为<mask>
        if redesigned_residues:
            for pos, residue in enumerate(protein.all_residue_seq[chain_id]):
                residue_id = f"{chain_id}{residue}"
                if residue_id in redesigned_set:
                    chain_seq[pos] = '<mask>'  # 执行替换
        
        masked_sequence = ''.join(chain_seq)  # 重新组合为字符串
        # 获取ESM嵌入（不再需要mask_idx参数）
        esm_embedding_dic[chain_id] = get_per_residue_esm_embedding(
            esm_model, batch_converter, masked_sequence, device
        )
    
    # 图特征生成
    feature = GraphFeature(inter_pl, inter_pp, esm_embedding_dic)
    g = feature.generate_graph()
    g['residue'].mask = mask
    
    if not use_interaction:
        print('[info] Predict without using the interaction information of the original residues...')

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
        print('[info] Predict with using the interaction information of the original residues...')

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


def design(graph, model_path="best_model.pth", output_file='output.txt', redesigned_residues=None):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #print(f"Using device: {device}")
    
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
    output_content.append("Sequence Design Output\n")
    output_content.append("="*50 + "\n\n")

    with torch.no_grad():
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
        masks = graph['residue'].mask
        
        total_residues = sum(len(seq) for seq in chain_full_seq.values()) if chain_full_seq else len(graph['residue'].x)
        output_content.append(f"Total residues: {total_residues}\n\n")
        
        # 整个序列的预测序列（所有位置的最大概率）
        all_pred_seq = residue_probs.argmax(dim=-1)
        
        if redesigned_residues:
            # 指定了重设计残基的情况 ===============================
            mask_to_use = masks
            residue_count = len(redesigned_residues.split(','))
            
            # 只提取重设计位置的标签和预测
            redesigned_labels = labels[mask_to_use]
            redesigned_logprobs = residue_logprobs[mask_to_use]
            redesigned_probs = residue_probs[mask_to_use]
            redesigned_pred_seq = all_pred_seq[mask_to_use]

            # 重设计残基的恢复率计算
            correct = (redesigned_pred_seq == redesigned_labels).sum().item()
            recovery = correct / len(redesigned_labels)
            
            # 重设计残基的置信度和适应度
            confidence_redesigned = calc_confidence(redesigned_pred_seq, redesigned_logprobs)
            fitness_redesigned = calc_fitness(redesigned_pred_seq, redesigned_probs)
            
            # 完整序列的置信度和适应度计算
            # 使用混合标签：重设计位置用预测标签，其他位置用原始标签
            hybrid_labels = labels.clone()
            hybrid_labels[masks] = all_pred_seq[masks]  # 重设计位置使用预测标签
            
            # 完整序列的置信度：使用混合标签（预测标签+原始标签）和整个序列的logprobs
            confidence_full = calc_confidence(hybrid_labels, residue_logprobs)
            
            # 完整序列的适应度：使用混合标签（预测标签+原始标签）和整个序列的概率
            fitness_full = calc_fitness(hybrid_labels, residue_probs)
            
            # 输出重设计残基指标
            output_content.append(f"Metrics computed on {residue_count} redesigned residues\n")
            output_content.append("-"*50 + "\n")
            output_content.append(f"  Fitness of redesigned residues: {fitness_redesigned:.6f}\n")
            output_content.append(f"  Confidence of redesigned residues: {confidence_redesigned:.6f}\n")
            output_content.append(f"  Recovery of redesigned residues: {recovery:.4f}\n\n")
            
            # 输出完整序列指标
            output_content.append(f"Metrics computed on all {len(labels)} residues\n")
            output_content.append("-"*50 + "\n")
            output_content.append(f"  Fitness of full output sequence: {fitness_full:.6f}\n")
            output_content.append(f"  Confidence of full output sequence: {confidence_full:.6f}\n\n")
        else:
            # 未指定重设计残基的情况 ===============================
            residue_count = masks.sum().item()
            output_content.append(f"Metrics computed on all {len(labels)} residues\n")
            output_content.append("-"*50 + "\n")

            # 整个序列的置信度和适应度
            confidence_all = calc_confidence(all_pred_seq, residue_logprobs)
            fitness_all = calc_fitness(labels, residue_probs)
            
            # 整个序列的恢复率（所有位置的预测准确性）
            correct = (all_pred_seq == labels).sum().item()
            recovery = correct / len(labels)
            
            # 输出所有指标
            output_content.append(f"  Fitness of full output sequence:{fitness_all}\n")
            output_content.append(f"  Confidence of full output sequence:{confidence_all:.4f}\n")
            output_content.append(f"  Recovery of full output sequence:{recovery:.4f}\n\n")

        # 构建链级输出序列（保持不变）
        if reserved_chain_ids and chain_full_seq:
            output_content.append("\nPer-chain Sequence Details\n")
            output_content.append("="*50 + "\n\n")
            pred_indices_all = all_pred_seq.cpu().tolist()
            pred_letters_all = seq_idx_to_residue(pred_indices_all)

            redesigned_set = set()
            if redesigned_residues:
                redesigned_set = {res.strip() for res in redesigned_residues.split(',')}

            for chain_id, orig_seq in chain_full_seq.items():
                ori_list = list(orig_seq)
                pred_list = ori_list.copy()

                for idx, (res_chain, pos) in enumerate(zip(reserved_chain_ids, reserved_pos_in_chain)):
                    if res_chain == chain_id:
                        res_str = f"{res_chain}{all_residue_seq[res_chain][pos]}"
                        # 如果指定了重设计残基且当前残基不在集合中，跳过替换
                        if redesigned_residues and (res_str not in redesigned_set):
                            continue
                        pred_list[pos] = pred_letters_all[idx]

                pred_str = ''.join(pred_list)
                output_content.append(f"Chain {chain_id}:\n")
                output_content.append(f"  Original: {orig_seq}\n")
                output_content.append(f"  Redesign: {pred_str}\n\n")
        else:
            pred_sequence_1letter = seq_idx_to_residue(all_pred_seq.cpu().tolist())
            output_content.append(f"Predicted sequence: {pred_sequence_1letter}\n")

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
    parser.add_argument("-a", "--redesigned_residues", type=str, 
                        help="Redesigned residues. Example: A36,A109,A221")
    parser.add_argument("-o", "--output", type=str, default="output.txt",
                        help="Output file path.")
    parser.add_argument("-c", "--cutoff", type=float, default=40,
                        help="Cut off for detecting reserved residues.")
    parser.add_argument("-u", "--use_interaction", action="store_true", default=False,
                        help=("The interaction information of the residues specified "
                              "in --redesigned_residues is ignored by default. "
                              "For specific purposes such as finding residues "
                              "with similar interaction patterns, you can set "
                              "--use_interaction to retrieve the interaction environment."))
    args = parser.parse_args()

    graph = get_feature(
        ligand_mol2_file=args.ligand, 
        receptor_pdb_file=args.receptor, 
        redesigned_residues=args.redesigned_residues, 
        cutoff=args.cutoff, 
        use_interaction=args.use_interaction        
    )

    design(
        graph, 
        model_path=args.model, 
        output_file=args.output, 
        redesigned_residues=args.redesigned_residues
    )