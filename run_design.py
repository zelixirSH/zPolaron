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
from scripts.utils import calc_confidence, calc_fitness, seq_idx_to_residue, get_esm_embedding_with_gap_split

_esm_model = None
_esm_batch_converter = None


def _load_esm(device):
    global _esm_model, _esm_batch_converter
    if _esm_model is None:
        print("[info] Loading ESM model (once, cached for subsequent rounds)...")
        esm_model, alphabet = esm.pretrained.esm2_t30_150M_UR50D()
        esm_model = esm_model.to(device)
        esm_model.eval()
        _esm_batch_converter = alphabet.get_batch_converter()
        _esm_model = esm_model
    return _esm_model, _esm_batch_converter


def get_feature(ligand_mol2_file, receptor_pdb_file, redesigned_residues=None, cutoff=40., use_interaction=False, use_esm=False, predicted_sequences=None, locked_indices=None):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Ligand and receptor processing
    ligand = Ligand(ligand_mol2_file)
    pocket_xyz = ligand.pocket_center
    protein = Protein(receptor_pdb_file, pocket_xyz, cutoff)

    # Support "all_res": auto-expand to all pocket residues
    if redesigned_residues == "all_res":
        all_res_list = []
        for chain_id, chain_data in protein.reserved_residues_dic.items():
            for res_seq in chain_data.keys():
                all_res_list.append(f"{chain_id}{res_seq}")
        redesigned_residues = ','.join(all_res_list)
        print(f"[info] Expanded 'all_res' to {len(all_res_list)} residues")

        # Masked residue processing
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
    if locked_indices is not None:
        mask[locked_indices] = False

        # Interaction computation
    inter_pl = Interaction_PL(protein, ligand)
    inter_pp = Interaction_PP(protein)

    # Load ESM model (use global cache to avoid reloading across iterations)
    esm_model, batch_converter = _load_esm(device)

    # Build locked_set for ESM sequence to decide which positions keep predicted values
    locked_set_esm = set(locked_indices) if locked_indices is not None else set()

    # Masked sequence generation and embedding computation
    esm_embedding_dic = {}
    for chain_id, seq in protein.all_residue_1_letter.items():
        chain_seq = list(seq)  # Prepare chain sequence list

        if redesigned_residues and not use_esm:
            # Step 1: Replace all redesigned positions with <mask>
            for pos, residue in enumerate(protein.all_residue_seq[chain_id]):
                residue_id = f"{chain_id}{residue}"
                if residue_id in redesigned_set:
                    chain_seq[pos] = '<mask>'

            # Step 2: If predicted_sequences provided, overwrite locked positions with predicted amino acids
            if predicted_sequences is not None and chain_id in predicted_sequences:
                pred_seq = predicted_sequences[chain_id]
                for pos, residue in enumerate(protein.all_residue_seq[chain_id]):
                    residue_id = f"{chain_id}{residue}"
                    if residue_id in redesigned_set:
                        # Check if this position is locked via global residue index
                        try:
                            global_idx = protein.reserved_residues_dic[chain_id][residue]['residue_index']
                            if global_idx in locked_set_esm:
                                chain_seq[pos] = pred_seq[pos]
                        except KeyError:
                            pass
                print(f"[info] Using predicted sequence for chain {chain_id} (iterative refinement)")
            else:
                print(f"[info] Applied ESM mask for chain {chain_id}")
        else:
            print(f"[info] No ESM mask applied for chain {chain_id}")

        masked_sequence = ''.join(chain_seq)  # Rejoin into string
        # Get ESM embeddings (auto-segmented when residue numbering is discontinuous)
        esm_embedding_dic[chain_id] = get_esm_embedding_with_gap_split(
            esm_model, batch_converter, masked_sequence,
            protein.all_residue_seq[chain_id], device
        )

    # Graph feature generation
    feature = GraphFeature(inter_pl, inter_pp, esm_embedding_dic, mask)
    g = feature.generate_graph()
    g['residue'].mask = mask

    if not use_interaction:
        print('[info] Predict without using the interaction information of the original residues...')

        # Process ligand-interacts_with-residue edges
        edge_index = g[('ligand', 'interacts_with', 'residue')].edge_index
        edge_attr = g[('ligand', 'interacts_with', 'residue')].edge_attr
        dst_nodes = edge_index[1]
        edge_mask = g['residue'].mask[dst_nodes]

        g[('ligand', 'interacts_with', 'residue')].edge_y = edge_attr.clone()
        g[('ligand', 'interacts_with', 'residue')].edge_mask = edge_mask
        edge_attr[edge_mask] = 0
        g[('ligand', 'interacts_with', 'residue')].edge_attr = edge_attr

        # Process residue-interacts_between-residue edges
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
    g.redesigned_residues = redesigned_residues

    return g


def design(graph, model_path="best_model.pth", output_file='output.txt', redesigned_residues=None,
           iter_num=1, ligand_mol2_file=None, receptor_pdb_file=None, cutoff=40., use_interaction=False):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    graph = graph.to(device)

    # Get expanded redesigned_residues from graph (supports "all_res" auto-expand)
    redesigned_residues = getattr(graph, 'redesigned_residues', redesigned_residues)

    # Retrieve chain-related info from graph
    reserved_chain_ids = getattr(graph, 'reserved_chain_ids', None)
    reserved_pos_in_chain = getattr(graph, 'reserved_pos_in_chain', None)
    chain_full_seq = getattr(graph, 'chain_full_seq', None)
    all_residue_seq = getattr(graph, 'all_residue_seq', None)
    reserved_seq_to_idx = getattr(graph, 'reserved_seq_to_idx', None)

    # Dimension info
    num_residue_types = graph['residue'].x.shape[1]
    num_lig_atom_types = graph['ligand'].x.shape[1]
    num_interac_PP_type = graph[('residue', 'interacts_between', 'residue')].edge_attr.shape[1]
    num_interac_PL_type = graph[('ligand', 'interacts_with', 'residue')].edge_attr.shape[1]

    # Initialize model
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

    # Save original labels and initial mask (for final output, unaffected by per-round locking)
    labels = graph['residue'].y.to(device)
    initial_mask = graph['residue'].mask.clone()

    # Build index mapping: model output idx -> global residue index
    idx_to_global = {}
    if reserved_chain_ids and reserved_pos_in_chain and all_residue_seq and reserved_seq_to_idx:
        for idx, (chain, pos) in enumerate(zip(reserved_chain_ids, reserved_pos_in_chain)):
            res_name = all_residue_seq[chain][pos]
            global_idx = reserved_seq_to_idx[chain][res_name]
            idx_to_global[idx] = global_idx

            # ---- Per-round confidence-ranked iterative design ----
    total_masked = int(initial_mask.sum().item())
    residues_per_round = max(1, (total_masked + iter_num - 1) // iter_num) if iter_num > 1 else total_masked
    locked_set: set = set()
    current_graph = graph

    for it in range(iter_num):
        current_graph = current_graph.to(device)

        with torch.no_grad():
            out = model(
                current_graph.x_dict,
                current_graph.edge_index_dict,
                current_graph.edge_attr_dict,
                current_graph
            )
            residue_logits = out['residue']
            residue_probs = F.softmax(residue_logits, dim=1)
            residue_logprobs = F.log_softmax(residue_logits, dim=1)
            all_pred_seq = residue_probs.argmax(dim=-1)

        current_mask = current_graph['residue'].mask

        # Non-final round: lock top-confidence residues this round, defer the rest
        if it < iter_num - 1 and current_mask.any():
            masked_indices = torch.where(current_mask)[0]
            masked_probs = residue_probs[current_mask]
            masked_pred = all_pred_seq[current_mask]

            # Confidence = softmax probability of predicted class
            confidence = torch.gather(masked_probs, 1, masked_pred.unsqueeze(1)).squeeze(1)
            sorted_order = torch.argsort(confidence, descending=True)

            n_to_lock = min(residues_per_round, len(masked_indices))
            for rank in range(n_to_lock):
                locked_set.add(masked_indices[sorted_order[rank]].item())

                # Build next-round predicted_sequences: only locked positions filled with predicted amino acids;
                # unlocked positions keep original amino acids (get_feature applies <mask> first, then overwrites locked)
            pred_seqs = {}
            if reserved_chain_ids and chain_full_seq:
                for chain_id, orig_seq in chain_full_seq.items():
                    pred_list = list(orig_seq)
                    for idx, (res_chain, pos) in enumerate(zip(reserved_chain_ids, reserved_pos_in_chain)):
                        if res_chain == chain_id:
                            global_idx = idx_to_global[idx]
                            if global_idx in locked_set:
                                pred_list[pos] = seq_idx_to_residue([all_pred_seq[idx].item()])[0]
                    pred_seqs[chain_id] = ''.join(pred_list)

            current_graph = get_feature(
                ligand_mol2_file=ligand_mol2_file,
                receptor_pdb_file=receptor_pdb_file,
                redesigned_residues=redesigned_residues,
                cutoff=cutoff,
                use_interaction=use_interaction,
                use_esm=False,
                predicted_sequences=pred_seqs,
                locked_indices=list(locked_set),
            )

            # ---- Final output (last-round model output + initial mask covering all redesigned residues) ----
    current_graph = current_graph.to(device)
    with torch.no_grad():
        out = model(
            current_graph.x_dict,
            current_graph.edge_index_dict,
            current_graph.edge_attr_dict,
            current_graph
        )
        residue_logits = out['residue']
        residue_probs = F.softmax(residue_logits, dim=1)
        residue_logprobs = F.log_softmax(residue_logits, dim=1)
        all_pred_seq = residue_probs.argmax(dim=-1)

    output_content = []
    output_content.append("Sequence Design Output\n")
    output_content.append("=" * 50 + "\n\n")

    total_residues = sum(len(seq) for seq in chain_full_seq.values()) if chain_full_seq else len(labels)
    output_content.append(f"Total residues: {total_residues}\n\n")

    mask_to_use = initial_mask.to(device)

    if redesigned_residues:
        residue_count = len(redesigned_residues.split(','))

        red_labels = labels[mask_to_use]
        red_logprobs = residue_logprobs[mask_to_use]
        red_probs = residue_probs[mask_to_use]
        red_pred_seq = all_pred_seq[mask_to_use]

        correct = (red_pred_seq == red_labels).sum().item()
        recovery = correct / len(red_labels)

        confidence_red = calc_confidence(red_pred_seq, red_logprobs)
        fitness_red = calc_fitness(red_pred_seq, red_probs)

        hybrid_labels = labels.clone()
        hybrid_labels[mask_to_use] = all_pred_seq[mask_to_use]
        confidence_full = calc_confidence(hybrid_labels, residue_logprobs)
        fitness_full = calc_fitness(hybrid_labels, residue_probs)

        output_content.append(f"Metrics computed on {residue_count} redesigned residues\n")
        output_content.append("-" * 50 + "\n")
        output_content.append(f"  Fitness of redesigned residues: {fitness_red:.6f}\n")
        output_content.append(f"  Confidence of redesigned residues: {confidence_red:.6f}\n")
        output_content.append(f"  Recovery of redesigned residues: {recovery:.4f}\n\n")
        output_content.append(f"Metrics computed on all {len(labels)} residues\n")
        output_content.append("-" * 50 + "\n")
        output_content.append(f"  Fitness of full output sequence: {fitness_full:.6f}\n")
        output_content.append(f"  Confidence of full output sequence: {confidence_full:.6f}\n\n")
    else:
        residue_count = mask_to_use.sum().item()
        output_content.append(f"Metrics computed on all {len(labels)} residues\n")
        output_content.append("-" * 50 + "\n")
        confidence_all = calc_confidence(all_pred_seq, residue_logprobs)
        fitness_all = calc_fitness(labels, residue_probs)
        correct = (all_pred_seq == labels).sum().item()
        recovery = correct / len(labels)
        output_content.append(f"  Fitness of full output sequence:{fitness_all}\n")
        output_content.append(f"  Confidence of full output sequence:{confidence_all:.4f}\n")
        output_content.append(f"  Recovery of full output sequence:{recovery:.4f}\n\n")

        # Per-chain sequence output
    if reserved_chain_ids and chain_full_seq:
        output_content.append("\nPer-chain Sequence Details\n")
        output_content.append("=" * 50 + "\n\n")
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

    full_output = "".join(output_content)
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
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts/model_seq_design.pth"),
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
    parser.add_argument("-e", "--use_esm", action="store_true", default=False,
                        help=("By default, redesigned residues are masked in ESM embeddings. "
                              "Set this flag to use original residue identities in ESM embeddings "
                              "without masking."))
    parser.add_argument("-n", "--iter_num", type=int, default=1,
                        help="Number of iterative refinement rounds. "
                             ">1: subsequent rounds use predicted sequences as ESM input "
                             "instead of mask tokens.")
    args = parser.parse_args()

    graph = get_feature(
        ligand_mol2_file=args.ligand,
        receptor_pdb_file=args.receptor,
        redesigned_residues=args.redesigned_residues,
        cutoff=args.cutoff,
        use_interaction=args.use_interaction,
        use_esm=args.use_esm
    )

    design(
        graph,
        model_path=args.model,
        output_file=args.output,
        redesigned_residues=args.redesigned_residues,
        iter_num=args.iter_num,
        ligand_mol2_file=args.ligand,
        receptor_pdb_file=args.receptor,
        cutoff=args.cutoff,
        use_interaction=args.use_interaction
    )
