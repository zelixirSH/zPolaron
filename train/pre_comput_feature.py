#!/usr/bin/env python

import os
import sys
import pickle
import argparse
from tqdm import tqdm

# ---- OpenBabel plugin path (must be set before importing openbabel) ----
_BABEL_LIBDIR = "/media/data/wzh/env/lib/openbabel/3.1.0"
if "BABEL_LIBDIR" not in os.environ:
    os.environ["BABEL_LIBDIR"] = _BABEL_LIBDIR

    # ---- Add project root to sys.path ----
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
sys.path.insert(0, _PROJECT_ROOT)

import torch
import lmdb
import esm

from scripts.interaction import Interaction_PL, Interaction_PP
from scripts.parse_protein import Protein
from scripts.parse_ligand import Ligand
from scripts.generate_feature import GraphFeature
from scripts.utils import get_esm_embedding_with_gap_split


# -----------------------------------------------------------------------
# Data paths (consistent with process_feature.py / gene_task_feature.py)
# -----------------------------------------------------------------------
def get_complex_files(data_root, pdb_id):
    """
    Return (protein_path, ligand_path), or (None, None) if files are missing.

    Path format:
      {data_root}/{pdb_id}/{pdb_id}_protein.pdb
      {data_root}/{pdb_id}/{pdb_id}_ligand.mol2
    """
    protein_path = os.path.join(data_root, pdb_id, f"{pdb_id}_protein.pdb")
    ligand_path = os.path.join(data_root, pdb_id, f"{pdb_id}_ligand.mol2")
    if os.path.exists(protein_path) and os.path.exists(ligand_path):
        return protein_path, ligand_path
    return None, None


def parse_mask_file(mask_file):
    """Parse mask file, return [(pdb_id, [mask_res, ...]), ...]"""
    tasks = []
    with open(mask_file, "r") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.strip().split()
            pdb_id = parts[0]
            m_list = parts[1:]
            tasks.append((pdb_id, m_list))
    return tasks


    # -----------------------------------------------------------------------
    # Graph feature generation (same logic as run_get_feature in process_feature.py)
    # -----------------------------------------------------------------------
def generate_graph(protein_file, ligand_file, mask_res_list,
                   esm_model, batch_converter, device, cutoff=200.0):
    ligand = Ligand(ligand_file)
    pocket_xyz = ligand.pocket_center
    protein = Protein(protein_file, pocket_xyz, cutoff)

    redesigned_set = set(mask_res_list)
    indeces_in_reserved = []

    for res_id in redesigned_set:
        chain = res_id[0]
        res_seq = res_id[1:]
        if chain in protein.reserved_residues_dic and res_seq in protein.reserved_residues_dic[chain]:
            idx = protein.reserved_residues_dic[chain][res_seq]["residue_index"]
            indeces_in_reserved.append(idx)

    mask_tensor = torch.zeros(len(protein.reserved_residue_names), dtype=torch.bool)
    if indeces_in_reserved:
        mask_tensor[indeces_in_reserved] = True

    inter_pl = Interaction_PL(protein, ligand)
    inter_pp = Interaction_PP(protein)

    # ESM embeddings (identity masking)
    esm_embedding_dic = {}
    for chain_id, seq in protein.all_residue_1_letter.items():
        chain_seq_list = list(seq)
        for pos, res_seq in enumerate(protein.all_residue_seq[chain_id]):
            if f"{chain_id}{res_seq}" in redesigned_set:
                chain_seq_list[pos] = "<mask>"
        masked_sequence = "".join(chain_seq_list)
        esm_embedding_dic[chain_id] = get_esm_embedding_with_gap_split(
            esm_model, batch_converter, masked_sequence,
            protein.all_residue_seq[chain_id], device,
        )

    feat_gen = GraphFeature(inter_pl, inter_pp, esm_embedding_dic, mask_tensor)
    g = feat_gen.generate_graph()
    g["residue"].mask = mask_tensor

    # PL edges: mask interaction features for masked residues
    pl_edge = g["ligand", "interacts_with", "residue"]
    if pl_edge.edge_index.numel() > 0:
        e_mask_pl = g["residue"].mask[pl_edge.edge_index[1]]
        pl_edge.edge_y = pl_edge.edge_attr.clone()
        pl_edge.edge_mask = e_mask_pl
        pl_edge.edge_attr[e_mask_pl] = 0

        # PP edges: mask interaction features for masked residues
    pp_edge = g["residue", "interacts_between", "residue"]
    if pp_edge.edge_index.numel() > 0:
        src_n, dst_n = pp_edge.edge_index
        e_mask_pp = g["residue"].mask[src_n] | g["residue"].mask[dst_n]
        pp_edge.edge_y = pp_edge.edge_attr.clone()
        pp_edge.edge_mask = e_mask_pp
        pp_edge.edge_attr[e_mask_pp] = 0

    return g


    # -----------------------------------------------------------------------
    # LMDB writing (format identical to convert_to_lmdb.py)
    # -----------------------------------------------------------------------
def process_to_lmdb(mask_file, data_root, lmdb_path,
                    esm_model, batch_converter, device,
                    cutoff=200.0, test_n=None, commit_interval=1000):
    tasks = parse_mask_file(mask_file)
    if test_n:
        tasks = tasks[:test_n]

    total = len(tasks)
    print(f"  Total tasks: {total}")

    map_size = 1 << 40  # 1 TB virtual allocation, actual disk usage on demand

    env = lmdb.open(
        lmdb_path,
        map_size=map_size,
        subdir=False,
        readonly=False,
        meminit=False,
        map_async=True,
    )

    txn = env.begin(write=True)
    valid_count = 0
    skipped = 0

    for pdb_id, m_list in tqdm(tasks, desc=os.path.basename(lmdb_path)):
        protein_file, ligand_file = get_complex_files(data_root, pdb_id)
        if protein_file is None:
            skipped += 1
            continue

        try:
            graph = generate_graph(
                protein_file, ligand_file, m_list,
                esm_model, batch_converter, device, cutoff,
            )
        except Exception as e:
            print(f"\n  [Error] {pdb_id}: {e}")
            skipped += 1
            continue

        serialized = pickle.dumps(graph)
        key = f"{valid_count:08d}".encode()
        txn.put(key, serialized)
        valid_count += 1

        if valid_count % commit_interval == 0:
            txn.commit()
            txn = env.begin(write=True)

    txn.put(b"__len__", str(valid_count).encode())
    txn.commit()
    env.sync()
    env.close()

    print(f"  Done: {valid_count} samples written, {skipped} skipped")
    return valid_count


    # -----------------------------------------------------------------------
    # Main program
    # -----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Pre-compute graph features and write to LMDB directly",
    )
    parser.add_argument(
        "--data_root_train", type=str,
        default="/media/data/wzh/datasets/LigandMPNN_dataset/split_two_part_train",
        help="Root directory for training data",
    )
    parser.add_argument(
        "--data_root_valid", type=str,
        default="/media/data/wzh/datasets/LigandMPNN_dataset/split_two_part_valid",
        help="Root directory for validation data",
    )
    parser.add_argument(
        "--mask_dir", type=str, default=_SCRIPT_DIR,
        help="Directory containing mask_list_train/ and mask_list_valid/",
    )
    parser.add_argument(
        "--out_dir", type=str,
        default='./lmdb_data',
        help="Output directory for LMDB files",
    )
    parser.add_argument(
        "-cutoff", type=float, default=200.0,
        help="Cutoff distance for reserved residues (default: 200)",
    )
    parser.add_argument(
        "-device", type=str, default="cuda:0",
        help="Device for ESM model (default: cuda:0)",
    )
    parser.add_argument(
        "--test_n_train", type=int, default=None,
        help="Test mode: only process N samples per train LMDB",
    )
    parser.add_argument(
        "--test_n_valid", type=int, default=None,
        help="Test mode: only process N samples per valid LMDB",
    )
    parser.add_argument(
        "--train_only", type=int, default=None,
        help="Only process train_N.lmdb (1-10), skip the rest",
    )
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    # Load ESM model (only once)
    print(f"[{device}] Loading ESM-2 (esm2_t30_150M_UR50D)...")
    esm_model, alphabet = esm.pretrained.esm2_t30_150M_UR50D()
    esm_model = esm_model.to(device).eval()
    batch_converter = alphabet.get_batch_converter()
    print("ESM-2 loaded.")

    # ---- Training set ----
    mask_train_dir = os.path.join(args.mask_dir, "mask_list_train")
    train_indices = [args.train_only] if args.train_only else range(1, 11)

    for n in train_indices:
        mask_file = os.path.join(mask_train_dir, f"mask_35%_{n}.dat")
        if not os.path.exists(mask_file):
            print(f"[Warning] Mask file not found: {mask_file}, skipping")
            continue

        lmdb_path = os.path.join(args.out_dir, f"train_{n}.lmdb")
        print(f"\n{'='*60}")
        print(f"Processing train_{n}.lmdb")
        print(f"  Mask file : {mask_file}")
        print(f"  Data root : {args.data_root_train}")
        print(f"  LMDB path : {lmdb_path}")
        print(f"{'='*60}")

        process_to_lmdb(
            mask_file, args.data_root_train, lmdb_path,
            esm_model, batch_converter, device,
            args.cutoff, args.test_n_train,
        )

        # ---- Validation set ----
    mask_valid_dir = os.path.join(args.mask_dir, "mask_list_valid")
    mask_file = os.path.join(mask_valid_dir, "mask_35%_1.dat")
    if os.path.exists(mask_file):
        lmdb_path = os.path.join(args.out_dir, "valid.lmdb")
        print(f"\n{'='*60}")
        print(f"Processing valid.lmdb")
        print(f"  Mask file : {mask_file}")
        print(f"  Data root : {args.data_root_valid}")
        print(f"  LMDB path : {lmdb_path}")
        print(f"{'='*60}")

        process_to_lmdb(
            mask_file, args.data_root_valid, lmdb_path,
            esm_model, batch_converter, device,
            args.cutoff, args.test_n_valid,
        )
    else:
        print(f"[Warning] Valid mask file not found: {mask_file}")

    print(f"\nAll done. LMDB files in: {args.out_dir}")


if __name__ == "__main__":
    main()
  
