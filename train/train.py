import os
import io
import sys
import glob
import lmdb
import pickle
import random
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch_geometric.loader import DataLoader
from torch.utils.data import Dataset, DistributedSampler
from torch_geometric.data import HeteroData
from tqdm import tqdm
import argparse
from contextlib import nullcontext

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.model import MPNN_HeteroGNN


class LMDBHeteroDataset(Dataset):
    
    def __init__(self, lmdb_path: str):
        self.lmdb_path = lmdb_path
        self.env = None
        self._length = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["env"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.env = None

    def _open_env(self):
        if self.env is not None:
            return

        if not os.path.exists(self.lmdb_path):
            raise FileNotFoundError(f"LMDB file not found: {self.lmdb_path}")

        self.env = lmdb.open(
            self.lmdb_path,
            readonly=True,
            lock=False,
            readahead=True,
            meminit=False,
            subdir=False,
            max_readers=512,
        )

        with self.env.begin(write=False) as txn:
            length_buf = txn.get(b"__len__")
            if length_buf is None:
                raise KeyError(f"LMDB file missing __len__ key: {self.lmdb_path}")
            self._length = int(length_buf.decode("utf-8"))

    def __len__(self):
        if self._length is None:
            self._open_env()
        return self._length

    @staticmethod
    def _deserialize(buffer):
        
        bio = io.BytesIO(buffer)

        try:
            return torch.load(bio, map_location="cpu", weights_only=False)
        except TypeError:
            bio.seek(0)
            try:
                return torch.load(bio, map_location="cpu")
            except Exception:
                bio.seek(0)
                return pickle.loads(bio.read())
        except Exception:
            bio.seek(0)
            try:
                return pickle.loads(bio.read())
            except Exception as e:
                raise RuntimeError(f"Failed to deserialize LMDB value: {e}")

    @staticmethod
    def _sanitize_heterodata(raw_data):
        clean_data = HeteroData()

        for node_type in raw_data.node_types:
            src = raw_data[node_type]
            dst = clean_data[node_type]

            if hasattr(src, "x"):
                dst.x = src.x
            if hasattr(src, "y"):
                dst.y = src.y
            if hasattr(src, "mask"):
                dst.mask = src.mask

        for edge_type in raw_data.edge_types:
            src = raw_data[edge_type]
            dst = clean_data[edge_type]

            if hasattr(src, "edge_index"):
                dst.edge_index = src.edge_index
            if hasattr(src, "edge_attr"):
                dst.edge_attr = src.edge_attr
            if hasattr(src, "edge_y"):
                dst.edge_y = src.edge_y
            if hasattr(src, "edge_mask"):
                dst.edge_mask = src.edge_mask

        return clean_data

    def __getitem__(self, idx):
        self._open_env()
        key = f"{idx:08d}".encode("utf-8")

        with self.env.begin(write=False) as txn:
            buffer = txn.get(key)

        if buffer is None:
            raise IndexError(f"Missing LMDB key: {key.decode('utf-8')} in {self.lmdb_path}")

        raw_data = self._deserialize(buffer)
        data = self._sanitize_heterodata(raw_data)
        return data


class RandomMultiLMDBHeteroDataset(Dataset):

    def __init__(self, lmdb_paths: Sequence[str], strict_same_length: bool = True):
        if not lmdb_paths:
            raise ValueError("lmdb_paths is empty")

        self.lmdb_paths = list(lmdb_paths)
        self.strict_same_length = strict_same_length
        self.datasets = [LMDBHeteroDataset(p) for p in self.lmdb_paths]
        self._length = None

        lengths = [len(ds) for ds in self.datasets]
        if strict_same_length and len(set(lengths)) != 1:
            raise ValueError(f"All LMDBs must have the same length, got: {dict(zip(self.lmdb_paths, lengths))}")

        self._length = min(lengths)
        self._lengths = lengths

    def __getstate__(self):
        state = self.__dict__.copy()
        for ds in state["datasets"]:
            ds.env = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        for ds in self.datasets:
            ds.env = None

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        ds = random.choice(self.datasets)
        return ds[idx]


class TypedZeroInflatedPoissonLoss(nn.Module):
    def __init__(self, edge_type_info, device='cuda'):
        super().__init__()
        self.device = device
        self.alpha_params = nn.ParameterDict()
        for edge_type, zero_ratio in edge_type_info.items():
            str_key = str(edge_type)
            init_value = torch.logit(torch.tensor(zero_ratio, dtype=torch.float32).clamp(0.01, 0.99))
            param = nn.Parameter(init_value, requires_grad=True)
            self.alpha_params[str_key] = param

        self.poisson_loss_fn = nn.PoissonNLLLoss(log_input=False, full=True, reduction='none', eps=1e-8)
        self.zero_cls_loss = nn.BCEWithLogitsLoss(reduction='none')

    def get_alpha(self, edge_type):
        str_key = str(edge_type)
        return torch.sigmoid(self.alpha_params[str_key]).clamp(min=1e-4, max=1-1e-4)

    def forward(self, edge_type, pred_count, pred_zero_logits, true_count):
        true_count = true_count.to(self.device).float()
        alpha = self.get_alpha(edge_type)

        safe_pred_count = torch.clamp(pred_count, min=1e-5, max=50.0)

        zero_mask = (true_count == 0).float()
        cls_loss = self.zero_cls_loss(pred_zero_logits, zero_mask)

        raw_poisson = self.poisson_loss_fn(safe_pred_count, true_count)
        count_mask = (true_count > 0).float()
        poisson_loss = raw_poisson * count_mask

        cls_loss = torch.where(torch.isfinite(cls_loss), cls_loss, torch.zeros_like(cls_loss))
        poisson_loss = torch.where(torch.isfinite(poisson_loss), poisson_loss, torch.zeros_like(poisson_loss))

        term1 = alpha * cls_loss
        term2 = (1.0 - alpha) * poisson_loss
        total_loss = (term1 + term2).mean()

        if not torch.isfinite(total_loss):
            return (pred_zero_logits.sum() + pred_count.sum()) * 0.0

        return total_loss


class HeteroNodeClassificationLoss(nn.Module):
    def __init__(
        self,
        num_classes=20,
        class_counts=None,
        class_weights=None,
        focal_gamma=2.0,
        base_label_smoothing=0.05,
        ldas_power=1.0,
        edge_loss_weight=1.0,
        edge_warmup_epochs=10,
        confidence_penalty_weight=0.01,
        edge_sparsity_weight=1e-5,
        edge_type_info=None,
        device='cuda'
    ):
        super().__init__()
        self.num_classes = num_classes
        self.focal_gamma = focal_gamma
        self.base_label_smoothing = base_label_smoothing
        self.ldas_power = ldas_power
        self.edge_loss_weight = edge_loss_weight
        self.edge_warmup_epochs = edge_warmup_epochs
        self.confidence_penalty_weight = confidence_penalty_weight
        self.edge_sparsity_weight = edge_sparsity_weight
        self.device = torch.device(device)

        if class_counts is None:
            class_counts = torch.ones(num_classes, dtype=torch.float32)
        else:
            class_counts = torch.as_tensor(class_counts, dtype=torch.float32)

        class_counts = class_counts.clamp_min(1.0)
        class_freq = class_counts / class_counts.sum()
        freq_norm = class_freq / class_freq.max().clamp_min(1e-8)

        # Label-distribution-aware smoothing:
        # head class -> larger smoothing, tail class -> smaller smoothing
        class_smoothing = self.base_label_smoothing * freq_norm.pow(self.ldas_power)
        class_smoothing = class_smoothing.clamp(min=0.0, max=self.base_label_smoothing)

        self.register_buffer("class_counts_buf", class_counts.to(self.device))
        self.register_buffer("class_freq_buf", class_freq.to(self.device))
        self.register_buffer("class_smoothing_buf", class_smoothing.to(self.device))

        # class weights are optional
        if class_weights is None:
            class_weights = torch.ones(num_classes, dtype=torch.float32)
        else:
            class_weights = torch.as_tensor(class_weights, dtype=torch.float32)
        self.register_buffer("class_weights_buf", class_weights.to(self.device))

        if edge_type_info is None:
            edge_type_info = {}
        self.edge_loss_fn = TypedZeroInflatedPoissonLoss(
            edge_type_info=edge_type_info,
            device=device
        )

        self.edge_type_mapping = {
            ('ligand', 'interacts_with', 'residue'): 'ligand_interacts_with_residue',
            ('residue', 'interacts_between', 'residue'): 'residue_interacts_between_residue'
        }

    def _get_dynamic_edge_weight(self, epoch=None):
        if epoch is None or self.edge_warmup_epochs <= 0:
            return float(self.edge_loss_weight)

        warmup_ratio = min(1.0, float(epoch + 1) / float(self.edge_warmup_epochs))
        return float(self.edge_loss_weight * warmup_ratio)

    def _calc_node_loss(self, pred_dict, data):
        node_type = 'residue'
        logits = pred_dict[node_type]
        labels = data[node_type].y.to(self.device)
        mask = data[node_type].mask

        if mask.sum() == 0:
            zero = logits.sum() * 0.0
            return zero, {
                'node_cls_loss': 0.0,
                'node_conf_penalty': 0.0,
                'node_smoothing_mean': 0.0,
                'node_loss': 0.0
            }

        logits = logits[mask]
        labels = labels[mask]

        log_probs = F.log_softmax(logits, dim=-1)
        probs = torch.softmax(logits, dim=-1)

        # label-distribution-aware smoothing
        eps = self.class_smoothing_buf[labels]  # [N]
        one_hot = F.one_hot(labels, num_classes=self.num_classes).float()
        soft_targets = one_hot * (1.0 - eps.unsqueeze(1)) + eps.unsqueeze(1) / self.num_classes

        # smoothed CE
        ce_per_sample = -(soft_targets * log_probs).sum(dim=-1)  # [N]

        # focal factor
        pt = probs.gather(1, labels.unsqueeze(1)).squeeze(1).clamp_min(1e-8)
        focal_factor = (1.0 - pt).pow(self.focal_gamma)

        # optional class weights
        sample_weight = self.class_weights_buf[labels]
        weighted_cls_loss = (focal_factor * ce_per_sample * sample_weight).sum() / sample_weight.sum().clamp_min(1.0)

        # confidence penalty: encourage higher entropy
        entropy = -(probs * log_probs).sum(dim=-1).mean()
        conf_penalty = -self.confidence_penalty_weight * entropy

        node_total = weighted_cls_loss + conf_penalty

        details = {
            'node_cls_loss': weighted_cls_loss.item(),
            'node_conf_penalty': conf_penalty.item(),
            'node_smoothing_mean': eps.mean().item(),
            'node_loss': node_total.item()
        }
        return node_total, details

    def _calc_edge_loss(self, pred_dict, data, epoch=None):
        dynamic_edge_weight = self._get_dynamic_edge_weight(epoch)

        edge_loss_sum = 0.0
        edge_count_sum = 0
        edge_sparsity_sum = 0.0
        valid_edge_types = 0
        edge_loss_details = {
            'edge_dynamic_weight': dynamic_edge_weight,
            'edge_sparsity_loss': 0.0,
            'edge_base_loss': 0.0,
            'edge_loss': 0.0
        }

        for orig_edge_type, str_edge_type in self.edge_type_mapping.items():
            if orig_edge_type not in data.edge_types or orig_edge_type not in pred_dict['edges']:
                continue

            pred_data = pred_dict['edges'][orig_edge_type]
            mask = data[orig_edge_type].edge_mask
            if mask is None or mask.sum() == 0:
                continue

            masked_pred_count = pred_data['count'][mask]
            masked_pred_zero = pred_data['zero_logits'][mask]
            masked_true = data[orig_edge_type].edge_y[mask].to(self.device)

            # raw ZIP loss for this edge type
            raw_edge_loss = self.edge_loss_fn(
                orig_edge_type,
                masked_pred_count,
                masked_pred_zero,
                masked_true
            )

            n_masked = int(mask.sum().item())
            edge_loss_sum += raw_edge_loss * n_masked
            edge_count_sum += n_masked
            valid_edge_types += 1

            # edge sparsity regularization on all predicted counts
            # keep coefficient tiny, otherwise it will suppress legitimate contacts too much
            edge_sparsity_sum += pred_data['count'].mean()

            edge_loss_details[f'edge_{str_edge_type}_loss'] = raw_edge_loss.item()
            edge_loss_details[f'edge_{str_edge_type}_alpha'] = self.edge_loss_fn.get_alpha(orig_edge_type).item()

        if edge_count_sum > 0:
            base_edge_loss = edge_loss_sum / edge_count_sum
            edge_sparsity_loss = self.edge_sparsity_weight * (edge_sparsity_sum / max(valid_edge_types, 1))
            edge_total = dynamic_edge_weight * base_edge_loss + edge_sparsity_loss
        else:
            zero = pred_dict['residue'].sum() * 0.0
            base_edge_loss = zero
            edge_sparsity_loss = zero
            edge_total = zero

        edge_loss_details['edge_base_loss'] = float(base_edge_loss.item()) if torch.is_tensor(base_edge_loss) else float(base_edge_loss)
        edge_loss_details['edge_sparsity_loss'] = float(edge_sparsity_loss.item()) if torch.is_tensor(edge_sparsity_loss) else float(edge_sparsity_loss)
        edge_loss_details['edge_loss'] = float(edge_total.item()) if torch.is_tensor(edge_total) else float(edge_total)
        edge_loss_details['edge_dynamic_weight'] = float(dynamic_edge_weight)

        return edge_total, edge_loss_details

    def forward(self, pred_dict, data, epoch=None):
        node_loss, node_loss_details = self._calc_node_loss(pred_dict, data)
        edge_loss, edge_loss_details = self._calc_edge_loss(pred_dict, data, epoch=epoch)

        total_loss = node_loss + edge_loss
        loss_details = {
            'total_loss': total_loss.item(),
            'node_loss': node_loss.item(),
            'edge_loss': edge_loss.item(),
            **node_loss_details,
            **edge_loss_details
        }
        return total_loss, loss_details


class EarlyStopping:
    def __init__(self, patience=10, delta=0.001):
        self.patience = patience
        self.delta = delta
        self.best_loss = float('inf')
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


def seed_everything(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def get_available_device(preferred_device=None):
    if preferred_device is not None:
        try:
            if preferred_device.startswith('cuda'):
                device_id = int(preferred_device.split(':')[1]) if ':' in preferred_device else 0
                if device_id < torch.cuda.device_count():
                    device = torch.device(preferred_device)
                    torch.zeros(1).to(device)
                    return device
            else:
                return torch.device(preferred_device)
        except Exception:
            print(f"Warning: Preferred device {preferred_device} not available, trying auto-detection...")

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            try:
                torch.cuda.set_device(i)
                torch.cuda.empty_cache()
                test_tensor = torch.zeros(1).cuda()
                del test_tensor
                torch.cuda.synchronize()

                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3

                device = torch.device(f'cuda:{i}')
                print(f"  Using GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
                return device
            except Exception as e:
                print(f"Warning: GPU {i} not available: {e}")
                continue

        print("Warning: CUDA is available but no working GPU found, using CPU")
        return torch.device('cpu')
    else:
        print("CUDA not available, using CPU")
        return torch.device('cpu')


def build_loader(dataset, batch_size, shuffle, sampler, num_workers, pin_memory, drop_last, prefetch_factor):
    kwargs = dict(
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        worker_init_fn=seed_worker if num_workers > 0 else None,
    )
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(dataset, **kwargs)


def collect_train_lmdbs(args) -> List[str]:
    if args.train_lmdbs:
        lmdbs = [p for p in args.train_lmdbs if p]
    else:
        pattern = os.path.join(args.train_lmdb_dir, "train_*.lmdb")
        lmdbs = sorted(glob.glob(pattern))

    if not lmdbs:
        raise FileNotFoundError(
            f"No training LMDB files found. Provide --train_lmdbs or place train_*.lmdb under {args.train_lmdb_dir}"
        )

    return lmdbs


def calculate_zero_ratios_streaming(dataset, max_samples=1000):
    print("Estimating zero ratios from subset...")
    edge_counts = {
        ('residue', 'interacts_between', 'residue'): {'total': 0, 'zeros': 0},
        ('ligand', 'interacts_with', 'residue'): {'total': 0, 'zeros': 0}
    }
    indices = random.sample(range(len(dataset)), min(len(dataset), max_samples))
    for idx in indices:
        data = dataset[idx]
        for et in edge_counts:
            if et in data.edge_types:
                edge_y = data[et].edge_y
                edge_counts[et]['total'] += edge_y.numel()
                edge_counts[et]['zeros'] += (edge_y == 0).sum().item()
    return {et: (c['zeros']/c['total'] if c['total'] > 0 else 0.5) for et, c in edge_counts.items()}


def build_scheduler(optimizer, total_steps):
    if total_steps <= 1:
        return None

    warmup_steps = min(5000, max(1, int(total_steps * 0.1)))
    warmup_steps = min(warmup_steps, total_steps - 1)
    warmup_steps = 2000
    #warmup_steps = 1000
    if warmup_steps <= 0:
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, total_steps),
            eta_min=1e-7,
        )

    return torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=0.01,
                total_iters=warmup_steps,
            ),
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(1, total_steps - warmup_steps),
                eta_min=1e-7,
            ),
        ],
        milestones=[warmup_steps]
    )


def load_model_weights(model, basemodel_path, device, rank=0):
    """
    Load model weights from a checkpoint file.
    For DDP, only rank 0 prints the status message.
    """
    if not os.path.isfile(basemodel_path):
        raise FileNotFoundError(f"Base model checkpoint not found: {basemodel_path}")

    if rank == 0:
        print(f"Loading base model weights from: {basemodel_path}")

    state_dict = torch.load(basemodel_path, map_location=device)

    # Handle DDP-wrapped checkpoint (module. prefix)
    if all(k.startswith('module.') for k in state_dict.keys()):
        if rank == 0:
            print("Detected DDP-wrapped checkpoint, stripping 'module.' prefix")
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

    if rank == 0:
        if missing_keys:
            print(f"  Missing keys in checkpoint: {missing_keys}")
        if unexpected_keys:
            print(f"  Unexpected keys in checkpoint: {unexpected_keys}")
        if not missing_keys and not unexpected_keys:
            print("  All model weights loaded successfully (strict match).")
        else:
            print("  Model weights loaded with non-strict match.")

    return model


def save_checkpoint(epoch, model, optimizer, scheduler, loss_fn, early_stopping, best_val_loss, save_path, is_ddp=False):
    ckpt = {
        'epoch': epoch,
        'model_state_dict': model.module.state_dict() if is_ddp else model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'loss_fn_state_dict': loss_fn.state_dict(),
        'best_val_loss': best_val_loss,
        'early_stopping_best_loss': early_stopping.best_loss,
        'early_stopping_counter': early_stopping.counter,
    }
    torch.save(ckpt, save_path)


def train_single_gpu(args, early_stopping):
    torch.autograd.set_detect_anomaly(False)
    seed_everything(args.seed)

    # ---- Resume handling ----
    start_epoch = 0
    resume_checkpoint = None
    if args.resume:
        if not os.path.isfile(args.resume):
            raise FileNotFoundError(f"Resume checkpoint not found: {args.resume}")
        resume_checkpoint = torch.load(args.resume, map_location=torch.device('cpu'))
        start_epoch = resume_checkpoint['epoch'] + 1
        early_stopping.best_loss = resume_checkpoint['early_stopping_best_loss']
        early_stopping.counter = resume_checkpoint['early_stopping_counter']
        print(f"Resuming from epoch {start_epoch}, checkpoint: {args.resume}")

    device = get_available_device(args.device)
    print(f"Using device: {device}")

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

        gpu_id = device.index if device.index is not None else 0
        print(f"GPU Memory: {torch.cuda.get_device_properties(gpu_id).total_memory / 1024**3:.1f} GB")
        torch.cuda.empty_cache()

    train_lmdbs = collect_train_lmdbs(args)
    print(f"Training LMDBs ({len(train_lmdbs)}):")
    for p in train_lmdbs:
        print(f"  - {p}")

    train_dataset = RandomMultiLMDBHeteroDataset(train_lmdbs, strict_same_length=True)
    val_dataset = LMDBHeteroDataset(args.valid_lmdb)

    num_workers = args.num_workers if device.type == 'cuda' else 0
    pin_memory = True if device.type == 'cuda' else False

    train_loader = build_loader(
        dataset=train_dataset,
        batch_size=args.batchsz,
        shuffle=True,
        sampler=None,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        prefetch_factor=args.prefetch_factor,
    )
    val_loader = build_loader(
        dataset=val_dataset,
        batch_size=args.batchsz,
        shuffle=False,
        sampler=None,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        prefetch_factor=args.prefetch_factor,
    )

    # edge zero ratio statistics
    edge_type_info = calculate_zero_ratios_streaming(
        train_dataset, max_samples=args.zero_ratio_samples
    )

    # class counts statistics (for LDAS / focal loss)
    class_counts = calculate_residue_class_counts_streaming(
        train_dataset,
        max_samples=args.class_count_samples,
        num_classes=20,
    )

    sample_data = train_dataset[0]
    model = MPNN_HeteroGNN(
        num_residue_types=sample_data["residue"].x.shape[1],
        num_lig_atom_types=sample_data["ligand"].x.shape[1],
        num_interac_PP_type=sample_data[('residue', 'interacts_between', 'residue')].edge_attr.shape[1],
        num_interac_PL_type=sample_data[('ligand', 'interacts_with', 'residue')].edge_attr.shape[1],
        num_blocks=4,
        hidden_dim=256,
        num_heads=4,
        dropout_rate=args.dropout,
    ).to(device)

    if resume_checkpoint is not None:
        model.load_state_dict(resume_checkpoint['model_state_dict'])
        print(f"Loaded model weights from resume checkpoint (epoch {resume_checkpoint['epoch']})")
    elif args.basemodel is not None:
        model = load_model_weights(model, args.basemodel, device, rank=0)

    # loss function (parameters match DDP version)
    loss_fn = HeteroNodeClassificationLoss(
        num_classes=20,
        class_counts=class_counts,
        focal_gamma=args.focal_gamma,
        base_label_smoothing=args.base_label_smoothing,
        ldas_power=args.ldas_power,
        edge_loss_weight=args.edge_loss_weight,
        edge_warmup_epochs=args.edge_warmup_epochs,
        confidence_penalty_weight=args.confidence_penalty_weight,
        edge_sparsity_weight=args.edge_sparsity_weight,
        edge_type_info=edge_type_info,
        device=device,
    ).to(device)

    if resume_checkpoint is not None:
        loss_fn.load_state_dict(resume_checkpoint['loss_fn_state_dict'])
        print("Loaded loss function state from resume checkpoint")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.98),
        eps=1e-9,
    )

    total_steps = args.epochs * ((len(train_loader) + args.accum_steps - 1) // args.accum_steps)
    scheduler = build_scheduler(optimizer, total_steps)

    if resume_checkpoint is not None and resume_checkpoint['scheduler_state_dict'] is not None:
        scheduler.load_state_dict(resume_checkpoint['scheduler_state_dict'])
        print("Loaded scheduler state from resume checkpoint")

    if resume_checkpoint is not None:
        optimizer.load_state_dict(resume_checkpoint['optimizer_state_dict'])
        print("Loaded optimizer state from resume checkpoint")

    if resume_checkpoint is not None:
        best_val_loss = resume_checkpoint['best_val_loss']
        log_mode = 'a'
        print(f"Resumed best_val_loss = {best_val_loss:.6f}, appending to existing log.")
    else:
        best_val_loss = float('inf')
        log_mode = 'w'

    with open(args.log, log_mode) as log_file:
        if log_mode == 'w':
            headers = [
                'Epoch',
                'Tra_Loss', 'Tra_Node', 'Tra_Edge',
                'Tra_NodeCls', 'Tra_NodeConf', 'Tra_NodeSmooth',
                'Tra_EdgeBase', 'Tra_EdgeSparse', 'Tra_EdgeDynW',
                'Tra_PP_Loss', 'Tra_PP_Alpha',
                'Tra_PL_Loss', 'Tra_PL_Alpha',
                'Val_Loss', 'Val_Node', 'Val_Edge',
                'Val_NodeCls', 'Val_NodeConf', 'Val_NodeSmooth',
                'Val_EdgeBase', 'Val_EdgeSparse', 'Val_EdgeDynW',
                'L_Rate'
            ]
            log_file.write('  '.join(headers) + '\n')
            log_file.flush()

    for epoch in range(start_epoch, args.epochs):
        model.train()
        loss_fn.train()
        optimizer.zero_grad(set_to_none=True)

        total_loss_sum = 0.0
        tra_node_loss, tra_edge_loss = 0.0, 0.0
        tra_node_cls, tra_node_conf, tra_node_smooth = 0.0, 0.0, 0.0
        tra_edge_base, tra_edge_sparse, tra_edge_dynw = 0.0, 0.0, 0.0
        tra_pp_loss, tra_pl_loss = 0.0, 0.0
        tra_pp_alpha, tra_pl_alpha = 0.0, 0.0
        train_batches = 0

        train_pbar = tqdm(
            train_loader,
            desc=f'Epoch {epoch+1}/{args.epochs} [Train]',
            bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'
        )

        accum_steps = args.accum_steps

        for batch_idx, batch in enumerate(train_pbar):
            batch = batch.to(device)

            is_update_step = (
                (batch_idx + 1) % accum_steps == 0
                or (batch_idx + 1) == len(train_loader)
            )

            if not is_update_step:
                out = model(batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict, batch)
                loss, loss_details = loss_fn(out, batch, epoch=epoch)

                if not torch.isfinite(loss):
                    print(f"Warning: infinite loss for Batch {batch_idx}")
                    optimizer.zero_grad(set_to_none=True)
                    continue

                (loss / accum_steps).backward()
            else:
                out = model(batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict, batch)
                loss, loss_details = loss_fn(out, batch, epoch=epoch)

                if not torch.isfinite(loss):
                    print(f"Warning: infinite loss in Batch {batch_idx}")
                    optimizer.zero_grad(set_to_none=True)
                    continue

                (loss / accum_steps).backward()

                grad_norm_model = nn.utils.clip_grad_norm_(model.parameters(), 0.8)
                if torch.isnan(grad_norm_model) or torch.isinf(grad_norm_model):
                    print(f"Warning: invalid gradient in Batch {batch_idx}")
                    optimizer.zero_grad(set_to_none=True)
                    continue

                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            total_loss_sum += loss.item()
            tra_node_loss += loss_details.get('node_loss', 0.0)
            tra_edge_loss += loss_details.get('edge_loss', 0.0)
            tra_node_cls += loss_details.get('node_cls_loss', 0.0)
            tra_node_conf += loss_details.get('node_conf_penalty', 0.0)
            tra_node_smooth += loss_details.get('node_smoothing_mean', 0.0)
            tra_edge_base += loss_details.get('edge_base_loss', 0.0)
            tra_edge_sparse += loss_details.get('edge_sparsity_loss', 0.0)
            tra_edge_dynw += loss_details.get('edge_dynamic_weight', 0.0)
            tra_pp_loss += loss_details.get('edge_residue_interacts_between_residue_loss', 0.0)
            tra_pl_loss += loss_details.get('edge_ligand_interacts_with_residue_loss', 0.0)
            tra_pp_alpha += loss_details.get('edge_residue_interacts_between_residue_alpha', 0.0)
            tra_pl_alpha += loss_details.get('edge_ligand_interacts_with_residue_alpha', 0.0)
            train_batches += 1

            train_pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{optimizer.param_groups[0]['lr']:.2e}"
            })

        train_batches = max(train_batches, 1)

        avg_total_loss = total_loss_sum / train_batches
        avg_tra_node_loss = tra_node_loss / train_batches
        avg_tra_edge_loss = tra_edge_loss / train_batches
        avg_tra_node_cls = tra_node_cls / train_batches
        avg_tra_node_conf = tra_node_conf / train_batches
        avg_tra_node_smooth = tra_node_smooth / train_batches
        avg_tra_edge_base = tra_edge_base / train_batches
        avg_tra_edge_sparse = tra_edge_sparse / train_batches
        avg_tra_edge_dynw = tra_edge_dynw / train_batches
        avg_tra_pp_loss = tra_pp_loss / train_batches
        avg_tra_pl_loss = tra_pl_loss / train_batches
        avg_tra_pp_alpha = tra_pp_alpha / train_batches
        avg_tra_pl_alpha = tra_pl_alpha / train_batches

        # ---- Validation ----
        model.eval()
        loss_fn.eval()

        val_loss_sum, val_node_loss, val_edge_loss = 0.0, 0.0, 0.0
        val_node_cls, val_node_conf, val_node_smooth = 0.0, 0.0, 0.0
        val_edge_base, val_edge_sparse, val_edge_dynw = 0.0, 0.0, 0.0
        val_batches = 0

        val_pbar = tqdm(
            val_loader,
            desc=f'Epoch {epoch+1}/{args.epochs} [Val]',
            bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}',
            colour='green'
        )

        with torch.no_grad():
            for batch in val_pbar:
                batch = batch.to(device)
                out = model(batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict, batch)
                loss, loss_details = loss_fn(out, batch, epoch=epoch)

                val_loss_sum += loss_details.get('total_loss', loss.item())
                val_node_loss += loss_details.get('node_loss', 0.0)
                val_edge_loss += loss_details.get('edge_loss', 0.0)
                val_node_cls += loss_details.get('node_cls_loss', 0.0)
                val_node_conf += loss_details.get('node_conf_penalty', 0.0)
                val_node_smooth += loss_details.get('node_smoothing_mean', 0.0)
                val_edge_base += loss_details.get('edge_base_loss', 0.0)
                val_edge_sparse += loss_details.get('edge_sparsity_loss', 0.0)
                val_edge_dynw += loss_details.get('edge_dynamic_weight', 0.0)
                val_batches += 1

                val_pbar.set_postfix({'v_loss': f"{loss_details.get('total_loss', loss.item()):.4f}"})

        val_batches = max(val_batches, 1)

        avg_val_loss = val_loss_sum / val_batches
        avg_val_node_loss = val_node_loss / val_batches
        avg_val_edge_loss = val_edge_loss / val_batches
        avg_val_node_cls = val_node_cls / val_batches
        avg_val_node_conf = val_node_conf / val_batches
        avg_val_node_smooth = val_node_smooth / val_batches
        avg_val_edge_base = val_edge_base / val_batches
        avg_val_edge_sparse = val_edge_sparse / val_batches
        avg_val_edge_dynw = val_edge_dynw / val_batches

        print("\n" + "=" * 80)
        print(f"Epoch {epoch+1:03d} Detailed loss (single GPU):")
        print("-" * 80)
        print("Train loss:")
        print(f"   Total loss: {avg_total_loss:.6f}")
        print(f"   node loss: {avg_tra_node_loss:.6f}")
        print(f"   edge loss: {avg_tra_edge_loss:.6f}")
        print(f"   node classification loss {avg_tra_node_cls:.6f}")
        print(f"   node confidence penalty: {avg_tra_node_conf:.6f}")
        print(f"   node smooth: {avg_tra_node_smooth:.6f}")
        print(f"   edge basic loss: {avg_tra_edge_base:.6f}")
        print(f"   edge sparse: {avg_tra_edge_sparse:.6f}")
        print(f"   edge weight: {avg_tra_edge_dynw:.6f}")
        print(f"   PP loss: {avg_tra_pp_loss:.6f} (alpha: {avg_tra_pp_alpha:.6f})")
        print(f"   PL loss: {avg_tra_pl_loss:.6f} (alpha: {avg_tra_pl_alpha:.6f})")
        print("-" * 80)
        print("Valid loss:")
        print(f"   Total loss: {avg_val_loss:.6f}")
        print(f"  node loss: {avg_val_node_loss:.6f}")
        print(f"  edge loss: {avg_val_edge_loss:.6f}")
        print(f"  node classification loss: {avg_val_node_cls:.6f}")
        print(f"  node confidence penalty: {avg_val_node_conf:.6f}")
        print(f"  node smooth: {avg_val_node_smooth:.6f}")
        print(f"  edge basic loss: {avg_val_edge_base:.6f}")
        print(f"  edge sparse: {avg_val_edge_sparse:.6f}")
        print(f"  edge weight: {avg_val_edge_dynw:.6f}")
        print("-" * 80)
        print(f"LR: {optimizer.param_groups[0]['lr']:.2e}")
        print("=" * 80)

        log_data = [
            epoch + 1,
            avg_total_loss, avg_tra_node_loss, avg_tra_edge_loss,
            avg_tra_node_cls, avg_tra_node_conf, avg_tra_node_smooth,
            avg_tra_edge_base, avg_tra_edge_sparse, avg_tra_edge_dynw,
            avg_tra_pp_loss, avg_tra_pp_alpha, avg_tra_pl_loss, avg_tra_pl_alpha,
            avg_val_loss, avg_val_node_loss, avg_val_edge_loss,
            avg_val_node_cls, avg_val_node_conf, avg_val_node_smooth,
            avg_val_edge_base, avg_val_edge_sparse, avg_val_edge_dynw,
            optimizer.param_groups[0]['lr']
        ]
        with open(args.log, 'a') as f:
            f.write(
                '    '.join(
                    map(lambda x: f"{x:.5f}" if isinstance(x, float) else str(x), log_data)
                ) + '\n'
            )
            f.flush()

        if avg_val_node_cls < best_val_loss:
            best_val_loss = avg_val_node_cls
            torch.save(model.state_dict(), args.bestmodel)
            print(f" Bestmodel saved (node classification loss: {best_val_loss:.6f})")

        if (epoch + 1) % args.save_interval == 0:
            ckpt_path = os.path.join(args.models_dir, f"model_{epoch+1:03d}.pth")
            torch.save(model.state_dict(), ckpt_path)
            full_ckpt_path = os.path.join(args.models_dir, f"checkpoint_epoch_{epoch+1:03d}.pth")
            save_checkpoint(epoch, model, optimizer, scheduler, loss_fn,
                            early_stopping, best_val_loss, full_ckpt_path, is_ddp=False)
            print(f" Save model: {ckpt_path}")
            print(f" Save checkpoint: {full_ckpt_path}")

        latest_path = os.path.join(args.models_dir, "checkpoint_latest.pth")
        save_checkpoint(epoch, model, optimizer, scheduler, loss_fn,
                        early_stopping, best_val_loss, latest_path, is_ddp=False)

        early_stopping(avg_val_node_cls)
        if early_stopping.early_stop:
            print(f" Early stopped at epoch {epoch+1}")
            break

    torch.save(model.state_dict(), args.finalmodel)
    final_ckpt_path = os.path.join(args.models_dir, "checkpoint_final.pth")
    save_checkpoint(epoch, model, optimizer, scheduler, loss_fn,
                    early_stopping, best_val_loss, final_ckpt_path, is_ddp=False)
    print("Finish.")


def calculate_residue_class_counts_streaming(dataset, max_samples=500, num_classes=20):
    counts = torch.zeros(num_classes, dtype=torch.long)
    n = len(dataset) if max_samples is None else min(len(dataset), max_samples)

    for i in range(n):
        data = dataset[i]
        y = data['residue'].y.cpu()
        counts += torch.bincount(y, minlength=num_classes)

    return counts

def main_worker(rank, world_size, args):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(args.master_port)

    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    device = torch.device(f"cuda:{rank}")

    torch.autograd.set_detect_anomaly(False)
    seed_everything(args.seed + rank)

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    # ---- Resume handling ----
    start_epoch = 0
    resume_checkpoint = None
    if args.resume:
        if not os.path.isfile(args.resume):
            raise FileNotFoundError(f"Resume checkpoint not found: {args.resume}")
        resume_checkpoint = torch.load(args.resume, map_location=torch.device('cpu'))
        start_epoch = resume_checkpoint['epoch'] + 1
        if rank == 0:
            print(f"Resuming from epoch {start_epoch}, checkpoint: {args.resume}")

    early_stopping = EarlyStopping(patience=args.patience, delta=args.delta)
    if rank == 0 and resume_checkpoint is not None:
        early_stopping.best_loss = resume_checkpoint['early_stopping_best_loss']
        early_stopping.counter = resume_checkpoint['early_stopping_counter']

    train_lmdbs = collect_train_lmdbs(args)
    train_dataset = RandomMultiLMDBHeteroDataset(train_lmdbs, strict_same_length=True)
    val_dataset = LMDBHeteroDataset(args.valid_lmdb)

    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=True,
    )
    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        drop_last=True,
    )

    num_workers = args.num_workers
    pin_memory = True

    train_loader = build_loader(
        dataset=train_dataset,
        batch_size=args.batchsz,
        shuffle=False,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        prefetch_factor=args.prefetch_factor,
    )
    val_loader = build_loader(
        dataset=val_dataset,
        batch_size=args.batchsz,
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        prefetch_factor=args.prefetch_factor,
    )

    # edge zero ratio statistics
    if rank == 0:
        edge_type_info = calculate_zero_ratios_streaming(
            train_dataset,
            max_samples=args.zero_ratio_samples
        )
    else:
        edge_type_info = {}

    edge_type_info_list = [
        edge_type_info.get(('residue', 'interacts_between', 'residue'), 0.5),
        edge_type_info.get(('ligand', 'interacts_with', 'residue'), 0.5),
    ]
    edge_tensor = torch.tensor(edge_type_info_list, device=device, dtype=torch.float32)
    dist.broadcast(edge_tensor, src=0)

    edge_type_info = {
        ('residue', 'interacts_between', 'residue'): edge_tensor[0].item(),
        ('ligand', 'interacts_with', 'residue'): edge_tensor[1].item(),
    }

    # class counts statistics
    if rank == 0:
        class_counts = calculate_residue_class_counts_streaming(
            train_dataset,
            max_samples=args.class_count_samples,
            num_classes=20
        )
    else:
        class_counts = torch.zeros(20, dtype=torch.long)

    class_count_tensor = class_counts.to(device=device, dtype=torch.float32)
    dist.broadcast(class_count_tensor, src=0)
    class_counts = class_count_tensor.cpu()

    sample_data = train_dataset[0]
    model = MPNN_HeteroGNN(
        num_residue_types=sample_data["residue"].x.shape[1],
        num_lig_atom_types=sample_data["ligand"].x.shape[1],
        num_interac_PP_type=sample_data[('residue', 'interacts_between', 'residue')].edge_attr.shape[1],
        num_interac_PL_type=sample_data[('ligand', 'interacts_with', 'residue')].edge_attr.shape[1],
        num_blocks=4,
        hidden_dim=256,
        num_heads=4,
        dropout_rate=args.dropout,
    ).to(device)

    if resume_checkpoint is not None:
        model.load_state_dict(resume_checkpoint['model_state_dict'])
        if rank == 0:
            print(f"Loaded model weights from resume checkpoint (epoch {resume_checkpoint['epoch']})")
    elif args.basemodel is not None:
        model = load_model_weights(model, args.basemodel, device, rank=rank)

    model = DistributedDataParallel(model, device_ids=[rank], find_unused_parameters=False)

    # loss no longer needs to be optimized
    loss_fn = HeteroNodeClassificationLoss(
        num_classes=20,
        class_counts=class_counts,
        focal_gamma=args.focal_gamma,
        base_label_smoothing=args.base_label_smoothing,
        ldas_power=args.ldas_power,
        edge_loss_weight=args.edge_loss_weight,
        edge_warmup_epochs=args.edge_warmup_epochs,
        confidence_penalty_weight=args.confidence_penalty_weight,
        edge_sparsity_weight=args.edge_sparsity_weight,
        edge_type_info=edge_type_info,
        device=device,
    ).to(device)

    if resume_checkpoint is not None:
        loss_fn.load_state_dict(resume_checkpoint['loss_fn_state_dict'])
        if rank == 0:
            print("Loaded loss function state from resume checkpoint")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.98),
        eps=1e-9,
    )

    total_steps = args.epochs * ((len(train_loader) + args.accum_steps - 1) // args.accum_steps)
    scheduler = build_scheduler(optimizer, total_steps)

    if resume_checkpoint is not None and resume_checkpoint['scheduler_state_dict'] is not None:
        scheduler.load_state_dict(resume_checkpoint['scheduler_state_dict'])
        if rank == 0:
            print("Loaded scheduler state from resume checkpoint")

    if resume_checkpoint is not None:
        optimizer.load_state_dict(resume_checkpoint['optimizer_state_dict'])
        if rank == 0:
            print("Loaded optimizer state from resume checkpoint")

    if rank == 0:
        if resume_checkpoint is not None:
            best_val_loss = resume_checkpoint['best_val_loss']
            log_mode = 'a'
            print(f"Resumed best_val_loss = {best_val_loss:.6f}, appending to existing log.")
        else:
            best_val_loss = float('inf')
            log_mode = 'w'

        with open(args.log, log_mode) as log_file:
            if log_mode == 'w':
                headers = [
                    'Epoch',
                    'Tra_Loss', 'Tra_Node', 'Tra_Edge',
                    'Tra_NodeCls', 'Tra_NodeConf', 'Tra_NodeSmooth',
                    'Tra_EdgeBase', 'Tra_EdgeSparse', 'Tra_EdgeDynW',
                    'Tra_PP_Loss', 'Tra_PP_Alpha',
                    'Tra_PL_Loss', 'Tra_PL_Alpha',
                    'Val_Loss', 'Val_Node', 'Val_Edge',
                    'Val_NodeCls', 'Val_NodeConf', 'Val_NodeSmooth',
                    'Val_EdgeBase', 'Val_EdgeSparse', 'Val_EdgeDynW',
                    'L_Rate'
                ]
                log_file.write('  '.join(headers) + '\n')
                log_file.flush()
    else:
        best_val_loss = None

    for epoch in range(start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)

        model.train()
        loss_fn.train()
        optimizer.zero_grad(set_to_none=True)

        total_loss_sum = 0.0
        tra_node_loss = 0.0
        tra_edge_loss = 0.0
        tra_node_cls = 0.0
        tra_node_conf = 0.0
        tra_node_smooth = 0.0
        tra_edge_base = 0.0
        tra_edge_sparse = 0.0
        tra_edge_dynw = 0.0
        tra_pp_loss, tra_pl_loss = 0.0, 0.0
        tra_pp_alpha, tra_pl_alpha = 0.0, 0.0
        train_batches = 0

        if rank == 0:
            train_pbar = tqdm(
                train_loader,
                desc=f'Epoch {epoch+1}/{args.epochs} [Train]',
                bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'
            )
        else:
            train_pbar = train_loader

        accum_steps = args.accum_steps

        for batch_idx, batch in enumerate(train_pbar):
            batch = batch.to(device)

            is_update_step = (
                (batch_idx + 1) % accum_steps == 0
                or (batch_idx + 1) == len(train_loader)
            )

            if not is_update_step:
                with model.no_sync():
                    out = model(batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict, batch)
                    loss, loss_details = loss_fn(out, batch, epoch=epoch)

                    if not torch.isfinite(loss):
                        if rank == 0:
                            print(f"Warning: infinite loss for Batch {batch_idx}")
                        optimizer.zero_grad(set_to_none=True)
                        continue

                    (loss / accum_steps).backward()
            else:
                out = model(batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict, batch)
                loss, loss_details = loss_fn(out, batch, epoch=epoch)

                if not torch.isfinite(loss):
                    if rank == 0:
                        print(f"Warning: infinite loss in Batch {batch_idx}")
                    optimizer.zero_grad(set_to_none=True)
                    continue

                (loss / accum_steps).backward()

                grad_norm_model = nn.utils.clip_grad_norm_(model.parameters(), 0.8)
                if torch.isnan(grad_norm_model) or torch.isinf(grad_norm_model):
                    if rank == 0:
                        print(f"Warning: invalid gradient in Batch {batch_idx}")
                    optimizer.zero_grad(set_to_none=True)
                    continue

                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            total_loss_sum += loss.item()
            tra_node_loss += loss_details.get('node_loss', 0.0)
            tra_edge_loss += loss_details.get('edge_loss', 0.0)
            tra_node_cls += loss_details.get('node_cls_loss', 0.0)
            tra_node_conf += loss_details.get('node_conf_penalty', 0.0)
            tra_node_smooth += loss_details.get('node_smoothing_mean', 0.0)
            tra_edge_base += loss_details.get('edge_base_loss', 0.0)
            tra_edge_sparse += loss_details.get('edge_sparsity_loss', 0.0)
            tra_edge_dynw += loss_details.get('edge_dynamic_weight', 0.0)
            tra_pp_loss += loss_details.get('edge_residue_interacts_between_residue_loss', 0.0)
            tra_pl_loss += loss_details.get('edge_ligand_interacts_with_residue_loss', 0.0)
            tra_pp_alpha += loss_details.get('edge_residue_interacts_between_residue_alpha', 0.0)
            tra_pl_alpha += loss_details.get('edge_ligand_interacts_with_residue_alpha', 0.0)
            train_batches += 1

            if rank == 0:
                train_pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'lr': f"{optimizer.param_groups[0]['lr']:.2e}"
                })

        model.eval()
        loss_fn.eval()

        val_loss_sum = 0.0
        val_node_loss = 0.0
        val_edge_loss = 0.0
        val_node_cls = 0.0
        val_node_conf = 0.0
        val_node_smooth = 0.0
        val_edge_base = 0.0
        val_edge_sparse = 0.0
        val_edge_dynw = 0.0
        val_batches = 0

        if rank == 0:
            val_pbar = tqdm(
                val_loader,
                desc=f'Epoch {epoch+1}/{args.epochs} [Val]',
                bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}',
                colour='green'
            )
        else:
            val_pbar = val_loader

        with torch.no_grad():
            for batch in val_pbar:
                batch = batch.to(device)
                out = model(batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict, batch)
                loss, loss_details = loss_fn(out, batch, epoch=epoch)

                val_loss_sum += loss_details.get('total_loss', loss.item())
                val_node_loss += loss_details.get('node_loss', 0.0)
                val_edge_loss += loss_details.get('edge_loss', 0.0)
                val_node_cls += loss_details.get('node_cls_loss', 0.0)
                val_node_conf += loss_details.get('node_conf_penalty', 0.0)
                val_node_smooth += loss_details.get('node_smoothing_mean', 0.0)
                val_edge_base += loss_details.get('edge_base_loss', 0.0)
                val_edge_sparse += loss_details.get('edge_sparsity_loss', 0.0)
                val_edge_dynw += loss_details.get('edge_dynamic_weight', 0.0)
                val_batches += 1

                if rank == 0:
                    val_pbar.set_postfix({'v_loss': f"{loss_details.get('total_loss', loss.item()):.4f}"})

        train_batches = max(train_batches, 1)
        val_batches = max(val_batches, 1)

        stats_tensor = torch.tensor(
            [
                total_loss_sum,
                tra_node_loss,
                tra_edge_loss,
                tra_node_cls,
                tra_node_conf,
                tra_node_smooth,
                tra_edge_base,
                tra_edge_sparse,
                tra_edge_dynw,
                tra_pp_loss,
                tra_pp_alpha,
                tra_pl_loss,
                tra_pl_alpha,
                val_loss_sum,
                val_node_loss,
                val_edge_loss,
                val_node_cls,
                val_node_conf,
                val_node_smooth,
                val_edge_base,
                val_edge_sparse,
                val_edge_dynw,
            ],
            device=device,
            dtype=torch.float64,
        )

        dist.all_reduce(stats_tensor, op=dist.ReduceOp.SUM)

        avg_total_loss = stats_tensor[0].item() / (world_size * train_batches)
        avg_tra_node_loss = stats_tensor[1].item() / (world_size * train_batches)
        avg_tra_edge_loss = stats_tensor[2].item() / (world_size * train_batches)
        avg_tra_node_cls = stats_tensor[3].item() / (world_size * train_batches)
        avg_tra_node_conf = stats_tensor[4].item() / (world_size * train_batches)
        avg_tra_node_smooth = stats_tensor[5].item() / (world_size * train_batches)
        avg_tra_edge_base = stats_tensor[6].item() / (world_size * train_batches)
        avg_tra_edge_sparse = stats_tensor[7].item() / (world_size * train_batches)
        avg_tra_edge_dynw = stats_tensor[8].item() / (world_size * train_batches)
        avg_tra_pp_loss = stats_tensor[9].item() / (world_size * train_batches)
        avg_tra_pp_alpha = stats_tensor[10].item() / (world_size * train_batches)
        avg_tra_pl_loss = stats_tensor[11].item() / (world_size * train_batches)
        avg_tra_pl_alpha = stats_tensor[12].item() / (world_size * train_batches)

        avg_val_loss = stats_tensor[13].item() / (world_size * val_batches)
        avg_val_node_loss = stats_tensor[14].item() / (world_size * val_batches)
        avg_val_edge_loss = stats_tensor[15].item() / (world_size * val_batches)
        avg_val_node_cls = stats_tensor[16].item() / (world_size * val_batches)
        avg_val_node_conf = stats_tensor[17].item() / (world_size * val_batches)
        avg_val_node_smooth = stats_tensor[18].item() / (world_size * val_batches)
        avg_val_edge_base = stats_tensor[19].item() / (world_size * val_batches)
        avg_val_edge_sparse = stats_tensor[20].item() / (world_size * val_batches)
        avg_val_edge_dynw = stats_tensor[21].item() / (world_size * val_batches)

        if rank == 0:
            print("\n" + "=" * 80)
            print(f"Epoch {epoch+1:03d} Detailed loss (DDP, {world_size} GPUs):")
            print("-" * 80)
            print("Train loss:")
            print(f"  Total loss: {avg_total_loss:.6f}")
            print(f"   node loss: {avg_tra_node_loss:.6f}")
            print(f"   edge loss: {avg_tra_edge_loss:.6f}")
            print(f"   node classification loss {avg_tra_node_cls:.6f}")
            print(f"   node confidence penalty: {avg_tra_node_conf:.6f}")
            print(f"   node smooth: {avg_tra_node_smooth:.6f}")
            print(f"   edge basic loss: {avg_tra_edge_base:.6f}")
            print(f"   edge sparse: {avg_tra_edge_sparse:.6f}")
            print(f"   edge weight: {avg_tra_edge_dynw:.6f}")
            print(f"   PP loss: {avg_tra_pp_loss:.6f} (alpha: {avg_tra_pp_alpha:.6f})")
            print(f"   PL loss: {avg_tra_pl_loss:.6f} (alpha: {avg_tra_pl_alpha:.6f})")
            print("-" * 80)
            print("Valid loss:")
            print(f"   Total loss: {avg_val_loss:.6f}")
            print(f"  node loss: {avg_val_node_loss:.6f}")
            print(f"  edge loss: {avg_val_edge_loss:.6f}")
            print(f"  node classification loss: {avg_val_node_cls:.6f}")
            print(f"  node confidence penalty: {avg_val_node_conf:.6f}")
            print(f"  node smooth: {avg_val_node_smooth:.6f}")
            print(f"  edge basic loss: {avg_val_edge_base:.6f}")
            print(f"  edge sparse: {avg_val_edge_sparse:.6f}")
            print(f"  edge weight: {avg_val_edge_dynw:.6f}")
            print("-" * 80)
            print(f"LR: {optimizer.param_groups[0]['lr']:.2e}")
            print("=" * 80)

            log_data = [
                epoch + 1,
                avg_total_loss, avg_tra_node_loss, avg_tra_edge_loss,
                avg_tra_node_cls, avg_tra_node_conf, avg_tra_node_smooth,
                avg_tra_edge_base, avg_tra_edge_sparse, avg_tra_edge_dynw,
                avg_tra_pp_loss, avg_tra_pp_alpha, avg_tra_pl_loss, avg_tra_pl_alpha,
                avg_val_loss, avg_val_node_loss, avg_val_edge_loss,
                avg_val_node_cls, avg_val_node_conf, avg_val_node_smooth,
                avg_val_edge_base, avg_val_edge_sparse, avg_val_edge_dynw,
                optimizer.param_groups[0]['lr']
            ]
            with open(args.log, 'a') as f:
                f.write(
                    '    '.join(
                        map(lambda x: f"{x:.5f}" if isinstance(x, float) else str(x), log_data)
                    ) + '\n'
                )
                f.flush()

        if rank == 0 and avg_val_node_cls < best_val_loss:
            best_val_loss = avg_val_node_cls
            torch.save(model.module.state_dict(), args.bestmodel)
            print(f" Bestmodel saved (node classification loss: {best_val_loss:.6f})")

        if rank == 0:
            if (epoch + 1) % args.save_interval == 0:
                ckpt_path = os.path.join(args.models_dir, f"model_{epoch+1:03d}.pth")
                torch.save(model.module.state_dict(), ckpt_path)
                full_ckpt_path = os.path.join(args.models_dir, f"checkpoint_epoch_{epoch+1:03d}.pth")
                save_checkpoint(epoch, model, optimizer, scheduler, loss_fn,
                                early_stopping, best_val_loss, full_ckpt_path, is_ddp=True)
                print(f" Save model: {ckpt_path}")
                print(f" Save checkpoint: {full_ckpt_path}")
            latest_path = os.path.join(args.models_dir, "checkpoint_latest.pth")
            save_checkpoint(epoch, model, optimizer, scheduler, loss_fn,
                            early_stopping, best_val_loss, latest_path, is_ddp=True)

        if rank == 0:
            early_stopping(avg_val_node_cls)

        stop_signal = torch.tensor([int(early_stopping.early_stop)], dtype=torch.int, device=device)
        dist.broadcast(stop_signal, src=0)

        if stop_signal.item() == 1:
            if rank == 0:
                print(f" Early stopped at epoch {epoch+1}")
            break

    if rank == 0:
        torch.save(model.module.state_dict(), args.finalmodel)
        final_ckpt_path = os.path.join(args.models_dir, "checkpoint_final.pth")
        save_checkpoint(epoch, model, optimizer, scheduler, loss_fn,
                        early_stopping, best_val_loss, final_ckpt_path, is_ddp=True)
        print("Finish.")

    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_lmdb_dir", type=str, default="../lmdb_data")
    parser.add_argument("--train_lmdbs", type=str, nargs='*', default=None,
                        help="Optional explicit list of training LMDB files. If omitted, use train_*.lmdb under --train_lmdb_dir")
    parser.add_argument("--valid_lmdb", type=str, default="../lmdb_data/valid.lmdb")

    parser.add_argument("-lr", type=float, default=0.001)
    parser.add_argument("-batchsz", type=int, default=16)
    parser.add_argument("-epochs", type=int, default=1000)
    parser.add_argument("-device", type=str, default=None)
    parser.add_argument("-log", type=str, default="log.dat")
    parser.add_argument("-bestmodel", type=str, default="best_model.pth")
    parser.add_argument("-finalmodel", type=str, default="final_model.pth")
    parser.add_argument("-dropout", type=float, default=0.5)
    parser.add_argument("-save_interval", type=int, default=10,
                        help="Save checkpoint every N epochs (default: 10)")
    parser.add_argument("-save_dir", type=str, default='models',
                        help="Save checkpoint to save_dir/ directory (default: models)")

    parser.add_argument("-basemodel", type=str, default=None,
                        help="Path to a pre-trained .pth checkpoint. If specified, load these weights instead of random initialization.")

    parser.add_argument("-edge_loss_weight", type=float, default=8)

    parser.add_argument("-label_smoothing", type=float, default=0.05)

    parser.add_argument("-weight_decay", type=float, default=0.01)
    parser.add_argument("-patience", type=int, default=50)
    parser.add_argument("-delta", type=float, default=0.0003)

    parser.add_argument("-multi_gpu", action="store_true", help="Enable multi-gpu training")
    parser.add_argument('--accum_steps', type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--prefetch_factor", type=int, default=4)
    parser.add_argument("--zero_ratio_samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--master_port", type=int, default=12355)
    
    parser.add_argument("--class_count_samples", type=int, default=500,
                        help="Number of training samples used to estimate residue class counts")
    parser.add_argument("--focal_gamma", type=float, default=2.0,
                        help="Gamma for focal loss")
    parser.add_argument("--base_label_smoothing", type=float, default=0.05,
                        help="Base label smoothing for LDAS")
    parser.add_argument("--ldas_power", type=float, default=1.0,
                        help="Power term for label-distribution-aware smoothing")
    parser.add_argument("--edge_warmup_epochs", type=int, default=10,
                        help="Warmup epochs for dynamic edge loss weight")
    parser.add_argument("--confidence_penalty_weight", type=float, default=0.01,
                        help="Weight for entropy-based confidence penalty")
    parser.add_argument("--edge_sparsity_weight", type=float, default=1e-5,
                        help="Weight for edge sparsity regularization")

    parser.add_argument("-resume", type=str, default=None,
                        help="Path to checkpoint file for resuming training. Overrides -basemodel.")

    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(script_dir, args.save_dir)
    os.makedirs(models_dir, exist_ok=True)
    print(f"Checkpoints will be saved to: {models_dir}")
    args.models_dir = models_dir

    if args.multi_gpu and torch.cuda.device_count() > 1:
        import torch.multiprocessing as mp

        world_size = torch.cuda.device_count()
        print(f"Train with DistributedDataParallel, {world_size} GPU")

        mp.set_start_method("spawn", force=True)
        mp.spawn(
            main_worker,
            args=(world_size, args),
            nprocs=world_size,
            join=True,
        )
    else:
        print("Train with one single GPU")
        early_stopping = EarlyStopping(patience=args.patience, delta=args.delta)
        train_single_gpu(args, early_stopping)
