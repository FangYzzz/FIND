"""Offline embedding collection and RL-token bottleneck pretraining."""

from __future__ import annotations

import hashlib
import json
import dataclasses
import bisect
import fcntl
import os
import socket
from contextlib import contextmanager
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from resfit.rl_finetuning.off_policy.rl.rl_token import RLTokenVAE


@contextmanager
def embedding_store_lock(root: Path):
    """Fail fast when another process is using the same embedding store."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / ".collection.lock"
    lock_file = lock_path.open("a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            lock_file.seek(0)
            owner = lock_file.read().strip() or "unknown process"
            raise RuntimeError(
                f"Embedding store is already in use: {root} (lock owner: {owner})"
            ) from exc

        lock_file.seek(0)
        lock_file.truncate()
        lock_file.write(f"pid={os.getpid()} host={socket.gethostname()}\n")
        lock_file.flush()
        os.fsync(lock_file.fileno())
        yield
    finally:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        finally:
            lock_file.close()


class ShardedEmbeddingDataset(Dataset):
    """Disk-backed pi0 embedding dataset with a one-shard read cache."""

    def __init__(self, root: Path):
        self.root = Path(root)
        with (self.root / "manifest.json").open("r", encoding="utf-8") as file:
            self.manifest = json.load(file)
        self.shards = self.manifest["shards"]
        self.cumulative = []
        total = 0
        for shard in self.shards:
            total += int(shard["count"])
            self.cumulative.append(total)
        self.episode_ids = [int(value) for value in self.manifest.get("episode_ids", [])]
        self._cached_shard_index = None
        self._cached_payload = None

    @property
    def complete(self):
        return bool(self.manifest.get("complete", False))

    def __len__(self):
        return self.cumulative[-1] if self.cumulative else 0

    def _load_shard(self, shard_index):
        if self._cached_shard_index != shard_index:
            path = self.root / self.shards[shard_index]["file"]
            self._cached_payload = torch.load(path, map_location="cpu", weights_only=True)
            self._cached_shard_index = shard_index
        return self._cached_payload

    def __getitem__(self, index):
        if index < 0:
            index += len(self)
        shard_index = bisect.bisect_right(self.cumulative, index)
        shard_start = 0 if shard_index == 0 else self.cumulative[shard_index - 1]
        local_index = index - shard_start
        payload = self._load_shard(shard_index)
        if "quantized" in payload:
            embedding = payload["quantized"][local_index].float() * payload["scale"][local_index].float()
        else:
            embedding = payload["embeddings"][local_index].float()
        return embedding, payload["masks"][local_index].bool()


def _write_manifest(root, manifest):
    tmp = root / "manifest.tmp"
    with tmp.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)
    os.replace(tmp, root / "manifest.json")


def initialize_embedding_store(root: Path, expected_frames: int, storage_dtype: str, reset=False):
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"
    if manifest_path.exists() and not reset:
        return ShardedEmbeddingDataset(root)
    if storage_dtype not in ("int8", "float16"):
        raise ValueError("rl_token.embedding_storage_dtype must be 'int8' or 'float16'")
    _write_manifest(root, {
        "schema": "pi0_embedding_shards_v1",
        "storage_dtype": storage_dtype,
        "expected_frames": int(expected_frames),
        "complete": False,
        "shards": [],
        "episode_ids": [],
    })
    return ShardedEmbeddingDataset(root)


def append_embedding_shard(root, embeddings, masks, episode_ids, complete=False):
    store = ShardedEmbeddingDataset(root)
    if embeddings:
        x = pad_sequence([value.float() for value in embeddings], batch_first=True)
        valid = pad_sequence([value.bool() for value in masks], batch_first=True)
        if store.manifest["storage_dtype"] == "int8":
            scale = x.abs().amax(dim=(1, 2), keepdim=True).clamp_min(1e-8) / 127.0
            payload = {"quantized": torch.round(x / scale).clamp(-127, 127).to(torch.int8),
                       "scale": scale, "masks": valid}
        else:
            payload = {"embeddings": x.to(torch.float16), "masks": valid}
        shard_index = len(store.shards)
        relative_path = f"shards/part-{shard_index:06d}.pt"
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        torch.save(payload, tmp)
        os.replace(tmp, path)
        store.manifest["shards"].append({"file": relative_path, "count": len(embeddings)})
        store.manifest["episode_ids"].extend(int(value) for value in episode_ids)
    store.manifest["complete"] = bool(complete)
    _write_manifest(root, store.manifest)
    return ShardedEmbeddingDataset(root)


def artifact_id(dataset_name, num_episodes, cfg):
    payload = {
        "schema": "pi0_final_prefix_sharded_v2",
        "dataset": dataset_name,
        "episodes": num_episodes,
        "embedding_key": cfg.vla_embedding_key,
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:10]


def bottleneck_id(embedding_id, cfg):
    cfg_dict = dataclasses.asdict(cfg) if dataclasses.is_dataclass(cfg) else dict(cfg)
    # Cache-control and collection-only fields do not change model weights.
    for key in (
        "force_recollect_embeddings", "force_retrain", "embedding_cache_dir",
        "checkpoint_dir", "offline_num_episodes", "request_field", "obs_key",
        "vla_embedding_key", "batch_size", "max_tokens_per_batch", "num_workers",
    ):
        cfg_dict.pop(key, None)
    payload = {"schema": "rl_token_vae_v2_normalized", "embedding_id": embedding_id, "cfg": cfg_dict}
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:10]


def _split_indices(num_samples, episode_ids, validation_fraction, seed):
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("rl_token.validation_fraction must be in [0, 1)")
    all_indices = list(range(num_samples))
    if validation_fraction == 0.0 or num_samples < 2:
        return all_indices, []
    generator = torch.Generator().manual_seed(seed)
    unique_episodes = sorted(set(episode_ids)) if episode_ids is not None else []
    if len(unique_episodes) >= 2:
        order = torch.randperm(len(unique_episodes), generator=generator).tolist()
        num_val_episodes = max(1, round(len(unique_episodes) * validation_fraction))
        num_val_episodes = min(num_val_episodes, len(unique_episodes) - 1)
        val_episodes = {unique_episodes[i] for i in order[:num_val_episodes]}
        val_indices = [i for i, episode in enumerate(episode_ids) if episode in val_episodes]
        train_indices = [i for i, episode in enumerate(episode_ids) if episode not in val_episodes]
        return train_indices, val_indices
    # A one-episode smoke test cannot be group-split; use a deterministic
    # frame split so validation and early-stopping plumbing can still run.
    order = torch.randperm(num_samples, generator=generator).tolist()
    num_val = min(max(1, round(num_samples * validation_fraction)), num_samples - 1)
    return order[num_val:], order[:num_val]


def _run_validation(model, loader, cfg, device):
    sums = {"loss": 0.0, "recon": 0.0, "kl": 0.0}
    count = 0
    model.eval()
    with torch.no_grad():
        for batch_x, batch_valid in loader:
            batch_x, batch_valid = batch_x.to(device), batch_valid.to(device)
            reconstruction, mu, logvar = model(batch_x, ~batch_valid)
            loss, reconstruction_loss, kl = model.loss(
                reconstruction, batch_x, mu, logvar, batch_valid, cfg.beta_kl
            )
            sums["loss"] += float(loss)
            sums["recon"] += float(reconstruction_loss)
            sums["kl"] += float(kl)
            count += 1
    return {key: value / max(count, 1) for key, value in sums.items()}


def _pad_embedding_batch(batch):
    embeddings, masks = zip(*batch, strict=True)
    # Stored sequences include hundreds of padded prompt positions. Remove
    # them before attention; this preserves every valid VLA token while
    # substantially reducing quadratic Transformer cost.
    compact = [embedding[mask.bool()] for embedding, mask in zip(embeddings, masks, strict=True)]
    padded = pad_sequence(compact, batch_first=True)
    valid = pad_sequence(
        [torch.ones(value.shape[0], dtype=torch.bool) for value in compact], batch_first=True
    )
    return padded, valid


def _evenly_limit_indices(indices, limit):
    if limit is None or limit <= 0 or len(indices) <= limit:
        return indices
    positions = torch.linspace(0, len(indices) - 1, steps=limit).round().long().tolist()
    return [indices[position] for position in positions]


def train_or_load_rl_token(embedding_dataset, cfg, root: Path, device):
    """Train once, save a portable checkpoint, then return a frozen module."""
    ckpt = root / "rl_token.pt"
    if ckpt.exists() and not cfg.force_retrain:
        state = torch.load(ckpt, map_location=device, weights_only=True)
        input_dim = state.get("input_dim")
        if input_dim is None:
            # Backward compatibility for checkpoints created before input_dim
            # was saved explicitly.
            input_proj_weight = state.get("model", {}).get("input_proj.weight")
            if input_proj_weight is None:
                raise KeyError(
                    f"RL-token checkpoint {ckpt} has neither 'input_dim' nor "
                    "'model.input_proj.weight'"
                )
            input_dim = input_proj_weight.shape[1]
        model = RLTokenVAE(
            int(input_dim), cfg.token_dim, cfg.model_dim, cfg.num_heads,
            cfg.encoder_layers, cfg.decoder_layers, cfg.dropout,
        ).to(device)
        model.load_state_dict(state["model"])
    else:
        if embedding_dataset is None or len(embedding_dataset) == 0:
            raise ValueError(
                "VLA embeddings are required to train the RL-token model, but "
                "no embedding dataset was provided"
            )
        first_embedding, _ = embedding_dataset[0]
        input_dim = first_embedding.shape[-1]
        model = RLTokenVAE(
            input_dim, cfg.token_dim, cfg.model_dim, cfg.num_heads,
            cfg.encoder_layers, cfg.decoder_layers, cfg.dropout,
        ).to(device)
        root.mkdir(parents=True, exist_ok=True)
        sequence_length = int(first_embedding.shape[0])
        token_limited_batch_size = max(1, int(cfg.max_tokens_per_batch) // sequence_length)
        effective_batch_size = min(int(cfg.batch_size), token_limited_batch_size, len(embedding_dataset))
        print(
            "RL-token VAE batching: "
            f"requested={cfg.batch_size}, effective={effective_batch_size}, "
            f"sequence_length={sequence_length}, max_tokens_per_batch={cfg.max_tokens_per_batch}"
        )
        train_indices, val_indices = _split_indices(
            len(embedding_dataset), embedding_dataset.episode_ids, cfg.validation_fraction, cfg.split_seed
        )
        available_train, available_val = len(train_indices), len(val_indices)
        train_indices = _evenly_limit_indices(train_indices, cfg.max_train_samples_per_epoch)
        val_indices = _evenly_limit_indices(val_indices, cfg.max_validation_samples)
        train_loader = DataLoader(
            Subset(embedding_dataset, train_indices), batch_size=effective_batch_size,
            # Sequential indices keep disk access shard-local. Random frame
            # shuffling would reload a different multi-MB shard per sample.
            shuffle=False, num_workers=cfg.num_workers, collate_fn=_pad_embedding_batch,
        )
        val_loader = DataLoader(
            Subset(embedding_dataset, val_indices), batch_size=effective_batch_size,
            shuffle=False, num_workers=cfg.num_workers, collate_fn=_pad_embedding_batch,
        ) if val_indices else None
        print(
            f"RL-token split: train={len(train_indices)}/{available_train} frames, "
            f"validation={len(val_indices)}/{available_val} frames "
            f"(episode-grouped={len(set(embedding_dataset.episode_ids)) >= 2})"
        )
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        best_val_loss = float("inf")
        best_epoch = 0
        best_state = None
        epochs_without_improvement = 0
        epoch_bar = tqdm(range(cfg.epochs), desc="Training RL-token VAE")
        for epoch in epoch_bar:
            model.train()
            total_sum = 0.0
            reconstruction_sum = 0.0
            kl_sum = 0.0
            num_batches = 0
            for batch_x, batch_valid in train_loader:
                batch_x, batch_valid = batch_x.to(device), batch_valid.to(device)
                reconstruction, mu, logvar = model(batch_x, ~batch_valid)
                loss, reconstruction_loss, kl = model.loss(
                    reconstruction, batch_x, mu, logvar, batch_valid, cfg.beta_kl
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        "RL-token VAE loss became non-finite. Reduce rl_token.learning_rate "
                        "or inspect the reported reconstruction/KL values."
                    )
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip_norm)
                opt.step()
                total_sum += float(loss.detach())
                reconstruction_sum += float(reconstruction_loss.detach())
                kl_sum += float(kl.detach())
                num_batches += 1
            denom = max(num_batches, 1)
            train_metrics = {
                "loss": total_sum / denom,
                "recon": reconstruction_sum / denom,
                "kl": kl_sum / denom,
            }
            val_metrics = _run_validation(model, val_loader, cfg, device) if val_loader is not None else train_metrics
            epoch_bar.set_postfix({
                "train": f"{train_metrics['loss']:.6f}",
                "val": f"{val_metrics['loss']:.6f}",
            })
            print(
                f"[RLToken VAE] epoch {epoch + 1:03d}/{cfg.epochs:03d} "
                f"train_loss={train_metrics['loss']:.6f} "
                f"train_reconstruction={train_metrics['recon']:.6f} "
                f"train_kl={train_metrics['kl']:.6f} "
                f"val_loss={val_metrics['loss']:.6f} "
                f"val_reconstruction={val_metrics['recon']:.6f} "
                f"val_kl={val_metrics['kl']:.6f}"
            )
            if val_metrics["loss"] < best_val_loss - cfg.early_stopping_min_delta:
                best_val_loss = val_metrics["loss"]
                best_epoch = epoch + 1
                epochs_without_improvement = 0
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            else:
                epochs_without_improvement += 1
                if val_loader is not None and epochs_without_improvement >= cfg.early_stopping_patience:
                    print(
                        f"[RLToken VAE] early stopping at epoch {epoch + 1}; "
                        f"best epoch={best_epoch}, best val_loss={best_val_loss:.6f}"
                    )
                    break
        if best_state is not None:
            model.load_state_dict(best_state)
        torch.save({
            "model": model.state_dict(), "input_dim": input_dim,
            "best_epoch": best_epoch, "best_val_loss": best_val_loss,
            "train_size": len(train_indices), "validation_size": len(val_indices),
        }, ckpt)
    model.eval()
    model.requires_grad_(False)
    return model


@torch.no_grad()
def encode_all(model, embedding_dataset, device):
    tokens = []
    for embedding, valid in tqdm(embedding_dataset, total=len(embedding_dataset), desc="Encoding RL tokens"):
        embedding = embedding[valid]
        token = model.encode(embedding.unsqueeze(0).to(device), None)
        tokens.append(token.squeeze(0).cpu())
    return tokens
