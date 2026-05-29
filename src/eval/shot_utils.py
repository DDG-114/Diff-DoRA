"""
src/eval/shot_utils.py
----------------------
Shared utilities for full-shot / few-shot / zero-shot shared-adapter runs.
"""
from __future__ import annotations

import gc
import random
from pathlib import Path

import numpy as np
import torch
from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments

from src.eval.eval_paper_ablation import _build_prompt
from src.eval.metrics import per_horizon_metrics
from src.models.qwen_peft import generate_batch, get_lora_model, load_model_and_tokenizer
from src.retrieval.knn_retriever import KNNRetriever
from src.train.train_single import EVDataset
from src.utils.node_context import extract_node_static_context, normalise_domain_label


def parse_ratio_csv(raw: str) -> list[float]:
    values = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        values.append(float(part))
    if not values:
        raise ValueError("Expected at least one ratio.")
    return values


def select_sample_subset_with_indices(
    all_samples: list[dict],
    max_eval: int,
    sampling: str,
    seed: int,
) -> list[tuple[int, dict]]:
    indexed = list(enumerate(all_samples))
    if max_eval <= 0:
        count = len(indexed)
    else:
        count = min(max_eval, len(indexed))
    if sampling == "random":
        rng = random.Random(seed)
        return rng.sample(indexed, count)
    return indexed[:count]


def select_sample_subset(all_samples: list[dict], max_eval: int, sampling: str, seed: int) -> list[dict]:
    return [sample for _, sample in select_sample_subset_with_indices(all_samples, max_eval, sampling, seed)]


def select_node_subset(
    all_nodes: list[int],
    max_nodes: int,
    sampling: str,
    seed: int,
) -> list[int]:
    if max_nodes <= 0 or max_nodes >= len(all_nodes):
        return list(all_nodes)
    if sampling == "random":
        rng = random.Random(seed)
        return sorted(rng.sample(all_nodes, max_nodes))
    return list(all_nodes[:max_nodes])


def build_tagged_node_samples(
    raw_samples: list[dict],
    node_indices: list[int],
    *,
    max_items: int = 0,
) -> list[dict]:
    tagged = []
    for sample in raw_samples:
        for node_idx in node_indices:
            tagged.append(dict(sample, node_idx=node_idx))
            if max_items > 0 and len(tagged) >= max_items:
                return tagged
    return tagged


def build_tagged_node_samples_shuffled_pairs(
    raw_samples: list[dict],
    node_indices: list[int],
    *,
    seed: int,
    max_items: int = 0,
) -> list[dict]:
    pairs = [(sample_idx, node_idx) for sample_idx in range(len(raw_samples)) for node_idx in node_indices]
    rng = random.Random(seed)
    rng.shuffle(pairs)
    if max_items > 0:
        pairs = pairs[:max_items]
    return [dict(raw_samples[sample_idx], node_idx=node_idx) for sample_idx, node_idx in pairs]


def make_retriever(
    train_raw_samples: list[dict],
    *,
    use_rag: bool,
    query_device: str = "cpu",
    query_batch_size: int = 128,
    corpus_chunk_size: int = 65536,
) -> KNNRetriever | None:
    if not use_rag:
        return None
    return KNNRetriever(
        train_raw_samples,
        top_k=2,
        query_device=query_device,
        query_batch_size=query_batch_size,
        corpus_chunk_size=corpus_chunk_size,
    )


def load_retriever(path: str | Path, *, use_rag: bool) -> KNNRetriever | None:
    if not use_rag:
        return None
    retriever_path = Path(path)
    if not retriever_path.exists():
        raise FileNotFoundError(f"Retrieval cache not found: {retriever_path}")
    return KNNRetriever.load(retriever_path)


def mask_occ_and_adj(
    occ: np.ndarray,
    adj: np.ndarray | None,
    keep_nodes: list[int],
) -> tuple[np.ndarray, np.ndarray | None]:
    keep = np.zeros(occ.shape[1], dtype=bool)
    keep[np.array(keep_nodes, dtype=int)] = True

    occ_masked = occ.copy()
    occ_masked[:, ~keep] = 0.0

    if adj is None:
        return occ_masked, None

    adj_masked = adj.copy().astype(np.float32)
    adj_masked[~keep, :] = 0.0
    adj_masked[:, ~keep] = 0.0
    return occ_masked, adj_masked


def mask_sample_for_nodes(sample: dict, keep_nodes: list[int]) -> dict:
    keep = np.zeros(sample["x_hist"].shape[1], dtype=bool)
    keep[np.array(keep_nodes, dtype=int)] = True

    masked = dict(sample)
    masked["x_hist"] = np.array(sample["x_hist"], copy=True)
    masked["nbr_feat"] = np.array(sample["nbr_feat"], copy=True)
    masked["y"] = np.array(sample["y"], copy=True)

    masked["x_hist"][:, ~keep] = 0.0
    masked["nbr_feat"][:, ~keep] = 0.0
    masked["y"][:, ~keep] = 0.0
    return masked


def mask_samples_for_nodes(samples: list[dict], keep_nodes: list[int]) -> list[dict]:
    return [mask_sample_for_nodes(sample, keep_nodes) for sample in samples]


def train_shared_adapter(
    *,
    train_items: list[dict],
    splits: dict,
    horizon: int,
    out_dir: Path,
    args,
    retriever: KNNRetriever | None,
):
    base_model, tokenizer = load_model_and_tokenizer()
    model = get_lora_model(
        base_model,
        use_dora=args.use_dora,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
    )
    if args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    elif hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()

    train_ds = EVDataset(
        train_items,
        tokenizer,
        horizon,
        max_length=args.max_length,
        node_idx=0,
        use_rag=args.use_rag,
        retriever=retriever,
        weather=splits.get("weather"),
        price=splits.get("price"),
        node_meta=splits.get("node_meta"),
        node_ids=splits.get("node_ids"),
        poi=splits.get("poi"),
        include_env_diff=args.use_diff_dora,
        prompt_style=args.prompt_style,
        target_style=getattr(args, "target_style", "auto"),
    )
    collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True, pad_to_multiple_of=8)
    training_args = TrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        fp16=True,
        logging_steps=args.logging_steps,
        eval_strategy="no",
        save_strategy="no",
        report_to="none",
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=collator,
    )
    trainer.label_names = ["labels"]
    trainer.train()

    adapter_dir = out_dir / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    train_metrics = dict(getattr(trainer.state, "log_history", [])[-1]) if getattr(trainer.state, "log_history", None) else {}
    del trainer
    del collator
    del train_ds
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return model, tokenizer, adapter_dir, train_metrics


def evaluate_shared_adapter(
    *,
    model,
    tokenizer,
    test_samples: list[dict],
    node_indices: list[int],
    splits: dict,
    horizon: int,
    args,
    retriever: KNNRetriever | None,
    retrieval_query_samples: list[dict] | None = None,
) -> dict:
    subset = select_sample_subset_with_indices(test_samples, args.max_eval, args.sampling, args.seed)
    preds, trues = [], []
    parse_ok = 0
    requested = len(subset) * len(node_indices)

    for sample_index, sample in subset:
        jobs = []
        retrieval_query_sample = None
        if retrieval_query_samples is not None:
            retrieval_query_sample = retrieval_query_samples[sample_index]
        for node_idx in node_indices:
            static_context = extract_node_static_context(
                node_idx,
                node_ids=splits.get("node_ids"),
                node_meta=splits.get("node_meta"),
            )
            domain_label = normalise_domain_label(static_context.get("zone_type"))
            sys_msg, usr_msg = _build_prompt(
                sample=sample,
                node_idx=node_idx,
                horizon=horizon,
                prompt_style=args.prompt_style,
                use_rag=args.use_rag,
                use_diff_dora=args.use_diff_dora,
                retriever=retriever,
                retrieval_query_sample=retrieval_query_sample,
                static_context=static_context,
                domain_label=domain_label,
                splits=splits,
            )
            jobs.append({
                "prompt": (sys_msg, usr_msg),
                "target": sample["y"][:horizon, node_idx],
            })

        for start in range(0, len(jobs), max(1, args.infer_batch_size)):
            chunk = jobs[start:start + max(1, args.infer_batch_size)]
            prompts = [job["prompt"] for job in chunk]
            raw_outputs = generate_batch(
                model,
                tokenizer,
                prompts,
                max_new_tokens=args.max_new_tokens,
            )
            for job, raw_output in zip(chunk, raw_outputs):
                from src.prompts.parser import parse_output
                arr = parse_output(raw_output, expected_len=horizon)
                if arr is not None and len(arr) == horizon:
                    preds.append(arr)
                    trues.append(job["target"])
                    parse_ok += 1

    result = {
        "requested_predictions": requested,
        "parsed_predictions": parse_ok,
        "parse_success_rate": parse_ok / max(requested, 1),
        "metrics": None,
    }
    if preds:
        result["metrics"] = per_horizon_metrics(
            preds,
            trues,
            horizon,
            splits["norm_min"],
            splits["norm_max"],
        )
    return result


def cleanup_model(*objs) -> None:
    for obj in objs:
        del obj
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
