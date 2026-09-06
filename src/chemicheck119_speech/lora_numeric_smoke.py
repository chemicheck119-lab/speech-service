"""Bounded MPS numeric smoke gate for Whisper LoRA training."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from importlib.metadata import version as distribution_version
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

from .lora_data_preflight import (
    _load_execution_config,
    _object,
    validate_lora_data_preflight,
)
from .lora_protocol import load_experiment_config
from .lora_tokenizer_preflight import validate_lora_tokenizer_preflight
from .lora_training import (
    CONFIRMATION_PHRASE,
    WhisperDataCollator,
    WhisperSegmentDataset,
    _clean_metrics,
    _expected_snapshot_map,
    _install_deadline,
    _secure_write,
    build_finite_training_callback,
    build_whisper_lora_config,
    materialize_training_examples,
    validate_authorization_claim,
    validate_cost_quote,
    validate_gpu_runtime,
)


NUMERIC_SMOKE_PROTOCOL_ID = "whisper-small-lora-numeric-smoke-v1"
SMOKE_OPTIMIZER_STEPS = 2
SMOKE_EXAMPLES = 32
SMOKE_DEADLINE_SECONDS = 30 * 60


def run_numeric_smoke(
    *,
    execution_config_path: Path,
    experiment_config_path: Path,
    artifact_root: Path,
    cost_quote_path: Path,
    authorization_claim_path: Path,
    output_dir: Path,
    confirmation: str,
    runner_revision: str,
) -> dict[str, object]:
    """Run two optimizer steps and write only an aggregate numeric report."""

    if confirmation != CONFIRMATION_PHRASE:
        raise PermissionError("explicit bounded-experiment confirmation is required")
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError("refusing to overwrite numeric smoke output")
    execution, execution_bytes = _load_execution_config(execution_config_path)
    experiment, experiment_bytes = load_experiment_config(experiment_config_path)
    training = _object(experiment["training"], "experiment training")
    if (
        training.get("parameter_dtype") != "float32"
        or training.get("mixed_precision") != "fp16"
    ):
        raise ValueError("numeric smoke requires FP32 master weights and FP16 autocast")
    cost = validate_cost_quote(
        quote_path=cost_quote_path,
        execution_config=execution,
        runner_revision=runner_revision,
    )
    claim = validate_authorization_claim(
        claim_path=authorization_claim_path,
        cost=cost,
        runner_revision=runner_revision,
        expected_gcs_prefix=str(execution["private_output"]["gcs_prefix"]),
    )
    runtime_report = validate_gpu_runtime(execution)
    started = datetime.now(timezone.utc)
    started_monotonic = time.monotonic()

    os.environ["USE_TF"] = "0"
    try:
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import (
            Seq2SeqTrainer,
            Seq2SeqTrainingArguments,
            TrainerCallback,
            WhisperForConditionalGeneration,
            WhisperProcessor,
        )
    except ImportError as error:
        raise RuntimeError("pinned LoRA dependencies are unavailable") from error

    models = _object(experiment["models"], "experiment models")
    base = _object(models["transformers_base"], "transformers base")
    processor = WhisperProcessor.from_pretrained(
        str(base["id"]),
        revision=str(base["revision"]),
        language=str(training["language"]),
        task=str(training["task"]),
    )
    tokenizer_report = validate_lora_tokenizer_preflight(
        execution_config_path=execution_config_path,
        experiment_config_path=experiment_config_path,
        artifact_root=artifact_root,
        tokenizer=processor.tokenizer,
        tokenizer_version=distribution_version("transformers"),
    )
    if (
        tokenizer_report["status"] != "limited"
        or not tokenizer_report["training_eligible_by_token_limit"]
    ):
        raise RuntimeError("tokenizer preflight rejected numeric smoke")
    data_report = validate_lora_data_preflight(
        execution_config_path=execution_config_path,
        experiment_config_path=experiment_config_path,
        artifact_root=artifact_root,
    )

    output_dir.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    output_dir.parent.chmod(0o700)
    stage = output_dir.parent / f".{output_dir.name}.stage"
    work = output_dir.parent / f".{output_dir.name}.work"
    if any(path.exists() or path.is_symlink() for path in (stage, work)):
        raise FileExistsError("refusing to reuse numeric smoke staging paths")
    stage.mkdir(mode=0o700)
    work.mkdir(mode=0o700)
    restore_deadline = _install_deadline(SMOKE_DEADLINE_SECONDS)
    try:
        examples, assignment_counts = materialize_training_examples(
            artifact_root=artifact_root,
            output_dir=work / "audio",
            execution_config=execution,
            expected_snapshots=_expected_snapshot_map(data_report),
        )
        if len(examples) < SMOKE_EXAMPLES:
            raise ValueError("numeric smoke does not have enough training examples")
        dataset = WhisperSegmentDataset(
            examples[:SMOKE_EXAMPLES],
            processor,
            sample_rate=int(execution["segment_contract"]["resample_hz"]),
            max_label_tokens=int(execution["segment_contract"]["max_label_tokens"]),
        )
        model = WhisperForConditionalGeneration.from_pretrained(
            str(base["id"]),
            revision=str(base["revision"]),
            torch_dtype=torch.float32,
        )
        model.config.use_cache = False
        model.enable_input_require_grads()
        model = get_peft_model(
            model,
            build_whisper_lora_config(
                _object(experiment["lora"], "LoRA config"), LoraConfig
            ),
        )
        trainable_before = {
            name: parameter.detach().cpu().clone()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        frozen_sample = next(
            parameter.detach().cpu().clone()
            for parameter in model.parameters()
            if not parameter.requires_grad and parameter.numel() <= 1_000_000
        )
        frozen_sample_after = next(
            parameter
            for parameter in model.parameters()
            if not parameter.requires_grad and parameter.numel() <= 1_000_000
        )
        if not trainable_before:
            raise RuntimeError("numeric smoke found no trainable LoRA parameters")

        arguments = Seq2SeqTrainingArguments(
            output_dir=str(work / "trainer"),
            overwrite_output_dir=False,
            max_steps=SMOKE_OPTIMIZER_STEPS,
            per_device_train_batch_size=int(training["per_device_batch_size"]),
            gradient_accumulation_steps=int(training["gradient_accumulation_steps"]),
            learning_rate=float(training["learning_rate"]),
            warmup_ratio=float(training["warmup_ratio"]),
            weight_decay=float(training["weight_decay"]),
            max_grad_norm=float(training["max_gradient_norm"]),
            fp16=True,
            gradient_checkpointing=bool(training["gradient_checkpointing"]),
            gradient_checkpointing_kwargs={
                "use_reentrant": bool(
                    execution["runtime"]["gradient_checkpointing_use_reentrant"]
                )
            },
            eval_strategy="no",
            save_strategy="no",
            logging_strategy="steps",
            logging_steps=1,
            logging_first_step=True,
            logging_nan_inf_filter=False,
            report_to=[],
            remove_unused_columns=False,
            dataloader_num_workers=int(execution["runtime"]["dataloader_num_workers"]),
            dataloader_pin_memory=bool(
                execution["runtime"]["dataloader_pin_memory"]
            ),
            seed=int(training["seed"]),
            data_seed=int(training["seed"]),
            full_determinism=bool(execution["runtime"]["deterministic_algorithms"]),
            tf32=bool(execution["runtime"]["allow_tf32"]),
            predict_with_generate=False,
        )
        torch.use_deterministic_algorithms(True)
        finite_callback = build_finite_training_callback(TrainerCallback, torch)
        trainer = Seq2SeqTrainer(
            model=model,
            args=arguments,
            train_dataset=dataset,
            data_collator=WhisperDataCollator(
                processor, model.config.decoder_start_token_id
            ),
            processing_class=processor,
            callbacks=[finite_callback],
        )
        result = trainer.train()
        parameters_after = dict(model.named_parameters())
        changed = sum(
            not torch.equal(before, parameters_after[name].detach().cpu())
            for name, before in trainable_before.items()
        )
        if changed == 0:
            raise FloatingPointError("numeric smoke did not update any LoRA tensor")
        if not torch.equal(frozen_sample, frozen_sample_after.detach().cpu()):
            raise FloatingPointError("numeric smoke changed a frozen base parameter")
        if len(finite_callback.logged_losses) != SMOKE_OPTIMIZER_STEPS:
            raise FloatingPointError("numeric smoke did not validate every logged loss")

        completed = datetime.now(timezone.utc)
        report = {
            "schema_version": "1.0.0",
            "protocol_id": NUMERIC_SMOKE_PROTOCOL_ID,
            "status": "numeric_smoke_passed",
            "fact_status": "부분 구현 또는 개발용 데모",
            "started_at": started.isoformat().replace("+00:00", "Z"),
            "completed_at": completed.isoformat().replace("+00:00", "Z"),
            "wall_seconds": round(time.monotonic() - started_monotonic, 3),
            "runner_revision": runner_revision,
            "execution_config_sha256": hashlib.sha256(execution_bytes).hexdigest(),
            "experiment_config_sha256": hashlib.sha256(experiment_bytes).hexdigest(),
            "runtime": runtime_report,
            "cost_guard": {
                "authorization_id": cost.authorization_id,
                "authorization_claim_sha256": claim.claim_sha256,
                "authorization_claim_uri": claim.remote_object_uri,
                "quoted_total_krw_with_contingency": cost.quoted_total_krw_with_contingency,
            },
            "smoke": {
                "optimizer_steps": SMOKE_OPTIMIZER_STEPS,
                "example_count": SMOKE_EXAMPLES,
                "parameter_dtype": training["parameter_dtype"],
                "mixed_precision": training["mixed_precision"],
                "trainable_tensor_count": len(trainable_before),
                "changed_trainable_tensor_count": changed,
                "gradient_tensor_checks": finite_callback.gradient_checks,
                "parameter_tensor_checks": finite_callback.parameter_checks,
                "logged_losses": [
                    round(value, 8) for value in finite_callback.logged_losses
                ],
                "frozen_parameter_sample_unchanged": True,
                "metrics": _clean_metrics(result.metrics),
                "source_assignment_counts": assignment_counts,
            },
            "privacy": {
                "contains_record_ids": False,
                "contains_transcripts": False,
                "contains_addresses": False,
                "private_storage_required": True,
                "git_commit_allowed": False,
            },
            "automatic_full_training_allowed": False,
            "next_gate": (
                "review smoke result, then issue a new single-use full-training "
                "authorization"
            ),
            "claim_scope": (
                "numeric stability only; no accuracy, safety, adoption, or deployment "
                "claim"
            ),
        }
        _secure_write(
            stage / "numeric-smoke-report.json",
            (json.dumps(report, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
        )
        stage.chmod(0o700)
        (stage / "numeric-smoke-report.json").chmod(0o600)
        stage.rename(output_dir)
        return report
    finally:
        restore_deadline()
        shutil.rmtree(work, ignore_errors=True)
        if stage.exists():
            shutil.rmtree(stage, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execution-config", type=Path, required=True)
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--cost-quote", type=Path, required=True)
    parser.add_argument("--authorization-claim", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--confirm-bounded-experiment", required=True)
    parser.add_argument("--runner-revision", required=True)
    args = parser.parse_args(argv)
    report = run_numeric_smoke(
        execution_config_path=args.execution_config,
        experiment_config_path=args.experiment_config,
        artifact_root=args.artifact_root,
        cost_quote_path=args.cost_quote,
        authorization_claim_path=args.authorization_claim,
        output_dir=args.output_dir,
        confirmation=args.confirm_bounded_experiment,
        runner_revision=args.runner_revision,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "fact_status": report["fact_status"],
                "automatic_full_training_allowed": report[
                    "automatic_full_training_allowed"
                ],
                "output_dir": str(args.output_dir),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
