import random
from dataclasses import replace
import torch

from deepspec.data import CacheCollator
from deepspec.data.cuda_prefetcher import move_batch_to_device
from deepspec.utils.metrics import add_metric
from deepspec.modeling.dspark.gemma4 import Gemma4DSparkModel
from deepspec.modeling.dspark.gemma4.config import (
    build_draft_config as build_gemma4_draft_config,
)
from deepspec.modeling.dspark.loss import compute_dspark_loss
from deepspec.modeling.dspark.ouro import compute_ouro_loss
from deepspec.modeling.dspark.qwen3 import Qwen3DSparkModel
from deepspec.modeling.dspark.qwen3.config import (
    build_draft_config as build_qwen3_draft_config,
)
from deepspec.trainer.base_trainer import BaseTrainer


class Qwen3DSparkTrainer(BaseTrainer):
    data_collator_cls = CacheCollator

    def prepare_model_for_training(self):
        mode = self.args.model.get("ouro_training_mode", "baseline")
        if mode != "baseline":
            if self.args.model.get("sample_loop_count", False):
                raise ValueError("Ouro training requires full loop depth")
            if self.draft_model.markov_head is not None or self.draft_model.confidence_head is not None:
                raise ValueError("The initial Ouro implementation is CE-only DFlash")
            if float(self.args.model.get("l1_loss_alpha", 0)) != 0:
                raise ValueError("Ouro training currently uses CE only")
        if mode == "gate_only":
            if not self.args.model.get("init_checkpoint"):
                raise ValueError("gate_only requires init_checkpoint")
            self.draft_model.requires_grad_(False)
            self.draft_model.exit_gate.requires_grad_(True)

    def save_and_eval_checkpoint(self):
        if getattr(self.draft_model, "exit_gate", None) is not None and self.global_step > 0:
            self.draft_model.config.exit_gate_trained = True
        return super().save_and_eval_checkpoint()

    def _build_draft_model(self, *, target_config, model_args):
        draft_config = build_qwen3_draft_config(
            target_config=target_config,
            model_args=model_args,
        )
        return Qwen3DSparkModel(draft_config)

    # Training step.
    def run_batch(self, batch):
        loops = int(getattr(self.draft_model, "num_loops", 1))
        if self.args.model.get("sample_loop_count", False):
            # All ranks choose the same depth; loss collectives stay aligned.
            loops = random.Random(int(self.args.seed) + self.global_step).randint(1, loops)
        # Make anchors independent of the number of random weights initialized.
        torch.manual_seed(int(self.args.seed) + self.next_micro_step * self.world_size + self.global_rank)
        forward_kwargs = {}
        if hasattr(self.draft_model, "num_loops"):
            forward_kwargs["num_loops"] = loops
        if hasattr(self.draft_model, "loop_boundary_norm"):
            interval = int(self.args.logging.get("diagnostic_steps", 0))
            forward_kwargs["diagnostics"] = interval > 0 and (self.global_step + 1) % interval == 0 and self.next_micro_step % self.gradient_accumulation_steps == 0
        pure_ce = float(self.args.model.l1_loss_alpha) == 0 and float(self.args.model.confidence_head_alpha) == 0
        outputs = self.model(
            input_ids=batch["input_ids"],
            target_hidden_states=batch["target_hidden_states"],
            loss_mask=batch["loss_mask"],
            target_last_hidden_states=None if pure_ce else batch["target_last_hidden_states"],
            **forward_kwargs,
        )
        is_multi_exit = isinstance(outputs, tuple)
        exit_outputs = outputs if is_multi_exit else (outputs,)
        if self.args.model.get("ouro_training_mode", "baseline") != "baseline":
            loss = compute_ouro_loss(exit_outputs, self.draft_model.config)
            add_metric("objective", loss, reduction="dp_mean", tag="train")
            add_metric("num_loops", loops, reduction="last", tag="train")
            return loss
        weights = self.args.model.get("loop_loss_weights", [1.0] * len(exit_outputs))
        active_weights = [float(w) for w in weights[:len(exit_outputs)]]
        if len(active_weights) != len(exit_outputs) or min(active_weights) < 0 or sum(active_weights) <= 0:
            raise ValueError("Active loop loss weights must be nonnegative and have positive sum")
        loss = 0.0
        for index, output in enumerate(exit_outputs):
            exit_loss = compute_dspark_loss(
                outputs=output,
                loss_decay_gamma=self.args.model.loss_decay_gamma,
                ce_loss_alpha=float(self.args.model.ce_loss_alpha),
                l1_loss_alpha=float(self.args.model.l1_loss_alpha),
                confidence_head_alpha=float(self.args.model.confidence_head_alpha),
                metric_tag=f"train/loop{index + 1}",
            )
            if output.side_aux_logits is not None:
                alpha = float(self.draft_model.config.side_aux_loss_alpha)
                aux_loss = compute_dspark_loss(
                    outputs=replace(output, draft_logits=output.side_aux_logits,
                                    side_aux_logits=None, aligned_target_logits=None,
                                    confidence_pred=None, exit_gate_logits=None),
                    loss_decay_gamma=self.args.model.loss_decay_gamma,
                    ce_loss_alpha=1.0, l1_loss_alpha=0.0, confidence_head_alpha=0.0,
                    metric_tag="train/side_grouped",
                )
                exit_loss = exit_loss + alpha * aux_loss
            contribution = active_weights[index] / sum(active_weights) * exit_loss
            loss = loss + contribution
            add_metric("weighted_contribution", contribution, reduction="dp_mean", tag=f"train/loop{index + 1}")
            add_metric("weight", active_weights[index] / sum(active_weights), reduction="last", tag=f"train/loop{index + 1}")
        add_metric("objective", loss, reduction="dp_mean", tag="train")
        add_metric("num_loops", loops, reduction="last", tag="train")
        return loss

    @torch.no_grad()
    def evaluate_cache(self):
        """Fixed disjoint samples and anchors; no generation or throughput claims."""
        import torch.distributed as dist
        import torch.nn.functional as F
        from deepspec.utils.training_logger import log_validation
        from deepspec.utils.metrics import isolated_metrics
        dataset = getattr(self, "validation_dataset", None)
        if dataset is None:
            return
        was_training = self.model.training
        self.model.eval()
        loops = int(getattr(self.draft_model, "num_loops", 1))
        totals = torch.zeros(loops + 1, device=self.device, dtype=torch.float64)
        # Each rank executes the same number of FSDP forwards, including when
        # the held-out count is not divisible by world size. Padding is not scored.
        rounds = (len(dataset) + self.world_size - 1) // self.world_size
        try:
            with torch.random.fork_rng(devices=[self.device.index] if self.device.type == "cuda" else []), isolated_metrics():
                for turn in range(rounds):
                    index = turn * self.world_size + self.global_rank
                    valid = index < len(dataset)
                    index %= len(dataset)
                    torch.manual_seed(int(self.args.seed) + int(dataset.source_ids[index]) + 209759)
                    batch = self.data_collator_cls()([dataset[index]])
                    batch = move_batch_to_device(batch, self.device)
                    outputs = self.model(input_ids=batch["input_ids"],
                        target_hidden_states=batch["target_hidden_states"], loss_mask=batch["loss_mask"],
                        target_last_hidden_states=None)
                    outputs = outputs if isinstance(outputs, tuple) else (outputs,)
                    if valid:
                        for exit_index, output in enumerate(outputs):
                            ce = F.cross_entropy(output.draft_logits.float().reshape(-1, output.draft_logits.shape[-1]),
                                output.target_ids.reshape(-1), reduction="none").reshape_as(output.eval_mask)
                            totals[exit_index] += (ce * output.eval_mask).sum().double()
                        totals[-1] += outputs[0].eval_mask.sum().double()
            dist.all_reduce(totals)
            if totals[-1].item() == 0:
                raise ValueError("Held-out samples have no usable supervision tokens")
            values = {f"validation/loop{i+1}/ce_unweighted": (totals[i] / totals[-1].clamp_min(1)).item() for i in range(loops)}
            values["validation/tokens"] = totals[-1].item()
            log_validation(values, global_step=self.global_step)
        finally:
            self.model.train(was_training)


class Gemma4DSparkTrainer(Qwen3DSparkTrainer):
    def _build_draft_model(self, *, target_config, model_args):
        draft_config = build_gemma4_draft_config(
            target_config=target_config,
            model_args=model_args,
        )
        return Gemma4DSparkModel(draft_config)

