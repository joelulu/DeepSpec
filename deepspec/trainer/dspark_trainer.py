import random
import torch

from deepspec.data import CacheCollator
from deepspec.utils.metrics import add_metric
from deepspec.modeling.dspark.gemma4 import Gemma4DSparkModel
from deepspec.modeling.dspark.gemma4.config import (
    build_draft_config as build_gemma4_draft_config,
)
from deepspec.modeling.dspark.loss import compute_dspark_loss
from deepspec.modeling.dspark.qwen3 import Qwen3DSparkModel
from deepspec.modeling.dspark.qwen3.config import (
    build_draft_config as build_qwen3_draft_config,
)
from deepspec.trainer.base_trainer import BaseTrainer


class Qwen3DSparkTrainer(BaseTrainer):
    data_collator_cls = CacheCollator

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
        outputs = self.model(
            input_ids=batch["input_ids"],
            target_hidden_states=batch["target_hidden_states"],
            loss_mask=batch["loss_mask"],
            target_last_hidden_states=batch["target_last_hidden_states"],
            **forward_kwargs,
        )
        is_multi_exit = isinstance(outputs, tuple)
        exit_outputs = outputs if is_multi_exit else (outputs,)
        weights = self.args.model.get("loop_loss_weights", [1.0] * len(exit_outputs))
        active_weights = [float(w) for w in weights[:len(exit_outputs)]]
        loss = 0.0
        for index, output in enumerate(exit_outputs):
            exit_loss = compute_dspark_loss(
                outputs=output,
                loss_decay_gamma=self.args.model.loss_decay_gamma,
                ce_loss_alpha=float(self.args.model.ce_loss_alpha),
                l1_loss_alpha=float(self.args.model.l1_loss_alpha),
                confidence_head_alpha=float(self.args.model.confidence_head_alpha),
                metric_tag=f"train/loop{index + 1}" if is_multi_exit else "train",
            )
            loss = loss + active_weights[index] / sum(active_weights) * exit_loss
        if is_multi_exit:
            add_metric("weighted_loss", loss, reduction="mean", tag="train")
            add_metric("num_loops", loops, reduction="mean", tag="train")
        return loss


class Gemma4DSparkTrainer(Qwen3DSparkTrainer):
    def _build_draft_model(self, *, target_config, model_args):
        draft_config = build_gemma4_draft_config(
            target_config=target_config,
            model_args=model_args,
        )
        return Gemma4DSparkModel(draft_config)

