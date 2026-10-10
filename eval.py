from __future__ import annotations
import argparse
import json
from pathlib import Path
import torch.distributed as dist
import torch
from transformers import AutoConfig
from deepspec.eval.dspark import Gemma4DSparkEvaluator, Qwen3DSparkEvaluator
from deepspec.eval.eagle3 import Gemma4Eagle3Evaluator, Qwen3Eagle3Evaluator
from deepspec.utils import CustomJSONEncoder
from deepspec.eval.ar_evaluator import AREvaluator
from deepspec.eval.dspark.loopcd import LoopCDConfig

EVALUATORS = {
    "Qwen3DSparkModel": Qwen3DSparkEvaluator,
    "Gemma4DSparkModel": Gemma4DSparkEvaluator,
    "Qwen3Eagle3Model": Qwen3Eagle3Evaluator,
    "Gemma4Eagle3Model": Gemma4Eagle3Evaluator,
    "Eagle3DraftModel": Qwen3Eagle3Evaluator,
}

TASKS = [
    ("gsm8k", 500),
    ("math500", 500),
    ("aime25",30),
    ("humaneval", 164),
    ("mbpp", 256),
    ("livecodebench", 500),
    ("mt-bench", 80),
    ("alpaca", 500),
    ("arena-hard-v2", 500),
]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_name_or_path", type=str, required=True)
    parser.add_argument("--draft_name_or_path",type=str,default=None)
    parser.add_argument("--autoregressive", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.0,
        help=("Confidence-head early-stop threshold. Confidence calibration metrics are collected only when this is 0.0."),
    )
    parser.add_argument("--tensorboard-dir", type=str, default=None)
    parser.add_argument("--step", type=int, default=None,help=("step for tensorboard logging"),)
    parser.add_argument("--seed", type=int, default=980406)
    parser.add_argument("--tasks", type=str, default=None, help="Comma-separated task names")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--num-loops", type=int, default=None)
    parser.add_argument("--loopcd", action="store_true", help="Contrast final and early loop logits (frozen Qwen3 DFlash only)")
    parser.add_argument("--loopcd-early-loop", type=int, default=1, help="One-based early exit index")
    parser.add_argument("--loopcd-lambda", type=float, default=0.2)
    parser.add_argument("--loopcd-alpha", type=float, default=0.1, help="Final-loop relative plausibility threshold")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--warmup-samples", type=int, default=1)
    parser.add_argument("--output-json", type=str, default=None)
    args = parser.parse_args()
    if args.loopcd:
        if args.autoregressive:
            parser.error("--loopcd requires a looped Qwen3 DFlash checkpoint")
        try:
            loopcd = LoopCDConfig(args.loopcd_early_loop, args.loopcd_lambda, args.loopcd_alpha)
        except ValueError as exc:
            parser.error(str(exc))
        if args.num_loops is not None and loopcd.early_loop >= args.num_loops:
            parser.error("--loopcd-early-loop must be less than --num-loops")
    if not args.autoregressive and args.draft_name_or_path is None:
        parser.error("--draft_name_or_path is required unless --autoregressive is set")
    if args.autoregressive:
        args.draft_name_or_path = "AR"
    selected = args.tasks
    args.tasks = list(TASKS)
    if args.max_samples is not None and args.max_samples < 1:
        parser.error("--max-samples must be positive")
    if args.warmup_samples < 0 or args.max_new_tokens < 1:
        parser.error("warmup must be nonnegative and max-new-tokens positive")
    if selected is not None:
        names = selected.split(",")
        defaults = dict(TASKS)
        if any(name not in defaults for name in names):
            parser.error("unknown task in --tasks")
        args.tasks = [(name, defaults[name]) for name in names]
    if args.max_samples is not None:
        args.tasks = [(name, args.max_samples) for name, _ in args.tasks]
    return args


def main(local_rank: int, args):
    if local_rank == 0:
        print(json.dumps(args, indent=4, cls=CustomJSONEncoder), flush=True)
    if args.autoregressive:
        evaluator_cls = AREvaluator
    else:
        draft_config = AutoConfig.from_pretrained(args.draft_name_or_path)
        evaluator_cls = EVALUATORS[draft_config.architectures[0]]
        if args.loopcd and evaluator_cls is not Qwen3DSparkEvaluator:
            raise ValueError("--loopcd currently supports Qwen3 DFlash only")
    evaluator = evaluator_cls(local_rank, args)
    evaluator.evaluate()
    if args.output_json and dist.get_rank() == 0:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(
            args=vars(args),
            num_draft_layers=0 if args.autoregressive else evaluator.draft_model.config.num_hidden_layers,
            num_loops=0 if args.autoregressive else getattr(evaluator.draft_model, "eval_num_loops", getattr(evaluator.draft_model, "num_loops", 1)),
            loop_boundary_norm=False if args.autoregressive else getattr(evaluator.draft_model.config, "loop_boundary_norm", False),
            loop_loss_weights=[] if args.autoregressive else getattr(evaluator.draft_model.config, "loop_loss_weights", [1.0]),
            device_name=torch.cuda.get_device_name(evaluator.device),
            metrics=evaluator.metrics_rows,
            timing_scope="serial requests per GPU; not a serving-throughput benchmark",
        )
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    evaluator.clean_up()

if __name__ == "__main__":
    args = parse_args()
    torch.multiprocessing.spawn(
        main,
        args=(args,),
        nprocs=torch.cuda.device_count(),
    )

