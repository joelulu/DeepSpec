"""Serial Transformers AR reference using the same prompts and decode loop."""
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

from deepspec.eval.base_evaluator import BaseEvaluator, DraftProposal, generate_decoding_sample


class AREvaluator(BaseEvaluator):
    @property
    def max_proposal_tokens(self):
        return 1

    def build_models(self):
        target = AutoModelForCausalLM.from_pretrained(self.args.target_name_or_path,
            dtype=torch.bfloat16, attn_implementation="sdpa").to(self.device).eval()
        return target, target, AutoTokenizer.from_pretrained(self.args.target_name_or_path)

    def generate_one_sample(self, *, input_ids, stop_token_ids):
        def propose(*, output_ids, start, **kwargs):
            return DraftProposal(draft_token_count=0, verify_input_ids=output_ids[:, start:start + 1], draft_probs=None)
        return generate_decoding_sample(target_model=self.target_model, input_ids=input_ids,
            max_new_tokens=int(self.args.max_new_tokens), max_proposal_tokens=1,
            temperature=float(self.args.temperature), stop_token_ids=stop_token_ids,
            init_context=lambda **kwargs: None, propose=propose, update=lambda *args: None,
            profile=bool(self.args.profile), capture_hidden_states=False)

    def evaluate(self):
        for dataset_name, max_samples in self.tasks:
            responses = self.run_dataset(dataset_name=dataset_name, max_samples=max_samples)
            summary = self.allreduce_response_metrics(responses)
            self.record_dataset_metrics(dataset_name=dataset_name, metric_summary=summary)
        self.report_results()
