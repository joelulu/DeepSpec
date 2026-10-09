import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from transformers import Qwen3ForCausalLM

from deepspec.data.cache_selection import select_cache_ids
from deepspec.data.target_cache_dataset import (CacheDataset, LocalTargetCacheWriter,
    build_target_cache_manifest, write_target_cache_manifest)
from deepspec.modeling.dspark.context_cache import ContextKVCache
from deepspec.modeling.dspark.qwen3.modeling import Qwen3DSparkModel
from deepspec.trainer.run_identity import check_run_identity
from deepspec.trainer.base_trainer import _compute_training_schedule
from deepspec.utils.config import load_config, parse_opts_to_config
from test_loop_pilot import load_script
from test_loop_qwen3 import config, inputs, dense_mask


def make_cache(path, count=100):
    path.mkdir()
    writer = LocalTargetCacheWriter(rank_dir=str(path), max_shard_bytes=2048)
    for i in range(count):
        writer.write_sample(sample_id=i, input_ids=torch.tensor([i, i+1]),
            attention_mask=torch.ones(2), loss_mask=torch.ones(2),
            target_hidden_states=torch.full((2, 4), i, dtype=torch.bfloat16),
            target_last_hidden_states=torch.full((2, 4), i+1, dtype=torch.bfloat16))
    writer.close()
    (path / 'samples.local.idx').rename(path / 'samples.idx')
    manifest = build_target_cache_manifest(num_samples=count,
        shards=[dict(shard_id=i, file_name=name) for i, name in enumerate(writer.local_shard_files)],
        target_layer_ids=[0], hidden_size=4, extra_fields=dict(target_model_name_or_path='target'))
    write_target_cache_manifest(output_dir=str(path), manifest=manifest)


def test_logical_selection_reads_correct_tensors_without_writing_cache(tmp_path):
    source = tmp_path / 'cache'; make_cache(source)
    before = {p.name: p.read_bytes() for p in source.iterdir()}
    view = CacheDataset(str(source), data_percent='12.5', subset_seed=42, holdout_samples=4)
    repeated = CacheDataset(str(source), data_percent='12.5', subset_seed=42, holdout_samples=4)
    held = CacheDataset(str(source), data_percent='12.5', subset_seed=42, holdout_samples=4, split='validation')
    full = CacheDataset(str(source))
    assert len(view) == 12 and len(held) == 4
    np.testing.assert_array_equal(view.source_ids, repeated.source_ids)
    assert not set(view.source_ids) & set(held.source_ids)
    assert view.selection_id == repeated.selection_id != held.selection_id
    for index, original_id in enumerate(view.source_ids):
        for key, value in full[int(original_id)].items():
            torch.testing.assert_close(view[index][key], value)
    for dataset in (view, repeated, held, full):
        dataset.close()
    assert before == {p.name: p.read_bytes() for p in source.iterdir()}
    assert full.source_ids is None
    train_all = select_cache_ids(100, 100, holdout_samples=4)
    assert len(train_all) == 96 and not set(train_all) & set(held.source_ids)


@pytest.mark.parametrize('percent', ['0', '-1', '100.1', 'nan', 'inf', '0.000001'])
def test_invalid_or_empty_percent_rejected(percent):
    with pytest.raises(ValueError):
        select_cache_ids(100, percent)


def test_flex_launch_epoch_schedule_and_real_config(tmp_path):
    # Match the user's cache magnitude without manufacturing tensor data.
    (tmp_path/'manifest.json').write_text(json.dumps(dict(num_samples=1339767,
        target_model_name_or_path='target', target_layer_ids=[1,9,17,25,33])))
    env = dict(SOURCE_CACHE=str(tmp_path), DATA_PERCENT='3', HOLDOUT_SAMPLES='32', SEED='42',
        EPOCHS='3', LOCAL_BATCH='4', GLOBAL_BATCH='256', LOOP_TRANSFER='rmsnorm', LOOP_LOSS='final',
        RUN_ROOT=str(tmp_path/'runs'), ANCHORS='32', LR='6e-4', SHARDING='no_shard', LOG_EVERY='5', DIAGNOSTIC_EVERY='20')
    plan, command, opts = load_script('launch_flex').build_launch(1, 5, env)
    assert plan['train_samples'] == 40192
    assert plan['steps_per_epoch'] == 157 and plan['max_train_steps'] == 471
    assert plan['loss_weights'] == [0,0,0,0,1]
    root = Path(__file__).resolve().parents[1]
    cfg = parse_opts_to_config([key+'='+json.dumps(value) for key,value in opts.items()],
        load_config(root/'config/loop/dflash_flex_qwen3_4b.py'))
    assert cfg.model.num_draft_layers == 1 and cfg.model.num_loops == 5
    assert cfg.model.loop_boundary_norm and cfg.train.max_train_steps is None
    schedule = _compute_training_schedule(world_size=8, dataset_size=plan['train_samples'],
        local_batch_size=4, global_batch_size=256, num_train_epochs=3)
    assert schedule[0] == 8 and schedule[4:6] == (157,471)
    env['LOOP_LOSS'] = 'uniform'
    assert load_script('launch_flex').build_launch(3, 4, env)[0]['loss_weights'] == [1]*4


def test_resume_rejects_changed_data_or_settings(tmp_path):
    check_run_identity(tmp_path, {'selection':'abc','loops':3}, write=True)
    check_run_identity(tmp_path, {'selection':'abc','loops':3}, resuming=True)
    with pytest.raises(ValueError, match='differ'):
        check_run_identity(tmp_path, {'selection':'xyz','loops':3}, resuming=True)
    with pytest.raises(ValueError, match='Missing'):
        check_run_identity(tmp_path/'other', {}, resuming=True)


@pytest.mark.parametrize('layers,loops,norm', [(15,1,False),(1,5,False),(5,3,True),(3,4,True)])
def test_all_physical_layers_receive_gradient_from_final_exit(monkeypatch, layers, loops, norm):
    import deepspec.modeling.dspark.qwen3.modeling as modeling
    monkeypatch.setattr(modeling, 'create_dspark_attention_mask', dense_mask)
    torch.manual_seed(42); torch.set_num_threads(1)
    cfg = config(loops); cfg.num_hidden_layers = layers; cfg.layer_types = ['full_attention']*layers
    cfg.loop_boundary_norm = norm; cfg.loop_loss_weights = [0.]*(loops-1)+[1.]
    model = Qwen3DSparkModel(cfg); model.set_embedding_head_trainable(False)
    outputs = model(input_ids=torch.randint(0,60,(1,12)), loss_mask=torch.ones(1,12),
        target_hidden_states=torch.randn(1,12,64), diagnostics=True)
    outputs = outputs if isinstance(outputs, tuple) else (outputs,)
    assert all(not output.draft_logits.requires_grad for output in outputs[:-1])
    final = outputs[-1]
    torch.nn.functional.cross_entropy(final.draft_logits.flatten(0,2), final.target_ids.flatten()).backward()
    assert len({id(layer.self_attn.q_proj.weight) for layer in model.layers}) == layers
    for layer in model.layers:
        for name in ('q_proj','k_proj','v_proj','o_proj'):
            grad = getattr(layer.self_attn, name).weight.grad
            assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
        assert layer.mlp.down_proj.weight.grad.abs().sum() > 0
    assert model.fc.weight.grad.abs().sum() > 0
    assert model.embed_tokens.weight.grad is None and model.lm_head.weight.grad is None


def test_boundary_norm_feeds_normalized_state_and_cache_remains_exact(tmp_path):
    torch.manual_seed(42); torch.set_num_threads(1)
    cfg = config(5); cfg.num_hidden_layers=1; cfg.layer_types=['full_attention']; cfg.loop_boundary_norm=True
    model = Qwen3DSparkModel(cfg).eval(); batch = inputs()
    layer_inputs=[]
    handle = model.layers[0].register_forward_pre_hook(lambda module, args, kwargs: layer_inputs.append(kwargs['hidden_states'].detach().clone()), with_kwargs=True)
    with torch.no_grad():
        exits = model._forward_backbone(**batch, return_all_loop_hidden=True)
    handle.remove()
    for index in range(1,5):
        torch.testing.assert_close(layer_inputs[index], exits[index-1])
    raw_cfg=copy.deepcopy(cfg); raw_cfg.loop_boundary_norm=False
    raw=Qwen3DSparkModel(raw_cfg).eval(); raw.load_state_dict(model.state_dict())
    torch.testing.assert_close(model._forward_backbone(**batch,num_loops=1), raw._forward_backbone(**batch,num_loops=1))
    cache=ContextKVCache()
    with torch.no_grad():
        model._forward_backbone(**batch,past_key_values=cache)
        new=torch.randn(1,2,64); noise=torch.randn(1,3,32)
        actual=model._forward_backbone(target_hidden_states=new,noise_embedding=noise,
            position_ids=torch.arange(6,11)[None],past_key_values=cache)
        expected=model._forward_backbone(target_hidden_states=torch.cat((batch['target_hidden_states'],new),dim=1),
            noise_embedding=noise,position_ids=torch.arange(11)[None])
    torch.testing.assert_close(actual,expected,atol=3e-6,rtol=3e-5)
    assert cache.layers[0][0].shape[-2]==8
    model.save_pretrained(tmp_path)
    restored=Qwen3DSparkModel.from_pretrained(tmp_path,attn_implementation='sdpa')
    assert restored.loop_boundary_norm and restored.num_loops==5
    torch.testing.assert_close(restored._forward_backbone(**batch), model._forward_backbone(**batch))


@pytest.mark.parametrize('count',[1,4,5])
def test_prefix_summary_supports_arbitrary_depth_and_matches_bruteforce(count):
    import itertools
    script=load_script('summarize_prefix'); rounds=[]
    for shift in (0,1):
        exits=[dict(num_loops=i+1,round_wall_ms=10+i*2,progress_tokens=2+i+shift,
            accepted_draft_tokens=1+i+shift,progress_per_second_proxy=(2+i+shift)*1000/(10+i*2)) for i in range(count)]
        rounds.append(dict(exits=exits,valid_for_utility=True,best_loop_by_proxy=count))
    brute=max(sum(rounds[r]['exits'][depth]['progress_tokens'] for r,depth in enumerate(choice))*1000 /
        sum(rounds[r]['exits'][depth]['round_wall_ms'] for r,depth in enumerate(choice))
        for choice in itertools.product(range(count),repeat=2))
    result=script.summarize(rounds)
    assert len(result['fixed_depths'])==count and len(result['acceptance_transitions'])==count-1
    assert result['aggregate_oracle_progress_per_second_proxy']==pytest.approx(brute)


def test_real_trainer_final_loss_and_validation_are_deterministic_and_isolated(monkeypatch, tmp_path):
    import torch.distributed as dist
    import deepspec.modeling.dspark.qwen3.modeling as modeling
    from deepspec.trainer.dspark_trainer import Qwen3DSparkTrainer
    from deepspec.utils.config import to_config_node
    from deepspec.utils import metrics, training_logger
    monkeypatch.setattr(modeling, 'create_dspark_attention_mask', dense_mask)
    monkeypatch.setattr(dist, 'get_world_size', lambda: 1)
    monkeypatch.setattr(dist, 'all_reduce', lambda *args, **kwargs: None)
    torch.manual_seed(42); torch.set_num_threads(1); metrics.reset()
    cfg=config(3); cfg.loop_loss_weights=[0,0,1]; cfg.loop_boundary_norm=True
    model=Qwen3DSparkModel(cfg); model.set_embedding_head_trainable(False)
    trainer=Qwen3DSparkTrainer.__new__(Qwen3DSparkTrainer)
    trainer.model=trainer.draft_model=model
    trainer.args=to_config_node(dict(seed=42,model=dict(loop_loss_weights=[0,0,1],
        l1_loss_alpha=0.,ce_loss_alpha=1.,confidence_head_alpha=0.,loss_decay_gamma=4.),
        logging=dict(diagnostic_steps=1)))
    trainer.next_micro_step=0; trainer.gradient_accumulation_steps=1
    trainer.global_rank=0; trainer.world_size=1; trainer.device=torch.device('cpu')
    batch=dict(input_ids=torch.randint(0,60,(1,12)),loss_mask=torch.ones(1,12),
        target_hidden_states=torch.randn(1,12,64),target_last_hidden_states=torch.randn(1,12,32))
    loss=trainer.run_batch(batch)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.layers[0].self_attn.q_proj.weight.grad.abs().sum()>0
    assert metrics._metrics['train/loop1/weighted_contribution']['values'][0].item()==0
    assert metrics._metrics['train/loop2/weighted_contribution']['values'][0].item()==0
    assert metrics._metrics['train/objective']['values'][0].item()==pytest.approx(loss.item())
    assert all(f'train/loop{i}/ce_unweighted' in metrics._metrics for i in (1,2,3))
    # Exercise the writer -> reader -> collator path: real caches store int32
    # token IDs and bfloat16 features, unlike torch.randint's int64 default.
    cache_path=tmp_path/'validation_cache'; cache_path.mkdir()
    writer=LocalTargetCacheWriter(rank_dir=str(cache_path), max_shard_bytes=65536)
    for index in range(10):
        writer.write_sample(sample_id=index, attention_mask=torch.ones(12),
            **{key:value[0] for key,value in batch.items()})
    writer.close()
    (cache_path/'samples.local.idx').rename(cache_path/'samples.idx')
    manifest=build_target_cache_manifest(num_samples=10,
        shards=[dict(shard_id=i,file_name=name) for i,name in enumerate(writer.local_shard_files)],
        target_layer_ids=[0,2],hidden_size=32)
    write_target_cache_manifest(output_dir=str(cache_path),manifest=manifest)
    held=CacheDataset(str(cache_path),data_percent=100,holdout_samples=3,split='validation')
    assert held[0]['input_ids'].dtype==torch.int32
    model.to(torch.bfloat16)
    trainer.validation_dataset=held
    previous_schema=metrics._schema(); collected=[]
    monkeypatch.setattr(training_logger,'log_validation',lambda values,**kwargs:collected.append(values))
    rng_before=torch.get_rng_state().clone()
    trainer.evaluate_cache(); trainer.evaluate_cache()
    assert collected[0]==collected[1]
    class LongHeld:
        source_ids=held.source_ids
        def __len__(self): return len(held)
        def __getitem__(self,index):
            sample=held[index]
            return {**sample,'input_ids':sample['input_ids'].long()}
    trainer.validation_dataset=LongHeld()
    trainer.evaluate_cache()
    assert collected[0]==collected[2]
    assert held[0]['input_ids'].dtype==torch.int32
    held.close()
    assert collected[0]['validation/tokens']>0
    assert all(collected[0][f'validation/loop{i}/ce_unweighted']>0 for i in (1,2,3))
    assert torch.equal(rng_before,torch.get_rng_state())
    assert metrics._schema()==previous_schema and model.training
    metrics.reset()
