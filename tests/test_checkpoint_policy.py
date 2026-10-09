import json
from types import SimpleNamespace

import pytest
import torch

from deepspec.modeling.dspark.qwen3.modeling import Qwen3DSparkModel
from deepspec.trainer import ckpt_manager
from deepspec.trainer.base_trainer import BaseTrainer
from deepspec.utils.config import to_config_node
from test_loop_qwen3 import config


def test_final_only_keeps_epoch_validation_without_intermediate_saves(monkeypatch):
    import deepspec.trainer.base_trainer as base
    monkeypatch.setattr(base.dist, 'barrier', lambda: None)
    monkeypatch.setattr(base, 'is_global_main_process', lambda: False)
    validations=[]; saves=[]
    monkeypatch.setattr(base, 'save_checkpoint', lambda **kwargs: saves.append(kwargs) or 'final')
    trainer=BaseTrainer.__new__(BaseTrainer)
    trainer.args=to_config_node(dict(logging=dict(save_only_final=True,save_training_state=False),
        data=dict(holdout_samples=32),train=dict(local_batch_size=4)))
    trainer.gradient_accumulation_steps=1; trainer.max_train_steps=3
    trainer.model=trainer.draft_model=trainer.optimizer=object()
    trainer.checkpoint_dir_root='unused'; trainer.global_rank=0; trainer.world_size=1
    trainer.evaluate_cache=lambda: validations.append(trainer.global_step)
    for step in (1,2,3,3):
        trainer.next_micro_step=step
        trainer.save_and_eval_checkpoint()
    assert validations==[1,2,3]
    assert len(saves)==1 and saves[0]['next_micro_step']==3
    assert saves[0]['save_training_state'] is False
    assert saves[0]['training_complete'] is True


@pytest.mark.parametrize('with_state', [False,True])
def test_checkpoint_file_contents_and_completed_model_reload(monkeypatch,tmp_path,with_state):
    torch.set_num_threads(1)
    model=Qwen3DSparkModel(config(1))
    model.set_embedding_head_trainable(False)
    monkeypatch.setattr(ckpt_manager.dist,'barrier',lambda:None)
    monkeypatch.setattr(ckpt_manager.dist,'get_rank',lambda:0)
    monkeypatch.setattr(ckpt_manager,'is_global_main_process',lambda:True)
    monkeypatch.setattr(ckpt_manager,'_full_model_state_dict',lambda wrapped:model.state_dict())
    serializations=[]
    monkeypatch.setattr(ckpt_manager,'_serialize_training_state',
        lambda **kwargs: serializations.append(kwargs) or {'sentinel':1})
    origin=tmp_path/'config.py'; origin.write_text('model = {}\n')
    train_config=SimpleNamespace(_origin_config_path=str(origin),_origin_opts=[])
    root=tmp_path/'checkpoints'
    saved=ckpt_manager.save_checkpoint(model=model,draft_model=model,optimizer=object(),
        checkpoint_dir_root=str(root),train_config=train_config,next_micro_step=6,
        gradient_accumulation_steps=2,global_rank=0,world_size=1,local_batch_size=4,
        save_training_state=with_state,training_complete=True)
    assert ckpt_manager.discover_latest_checkpoint(str(root))==saved
    assert {p.name for p in root.iterdir()}=={'step_3','step_latest'}
    assert bool(serializations)==with_state
    assert (root/'step_3'/'training_state.rank0.pt').exists()==with_state
    if not with_state:
        assert not list(root.rglob('*.pt'))
        metadata=json.loads((root/'step_3'/'checkpoint_meta.json').read_text())
        assert metadata['training_complete'] is True and metadata['next_micro_step']==6
        resumed=ckpt_manager.load_resume_draft_model(resume_checkpoint_dir=saved,
            draft_model=model,device=torch.device('cpu'),precision_dtype=torch.float32,global_rank=0)
        for name,tensor in model.state_dict().items():
            torch.testing.assert_close(resumed.state_dict()[name],tensor)
        # An object without load_state_dict proves no optimizer restore occurs.
        state=ckpt_manager.load_training_state(resume_checkpoint_dir=saved,optimizer=object(),
            global_rank=0,world_size=1,local_batch_size=4,gradient_accumulation_steps=2,
            micro_batches_per_epoch=2)
        assert state.next_micro_step==6
        trainer=BaseTrainer.__new__(BaseTrainer)
        trainer.model=resumed; trainer.next_micro_step=state.next_micro_step
        trainer.gradient_accumulation_steps=2; trainer.max_train_steps=3
        trainer.train()  # Completed runs return before constructing a CUDA loader.


def test_incomplete_weights_without_training_state_cannot_auto_resume(tmp_path):
    (tmp_path/'checkpoint_meta.json').write_text(json.dumps(dict(
        training_complete=False,save_training_state=False,next_micro_step=2)))
    with pytest.raises(ValueError,match='No resumable training state'):
        ckpt_manager.load_training_state(resume_checkpoint_dir=str(tmp_path),optimizer=object(),
            global_rank=0,world_size=1,local_batch_size=4,gradient_accumulation_steps=2,
            micro_batches_per_epoch=2)
