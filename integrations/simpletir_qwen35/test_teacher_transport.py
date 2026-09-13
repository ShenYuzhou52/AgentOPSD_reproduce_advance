"""Exercise teacher-field transport and variable-length credit on CPU.

中文说明：CPU 上端到端验证 teacher 字段经 TransferQueue 抵达 actor 的
传输链路（字段元数据注册）与变长轨迹上的信用重塑——对应预跑时修过的
"写入了字段但 actor 看不见"的问题。
"""
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch
import torch
from tensordict import TensorDict
from agentopsd.credit import AgentOPSDConfig
from integrations.simpletir_qwen35.trainer import SimpleTIRTrainer, PPOTrainerSync

class Batch(SimpleNamespace):
    def __len__(self):
        return len(self.keys)

class TestTeacherTransport(TestCase):
    def run_method(self, method):
        def nested(rows):
            return torch.nested.as_nested_tensor([torch.tensor(r, dtype=torch.float32) for r in rows], layout=torch.jagged)
        storage = TensorDict({
            'response_mask': nested([[1,1],[1],[1,1],[0]]),
            'old_log_probs': nested([[-1,-1],[-1],[-1,-1],[0]]),
            'advantages': nested([[1,1],[1],[-1,-1],[0]]),
            'returns': nested([[1,1],[1],[-1,-1],[0]]),
        }, batch_size=4)
        original = Batch(keys=['g_a_0','g_a_1','g_b_0','padding_x_0'],
            tags=[{},{},{},{'is_padding':True}], partition_id='test', fields=list(storage.keys()))
        extras = [
            {'turn_step':0,'teacher_response_log_probs':[-.5,-.6],'reward_extra_info':{'answer_accuracy':1.}},
            {'turn_step':1,'teacher_response_log_probs':[-.3],'reward_extra_info':{'answer_accuracy':1.}},
            {'turn_step':0,'teacher_response_log_probs':[-1.3,-1.1],'reward_extra_info':{'answer_accuracy':0.}}, {},
        ]
        trainer=object.__new__(SimpleTIRTrainer)
        trainer.method=method
        trainer._agentopsd_cfg=AgentOPSDConfig()
        trainer._agentopsd_cfg.enabled=method=='agentopsd'
        trainer._get_extra_fields=lambda batch: extras
        def get(**kw):
            return storage.select(*kw['select_fields'])
        def put(**kw):
            storage.update(kw['fields'])
            return Batch(keys=original.keys,tags=original.tags,partition_id='test',fields=list(storage.keys()))
        metrics={}
        with patch.object(PPOTrainerSync,'_compute_advantage',return_value=original), \
             patch('integrations.simpletir_qwen35.trainer.tq.kv_batch_get',side_effect=get), \
             patch('integrations.simpletir_qwen35.trainer.tq.kv_batch_put',side_effect=put):
            result=trainer._compute_advantage(original,metrics)
        self.assertIn('teacher_response_log_probs',result.fields)
        self.assertNotIn('teacher_response_log_probs',original.fields)
        self.assertEqual(metrics['simpletir/teacher_forward_applied'],1.)
        self.assertTrue(storage['advantages'].is_nested)
        self.assertEqual(storage['advantages'].offsets().diff().tolist(),[2,1,2,1])
        self.assertEqual(storage.to_padded_tensor()['advantages'][-1].abs().sum().item(),0.)
        self.assertTrue(torch.isfinite(storage['advantages'].values()).all())
        return metrics

    def test_opsd_metadata_includes_teacher_field(self):
        self.run_method('opsd_author_code')

    def test_agentopsd_ragged_turns_and_padding(self):
        self.assertEqual(self.run_method('agentopsd')['agentopsd/reshape_applied'],1.)
