"""Real OPSD backward pass with ragged responses and a zero-loss padding row."""
from types import SimpleNamespace
from unittest import TestCase
import torch
from tensordict import TensorDict
from integrations.simpletir_qwen35.opsd_loss import sdar_only_loss

class TestOPSDBackward(TestCase):
    def test_gated_loss_has_gradient_only_on_real_response(self):
        def nested(rows):
            return torch.nested.as_nested_tensor([torch.tensor(r) for r in rows],layout=torch.jagged)
        data=TensorDict({
            'prompts':nested([[10],[10]]), 'responses':nested([[20,21],[0]]),
            'response_mask':nested([[1.,1.],[0.]]),
            'teacher_response_log_probs':nested([[-.5,-.5],[0.]]),
            'ref_log_prob':nested([[-1.,-1.],[0.]]),
        }, batch_size=2)
        for key,value in {'dp_size':1,'batch_num_tokens':None,'global_batch_size':None}.items():
            data.set_non_tensor(key,value)
        config=SimpleNamespace(global_batch_info={},loss_scale_factor=None,loss_agg_mode='token-mean',
            entropy_coeff=0.,use_kl_loss=True,kl_loss_type='low_var_kl',kl_loss_coef=.01)
        values=torch.tensor([-1.,-1.,-9.,-2.,-9.],requires_grad=True)
        loss,metrics=sdar_only_loss(config,{'log_probs':values},data)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(values.grad).all())
        self.assertGreater(values.grad[:2].abs().sum().item(),0.)
        self.assertEqual(values.grad[2:].abs().sum().item(),0.)
        self.assertIn('opsd/sdar_loss',metrics)
