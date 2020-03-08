import torch
import os
import numpy as np
import string
import random

class Linear_Decay(object):

    def __init__ (self, total_steps, init_value, end_value):
        self.total_steps = total_steps
        self.init_value = init_value
        self.end_value = end_value

    def _get_value(self, step):
        frac = min(float(step) / self.total_steps, 1.0)
        return self.init_value + frac * (self.end_value-self.init_value)

def get_conditional_argmax(Q, action_idxes, action_space):
    size = []
    size.append(Q.shape[0])
    size += action_space
    _Q = Q.view(size)
    a_idxes = []
    for q_i in range(_Q.shape[0]):
        N = _Q[q_i].shape
        masks = []
        for i, idx in enumerate(action_idxes[q_i]):
            if idx == -1:
                m = torch.ones(N[i]).byte()
            else:
                m = torch.zeros(N[i]).byte()
                m[idx] = 1
            masks.append(m)
        letters = string.ascii_letters[:_Q[q_i].ndimension()]
        rule = ','.join(letters) + '->' + letters
        mask = torch.einsum(rule, *masks)
        Qmasked = _Q[q_i].where(mask, torch.tensor(-float('inf')))
        a_idxes.append(Qmasked.argmax())
    return torch.tensor(a_idxes).view(-1,1)

def get_conditional_action(actions, v):
    condi_a = actions.clone()
    condi_a[v] = -1
    return condi_a 

def save_check_point(agents, cen_controller, cur_step, n_epi, cur_hysteretic, cur_eps, save_dir, mem_cen, mem_dec, run_id, test_perform, max_save=2):

    PATH = "./performance/" + save_dir + "/check_point/" + str(run_id) + "_cen_controller_" + "{}.tar"

    for n in list(range(max_save-1, 0, -1)):
        os.system('cp -rf ' + PATH.format(n) + ' ' + PATH.format(n+1) )
    PATH = PATH.format(1)

    torch.save({
                'cur_step': cur_step,
                'n_epi': n_epi,
                'policy_net_state_dict':cen_controller.policy_net.state_dict(),
                'target_net_state_dict':cen_controller.target_net.state_dict(),
                'optimizer_state_dict':cen_controller.optimizer.state_dict(),
                'cur_hysteretic':cur_hysteretic,
                'cur_eps':cur_eps,
                'TEST_PERFORM': test_perform,
                'mem_cen_buf':mem_cen.buf,
                'random_state':random.getstate(),
                'np_random_state': np.random.get_state(),
                'torch_random_state': torch.random.get_rng_state()
                }, PATH)

    for idx, agent in enumerate(agents):
        PATH = "./performance/" + save_dir + "/check_point/" + str(run_id) + "_agent_" + str(idx) + "{}.tar"

        for n in list(range(max_save-1, 0, -1)):
            os.system('cp -rf ' + PATH.format(n) + ' ' + PATH.format(n+1) )
        PATH = PATH.format(1)

        torch.save({
                    'cur_step': cur_step,
                    'n_epi': n_epi,
                    'policy_net_state_dict':agent.policy_net.state_dict(),
                    'target_net_state_dict':agent.target_net.state_dict(),
                    'optimizer_state_dict':agent.optimizer.state_dict(),
                    'cur_hysteretic':cur_hysteretic,
                    'cur_eps':cur_eps,
                    'TEST_PERFORM': test_perform,
                    'mem_dec_buf':mem_dec.buf,
                    'random_state':random.getstate(),
                    'np_random_state': np.random.get_state(),
                    'torch_random_state': torch.random.get_rng_state()
                    }, PATH)




