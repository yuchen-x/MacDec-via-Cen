import argparse
import numpy as np
import torch
import os
import sys
sys.path.append("..")
import time
import IPython
import logging

from rlmamr.my_env.osd_ma_single_room import ObjSearchDelivery_v4 as OSD_S_4

from rlmamr.MA_hddrqn.utils.Agent import Agent
from IPython.core.debugger import set_trace

ENVIRONMENTS = {
        'OSD_S_4':OSD_S_4
        }

def get_actions_and_h_states(env, agent, joint_obs, h_states_in, last_action, last_valid, log=False):
    with torch.no_grad():
        actions = []
        h_states_out = []
        for idx,agent in enumerate(agent):
            if last_valid[agent.idx]:
                jobs = joint_obs[agent.idx].view(1,1,env.obs_size[agent.idx])
                one_h_state = h_states_in[agent.idx]
                Q, h = agent.policy_net(jobs, one_h_state)
                a = Q.squeeze(1).max(1)[1].item()
                if log:
                    update_log(agent.idx, jobs, a)
                actions.append(a)
                h_states_out.append(h)
            else:
                actions.append(env.agents[agent.idx].cur_action.idx)
                h_states_out.append(h_states_in[agent.idx])

    return actions, h_states_out

def get_init_inputs(env,n_agent):
    return [torch.from_numpy(i).float() for i in env.reset(True)], [None]*n_agent

def test(env_name, env_terminate_step, n_agent, n_episode, p_id):
    ENV = ENVIRONMENTS[env_name]
    env = ENV(n_obj=3, 
              fetch_pass_obj_tc=4, 
              fetch_look_for_obj_tc=6, 
              human_speed_per_step=[[18]], 
              TB_move_speed=0.6, 
              delay_delivery_penalty=False,
              terminate_step=150)

    env.reset(True)

    agents = []

    for i in range(n_agent):
        agent = Agent()
        agent.idx = i
        agent.policy_net = torch.load("./policy_nns/OSD_S/" + str(p_id) + "_agent_" + str(i) + ".pt")
        agent.policy_net.eval()
        agents.append(agent)

    R = 0
    discount = 1.0

    for e in range(n_episode):
        t = 0
        step=0
        last_obs, h_states = get_init_inputs(env, n_agent)
        last_valid = [torch.tensor([[1]]).byte()] * n_agent
        last_action = [torch.tensor([[-1]])] * n_agent
        while not t:
            a, h_states = get_actions_and_h_states(env, agents, last_obs, h_states, last_action, last_valid)
            #set_trace()
            a, last_obs, r, t, v = env.step(a,True)
            time.sleep(0.2)
            last_obs = [torch.from_numpy(o).float() for o in last_obs]
            last_action = [torch.tensor(a_idx).view(1,1) for a_idx in a]
            last_valid = [torch.tensor(_v, dtype=torch.uint8).view(1,-1) for _v in v]
            R += discount**step*r
            step += 1

        if t:
            print(R)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', action='store', type=str, default='OSD_S_4')
    parser.add_argument('--env_terminate_step', action='store', type=int, default=150)
    parser.add_argument('--n_agent', action='store', type=int, default=3)
    parser.add_argument('--n_episode', action='store', type=int, default=1)
    parser.add_argument('--p_id', action='store', type=int, default=0)

    test(**vars(parser.parse_args()))

if __name__ == '__main__':
    main()
