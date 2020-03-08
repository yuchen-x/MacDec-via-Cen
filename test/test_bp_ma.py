import argparse
import numpy as np
import torch
import os
import sys
sys.path.append("..")
import time
import IPython

from rlmamr.my_env.box_pushing_MA import BoxPushing_harder as BP_MA

from rlmamr.MA_hddrqn.utils.Agent import Agent
from IPython.core.debugger import set_trace

ACTIONS = ["GT_SB0", "GT_SB1", "GT_BB0", "GT_BB1", "PUSH", "T_L", "T_R", "STAY"]

ENVIRONMENTS = {
        'BP_MA': BP_MA}

def get_actions_and_h_states(env, agents, last_valid, joint_obs, joint_h_states):
    with torch.no_grad():
        actions = []
        h_states = []
        for idx,agent in enumerate(agents):
            if last_valid[agent.idx]:
                Q, h = agent.policy_net(joint_obs[agent.idx].view(1,1,env.obs_size[agent.idx]), joint_h_states[agent.idx])
                #print(f'agent{agent.idx}', Q)
                a = Q.squeeze(1).max(1)[1].item()
                actions.append(a)
                h_states.append(h)
            else:
                actions.append(env.agents[agent.idx].cur_action.idx)
                #actions.append(env.envs[idx].agents[0].cur_action.idx)
                h_states.append(joint_h_states[agent.idx])

    return actions, h_states

def get_init_inputs(env,n_agent):
    return [torch.from_numpy(i).float() for i in env.reset(True)], [None]*n_agent

def test(env_name, env_terminate_step, grid_dim, n_agent, n_episode):
    ENV = ENVIRONMENTS[env_name]
    env = ENV(tuple(grid_dim), terminate_step=env_terminate_step,obs_teammate=True, random_init=True)

    agents = []
    
    for i in range(n_agent):
        agent = Agent()
        agent.idx = i
        agent.policy_net = torch.load("./policy_nns/BP_MA/agent_" + str(i) + str(grid_dim[0]) + ".pt")
        agent.policy_net.eval()
        agents.append(agent)

    R = 0
    discount=0.95
    step = 0.0

    for e in range(n_episode):
        t = 0
        last_obs, h_states = get_init_inputs(env, n_agent)
        last_valid = [1.0] * n_agent
        while not t:
            a, h_states = get_actions_and_h_states(env, agents, last_valid, last_obs, h_states)
            time.sleep(0.1)
            a, last_obs, r, t, v = env.step(a,True)
            last_obs = [torch.from_numpy(o).float() for o in last_obs]
            last_valid = v
            R += discount**step*r
            step += 1.0
        time.sleep(0.1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', action='store', type=str, default='BP_MA')
    parser.add_argument('--env_terminate_step', action='store', type=int, default=100)
    parser.add_argument('--grid_dim', action='store', type=int, nargs=2, default=[10,10])
    parser.add_argument('--n_agent', action='store', type=int, default=2)
    parser.add_argument('--n_episode', action='store', type=int, default=20)

    test(**vars(parser.parse_args()))

if __name__ == '__main__':
    main()


