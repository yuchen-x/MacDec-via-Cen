import numpy as np
import IPython
import matplotlib.pyplot as plt
import torch
import pickle
import math
import copy
import random

from torch.optim import Adam, RMSprop
from itertools import chain

from .model import DDRQN, Cen_DDRQN
from .utils.Agent import Agent
from .utils.Cen_ctrl import Cen_Controller
from .utils.utils import Linear_Decay, get_conditional_argmax, get_conditional_action
from .envs_runner import EnvsRunner

from IPython.core.debugger import set_trace

# parameters for e-greedy policy
EPS_START = 1.0
EPS_DECAY = 2000
EPS_DECAY_LINEAR_RATE = 0.9999

OPTIMIZERS = {'Adam': Adam,
              'RMSprop': RMSprop}

class Team:
    
    def __init__(self, env, memory, n_agent, h_stable_at, dynamic_h=False, hysteretic=None, discount=0.99,
                 epsilon_end = 0.1, epsilon_linear_decay=False, epsilon_linear_decay_steps=0):
        self.env = env
        self.n_agent = n_agent
        self.memory = memory

        self.step_count = 0.0
        self.episode_count = 0.0
        self.episode_rewards = 0.0
 
        # hysteretic settings
        self.dynamic_h = dynamic_h
        (self.init_hysteretic, self.end_hysteretic) = hysteretic
        self.hysteretic = self.init_hysteretic
        self.discount = discount

        # epsilon for e-greedy
        self.epsilon = EPS_START
        self.epsilon_end = epsilon_end
        self.epsilon_linear_decay = epsilon_linear_decay
        self.eps_l_d = Linear_Decay(epsilon_linear_decay_steps, EPS_START, epsilon_end)
        
        self.HYSTERESIS_STABLE_AT = h_stable_at

    def create_agents(self):
        raise NotImplementedError

    def create_cen_controller(self):
        raise NotImplementedError

    def step(self):
        raise NotImplementedError

    def get_next_actions(self):
        raise NotImplementedError

    def update_dec_target_net(self):
        for agent in self.agents:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
    
    def update_cen_target_net(self):
        self.cen_controller.target_net.load_state_dict(self.cen_controller.policy_net.state_dict())

    def train(self):
        raise NotImplementedError

    def update_epsilon(self, step):
        # update epsilon:
        if self.epsilon_linear_decay:
            #self.epsilon = max(self.epsilon*EPS_DECAY_LINEAR_RATE, EPS_END)
            self.epsilon = self.eps_l_d._get_value(step)
        else:
            self.epsilon = self.epsilon_end + (EPS_START - self.epsilon_end) * math.exp(-1. * (step//8)  / EPS_DECAY)
    
    def update_hysteretic(self, step):
        if self.dynamic_h:
            self.hysteretic = min(self.end_hysteretic,
                                  ((self.end_hysteretic - self.init_hysteretic) / self.HYSTERESIS_STABLE_AT) * step + self.init_hysteretic)
        else:
            self.hysteretic = 1 - self.epsilon
    
    def get_init_inputs(self):
        raise NotImplementedError

    def sep_joint_exps(self, joint_exps):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError

    def load_check_point(self, idx_run):
        PATH = "./performance/" + self.save_dir + "/check_point/" + str(idx_run) + "_cen_controller_" + "1.tar"
        ckpt = torch.load(PATH)
        self.cen_controller.policy_net.load_state_dict(ckpt['policy_net_state_dict'])
        self.cen_controller.target_net.load_state_dict(ckpt['target_net_state_dict'])
        self.cen_controller.optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        for idx, agent in enumerate(self.agents):
            PATH = "./performance/" + self.save_dir + "/check_point/" + str(idx_run) + "_agent_" + str(idx) + "1.tar"
            ckpt = torch.load(PATH)
            agent.policy_net.load_state_dict(ckpt['policy_net_state_dict'])
            agent.target_net.load_state_dict(ckpt['target_net_state_dict'])
            agent.optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        self.hysteretic = ckpt['cur_hysteretic']
        self.epsilon = ckpt['cur_eps']
        self.episode_count = ckpt['n_epi']
        self.step_count = ckpt['cur_step']
        self.TEST_PERFORM = ckpt['TEST_PERFORM']
        self.memory.buf = ckpt['mem_buf']
        random.setstate(ckpt['random_state'])
        np.random.set_state(ckpt['np_random_state'])
        torch.set_rng_state(ckpt['torch_random_state'])

class Team_RNN(Team):

    def __init__(self, env, n_env, memory, n_agent, training_method, h_stable_at, discount=0.99, sample_epi=False, 
                 dynamic_h=False, hysteretic=None, h_explore=False, cen_explore=False, cen_explore_end=float('inf'), 
                 explore_switch=False, soft_action_selection=False, epsilon_end=0.1, epsilon_linear_decay=False, epsilon_linear_decay_steps=0, epsilon_exp_decay=False, 
                 optimizer='Adam', learning_rate=0.001, device='cpu', save_dir=None, 
                 nn_model_params={}, **hyper_params):

        super(Team_RNN, self).__init__(env, memory, n_agent, h_stable_at, dynamic_h, hysteretic, discount,
                                       epsilon_end, epsilon_linear_decay, epsilon_linear_decay_steps)

        # create multiprocessor for multiple envs running parallel
        self.envs_runner = EnvsRunner(self.env, self.memory, n_env, h_explore, self.get_next_actions)
        self.envs_runner.reset()
        self.n_env = n_env
        self.cen_explore = cen_explore
        self.cen_explore_end = cen_explore_end
        self.exp_switch = explore_switch

        # sample the whole episode for training
        self.sample_epi = sample_epi
        
        # training method
        self.soft_action_selection = soft_action_selection
        self.training_method = training_method
        self.nn_model_params = nn_model_params
        self.hyper_params = hyper_params
        self.optimizer = optimizer
        self.lr = learning_rate
        
        # save model
        self.save_dir = save_dir
        self.device = device

        # statistic of training and testing
        self.TRAIN_PERFORM = []
        self.TEST_PERFORM = []

        # create agents
        self.create_agents()

        # create cen_controller
        self.create_cen_controller()

    def create_agents(self):
        self.agents=[]
        for i in range(self.n_agent):
            agent = Agent()
            agent.idx = i
            agent.policy_net = DDRQN(self.env.obs_size[i], self.env.n_action[i], **self.nn_model_params).to(self.device)
            agent.target_net = DDRQN(self.env.obs_size[i], self.env.n_action[i], **self.nn_model_params).to(self.device)
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
            agent.optimizer = OPTIMIZERS[self.optimizer](agent.policy_net.parameters(), lr=self.lr)
            self.agents.append(agent)

    def create_cen_controller(self):
        self.cen_controller = Cen_Controller()
        self.cen_controller.policy_net = Cen_DDRQN(np.sum(self.env.obs_size),np.prod(self.env.n_action), **self.nn_model_params).to(self.device)
        self.cen_controller.target_net = Cen_DDRQN(np.sum(self.env.obs_size),np.prod(self.env.n_action), **self.nn_model_params).to(self.device)
        self.cen_controller.target_net.load_state_dict(self.cen_controller.policy_net.state_dict())
        self.cen_controller.optimizer = OPTIMIZERS[self.optimizer](self.cen_controller.policy_net.parameters(), lr=self.lr)

    def step(self, idx_run):
        if self.step_count == 0:
            self.evaluate()
            with open("./performance/" + self.save_dir + "/test/test_perform" + str(idx_run) + ".pickle", 'wb') as handle:
                pickle.dump(self.TEST_PERFORM, handle)

        self.step_count += 1.0

        n_episode_done = self.envs_runner.step()
        self.episode_count += n_episode_done

        if n_episode_done > 0 and not self.episode_count % 10:
            self.evaluate()
            with open("./performance/" + self.save_dir + "/test/test_perform" + str(idx_run) + ".pickle", 'wb') as handle:
                pickle.dump(self.TEST_PERFORM, handle)

            if self.TEST_PERFORM[-1] == np.max(self.TEST_PERFORM):
                for agent in self.agents:
                     PATH = "./policy_nns/" + self.save_dir + "/best_agent_" + str(agent.idx) + ".pt"
                     torch.save(agent.policy_net, PATH)

            for agent in self.agents:
                 PATH = "./policy_nns/" + self.save_dir + "/agent_" + str(agent.idx) + ".pt"
                 torch.save(agent.policy_net, PATH)

            PATH = "./policy_nns/" + self.save_dir + "/cen_controller.pt"
            torch.save(self.cen_controller.policy_net, PATH)

    def get_next_actions(self, joint_obs, dec_h_states, cen_h_state, cen_last_action, last_valid, eval=False):
        # explore using dec policies
        with torch.no_grad():
            dec_actions = []
            new_dec_h_states = []
            for i, agent in enumerate(self.agents):
                if last_valid[agent.idx]:
                    agent.policy_net.eval()
                    Q, h = agent.policy_net(joint_obs[agent.idx].view(1,1,self.env.obs_size[agent.idx]), dec_h_states[agent.idx])
                    a = Q.squeeze(1).max(1)[1].item()

                    if not eval:
                        if self.soft_action_selection:
                            logits = torch.log_softmax((Q/self.epsilon), 1)
                            a = torch.distributions.Categorical(logits=logits).sample().item()
                        else:
                            if np.random.random() < self.epsilon:
                                a = self.env.action_space_sample(i)

                    dec_actions.append(a)
                    new_dec_h_states.append(h)

                else:
                    dec_actions.append(-1)
                    new_dec_h_states.append(dec_h_states[agent.idx])

        if not eval:
            # explore using cen policy
            with torch.no_grad():
                if max(last_valid) == 1.0:
                    self.cen_controller.policy_net.eval()
                    Q, h = self.cen_controller.policy_net(torch.cat(joint_obs).view(1,1,np.sum(self.env.obs_size)), cen_h_state)
                    a = get_conditional_argmax(Q, get_conditional_action(torch.cat(cen_last_action).view(1,-1), torch.cat(last_valid).view(1,-1)), self.env.n_action).item()

                    if self.soft_action_selection:
                        logits = torch.log_softmax((Q/self.epsilon), 1)
                        a = torch.distributions.Categorical(logits=logits).sample().item()
                    else:
                        if np.random.random() < self.epsilon:
                            a = np.random.choice(range(Q.shape[-1]), 1)[0]

                    cen_actions = np.unravel_index(a, self.env.n_action)
                    cen_h_state = h
                else:
                    cen_actions = [-1] * self.n_agent

        if self.cen_explore and not eval and self.episode_count <= self.cen_explore_end:
            # switch explore between cen_Q and dec_Q according to epsilon
            if self.exp_switch and np.random.random() > self.epsilon:
                return dec_actions, new_dec_h_states, cen_h_state
            else:
                return cen_actions, new_dec_h_states, cen_h_state
        else:
            return dec_actions, new_dec_h_states, cen_h_state

    def train(self):
        if self.sample_epi:
            batch, trace_len = self.memory.sample()
            self.hyper_params['trace_len'] = trace_len
        else:
            batch = self.memory.sample()
        cen_batch = self.cen_sep_joint_exps(batch)
        dec_batch = self.dec_sep_joint_exps(batch)

        self.training_method(self.env, self.agents, self.cen_controller, (cen_batch, dec_batch), self.hysteretic, self.discount, **self.hyper_params)

    def get_init_inputs(self):
        return [torch.from_numpy(i).float() for i in self.env.reset()], [None]*self.n_agent, None

    def dec_sep_joint_exps(self, joint_exps):
        # seperate the joint experience for individual agents
        exps = [[] for _ in range(self.n_agent)]
        for o, a, id_r, j_r, o_n, t, id_v, j_v in chain(*joint_exps):
            for i in range(self.n_agent):
                exps[i].append([o[i], a[i], id_r[i], j_r, o_n[i], t, id_v[i], j_v])
        return exps
    
    def cen_sep_joint_exps(self, joint_exps):
        # seperate the joint experience for individual agents
        exp = []
        for o, a, id_r, j_r, o_n, t, id_v, j_v in chain(*joint_exps):
            exp.append([torch.cat(o).view(1,-1), 
                        torch.cat(a).view(1,-1),
                        torch.tensor(np.ravel_multi_index(a, self.env.n_action)).view(1,-1),
                        id_r, 
                        j_r, 
                        torch.cat(o_n).view(1,-1),
                        t,
                        torch.cat(id_v).view(-1),
                        j_v])
        return exp

    def evaluate(self, n_episode=1):

        R, L = 0, 0

        for _ in range(n_episode):
            t = 0
            step = 0

            last_obs, dec_h_states, cen_h_state = self.get_init_inputs()
            last_valid = [torch.tensor([[1]]).byte()] * self.n_agent
            cen_last_action = [torch.tensor([[-1]])] * self.n_agent

            while not t:
                a, dec_h_states, cen_h_state = self.get_next_actions(last_obs, dec_h_states, cen_h_state, cen_last_action, last_valid, eval=True)
                a, obs, r, t, v = self.env.step(a)

                last_obs = [torch.from_numpy(o).float() for o in obs]
                cen_last_action = [torch.tensor(a_idx).view(1,1) for a_idx in a]
                last_valid = [torch.tensor(_v, dtype=torch.uint8).view(1,-1) for _v in v]

                R += self.discount**step * r
                step += 1

        self.TEST_PERFORM.append(R/n_episode)

