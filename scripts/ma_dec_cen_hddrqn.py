import argparse
import numpy as np
import torch
import os
import sys
import time
import random
import IPython

from rlmamr.my_env.box_pushing_MA import BoxPushing_harder as BP_MA
from rlmamr.my_env.osd_ma_single_room import ObjSearchDelivery_v4 as OSD_S_4

from rlmamr.MA_dec_cen_hddrqn.team import Team_RNN
from rlmamr.MA_dec_cen_hddrqn.utils.memory import ReplayMemory_rand, ReplayMemory_epi
from rlmamr.MA_dec_cen_hddrqn.utils.utils import save_check_point
from rlmamr.MA_dec_cen_hddrqn.learning_methods import QLearn_squ_dec_cen_0, QLearn_squ_dec_cen_1

from IPython.core.debugger import set_trace

ENVIRONMENTS = {
        'BP_MA': BP_MA,
        'OSD_S_4': OSD_S_4
        }

QLearn_squ_dec_cen = [QLearn_squ_dec_cen_0, QLearn_squ_dec_cen_1]

def train(env_name, grid_dim, obs_one_hot, target_flick_prob, agent_trans_noise, target_rand_move, 
          h_speed_ps, tb_m_speed, tb_m_noisy, f_p_obj_tc, f_l_obj_tc, d_pen,
          env_terminate_step, n_env, n_agent, total_epi, replay_buffer_size, sample_epi, 
          dynamic_h, init_h, end_h, h_stable_at, eps_l_d, eps_l_d_steps, eps_end, eps_e_d, softmax_explore, h_explore, cen_explore, cen_explore_end, explore_switch, db_step,
          optim, l_rate, discount, huber_l, g_clip, g_clip_v, g_clip_norm, g_clip_max_norm, l_mode,
          start_train, train_freq, target_update_freq, trace_len, sub_trace_len, batch_size,
          sort_traj, dec_mlp_layer_size, cen_mlp_layer_size, rnn, rnn_input_dim, rnn_layer_num, dec_rnn_h_size, cen_rnn_h_size, run_id, resume, save_ckpt, seed, save_dir, device, **kwargs):

    # define the name of the directory to be created
    os.makedirs("./performance/"+save_dir+"/train", exist_ok=True)
    os.makedirs("./performance/"+save_dir+"/test", exist_ok=True)
    os.makedirs("./performance/"+save_dir+"/check_point", exist_ok=True)
    os.makedirs("./policy_nns/"+save_dir, exist_ok=True)

    if seed is not None:
        seed = seed *10 
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
    torch.set_num_threads(1)

    # create env 
    ENV = ENVIRONMENTS[env_name]
    if env_name.startswith('OSD'):
        env = ENV(terminate_step=env_terminate_step, 
                  human_speed_per_step=[h_speed_ps], 
                  TB_move_speed=tb_m_speed, 
                  TB_move_noisy=tb_m_noisy, 
                  fetch_pass_obj_tc=f_p_obj_tc, 
                  fetch_look_for_obj_tc=f_l_obj_tc, 
                  delay_delivery_penalty=d_pen)
    elif env_name.startswith('BP'):
        env = ENV(tuple(grid_dim),terminate_step=env_terminate_step)

    # create replay buffer
    if sample_epi:
        memory = ReplayMemory_epi(env.n_agent, env.obs_size, batch_size, size=replay_buffer_size)
    else:
        memory = ReplayMemory_rand(env.n_agent, env.obs_size, trace_len, batch_size, size=replay_buffer_size)

    # collect hyper params:
    hyper_params = {'epsilon_linear_decay': eps_l_d,
                    'epsilon_linear_decay_steps': eps_l_d_steps,
                    'epsilon_end': eps_end,
                    'soft_action_selection': softmax_explore,
                    'h_explore': h_explore,
                    'cen_explore': cen_explore,
                    'cen_explore_end': cen_explore_end,
                    'explore_switch': explore_switch,
                    'epsilon_exp_decay': eps_e_d,
                    'dynamic_h': dynamic_h,
                    'hysteretic': (init_h, end_h),
                    'optimizer': optim,
                    'learning_rate': l_rate,
                    'discount': discount,
                    'huber_loss': huber_l,
                    'grad_clip': g_clip,
                    'grad_clip_value': g_clip_v,
                    'grad_clip_norm': g_clip_norm,
                    'grad_clip_max_norm': g_clip_max_norm,
                    'sample_epi': sample_epi,
                    'trace_len': trace_len,
                    'sub_trace_len': sub_trace_len,
                    'batch_size': batch_size,
                    'sort_traj': sort_traj,
                    'device':device}

    model_params = {'dec_mlp_layer_size': dec_mlp_layer_size,
                    'cen_mlp_layer_size': cen_mlp_layer_size,
                    'rnn_input_dim': rnn_input_dim,
                    'rnn_layer_num': rnn_layer_num,
                    'dec_rnn_h_size': dec_rnn_h_size,
                    'cen_rnn_h_size': cen_rnn_h_size}

    # create team
    team = Team_RNN(env, n_env, memory, env.n_agent, QLearn_squ_dec_cen[l_mode], h_stable_at,
            save_dir=save_dir, nn_model_params=model_params, **hyper_params)

    t = time.time()
    t_ckpt = time.time()

    training_count=0
    target_updating_count = 0

    step = 0

    # continue training using the lastest check point
    if resume:
        team.load_check_point(run_id)
        step = team.step_count

    while team.episode_count < total_epi:
        team.step(run_id)
        if (not step % train_freq) and team.episode_count >= start_train:
            # update hysteretic learning rate
            if db_step:
                team.update_hysteretic(step)
            else:
                team.update_hysteretic(team.episode_count)

            for _ in range(n_env):
                team.train()

            # update epsilon
            if db_step:
                team.update_epsilon(step)
            else:
                team.update_epsilon(team.episode_count)

            training_count += 1

        if not step % target_update_freq: 
            team.update_dec_target_net() 
            team.update_cen_target_net() 
            if (time.time() - t_ckpt) // 3600 >= 23 and save_ckpt:
                save_check_point(team.agents, team.cen_controller, step, team.episode_count, team.hysteretic, team.epsilon, save_dir, team.memory, run_id, team.TEST_PERFORM) 
                t_ckpt = time.time()
                sys.exit()
            target_updating_count += 1 
            print('[{}]run, [{:.1f}K] took {:.3f}hr to finish {} episodes {} trainning and {} target_net updating (eps={})'.format(
                    run_id, step/1000, (time.time()-t)/3600, team.episode_count, training_count, target_updating_count, team.epsilon), flush=True)
        step += 1

    if team.episode_count == total_epi and save_ckpt:
        save_check_point(team.agents, team.cen_controller, step, team.episode_count, team.hysteretic, team.epsilon, save_dir, team.memory, run_id, team.TEST_PERFORM) 
        print("Finish entire training ... ")
    team.envs_runner.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name',           action='store',        type=str,             default='OSD_S_4',     help='Domain name')
    parser.add_argument('--env_terminate_step', action='store',        type=int,             default=150,           help='Maximal steps for termination')
    parser.add_argument('--grid_dim',           action='store',        type=int,  nargs=2,   default=[6,6],         help='Grid world size')
    parser.add_argument('--obs_one_hot',        action='store_true',                                                help='Whether represents observation as a one-hot vector')
    parser.add_argument('--target_flick_prob',  action='store',        type=float,           default=0.3,           help='Probability of not observing target in Capture Target domain')
    parser.add_argument('--agent_trans_noise',  action='store',        type=float,           default=0.1,           help='Agent dynamics noise in Capture Target domain')
    parser.add_argument('--target_rand_move',   action='store_true',                                                help='Whether apply target random movement')
    parser.add_argument('--n_env',              action='store',        type=int,             default=1,             help='Number of envs running in parallel')
    parser.add_argument('--n_agent',            action='store',        type=int,             default=2,             help='Number of agents')
    
    #OSD params
    parser.add_argument('--h_speed_ps',         action='store',        type=int,  nargs='+', default=[18],          help='Time-step cost for each work step of human task')
    parser.add_argument('--tb_m_speed',         action='store',        type=float,           default=0.6,           help='Turtlebot move speed m/s')
    parser.add_argument('--tb_m_noisy',         action='store',        type=float,           default=0.0,           help='Turtlebot dynamics noise')
    parser.add_argument('--f_p_obj_tc',         action='store',        type=int,             default=4,             help='Time-step cost for finishing macro-action Pass_obj by Fetch robot')
    parser.add_argument('--f_l_obj_tc',         action='store',        type=int,             default=6,             help='Time-step cost for finishing macro-action Look_for_obj by Fetch robot')
    parser.add_argument('--d_pen',              action='store_true',                                                help='Whether apply penatly for delayed tool delivery')

    parser.add_argument('--total_epi',          action='store',        type=int,             default=40*1000,       help='Number of training episodes')
    parser.add_argument('--replay_buffer_size', action='store',        type=int,             default=1000,          help='Number of episodes/sequences in replay buffer')
    parser.add_argument('--sample_epi',         action='store_true',                                                help='Whether use full-episode-based replay buffer or not')
    parser.add_argument('--dynamic_h',          action='store_true',                                                help='Whether apply hysteritic learning rate decay or not')
    parser.add_argument('--init_h',             action='store',        type=float,           default=1.0,           help='Initial value of hysteretic learning rate')
    parser.add_argument('--end_h',              action='store',        type=float,           default=1.0,           help='Ending value of hysteretic learning rate')
    parser.add_argument('--h_stable_at',        action='store',        type=int,             default=4*1000,        help='Decaying period according to episodes/steps')

    parser.add_argument('--eps_l_d',            action='store_true',                                                help='Whether use epsilon linear decay for exploartion or not')
    parser.add_argument('--eps_l_d_steps',      action='store',        type=int,             default=6*1000,        help='Decaying period according to episodes/steps')
    parser.add_argument('--eps_end',            action='store',        type=float,           default=0.1,           help='Ending value of epsilon')
    parser.add_argument('--eps_e_d',            action='store_true',                                                help='Whether use episode-based epsilon linear decay or not')
    parser.add_argument('--softmax_explore',    action='store_true',                                                help='Whether apply softmac for exploration')
    parser.add_argument('--h_explore',          action='store_true',                                                help='Whether use history-based policy for exploration or not')
    parser.add_argument('--cen_explore',        action='store_true',                                                help='Whether use centralized explore or not')
    parser.add_argument('--cen_explore_end',    action='store',        type=int,             default=100*1000,      help='Number of episodes for centralied explore')
    parser.add_argument('--explore_switch',     action='store_true',                                                help='Whether switch between centralized explore and decentralized explore')
    parser.add_argument('--db_step',            action='store_true',                                                help='Whether use step-based decaying manner or not')

    parser.add_argument('--l_mode',             action='store',         type=int,            default=0,             help='Index of centralized training manner')
    parser.add_argument('--optim',              action='store',         type=str,            default='Adam',        help='Optimizer')
    parser.add_argument('--l_rate',             action='store',         type=float,          default=0.0006,        help='Learning rate')
    parser.add_argument('--discount',           action='store',         type=float,          default=0.95,          help='Discount factor')
    parser.add_argument('--huber_l',            action='store_true',                                                help='Whether use huber loss or not')
    parser.add_argument('--g_clip',             action='store_true',                                                help='Whether use gradient clip or not')
    parser.add_argument('--g_clip_v',           action='store',         type=float,          default=0.0,           help='Absolute Value limitation for gradient clip')
    parser.add_argument('--g_clip_norm',        action='store_true',                                                help='Whether use norm-based gradient clip')
    parser.add_argument('--g_clip_max_norm',    action='store',         type=float,          default=0.0,           help='Norm limitation for gradient clip')

    parser.add_argument('--start_train',        action='store',         type=int,            default=2,             help='Training starts after a number of episodes')
    parser.add_argument('--train_freq',         action='store',         type=int,            default=30,            help='Updating performs every a number of steps')
    parser.add_argument('--target_update_freq', action='store',         type=int,            default=5000,          help='Updating target net every a number of steps')
    parser.add_argument('--trace_len',          action='store',         type=int,            default=10,            help='The length of each sequence saved in replay buffer when not using full-episode-based replay buffer') 
    parser.add_argument('--sub_trace_len',      action='store',         type=int,            default=1,             help='Minimal length of a sequence for traning data filtering')
    parser.add_argument('--sort_traj',          action='store_true',                                                help='Whether sort sequences based on its valid length after squeezing experiences or not, redundant param for pytorch version later than 1.1.0')
    parser.add_argument('--batch_size',         action='store',         type=int,            default=16,            help='Number of episodes/sequences in a batch')

    parser.add_argument('--rnn',                action='store_true',                                                help='Whether using rnn-based agent')
    parser.add_argument('--rnn_input_dim',      action='store',         type=int,            default=128,           help='Input dimension of RNN layer')

    parser.add_argument('--dec_mlp_layer_size', action='store',         type=int,            default=32,            help='MLP layer dimension of decentralized policy-net')
    parser.add_argument('--cen_mlp_layer_size', action='store',         type=int,            default=32,            help='MLP layer dimension of centralized policy-net')
    parser.add_argument('--rnn_layer_num',      action='store',         type=int,            default=1,             help='Number of RNN layers')
    parser.add_argument('--dec_rnn_h_size',     action='store',         type=int,            default=64,            help='RNN hidden layer dimension of decentralized policy-net')
    parser.add_argument('--cen_rnn_h_size',     action='store',         type=int,            default=128,           help='RNN hidden layer dimension of centralized policy-net')

    parser.add_argument('--resume',             action='store_true',                                                help='Whether use saved ckpt to continue training or not')
    parser.add_argument('--save_ckpt',          action='store_true',                                                help='Whether save ckpt or not')
    parser.add_argument('--run_id',             action='store',         type=int,            default=0,             help='Index of a run')
    parser.add_argument('--seed',               action='store',         type=int,            default=None,          help='Random seed of a run')
    parser.add_argument('--save_dir',           action='store',         type=str,            default=None,          help='Directory name for storing trainning results')
    parser.add_argument('--device',             action='store',         type=str,            default='cpu',         help='Which device (CPU/GPU) to use.')

    params = vars(parser.parse_args())

    train(**params)

if __name__ == '__main__':
    main()
