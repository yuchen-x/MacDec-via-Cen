import argparse
import numpy as np
import torch
import os
import sys
import time
import random
import IPython
import warnings
warnings.filterwarnings('error')

from rlmamr.my_env.box_pushing_MA import BoxPushing_harder as BP_MA
from rlmamr.my_env.osd_ma_single_room import ObjSearchDelivery_v4 as OSD_S_4

from rlmamr.MA_cen_condi_ddrqn.team_cen_condi import Team_RNN
from rlmamr.MA_cen_condi_ddrqn.utils.memory_cen_condi import ReplayMemory_rand, ReplayMemory_epi
from rlmamr.MA_cen_condi_ddrqn.utils.utils import save_check_point
from rlmamr.MA_cen_condi_ddrqn.learning_methods_cen_condi import QLearn_squ_cen_condi_0

from IPython.core.debugger import set_trace

ENVIRONMENTS = {
        'BP_MA': BP_MA,
        'OSD_S_4': OSD_S_4
        }

QLearns = [QLearn_squ_cen_condi_0]

def train(env_name, grid_dim, goal, obs_one_hot, target_flick_prob, agent_trans_noise, target_rand_move, 
          h_speed_ps, tb_m_speed, tb_m_noisy, f_p_obj_tc, f_l_obj_tc, d_pen,
          env_terminate_step, n_env, n_agent, cen_mode, total_epi, replay_buffer_size, sample_epi, 
          dynamic_h, init_h, end_h, h_stable_at, eps_l_d, eps_l_d_steps, eps_end, eps_e_d, softmax_explore, h_explore, db_step,
          optim, l_rate, discount, huber_l, g_clip, g_clip_v, g_clip_norm, g_clip_max_norm,
          start_train, train_freq, target_update_freq, trace_len, sub_trace_len, batch_size,
          sort_traj, batch_norm, rnn, mlp_layer_size, rnn_layer_num, rnn_h_size, gru, run_id, resume, save_ckpt, seed, save_dir, device, **kwargs):

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
        env = ENV(terminate_step=env_terminate_step, human_speed_per_step=[h_speed_ps], TB_move_speed=tb_m_speed, TB_move_noisy=tb_m_noisy, fetch_pass_obj_tc=f_p_obj_tc, fetch_look_for_obj_tc=f_l_obj_tc, delay_delivery_penalty=d_pen)
    else:
        env = ENV(tuple(grid_dim),terminate_step=env_terminate_step)

    # create replay buffer
    if sample_epi:
        memory = ReplayMemory_epi(env.n_agent, env.obs_size, batch_size, size=replay_buffer_size)
    else:
        memory = ReplayMemory_rand(env.n_agent, env.obs_size, trace_len, batch_size, size=replay_buffer_size)

    # collect hyper params:
    hyper_params = {'cen_mode': cen_mode,
                    'batch_norm': batch_norm,
                    'soft_action_selection': softmax_explore,
                    'epsilon_linear_decay': eps_l_d,
                    'epsilon_linear_decay_steps': eps_l_d_steps,
                    'epsilon_end': eps_end,
                    'h_explore': h_explore,
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


    model_params = {'mlp_layer_size': mlp_layer_size,
                    'rnn_layer_num': rnn_layer_num,
                    'rnn_h_size': rnn_h_size,
                    'GRU': gru}

    # create team
    team = Team_RNN(env, n_env, memory, env.n_agent, QLearns[cen_mode], h_stable_at,
            save_dir=save_dir, nn_model_params=model_params, **hyper_params)

    t = time.time()
    training_count=0
    target_updating_count = 0

    step = 0

    if resume:
        team.load_check_point(run_id)
        step = team.step_count

    while team.episode_count < total_epi:
        team.step(run_id)
        if (not step % train_freq) and team.episode_count >= start_train:
            if db_step:
                team.update_hysteretic(step)
            else:
                team.update_hysteretic(team.episode_count)

            for _ in range(n_env):
                team.train()

            if db_step:
                team.update_epsilon(step)
            else:
                team.update_epsilon(team.episode_count)

            training_count += 1
            #save_check_point(team.agents, step, team.hysteretic, team.epsilon, save_dir) 
        if not step % target_update_freq: 
            team.update_target_net() 
            if (time.time()-t) / 3600 >= 23.0 and save_ckpt:
                save_check_point(team.cen_controller, step, team.episode_count, team.hysteretic, team.epsilon, save_dir, team.memory, run_id, team.TEST_PERFORM) 
                sys.exit()
            target_updating_count += 1 
            print('[{}]run, [{:.1f}K] took {:.3f}hr to finish {} episodes {} trainning and {} target_net updating (eps={})'.format(
                    run_id, step/1000, (time.time()-t)/3600, team.episode_count, training_count, target_updating_count, team.epsilon), flush=True)
        step += 1

    if team.episode_count == total_epi and save_ckpt:
        save_check_point(team.cen_controller, step, team.episode_count, team.hysteretic, team.epsilon, save_dir, team.memory, run_id, team.TEST_PERFORM) 
        print("Finish entire training ...")
    team.envs_runner.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', action='store', type=str, default='OSD_S_4')
    parser.add_argument('--env_terminate_step', action='store', type=int, default=150)
    parser.add_argument('--grid_dim', action='store', type=int, nargs=2, default=[6,6])
    parser.add_argument('--obs_one_hot', action='store_true')
    parser.add_argument('--target_flick_prob', action='store', type=float, default=0.3)
    parser.add_argument('--agent_trans_noise', action='store', type=float, default=0.1)
    parser.add_argument('--target_rand_move', action='store_true')
    parser.add_argument('--n_env', action='store', type=int, default=1)
    parser.add_argument('--n_agent', action='store', type=int, default=2)
    parser.add_argument('--cen_mode', action='store', type=int, default=0)
    parser.add_argument('--goal', action='store', type=int, nargs='+', default=[0,1])

    #OSD params
    parser.add_argument('--h_speed_ps', action='store', type=int, nargs='+', default=[18])
    parser.add_argument('--tb_m_speed', action='store', type=float, default=0.6)
    parser.add_argument('--tb_m_noisy', action='store', type=float, default=0.0)
    parser.add_argument('--f_p_obj_tc', action='store', type=int, default=4)
    parser.add_argument('--f_l_obj_tc', action='store', type=int, default=6)
    parser.add_argument('--d_pen', action='store', type=float, default=0.0) # delay delivery penalty

    parser.add_argument('--total_epi', action='store', type=int, default=40*1000)
    parser.add_argument('--replay_buffer_size', action='store', type=int, default=1000)
    parser.add_argument('--sample_epi', action='store_true')
    parser.add_argument('--dynamic_h', action='store_true')
    parser.add_argument('--init_h', action='store', type=float, default=1.0)
    parser.add_argument('--end_h', action='store', type=float, default=1.0)
    parser.add_argument('--h_stable_at', action='store', type=int, default=6*1000)

    parser.add_argument('--eps_l_d', action='store_true')
    parser.add_argument('--eps_l_d_steps', action='store', type=int, default=6*1000)
    parser.add_argument('--eps_end', action='store', type=float, default=0.1)
    parser.add_argument('--eps_e_d', action='store_true')
    parser.add_argument('--softmax_explore', action='store_true')
    parser.add_argument('--h_explore', action='store_true')
    parser.add_argument('--db_step', action='store_true')

    parser.add_argument('--optim', action='store', type=str, default='Adam')
    parser.add_argument('--l_rate', action='store', type=float, default=0.0006)
    parser.add_argument('--discount', action='store', type=float, default=1.0)
    parser.add_argument('--huber_l', action='store_true')
    parser.add_argument('--g_clip', action='store_true')
    parser.add_argument('--g_clip_v', action='store', type=float, default=0.0)
    parser.add_argument('--g_clip_norm', action='store_true')
    parser.add_argument('--g_clip_max_norm', action='store', type=float, default=0.0)

    parser.add_argument('--start_train', action='store', type=int, default=2)
    parser.add_argument('--train_freq', action='store', type=int, default=30)
    parser.add_argument('--target_update_freq', action='store', type=int, default=5000)
    parser.add_argument('--trace_len', action='store', type=int, default=10) 
    parser.add_argument('--sub_trace_len', action='store', type=int, default=1)
    parser.add_argument('--sort_traj', action='store_true')
    parser.add_argument('--batch_size', action='store', type=int, default=16)

    parser.add_argument('--batch_norm', action='store_true')
    parser.add_argument('--rnn', action='store_true')

    parser.add_argument('--mlp_layer_size', action='store', type=int, default=32)
    parser.add_argument('--rnn_layer_num', action='store', type=int, default=1)
    parser.add_argument('--rnn_h_size', action='store', type=int, default=64)
    parser.add_argument('--gru', action='store_true')

    parser.add_argument('--n_run', action='store', type=int, default=1)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--save_ckpt', action='store_true')
    parser.add_argument('--run_id', action='store', type=int, default=0)
    parser.add_argument('--seed', action='store', type=int, default=0)
    parser.add_argument('--save_dir', action='store', type=str, default=None)
    parser.add_argument('--device', action='store', type=str, default='cpu')

    params = vars(parser.parse_args())

    if params['n_run'] > 1:
        for i in range(params['n_run']):
            params['run_id'] += i
            params['seed'] += i
            train(**params)
    else:
        train(**params)

if __name__ == '__main__':
    main()
