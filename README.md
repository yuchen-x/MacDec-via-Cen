# Learning Multi-Robot Decentralized Macro-Action-Based Policies via a Centralized Q-Net

In this [paper](https://arxiv.org/pdf/1909.08776.pdf), we first introduce a new macro-action-based decentralized multi-agent double deep recurrent Q-net (**MacDec-MADDRQN**) which adopts *centralized trainning with decentralized execution* by allowing each decentralized Q-net update to use a centralized Q-net for action selection. In order to balance centralized and decentralized exploration, a general version, called **Parallel-MacDec-MADDRQN**, is also proposed. The code in this repo is to implement these two algorithms. 

- The decentralized macro-action-based policies learned via **MacDec-MADDRQN** enable the agents to collaboratively push the big box for higher credits:

<p align="center">
  <img src="https://github.com/yuchen-x/gifs/blob/master/bpma10.gif" width="25%" hspace="40">
  <img src="https://github.com/yuchen-x/gifs/blob/master/bpma30.GIF" width="25%" hspace="40">
</p>

- A team of robots collaborate to bring the correct tools to a human at the right time by running the decentralized macro-action-based policies learned via **Parallel-MacDec-MADDRQN**:
<p align="center">
  <img src="https://github.com/yuchen-x/gifs/blob/master/osd.GIF" width="50%">
</p>

## Installation

- To install the anaconda virtual env with all the dependencies:
  ```
  cd Anaconda_Env/
  conda env create -f icra2020.yml
  ```
- To install the python module:
  ```
  cd MacDec-via-Cen
  pip install -e .
  ```

## MacDec-MADDRQN
Use either decentralized Q-nets or centralized Q-net as the exploration policy to generate trainning data; Each decentralized Q-net is then optimized via a novel double-Q update rule by minimizing the loss:

<p align="center">
  <img src="https://github.com/yuchen-x/gifs/blob/master/new-double-q.png" width="60%">
</p>

where, the target value for updating each decentralized macro-action Q-net is calculated by using a centralized Q-net for macro-action selection and the corresponding decentralized Q-net for value estimation.

Training in Box Pushing domain and the warehouse tool delivery domain (single run):
- Box Pushing (10 x 10)
  ```
  ma_dec_cen_hddrqn.py --grid_dim 10 10 --env_name=BP_MA --env_terminate_step=100 --trace_len=15 --batch_size=128 --dec_rnn_h_size=32 --cen_rnn_h_size=64 --train_freq=15 --total_epi=15000 --replay_buffer_size=80000 --l_rate=0.001 --discount=0.98 --start_train=2 --l_mode=0 --cen_explore --eps_end=0.1 --dynamic_h --eps_l_d --save_dir=bpma10 --seed=0 --run_id=0
  ```

- Box Pushing (30 x 30)
  ```
  ma_dec_cen_hddrqn.py --grid_dim 30 30 --env_name=BP_MA --env_terminate_step=200 --trace_len=45 --batch_size=128 --dec_rnn_h_size=32 --cen_rnn_h_size=64 --train_freq=45 --total_epi=15000 --replay_buffer_size=80000 --eps_l_d_steps=6000 --l_rate=0.001 --discount=0.98 --start_train=2 --l_mode=0 --cen_explore --eps_end=0.1 --dynamic_h --eps_l_d --save_dir=bpma30 --seed=0 --run_id=0
  ```
- Warehouse Tool Delivery
  ```
  ma_dec_cen_hddrqn.py --env_name=OSD_S_4 --env_terminate_step=150 --batch_size=16 --dec_rnn_h_size=64 --cen_rnn_h_size=64 --train_freq=30 --total_epi=40000 --replay_buffer_size=1000 --eps_l_d_steps=6000 --l_rate=0.0006 --discount=1.0 --start_train=2 --l_mode=0 --eps_end=0.1 --cen_explore --h_explore --sample_epi --dynamic_h --eps_l_d --save_dir=warehouse --seed=0 --run_id=0
  ```

## Parallel-MacDec-MADDRQN
This approach differs **MacDec-MADDRQN** in the way that two parallel environments are involved with agents respectively performing centralized exploration (cen-e-greedy) and decentralized exploration (dec-e-greedy) in each. 

<p align="center">
  <img src="https://github.com/yuchen-x/gifs/blob/master/Parallel-MacDec-MADDRQN.png" width="70%">
</p>

The centralized Q-net is first trained purely using the centralized experiences, while each decentralized Q-net is then optimized using the above new double Q-update rule. 

- Training in the warehouse domain
  ```
  ma_dec_cen_hddrqn_sep.py --env_name=OSD_S_4 --env_terminate_step=150 --batch_size=16 --dec_rnn_h_size=64 --cen_rnn_h_size=64 --train_freq=30 --total_epi=40000 --replay_buffer_size=1000 --eps_l_d_steps=6000 --l_rate=0.0006 --discount=1.0 --start_train=2 --l_mode=0 --eps_end=0.1 --h_explore --sample_epi --dynamic_h --eps_l_d --save_dir=warehouse_parallel --seed=0 --run_id=0
  ```
## Dec-HDDRQN with Mac-CERTs and Cen-DDRQN with Mac-JERTs

These two methods are respectively the pure decentralized learning framework and the pure centralized learning framework for macro-action-based domains, proposed in our [CoRL2019 paper](https://drive.google.com/file/d/1R5bh7Hqs_Dhzz7FMmPP8TmMmk_IppcWL/view)


## Demo Videos
Please check our [YouTube channel](https://www.youtube.com/channel/UCQxF16jC0cO8uIWrsbGOmGg/) for the entire real robots videos.

## Paper Citation
If you used this code for your reasearch or found it helpful, please consider citing the following paper:
```
@InProceedings{xiao_corl_2019,
    author = "Xiao, Yuchen and Hoffman, Joshua and Amato, Christopher",
    title = "Macro-Action-Based Deep Multi-Agent Reinforcement Learning",
    booktitle = "3rd Annual Conference on Robot Learning",
    year = "2019"
}

@InProceedings{xiao_icra_2020,
    author = "Xiao, Yuchen and Hoffman, Joshua and Xia, Tian and Amato, Christopher",
    title = "Learning Multi-Robot Decentralized Macro-Action-Based Policies via a Centralized Q-Net",
    booktitle = "Proceedings of the International Conference on Robotics and Automation",
    year = "2020"
}
```
