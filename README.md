# Learning Multi-Robot Decentralized Macro-Action-Based Policies via a Centralized Q-Net

In this [paper](https://arxiv.org/pdf/1909.08776.pdf), we first introduce a new macro-action-based decentralized multi-agent double deep recurrent Q-net (**MacDec-MADDRQN**) which adopts *centralized trainning with decentralized execution* by allowing each decentralized Q-net update to use a centralized Q-net for action selection. In order to balance centralized and decentralized exploration, a general version, called **Parallel-MacDec-MADDRQN**, is also proposed. The code in this repo is to implement these two algorithms. 

- The decentralized macro-action-based policies learned via **MacDec-MADDRQN** enable the agents to collaboratively push the big box for higher credits:

<p align="center">
  <img src="https://github.com/yuchen-x/gifs/blob/master/bpma10.gif" width="25%" hspace="40">
  <img src="https://github.com/yuchen-x/gifs/blob/master/bpma30.GIF" width="25%" hspace="40">
</p>

- A team of robots collaborate to bring the correct tools to a human at the right time by running the decentralized macro-action-based policies learned via **Parallel-MacDec-MADDRQN**:
<p align="center">
  <img src="https://github.com/yuchen-x/MacDec-via-Cen/blob/master/images/osd.GIF" width="50%">
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
