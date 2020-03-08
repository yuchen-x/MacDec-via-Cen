# Learning Multi-Robot Decentralized Macro-Action-Based Policies via a Centralized Q-Net

In this [paper](https://arxiv.org/pdf/1909.08776.pdf), we first introduce a new macro-action-based decentralized multi-agent double deep recurrent Q-net (**MacDec-MADDRQN**) which adopts *centralized trainning with decentralized execution* by allowing each decentralized Q-net update to use a centralized Q-net for action selection. In order to balance centralized and decentralized exploration, a general version, called **Parallel-MacDec-MADDRQN**, is also proposed. The code in this repo is to implement these two algorithms. 

The advantages and the practical nature of our methods are demonstrated by achieving near-centralized results in simulation and having real robots accomplish a warehouse tool delivery task in an efficient way.

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
