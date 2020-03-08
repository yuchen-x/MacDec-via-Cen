# Learning Multi-Robot Decentralized Macro-Action-Based Policies via a Centralized Q-Net

In this [paper](https://arxiv.org/pdf/1909.08776.pdf), we first introduce a new macro-action-based decentralized multi-agent double deep recurrent Q-net (**MacDec-MADDRQN**) which adopts *centralized trainning with decentralized execution* by allowing each decentralized Q-net update to use a centralized Q-net for action selection. In order to balance centralized and decentralized exploration, a general version, called **Parallel-MacDec-MADDRQN**, is also proposed. 
