#!/usr/bin/python

import gym
import numpy as np
import IPython

from gym import spaces
from .box_pushing_MA_core import astar_Agent_, Box, BeliefWayPoint, MacroAction

DIRECTION = [(0,1), (1,0), (0,-1), (-1,0)]

class BoxPushing_harder(gym.Env):

    """
       Box Pushing Domain Description
       ------------------------------

       1) Agent is allowed to push any small box;
       2) Agent is allowed to go to any one of the two waypoints to push the big box;
       3) Big box is only able to be pushed when the two agents locating in the two waypoints seperately."""

    metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second' : 50
            }

    def __init__(self, grid_dim, terminate_step=100, random_init=False, *args, **kwargs):

        """
        Parameters
        ----------
        gird_dim : tuple(int, int)
            The size of the grid world.
        terminate_step : int
            The maximal steps per episode.
        random_init : bool
            Whether to randomly initialize agents' positions.
        """

        self.n_agent = 2

        self.action_space = spaces.Discrete(8)
        self.observation_space = spaces.MultiBinary(5)

        self.xlen, self.ylen = grid_dim

        # create waypoint for pushing box
        if self.ylen >= 8.0:
            BWP0 = BeliefWayPoint('Small_box_0_spot', 0, 1.5, self.ylen/2-0.5)
            BWP1 = BeliefWayPoint('Small_box_1_spot', 1, self.ylen-1.5, self.ylen/2-0.5)
            BWP2_0 = BeliefWayPoint('Big_box_spot_0', 2, self.ylen/2.0-0.5, self.ylen/2-0.5)
            BWP2_1 = BeliefWayPoint('Big_box_spot_1', 3, self.ylen/2.0+0.5, self.ylen/2-0.5)
        elif self.ylen == 6.0:
            BWP0 = BeliefWayPoint('Small_box_0_spot', 0, 0.5, self.ylen/2-0.5)
            BWP1 = BeliefWayPoint('Small_box_1_spot', 1, 5.5, self.ylen/2-0.5)
            BWP2_0 = BeliefWayPoint('Big_box_spot_0', 2, 2.5, self.ylen/2-0.5)
            BWP2_1 = BeliefWayPoint('Big_box_spot_1', 3, 3.5, self.ylen/2-0.5)
        else:
            BWP0 = BeliefWayPoint('Small_box_0_spot', 0, 0.5, self.ylen/2-0.5)
            BWP1 = BeliefWayPoint('Small_box_1_spot', 1, 3.5, self.ylen/2-0.5)
            BWP2_0 = BeliefWayPoint('Big_box_spot_0', 2, 1.5, self.ylen/2-0.5)
            BWP2_1 = BeliefWayPoint('Big_box_spot_1', 3, 2.5, self.ylen/2-0.5)

        self.BWPs = [BWP0, BWP1, BWP2_0, BWP2_1]

        # def macro-actions
        MA0 = MacroAction('GT_SB0', 0, ma_bwpterm=0)
        MA1 = MacroAction('GT_SB1', 1, ma_bwpterm=1)
        MA2 = MacroAction('GT_BB0', 2, ma_bwpterm=2)
        MA3 = MacroAction('GT_BB1', 3, ma_bwpterm=3)
        MA4 = MacroAction('Push', 4)
        MA5 = MacroAction('T_L', 5, expected_t_cost=1)
        MA6 = MacroAction('T_R', 6, expected_t_cost=1)
        MA7 = MacroAction('Stay', 7, expected_t_cost=1)

        self.MAs = [MA0, MA1, MA2, MA3, MA4, MA5, MA6, MA7]

        self.random_init = random_init
        self.createAgents()
        self.createBoxes()

        self.terminate_step = terminate_step
        self.pushing_big_box = False

        self.viewer = None

        self.single_small_box = 0.0
        self.both_small_box = 0.0
        self.big_box = 0.0

    @property
    def obs_size(self):
        return [self.observation_space.n] *2

    @property
    def n_action(self):
        return [a.n for a in self.action_spaces]

    def action_space_sample(self, i):
        return np.random.randint(self.action_spaces[i].n)

    @property
    def action_spaces(self):
        return [self.action_space] * 2

    def createAgents(self):
        if self.random_init:
            init_ori = np.random.randint(4,size=2)
            init_xs = np.random.randint(self.xlen,size=2) + 0.5
            init_ys = np.random.randint(int(self.ylen/2-1),size=2) + 0.5
            A0 = astar_Agent_(0, init_xs[0], init_ys[0], init_ori[0], self.BWPs, self.MAs, (self.xlen, self.ylen))
            A1 = astar_Agent_(1, init_xs[1], init_ys[1], init_ori[1], self.BWPs, self.MAs, (self.xlen, self.ylen))
        else:
            if self.ylen >= 8.0:
                A0 = astar_Agent_(0, 1.5, 1.5, 1, self.BWPs, self.MAs, (self.xlen, self.ylen))
                A1 = astar_Agent_(1, self.ylen-1.5, 1.5, 3, self.BWPs, self.MAs, (self.xlen, self.ylen))
            elif self.ylen == 6.0:
                A0 = astar_Agent_(0, 0.5, 1.5, 1, self.BWPs, self.MAs, (self.xlen, self.ylen))
                A1 = astar_Agent_(1, 5.5, 1.5, 3, self.BWPs, self.MAs, (self.xlen, self.ylen))
            else:
                A0 = astar_Agent_(0, 0.5, 0.5, 1, self.BWPs, self.MAs, (self.xlen, self.ylen))
                A1 = astar_Agent_(1, 3.5, 0.5, 3, self.BWPs, self.MAs, (self.xlen, self.ylen))

        self.agents = [A0, A1]

    def createBoxes(self):
        if self.ylen >= 8.0:
            SB_0 = Box(0, 1.5, (self.ylen/2+0.5), 1.0, 1.0) 
            SB_1 = Box(1, self.ylen-1.5, (self.ylen/2+0.5), 1.0, 1.0) 
            BB_2 = Box(2, self.ylen/2.0, (self.ylen/2+0.5), 1.0, 2.0) 
        elif self.ylen == 6.0:
            SB_0 = Box(0, 0.5, (self.ylen/2+0.5), 1.0, 1.0) 
            SB_1 = Box(1, 5.5, (self.ylen/2+0.5), 1.0, 1.0) 
            BB_2 = Box(2, 3.0, (self.ylen/2+0.5), 1.0, 2.0) 
        else:
            SB_0 = Box(0, 0.5, (self.ylen/2+0.5), 1.0, 1.0) 
            SB_1 = Box(1, 3.5, (self.ylen/2+0.5), 1.0, 1.0) 
            BB_2 = Box(2, 2.0, (self.ylen/2+0.5), 1.0, 2.0) 

        self.boxes = [SB_0, SB_1, BB_2]

    def reset(self, debug=False):
        self.createAgents()
        self.createBoxes()
        self.t = 0
        self.count_step = 0
        self.pushing_big_box = False

        if debug:
            self.render()

        return self._getobs()

    def step(self, actions, debug=False):
        """
        Parameters
        ----------
        actions : int | List[..]
           The discrete macro-action index for one or more agents. 

        Returns
        -------
        cur_actions : int | List[..]
            The discrete macro-action indice which agents are executing in the current step.
        observations : ndarry | List[..]
            A list of  each agent's macor-observation.
        rewards : float | List[..]
            A global shared reward for each agent.
        done : bool
            Whether the current episode is over or not.
        cur_action_done : binary (1/0) | List[..]
            Whether each agent's curent macro-action is done or not.
        """

        rewards = -0.1
        terminate = 0

        cur_actions = []
        cur_actions_done = []

        self.count_step += 1

        # check if agents reach the pre-condition of pushing a big box 
        if (actions[0] == 4) and (actions[1] == 4) and self.agents[0].cur_action_done and self.agents[1].cur_action_done and \
                self.agents[0].ori == 0 and self.agents[1].ori == 0 and \
                ((self.agents[0].cur_BWP is not None and self.agents[0].cur_BWP.idx == 2 and self.agents[1].cur_BWP is not None and self.agents[1].cur_BWP.idx == 3) or \
                (self.agents[0].cur_BWP is not None and self.agents[0].cur_BWP.idx == 3 and (self.agents[1].cur_BWP is not None) and self.agents[1].cur_BWP.idx == 2)):
                    self.pushing_big_box = True

        if not self.pushing_big_box:
            for idx, agent in enumerate(self.agents):
                # if agent's previous macro-action is not done, continue running the current macro-action
                if not agent.cur_action_done:
                    reward = agent.step(agent.cur_action.idx, self.boxes)
                    cur_actions.append(agent.cur_action.idx)
                # if agent's previous macro-action is done, run the new input macro-action
                else:
                    reward = agent.step(actions[idx], self.boxes)
                    cur_actions.append(actions[idx])
                cur_actions_done.append(1 if agent.cur_action_done else 0)
                rewards += reward
        else:
            for agent in self.agents:
                agent.cur_action = agent.macro_actions[4]
                agent.cur_action_done = False
                agent.cur_action_time_left = -1.0
                agent.ycoord += 1.0
            self.boxes[2].ycoord += 1.0
            if self.boxes[2].ycoord == self.ylen-0.5:
                for agent in self.agents:
                    agent.cur_action_done = True
            cur_actions_done.append(1 if self.agents[0].cur_action_done else 0)
            cur_actions_done.append(1 if self.agents[1].cur_action_done else 0)
            cur_actions.append(4)
            cur_actions.append(4)

        # compute reward
        reward = 0.0
        small_box = 0.0
        for idx, box in enumerate(self.boxes):
            if box.ycoord == self.ylen-0.5:
                terminate = 1
                reward = reward + 10 if idx < 2 else reward + 100
                if idx == 2:
                    self.big_box += 1.0
                else:
                    small_box += 1.0

        if small_box == 1.0:
            self.single_small_box += 1.0
        elif small_box == 2.0:
            self.both_small_box += 1.0

        rewards += reward

        if debug:
            self.render()
            print(" ")
            print("Actions list:")
            print("Agent_0 \t action \t\t{}".format(self.agents[0].cur_action.name))
            print("        \t action_t_left \t\t{}".format(self.agents[0].cur_action_time_left))
            print("        \t action_done \t\t{}".format(self.agents[0].cur_action_done))
            print(" ")
            print("Agent_1 \t action \t\t{}".format(self.agents[1].cur_action.name))
            print("        \t action_t_left \t\t{}".format(self.agents[1].cur_action_time_left))
            print("        \t action_done \t\t{}".format(self.agents[1].cur_action_done))

        observations = self._getobs(debug)

        return cur_actions, observations, [rewards]*self.n_agent, terminate or self.count_step >= self.terminate_step, cur_actions_done

    def _getobs(self, debug=False):

        if self.t == 0:
            obs = np.zeros(self.observation_space.n)
            obs[2] = 1.0
            self.t = 1
            observations = [obs, obs]
            self.old_observations = observations
            
            return observations

        if debug:
            print("")
            print("Observations list:")

        observations = []
        for idx, agent in enumerate (self.agents):
            if not agent.cur_action_done:
                observations.append(self.old_observations[idx])
                if debug:
                    print("Agent_" + str(idx) + " \t small_box  \t\t{}".format(self.old_observations[idx][0]))
                    print("          " + " \t large_box \t\t{}".format(self.old_observations[idx][1]))
                    print("          " + " \t empty \t\t\t{}".format(self.old_observations[idx][2]))
                    print("          " + " \t wall \t\t\t{}".format(self.old_observations[idx][3]))
                    print("          " + " \t teammate \t\t{}".format(self.old_observations[idx][4]))
                    print("")
                continue

            obs = np.zeros(self.observation_space.n)

            # assume empty front
            obs[2] = 1.0

            # observe small box
            for box in self.boxes[0:2]:
                if box.xcoord == agent.xcoord + DIRECTION[agent.ori][0] and \
                        box.ycoord == agent.ycoord + DIRECTION[agent.ori][1]:
                            obs[0] = 1.0
                            obs[2] = 0.0
            # observe large box
            if (self.boxes[2].xcoord+0.5 == agent.xcoord + DIRECTION[agent.ori][0] or \
                    self.boxes[2].xcoord-0.5 == agent.xcoord + DIRECTION[agent.ori][0]) and \
                    self.boxes[2].ycoord  == agent.ycoord + DIRECTION[agent.ori][1]:
                        obs[1] = 1.0
                        obs[2] = 0.0
           
            # observe wall
            if agent.xcoord + DIRECTION[agent.ori][0] > self.xlen or \
                    agent.xcoord + DIRECTION[agent.ori][0] < 0.0 or \
                    agent.ycoord + DIRECTION[agent.ori][1] > self.ylen or \
                    agent.ycoord + DIRECTION[agent.ori][1] < 0.0:
                        obs[3] = 1.0
                        obs[2] = 0.0
            
            # observe agent
            if idx == 0:
                teamate_idx = 1
            else:
                teamate_idx = 0
            if (agent.xcoord + DIRECTION[agent.ori][0] == self.agents[teamate_idx].xcoord) and \
                    (agent.ycoord + DIRECTION[agent.ori][1] == self.agents[teamate_idx].ycoord):
                obs[4] = 1.0
                obs[2] = 0.0

            if debug:
                    print("Agent_" + str(idx) + " \t small_box  \t\t{}".format(obs[0]))
                    print("          " + " \t large_box \t\t{}".format(obs[1]))
                    print("          " + " \t empty \t\t\t{}".format(obs[2]))
                    print("          " + " \t wall \t\t\t{}".format(obs[3]))
                    print("          " + " \t teammate \t\t{}".format(obs[4]))
                    print("")

            observations.append(obs)

        self.old_observations = observations

        return observations

    def render(self, mode='human'):
        
        screen_width = 8*100
        screen_height = 8*100

        scale = 8 / self.ylen

        agent_size = 30.0
        agent_in_size = 25.0
        agent_clrs = [((0.15,0.65,0.15), (0.0,0.8,0.4)), ((0.15,0.15,0.65), (0.0, 0.4,0.8))]

        small_box_size = 85.0
        small_box_clrs = [(0.43,0.28,0.02), (0.67,0.43,0.02)]
        small_box_in_size = 75.0

        big_box_l = 185.0
        big_box_in_l = 175.0
        big_box_w = 85.0
        big_box_in_w = 75.0
        big_box_clrs = [(0.43,0.28,0.02), (0.67,0.43,0.02)]
        
        if self.viewer is None:
            from rlmamr.my_env import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)

            #-------------------draw line-----------------
            for l in range(1, self.ylen):
                line = rendering.Line((0.0, l*100*scale), (screen_width, l*100*scale))
                line.linewidth.stroke = 4
                line.set_color(0.50, 0.50, 0.50)
                self.viewer.add_geom(line)

            for l in range(1, self.ylen):
                line = rendering.Line((l*100*scale, 0.0), (l*100*scale, screen_width))
                line.linewidth.stroke = 4
                line.set_color(0.50, 0.50, 0.50)
                self.viewer.add_geom(line)

            line = rendering.Line((0.0, 0.0), (0.0, screen_width))
            line.linewidth.stroke = 60
            line.set_color(0.0, 0.0, 0.0)
            self.viewer.add_geom(line)

            line = rendering.Line((0.0, 0.0), (screen_width, 0.0))
            line.linewidth.stroke = 60
            line.set_color(0.0, 0.0, 0.0)
            self.viewer.add_geom(line)

            line = rendering.Line((screen_width, 0.0), (screen_width, screen_width))
            line.linewidth.stroke = 60
            line.set_color(0.0, 0.0, 0.0)
            self.viewer.add_geom(line)

            line = rendering.Line((0.0, screen_width), (screen_width, screen_width))
            line.linewidth.stroke = 60
            line.set_color(0.0, 0.0, 0.0)
            self.viewer.add_geom(line)

            #-------------------draw waypoints
            for i in range(len(self.BWPs)):
                BWP = rendering.make_circle(radius=8*scale)
                BWP.set_color(178.0/255.0, 34.0/255.0, 34.0/255.0)
                BWPtrans = rendering.Transform(translation=(self.BWPs[i].xcoord*100*scale, self.BWPs[i].ycoord*100*scale))
                BWP.add_attr(BWPtrans)
                self.viewer.add_geom(BWP)

            #-------------------draw goal
            goal = rendering.FilledPolygon([(-(screen_width-8)/2.0,(-50+2)*scale), (-(screen_width-8)/2.0,(50-2)*scale), ((screen_width-8)/2.0,(50-2)*scale), ((screen_width-8)/2.0,-(50-2)*scale)])
            goal.set_color(1.0,1.0,0.0)
            goal_trans = rendering.Transform(translation=(screen_width/2.0,(self.ylen-0.5)*100*scale))
            goal.add_attr(goal_trans)
            self.viewer.add_geom(goal)

            #-------------------draw small box
            small_box_0 = rendering.FilledPolygon([(-small_box_size/2.0*scale,-small_box_size/2.0*scale), (-small_box_size/2.0*scale,small_box_size/2.0*scale), (small_box_size/2.0*scale,small_box_size/2.0*scale), (small_box_size/2.0*scale,-small_box_size/2.0*scale)])
            small_box_0.set_color(*small_box_clrs[0])
            self.small_box_0_trans = rendering.Transform(translation=(self.boxes[0].xcoord*100*scale, self.boxes[0].ycoord*100*scale))
            small_box_0.add_attr(self.small_box_0_trans)
            self.viewer.add_geom(small_box_0)

            small_box_0_in = rendering.FilledPolygon([(-small_box_in_size/2.0*scale,-small_box_in_size/2.0*scale), (-small_box_in_size/2.0*scale,small_box_in_size/2.0*scale), (small_box_in_size/2.0*scale,small_box_in_size/2.0*scale), (small_box_in_size/2.0*scale,-small_box_in_size/2.0*scale)])
            small_box_0_in.set_color(*small_box_clrs[1])
            self.small_box_0_in_trans = rendering.Transform(translation=(self.boxes[0].xcoord*100*scale, self.boxes[0].ycoord*100*scale))
            small_box_0_in.add_attr(self.small_box_0_in_trans)
            self.viewer.add_geom(small_box_0_in)
            
            small_box_1 = rendering.FilledPolygon([(-small_box_size/2.0*scale,-small_box_size/2.0*scale), (-small_box_size/2.0*scale,small_box_size/2.0*scale), (small_box_size/2.0*scale,small_box_size/2.0*scale), (small_box_size/2.0*scale,-small_box_size/2.0*scale)])
            small_box_1.set_color(*small_box_clrs[0])
            self.small_box_1_trans = rendering.Transform(translation=(self.boxes[1].xcoord*100*scale, self.boxes[1].ycoord*100*scale))
            small_box_1.add_attr(self.small_box_1_trans)
            self.viewer.add_geom(small_box_1)

            small_box_1_in = rendering.FilledPolygon([(-small_box_in_size/2.0*scale,-small_box_in_size/2.0*scale), (-small_box_in_size/2.0*scale,small_box_in_size/2.0*scale), (small_box_in_size/2.0*scale,small_box_in_size/2.0*scale), (small_box_in_size/2.0*scale,-small_box_in_size/2.0*scale)])
            small_box_1_in.set_color(*small_box_clrs[1])
            self.small_box_1_in_trans = rendering.Transform(translation=(self.boxes[1].xcoord*100*scale, self.boxes[1].ycoord*100*scale))
            small_box_1_in.add_attr(self.small_box_1_in_trans)
            self.viewer.add_geom(small_box_1_in)

            # -------------------draw big box
            big_box_2 = rendering.FilledPolygon([(-big_box_l/2.0*scale,-big_box_w/2.0*scale), (-big_box_l/2.0*scale,big_box_w/2.0*scale), (big_box_l/2.0*scale,big_box_w/2.0*scale), (big_box_l/2.0*scale,-big_box_w/2.0*scale)])
            big_box_2.set_color(*big_box_clrs[0])
            self.big_box_2_trans = rendering.Transform(translation=(self.boxes[2].xcoord*100*scale, self.boxes[2].ycoord*100*scale))
            big_box_2.add_attr(self.big_box_2_trans)
            self.viewer.add_geom(big_box_2)

            big_box_2_in = rendering.FilledPolygon([(-big_box_in_l/2.0*scale,-big_box_in_w/2.0*scale), (-big_box_in_l/2.0*scale,big_box_in_w/2.0*scale), (big_box_in_l/2.0*scale,big_box_in_w/2.0*scale), (big_box_in_l/2.0*scale,-big_box_in_w/2.0*scale)])
            big_box_2_in.set_color(*big_box_clrs[1])
            self.big_box_2_in_trans = rendering.Transform(translation=(self.boxes[2].xcoord*100*scale, self.boxes[2].ycoord*100*scale))
            big_box_2_in.add_attr(self.big_box_2_in_trans)
            self.viewer.add_geom(big_box_2_in)

            #-------------------draw agent
            agent_0 = rendering.make_circle(radius=agent_size*scale)
            agent_0.set_color(*agent_clrs[0][0])
            self.agent_0_trans = rendering.Transform(translation=(self.agents[0].xcoord*100*scale, self.agents[0].ycoord*100*scale))
            agent_0.add_attr(self.agent_0_trans)
            self.viewer.add_geom(agent_0)

            agent_0_in = rendering.make_circle(radius=agent_in_size*scale)
            agent_0_in.set_color(*agent_clrs[0][1])
            self.agent_0_in_trans = rendering.Transform(translation=(self.agents[0].xcoord*100*scale, self.agents[0].ycoord*100*scale))
            agent_0_in.add_attr(self.agent_0_in_trans)
            self.viewer.add_geom(agent_0_in)
            
            agent_1 = rendering.make_circle(radius=agent_size*scale)
            agent_1.set_color(*agent_clrs[1][0])
            self.agent_1_trans = rendering.Transform(translation=(self.agents[1].xcoord*100*scale, self.agents[1].ycoord*100*scale))
            agent_1.add_attr(self.agent_1_trans)
            self.viewer.add_geom(agent_1)

            agent_1_in = rendering.make_circle(radius=agent_in_size*scale)
            agent_1_in.set_color(*agent_clrs[1][1])
            self.agent_1_in_trans = rendering.Transform(translation=(self.agents[1].xcoord*100*scale, self.agents[1].ycoord*100*scale))
            agent_1_in.add_attr(self.agent_1_in_trans)
            self.viewer.add_geom(agent_1_in)

            #-------------------draw agent sensor
            sensor_size = 20.0
            sensor_in_size = 14.0
            sensor_clrs = ((0.65,0.15,0.15), (1.0, 0.2,0.2))

            sensor_0 = rendering.FilledPolygon([(-sensor_size/2.0*scale,-sensor_size/2.0*scale), (-sensor_size/2.0*scale,sensor_size/2.0*scale), (sensor_size/2.0*scale,sensor_size/2.0*scale), (sensor_size/2.0*scale,-sensor_size/2.0*scale)])
            sensor_0.set_color(*sensor_clrs[0])
            self.sensor_0_trans = rendering.Transform(translation=(self.agents[0].xcoord*100*scale+(agent_size)*DIRECTION[self.agents[0].ori][0]*scale, 
                                                                   self.agents[0].ycoord*100*scale+(agent_size)*DIRECTION[self.agents[0].ori][1]*scale))
            sensor_0.add_attr(self.sensor_0_trans)
            self.viewer.add_geom(sensor_0)

            sensor_0_in = rendering.FilledPolygon([(-sensor_in_size/2.0*scale,-sensor_in_size/2.0*scale), (-sensor_in_size/2.0*scale,sensor_in_size/2.0*scale), (sensor_in_size/2.0*scale,sensor_in_size/2.0*scale), (sensor_in_size/2.0*scale,-sensor_in_size/2.0*scale)])
            sensor_0_in.set_color(*sensor_clrs[1])
            self.sensor_0_in_trans = rendering.Transform(translation=(self.agents[0].xcoord*100*scale+(agent_size)*DIRECTION[self.agents[0].ori][0]*scale, 
                                                                   self.agents[0].ycoord*100*scale+(agent_size)*DIRECTION[self.agents[0].ori][1]*scale))
            sensor_0_in.add_attr(self.sensor_0_in_trans)
            self.viewer.add_geom(sensor_0_in)
            
            sensor_1 = rendering.FilledPolygon([(-sensor_size/2.0*scale,-sensor_size/2.0*scale), (-sensor_size/2.0*scale,sensor_size/2.0*scale), (sensor_size/2.0*scale,sensor_size/2.0*scale), (sensor_size/2.0*scale,-sensor_size/2.0*scale)])
            sensor_1.set_color(*sensor_clrs[0])
            self.sensor_1_trans = rendering.Transform(translation=(self.agents[1].xcoord*100*scale+(agent_size)*DIRECTION[self.agents[1].ori][0]*scale, 
                                                                   self.agents[1].ycoord*100*scale+(agent_size)*DIRECTION[self.agents[1].ori][1]*scale))
            sensor_1.add_attr(self.sensor_1_trans)
            self.viewer.add_geom(sensor_1)

            sensor_1_in = rendering.FilledPolygon([(-sensor_in_size/2.0*scale,-sensor_in_size/2.0*scale), (-sensor_in_size/2.0*scale,sensor_in_size/2.0*scale), (sensor_in_size/2.0*scale,sensor_in_size/2.0*scale), (sensor_in_size/2.0*scale,-sensor_in_size/2.0*scale)])
            sensor_1_in.set_color(*sensor_clrs[1])
            self.sensor_1_in_trans = rendering.Transform(translation=(self.agents[1].xcoord*100*scale+(agent_size)*DIRECTION[self.agents[1].ori][0]*scale, 
                                                                   self.agents[1].ycoord*100*scale+(agent_size)*DIRECTION[self.agents[1].ori][1]*scale))
            sensor_1_in.add_attr(self.sensor_1_in_trans)
            self.viewer.add_geom(sensor_1_in)

        self.small_box_0_trans.set_translation(self.boxes[0].xcoord*100*scale, self.boxes[0].ycoord*100*scale)
        self.small_box_0_in_trans.set_translation(self.boxes[0].xcoord*100*scale, self.boxes[0].ycoord*100*scale)
        self.small_box_1_trans.set_translation(self.boxes[1].xcoord*100*scale, self.boxes[1].ycoord*100*scale)
        self.small_box_1_in_trans.set_translation(self.boxes[1].xcoord*100*scale, self.boxes[1].ycoord*100*scale)
        self.big_box_2_trans.set_translation(self.boxes[2].xcoord*100*scale, self.boxes[2].ycoord*100*scale)
        self.big_box_2_in_trans.set_translation(self.boxes[2].xcoord*100*scale, self.boxes[2].ycoord*100*scale)
        
        self.agent_0_trans.set_translation(self.agents[0].xcoord*100*scale, self.agents[0].ycoord*100*scale)
        self.agent_0_in_trans.set_translation(self.agents[0].xcoord*100*scale, self.agents[0].ycoord*100*scale)
        self.agent_1_trans.set_translation(self.agents[1].xcoord*100*scale, self.agents[1].ycoord*100*scale)
        self.agent_1_in_trans.set_translation(self.agents[1].xcoord*100*scale, self.agents[1].ycoord*100*scale)

        if self.agents[0].cur_action_done or self.agents[0].cur_action.idx <= 4: #action hardcode
            self.sensor_0_trans.set_translation(self.agents[0].xcoord*100*scale+(agent_size)*DIRECTION[self.agents[0].ori][0]*scale, 
                                                self.agents[0].ycoord*100*scale+(agent_size)*DIRECTION[self.agents[0].ori][1]*scale)
            self.sensor_0_trans.set_rotation(0.0)
            self.sensor_0_in_trans.set_translation(self.agents[0].xcoord*100*scale+(agent_size)*DIRECTION[self.agents[0].ori][0]*scale, 
                                                self.agents[0].ycoord*100*scale+(agent_size)*DIRECTION[self.agents[0].ori][1]*scale)
            self.sensor_0_in_trans.set_rotation(0.0)

        else:
            x = self.agents[0].xcoord*100*scale + (agent_size)*self.agents[0].direct[0]*scale
            y = self.agents[0].ycoord*100*scale + (agent_size)*self.agents[0].direct[1]*scale
            angle = np.arccos(np.dot(self.agents[0].direct,np.array([1.0,0.0])))
            angle = angle * -1.0 if self.agents[0].direct[1] < 0.0 else angle
            self.sensor_0_trans.set_translation(x, y)
            self.sensor_0_trans.set_rotation(angle)
            self.sensor_0_in_trans.set_translation(x, y)
            self.sensor_0_in_trans.set_rotation(angle)

        if self.agents[1].cur_action_done or self.agents[1].cur_action.idx <= 4: #action hardcode
            self.sensor_1_trans.set_translation(self.agents[1].xcoord*100*scale+(agent_size)*DIRECTION[self.agents[1].ori][0]*scale,
                                                self.agents[1].ycoord*100*scale+(agent_size)*DIRECTION[self.agents[1].ori][1]*scale)
            self.sensor_1_trans.set_rotation(0.0)
            self.sensor_1_in_trans.set_translation(self.agents[1].xcoord*100*scale+(agent_size)*DIRECTION[self.agents[1].ori][0]*scale,
                                                self.agents[1].ycoord*100*scale+(agent_size)*DIRECTION[self.agents[1].ori][1]*scale)
            self.sensor_1_in_trans.set_rotation(0.0)
        else:
            x = self.agents[1].xcoord*100*scale + (agent_size)*self.agents[1].direct[0]*scale
            y = self.agents[1].ycoord*100*scale + (agent_size)*self.agents[1].direct[1]*scale
            angle = np.arccos(np.dot(self.agents[1].direct,np.array([1.0,0.0])))
            angle = angle * -1.0 if self.agents[1].direct[1] < 0.0 else angle
            self.sensor_1_trans.set_translation(x, y)
            self.sensor_1_trans.set_rotation(angle)
            self.sensor_1_in_trans.set_translation(x, y)
            self.sensor_1_in_trans.set_rotation(angle)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')
