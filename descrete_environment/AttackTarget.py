import math
import gym
from gym import spaces
import numpy as np
import time


class AttackTargetEnv(gym.Env):
    """
    Description:
        An agent steps toward to the target
    Observation:
        Type: Box(2)
        Num Observation    Min    Max
        0   Agent's pos_x  -50    50
        1   Agent's pos_y  -50    50
        2   Target's pos_x -50    50
        3   Target's pos_y -50    50 
        4   absolute distance in X 0 100
        5   absolute distance in Y 0 100
        6   distance        0     140
    Actions:
        Type: Discrete(4)
        Num Action        
        0   Velocity is 1 in X  
        1   Velocity is -1 in X  
        2   Velocity is 1 in Y  
        3   Velocity is -1 in Y
    Reward: 
        If Agent is closer to the target, the reward is bigger.
        Simplicilty, set reward is -np.sqrt((X1-X0)**2+(Y1-Y0)**2)/100
    Starting State:
        Target is randomly located in [X0, Y0], where X0 and Y0 range in [-50, 50]
    Episode Termination:
        Agent achieves the target, which means reward is bigger than 
    """
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 2
    }

    def __init__(self):

        self.high_pos = np.array([50, 50])
        self.low_pos = -self.high_pos
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(
            self.low_pos, self.high_pos, dtype=np.float32)

        self.viewer = None
        self.state = None


    def step(self, action):
        assert self.action_space.contains(
            action), "action: {0}, type: {1} is invalid".format(action, type(action))
        if action == 0:
            self.state += np.array([1, 0])
        elif action == 1:
            self.state += np.array([-1, 0])
        elif action == 2:
            self.state += np.array([0, 1])
        elif action == 3:
            self.state += np.array([0, -1])
        elif action == 4:
            self.state += 0
        # self.state = np.clip(
        #     self.state, self.observation_space.low, self.observation_space.high)

        # # reward style #1
        # X1, Y1 = self.state
        # X0, Y0 = self.target_pos
        # reward = -np.sqrt((X1 - X0)**2 + (Y1 - Y0)**2) / 100
        # # print("Agent: {0}, Target: {1}, reward: {2}".format(self.state, self.target_pos, reward))
        # if np.abs(X1 - X0) + np.abs(Y1 - Y0) < 5:
        #     done = True
        # else:
        #     done = False

        # # reward style #2
        # X1, Y1 = self.state
        # X0, Y0 = self.target_pos
        # done = (np.abs(X1-X0)+np.abs(Y1-Y0)<=1)
        # if not done:
        #     reward = -0.1
        # else:
        #     reward = 10


        # #reward style #3
        # X1, Y1 = self.state
        # X0, Y0 = self.target_pos
        # done = (np.abs(X1-X0)+np.abs(Y1-Y0)<=1) 
        # if not done:
        #     reward = -0.1
        # else:
        #     reward = 10

        # #reward style #4
        # X1, Y1 = self.state
        # X0, Y0 = self.target_pos
        # reward = -np.sqrt((X1-X0)**2+(Y1-Y0)**2)/20
        # done = np.abs(X1-X0)+np.abs(Y1-Y0)<= 1
        # if done:
        #     reward += 20
        # else:
        #     reward -= 0.05

        #reward style #5
        self.state = np.clip(self.state, -50, 50)

        X1, Y1 = self.state
        X0, Y0 = self.target_pos
        dist = np.sqrt((X1-X0)**2+(Y1-Y0)**2)
        reward = -dist/20.
        if dist < 15.:
            done = True
            reward += 20
        else:
            done = False
            reward -= 0.05

        


        # self.state = np.concatenate(X1, Y1, X0, Y0, abs(X1-X0), abs(Y1-Y0), np.sqrt((X1-X0)**2+(Y1-Y0)**2)/20)   

        # print(self.state, action, reward, done)
        return self.state, reward, done, {}

    def reset(self):
        # random set tartget pos in range (-50, 50)
        self.target_pos = np.array([0., 0.])
        # random set agent pos in range (-50, 50)
        self.state = np.array([30., -20.])
        
        

        return self.state

    def render(self, mode='human'):
        screen_width = self.high_pos[0] - self.low_pos[0]
        screen_height = self.high_pos[1] - self.low_pos[1]
        cart_origin = np.zeros(2) - self.low_pos

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            self.draw_agent = rendering.make_circle(5)
            self.draw_agent.set_color(0.5, 0.5, 0.8)
            self.trans_agent = rendering.Transform()
            self.draw_agent.add_attr(self.trans_agent)
            self.viewer.add_geom(self.draw_agent)

            draw_target = rendering.make_circle(5)
            draw_target.set_color(1, 1, 0)
            cart_target_x = self.target_pos[0] + cart_origin[0]
            cart_target_y = self.target_pos[1] + cart_origin[1]
            trans_target = rendering.Transform(
                translation=(cart_target_x, cart_target_y))
            draw_target.add_attr(trans_target)
            self.viewer.add_geom(draw_target)

        if self.state is None:
            return None

        cart_agent_x = self.state[0] + cart_origin[0]
        cart_agent_y = self.state[1] + cart_origin[1]
        self.trans_agent.set_translation(cart_agent_x, cart_agent_y)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

# if __name__ == "__main__":
#     env = AttackTargetEnv()
#     state = env.reset()
#     while True:
#         env.render()
#         action = env.action_space.sample()
#         next_state, reward, done, _ = env.step(action)
#         state = next_state
#         # print("reward is {}".format(reward))
#         # print("Agent state is {}".format(state))
#         # print("Target state is {}".format(env.target_pos))
#         # print("*"*20)
#         if done:
#             print("Agent has achieved target")
#             print("Agent pos: {0}, target pos: {1}".format(
#                 state, env.target_pos))
#             time.sleep(5)
#             break
