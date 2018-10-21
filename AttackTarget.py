import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import time


HIGH_POS = 100.0
LOW_POS = 0
HIGH_VEL = 1.0
LOW_VEL = -1.0


class AttackTargetEnv(gym.Env):
    """
    Description:
        An agent steps toward to the target
    Observation:
        Type: Box(2)
        Num Observation    Min    Max
        0   Position in X  0      100
        1   Position in Y  0      100
    Actions:
        Type: Discrete(2)
        Num Action        
        0   Velocity is 1 in X  
        1   Velocity is -1 in X  
        2   Velocity is 1 in Y  
        3   Velocity is -1 in Y
    Reward: 
        If Agent is closer to the target, the reward is bigger.
        Simplicilty, set reward is -log(((PosX-X0)^2+(PosY-Y0)^2)/2)+4
    Starting State:
        Target is randomly located in [X0, Y0], where X0 and Y0 range in [0, 100]
    Episode Termination:
        Agent achieves the target, which means reward is bigger than 
    """
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self):

        high_pos = np.array([HIGH_POS, HIGH_POS])
        low_pos = np.array([LOW_POS, LOW_POS])
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low_pos, high_pos, dtype=np.float32)
        self.reward_threshold = 100000
        self.seed()
        self.viewer = None
        self.state = None
        self.index = 0

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(
            action), "action: {0}, type: {1} is invalid".format(action, type(action))
        state = self.state
        pos_x1_old = state[0]   #store the old X position
        pos_y1_old = state[1]   #store the old Y position
        state_alpha = {
            0: np.array([1, 0]), 
            1: np.array([-1, 0]),
            2: np.array([0, 1]), 
            3: np.array([0, -1])
        }

        next_state = state + state_alpha[action]
        # Edge detection
        # next_state = np.clip(next_state, LOW_POS, HIGH_POS)
        next_state[next_state > 100] = 100
        next_state[next_state < 0] = 0
        self.index += 1
        self.state = next_state
        pos_x1_new = self.state[0]  #store the new X position
        pos_y1_new = self.state[1]  #store the new Y position
        pos_x0 = self.target_pos[0]
        pos_y0 = self.target_pos[1]
        # reward = -np.log10(((pos_x1 - pos_x0)**2 +
        #                     (pos_y1 - pos_y0)**2) / 2) + 4
        
        # reward =1/np.sqrt((pos_x1-pos_x0)**2+(pos_y1-pos_y0)**2)
        reward_old = 1/(np.abs(pos_x1_old-pos_x0)+np.abs(pos_y1_old-pos_y0))
        reward_new = 1/(np.abs(pos_x1_new-pos_x0)+np.abs(pos_y1_new-pos_y0))
        reward = reward_old-reward_new
        if reward > self.reward_threshold:
            done = True
        else:
            done = False
        return self.state, reward, done, {}

    def reset(self):
        # random set tartget pos in range(0, 100)
        self.target_pos = np.random.rand(2) * 100
        # random set agent pos in range (0, 100)
        self.state = np.random.rand(2) * 100

        return self.state

    def render(self, mode='human'):
        screen_width = screen_height = 600
        scale = screen_width / (HIGH_POS - LOW_POS)

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            self.draw_agent = rendering.make_circle(30)
            self.draw_agent.set_color(0.5, 0.5, 0.8)
            self.trans_agent = rendering.Transform(translation=(100, 100))
            self.draw_agent.add_attr(self.trans_agent)
            self.viewer.add_geom(self.draw_agent)

            draw_target = rendering.make_circle(30)
            draw_target.set_color(1, 1, 0)
            cart_target_x = scale * self.target_pos[0]
            cart_target_y = scale * self.target_pos[1]
            trans_target = rendering.Transform(
                translation=(cart_target_x, cart_target_y))
            draw_target.add_attr(trans_target)
            self.viewer.add_geom(draw_target)

        if self.state is None:
            return None

        cart_agent_x = scale * self.state[0]
        cart_agent_y = scale * self.state[1]
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

