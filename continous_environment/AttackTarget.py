import gym
import numpy as np
from gym import spaces, logger
from gym.utils import seeding
import time


class AttackTarget(gym.Env):
    """
    Description:
        An agent steps toward to the target
    Observation:
        Type: Box(2)
        Num Observation    Min    Max
        0   Position in X  0      100
        1   Position in Y  0      100
    Actions:
        Type: Box(2)
        Num Action         Min Max       
        0   Velocity in X  -1   1
        1   Velocity in Y  -1   1

    Reward: 
        If Agent is closer to the target, the reward is bigger.
        Simplicilty, set reward is -sqrt((X1-X0)**2+(Y1-Y0)**2)
    Starting State:
        Target is randomly located in [X0, Y0], where X0 and Y0 range in [0, 100]
        Agent is randomly located in [X1, Y1], where X1 and Y1 range in [0, 100]
    Episode Termination:
        Agent achieves the target, which means reward is bigger than -3
        or meet the max step of each episode
    """
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self):
        self.max_vel = np.array([1, 1])  # max velocity in X and Y coordinates
        self.min_vel = np.array([-1, -1])
        self.max_pos = np.array([100, 100])
        self.min_pos = np.array([0, 0])

        self.action_space = spaces.Box(
            self.min_vel, self.max_vel, dtype=np.float32)
        self.observation_space = spaces.Box(
            self.min_pos, self.max_pos, dtype=np.float32)

        self.seed()
        self.viewer = None
        self.state = None

    def seed(self, seed=1):
        seeding.np_random(seed)
        np.random.seed(seed)

    def step(self, action):
        self.state += action
        self.state = np.clip(self.state, self.min_pos[0], self.max_pos[1])
        X1, Y1 = self.state[0], self.state[1]
        X0, Y0 = self.target_pos[0], self.target_pos[1]

        reward = -np.sqrt((X1 - X0)**2 + (Y1 - Y0)**2)/100
        if reward > -0.05:
            reward += 10
            done = True
        else:
            done = False

        return self.state, reward, done, {}

    def render(self, mode="human"):
        screen_width = screen_height = 400
        scale = screen_width / (self.max_pos[0] - self.min_pos[0])

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            self.draw_agent = rendering.make_circle(10)
            self.draw_agent.set_color(0.5, 0.5, 0.8)
            self.trans_agent = rendering.Transform(translation=(100, 100))
            self.draw_agent.add_attr(self.trans_agent)
            self.viewer.add_geom(self.draw_agent)

            draw_target = rendering.make_circle(10)
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

    def reset(self):
        self.target_pos = np.array([70., 80.])
        self.state = np.array([10.,10.])
        # self.state = np.random.rand(2) * 100
        return self.state

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def choose_action(self, state):
        action = np.random.rand(2)*2-1
        return action


MAX_EPISODES = 500
MAX_EP_STEPS = 200

if __name__ == "__main__":
    env = AttackTarget()
    state = env.reset()

    for episode in range(MAX_EPISODES):
        for step in range(MAX_EP_STEPS):
            env.render()
            action = env.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            state = next_state
            print(state)
            if done or step == MAX_EP_STEPS - 1:
                print("episode: {0}, step: {1}, agent_pos: {2}, reward: {3}".format(episode, step, state, reward))
                break
