from AttackTarget_v1 import AttackTarget
from DDPG import DDPG
import matplotlib.pyplot as plt

MAX_EPISODES = 500
MAX_EP_STEPS = 500


if __name__ == "__main__":
    env = AttackTarget()

    

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_bound = env.action_space.high
    
    rl = DDPG(action_dim, state_dim, action_bound)

    steps = []


    for episode in range(MAX_EPISODES):
        state = env.reset()
        state_hold = 0
        for step in range(MAX_EP_STEPS):
            env.render()
            action = rl.choose_action(state)

            next_state, reward, done, _ = env.step(action)
            rl.store_transition(state, action, reward, next_state)
            
            if rl.memory_full:
                rl.learn()

            state = next_state
            
            if done or step == MAX_EP_STEPS - 1:
                print("episode: {0}, step: {1}, agent_pos: {2}, reward: {3}".format(episode, step, state, reward))
                steps.append(step)
                break
    plt.plot(steps)
    plt.show()
    rl.save()
