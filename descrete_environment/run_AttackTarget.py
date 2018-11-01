from AttackTarget import AttackTargetEnv

from DQN import DQN
import matplotlib.pyplot as plt

if __name__ == "__main__":
    env = AttackTargetEnv()
    agent = DQN(env)
    step_ls = []
    for episode in range(500):
        state = env.reset()
        for step in range(200):
            # env.render()
            action = agent.egreedy_action(state)
            
            next_state, reward, done, _ = env.step(action)
            agent.perceive(state, action, reward, next_state, done)
            state = next_state
            if done or step == 200-1:
                step_ls.append(step)
                print(episode, state, action, reward, done, step)    
                break

    plt.plot(step_ls)
    plt.show()