import gym
import tensorflow as tf
from DQN import DQN
from AttackTarget import AttackTargetEnv

EPISODE = 100
TRAIN_STEP = 1000
TEST_STEP = 1000

def main():
    env = AttackTargetEnv()
    agent = DQN(env)
    for episode in range(EPISODE):
        state = env.reset()
        # start train loop
        for step in range(TRAIN_STEP):
            env.render()
            action = agent.egreedy_action(state)
            next_state, reward, done, _ = env.step(action)
            print(state)
            print(action)
            print(next_state)
            print(reward)
            print(env.target_pos)
            print("step: {}".format(step))
            print("="*20)
            agent.perceive(state,action,reward, next_state, done)
            state = next_state  # update state
            if done:  # if meet terminational condition, end the for loop
                break
        # # start test loop, only test for each 10 episode
        # if episode % 10 == 0:
        #     for step in range(TEST_STEP):
        #         env.render() #refresh screeen
        #         # select action from trained network
        #         action = agent.action(state)
        #         next_state, reward, done, _ = env.step(action)
        #         state = next_state
        #         if (step+1)%100==0:
        #             print("At step: {}, reward: {}".format(step+1, reward))
        #             print("Agent pos: {}, Target pos: {}".format(state, env.target_pos))
        #         if done:
        #             print("At step: {}, reward: {}, Agent achieved Target!".format(step+1, reward))
        #             print("Agent pos: {}, Target pos: {}".format(state, env.target_pos))
        #             break


if __name__ == "__main__":
    main()

