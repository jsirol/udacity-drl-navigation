from collections import deque
import numpy as np


def train_dqn(agent, env, num_episodes=100, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    """
    Training loop for Deep Q-Learning.
    :param agent: learning agent
    :param env: handle to the environment
    :param num_episodes: number of episodes
    :param max_t: max steps per episode
    :param eps_start: epsilon start value
    :param eps_end: epsilon end value
    :param eps_decay: epsilon decay rate
    :return: array of reward per iteration
    """
    # get the default brain
    brain_name = env.brain_names[0]
    scores = []
    scores_window = deque(maxlen=100)
    eps = eps_start
    for episode in range(1, num_episodes+1):
        env_info = env.reset()[brain_name]
        state = env_info.vector_observations[0] # reset environment for new episode
        score = 0 # reset score for current episode
        for t in range(max_t):
            # choose action, execute it, and get the reward, next_state
            action = agent.act(state, eps)
            env_info = env.step(action)[brain_name]
            next_state = env_info.vector_observations[0]
            reward = env_info.rewards[0]
            done = env_info.local_done[0]
            # next_state, reward, done, _ = pass
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        scores_window.append(score)
        scores.append(score)
        eps = max(eps_end, eps_decay*eps)
        if episode % 10 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, np.mean(scores_window)))
    return scores
