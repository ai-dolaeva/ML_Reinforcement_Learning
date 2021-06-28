import os
import gym
import torch
import random
import DQNAgent
import save_plots

if __name__ == '__main__':

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.set_default_tensor_type('torch.FloatTensor')

    env_name = 'LunarLander-v2'
    environment = gym.make(env_name)

    checkpoint_dir = './checkpoints'
    checkpoint_frequency = 10
    num_episodes = 1000
    seed = 42
    fc1_dim = 128
    fc2_dim = 128

    torch.manual_seed(seed)
    random.seed(seed)
    environment.seed(seed)

    state_size = environment.observation_space.shape[0]
    num_actions = environment.action_space.n
    agent = DQNAgent.create_agent(state_size, num_actions,
                                  device, fc1_dim, fc2_dim)

    total_steps = 0
    scores, eps_history = [], []
    # images = []
    counter = 0

    for episode in range(num_episodes):
        last_state = environment.reset()
        total_reward = 0
        while True:
            environment.render()
            # images.append(environment.render())

            action = agent.act(last_state)
            next_state, reward, is_terminal, info = environment.step(action)
            agent.observe_and_learn(last_state, action, reward,
                                    next_state, is_terminal)

            total_steps += 1
            total_reward += reward
            last_state = next_state

            scores.append(total_reward)
            eps_history.append(agent.eps_threshold)

            if is_terminal:
                print('Episode {}, steps: {}, reward: {}, counter {}'.format(
                    episode, total_steps, total_reward, counter))

                if episode % checkpoint_frequency == 0:
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    agent.save_weights(os.path.join(checkpoint_dir, env_name))
                break

        if total_reward < 0:
            counter = 0
            continue

        counter += 1
        if counter >= 100:
            print('finish')
            break

    environment.close()

    x = [i + 1 for i in range(len(eps_history))]
    filename = 'train_lunar_lander.png'
    save_plots.saveTrainingPlot(x, scores, eps_history, filename)

    # save_plots.saveTrainingGif(images)
