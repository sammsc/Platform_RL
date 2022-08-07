import gym
from gym_platform.envs.platform_env import PlatformEnv
import ray
from ray.tune.registry import register_env
import ray.rllib.agents.ppo as ppo
import shutil
import time


def train_agent(env_cur, n_episodes):
    '''
    Train the agent

        Parameters:
            env_cur: gym environment
            n_episodes: total number of training episodes

        Returns:
            agent: RL agent
            sav_file: last training checkpoint file saved
    '''

    save_dir = "train_dir"
    shutil.rmtree(save_dir, ignore_errors=True, onerror=None)

    ray.init()
    register_env(env_cur, lambda config: PlatformEnv())
    
    config = ppo.DEFAULT_CONFIG.copy()
    config["log_level"] = "WARN"
    agent = ppo.PPOTrainer(config, env=env_cur)
    
    # train agent
    for i in range(n_episodes):
        result = agent.train()
        sav_file = agent.save(save_dir)
        print(f'{i}: reward: mean={result["episode_reward_mean"]}, '
              f'max={result["episode_reward_max"]}, min={result["episode_reward_min"]}. '
              f'length={result["episode_len_mean"]}. file={sav_file}')
        
    return agent, sav_file


def view_agent(env_cur, agent, sav_file):
    '''
    Display the trained agent

        Parameters:
            env_cur: gym environment
            agent: trained RL agent
            sav_file: checkpoint file saved from training

        Returns:
            None
    '''

    # run the policy
    agent.restore(sav_file)
    env = gym.make(env_cur)

    obs = env.reset()
    ttl_reward = 0
    n_step = 10

    for step in range(n_step):
        action = agent.compute_action(obs)
        obs, reward, done, info = env.step(action)
        ttl_reward += reward

        env.render()
        time.sleep(0.001)
        if done:
            # reward at the end of episode
            print("reward", ttl_reward)
            obs = env.reset()
            ttl_reward = 0
            
    env.close()
    
    
def main():
    training_episodes = 5000
    env_cur = "Platform-v0"
    agent, sav_file = train_agent(env_cur, training_episodes)
    view_agent(env_cur, agent, sav_file)


if __name__ == "__main__":
    main()