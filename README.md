# Platform_RL
Train RL agent to play Platform Domain

## Dependencies

- gym==0.21
- ray[rllib]
- gym_platform

## Usage

  python agent_trainer.py


## Explanation

Platform is an OpenAI Gym environment [[Bester et al. 2019]](https://github.com/cycraig/gym-platform) with continuous state space and parameterised action space. The Gym environment specification is as follow:

observation_space:
Tuple(	Box(0.0, 1.0, (9,), float32), 
	      Discrete(200)
)

action_space:
Tuple(	Discrete(3), 
	Tuple(	Box(0.0, 30.0, (1,), float32), 
		      Box(0.0, 720.0, (1,), float32), 
		      Box(0.0, 430.0, (1,), float32))
)


The agent is trained with the RLlib RAY library using the PPO algorithm.

## Future work

- Train with other RL algorithms
- Noise in environment
- Change reward
- Change exploraton behavior
- Change NN architecture
- Change replay
- Change observations

