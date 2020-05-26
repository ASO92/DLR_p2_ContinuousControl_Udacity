# Project 2: Continuous Control

## Environment to be solved

In a first attempt it was intended to solve the environment with 20 agents, nevertheless the time spent on training the model made really difficult the trial and error attempts.
For that reason, finally, I decided to solve the **Version 1** of the environment, with just ONE agent.

Goal of Version 1:
The task is episodic, and in order to solve the environment, your agent must get an average score of +30 over 100 consecutive episodes.

##### &nbsp;

## Approach
Here are the high-level steps taken in building an agent that solves this environment.

1. Evaluate the state and action space.
2. Establish performance baseline using a random action policy.
3. Select an appropriate algorithm and begin implementing it.
4. Run experiments, make revisions, and retrain the agent until the performance threshold is reached.

##### &nbsp;

### 1. Evaluate State & Action Space
The state space space has 33 dimensions corresponding to the position, rotation, velocity, and angular velocities of the robotic arm. There are two sections of the arm &mdash; analogous to those connecting the shoulder and elbow (i.e., the humerus), and the elbow to the wrist (i.e., the forearm) on a human body.

Each action is a vector with four numbers, corresponding to the torque applied to the two joints (shoulder and elbow). Every element in the action vector must be a number between -1 and 1, making the action space continuous.

##### &nbsp;

### 2. Establish Baseline
Before building an agent that learns, I started by testing an agent that selects actions (uniformly) at random at each time step.

```python
env_info = env.reset(train_mode=False)[brain_name]     # reset the environment
states = env_info.vector_observations                  # get the current state (for each agent)
scores = np.zeros(num_agents)                          # initialize the score (for each agent)
while True:
    actions = np.random.randn(num_agents, action_size) # select an action (for each agent)
    actions = np.clip(actions, -1, 1)                  # all actions between -1 and 1
    env_info = env.step(actions)[brain_name]           # send all actions to tne environment
    next_states = env_info.vector_observations         # get next state (for each agent)
    rewards = env_info.rewards                         # get reward (for each agent)
    dones = env_info.local_done                        # see if episode finished
    scores += env_info.rewards                         # update the score (for each agent)
    states = next_states                               # roll over states to next time step
    if np.any(dones):                                  # exit loop if episode finished
        break
print('Total score (averaged over agents) this episode: {}'.format(np.mean(scores)))
```

Running this agent a few times resulted in scores from 0.03 to 0.09. Obviously, if the agent needs to achieve an average score of 30 over 100 consecutive episodes, then choosing actions at random won't work.

##### &nbsp;

### 3. Implement Learning Algorithm

#### Deep Deterministic Policy Gradient (DDPG)
The chosen algorithm for this project is outlined in [this paper](https://arxiv.org/pdf/1509.02971.pdf), _Continuous Control with Deep Reinforcement Learning_, by researchers at Google Deepmind. In this paper, the authors present "a model-free, off-policy actor-critic algorithm using deep function approximators that can learn policies in high-dimensional, continuous action spaces." They highlight that DDPG can be viewed as an extension of Deep Q-learning to continuous tasks.

To start with, I used [this single-agent DDPG](https://github.com/udacity/deep-reinforcement-learning/tree/master/ddpg-pendulum) as a template, and also [this other](https://github.com/udacity/deep-reinforcement-learning/tree/master/ddpg-bipedal). To fo through the algorithm it also has been used the Udacity's classroom concepts and lessons.




## Ideas for Future Work


3. Negative rewards could be introduced to discourage the Agent from taking aleatory moves away from its goal of keeping in touch with the target.  

5. There are other actor-critic algorithms proposed to solve this kind of environment. So, future works may implement them to verify their performance. Some of those algorithms are:  
   * [TRPO - Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477)
   * [GAE - Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438)
   * [A3C - Asynchronous Advantage Actor-Critic](https://arxiv.org/abs/1602.01783)  
   * A2C - Advantage Actor-Critic
   * [ACER - Actor Critic with Experience Replay](https://arxiv.org/abs/1611.01224)
   * [PPO - Proximal Policy Optimization](https://arxiv.org/pdf/1707.06347.pdf)
   * [D4PG - Distributed Distributional Deterministic Policy Gradients](https://arxiv.org/pdf/1804.08617.pdf)
