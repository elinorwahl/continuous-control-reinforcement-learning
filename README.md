# Reinforcement Learning Project: Continuous Control

[image1]: https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif "Trained Agent"

## Introduction

In this project, we teach an AI reinforcement learning agent in the Unity [Reacher environment](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) to direct a double-jointed robot arm to a target location - marked below by a green bubble - and to maintain contact with the target location for as long as possible. A reward of +0.1 is given for each time step that the agent's hand is in the target location, and the environment is considered solved when the robot arm agent attains an average score of 30+ points over 100 consecutive episodes.

![Trained Agent][image1]

The state space in this environment has 33 variables corresponding to the position, rotation, velocity, and angular velocities of the robot arm. The action space is a vector of 4 variables corresponding to the torque applied to the two joints of the robot arm, and each number in the action vector is clipped between -1 and 1. This environment is marked by a **continuous action space**, with a highly variable range of potential action values and a wide range of motion for the arm to use.

There are two versions of this project environment. The first version contains a single agent; the second version contains 20 identical agents operating in their own copies of the environment, whose learning experience is gathered and then shared across all the agents. For my own implementation, I've chosen to work with the 20-agent environment. This type of multi-agent learning is useful for AI algorithms like [proximal policy optimization](https://arxiv.org/pdf/1707.06347.pdf), [asynchronous methods](https://arxiv.org/pdf/1602.01783.pdf), and [distributed distributional deterministic policy gradients](https://openreview.net/pdf?id=SyZipzbCb).

## Methods

The reinforcement learning algorithm being used in this project is deep deterministic policy gradients, or [DDPG](https://arxiv.org/pdf/1509.02971.pdf). DDPG combines the strengths of **policy-based (stochastic)** and **value-based (deterministic)** AI learning methods by using two agents, called the Actor and the Critic. The actor directly estimates the optimal policy, or action, for a given state, and applies gradient ascent to maximize rewards. The critic takes the actor's output and uses it to estimate the value (or cumulative future reward) of state-action pairs. The weights of the actor are then updated with the critic’s output, and the critic is updated with the gradients from the temporal-difference error signal at each step. This hybrid algorithm can be a very robust form of artificial intelligence, because it needs fewer training samples than a purely policy-based agent, and demonstrates more stable learning than a purely value-based one.

[This baseline DDPG agent and model](https://github.com/udacity/deep-reinforcement-learning/tree/master/ddpg-pendulum) was used as a starting point. The modifications I've made to `ddpg_agent.py` are as follows:

- Increasing the size of the experience replay buffer from `128` to `512`.

- Decreasing the weight decay rate of the Critic optimizer from `0.0001` to `0.0`. This means that the critic's weights never decrease in value.

- Modifying the `Agent.step()` method to accommodate multiple agents, and to employ a learning interval. This ensures that the agent performs the learning step only once every 20 time steps during training, and each time 10 passes are made through experience sampling and the `Agent.learn()` method:

```
def step(self, state, action, reward, next_state, done, t):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward for each agent
        for state, action, reward, next_state, done in zip(state, action, reward, next_state, done):
            self.memory.add(state, action, reward, next_state, done)

        # Learn, if enough samples are available in memory
        if len(self.memory) > BATCH_SIZE and t % 20 == 0:
            for i in range(10):
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)
```

- Adding gradient clipping to the critic's loss in the `Agent.learn()` method. This bounds the upper limits of the gradients close to 1, and prevents the 'exploding gradient problem', in which a network risks making the updated weights too large to properly learn from:

```
torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
```

- In the `OUNoise.sample()` method, changing `random.random()` to `np.random.standard_normal()`. This means that the random noise being added to the experience replay buffer samples via the Ornstein-Uhlenbeck process follows a Gaussian distribution, and turns out to perform much better than a completely random distribution in this case!

The modifications I've made to `model.py` are as follows:

- Adding batch normalization to the first hidden layer of both the Actor and Critic models. This also addresses the exploding gradient problem by limiting the activation functions of these networks to a stable distribution of inputs.

- Decreasing the size of the two hidden layers of the Actor network from `400` and `300` nodes to `256` and `128` nodes, respectively. Experience has taught me that smaller networks can often perform more nimbly than large ones.

- Giving the Critic network three hidden layers, the first having `128` nodes, the second having `64`, and the third having `32`. Making the Critic network smaller and longer than the Actor also had the apparent effect of improving overall performance!

To recap, these are the final hyperparameters:

```
BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 512        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor 
LR_CRITIC = 1e-3        # learning rate of the critic
WEIGHT_DECAY = 0.0      # L2 weight decay
```

And the final network architecture consists of an Actor network with two hidden layers of size `256, 128`, with batch normalization in the first hidden layer, and a `tanh` output activation function; and a Critic network with three hidden layers of size `128, 64, 32`, batch normalization in the first hidden layer, and a `relu` output activation function.

This is the learning progress of the resulting DDPG agent architecture, simultaneously controlling 20 robot arms in their simulated Unity environments:

![Graph of DDPG agent performance](/images/ddpg_results.png)

The agent solved the environment, and obtained an average score of 30+ points, within 109 episodes! It would also appear that all of the agents steadily decrease their margin of error over time, and learn more smoothly from their shared experience.

## Further Study

It might be useful to experiment with other network architectures for this project - different numbers of hidden layers, different numbers of nodes, and additional features such as dropout.

Increasing the size of the experience replay buffer had a major effect on the performance of the agent, and it might perform better with an even larger buffer. It might also be useful to try implementing prioritized experience replay, instead of a random buffer.

The multi-agent learning algorithms mentioned above would also be applicable to this Unity learning application, and it could be informative to explore their effectiveness at this particular task.

Finally, a very simple change to make would be to raise the target score - say, to 40+ - to more accurately gauge the ability of any agent architecture to learn over time.

## Usage

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:

    - **_Version 1: One (1) Agent_**
        - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
        - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
        - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip)
        - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)

    - **_Version 2: Twenty (20) Agents_**
        - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
        - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
        - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
        - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux_NoVis.zip) (version 1) or [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux_NoVis.zip) (version 2) to obtain the "headless" version of the environment.  You will **not** be able to watch the agent without enabling a virtual screen, but you will be able to train the agent.  (_To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above._)

2. Place the file in the DRLND GitHub repository, in the `p2_continuous-control/` folder, and unzip (or decompress) the file.