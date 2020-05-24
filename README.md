
[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif "Trained Agent"
[image2]: https://3.bp.blogspot.com/-I6UKhtpt-pI/WzP8ThUgMRI/AAAAAAAADFQ/mmbmu0YtDeAGT1RJj0pDPPm_jYyyYYg0gCLcBGAs/s1600/image8.gif "GoogleArms"
[image10]: https://user-images.githubusercontent.com/10624937/43851646-d899bf20-9b00-11e8-858c-29b5c2c94ccc.png "Crawler"


# Project 2: Continuous Control

## 1.- Introduction

This repository includes the code of my personal attempt to solve the second work project of the Udacity Nanodegree. In the following photograph it is posible to see the environment solved.

![Trained Agent][image1]
*Credit image: [Trained agent in Reacher Environment](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher)*

It may be a representation of a real problem of several robotic arms trying to grab any object, as we can see in the following image.

![Google](https://3.bp.blogspot.com/-I6UKhtpt-pI/WzP8ThUgMRI/AAAAAAAADFQ/mmbmu0YtDeAGT1RJj0pDPPm_jYyyYYg0gCLcBGAs/s1600/image8.gif)
*Credit image: [Google AI Blog](https://ai.googleblog.com/2018/06/scalable-deep-reinforcement-learning.html)*

## 2. First steps in the environment

This environment has been built using the **Unity Machine Learning Agents Toolkit (ML-Agents)**, which is an open-source Unity plugin that enables games and simulations to serve as environments for training intelligent agents. You can read more about ML-Agents by perusing this [GitHub repository](https://github.com/Unity-Technologies/ml-agents).  

The project environment provided by Udacity is similar to, but not identical to the Reacher environment on the [Unity ML-Agents GitHub page](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher). All the code used in this repository is a variation of the....

In this environment, a double-jointed arm can move to target locations. A reward of `+0.1` is provided for each step that the agent's hand is in the goal location.  
Thus, the goal of the agent is to maintain its position at the target location for as many time steps as possible.

#### 2.1. State and action spaces
The observation space consists of **33 variables** corresponding to position, rotation, velocity, and angular velocities of the arm.  

Each action is **a vector with four numbers**, corresponding to torque applicable to two joints. Every entry in the action vector should be a number **between -1 and 1**.

#### 2.2. Solving the environment

There are two versions of the environment.

* **Version 1: One (1) Agent**  
The task is episodic, and in order to solve the environment, the agent must get an **average score of +30 over 100 consecutive episodes**.

* **Version 2: Twenty (20) Agents**  
The barrier to solving the second version of the environment is slightly different, to take into account the presence of many agents. In particular, the agents must get an average score of +30 (over 100 consecutive episodes, and over all agents). Specifically,
 * After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. That yields 20 (potentially different) scores. We then take the average of these 20 scores.  
 * That yields an average score for each episode (where the average is over all 20 agents).  
 * The environment is considered solved, when the **moving average over 100 episodes** of those average scores **is at least +30**.

## 3. Included in this repository

* The code used to create and train the Agent
  * Continuous_Control.ipynb
  * ddpg_agent.py
  * model.py
* The trained model
  * checkpoint.pt
* A file describing all the packages required to set up the environment
  * environment.yml
* A Report.md file describing the development process and the learning algorithm, along with ideas for future work
* This README.md file

## 4. Setting up the environment

This section describes how to get the code for this project, configure the local environment, and download the Unity environment with the Agents.

#### 4.1 Getting the code
You have two options to get the code contained in this repository:
##### Option 1. Download it as a zip file

* [Click here](https://github.com/ASO92/DRL_p2_ContinuousControl_Udacity/archive/master.zip) to download all the content of this repository as a zip file
* Decompress the downloaded file into a folder of your choice

##### Option 2. Clone this repository using Git version control system
If you are not sure about having Git installed in your system, run the following command to verify that:
```
$ git --version
```
Having Git installed in your system, you can clone this repository by running the following command:

```
$ git clone https://github.com/ASO92/DRL_p2_ContinuousControl_Udacity/archive.git
```
#### 4.2 Installing Ananconda/Miniconda
Miniconda is a free minimal installer for conda. It is a small, bootstrap version of Anaconda that includes only conda, Python, the packages they depend on, and a small number of other useful packages, including pip, zlib, and a few others.  

If you would like to know more about Anaconda, visit [this link](https://www.anaconda.com/).

In the following links, you find all the information to install **Miniconda** (*recommended*)

* Download the installer: [https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html)
* Installation Guide: [https://conda.io/projects/conda/en/latest/user-guide/install/index.html](https://conda.io/projects/conda/en/latest/user-guide/install/index.html)

Alternatively, you can install the complete Anaconda Platform

* Download the installer: [https://www.anaconda.com/distribution/](https://www.anaconda.com/distribution/)
* Installation Guide: [https://docs.anaconda.com/anaconda/install/](https://docs.anaconda.com/anaconda/install/)

####  4.3 Configuring the local environment

#### Installing the packages
The following steps will guide you to install the necesary dependences to execute correctly the environment.

**1. Create the environment**  
```
$ conda create --name drlnd-p2-control python=3.6
$ conda activate drlnd-p2-control
```  

**2. Install PyTorch**  
Follow [this link](https://pytorch.org/get-started/locally) to select the right command for your system.  
Here, there are some examples which you can use, if it fit in your system:  

**a.** Linux or Windows

```
## Run this command if you have an NVIDIA graphic card and want to use it  
$ conda install pytorch cudatoolkit=10.1 -c pytorch

## Run this command, otherwise
$ conda install pytorch cpuonly -c pytorch
```  

**b.** Mac  
MacOS Binaries do not support CUDA, install from source if CUDA is needed

```
$ conda install pytorch -c pytorch  
```  


**3. Install Unity Agents**  

```
$ pip install unityagents
```  

#### 4.4. Download the Unity environment with the Agents  

Download the environment from one of the links below and decompress the file into your project folder.  
You need only select the environment that matches your operating system:

 * Version 1: One (1) Agent
     * Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
     * Linux Headless: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux_NoVis.zip)
     * Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
     * Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip)
     * Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)
 * Version 2: Twenty (20) Agents
     * Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
     * Linux Headless: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux_NoVis.zip)
     * Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
     * Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
     * Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)     

(_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining whether your computer is running a 32-bit version or 64-bit version of the Windows operating system.

(_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux_NoVis.zip) (version 1) or [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux_NoVis.zip) (version 2) to obtain the "headless" version of the environment.  You will **not** be able to watch the agent without enabling a virtual screen, but you will be able to train the agent.  (_To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above._)

## 5. Train the agent


#### 5.2 Adjusting the Hyperparameters
To experiment with how the Agent learns through distinct parameters, you can tune these variables:  

**1.** In the **Continuous_Control.ipynb** file  

* **\_target_score**: How many points your agent must obtain to consider the environment solved
* **\_target_episodes**: How many episodes to consider when calculating the moving average
* **n_episodes**: Maximum number of training episodes
* **max_t**: Maximum number of time steps per episode
* **random_seed**: The number used to initialize the pseudorandom number generator

**2.** In the **ddpg_agent.py** file

* **\_batch_size**: Minibatch size
* **\_buffer_size**: Replay buffer size
* **\_gamma**: Discount factor for expected rewards
* **\_lr_actor**: Learning rate for the local actor's network
* **\_lr_critic**: Learning rate for the local critic's network
* **\_tau**: Multiplicative factor for the soft-update of the target networks's weights
* **\_noise_decay**: Multiplicative factor for the noise-process rate decay
* **fc_layers for the actor network**: Number and size of the actor network's layers
* **fc_layers for the critic network**: Number and size of the critic network's layers

## 6. Uninstall
If you wish to revert all the modifies in your system, and remove all the code, dependencies and programs installed in the steps above, you will want to follow the next steps.

#### 6.1 Uninstall Miniconda or Anaconda
To do so, please refer to [this link](https://docs.anaconda.com/anaconda/install/uninstall/).


#### 6.2 Remove the code
Simply delete the entire folder containing the code you downloaded in the step "Getting the code"


## 10.- (Optional) Challenge: Crawler Environment

After you have successfully completed the project, you might like to solve the more difficult **Crawler** environment.

![Crawler][image10]

In this continuous control environment, the goal is to teach a creature with four legs to walk forward without falling. It is one environment of the [ML Unity-Technologies](https://github.com/Unity-Technologies/ml-agents/tree/master/docs).

You can read more about this environment in the ML-Agents GitHub [here](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#crawler).  To solve this harder task, you'll need to download a new Unity environment.  (**Note**: Udacity students should not submit a project with this new environment.)

You need only select the environment that matches your operating system:
- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Crawler/Crawler_Linux.zip)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Crawler/Crawler.app.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Crawler/Crawler_Windows_x86.zip)
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Crawler/Crawler_Windows_x86_64.zip)

Then, place the file in the `p2_continuous-control/` folder in the DRLND GitHub repository, and unzip (or decompress) the file.  Next, open `Crawler.ipynb` and follow the instructions to learn how to use the Python API to control the agent.

(_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Crawler/Crawler_Linux_NoVis.zip) to obtain the "headless" version of the environment.  You will **not** be able to watch the agent without enabling a virtual screen, but you will be able to train the agent.  (_To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above._)
