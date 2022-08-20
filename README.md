# FlappyBirRL <img src="assets/icon.png" title="icon" alt="icon" width="50" height="40"/>&nbsp;

This repository contains the code and the report for the project of the exam [Autonomous and Adaptive Systems](https://www.unibo.it/en/teaching/course-unit-catalogue/course-unit/2021/454632) of the [Master's degree in Artificial Intelligence, University of Bologna](https://corsi.unibo.it/2cycle/artificial-intelligence).

In this project the A2C reinforcement learning method is used to learn to play [Flappy Bird](https://en.wikipedia.org/wiki/Flappy_Bird) mobile game.

## Game

This is part of a game played by an agent trained using A2C algorithm

 ![game gif](assets/game.gif)

## Structure

The repository is composed as follows:
- *agents*: agents and networks
- *assets*: resources for the README.md
- *envs*: customized version of the original environment
- *evaluation*: functions to evaluate pretrained models and compare them using a boxplot
- *training*: A2C algorithm, data, plots and utility related to training
- *play.py*: run a game with an agent
- *report.pdf*: final report of the project
- *train.py*: train an agent
- *utils.py*: constants and functions to plot graphs

## Requirements

Python 3.7 must be used to be able to run correctly the project

### OpenAI Gym Environment

The environment has been partially developed reusing the implementation provided by [Talendar](https://github.com/Talendar/flappy-bird-gym). To install the library follow these steps:

1. Clone the repository in the root of this project folder
```console
git clone https://github.com/Talendar/flappy-bird-gym.git
```
2. Copy the folder **envs** of the current repository to __flappy-bird-gym/flappy_bird_gym__ replacing the folder with the same name. This operation allows to replace the original implementation of the environment with the new customized one.
3. Install the library **flappy-bird-gym** directly from a local folder.
```console
pip install -e flappy-bird-gym
```
Now **flappy-bird-gym** should be recognized as package, it is possible to check it with Python

```python
import flappy_bird_gym
```

### Packages
Because of **flappy-bird-gym** requirements and **tensorflow** compatibility it is strongly suggested to use these packages with the corresponding versions:
- numpy version=1.20
- tensorflow version=2.8

## Run
To train a specific version of an agent it is possible to use the script **train.py**.
Run:
```console
py train.py <agent> <num_episodes> <num_processes> <discount_rate> <learning_rate>
```
where:
- *agent* is the type of agent (base version, cnn version or base version trained using entropy regularization)
- *num_episodes* is the number of game to train the agent
- *num_processes* is the number of parallel processes to train the agent
- *discount_rate* is the value used to obtain expected return
- *learning_rate* is the rate used by the optimizer to minimize the loss function

To play a game with a specific agent the script **play.py** can be used as follows:
```console
py play.py <agent>
```
where *agent* is the name of a pretrained agent
