from kaggle_environments import make
from keras.models import load_model

from Model import EvalModel, TargetModel
import tensorflow as tf

from TrainAgent_DDQN import DeepQNetwork, DQN_agent

env = make("lux_ai_2021", configuration={"seed": 4242, "loglevel": 2, "annotations": True, "width": 12, "height": 12}, debug=True)
steps = env.run([DQN_agent, "simple_agent"])
env.render(mode="ipython", width=1200, height=800)