from kaggle_environments import make

from AgentTeacher import agent_teacher
from Model import EvalModel, TargetModel
from TrainAgent_DDQN import DeepQNetwork, DQN_agent

global RL

eval_model = EvalModel(32)
target_model = TargetModel(32)
RL = DeepQNetwork(32, 8, 32 * 32 * 17, eval_model, target_model)
env = make("lux_ai_2021", configuration={"seed": 4242, "loglevel": 2}, debug=True)
steps = env.run([DQN_agent, agent_teacher])
env.render(mode="ipython", width=1200, height=800)
