import numpy as np
from Model import get_inputs, get_model
from lux.game import Game
import random
from IPython.display import clear_output
from kaggle_environments import make
import tensorflow as tf
import tensorflow.keras.backend as K
from tqdm import tqdm


def custom_mean_squared_error_temp(y_true, y_pred):

    y_units_true = y_true[:, :, :, :6]
    y_cities_true = y_true[:, :, :, 6:]

    y_units_pred = y_pred[:, :, :, :6]
    y_cities_pred = y_pred[:, :, :, 6:]

    # 用来区分unit和city和啥都没的块
    is_unit = tf.keras.backend.max(y_units_true, axis=-1)
    is_city = tf.keras.backend.max(y_cities_true, axis=-1)

    y_units_pred *= K.stack([is_unit] * 6, axis=-1)
    y_cities_pred *= K.stack([is_city] * 2, axis=-1)

    loss1 = K.square(y_units_pred - y_units_true)  # /K.sum(is_unit)
    loss2 = K.square(y_cities_pred - y_cities_true)  # /K.sum(is_city)
    return K.concatenate([loss1, loss2])


def get_prediction_actions(y, player):
    # move
    option = np.argmax(y, axis=2)
    # c s n w e build_city & research & buid_worker
    actions = []
    for i in player.units:
        #         print(option.shape,i.pos.y,i.pos.x)
        d = "csnwe#############"[option[i.pos.y, i.pos.x]]
        if option[i.pos.y, i.pos.x] < 5:
            actions.append(i.move(d))
        elif option[i.pos.y, i.pos.x] == 5 and i.can_build(game_state.map):
            actions.append(i.build_city())

    for city in player.cities.values():
        for city_tile in city.citytiles:
            #             city_tiles.append(city_tile)
            if option[city_tile.pos.y, city_tile.pos.x] == 6:
                action = city_tile.research()
                actions.append(action)
            if option[city_tile.pos.y, city_tile.pos.x] == 7:
                action = city_tile.build_worker()
                actions.append(action)
    return actions, option


Last_State = {}
learning_rate = 0.01
gamma = 0.95
epsilon = 1.0
epsilon_final = 0.01
epsilon_decay = 0.995
game_state = None
model = None
last_reward = 0
W = 0
summary_writer = None
iteration = 0
epoch = 0


def agent(observation, configuration):
    global game_state, epsilon, model, last_reward, W, summary_writer, iteration, epoch, reward_count

    ### Do not edit ###
    if observation["step"] == 0:
        game_state = Game()
        game_state._initialize(observation["updates"])
        game_state._update(observation["updates"][2:])
        game_state.id = observation.player
    else:
        game_state._update(observation["updates"])

    ### AI Code goes down here! ###
    player = game_state.players[observation.player]
    opponent = game_state.players[(observation.player + 1) % 2]
    width, height = game_state.map.width, game_state.map.height

    # Get Prediction of actions
    x = get_inputs(game_state)
    y = model.predict(np.asarray([x]))[0]

    if random.random() < epsilon:
        y = np.random.rand(*y.shape)
    print("eps ", epsilon, end=" | ")
    actions, option = get_prediction_actions(y, player)

    print("Reward", observation["reward"])

    if observation["reward"] not in reward_count:
        reward_count[observation["reward"]] = 1
    else:
        reward_count[observation["reward"]] += 1

    if observation.player in Last_State:
        _x, _y, _player, _option = Last_State[observation.player]
        state, next_state, reward = _x, x, observation["reward"]

        # Reward
        if reward > last_reward:
            r = 1
        elif reward < last_reward:
            r = -1
        else:
            r = 0

        # Q-learning update

        for i in _player.units:
            Q1 = _y[i.pos.y, i.pos.x][_option[i.pos.y, i.pos.x]]
            Q2 = y[i.pos.y, i.pos.x][_option[i.pos.y, i.pos.x]]
            v = r + gamma * Q2 - Q1   # gamma * Q' - Q || gamma * (Q' - Q)
            _y[i.pos.y, i.pos.x][_option[i.pos.y, i.pos.x]] += learning_rate * v

        _y = y + learning_rate * _y

        states = [state]
        _y_ = [_y]

        test = custom_mean_squared_error_temp(_y[np.newaxis, :], model.predict(np.asarray(states)))
        log = model.fit(np.asarray(states), np.asarray(_y_), epochs=1, verbose=1)
        # with summary_writer.as_default():
        #     tf.summary.scalar(f"episode{epoch}_loss", log.history['loss'][0], step=iteration)
        #     iteration += 1
        if epsilon > epsilon_final:
            epsilon *= epsilon_decay
    Last_State[observation.player] = [x, y, player, option]
    last_reward = observation["reward"]
    return actions


if __name__ == '__main__':

    episodes = 100

    # RL training
    # sizes = [12, 16, 24, 32]
    sizes = [12]

    reward_count = {}

    for size in sizes:
        # Inistialise the model
        summary_writer = tf.summary.create_file_writer(f'../test/experiment_Q_log/size{size}')
        model = get_model(size)
        Last_State = {}
        for eps in tqdm(range(episodes)):
            iteration = 0
            epoch = eps
            epsilon = 0.2  # Maintaining exploration
            clear_output()
            print("=== Episode {} ===".format(eps))
            env = make("lux_ai_2021", debug=True,
                       configuration={"annotations": True, "width": size, "height": size})
            steps = env.run(["simple_agent", agent])

            with summary_writer.as_default():
                tf.summary.scalar(f"{size}x{size}_cities", Last_State[1][2].city_tile_count, eps)
                tf.summary.scalar(f"{size}x{size}_units", len(Last_State[1][2].units), eps)


        # Save the model
        model.save_weights("model_%d.h5" % size)

    print(reward_count)
