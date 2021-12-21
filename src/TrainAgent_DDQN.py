import tensorflow as tf
import numpy as np
from IPython.display import clear_output
from kaggle_environments import make
from Model import get_inputs, EvalModel, TargetModel, EvalModel_Conv, TargetModel_Conv
from lux.game import Game
from typing import List
import tensorflow.keras.backend as K
from AgentTeacher import agent_teacher


def custom_mean_squared_error(y_true, y_pred):

    y_units_true = K.eval(y_true[:, :, :, :6])
    y_cities_true = K.eval(y_true[:, :, :, 6:])
    y_units_true[y_units_true != 0] = 1
    y_cities_true[y_cities_true != 0] = 1

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


def custom_accuracy(y_true, y_pred):

    # TODO 看看有没有正确置为1
    units_true_action = K.argmax(y_true[:, :, :, :6], axis=-1)
    cities_true_action = K.argmax(y_true[:, :, :, 6:], axis=-1)

    y_units_true = K.eval(y_true[:, :, :, :6])
    y_cities_true = K.eval(y_true[:, :, :, 6:])
    y_units_true[y_units_true != 0] = 1
    y_cities_true[y_cities_true != 0] = 1

    y_units_pred = y_pred[:, :, :, :6]
    y_cities_pred = y_pred[:, :, :, 6:]

    # 用来区分unit和city和啥都没的块
    is_unit = tf.keras.backend.max(y_units_true, axis=-1)
    is_city = tf.keras.backend.max(y_cities_true, axis=-1)

    y_units_pred *= K.stack([is_unit] * 6, axis=-1)
    y_cities_pred *= K.stack([is_city] * 2, axis=-1)

    units_pred_action = K.eval(K.argmax(y_units_pred, axis=-1))
    cities_pred_action = K.eval(K.argmax(y_cities_pred, axis=-1))

    total = K.sum(np.max(y_units_true, axis=-1)) + K.sum(np.max(y_cities_true, axis=-1))

    # test_true0 = K.eval(units_true_action[0, :, :])
    # test_true1 = K.eval(units_true_action[1, :, :])
    # test_true2 = K.eval(units_true_action[2, :, :])
    # test_true3 = K.eval(units_true_action[3, :, :])
    # test_pred0 = units_pred_action[0, :, :]
    # test_pred1 = units_pred_action[1, :, :]
    # test_pred2 = units_pred_action[2, :, :]
    # test_pred3 = units_pred_action[3, :, :]

    units_FP = np.count_nonzero(units_pred_action - units_true_action)
    cities_FP = np.count_nonzero(cities_pred_action - cities_true_action)
    return (total - (units_FP + cities_FP)) / total


def get_actions_with_e_greedy(y, player, e, training=True):

    def flip_coin():
        return np.random.uniform() < e

    assert y.shape[0] == 1
    # y 是 (1, map_size, map_size, 8)
    y = np.squeeze(y, axis=0)
    units_option = np.argmax(y[:, :, :6], axis=2)
    cities_option = np.argmax(y[:, :, 6:], axis=2)
    #option = np.argmax(np.squeeze(y, axis=0), axis=2)
    # c s n w e build_city & research & buid_worker
    actions = []
    # TODO 加上pillage
    for i in player.units:

        act_flag = False
        if not i.can_act():
            continue
        d = "csnwe#############"[units_option[i.pos.y, i.pos.x]]
        if flip_coin() and training:
            # Epsilon-greedy
            random_number = np.random.randint(0, 2)
            if random_number == 0:
                actions.append(i.random_move())
                act_flag = True
            elif random_number == 1 and i.can_build(game_state.map):
                actions.append(i.build_city())
                act_flag = True
            continue
        if units_option[i.pos.y, i.pos.x] < 5:
            actions.append(i.move(d))
            act_flag = True
        elif units_option[i.pos.y, i.pos.x] == 5 and i.can_build(game_state.map):
            actions.append(i.build_city())
            act_flag = True

        while not act_flag and training:
            random_number = np.random.randint(0, 2)
            if random_number == 0:
                actions.append(i.random_move())
                act_flag = True
            elif random_number == 1 and i.can_build(game_state.map):
                actions.append(i.build_city())
                act_flag = True

    # TODO 加上build_cart
    for city in player.cities.values():

        for city_tile in city.citytiles:
            #             city_tiles.append(city_tile)
            act_flag = False
            if not city_tile.can_act():
                continue

            if flip_coin() and training:
                random_number = np.random.randint(6, 8)
                if random_number == 6:
                    actions.append(city_tile.research())
                    act_flag = True
                elif random_number == 7 and player.city_tile_count > len(player.units):
                    actions.append(city_tile.build_worker())
                    act_flag = True
                continue
            if cities_option[city_tile.pos.y, city_tile.pos.x] == 0:
                action = city_tile.research()
                actions.append(action)
                act_flag = True
            if cities_option[city_tile.pos.y, city_tile.pos.x] == 1 and player.city_tile_count > len(player.units):
                action = city_tile.build_worker()
                actions.append(action)
                act_flag = True

            while not act_flag and training:
                random_number = np.random.randint(6, 8)
                if random_number == 6:
                    actions.append(city_tile.research())
                    act_flag = True
                elif random_number == 7 and player.city_tile_count > len(player.units):
                    actions.append(city_tile.build_worker())
                    act_flag = True

    return actions


class DeepQNetwork:
    def __init__(self, map_size, n_actions, n_features, eval_model, target_model):

        self.params = {
            'map_size': map_size,
            'n_actions': n_actions,
            'n_features': n_features,
            'learning_rate': 0.01,
            'gamma': 0.9,
            'e_greedy': .2,
            'replace_target_iter': 300,
            'memory_size': 500,
            'batch_size': 4,
            'e_greedy_increment': None
        }

        # total learning step

        self.learn_step_counter = 0

        # initialize zero memory [s, a, r, s_]
        self.epsilon = 0 if self.params['e_greedy_increment'] is not None else self.params['e_greedy']
        self.memory = np.zeros((self.params['memory_size'],
                                self.params['n_features'] * 2
                                + self.params['map_size'] * self.params['map_size'] * self.params['n_actions'] + 1))

        self.eval_model = eval_model
        self.target_model = target_model

        self.eval_model.compile("adam",
                                loss=custom_mean_squared_error,
                                metrics=[custom_accuracy],
                                run_eagerly=True,)
        self.cost_his = []

    def choose_action(self, observations, training=True) -> (List, np.array):
        # TODO 搞清楚这里的action要不要转(好像要)
        # to have batch dimension when feed into tf placeholder
        # game_state = get_game_from_obs(observations)
        global game_state
        x = np.array(get_inputs(game_state), float).flatten()
        x = x[np.newaxis, :]

        player = game_state.players[observations.player]

        # actions_value是个 map_size * map_size * #actions 的矩阵，这里扮演Q value的角色
        actions_value = self.eval_model.predict(x)
        # print(actions_value)
        # action = get_prediction_actions(actions_value, player)

        actions = get_actions_with_e_greedy(actions_value, player, self.params["e_greedy"], training)
        return actions, actions_value

    def store_transition(self, state, a, r, state_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

        #state = np.array(get_inputs(state), float).flatten()  # size^2 * 17
        #state_ = np.array(get_inputs(state_), float).flatten()  # size^2 * 17

        actions = a.flatten()
        transition = np.hstack((state, actions, r, state_))

        # replace the old memory with new memory
        index = self.memory_counter % self.params['memory_size']
        self.memory[index, :] = transition

        self.memory_counter += 1

    def learn(self):

        # TODO debug进来看看怎么改合适
        # sample batch memory from all memory
        if self.memory_counter > self.params['memory_size']:
            sample_index = np.random.choice(self.params['memory_size'], size=self.params['batch_size'])
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.params['batch_size'])

        # (batch_size, map_size, map_size, 17, action, reward, map_size, map_size, 17)
        batch_memory = self.memory[sample_index, :]

        # 输入是batchsize * (map_size * map_size * 17)
        # import ipdb
        # ipdb.set_trace()
        q_next = self.target_model.predict(batch_memory[:, -self.params['n_features']:])
        q_eval = self.eval_model.predict(batch_memory[:, :self.params['n_features']])

        # change q_target w.r.t q_eval's action
        q_target = q_eval.copy()

        batch_index = np.arange(self.params['batch_size'], dtype=np.int32)
        # 把squeezed的action转成 (batch_size, map_size, map_size, 8)
        action_length = self.params['map_size'] * self.params['map_size'] * self.params['n_actions']
        eval_act_index = np.resize(
            batch_memory[:, self.params['n_features']:self.params['n_features'] + action_length],
            (self.params['batch_size'], self.params['map_size'], self.params['map_size'], self.params['n_actions']))
        # TODO 统计reward的数值变化，画图分析
        # batch_size * 1
        reward = batch_memory[:, self.params['n_features'] + action_length + 1]

        # TODO reward 高度离散且稀疏怎么办, 加个log(x + 1)什么的会不会好点。高度离散的reward可能会导致mse算出来的loss波动非常大。
        # q_target[batch_index, np.argmax(eval_act_index, axis=3)] = \
        #     reward + self.params['gamma'] * np.max(q_next, axis=3)

        # TODO 下面的for太丑了，不知道怎么写的简单点
        chosen_act = np.argmax(eval_act_index, axis=3)
        next_max_q = np.max(q_next, axis=3)

        target_game_state = np.resize(batch_memory[:, -self.params['n_features']:],
                                      (self.params['batch_size'], self.params['map_size'], self.params['map_size'], 17))
        # eval_game_state = np.resize(batch_memory[:, :self.params['n_features']],
        #                             (self.params['batch_size'], self.params['map_size'], self.params['map_size'], 17))
        target_mask = np.minimum(target_game_state[:, :, :, 6], 1)[:, :, :, np.newaxis]

        for bi in batch_index:
            for i in range(self.params['map_size']):
                for j in range(self.params['map_size']):
                    q_target[bi, i, j, chosen_act[bi, i, j]] = reward[bi] + self.params['gamma'] * next_max_q[bi, i, j]

        q_target *= target_mask

        # check to replace target parameters
        if self.learn_step_counter % self.params['replace_target_iter'] == 0:
            for eval_layer, target_layer in zip(self.eval_model.layers, self.target_model.layers):
                target_layer.set_weights(eval_layer.get_weights())
            print('\ntarget_params_replaced\n')

        """
        For example in this batch I have 2 samples and 3 actions:
        q_eval =
        [[1, 2, 3],
         [4, 5, 6]]
        q_target = q_eval =
        [[1, 2, 3],
         [4, 5, 6]]
        Then change q_target with the real q_target value w.r.t the q_eval's action.
        For example in:
            sample 0, I took action 0, and the max q_target value is -1;
            sample 1, I took action 2, and the max q_target value is -2:
        q_target =
        [[-1, 2, 3],
         [4, 5, -2]]
        So the (q_target - q_eval) becomes:
        [[(-1)-(1), 0, 0],
         [0, 0, (-2)-(6)]]
        We then backpropagate this error w.r.t the corresponding action to network,
        leave other action as error=0 cause we didn't choose it.
        """

        #custom_mean_squared_error(q_target, self.eval_model.predict(batch_memory[:, :self.params['n_features']]))
        # train eval network
        custom_accuracy(q_target, self.eval_model.predict(batch_memory[:, :self.params['n_features']]))
        log = self.eval_model.train_on_batch(
            x=batch_memory[:, :self.params['n_features']],
            y=q_target,
            return_dict=True)
        size = self.params["map_size"]
        with summary_writer.as_default():
            tf.summary.scalar(f"{size}x{size}_loss", log['loss'], step=self.learn_step_counter)
            tf.summary.scalar(f"{size}x{size}_acc", log['custom_accuracy'], step=self.learn_step_counter)
        self.cost_his.append(log)

        # increasing epsilon
        # TODO 是否有其它更新epsilon的方法？原来代码的epsilon用0.9是不是太大了
        self.epsilon = self.epsilon + self.params['e_greedy_increment'] if self.epsilon < self.params['e_greedy'] \
            else self.params['e_greedy']
        self.learn_step_counter += 1


def get_game_from_obs(observation) -> Game:
    global game_state
    ### Do not edit ###
    if observation["step"] == 0:
        game_state = Game()
        game_state._initialize(observation["updates"])
        game_state._update(observation["updates"][2:])
        game_state.id = observation.player
    else:
        game_state._update(observation["updates"])

    return game_state


def DQN_agent(observation, configuration):
    global game_state, RL

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

    eval_model = EvalModel(12)
    input = tf.random.uniform([1, 12 * 12 * 17], 0, 100, dtype=tf.int32)
    _ = eval_model(input)
    eval_model.load_weights("/Users/yannik/PycharmProjects/Lux_RL/src/model_12.h5")
    target_model = TargetModel(12)
    RL = DeepQNetwork(12, 8, 12 * 12 * 17, eval_model, target_model)

    actions, _ = RL.choose_action(observation, training=False)
    #     print(translated_actions)

    return actions


if __name__ == '__main__':

    sizes = [12, 16, 24, 32]
    # sizes = [12]
    for size in sizes:
        # Inistialise the model
        summary_writer = tf.summary.create_file_writer(f'../logs/experiment_four_sizes_log/size{size}')
        eval_model = EvalModel(size)
        target_model = TargetModel(size)
        RL = DeepQNetwork(size, 8, size * size * 17, eval_model, target_model)
        step = 0
        for eps in range(100):
            clear_output()
            print(f"=== Episode {eps} ===")
            env = make("lux_ai_2021",
                       debug=True,
                       configuration={"annotations": True, "width": size, "height": size})
            trainer = env.train([None, agent_teacher])
            obs = trainer.reset()
            done = False
            game_state = get_game_from_obs(obs)
            while not done:
                action, network_output = RL.choose_action(obs, training=True)  # Action for the agent being trained.

                state_matrix = np.array(get_inputs(get_game_from_obs(obs)), float).flatten()
                obs, reward, done, info = trainer.step(action)
                #reward = game_state.players[0].city_tile_count * 3 + len(game_state.players[0].units) \
                #         + game_state.players[0].research_points / 100 + obs.step / 400
                state_matrix_ = np.array(get_inputs(get_game_from_obs(obs)), float).flatten()

                RL.store_transition(state=state_matrix, a=network_output, r=reward, state_=state_matrix_)
                # 确保有足够的训练集让我们采样？
                # if (step > 200) and (step % 5 == 0):
                with summary_writer.as_default():
                    tf.summary.scalar(f"{size}x{size}_units", len(game_state.players[obs.player].units), step)
                    tf.summary.scalar(f"{size}x{size}_reward", reward, step)
                if (step > 100) and (step % 5 == 0):
                    RL.learn()
                step += 1
            with summary_writer.as_default():
                tf.summary.scalar(f"{size}x{size}_cities", game_state.players[obs.player].city_tile_count, eps)
            #env.render()

        # Save the model
        eval_model.save_weights("conv_model_%d.h5" % size)
