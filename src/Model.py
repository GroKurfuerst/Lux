import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from collections import deque
import random
import math
from tensorflow.keras import backend as K


def get_inputs(game_state):
    # Teh shape of the map
    w, h = game_state.map.width, game_state.map.height
    # The map of ressources
    M = [
        [0 if game_state.map.map[j][i].resource == None else game_state.map.map[j][i].resource.amount for i in range(w)]
        for j in range(h)]

    M = np.array(M).reshape((h, w, 1))

    # The map of units features
    # U_player = np.zeros((w, h, 5))
    U_player = [[[0, 0, 0, 0, 0] for i in range(w)] for j in range(h)]
    units = game_state.player.units
    for i in units:
        U_player[i.pos.y][i.pos.x] = [i.type, i.cooldown, i.cargo.wood, i.cargo.coal, i.cargo.uranium]
    U_player = np.array(U_player)

    U_opponent = [[[0, 0, 0, 0, 0] for i in range(w)] for j in range(h)]
    units = game_state.opponent.units
    for i in units:
        U_opponent[i.pos.y][i.pos.x] = [i.type, i.cooldown, i.cargo.wood, i.cargo.coal, i.cargo.uranium]

    U_opponent = np.array(U_opponent)

    # The map of cities featrues
    e = game_state.player.cities
    C_player = [[[0, 0, 0] for i in range(w)] for j in range(h)]
    for k in e:
        citytiles = e[k].citytiles
        for i in citytiles:
            C_player[i.pos.y][i.pos.x] = [i.cooldown, e[k].fuel, e[k].light_upkeep]
    C_player = np.array(C_player)

    e = game_state.opponent.cities
    C_opponent = [[[0, 0, 0] for i in range(w)] for j in range(h)]
    for k in e:
        citytiles = e[k].citytiles
        for i in citytiles:
            C_opponent[i.pos.y][i.pos.x] = [i.cooldown, e[k].fuel, e[k].light_upkeep]
    C_opponent = np.array(C_opponent)

    # stacking all in one array
    E = np.dstack([M, U_opponent, U_player, C_opponent, C_player])
    return E


def get_inputs_with_day_night(game_state, obs):

    def id_day_night(step):
        res = (step % 40)
        if 0 <= res < 30:
            return 1
        else:
            return 2

    temp = get_inputs(game_state)
    w, h = game_state.map.width, game_state.map.height
    day_night = np.zeros((w, h))
    day_night[:, :] = id_day_night(obs["step"])
    e = np.dstack([temp, day_night])
    assert e.size == (w, h, 18)
    return e

"""
we add two lines:

e= tf.keras.backend.max(y_true,axis = -1)
y_pred*= K.stack([e]*8, axis=-1)

to make the positions which doesn't contain neither unit or city by zero in the prediction probabilities,
in order to focus only on the main occupied positions.
"""


def custom_mean_squared_error(y_true, y_pred):

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


def units_accuracy(y_true, y_pred):
    y_units_true = y_true[:, :, :, :6]
    y_cities_true = y_true[:, :, :, 6:]

    y_units_pred = y_pred[:, :, :, :6]
    y_cities_pred = y_pred[:, :, :, 6:]

    is_unit = tf.keras.backend.max(y_units_true, axis=-1)
    y_units_pred *= K.stack([is_unit] * 6, axis=-1)
    return K.cast(K.equal(y_units_true, K.round(y_units_pred)), "float32") / K.sum(is_unit)


def cities_accuracy(y_true, y_pred):
    y_units_true = y_true[:, :, :, :6]
    y_cities_true = y_true[:, :, :, 6:]

    y_units_pred = y_pred[:, :, :, :6]
    y_cities_pred = y_pred[:, :, :, 6:]

    is_city = tf.keras.backend.max(y_cities_true, axis=-1)
    y_cities_pred *= K.stack([is_city] * 2, axis=-1)

    return K.cast(K.equal(y_cities_true, K.round(y_cities_pred)), "float32") / K.sum(is_city)


def get_model(s):
    inputs = keras.Input(shape=(s, s, 17), name='The game map')
    f = layers.Flatten()(inputs)
    h, w = s, s
    f = layers.Dense(w * h, activation="sigmoid")(f)
    f = layers.Reshape((h, w, -1))(f)
    units = layers.Dense(6, activation="softmax", name="Units_actions")(f)
    cities = layers.Dense(2, activation="sigmoid", name="Cities_actions")(f)
    # units = layers.Dense(6,  name="Units_actions")(f)
    # cities = layers.Dense(2, name="Cities_actions")(f)
    output = layers.Concatenate()([units, cities])
    model = keras.Model(inputs=inputs, outputs=output)
    model.compile(optimizer="adam", loss=custom_mean_squared_error, metrics=["accuracy"])

    return model


model = get_model(12)
model.summary()


class EvalModel(tf.keras.Model):
    def __init__(self, size):
        super().__init__('mlp_q_network')
        # self.squeeze_input = layers.Flatten()
        self.feature_extractor = layers.Dense(size * size, activation='sigmoid')
        self.squeeze_feature = layers.Reshape((size, size, -1))
        self.unit_action = layers.Dense(6, activation=None, name="Units_actions")
        self.city_action = layers.Dense(2, activation=None, name="Cities_actions")
        self.cat = layers.Concatenate()

    def call(self, inputs):
        # x = self.squeeze_input(tf.convert_to_tensor(inputs))
        x = tf.convert_to_tensor(inputs)
        feature = self.feature_extractor(x)
        squeezed_feature = self.squeeze_feature(feature)
        unit_action_prob = self.unit_action(squeezed_feature)
        city_action_prob = self.city_action(squeezed_feature)
        output = self.cat([unit_action_prob, city_action_prob])
        return output


class TargetModel(tf.keras.Model):
    def __init__(self, size):
        super().__init__('mlp_target_network')
        # self.squeeze_input = layers.Flatten()
        self.feature_extractor = layers.Dense(size * size, activation='sigmoid', trainable=False)
        self.squeeze_feature = layers.Reshape((size, size, -1))
        self.unit_action = layers.Dense(6, activation=None, name="Units_actions", trainable=False)
        self.city_action = layers.Dense(2, activation=None, name="Cities_actions", trainable=False)
        self.cat = layers.Concatenate()

    def call(self, inputs):
        # x = self.squeeze_input(tf.convert_to_tensor(inputs))
        x = tf.convert_to_tensor(inputs)

        feature = self.feature_extractor(x)
        squeezed_feature = self.squeeze_feature(feature)
        unit_action_prob = self.unit_action(squeezed_feature)
        city_action_prob = self.city_action(squeezed_feature)
        output = self.cat([unit_action_prob, city_action_prob])
        return output


class EvalModel_Conv(tf.keras.Model):
    def __init__(self, size):
        super().__init__('mlp_q_network')
        # self.squeeze_input = layers.Flatten()
        self.reshape = layers.Reshape((size, size, -1))
        self.squeeze_feature = layers.Conv2D(8, 1, activation="relu", name="squeeze_feature")
        self.conv1 = layers.Conv2D(8, 3, activation="relu", name="conv1", padding="same")
        self.conv2 = layers.Conv2D(8, 3, activation="relu", name="conv2", padding="same")

        self.unit_conv = layers.Conv2D(6, 3, name="unit_conv", padding="same")
        self.city_conv = layers.Conv2D(2, 3, name="city_conv", padding="same")

        self.unit_action = layers.Dense(6, activation=None, name="Units_actions")
        self.city_action = layers.Dense(2, activation=None, name="Cities_actions")

        self.cat = layers.Concatenate()

    def call(self, inputs):
        # x = self.squeeze_input(tf.convert_to_tensor(inputs))
        x = tf.convert_to_tensor(inputs)
        x = self.reshape(x)
        x = self.squeeze_feature(x)
        x = self.conv1(x)
        x = self.conv2(x)

        unit_output = self.unit_conv(x)
        city_output = self.city_conv(x)

        unit_action_q = self.unit_action(unit_output)
        city_action_q = self.city_action(city_output)

        return self.cat([unit_action_q, city_action_q])


class TargetModel_Conv(tf.keras.Model):

    def __init__(self, size):
        super().__init__('mlp_q_network')
        # self.squeeze_input = layers.Flatten()
        self.reshape = layers.Reshape((size, size, -1))
        self.squeeze_feature = layers.Conv2D(8, 1, activation="relu", name="squeeze_feature", trainable=False)
        self.conv1 = layers.Conv2D(8, 3, activation="relu", name="conv1", padding="same", trainable=False)
        self.conv2 = layers.Conv2D(8, 3, activation="relu", name="conv2", padding="same", trainable=False)

        self.unit_conv = layers.Conv2D(6, 3, name="unit_conv", padding="same", trainable=False)
        self.city_conv = layers.Conv2D(2, 3, name="city_conv", padding="same", trainable=False)

        self.unit_action = layers.Dense(6, activation=None, name="Units_actions", trainable=False)
        self.city_action = layers.Dense(2, activation=None, name="Cities_actions", trainable=False)

        self.cat = layers.Concatenate()

    def call(self, inputs):
        # x = self.squeeze_input(tf.convert_to_tensor(inputs))
        x = tf.convert_to_tensor(inputs)
        x = self.reshape(x)
        x = self.squeeze_feature(x)
        x = self.conv1(x)
        x = self.conv2(x)

        unit_output = self.unit_conv(x)
        city_output = self.city_conv(x)

        unit_action_q = self.unit_action(unit_output)
        city_action_q = self.city_action(city_output)

        return self.cat([unit_action_q, city_action_q])

