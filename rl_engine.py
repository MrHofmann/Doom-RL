import numpy as np
import cv2
import itertools as it
import pickle
import random
from math import log, floor, ceil
from time import sleep
from pydoc import locate

import queue
import os
import torch
from vizdoom import *
import approximator
from replay_memory import UniformReplay, PrioritizedReplay
from util import doom_debug

#BITS_FOR_COUNT = 16

class RLEngine:
    def __init__(self, **kwargs):
        self.setup = kwargs
        if kwargs["network_args"] is None:
            network_args = dict()
        else:
            network_args = kwargs["network_args"]

        self.name = kwargs["name"]
        self.batchsize = kwargs["batchsize"]
        self.history_length = max(kwargs["history_length"], 1)
        self.update_pattern = kwargs["update_pattern"]
        self.epsilon = max(min(kwargs["start_epsilon"], 1.0), 0.0)
        self.end_epsilon = min(max(kwargs["end_epsilon"], 0.0), self.epsilon)
        self.epsilon_decay_stride = (self.epsilon - kwargs["end_epsilon"]) / kwargs["epsilon_decay_steps"]
        self.epsilon_decay_start = kwargs["epsilon_decay_start_step"]
        self.skiprate = max(kwargs["skiprate"], 0)
        self.steps = 0
        self.melt_interval = kwargs["melt_interval"]
        self.backprop_start_step = max(kwargs["backprop_start_step"], kwargs["batchsize"])
        self.one_hot_nactions = kwargs["one_hot_nactions"]
        self.agent_type = kwargs["agent_type"]
        self.gamma = network_args["gamma"]
        self.nstep = network_args["nstep"]
        #self.last_shaping_reward = 0
        #self.shaping_on = shaping_on
        self.training_mode = True

        self.r_n = []
        self.s_n = []
        self.a_n = []
        self.s2_n = []
        self.t_n = []

        self._env_init(kwargs["config_file"], kwargs["results_file"], kwargs["params_file"], kwargs["name"])
        if self.game.get_available_game_variables_size() > 0 and kwargs["use_game_variables"]:
            self.use_game_variables = True
        else:
            self.use_game_variables = False
        if kwargs["actions"] is None:
            self.actions = self._generate_default_actions(self.game)
        else:
            self.actions = kwargs["actions"]
        self.actions_num = len(self.actions)
        self.actions_stats = np.zeros([self.actions_num], np.int)

        img_shape = self._img_init(kwargs["reshaped_x"], kwargs["reshaped_y"])
        total_misc_len = self._misc_init(kwargs["remember_n_actions"])        
        state_format = dict()
        state_format["s_img"] = img_shape
        state_format["s_misc"] = total_misc_len
        network_args = kwargs["network_args"]
        network_args["state_format"] = state_format
        network_args["actions_number"] = len(self.actions)
        replay_memory_size = kwargs["replay_memory_size"]
        batchsize = kwargs["batchsize"]
        self._agent_init(state_format, network_args, kwargs["replay_memory_size"], kwargs["batchsize"])
        
        if "game" in kwargs:
            del kwargs["game"]

    def _generate_default_actions(self, the_game):
        n = the_game.get_available_buttons_size()

        actions = []
        for perm in it.product([0, 1], repeat=n):
            actions.append(list(perm))
        
        return actions

    def _prepare_for_save(self):
        self.setup["epsilon"] = self.epsilon
        self.setup["steps"] = self.steps
        self.setup["skiprate"] = self.skiprate

    def _env_init(self, config_file, results_file, params_file, name, game=None, grayscale=True, visible=False):
        
        if game is not None:
            self.game = game
            self.config_file = None
        elif config_file is not None:
            self.config_file = config_file
            self.game = DoomGame()
            #self.game.load_config("common.cfg")
            self.game.load_config(config_file)
            self.game.set_window_visible(visible)
            self.game.set_doom_game_path("wads/doom2.wad")
            self.game.set_render_hud(False)

            if grayscale:
                self.game.set_screen_format(ScreenFormat.GRAY8)

            print("Initializing DOOM ...")
            self.game.init()
            print("DOOM initialized.")

        else:
            raise Exception("Failed to initialize game. No game or config file specified.")

        if results_file:
            self.results_file = results_file
        else:
            self.results_file = "results/" + name + ".res"
        if params_file:
            self.params_file = params_file
        else:
            self.params_file = "params/" + name


    def _img_init(self, reshaped_x, reshaped_y):
        # changes img_shape according to the history size
        self.channels = self.game.get_screen_channels()
        if self.history_length > 1:
            self.channels *= self.history_length

        if reshaped_x is None:
            x = self.game.get_screen_width()
            y = self.game.get_screen_height()
            scale_x = scale_y = 1.0
        else:
            x = reshaped_x
            scale_x = float(x) / self.game.get_screen_width()

            if reshaped_y is None:
                y = int(self.game.get_screen_height() * scale_x)
                scale_y = scale_x
            else:
                y = reshaped_y
                scale_y = float(y) / self.game.get_screen_height()

        img_shape = [self.channels, y, x]

        # Check if this is slow. It seems that it isn't.
        if scale_x == 1 and scale_y == 1:
            def convert(img):
                img = img.astype(np.float32) / 255.0
                return img
        else:
            # Does not work with RGB.
            def convert(img):
                img = img.astype(np.float32) / 255.0
                #new_image = np.ndarray([img.shape[0], y, x], dtype=img.dtype)
                #for i in range(img.shape[0]):
                #    # new_image[i] = skimage.transform.resize(img[i], (y,x), preserve_range=True)
                #    new_image[i] = cv2.resize(img[i], (x, y), interpolation=cv2.INTER_AREA)
                new_image = np.ndarray([1, y, x], dtype=img.dtype)
                new_image = cv2.resize(img, (x, y), interpolation=cv2.INTER_AREA)
                return new_image
        self.convert_image = convert

        self.current_image_state = np.zeros(img_shape, dtype=np.float32)
        return img_shape

    def _misc_init(self, remember_n_actions):
        if self.use_game_variables:
            single_state_misc_len = int(self.game.get_available_game_variables_size())
        else:
            single_state_misc_len = 0
        self.single_state_misc_len = single_state_misc_len

        self.remember_n_actions = remember_n_actions
        total_misc_len = int(single_state_misc_len * self.history_length)

        if remember_n_actions > 0:
            self.remember_n_actions = remember_n_actions
            if self.one_hot_nactions:
                self.action_len = int(2 ** floor(log(len(self.actions), 2)))
            else:
                self.action_len = len(self.actions[0])
            self.last_action = np.zeros([self.action_len], dtype=np.float32)
            self.last_n_actions = np.zeros([remember_n_actions * self.action_len], dtype=np.float32)
            total_misc_len += len(self.last_n_actions)

        if total_misc_len > 0:
            self.misc_state_included = True
            self.current_misc_state = np.zeros(total_misc_len, dtype=np.float32)
        else:
            self.misc_state_included = False

        return total_misc_len

    def _agent_init(self, state_format, network_args, replay_memory_size, batchsize):
        if self.agent_type == "dqn":
            self.approximator = approximator.DQN(**network_args)
            self.replay_memory = UniformReplay(state_format, replay_memory_size, batchsize)
        elif self.agent_type == "double":
            self.approximator = approximator.DoubleDQN(**network_args)
            self.replay_memory = UniformReplay(state_format, replay_memory_size, batchsize)
        elif self.agent_type == "dueling":
            self.approximator = approximator.DuelingDQN(**network_args)
            self.replay_memory = UniformReplay(state_format, replay_memory_size, batchsize)
        elif self.agent_type == "prioritized":
            self.approximator = approximator.PrioritizedDQN(**network_args)
            self.replay_memory = PrioritizedReplay(state_format, replay_memory_size, batchsize)
        elif self.agent_type == "nstep":
            self.approximator = approximator.NStepDQN(**network_args)
            self.replay_memory = UniformReplay(state_format, replay_memory_size, batchsize)
        elif self.agent_type == "integrated":
            self.approximator = approximator.IntegratedDQN(**network_args)
            self.replay_memory = PrioritizedReplay(state_format, replay_memory_size, batchsize)
        elif self.agent_type == "ndouble":
            self.approximator = approximator.NDoubleDQN(**network_args)
            self.replay_memory = PrioritizedReplay(state_format, replay_memory_size, batchsize)
        elif self.agent_type == "ndueling":
            self.approximator = approximator.NDuelingDQN(**network_args)
            self.replay_memory = PrioritizedReplay(state_format, replay_memory_size, batchsize)        
        elif self.agent_type == "nnstep":
            self.approximator = approximator.NNStepDQN(**network_args)
            self.replay_memory = PrioritizedReplay(state_format, replay_memory_size, batchsize)
        elif self.agent_type == "nprioritized":
            self.approximator = approximator.NPrioritizedDQN(**network_args)
            self.replay_memory = UniformReplay(state_format, replay_memory_size, batchsize)
        else:
            if locate('approximator.' + agent_type) is not None:
                self.approximator = locate('approximator.' + agent_type)(**network_args)
            else:
                raise Exception("Invalid agent type.")

    def _update_state(self):
        raw_state = self.game.get_state()
        img = self.convert_image(raw_state.screen_buffer)
        state_misc = None
        if self.single_state_misc_len > 0:
            state_misc = np.zeros(self.single_state_misc_len, dtype=np.float32)
            if self.use_game_variables:
                game_variables = raw_state.game_variables.astype(np.float32)
                state_misc[0:len(game_variables)] = game_variables

        if self.history_length > 1:
            pure_channels = self.channels // self.history_length
            self.current_image_state[0:-pure_channels] = self.current_image_state[pure_channels:]
            self.current_image_state[-pure_channels:] = img

            if self.single_state_misc_len > 0:
                misc_len = len(state_misc)
                hist_len = self.history_length

                # TODO don't move count_time when it's one hot - it's useless and performance drops slightly
                #if self.rearrange_misc:
                #    for i in range(misc_len):
                #        cms_part = self.current_misc_state[i * hist_len:(i + 1) * hist_len]
                #        cms_part[0:hist_len - 1] = cms_part[1:]
                #        cms_part[-1] = state_misc[i]
                #else:
                cms = self.current_misc_state
                cms[0:(hist_len - 1) * misc_len] = cms[misc_len:hist_len * misc_len]
                cms[(hist_len - 1) * misc_len:hist_len * misc_len] = state_misc

        else:
            self.current_image_state[:] = img
            if self.single_state_misc_len > 0:
                self.current_misc_state[0:len(state_misc)] = state_misc

        if self.remember_n_actions:
            self.last_n_actions[:-self.action_len] = self.last_n_actions[self.action_len:]

            self.last_n_actions[-self.action_len:] = self.last_action
            self.current_misc_state[-len(self.last_n_actions):] = self.last_n_actions

    def new_episode(self, update_state=False):
        self.game.new_episode()
        self.reset_state()
        #self.last_shaping_reward = 0
        if update_state:
            self._update_state()

        self.s_n.clear()
        self.a_n.clear()
        self.r_n.clear()
        self.s2_n.clear()
        self.t_n.clear()

    def set_last_action(self, index):
        if self.one_hot_nactions:
            self.last_action.fill(0)
            self.last_action[index] = 1
        else:
            self.last_action[:] = self.actions[index]

    # Return current state including history
    def _current_state(self):
        if self.misc_state_included:
            s = [self.current_image_state, self.current_misc_state]
        else:
            s = [self.current_image_state]
        return s

    # Return current state's COPY including history.
    def _current_state_copy(self):
        if self.misc_state_included:
            s = [self.current_image_state.copy(), self.current_misc_state.copy()]
        else:
            s = [self.current_image_state.copy()]
        return s

    # Sets the whole state to zeros.
    def reset_state(self):
        self.current_image_state.fill(0.0)

        if self.misc_state_included:
            self.current_misc_state.fill(0.0)
            if self.remember_n_actions > 0:
                self.set_last_action(0)
                self.last_n_actions.fill(0)

    def make_step(self):
        self._update_state()
        # TODO Check if not making the copy still works
        a = self.approximator.estimate_best_action(self._current_state_copy())
        self.actions_stats[a] += 1
        self.game.make_action(self.actions[a], self.skiprate + 1)
        if self.remember_n_actions:
            self.set_last_action(a)

    def make_sleep_step(self, sleep_time=1 / 35.0):
        self._update_state()
        a = self.approximator.estimate_best_action(self._current_state_copy())
        self.actions_stats[a] += 1

        self.game.set_action(self.actions[a])
        if self.remember_n_actions:
            self.set_last_action(a)
        for i in range(self.skiprate):
            self.game.advance_action(1, True)
            sleep(sleep_time)
        self.game.advance_action()

        sleep(sleep_time)

    def check_timeout(self):
        return (self.game.get_episode_time() - self.game.get_episode_start_time() >= self.game.get_episode_timeout())

    def agent_start(self):
        start = doom_debug.start("DoomAgent::agent_start", 1)

        # Maybe current state should be deep coppied. Try both options.
        self.last_state = self._current_state_copy()
        
        # epsilon decay. This is ok, but should be part of a policy function.
        if self.steps > self.epsilon_decay_start and self.epsilon > self.end_epsilon:
            self.epsilon = max(self.epsilon - self.epsilon_decay_stride, 0)

        # This is ok, but should be part of a policy function.
        # With probability epsilon choose a random action:
        if self.epsilon >= random.random():
            a = random.randint(0, len(self.actions) - 1)
        else:
            a = self.approximator.estimate_best_action(self.last_state)
        self.actions_stats[a] += 1
        
        if self.remember_n_actions:
            self.set_last_action(a)

        doom_debug.end("DoomAgent::agent_start", 1, start)
        return self.actions[a]

    def agent_learning_step(self, reward):
        start = doom_debug.start("DoomAgent::agent_learning_step", 1)

        # This is ok, but should be part of a policy function.
        self.steps += 1
        # epsilon decay
        if self.steps > self.epsilon_decay_start and self.epsilon > self.end_epsilon:
            self.epsilon = max(self.epsilon - self.epsilon_decay_stride, 0)

        # This should be replaced with remembering it as last state at the end.
        # Copy because state will be changed in a second
        #s = self._current_state_copy();

        # This is ok, but should be part of a policy function.
        # With probability epsilon choose a random action:
        if self.epsilon >= random.random():
            a = random.randint(0, len(self.actions) - 1)
        else:
            a = self.approximator.estimate_best_action(self.last_state)
        self.actions_stats[a] += 1

        # Copy because state will be changed in a second
        #s = self._current_state_copy();
        action = 0
        for i in range(len(self.last_action)):
            if self.last_action[i] == 1:
                action = i
                break

        # Maybe last state should be deep coppied here.
        #s2 = self._current_state()                                                                      #
        #self.replay_memory.add_transition(s, a, s2, r, terminal=False)
        self.replay_memory.add_transition(self.last_state, action, self._current_state(), reward, terminal=False)

        # From here and below, this has to go out. It belongs to the environment.#########################
        #r = self.game.make_action(self.actions[a], self.skiprate + 1)                                   #
        #r = np.float32(r)                                                                               #
        #if self.shaping_on:                                                                             #
        #    sr = np.float32(doom_fixed_to_double(self.game.get_game_variable(GameVariable.USER1)))      #
        #    r += sr - self.last_shaping_reward                                                          #
        #    self.last_shaping_reward = sr                                                               #
        #                                                                                                #
        #r *= self.reward_scale                                                                          #
                                                                                                         #
        # Update state goes out, it belongs to environment. Add transition goes up after policy/estim.   #
        # update state s2 accordingly and add transition                                                 #
        #if self.game.is_episode_finished():                                                             #
        #    if (not self.no_timeout_terminal) or (not self.check_timeout()):                            #
        #        s2 = None                                                                               #
        #        self.replay_memory.add_transition(s, a, s2, r, terminal=True)                           #
        #else:                                                                                           #
        # Update state has no effect because action has not been forwarded to environment yet.           #
        # Replace this with remembering self.last_state and move update after when action is made        #
        #self._update_state()                                                                            #
        #s2 = self._current_state()                                                                      #
        #self.replay_memory.add_transition(s, a, s2, r, terminal=False)###################################

        # This can go before sampling and back propagation, but it can also stay here.
        # Melt the network sometimes
        if self.steps % self.melt_interval == 0:
            self.approximator.melt()

        # This is ok.
        # Perform q-learning once for a while
        if self.replay_memory.size >= self.backprop_start_step and self.steps % self.update_pattern[0] == 0:

            #self.save_replay()
            batch = self.replay_memory.get_sample()
            print(batch['s1_img'].shape)
            print(batch['s1_misc'].shape)
            print(batch['a'].shape)
            print(batch['s2_img'].shape)
            print(batch['s2_misc'].shape)
            print(batch['r'].shape)
            print(batch['nonterminal'].shape)
            for i in range(self.update_pattern[1]):
                self.approximator.learn(self.replay_memory.get_sample())

        self.last_state = self._current_state_copy()
        if self.remember_n_actions:
            self.set_last_action(a)
       
        doom_debug.end("DoomAgent::agent_learning_step", 1, start)
        return self.actions[a]

    def agent_end(self, reward):
        start = doom_debug.start("DoomAgent::agent_end", 1)

        self.steps += 1
        # This should be replaced with remembering it as last state at the end.
        # Copy because state will be changed in a second
        #s = self._current_state_copy();
        a = 0
        for i in range(len(self.last_action)):
            if self.last_action[i] == 1:
                a = i
                break
        r = reward        
        # update state s2 accordingly and add transition
        if (not self.no_timeout_terminal) or (not self.check_timeout()):
            s2 = None
            self.replay_memory.add_transition(self.last_state, a, s2, r, terminal=True)

        # This can go before sampling and back propagation, but it can also stay here.
        # Melt the network sometimes
        if self.steps % self.melt_interval == 0:
            self.approximator.melt()

        # This is ok.
        # Perform q-learning once for a while
        if self.replay_memory.size >= self.backprop_start_step and self.steps % self.update_pattern[0] == 0:
            for i in range(self.update_pattern[1]):
                self.approximator.learn(self.replay_memory.get_sample())

        doom_debug.end("DoomAgent::agent_end", 1, start)

    # Performs a learning step according to epsilon-greedy policy.
    # The step spans self.skiprate +1 actions.
    def make_learning_step(self):
        self.steps += 1
        # epsilon decay
        if self.steps > self.epsilon_decay_start and self.epsilon > self.end_epsilon:
            self.epsilon = max(self.epsilon - self.epsilon_decay_stride, 0)

            # Copy because state will be changed in a second
        s = self._current_state_copy();

        # With probability epsilon choose a random action:
        if self.epsilon >= random.random():
            a = random.randint(0, len(self.actions) - 1)
        else:
            a = self.approximator.estimate_best_action(s)
        self.actions_stats[a] += 1

        # make action and get the reward
        if self.remember_n_actions:
            self.set_last_action(a)

        r = self.game.make_action(self.actions[a], self.skiprate + 1)
        r = np.float32(r)
        #if self.shaping_on:
        #    sr = np.float32(doom_fixed_to_double(self.game.get_game_variable(GameVariable.USER1)))
        #    r += sr - self.last_shaping_reward
        #    self.last_shaping_reward = sr

        #r *= self.reward_scale

        # update state s2 accordingly and add transition
        if self.game.is_episode_finished():
            s2 = None
            t = True
        else:
            self._update_state()
            s2 = self._current_state()
            t = False

        if len(self.r_n) < self.nstep:
            self.s_n.append(s)
            self.a_n.append(a)
            self.r_n.append(r)
            self.s2_n.append(s2)
            self.t_n.append(t)
        elif s2 is None:
            idx = (self.steps-1) % self.nstep
            for i in range(self.nstep):
                r_t = 0.0
                for j in range(i, self.nstep):
                    r_t = self.r_n[idx-(j+1)] + self.gamma*r_t
                    self.replay_memory.add_transition(self.s_n[idx-(j+1)], self.a_n[idx-(j+1)], s2, r_t, t)
        # Add \gamma^n to target update as well
        else:
            idx = (self.steps-1) % self.nstep
            r_t = 0.0
            for i in range(self.nstep):
                r_t = self.r_n[idx-(i+1)] + self.gamma*r_t
            self.replay_memory.add_transition(self.s_n[idx], self.a_n[idx], s2, r_t, t)
            self.s_n[idx] = s
            self.a_n[idx] = a
            self.r_n[idx] = r
            self.s2_n[idx] = s2
            self.t_n[idx] = t
        
        # Perform q-learning once for a while
        if self.replay_memory.size >= self.backprop_start_step and self.steps % self.update_pattern[0] == 0:
            for a in range(self.update_pattern[1]):
                if self.agent_type in ["prioritized", "integrated", "ndouble", "ndueling", "nnstep"]:
                    b_idx, transitions, ISWeights = self.replay_memory.get_sample(self.batchsize)
                    abs_errors = self.approximator.learn(transitions, ISWeights)
                    self.replay_memory.batch_update(b_idx, abs_errors)
                else:
                    self.approximator.learn(self.replay_memory.get_sample())

        # Melt the network sometimes
        if self.steps % self.melt_interval == 0:
            self.approximator.melt()

    # Runs a single episode in current mode. It ignores the mode if learn==true/false
    def run_episode(self, sleep_time=0):
        self.new_episode()
        if sleep_time == 0:
            while not self.game.is_episode_finished():
                self.make_step()
        else:
            while not self.game.is_episode_finished():
                self.make_sleep_step(sleep_time)

        return np.float32(self.game.get_total_reward())

    # Utility stuff
    def get_actions_stats(self, clear=False, norm=True):
        stats = self.actions_stats.copy()
        if norm:
            stats = stats / np.float32(self.actions_stats.sum())
            stats[stats == 0.0] = -1
            stats = np.around(stats, 3)

        if clear:
            self.actions_stats.fill(0)
        return stats

    def get_steps(self):
        return self.steps

    def get_epsilon(self):
        return self.epsilon

    def get_network(self):
        return self.approximator.network

    def set_epsilon(self, eps):
        self.epsilon = eps

    def set_skiprate(self, skiprate):
        self.skiprate = max(skiprate, 0)

    def get_skiprate(self):
        return self.skiprate

    def get_mean_loss(self):
        return self.approximator.get_mean_loss()

    # Saves network weights to a file
    def save_params(self, filename, quiet=False):
        if not quiet:
            print("Saving network weights to " + filename + "...")
        self._prepare_for_save()
        torch.save(self.approximator.network, filename)
        if not quiet:
            print("Saving finished.")

    # Loads network weights from the file
    def load_params(self, filename, quiet=False):
        if not quiet:
            print("Loading network weights from " + filename + "...")
        self.approximator.network = torch.load(filename)
        self.approximator.frozen_network = torch.load(filename)

        if not quiet:
            print("Loading finished.")

            # Loads the whole engine with params from file

    def get_network_architecture(self):
        return self.get_network().state_dict()

    def print_setup(self):
        print("\nNetwork architecture:")
        for p in self.get_network_architecture():
            print(p)

    @staticmethod
    def load(filename, game=None, config_file=None, quiet=False):
        if not quiet:
            print("Loading qengine from " + filename + "...")

        params = pickle.load(open(filename, "rb"))

        qengine_args = params[0]
        network_weights = params[1]

        steps = qengine_args["steps"]
        epsilon = qengine_args["epsilon"]
        del (qengine_args["epsilon"])
        del (qengine_args["steps"])
        if game is None:
            if config_file is not None:
                game = initialize_doom(config_file)
                qengine_args["config_file"] = config_file
            elif "config_file" in qengine_args and qengine_args["config_file"] is not None:
                game = initialize_doom(qengine_args["config_file"])
            else:
                raise Exception("No game, no config file. Dunno how to initialize doom.")
        else:
            qengine_args["config_file"] = None

        qengine_args["game"] = game
        qengine = QEngine(**qengine_args)
        qengine.approximator.network.load_state_dict(network_weights)
        qengine.approximator.frozen_network.load_state_dict(network_weights)


        if not quiet:
            print("Loading finished.")
            qengine.steps = steps
            qengine.epsilon = epsilon
        return qengine

    # Saves the whole engine with params to a file
    def save(self, filename=None, quiet=False):
        if filename is None:
            filename = self.params_file
        if not quiet:
            print("Saving qengine to " + filename + "...")
        self._prepare_for_save()
        network_params = self.approximator.network.state_dict()
        params = [self.setup, network_params]
        pickle.dump(params, open(filename, "wb"))
        if not quiet:
            print("Saving finished.")
