#!/usr/bin/python

import numpy as np
import pickle
import json
import argparse
from inspect import getmembers, isfunction
from time import time
from tqdm import trange

import agents
from rl_engine import RLEngine
from util import *

FLAGS = None

def init_results(engine_setup, results_file=None):

    results = None
    if results_file is None:
        results = dict()
        results["epoch"] = []
        results["time"] = []
        results["overall_time"] = []
        results["mean"] = []
        results["std"] = []
        results["train_mean"] = []
        results["train_std"] = []
        results["max"] = []
        results["min"] = []
        results["epsilon"] = []
        results["training_episodes_finished"] = []
        results["loss"] = []
        results["setup"] = engine_setup
        results["best"] = None
        results["actions"] = []
        results["nmax"] = []
        results["pmax"] = []
    else:
        results = pickle.load(open(results_file, "r"))

    epoch = 1
    overall_start = time()
    best_result_so_far = None
    #if save_results and len(results["epoch"]) > 0:
    if len(results["epoch"]) > 0:
        overall_start -= results["overall_time"][-1]
        epoch = results["epoch"][-1] + 1
        best_result_so_far = results["best"]
        if "actions" not in results:
            results["actions"] = []
            for _ in len(results["epoch"]):
                results["actions"].append(0)

    return results, epoch, overall_start, best_result_so_far

def save_results(results_file, results, epoch, train_time, overall_time, test_rewards, train_rewards, epsilon, train_episodes_finished, mean_loss, best_result_so_far, steps, nmax, pmax):
    print ("Saving results to:", results_file)
    results["epoch"].append(epoch)
    results["time"].append(train_time)
    results["overall_time"].append(overall_time)
    results["mean"].append(test_rewards.mean())
    results["std"].append(test_rewards.std())
    results["max"].append(test_rewards.max())
    results["min"].append(test_rewards.min())
    results["train_mean"].append(train_rewards.mean())
    results["train_std"].append(train_rewards.std())

    results["epsilon"].append(epsilon)
    results["training_episodes_finished"].append(train_episodes_finished)
    results["loss"].append(mean_loss)
    results["best"] = best_result_so_far
    results["actions"].append(steps)
    results["nmax"].append(nmax)
    results["pmax"].append(pmax)
    res_f = open(results_file, 'wb')
    pickle.dump(results, res_f)
    res_f.close()

def main():
    engine = None
    results = None
    epoch = None
    overall_start = None
    best_result_so_far = None
    if FLAGS.agent_file:
        engine = RLEngine.load(a_agent_file[0], config_file=a_config_file[0])
        results, epoch, overall_start, best_results_so_far = init_results(engine.setup, engine.results_file)
    elif FLAGS.agent_json is not None and FLAGS.config_file is not None:
        engine_args = json.load(open(FLAGS.agent_json, "r"))
        engine_args["config_file"] = FLAGS.config_file
        engine_args["results_file"] = None
        engine_args["params_file"] = None
        engine_args["actions"] = None
        engine = RLEngine(**engine_args)
        
        results, epoch, overall_start, best_results_so_far = init_results(engine.setup)
    else:
        raise Exception("No agent json or config file specified.")

    game = engine.game
    engine.print_setup()
    print ("\n============================")


    while epoch - 1 < FLAGS.epochs:
#---------------------------- RL START ------------------------------------------------------#
        print ("\nEpoch", epoch)
        train_time = 0
        train_episodes_finished = 0
        mean_loss = 0
        #reward = 0.0
        #current_action = [0, 0, 0]
        train_rewards = []
        start = time()
        # Environment start.
        engine.new_episode(update_state=True)
        # Agent start.
        # last_action = engine.agent_start()

        print ("\nTraining ...")
        for step in trange(FLAGS.training_steps_per_epoch):
#---------------------------- RL STEP -------------------------------------------------------#
    
            # Environment step.
            #reward = game.make_action(current_action, engine_args["skiprate"] + 1)
            #reward = np.float32(reward)
            #if engine_args["shaping_on"]:
            #    sr = np.float32(doom_fixed_to_double(game.get_game_variable(GameVariable.USER1)))
            #    reward += sr - last_shaping_reward
            #    last_shaping_reward = sr
            #reward *= reward_scale
            # update_state moved to else: because it crashes when episode is finished.
            ##engine._update_state()

            # Agent step.
            #TODO move the check to RLEngine
            if game.is_episode_finished():
                r = game.get_total_reward()
                train_rewards.append(r)
                train_episodes_finished += 1
                #last_shaping_reward = 0
                #engine.agent_end(reward)
                engine.new_episode(update_state=True)
            #else:
                #engine.make_learning_step(reward)
                #engine.update_state()
                #current_action = engine.make_learning_step(reward)
                ##r = self.game.make_action(self.actions[a], self.skiprate + 1)

            engine.make_learning_step()

#----------------------------------------------------------------------------------------#
            # Should be outside of loop.
            end = time()
            train_time = end - start            

        print (train_episodes_finished, "training episodes played.")

#------------------------------ TRAIN RESULTS -------------------------------------------#

        print ("Training results:")
        print (engine.get_actions_stats(clear=True).reshape([-1, 4]))

        mean_loss = engine.get_mean_loss()

        if len(train_rewards) == 0:
            train_rewards.append(-123)
        train_rewards = np.array(train_rewards)

        print ("mean:", train_rewards.mean(), "std:", train_rewards.std(), "max:", train_rewards.max(), "min:", train_rewards.min(), "mean_loss:", mean_loss, "eps:", engine.get_epsilon())
        print ("t:", sec_to_str(train_time))

#----------------------------- TEST -----------------------------------------------------#

        # learning mode off
        new_best = False
####if test_episodes_per_epoch > 0:
        engine.training_mode = False
        test_rewards = []
        start = time()
        print ("Testing...")
        for test_episode in trange(FLAGS.test_episodes_per_epoch):
            r = engine.run_episode()
            test_rewards.append(r)
        end = time()

#---------------------------------------TEST RESULTS ------------------------------------#

        print ("Test results:")
        print (engine.get_actions_stats(clear=True, norm=False).reshape([-1, 4]))
        test_rewards = np.array(test_rewards)

        if best_result_so_far is None or test_rewards.mean() >= best_result_so_far:
            best_result_so_far = test_rewards.mean()
            new_best = True
        else:
            new_best = False
        print ("mean:", test_rewards.mean(), "std:", test_rewards.std(), "max:", test_rewards.max(), "min:", test_rewards.min())
        print ("t:", sec_to_str(end - start))
        print ("Best so far:", best_result_so_far)

        overall_end = time()
        overall_time = overall_end - overall_start

        count_max = np.count_nonzero(test_rewards == 2100)
        percent_max = count_max*100.0/FLAGS.test_episodes_per_epoch
        #print("max reward: ", count_max, ", ", count_max*100.0/FLAGS.test_episodes_per_epoch)

#--------------------------------- SAVE RESULTS ----------------------------------------#

        save_results(engine.results_file, results, epoch, train_time, overall_time, test_rewards, train_rewards, engine.get_epsilon(), train_episodes_finished, mean_loss, best_result_so_far, engine.steps, count_max, percent_max)

#------------------------------------ END -----------------------------------------#

        print ("")

        #if save_params:
        engine.save(engine.params_file + "-" + str(epoch))
        #if save_best and new_best:
        if new_best:
            engine.save(engine.params_file + "-best")

        epoch += 1
        print ("Elapsed time:", sec_to_str(overall_time))
        print ("=========================")

    overall_end = time()
    print ("Elapsed time:", sec_to_str(overall_end - overall_start))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent-file', type=str, default=None)
    parser.add_argument('--agent-json', type=str, default=None)
    parser.add_argument('--config-file', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=np.inf)
    parser.add_argument('--training-steps-per-epoch', type=int, default=200000)
    parser.add_argument('--test-episodes-per-epoch', type=int, default=300)

    FLAGS, unparsed = parser.parse_known_args()
    main()

