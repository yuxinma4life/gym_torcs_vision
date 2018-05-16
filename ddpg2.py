from gym_torcs import TorcsEnv
import numpy as np
import random
import argparse
from keras.models import model_from_json, Model
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
import tensorflow as tf
#from keras.engine.training import collect_trainable_weights
import json

from ReplayBuffer import ReplayBuffer
from ActorNetwork2 import ActorNetwork
from CriticNetwork2 import CriticNetwork
from OU import OU
import timeit
import cv2
import skimage as skimage
from collections import deque
import matplotlib.pyplot as plt
import time
from helper import *
import os                  
import math
import resource
import gc
import objgraph

PLOTGRAPH = 1
SCREEN_WIDTH = 256
SCREEN_HEIGHT = 192
OU = OU()       #Ornstein-Uhlenbeck Process

def save(actor, critic, episode_frames, total_reward, step):
    actor.model.save_weights("actormodel"+str(step)+".h5", overwrite=True)
    with open("actormodel"+str(step)+".json", "w") as outfile:
        json.dump(actor.model.to_json(), outfile)

    critic.model.save_weights("criticmodel"+str(step)+".h5", overwrite=True)
    with open("criticmodel"+str(step)+".h5", "w") as outfile:
        json.dump(critic.model.to_json(), outfile)


    time_per_step = 0.05
    images = np.array(episode_frames)
    #make_gif(images, './frames/image' + str(step) +'_reward_' + str(total_reward) + '.gif',
    #                     duration=len(images) * time_per_step, true_image=True, salience=False)

def playGame(train_indicator=0):    #1 means Train, 0 means simply Run
    BUFFER_SIZE = 100000
    BATCH_SIZE = 32
    GAMMA = 0.99
    TAU = 0.001     #Target Network HyperParameters
    LRA = 0.00008    #Learning rate for Actor
    LRC = 0.0008     #Lerning rate for Critic

    action_dim = 2  #Steering/Acceleration/Brake
    state_dim = (SCREEN_HEIGHT, SCREEN_WIDTH, 1)

    np.random.seed(1337)

    vision = True

    EXPLORE = 100000.
    episode_count = 2000
    warmup_episode_count = 0
    max_steps = 100000
    reward = 0
    done = False
    step = 0
    epsilon = 1
    indicator = 0
    done_count = 0

    #Tensorflow GPU optimization
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    tf.python.control_flow_ops = tf
    sess = tf.Session(config=config)
    from keras import backend as K
    K.set_session(sess)

    actor = ActorNetwork(sess, state_dim, action_dim, BATCH_SIZE, TAU, LRA)
    critic = CriticNetwork(sess, state_dim, action_dim, BATCH_SIZE, TAU, LRC)
    buff = ReplayBuffer(BUFFER_SIZE)    #Create replay buffer

    # Generate a Torcs environment
    env = TorcsEnv(vision=vision, throttle=True,gear_change=False, port=3101)

    #Now load the weight
    print("Now we load the weight")
    try:
        actor.model.load_weights("actormodel.h5")
        critic.model.load_weights("criticmodel.h5")
        actor.target_model.load_weights("actormodel.h5")
        critic.target_model.load_weights("criticmodel.h5")
        print("Weight load successfully")
    except:
        print("Cannot find the weight")

    print("TORCS Experiment Start.")

    if PLOTGRAPH == 1:
        # Create plots for reward and loss
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111)
        # fig2 = plt.figure()
        # ax2 = fig2.add_subplot(111)
        plt.ion()


    episode_frames = []
    for i in range(episode_count):
        del episode_frames[:]
        print("Episode : " + str(i) + " Replay Buffer " + str(buff.count()))

        if np.mod(i, 25) == 0:
            ob = env.reset(relaunch=True)   #relaunch TORCS every 3 episode because of the memory leak error
        else:
            ob = env.reset()


        # Create buffer of four most recent images
        s_t = np.reshape(ob.img, [1,SCREEN_HEIGHT,SCREEN_WIDTH,1])
        #to_gif = np.reshape(ob.img, (SCREEN_HEIGHT, SCREEN_WIDTH, 1))
        #episode_frames.append(to_gif)

        total_reward = 0.
        for j in range(max_steps):
            epsilon -= 1.0 / EXPLORE
            a_t = np.zeros([1,action_dim])
            noise_t = np.zeros([1,action_dim])
            
            a_t_original = actor.model.predict(s_t)

            noise_t[0][0] = train_indicator * max(epsilon, 0.1) * OU.function(a_t_original[0][0],  0.0 , 0.60, 0.30)
            noise_t[0][1] = train_indicator * max(epsilon, 0.1) * OU.function(a_t_original[0][1],  0.1 , 1.0, 0.30)
            #The following code do the stochastic brake
            #if random.random() <= 0.1:
            #    print("********Now we apply the brake***********")
            #    noise_t[0][2] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][2],  0.2 , 1.00, 0.10)

            a_t[0][0] = a_t_original[0][0] + noise_t[0][0]
            a_t[0][1] = a_t_original[0][1] + noise_t[0][1]

            #print("a_t_original", a_t_original, "a_t", a_t)

            ob, r_t, done, info = env.step(a_t[0])

            #r_t = r_t/1000

            s_t1 = np.reshape(ob.img,[1,SCREEN_HEIGHT,SCREEN_WIDTH,1])

            #to_gif = np.reshape(ob.img, (SCREEN_HEIGHT, SCREEN_WIDTH, 1))
            #episode_frames.append(to_gif)

            buff.add(s_t, a_t[0], r_t, s_t1, done)      #Add replay buffer
            
            total_reward += r_t

            s_t = s_t1
        
            #print("Episode", i, "Step", step, "Action", a_t, "Reward", r_t, "Total Reward", total_reward)

            step += 1
            if done:
                if r_t == 200 and train_indicator == 1:
                    print("Episode", i, "COMPLETED")
                    if np.mod(done_count, 5) == 0:
                        save(actor, critic, episode_frames, total_reward, i)
                    done_count += 1
                break

        #gc.collect()
        #print('Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        #objgraph.show_most_common_types()

        if (train_indicator == 0 or warmup_episode_count > buff.count()):
            continue

        # train after one episode to avoid losing frames in the game
        # (i.e. CNN takes too long =~ 0.3s per round)
        # train after episode is complete
        # assuming that we are training every 4 steps in the episode
        #batchrounds = math.ceil(j / 4)
        batchrounds = j;

        print("batch run", batchrounds, "times")

        total_loss = 0.
        for m in range(batchrounds):
            #Do the batch update
            batch = buff.getBatch(BATCH_SIZE)
            states = np.asarray([e[0][0] for e in batch])
            actions = np.asarray([e[1] for e in batch])
            rewards = np.asarray([e[2] for e in batch])
            # print e[3][0].shape
            new_states = np.asarray([e[3][0] for e in batch])
            dones = np.asarray([e[4] for e in batch])
            y_t = np.asarray([e[1] for e in batch])
            
            target_q_values = critic.target_model.predict([new_states, actor.target_model.predict(new_states)])  
            
            for k in range(len(batch)):
                if dones[k]:
                    y_t[k] = rewards[k]
                else:
                    y_t[k] = rewards[k] + GAMMA*target_q_values[k]
                
            #start = time.time()
            total_loss += critic.model.train_on_batch([states,actions], y_t)
            a_for_grad = actor.model.predict(states)
            grads = critic.gradients(states, a_for_grad)
            actor.train(states, grads)
            actor.target_train()
            critic.target_train()
            #end = time.time()
            #print("training time elapsed", end - start)
            #print('.', end='')
        print('')
        
        if np.mod(i, 50) == 0:
            time_per_step = 0.05
            images = np.array(episode_frames)
            #make_gif(images, './frames/image' + str(i) +'_reward_' + str(total_reward) + '.gif',
            #                 duration=len(images) * time_per_step, true_image=True, salience=False)

        if np.mod(i, 3) == 0:
            if (train_indicator):
                print("Now we save model")
                actor.model.save_weights("actormodel.h5", overwrite=True)
                with open("actormodel.json", "w") as outfile:
                    json.dump(actor.model.to_json(), outfile)

                critic.model.save_weights("criticmodel.h5", overwrite=True)
                with open("criticmodel.json", "w") as outfile:
                    json.dump(critic.model.to_json(), outfile)


        print("TOTAL REWARD @ " + str(i) +"-th Episode  : Reward " + str(total_reward))
        print("TOTAL LOSS @ " + str(i) +"-th Episode  : Loss " + str(total_loss))
        print("Total Step: " + str(step))
        print("")

        if PLOTGRAPH == 1:
            # Plotting
            ax1.plot(i, total_reward, 'ro')
            # ax2.plot(i, total_loss, 'bo')
            plt.pause(0.05)

    if PLOTGRAPH == 1:
        plt.ioff()
        plt.show()

    env.end()  # This is for shutting down TORCS
    print("Finish.")

if __name__ == "__main__":

    if not os.path.exists('./frames'):
        os.makedirs('./frames')

    playGame()
