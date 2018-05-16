import gym
from gym import spaces
# from os import path
import snakeoil3_gym as snakeoil3
import numpy as np
import copy
import collections as col
import os
import subprocess
import time
import signal
import cv2
import resource
 
SCREEN_WIDTH = 256
SCREEN_HEIGHT = 192

class TorcsEnv:
    terminal_judge_start = 100  # Speed limit is applied after this step
    termination_limit_progress = 5  # [km/h], episode terminates if car is running slower than this limit
    default_speed = 50

    initial_reset = True

    def start_torcs_process(self):
        if self.torcs_proc is not None:
            os.killpg(os.getpgid(self.torcs_proc.pid), signal.SIGKILL)
            time.sleep(0.5)
            self.torcs_proc = None
        window_title = str(self.port)
        print("Window title:", window_title)
        command = 'torcs -nofuel -nodamage -nolaptime -title {} -p {}'.format(window_title, self.port)
        if self.vision is True:
            command += ' -vision'
        self.torcs_proc = subprocess.Popen([command], shell=True, preexec_fn=os.setsid)
        time.sleep(0.5)
        os.system('sh autostart.sh {}'.format(window_title))
        time.sleep(0.5)

    def __init__(self, vision=False, throttle=True, gear_change=False, port=3101):
        print("Init vision = ", vision)
        self.vision = vision
        self.throttle = throttle
        self.gear_change = gear_change
        self.port = port
        self.torcs_proc = None

        self.initial_run = True

        print("launch torcs")
        time.sleep(0.5)
        self.start_torcs_process()
        time.sleep(0.5)

        """
        # Modify here if you use multiple tracks in the environment
        self.client = snakeoil3.Client(p=3101)  # Open new UDP in vtorcs
        self.client.MAX_STEPS = np.inf
        client = self.client
        client.get_servers_input()  # Get the initial input from torcs
        obs = client.S.d  # Get the current full-observation from torcs
        """
        if throttle is False:
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,))
        else:
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,))

        if vision is False:
            high = np.array([1., np.inf, np.inf, np.inf, 1., np.inf, 1., np.inf])
            low = np.array([0., -np.inf, -np.inf, -np.inf, 0., -np.inf, 0., -np.inf])
            self.observation_space = spaces.Box(low=low, high=high)
        else:
            high = np.array([1., np.inf, np.inf, np.inf, 1., np.inf, 1., np.inf, 255])
            low = np.array([0., -np.inf, -np.inf, -np.inf, 0., -np.inf, 0., -np.inf, 0])
            self.observation_space = spaces.Box(low=low, high=high)

    def step(self, u):
       #print("Step")
        # convert thisAction to the actual torcs actionstr
        client = self.client

        this_action = self.agent_to_torcs(u)

        # Apply Action
        action_torcs = client.R.d

        # Steering
        action_torcs['steer'] = this_action['steer']  # in [-1, 1]

        #  Simple Autnmatic Throttle Control by Snakeoil
        if self.throttle is False:
            target_speed = self.default_speed
            if client.S.d['speedX'] < target_speed - (client.R.d['steer']*50):
                client.R.d['accel'] += .02
            else:
                client.R.d['accel'] -= .02

            if client.R.d['accel'] > 0.4:
                client.R.d['accel'] = 0.4

            if client.S.d['speedX'] < target_speed:
                client.R.d['accel'] += 1/(client.S.d['speedX']+.1)

            # Traction Control System
            if ((client.S.d['wheelSpinVel'][2]+client.S.d['wheelSpinVel'][3]) -
               (client.S.d['wheelSpinVel'][0]+client.S.d['wheelSpinVel'][1]) > 5):
                action_torcs['accel'] -= .2
        else:
            if this_action['accel'] > 0.0:
                if this_action['accel'] > 0.6:
                    this_action['accel'] = 0.6
                action_torcs['brake'] = 0
                action_torcs['accel'] = this_action['accel']
            else:
                #action_torcs['brake'] = -this_action['accel']*0.03
                action_torcs['accel'] = 0
	    if ((client.S.d['wheelSpinVel'][2]+client.S.d['wheelSpinVel'][3]) -
               (client.S.d['wheelSpinVel'][0]+client.S.d['wheelSpinVel'][1]) > 5):
                action_torcs['accel'] -= .2

        #  Automatic Gear Change by Snakeoil
        if self.gear_change is True:
            action_torcs['gear'] = this_action['gear']
        else:
            #  Automatic Gear Change by Snakeoil is possible
            action_torcs['gear'] = 1
            if self.throttle:
                if client.S.d['speedX'] > 50:
                    action_torcs['gear'] = 2
                if client.S.d['speedX'] > 80:
                    action_torcs['gear'] = 3
                if client.S.d['speedX'] > 110:
                    action_torcs['gear'] = 4
                if client.S.d['speedX'] > 140:
                    action_torcs['gear'] = 5
                if client.S.d['speedX'] > 170:
                    action_torcs['gear'] = 6
        # Save the privious full-obs from torcs for the reward calculation
        obs_pre = copy.deepcopy(client.S.d)

        # One-Step Dynamics Update #################################
        # Apply the Agent's action into torcs
        client.respond_to_server()
        # Get the response of TORCS
        client.get_servers_input()

        # Get the current full-observation from torcs
        obs = client.S.d

        # Make an obsevation from a raw observation vector from TORCS
        self.observation = self.make_observaton(obs)

        # Reward setting Here #######################################
        # direction-dependent positive reward
        track = np.array(obs['track'])
        trackPos = np.array(obs['trackPos'])
        sp = np.array(obs['speedX'])
        spy = np.array(obs['speedY'])
        damage = np.array(obs['damage'])
        rpm = np.array(obs['rpm'])

        print("speedX", client.S.d['speedX'], "acc", client.R.d['accel'], "brake", client.R.d['brake'])

        ## REWARD
        #progress = sp*np.cos(obs['angle']) + spy*np.sin(obs['angle']) - np.abs(spy*np.cos(obs['angle']) - sp*np.sin(obs['angle'])) - sp*np.cos(obs['angle'])*np.abs(obs['trackPos'])
        # progress = sp*np.cos(obs['angle']) - np.abs(sp*np.sin(obs['angle'])) - sp * np.abs(obs['trackPos'])
        progress = sp*np.cos(obs['angle'])+ sp*np.cos(obs['angle'])*(2-2*np.abs(obs['trackPos']))-np.abs(spy*np.cos(obs['angle'])-0.15*sp*sp*np.abs(np.sin(obs['angle'])))-sp*np.cos(obs['angle'])*np.abs(obs['trackPos'])
        reward = progress

        # collision detection
        if obs['damage'] - obs_pre['damage'] > 0:
            # reward = -1
            print("COLLISION")
            reward = -20
            episode_terminate = True
            #client.R.d['meta'] = True

        # Termination judgement #########################
        episode_terminate = False
        if (abs(track.any()) > 1 or abs(trackPos) > 1):  # Episode is terminated if the car is out of track
           reward = -1
           # episode_terminate = True
           # client.R.d['meta'] = True

        if self.terminal_judge_start < self.time_step: # Episode terminates if the progress of agent is small
            if sp < self.termination_limit_progress:
           # if progress < self.termination_limit_progress:
               print("No progress")
               episode_terminate = True
               client.R.d['meta'] = True

        if np.cos(obs['angle']) < 0: # Episode is terminated if the agent runs backward
            episode_terminate = True
            client.R.d['meta'] = True

        if obs['lastLapTime'] > 0:
            print("lastLapTime", obs['lastLapTime'])
            reward = 200
            episode_terminate = True
            client.R.d['meta'] = True

        if client.R.d['meta'] is True: # Send a reset signal
            self.initial_run = False
            client.respond_to_server()

        self.time_step += 1

        return self.get_obs(), reward, client.R.d['meta'], {}

    def reset(self, relaunch=False):
        #print("Reset")

        self.time_step = 0

        if self.initial_reset is not True:
            self.client.R.d['meta'] = True
            self.client.respond_to_server()

            ## TENTATIVE. Restarting TORCS every episode suffers the memory leak bug!
            if relaunch is True:
                self.reset_torcs()
                print("### TORCS is RELAUNCHED ###")

        # Modify here if you use multiple tracks in the environment
        self.client = snakeoil3.Client(self.start_torcs_process, p=self.port)  # Open new UDP in vtorcs
        self.client.MAX_STEPS = np.inf

        client = self.client
        client.get_servers_input()  # Get the initial input from torcs

        obs = client.S.d  # Get the current full-observation from torcs
        self.observation = self.make_observaton(obs)

        self.last_u = None

        self.initial_reset = False
        return self.get_obs()

    def end(self):
        os.killpg(os.getpgid(self.torcs_proc.pid), signal.SIGKILL)

    def get_obs(self):
        return self.observation

    def reset_torcs(self):
        #print("relaunch torcs")
        #self.torcs_proc.terminate()
        #time.sleep(0.5)
        self.start_torcs_process()
        time.sleep(0.5)

    def agent_to_torcs(self, u):
        torcs_action = {'steer': u[0]}

        if self.throttle is True:  # throttle action is enabled
            torcs_action.update({'accel': u[1]})
            torcs_action.update({'brake': u[1]})

        if self.gear_change is True: # gear change action is enabled
            torcs_action.update({'gear': int(u[3])})

        return torcs_action

    def obs_vision_to_image_rgb_proto(self, obs_image):
        image_jpeg = np.asarray(obs_image).astype(np.uint8)
        image = cv2.imdecode(image_jpeg, cv2.IMREAD_COLOR)
        #cv2.imshow('Image', image)
        #cv2.waitKey()
        #return image.reshape((SCREEN_HEIGHT, SCREEN_WIDTH, 3))[::-1, :, :]

        r = image[:, :, 0]
        g = image[:, :, 1]
        b = image[:, :, 2]
        r = r[::-1]
        g = g[::-1]
        b = b[::-1]

        #return np.array([r, g, b], dtype=np.uint8)
        rgb = cv2.merge((r,g,b))
        return cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        #return rgb

    def obs_vision_to_image_rgb(self, obs_image_vec):
        image_vec =  obs_image_vec
        r = image_vec[0:len(image_vec):3]
        g = image_vec[1:len(image_vec):3]
        b = image_vec[2:len(image_vec):3]
        # print len(image_vec)

        sz = (SCREEN_HEIGHT, SCREEN_WIDTH)
        r = np.array(r).reshape(sz)
        r = r.astype(np.uint8)
        r = r[::-1]
        g = np.array(g).reshape(sz)
        g = g.astype(np.uint8)
        g = g[::-1]
        b = np.array(b).reshape(sz)
        b = b.astype(np.uint8)
        b = b[::-1]

        rgb = cv2.merge((r,g,b))
        return rgb

        # r = np.array(r).reshape(sz)
        # g = np.array(g).reshape(sz)
        # b = np.array(b).reshape(sz)
        # return np.array([r, g, b], dtype=np.uint8)

    # def obs_vision_to_image_rgb_proto(self, obs_image):
    #     image_jpeg = np.asarray(obs_image).astype(np.uint8)
    #     image = cv2.imdecode(image_jpeg, cv2.IMREAD_COLOR)
    #     print image
    #     return image.reshape((SCREEN_HEIGHT, SCREEN_WIDTH, 3))[::-1, :, :]

    # def obs_vision_to_image_rgb(self, obs_image_vec):
    #     image_vec =  obs_image_vec
    #     rgb = []
    #     # group rgb values together.
    #     # Format similar to the observation in openai gym

    #     #print "Total image bytes: %d" % len(image_vec)
    #     for i in range(0, SCREEN_WIDTH*SCREEN_HEIGHT):
    #         temp = tuple(image_vec[3*i:3*i+3])
    #         rgb.append(temp)
       
    #     return np.array(rgb, dtype=np.uint8).reshape((SCREEN_HEIGHT, SCREEN_WIDTH, 3))[::-1,:,:]

    def make_observaton(self, raw_obs):
        if self.vision is False:
            names = ['focus',
                     'speedX', 'speedY', 'speedZ', 'angle', 'damage',
                     'opponents',
                     'rpm',
                     'track', 
                     'trackPos',
                     'wheelSpinVel']
            Observation = col.namedtuple('Observaion', names)
            return Observation(focus=np.array(raw_obs['focus'], dtype=np.float32)/200.,
                               speedX=np.array(raw_obs['speedX'], dtype=np.float32)/300.0,
                               speedY=np.array(raw_obs['speedY'], dtype=np.float32)/300.0,
                               speedZ=np.array(raw_obs['speedZ'], dtype=np.float32)/300.0,
                               angle=np.array(raw_obs['angle'], dtype=np.float32)/3.1416,
                               damage=np.array(raw_obs['damage'], dtype=np.float32),
                               opponents=np.array(raw_obs['opponents'], dtype=np.float32)/200.,
                               rpm=np.array(raw_obs['rpm'], dtype=np.float32)/10000,
                               track=np.array(raw_obs['track'], dtype=np.float32)/200.,
                               trackPos=np.array(raw_obs['trackPos'], dtype=np.float32)/1.,
                               wheelSpinVel=np.array(raw_obs['wheelSpinVel'], dtype=np.float32))
        else:
            names = ['focus',
                     'speedX', 'speedY', 'speedZ', 'angle',
                     'opponents',
                     'rpm',
                     'track',
                     'trackPos',
                     'wheelSpinVel',
                     'img']
            Observation = col.namedtuple('Observaion', names)

            # Get RGB from observation
            # image_rgb = self.obs_vision_to_image_rgb(raw_obs[names[10]])
            image_rgb = self.obs_vision_to_image_rgb_proto(raw_obs[names[10]])

            return Observation(focus=np.array(raw_obs['focus'], dtype=np.float32)/200.,
                               speedX=np.array(raw_obs['speedX'], dtype=np.float32)/self.default_speed,
                               speedY=np.array(raw_obs['speedY'], dtype=np.float32)/self.default_speed,
                               speedZ=np.array(raw_obs['speedZ'], dtype=np.float32)/self.default_speed,
                               angle=np.array(raw_obs['angle'], dtype=np.float32)/3.1416,
                               opponents=np.array(raw_obs['opponents'], dtype=np.float32)/200.,
                               rpm=np.array(raw_obs['rpm'], dtype=np.float32),
                               track=np.array(raw_obs['track'], dtype=np.float32)/200.,
                               trackPos=np.array(raw_obs['trackPos'], dtype=np.float32)/1.,
                               wheelSpinVel=np.array(raw_obs['wheelSpinVel'], dtype=np.float32),
                               img=image_rgb)
