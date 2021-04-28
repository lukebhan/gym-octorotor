import gym
from gym import error, spaces, utils
from gym.utils import seeding
from gym.envs.classic_control  import rendering
from .Octocopter import Octocopter
from .Actuation import ControlAllocation
import pandas as pd
import numpy as np
import math

class OctorotorBaseEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    # Initialize the Octorotor
    # Simulation Parameters consist of the following and are passed via dictionary
    # g - gravity
    # mass - mass of octorotor
    # d - friction coefficient
    # b - rotor thrust constant
    # l - length of rotor arms (all equal)
    # omega_max - max angular velocity
    # Ixx - x moment of intertia
    # Iyy - y moment of inertia
    # Izz - z moment of inertia
    # dt  - time_Step of simulation
    # Motor - a motor object that inherits the Motor base class: Must have constructor and update functions implemented
    # motorController - A controller object that inherits the controller base class: Must have constructor and output methods implemented
    def __init__(self, OctorotorParams):
        super(OctorotorBaseEnv, self).__init__()
        # Octorotor Params
        self.octorotor = Octocopter(OctorotorParams)
        self.xrefarr = pd.read_csv("./Paths/EpathX2.csv", header=None)[1].to_numpy()
        self.yrefarr = pd.read_csv("./Paths/EpathY2.csv", header=None)[1].to_numpy()
        self.index = 0
        b = 10
        a = -10
        #self.octorotor.set_pos((b- a) * np.random.random_sample() + a, (b-a)*np.random.random_sample()+a)
        self.xref = 2
        self.yref = 2
        self.allocation = ControlAllocation(OctorotorParams)
        self.resistance = np.full(8, OctorotorParams["resistance"])
        self.state = np.append(np.append(np.append(self.octorotor.get_state(),self.resistance), abs(self.octorotor.get_state()[0]-self.xref)), abs(self.octorotor.get_state()[1]-self.yref))
        self.dt = OctorotorParams["dt"]
        self.motor = OctorotorParams["motor"]
        self.motorController = OctorotorParams["motorController"]
        self.posc = OctorotorParams["positionController"]
        self.attc = OctorotorParams["attitudeController"]
        self.altc = OctorotorParams["altitudeController"]
        self.OctorotorParams = OctorotorParams
        self.omega = np.zeros(8)
        self.step_count = 0
        self.total_step_count = OctorotorParams["total_step_count"]
        self.zref = 0
        self.psiref = np.zeros(3)
        self.reward_discount = OctorotorParams["reward_discount"]

        # OpenAI Gym Params
        # State vector
        # state[0:2] pos
        # state[3:5] vel
        # state[6:8] angle
        # state[9:11] angle vel
        self.observation_space = spaces.Box(np.full(22, -np.inf, dtype="float32"), np.full(22, np.inf, dtype="float32"), dtype="float32")
        #U = [T, tau]
        self.action_space = spaces.Box(np.array([-1, -1, -1, -1]), np.array([1, 1, 1, 1]), dtype="float32")
        self.viewer = None

    def step(self, action):
        # Run through control allocation, motor controller, motor, and octorotor dynamics in this order
        self.step_count += 1
        #if(self.step_count % 50 == 0 and self.index != len(self.xrefarr)):
        #self.xref = self.xrefarr[self.index]
        #self.yref = self.yrefarr[self.index]
        #self.index+=1
        targetValues = {"xref": self.xref, "yref": self.yref}
        self.psiref[1], self.psiref[0] = self.posc.output(self.state, targetValues)
        tau_des = self.attc.output(self.state, self.psiref)
        T_des = self.altc.output(self.state, self.zref)
        udes = np.array([T_des+action[0], tau_des[0]+action[1], tau_des[1]+action[2], tau_des[2]+action[3]], dtype="float32")
        #udes = np.array([T_des, tau_des[0], tau_des[1], tau_des[2]], dtype="float32")
        omega_ref = self.allocation.get_ref_velocity(udes)
        voltage = self.motorController.output(self.omega, omega_ref)
        self.omega = self.motor.update(voltage, self.dt)
        u = self.allocation.get_u(self.omega)
        self.octorotor.update_u(u)
        self.octorotor.update(self.dt)
        self.state = np.append(np.append(np.append(self.octorotor.get_state(),self.resistance), abs(self.octorotor.get_state()[0]-self.xref)), abs(self.octorotor.get_state()[1]-self.yref))
        return self.state, self.reward(), self.episode_over(), {"xerror": self.get_xerror(), "yerror": self.get_yerror()}

    def reset(self):
        OctorotorParams = self.OctorotorParams
        self.octorotor = Octocopter(OctorotorParams) 
        b = 10
        a = -10
        #self.octorotor.set_pos((b- a) * np.random.random_sample() + a, (b-a)*np.random.random_sample()+a)
        self.allocation = ControlAllocation(OctorotorParams)
        self.omega = np.zeros(8)
        self.dt = OctorotorParams["dt"]
        self.motor = OctorotorParams["motor"]
        self.motorController = OctorotorParams["motorController"]
        self.step_count = 0
        self.total_step_count = OctorotorParams["total_step_count"]
        self.viewer = None
        self.xref = 2
        self.yref = 2
        self.index = 0
        self.psiref = np.zeros(3)
        return self.state

    def render(self,mode='human'):
        xref = self.xref
        yref = self.yref
        screen_width = 600
        screen_height = 600
        # Set width to 100x100
        world_width = 600
        scale = screen_width/world_width
        rotorradius = 4
        armwidth = 1
        armlength = self.OctorotorParams["l"]*scale + rotorradius
        if self.viewer is None:
            # build Octorotor
            self.viewer = rendering.Viewer(screen_width, screen_height)
            rotor = rendering.make_circle(radius=rotorradius)
            self.rotortrans = rendering.Transform()
            rotor.add_attr(self.rotortrans)
            rotor.set_color(1, 0, 0)
            self.viewer.add_geom(rotor)
            self.add_arm((0, 0), (armlength, 0))
            self.add_arm((0, 0), (-armlength, 0))
            self.add_arm((0, 0), (0, armlength))
            self.add_arm((0, 0), (0, -armlength))
            self.add_arm((0, 0), (armlength, armlength))
            self.add_arm((0, 0), (-armlength, armlength))
            self.add_arm((0, 0), (-armlength, -armlength))
            self.add_arm((0, 0), (armlength, -armlength))
            # Build ref Point
            refPoint = rendering.make_circle(radius = rotorradius)
            self.refPointTrans = rendering.Transform()
            refPoint.add_attr(self.refPointTrans)
            refPoint.set_color(0, 0, 1)
            self.refPointTrans.set_translation(xref*scale+screen_width/2, yref*scale+screen_width/2)
            self.viewer.add_geom(refPoint)
            
        if self.state is None:
            return None
        # Translate Rotor according to x, y
        x = self.state[0]
        y = self.state[1]
        rotorx = x*scale + screen_width/2.0
        rotory = y*scale + screen_width/2.0
        self.rotortrans.set_translation(rotorx, rotory)
        self.refPointTrans.set_translation(xref*scale+screen_width/2, yref*scale+screen_width/2)
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def add_arm(self, start, end):
        arm = rendering.Line(start=start, end=end)
        arm.add_attr(self.rotortrans)
        self.viewer.add_geom(arm)

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def reward(self):
        error = math.sqrt((self.xref-self.state[0])*(self.xref-self.state[0]) + (self.yref-self.state[1]) * (self.yref-self.state[1])+ (self.zref-self.state[2]) * (self.zref-self.state[2]))
        return -error + 20

    def episode_over(self):
        error = math.sqrt((self.xref-self.state[0])*(self.xref-self.state[0]) + (self.yref-self.state[1]) * (self.yref-self.state[1]) + (self.zref-self.state[2]) * (self.zref-self.state[2]))
        return self.step_count > 10000 

    def update_r(self,r, idx):
        self.motor.update_r(r, idx)

    def get_state(self):
        return self.state

    def get_xerror(self):
        return self.state[0]

    def get_yerror(self):
        return self.state[1]

    def safe_ranges(self):
        return abs(self.state[3]) > 10 or abs(self.state[4]) > 10 or abs(self.state[5])>10 or abs(self.state[11]) > 4.5 or abs(self.state[10]) > 4.5 or abs(self.state[9]) > 4.5
