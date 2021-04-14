import gym
from gym import error, spaces, utils
from gym.utils import seeding
from gym.envs.classic_control  import rendering
from .Octocopter import Octocopter
from .Actuation import ControlAllocation
import numpy as np

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
        self.allocation = ControlAllocation(OctorotorParams)
        self.omega = np.zeros(8)
        self.state = self.octorotor.get_state()
        self.dt = OctorotorParams["dt"]
        self.motor = OctorotorParams["motor"]
        self.motorController = OctorotorParams["motorController"]
        self.OctorotorParams = OctorotorParams

        # OpenAI Gym Params
        # State vector
        # state[0:2] pos
        # state[3:5] vel
        # state[6:8] angle
        # state[9:11] angle vel
        #self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=len(self.state), dtype=np.float32)
        # U = [T, tau]
        #self.action_space = spaces.Box(low=-np.inf, high=np.inf, shape=4, dtype=np.float32)
        self.viewer = None

    def step(self, action):
        # Run through control allocation, motor controller, motor, and octorotor dynamics in this order
        omega_ref = self.allocation.get_ref_velocity(action)
        voltage = self.motorController.output(self.omega, omega_ref)
        self.omega = self.motor.update(voltage, self.dt)
        u = self.allocation.get_u(self.omega)
        self.octorotor.update_u(u)
        self.octorotor.update(self.dt)
        self.state = self.octorotor.get_state()
        return self.state, {}, {}, {}

    def reset(self):
        self.octorotor = Octocopter(self.OctorotorParams) 
        self.allocation = ControlAllocation(self.OctorotorParams)
        self.omega = np.zeros(8)
        self.state = self.octorotor.get_state()
        self.dt = self.OctorotorParams["dt"]

    def render(self,xref, yref,mode='human'):
        screen_width = 600
        screen_height = 600
        # Set width to 100x100
        world_width = 20
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
            refPointTrans = rendering.Transform()
            refPoint.add_attr(refPointTrans)
            refPoint.set_color(0, 0, 1)
            refPointTrans.set_translation(xref*scale+screen_width/2, yref*scale+screen_width/2)
            self.viewer.add_geom(refPoint)
            
        if self.state is None:
            return None
        
        # Translate Rotor according to x, y
        x = self.state[0]
        y = self.state[1]
        rotorx = x*scale + screen_width/2.0
        rotory = y*scale + screen_width/2.0
        self.rotortrans.set_translation(rotorx, rotory)
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def add_arm(self, start, end):
        arm = rendering.Line(start=start, end=end)
        arm.add_attr(self.rotortrans)
        self.viewer.add_geom(arm)

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def get_state(self):
        return self.state
