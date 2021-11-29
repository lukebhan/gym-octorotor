import numpy as np
import scipy.integrate
import math
from numba import jit

@jit 
def _calc_rotation_matrix(angles):
    ct = math.cos(angles[0])
    cp = math.cos(angles[1])
    cg = math.cos(angles[2])
    st = math.sin(angles[0])
    sp = math.sin(angles[1])
    sg = math.sin(angles[2])
    R = np.array([[cg*cp, cp*cg*st-sg*ct, sp*cg*ct+sg*st], [cp*sg, sg*cp*st+cg*ct, sg*cp*ct-st*sg], [-sp, cp*st, cp*ct]])
    return R


@jit
def _calc_rotation_inverse_a(angle):
    phi = angle[0]
    theta = angle[1]
    psi = angle[2]
    sphi = math.sin(phi)
    tt = math.tan(theta)
    nphi = math.cos(phi)
    ctheta = math.cos(theta)
    r = np.array([[1, sphi*tt, nphi*tt], [0, nphi, -sphi], [0, sphi/ctheta, nphi/ctheta]], dtype="float32")
    return r

def _state_dot(time, state, T, m0, g, tau, invJ, J):
    state_dot = np.zeros(12, dtype="float32")
    pos_dot = _calc_rotation_matrix(state[6:9]).dot(np.array([state[3], state[4], state[5]]))
    state_dot[0] = pos_dot[0]
    state_dot[1] = pos_dot[1]
    state_dot[2] = pos_dot[2]

    st = math.sin(state[7])
    ct = math.cos(state[7])
    sph = math.sin(state[6])
    cph = math.cos(state[6])
    r = state[9]
    q = state[10]
    p = state[11]
    pos_ddot = np.array([g*st-r*state[7]-q*state[8], -g*sph*ct-r*state[6]+p*state[8], T/m0-g*cph*ct+q*state[6]-p*state[7]])

    state_dot[3] = pos_ddot[0]
    state_dot[4] = pos_ddot[1]
    state_dot[5] = pos_ddot[2]
    u = np.array([pos_ddot[0].copy(), pos_ddot[1].copy(), pos_ddot[2].copy(), state[9].copy(), state[10].copy, state[11].copy()])

    ang_dot = _calc_rotation_inverse_a(state[6:9]).dot(np.array([state[9], state[10], state[11]]))
    state_dot[6] = ang_dot[0]
    state_dot[7] = ang_dot[1]
    state_dot[8] = ang_dot[2]
    s = np.array([[0, -r, q], [r, 0, -p], [-q, p, 0]], dtype="float32")

    ang_ddot = invJ.dot(tau - s.dot(J.dot(np.array([state[9], state[10], state[11]]))))
    state_dot[9] = ang_ddot[0]
    state_dot[10] = ang_ddot[1]
    state_dot[11] = ang_ddot[2]
    return state_dot, u

class Octocopter:
    def __init__(self, OctorotorParams, stepNum = 0, T=0, tau=np.array([0,0, 0])):
        self.prevx = [0, 0]
        self.xfilter = []
        self.yfilter = []
        self.stepNum = stepNum
        self.u = np.zeros(6)

        # State vector looks like
        # Pos (x, y, z) , Vel (x, y, z), Orient (Phi, Theta, Psi), AngVel (Phi, Theta, Psi)
        self.state = np.zeros(12, dtype = np.float32)
        self.T = np.float32(T)
        self.tau = np.float32(tau)
        self.m0 = np.float32(OctorotorParams["m0"])
        self.g = np.float32(OctorotorParams["g"])
        self.Ixx = np.float32(OctorotorParams["Ixx"])
        self.Iyy = np.float32(OctorotorParams["Iyy"])
        self.Izz = np.float32(OctorotorParams["Izz"])

        # J matrix 
        self.J = np.array([[self.Ixx, 0, 0], [0, self.Iyy, 0], [0, 0, self.Izz]], dtype="float32")

        self.invJ = np.array([[1/self.Ixx, 0, 0], [0, 1/self.Iyy, 0], [0,0, 1/self.Izz]], dtype="float32")


        # Setup ODE Integrator
        self.ode = scipy.integrate.ode(self.state_dot)

    def state_dot(self, time, state):
        state, u =_state_dot(time, self.state, np.float32(self.T), self.m0, self.g, np.float32(self.tau), self.invJ, self.J)
        self.u = u
        return state;

    def update(self, dt):
        self.stepNum += 1
        self.ode.set_initial_value(self.state, 0)
        self.state = self.ode.integrate(self.ode.t + dt)
        # Add constraint for z
        self.state[2] = max(self.state[2], 0)

    def get_velocity(self):
        return self.state[3:6]

    def get_position(self):
        return self.state[0:3]

    def get_angle(self):
        return self.state[6:9]
    
    def get_angle_vel(self):
        return self.state[9:12]

    def update_u(self, u):
        self.T = u[0]
        self.tau = u[1:4]

    def get_j_matrix(self):
        return self.J

    def get_state(self):
        return self.state

    def set_pos(self, x, y):
        self.state[0] = x
        self.state[1] = y

    def get_u(self):
        return self.u
