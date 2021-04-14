import numpy as np
import scipy.integrate
import math

class Octocopter:
    def __init__(self, OctorotorParams, stepNum = 0, T=0, tau=np.array([0,0, 0])):
        self.stepNum = stepNum
        # State vector looks like
        # Pos (x, y, z) , Vel (x, y, z), Orient (Phi, Theta, Psi), AngVel (Phi, Theta, Psi)
        self.state = np.zeros(12, dtype = np.float32)
        self.T = T
        self.tau = tau
        self.m0 = OctorotorParams["m0"]
        self.g = OctorotorParams["g"]
        self.Ixx = OctorotorParams["Ixx"]
        self.Iyy = OctorotorParams["Iyy"]
        self.Izz = OctorotorParams["Izz"]

        # J matrix 
        self.J = np.array([[self.Ixx, 0, 0], [0, self.Iyy, 0], [0, 0, self.Izz]], dtype="float32")

        self.invJ = np.array([[1/self.Ixx, 0, 0], [0, 1/self.Iyy, 0], [0,0, 1/self.Izz]], dtype="float32")


        # Setup ODE Integrator
        self.ode = scipy.integrate.ode(self.state_dot).set_integrator('vode', method='bdf')

       
    def calc_rotation_matrix(self, angles):
        ct = math.cos(angles[0])
        cp = math.cos(angles[1])
        cg = math.cos(angles[2])
        st = math.sin(angles[0])
        sp = math.sin(angles[1])
        sg = math.sin(angles[2])
        R_x = np.array([[1,0,0],[0,ct,-st],[0,st,ct]])
        R_y = np.array([[cp,0,sp],[0,1,0],[-sp,0,cp]])
        R_z = np.array([[cg,-sg,0],[sg,cg,0],[0,0,1]])
        R = np.dot(R_z, np.dot( R_y, R_x ))
        return R

    def calc_rotation_inverse_a(self, angle):
        phi = angle[0]
        theta = angle[1]
        psi = angle[2]
        sphi = math.sin(phi)
        tt = math.tan(theta)
        cphi = math.cos(phi)
        ctheta = math.cos(theta)
        r = np.array([[1, sphi*tt, cphi*tt], [0, cphi, -sphi], [0, sphi/ctheta, cphi/ctheta]], dtype="float32")
        return r

    def state_dot(self, time, state):
        state_dot = np.zeros(12, dtype="float32")
        pos_dot = self.calc_rotation_matrix(self.state[6:9]).dot(np.array([self.state[3], self.state[4], self.state[5]]))
        state_dot[0] = pos_dot[0]
        state_dot[1] = pos_dot[1]
        state_dot[2] = pos_dot[2]

        st = math.sin(self.state[7])
        ct = math.cos(self.state[7])
        sph = math.sin(self.state[6])
        cph = math.cos(self.state[6])

        r = self.state[9]
        q = self.state[10]
        p = self.state[11]
        s = np.array([[0, -r, q], [r, 0, -p], [-q, p, 0]], dtype="float32")
        pos_ddot = np.array([0, 0, self.T/self.m0], dtype="float32") + self.g*np.array([st, -sph*ct, -cph*ct])-1*s.dot(np.array([self.state[6], self.state[7], self.state[8]]))

        state_dot[3] = pos_ddot[0]
        state_dot[4] = pos_ddot[1]
        state_dot[5] = pos_ddot[2]
        
        ang_dot = self.calc_rotation_inverse_a(self.state[6:9]).dot(np.array([self.state[9], self.state[10], self.state[11]]))
        state_dot[6] = ang_dot[0]
        state_dot[7] = ang_dot[1]
        state_dot[8] = ang_dot[2]

        ang_ddot = self.invJ.dot(self.tau - s.dot(self.J.dot(np.array([self.state[9], self.state[10], self.state[11]]))))

        state_dot[9] = ang_ddot[0]
        state_dot[10] = ang_ddot[1]
        state_dot[11] = ang_ddot[2]
        return state_dot

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
