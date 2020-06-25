import numpy as np
from scipy.integrate import odeint

# Imports as required

class PIDController:
    """
        v = Final output of plant
        u = Input signal to the plant
    """
    def __init__(self, dt, initial_error, set_point):
        self.sum_error = 0
        self.dt = dt

        self.prev_error = initial_error
        self.set_point = set_point

    def set_set_point(self, sp):
        self.set_point = sp

    def set_k_constants(self, Ki, Kp, Kd):
        self.params = (Ki, Kp, Kd)

    def get_pid_output(self, v):
        """
            Given the latest measurement of v(t) we obtain new u(t)
        """
        Ki, Kp, Kd = self.params
        dt = self.dt
        
        error = self.set_point - v
        self.sum_error += error*dt
        derror = (error - self.prev_error)/dt
        
        u = Kp*error + Ki*self.sum_error + Kd*derror
        
        #temp acts as prev_error that needs to be send
        temp = self.prev_error
        self.prev_error = error
        
        return u, error, temp, derror
    

class Model:
    """
        u  = Plant input
        ycaps = y1, y2, y3 where y1 is the Plant Output
    """
    
    def __init__(self, initial_y_caps):
        self.y_caps = [initial_y_caps]
        self.u = 0
        
    def de_solver(self, y_cap, t, params):
        """
        Given a vector <y, y1, y2> returns <y1, y2, y3> according to equation
        above.
        """
        y, y1, y2 = y_cap # y, dy/dt, d2y/dt2
        x = params

        y3 = -2.14*y2 - 9.276*y1 - 4.228*y + 4.228*x
        dy_cap = y1, y2, y3

        return dy_cap
    
    def set_model_input(self, u_):
        self.u = u_
        
    def get_model_output(self, t_step):
        y_cap_solved = odeint(self.de_solver, self.y_caps[-1], t_step, args=(self.u,))[-1]
        self.y_caps.append(y_cap_solved)
        return self.y_caps[-1][0]


class PIDModel:
    """
        State must be a 5 length list of
        (Kd', Kp', alpha, e_t, de_t/dt)

        Action space is (Kd', Kp', alpha), this class handles all
        the denormalisation

        Kpmin, Kpmax, etc are parameters set in __init__
    """
    def __init__(self, ku, tu, t, SP):
        """
            Add arguments to this method as necessary
            Kp_min, Kp_max, etc should be arguments
        """
        self.ku = ku
        self.tu = tu
        self.kpmax = 0.6*ku
        self.kpmin = 0.32*ku
        self.kdmax = 0.15*ku*tu
        self.kdmin = 0.08*ku*tu
        
        self.count = 0
        self.setpoint = SP
        self.t = t
        
        self.model = Model(initial_y_caps=(0,0,0))
        self.pid = PIDController(dt=t[1]-t[0], initial_error=0, set_point=SP[0])
        
    def get_next_count(self):
        """Used to update the time-step through count""" 
        self.count = self.count + 1
        return self.count
    
    def get_reward(self, e_t, prev_e_t, alpha1=1, alpha2=1, epsilon=0.5):
        if np.abs(e_t) < epsilon:
            r1 = 0
        else:
            r1 = epsilon - e_t
        
        if np.abs(e_t) <= np.abs(prev_e_t):
            r2 = 0
        else:
            r2 = np.abs(e_t)-np.abs(prev_e_t)
        r = alpha1*r1 + alpha2*r2
        return r

    def step(self, action):
        """
            Support this:
            new_state, reward, done = env.step(action)

            done is a boolean whether or not episode is finished
        """
        kd_, kp_, alpha = action
        """
            De-Normalising kd', kp' values
        """
        kp = (self.kpmax - self.kpmin)*kp_ + self.kpmin
        kd = (self.kdmax - self.kdmin)*kd_ + self.kdmin
        ki = kp**2/(alpha*kd)
        
        #Updating PID parameters
        self.pid.set_k_constants(ki, kp, kd)
    
        #Getting plant output for step-time
        v = self.model.get_model_output(t_step=self.t[self.count: self.count+2])
        #Detting PID output, error, derivative of error corresponding to recent plant output
        u, e_t, prev_e_t, de_t = self.pid.get_pid_output(v)
        
        #Updating plants input to output of PID
        self.model.set_model_input(u)
        #Updating PIDs setpoint for next time-step also updating count for next time-step
        self.pid.set_set_point(self.setpoint[self.get_next_count()])
        
        reward = self.get_reward(e_t, prev_e_t)
        new_state = kd_, kp_, alpha, e_t, de_t
        
        if (self.count + 1) == len(self.t):
            done = True
        else:
            done = False
            
        return new_state, reward, done

    def reset(self):
        """
            Restart the episode, clear all data
        """
        self.model.__init__((0,0,0))
        self.pid.__init__(dt=self.t[1]-self.t[0], initial_error=0, set_point=self.setpoint[0])
        self.__init__(self.ku, self.tu, self.t, self.setpoint)
        
    def output(self):
        """
            Returns array of [y1, y2, y3] where y1 is plants output       
        """
        return self.model.y_caps
