# Motor Class
# An abstract base class for defining a motor
# Contains two methods: The intial constructor and an update method 
# Inherit this class when defining motor objects for the the Octorotor Gym

from abc import ABCMeta, abstractmethod

class Motor(metaclass=ABCMeta):
    # Initialize any motor arguments
    @abstractmethod
    def __init__(self, motorArgs):
        ...

    # Return angular velocity given a voltage according to motor dynamics and given time step
    @abstractmethod
    def update(self, voltage, dt):
        ...
