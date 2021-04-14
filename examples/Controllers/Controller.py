# Controller Class
# An abstract base class for defining a controller 
# Contains two methods: The intial constructor and a output method
# Inherit this class when building a controller object for the Octorotor Gym

from abc import ABCMeta, abstractmethod

class Controller(metaclass=ABCMeta):
    # Initialize any controller arguments such as gains in a pid controller
    @abstractmethod
    def __init__(self, controllerArgs):
        ...

    # Give control outputs based on the current state
    @abstractmethod
    def output(self, currentState, targetValue):
        ...
