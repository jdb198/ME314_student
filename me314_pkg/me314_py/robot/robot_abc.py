from abc import ABC, abstractmethod

class RobotABC(ABC):
    """
    Abstract base class for robot implementations.
    """

    def __init__(self, name: str):
        """
        Initialize the robot with a name.
        """
        self.name = name

    @abstractmethod
    def start(self):
        """
        Start the robot.
        """
        pass

    @abstractmethod
    def stop(self):
        """
        Stop the robot.
        """
        pass


    @abstractmethod
    def get_proprioception(self):
        """
        Get the proprioception of the robot.
        :return: Dictionary containing the proprioception information.
        """
        pass
