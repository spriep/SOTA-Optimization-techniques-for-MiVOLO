from abc import ABC, abstractmethod #Abstract base classes

class BaseDetector(ABC): 
    @abstractmethod
    def __init__(self, weights_path, device):
        pass

    @abstractmethod
    def detect(self, frame):
        """It should return a list of lists: [[x1, y1, x2, y2, score], ...]""" #($x_{min}, y_{min}, x_{max}, y_{max}$) --> coordinates of a bounding box
        pass

class BaseEstimator(ABC):
    @abstractmethod
    def __init__(self, weights_path, device):
        pass

    @abstractmethod
    def estimate(self, face_crop):
        """It should return a floating-point number (the age)."""
        pass