import numpy as np
from cromosim.micro import *

class SocialForce(Policy):
    def __init__(self):
        super.__init__()
        self.trainable = False
        self.multiagent_training = None
        self.kinematics = 'holonomic'
        self.radius = 0.3
        self.max_speed = 1
        self.sim = None