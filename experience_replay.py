from collections import deque
import random

class ReplayMeamory():
    
    # create FIFO queue - experience replay
    def __init__(self, maxlen, seed=None):
        self.meamory = deque([], maxlen=maxlen)
    
    def append(self, new_exp):
        self.meamory.append(new_exp)
        
    def sample(self, sample_size):
        return random.sample(self.meamory, sample_size)
        
    # curr buffer size
    def __len__(self):
        return len(self.meamory)