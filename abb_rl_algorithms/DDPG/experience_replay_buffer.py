import numpy as np
import random
import math
from sum_tree import SumTree

class ExperienceReplayBuffer():
    def __init__(self,buffer_size= 50000):
        self.buffer=[]
        self.buffer_size = buffer_size

    def add(self,experience):
        if len(self.buffer) + len(experience) >= self.buffer_size:
            self.buffer[0:(len(experience)+len(self.buffer))-self.buffer_size] = []
        self.buffer.extend(experience)

    def sample(self,batch_size,state_dimensions):
        return np.reshape(np.array(random.sample(self.buffer,batch_size)),[batch_size,state_dimensions])


class PrioritizedExperienceReplayBuffer(ExperienceReplayBuffer):
    def __init__(self, buffer_size, alpha=0.99, epsilon=0.01):
        """
        :param buffer_size: size of buffer (int)
        :param alpha: strength of prioritization (0 no, 1 full)
        """

        assert alpha > 0 and epsilon > 0

        super(PrioritizedExperienceReplayBuffer, self).__init__((int(2 ** int(math.log(buffer_size, 2)))))
        self.alpha = alpha
        self.epsilon = epsilon
        self.sum_tree = SumTree(capacity=self.buffer_size)

        self.priority_add_mean = 0
        self.priority_sample_mean = 0
        self.priority_add_step = 0
        self.priority_sample_step = 0
        self.chosen_samples = list()

    def add(self, td_error, experience):
        priority = self._calculate_priority(td_error)

        self.priority_add_mean += priority
        self.priority_add_step += 1

        self.sum_tree.add(experience=experience, new_priority_value=priority)

    def update(self, idx, td_error):
        priority = self._calculate_priority(np.abs(td_error))
        self.sum_tree.update(tree_idx=idx, new_priority_value=priority)

    def _calculate_priority(self, td_error):
        return (td_error + self.epsilon) ** self.alpha

    def sample(self, batch_size, state_dimensions):
        batch = []
        segment = self.sum_tree.get_root_value() // batch_size

        for i in range(batch_size):
            low = segment * i
            high = segment * (i + 1)

            seed = random.uniform(low, high)

            idx, priority, experience = self.sum_tree.get(seed=seed)

            batch.append((idx, priority, experience))

            self.priority_sample_mean += priority
            self.priority_sample_step += 1
            self.chosen_samples.append(idx)

        return np.array(batch)

    def get_node_priority(self, idx):
        return self.sum_tree.node_data[idx]
