import numpy as np
import math
class SumTree():
    """
    Used for prioritized experience replay buffer. Allows sampling
    according to priority given to an experience (based on time difference error of Bellman equation)

    """
    def __init__(self,capacity):

        assert capacity>0 and (capacity & capacity-1 == 0) # capacity needt to be power of 2 and larger than 0

        self.capacity = capacity
        print("Replay Buffersize:", self.capacity)

        self.node_nr = 2*capacity-1 # tree for storing priority values
        self.node_data = np.zeros(self.node_nr) # number of nodes in tree initialized to zero
        self.experience_buffer = np.zeros(capacity,dtype=object) #storing experiences
        self.buffer_pointer = 0

    def add(self, experience,new_priority_value):  # adds a new experience to experience_buffer from left to right, circular buffer

        if self.buffer_pointer >= self.capacity:
            self.buffer_pointer = 0 # make sure does not overflow

        #current index in buffer
        idx = self.buffer_pointer
        tree_idx= self.capacity-1+idx
         #calculate the difference of the new priority and priority in tree

        #advance buffer pointer, update experience buffer

        self.experience_buffer[self.buffer_pointer] = experience


        self.update(tree_idx,new_priority_value)


        self.buffer_pointer += 1


    def update(self,tree_idx,new_priority_value):
        #Update priority value according to the difference
        difference = new_priority_value - self.node_data[tree_idx]
        self.node_data[tree_idx] = new_priority_value

        #propagate the difference up to the root of the tree (O(log(N))
        self._propagate_up(tree_idx,difference)

    def _propagate_up(self,child_idx,difference):
        parent_idx = (child_idx - 1) // 2
        self.node_data[parent_idx] += difference

        if parent_idx != 0:
            self._propagate_up(parent_idx,difference)


    def get(self,seed):

        assert seed <= self.get_root_value()

        idx = self._get(0,seed)


        experience_idx = idx - self.capacity+1

        experience = self.experience_buffer[experience_idx]

        return idx,self.node_data[idx],experience #index need to be returned because we update the priority after new TD errors for each batch sample were calculated


    def _get(self,idx,seed):


        left_idx = 2*idx+1
        right_idx = left_idx+1

        if left_idx >= self.node_nr: #leaf node

            return idx

        if seed <= self.node_data[left_idx]:

            return self._get(idx=left_idx,seed=seed)
        else:

            return self._get(idx=right_idx,seed=seed-self.node_data[left_idx])




    def get_root_value(self):
        return self.node_data[0]


    def _debug_print_tree(self):

        print(self.node_data[0])
        i=1
        line_i=1
        while i < self.node_nr:

            print(self.node_data[i],end=" ")

            if i % 2**line_i:
                print(" ",end="\n")
                line_i += 1

            i += 1

