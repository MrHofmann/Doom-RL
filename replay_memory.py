import random

import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)

class UniformReplay:
    def __init__(self, format, capacity=10000, batch_size=40):
        sim_shape = list(format["s_img"])
        sim_shape.insert(0, capacity)
        self._s1_img = np.zeros(sim_shape, dtype=np.float32)
        self._s2_img = np.zeros(sim_shape, dtype=np.float32)
        self._a = np.zeros(capacity, dtype=np.int32)
        self._r = np.zeros(capacity, dtype=np.float32)
        self._nonterminal = np.zeros(capacity, dtype=np.bool_)

        sim_shape[0] = batch_size
        self._s1_img_buf = np.zeros(sim_shape, dtype=np.float32)
        self._s2_img_buf = np.zeros(sim_shape, dtype=np.float32)
        self._a_buf = np.zeros(batch_size, dtype=np.int32)
        self._r_buf = np.zeros(batch_size, dtype=np.float32)
        self._nonterminal_buf = np.zeros(batch_size, dtype=np.bool_)

        if format["s_misc"] > 0:
            self._s1_misc = np.zeros((capacity, format["s_misc"]), dtype=np.float32)
            self._s2_misc = np.zeros((capacity, format["s_misc"]), dtype=np.float32)
            self._s1_misc_buf = np.zeros((batch_size, format["s_misc"]), dtype=np.float32)
            self._s2_misc_buf = np.zeros((batch_size, format["s_misc"]), dtype=np.float32)
            self._misc = True
        else:
            self._s1_misc = None
            self._s2_misc = None
            self._s1_misc_buf = None
            self._s2_misc_buf = None
            self._misc = False

        self._capacity = capacity
        self.size = 0
        self._oldest_index = 0
        self._batch_size = batch_size

        ret = dict()
        ret["s1_img"] = self._s1_img_buf
        ret["s1_misc"] = self._s1_misc_buf
        ret["a"] = self._a_buf
        ret["s2_img"] = self._s2_img_buf
        ret["s2_misc"] = self._s2_misc_buf
        ret["r"] = self._r_buf
        ret["nonterminal"] = self._nonterminal_buf

        self._ret_dict = ret.copy()

    def add_transition(self, s1, a, s2, r, terminal=False):
        if self.size < self._capacity:
            self.size += 1

        self._s1_img[self._oldest_index] = s1[0]

        if not terminal:
            self._s2_img[self._oldest_index] = s2[0]
        #print(s1[1])
        if self._misc:
            self._s1_misc[self._oldest_index] = s1[1]
            if not terminal:
                self._s2_misc[self._oldest_index] = s2[1]

        self._a[self._oldest_index] = a
        self._r[self._oldest_index] = r
        self._nonterminal[self._oldest_index] = not terminal

        self._oldest_index = (self._oldest_index + 1) % self._capacity

    def get_sample(self):
        if self._batch_size > self.size:
            raise Exception("Transition bank doesn't contain " + str(self._batch_size) + " entries.")

        #indexes = random.sample(xrange(0, self.size), self._batch_size)
        indexes = random.sample(range(0, self.size), self._batch_size)
        self._s1_img_buf[:] = self._s1_img[indexes]
        self._s2_img_buf[:] = self._s2_img[indexes]
        if self._misc:
            self._s1_misc_buf[:] = self._s1_misc[indexes]
            self._s2_misc_buf[:] = self._s2_misc[indexes]
        self._a_buf[:] = self._a[indexes]
        self._r_buf[:] = self._r[indexes]
        self._nonterminal_buf[:] = self._nonterminal[indexes]
        return self._ret_dict

class SumTree(object):
    """
    This SumTree code is a modified version and the original code is from:
    https://github.com/jaara/AI-blog/blob/master/SumTree.py
    Story data with its priority in the tree.
    """

    def __init__(self, capacity=10000):
        self.capacity = capacity  # for all priority values
        self.tree = np.zeros(2 * capacity - 1)
        
        # [--------------Parent nodes-------------][-------leaves to recode priority-------]
        #             size: capacity - 1                       size: capacity
        #self.data = np.zeros(capacity, dtype=object)  # for all transitions
        # [--------------data frame-------------]
        #             size: capacity


    #def add_transition(self, s1, a, s2, r, terminal=False, p):
        #tree_idx = self.data_pointer + self.capacity - 1
        #self.data[self.data_pointer] = data  # update data_frame

        #self.update(tree_idx, p)  # update tree_frame

        #self.data_pointer += 1
        #if self.data_pointer >= self.capacity:  # replace when exceed the capacity
        #    self.data_pointer = 0

    def update(self, tree_idx, p):
        #print(tree_idx, p)
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p
        # then propagate the change through tree
        while tree_idx != 0:    # this method is faster than the recursive loop in the reference code
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, v):
        """
        Tree structure and array storage:
        Tree index:
             0         -> storing priority sum
            / \
          1     2
         / \   / \
        3   4 5   6    -> storing priority for transitions
        Array type for storing:
        [0,1,2,3,4,5,6]
        """
        parent_idx = 0
        while True:     # the while loop is faster than the method in the reference code
            cl_idx = 2 * parent_idx + 1         # this leaf's left and right kids
            cr_idx = cl_idx + 1
            if cl_idx >= len(self.tree):        # reach bottom, end search
                leaf_idx = parent_idx
                break
            else:       # downward search, always search for a higher priority node
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx

        data_idx = leaf_idx - self.capacity + 1
        #return leaf_idx, self.tree[leaf_idx], self.data[data_idx]
        return leaf_idx, self.tree[leaf_idx], data_idx

    @property
    def total_p(self):
        return self.tree[0]  # the root


class PrioritizedReplay(object):  # stored as ( s, a, r, s_ ) in SumTree
    """
    This Memory class is modified based on the original code from:
    https://github.com/jaara/AI-blog/blob/master/Seaquest-DDQN-PER.py
    """
    epsilon = 0.01  # small amount to avoid zero priority
    alpha = 0.6  # [0~1] convert the importance of TD error to priority
    beta = 0.4  # importance-sampling, from initial value increasing to 1
    #beta_increment_per_sampling = 0.001
    beta_increment_per_sampling = 0.0000034286  #700K
    #beta_increment_per_sampling = 0.000003      #800K
    #beta_increment_per_sampling = 0.0000024     #1M
    #beta_increment_per_sampling = 0.0000016     #1.5M
    #beta_increment_per_sampling = 0.0000012     #2M
    abs_err_upper = 1.  # clipped abs error

    data_pointer = 0
    size = 0
    #self._oldest_index = 0

    def __init__(self, format, capacity=10000, batch_size=40):
        sim_shape = list(format["s_img"])
        sim_shape.insert(0, capacity)
        self._s1_img = np.zeros(sim_shape, dtype=np.float32)
        self._s2_img = np.zeros(sim_shape, dtype=np.float32)
        self._a = np.zeros(capacity, dtype=np.int32)
        self._r = np.zeros(capacity, dtype=np.float32)
        self._nonterminal = np.zeros(capacity, dtype=np.bool_)
        self.tree = SumTree(capacity)
        
        sim_shape[0] = batch_size
        self._s1_img_buf = np.zeros(sim_shape, dtype=np.float32)
        self._s2_img_buf = np.zeros(sim_shape, dtype=np.float32)
        self._a_buf = np.zeros(batch_size, dtype=np.int32)
        self._r_buf = np.zeros(batch_size, dtype=np.float32)
        self._nonterminal_buf = np.zeros(batch_size, dtype=np.bool_)
        
        if format["s_misc"] > 0:
            self._s1_misc = np.zeros((capacity, format["s_misc"]), dtype=np.float32)
            self._s2_misc = np.zeros((capacity, format["s_misc"]), dtype=np.float32)
            self._s1_misc_buf = np.zeros((batch_size, format["s_misc"]), dtype=np.float32)
            self._s2_misc_buf = np.zeros((batch_size, format["s_misc"]), dtype=np.float32)
            self._misc = True
        else:
            self._s1_misc = None
            self._s2_misc = None
            self._s1_misc_buf = None
            self._s2_misc_buf = None
            self._misc = False
        
        self._capacity = capacity        
        self._batch_size = batch_size

        ret = dict()
        ret["s1_img"] = self._s1_img_buf
        ret["s1_misc"] = self._s1_misc_buf
        ret["a"] = self._a_buf
        ret["s2_img"] = self._s2_img_buf
        ret["s2_misc"] = self._s2_misc_buf
        ret["r"] = self._r_buf
        ret["nonterminal"] = self._nonterminal_buf

        self._ret_dict = ret.copy()

    def add_transition(self, s1, a, s2, r, terminal=False):
        if self.size < self._capacity:                              #
            self.size += 1                                          #

        tree_idx = self.data_pointer + self._capacity - 1

        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        if max_p == 0:
            max_p = self.abs_err_upper
        # Maybe save data here in ReplayMemory and send only max_p to tree.
        #self.tree.add(s1, a, s2, r, terminal, max_p)   # set the max p for new p
        
        self.tree.update(tree_idx, max_p)  # update tree_frame
        self._s1_img[self.data_pointer] = s1[0]
        if not terminal:
            self._s2_img[self.data_pointer] = s2[0]
        if self._misc:
            self._s1_misc[self.data_pointer] = s1[1]
            if not terminal:
                self._s2_misc[self.data_pointer] = s2[1]
        self._a[self.data_pointer] = a
        self._r[self.data_pointer] = r
        self._nonterminal[self.data_pointer] = not terminal

        #self._oldest_index = (self._oldest_index + 1) % self._capacity
        self.data_pointer += 1
        if self.data_pointer >= self._capacity:  # replace when exceed the capacity
            self.data_pointer = 0

    # Remove n from argument list. Replace with self.batch_size
    def get_sample(self, n):
        #b_idx, b_memory, ISWeights = np.empty((n,), dtype=np.int32), np.empty((n, self.tree.data[0].size)), np.empty((n, 1))
        b_idx, indexes, ISWeights = np.empty((n,), dtype=np.int32), np.empty((n,), dtype=np.int32), np.empty((n,))
        pri_seg = self.tree.total_p / n       # priority segment
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])  # max = 1
        #print(self.beta)
        l_lim = -self.tree.capacity
        #r_lim = -(self.tree.capacity-self.size)
        if self.tree.capacity == self.size:
            r_lim = self.tree.capacity
        else:
            r_lim = -(self.tree.capacity-self.size)
        min_prob = np.min(self.tree.tree[-self.tree.capacity:r_lim]) / self.tree.total_p     # for later calculate ISweight
        #ISWeights_0 = np.empty((n, ))
        #print(self.tree.tree[l_lim:r_lim])
        #print(self.tree.total_p)
        #print(len(self.tree.tree[l_lim:r_lim]))
        #print("min_prob ", min_prob)
        for i in range(n):
            a, b = pri_seg * i, pri_seg * (i + 1)
            #print(a, b, n, pri_seg, self.tree.total_p)
            v = np.random.uniform(a, b)
            #idx, p, data = self.tree.get_leaf(v)
            idx, p, data_idx = self.tree.get_leaf(v)
            prob = p / self.tree.total_p    
            #print("prob ", prob, " p ", p, " beta ", self.beta)
            #ISWeights_0[i, 0] = np.power(prob/min_prob, -self.beta)
            #ISWeights_0[i] = np.power(prob/min_prob, -self.beta)
            ISWeights[i] = np.power(self.size * prob, -self.beta)
            #b_idx[i], b_memory[i, :] = idx, data
            b_idx[i], indexes[i] = idx, data_idx
        #return b_idx, indexes, ISWeights
        
        ISWeights = ISWeights/np.max(ISWeights)
        #print(ISWeights)
        #print(indexes)  
        
        #indexes = random.sample(range(0, self.size), self._batch_size)
        self._s1_img_buf[:] = self._s1_img[indexes]
        self._s2_img_buf[:] = self._s2_img[indexes]
        if self._misc:
            self._s1_misc_buf[:] = self._s1_misc[indexes]
            self._s2_misc_buf[:] = self._s2_misc[indexes]
        self._a_buf[:] = self._a[indexes]
        self._r_buf[:] = self._r[indexes]
        self._nonterminal_buf[:] = self._nonterminal[indexes]
        return b_idx, self._ret_dict, ISWeights

    def batch_update(self, tree_idx, abs_errors):
        #print("batch_update")
        #print(abs_errors.sum(axis=1))
        #print(abs_errors)
        #print(tree_idx)
        #print(abs_errors.shape)
        #print(self.epsilon)
        
        #abs_errors += self.epsilon  # convert to abs and avoid 0
        #abs_errors = [x+self.epsilon for x in abs_errors.sum(axis=1)]
        abs_errors = [x+self.epsilon for x in abs_errors]

        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        #print(clipped_errors)
        ps = np.power(clipped_errors, self.alpha)
        #print(ps)
        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)


#class RankReplay(PrioritizedReplay):
#    def get_sample():
#        print("Rank Sample")

#class ProportionalReplay(PrioritizedReplay):
#    def get_sample():
#        print("Proportional Sample")


