#class imports
import numpy as np


class SumTree:
    '''
    SumTree data structure for Prioritized Experience Replay Buffer Implementation
    Args:
        capacity : the capacity of the buffer
    '''
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)  # Tree with sum nodes
        self.data = np.zeros(capacity, dtype=object)  # Holds experiences
        self.data_pointer = 0
        self.max_priority = 0
       

    def add(self, priority, data):
        tree_index = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data  # Store experience
        self.update(tree_index, priority)  # Update priority

        # Update max_priority
        self.max_priority = max(self.max_priority, priority)

        self.data_pointer += 1
        if self.data_pointer >= self.capacity:  # If capacity exceeded, overwrite
            self.data_pointer = 0

    def update(self, tree_index, priority):
        delta = priority - self.tree[tree_index]
        self.tree[tree_index] = priority
        self.propagate(tree_index, delta)
        
        # Update max_priority if necessary
        self.max_priority = max(self.max_priority, priority)

    def propagate(self, tree_index, delta):
        while tree_index != 0:
            parent = (tree_index - 1) // 2
            self.tree[parent] += delta
            tree_index = parent

    def get_leaf(self, value):
        parent = 0
        while True:  # Traverse down to a leaf
            left_child = 2 * parent + 1
            right_child = left_child + 1

            if left_child >= len(self.tree):  # Leaf node
                leaf_index = parent
                break
            else:  # Go deeper in the tree
                if value <= self.tree[left_child]:
                    parent = left_child
                else:
                    value -= self.tree[left_child]
                    parent = right_child

        data_index = leaf_index - self.capacity + 1
        return leaf_index, self.tree[leaf_index], self.data[data_index]

    def total_priority(self):
        return self.tree[0]  # Root node




class PrioritizedExperienceReplay:
    '''
    Prioritized Experience Replayvmechanism for training RL models. Priority is assigned to transitions as a
    functon of Temporal Difference Error (TDE).
    Args:
        capacity   : the capacity of the buffer
        alpha      : the extent to which the priority affects sampling probability- 0:uniform sample, 1: full priority sampling
        beta_start : Importance-Sampling Correction, corrects prioritized sampling bias- 0: ignores bias, 1: removes all bias
        beta_increment_per_sampling : increments (+) beta per sample, eventually to 1
    '''
    #                                                      anneals in ~14 million frames
    def __init__(self, capacity, alpha=.4, beta_start=0.6, beta_increment_per_sampling=0.000000035, alpha_increment_per_sampling=0.0000000624):
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.beta = beta_start
        self.beta_increment_per_sampling = beta_increment_per_sampling 
        self.alpha_increment_per_sampling = alpha_increment_per_sampling
        self.epsilon = .01  # Small value to avoid zero priorities
        self.capacity = capacity
        self.update_counter = 0 #init update counter for periodic max_priority calculation
        self.recalculate_schedule = 20_000 #recalculate max priority

    def add(self, experience, td_error=None):
        # Use max_priority for initialization if td_error is not provided
        if td_error is None:
            priority = self.tree.max_priority if self.tree.max_priority > 0 else 1.0  # Default max priority
        else:
            priority = (abs(td_error) + self.epsilon) ** self.alpha

        self.tree.add(priority, experience)


    def sample(self, batch_size):
        batch = []
        idxs = []
        self.priorities = []
        segment = self.tree.total_priority() / batch_size

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)

            value = np.random.uniform(a, b)
            index, priority, data = self.tree.get_leaf(value)

            self.priorities.append(priority)
            batch.append(data)
            idxs.append(index)

        sampling_probabilities = self.priorities / self.tree.total_priority()
        is_weights = np.power(self.capacity * sampling_probabilities, -self.beta)
        is_weights /= is_weights.max()  # Normalize weights

        self.beta = min(1.0, self.beta + self.beta_increment_per_sampling)
        self.alpha = min(.7, self.beta + self.alpha_increment_per_sampling)

        states_1, states_2, actions, rewards, dones = zip(*batch)
        return (
            np.array(states_1),
            np.array(states_2),
            np.array(actions),
            np.array(rewards),
            np.array(dones),
            np.array(is_weights, dtype=np.float32),
            idxs
        )

    def update_priorities(self, idxs, td_errors):
        for idx, td_error in zip(idxs, td_errors):
            priority = (abs(td_error) + self.epsilon) ** self.alpha
            self.tree.update(idx, priority)
        
    






