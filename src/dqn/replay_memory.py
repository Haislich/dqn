""" 
Module containing the definition of an `Experience` and a `ReplayBuffer`
"""

import random
from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass
class Experience:
    """
    An experience is a quadruple `(x,a,y,r)`, meaning that
    the execution of an action `a` in a state `x`
    results in a new state `y` and reward `r`

    This concept is  discussed in more detail in Section 3.5, "Experience Replay,"
    of "Reinforcement Learning for Robots Using Neural Networks."[1]

    References:
        - [1]: https://isl.iar.kit.edu/pdf/Lin1993.pdf
    """

    state: np.ndarray
    action: int | np.ndarray
    next_state: np.ndarray
    reward: float


class ReplayMemory:
    """A memory buffer for storing experiences.

    The `ReplayMemory` class maintains a limited size deque of experiences.
    It only retains the last `N` experiences, discarding older ones as new
    experiences are added. This is done to prioritize learning from more recent
    experiences, which are likely to be more relevant as they reflect the improved
    strategies developed over time.

    More details on the importance and mechanics of experience replay can be found
    in Section 3.5.2, "Over-Replay," of the document "Reinforcement Learning
    for Robots Using Neural Networks." [1] and Section 4
    "Deep reinforcement learning" of the paper
    "Playing Atari with deep reinforcement learning" [2]


    References:
        - [1]: https://isl.iar.kit.edu/pdf/Lin1993.pdf
        - [2]: https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf
    """

    def __init__(self, capacity: int) -> None:
        """Initializes the ReplayMemory with a specified capacity.

        Args:
            capacity (int): The maximum number of experiences the memory can hold.
        """
        # To be efficient in terms of computational time and memory space,
        # the memory can keep only the most recent experiences, because experiences
        # in the far past typically invove many bad choiches and thus are less worth
        # keeping than recent experiences
        #
        # This concept is  discussed in more detail in Section 3.5, "Experience Replay,"
        # of "Reinforcement Learning for Robots Using Neural Networks."[1]

        #  [1]: https://isl.iar.kit.edu/pdf/Lin1993.pdf
        #
        self.memory: "deque[Experience]" = deque([], maxlen=capacity)

    def store(self, experience: Experience):
        """Save the current experience into the replay memory"""
        self.memory.append(experience)

    def sample(self, minibatch_size: int):
        """Samples uniformly at random from the memory.
        The sampling size is adjusted if the requested
        minibatch size is greater than the number of stored experiences.

        Returns:
            list[Experience]: A list of random experiences
        """
        return random.sample(self.memory, min(len(self.memory), minibatch_size))
