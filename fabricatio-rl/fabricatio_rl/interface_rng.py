from typing import Union

import numpy as np


class FabricatioRNG:
    def __init__(self):
        self.__rng: Union[np.random.Generator, None] = None

    @property
    def rng(self):
        return self.__rng

    def reinitialize_rng(self, seed):
        if seed != -1:
            self.__rng = np.random.default_rng(seed)
        else:
            self.__rng = np.random.default_rng()
        return self
