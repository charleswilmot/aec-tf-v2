import pickle
import numpy as np


class Buffer(object):
    def __init__(self, size):
        self.size = size
        self.current_last = 0
        self._hparams = {
            "buffer_size": self.size,
        }
        self.dtype = None
        self.sample_index = 0

    def dump(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.buffer, f)

    def enough(self, n):
        return n <= self.current_last

    def integrate(self, data):
        if self.dtype is None: # must create buffer and dtype
            self.dtype = data.dtype
            self.buffer = np.zeros(self.size, dtype=self.dtype)
        data = data.flatten()
        n = data.shape[0]
        indices = self.get_insertion_indices(n)
        if self.current_last < self.size:
            self.current_last += n
        if self.current_last > self.size:
            self.current_last = self.size
        self.buffer[indices] = data

    def get_insertion_indices(self, n):
        if self.current_last < self.size:
            space_remaining = self.size - self.current_last
            if space_remaining < n:
                # not enough room to insert the full episode
                part1 = np.random.choice(
                    np.arange(self.current_last),
                    n - space_remaining,
                    replace=False
                )
                part2 = np.arange(self.current_last, self.size)
                return np.concatenate((part1, part2))
            else: # enough empty space
                return slice(self.current_last, self.current_last + n)
        else: # buffer already full
            return np.random.choice(self.size, n, replace=False)

    def sample(self, batch_size):
        if self.current_last < batch_size or batch_size > self.size:
            return np.arange(self.current_last), self.buffer[:self.current_last]
        batch_last = self.sample_index + batch_size
        if batch_last < self.current_last:
            ret = self.buffer[self.sample_index:batch_last]
            indices = np.arange(self.sample_index, batch_last)
            self.sample_index = batch_last
            return indices, ret
        else: # enough data in buffer but exceed its size
            part1 = self.buffer[self.sample_index:self.current_last]
            part2 = self.buffer[:batch_last - self.current_last]
            indices_1 = np.arange(self.sample_index, self.current_last)
            indices_2 = np.arange(batch_last - self.current_last)
            indices = np.concatenate((indices_1, indices_2))
            data = np.concatenate((part1, part2))
            self.sample_index = batch_last - self.current_last
            return indices, data
