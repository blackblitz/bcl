import numpy as np

test = np.lib.format.open_memmap(
    'test.npy', mode='w+', dtype=np.float32,
    shape=(1000000, 1000)
)
test[:] = 0.0
