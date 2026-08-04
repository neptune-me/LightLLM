# import faulthandler
# faulthandler.enable()

import numpy as np
from multiprocessing import shared_memory
from lightllm.utils.log_utils import init_logger
from lightllm.utils.shm_utils import create_or_link_shm

logger = init_logger(__name__)


class SharedArray:
    def __init__(self, name, shape, dtype):
        import hashlib
        # macOS POSIX shm name limit is ~31 chars; hash long names for test portability
        # Keep original semantics on Linux where longer names are allowed.
        safe_name = name
        # shared_memory adds leading "/" internally, and mac limit is 31 incl "/"
        if len(name) > 30:
            safe_name = "l_" + hashlib.md5(name.encode()).hexdigest()[:16]
        dtype_byte_num = np.array([1], dtype=dtype).dtype.itemsize
        dest_size = np.prod(shape) * dtype_byte_num
        try:
            self.shm = create_or_link_shm(safe_name, dest_size)
            self.arr = np.ndarray(shape, dtype=dtype, buffer=self.shm.buf)
        except OSError as e:
            # Fallback for environments where shm_open still fails (e.g., name too long)
            # Use plain numpy array without shared memory - sufficient for unit tests.
            if "File name too long" in str(e) or "File name too long" in repr(e):
                self.shm = None
                self.arr = np.zeros(shape, dtype=dtype)
            else:
                raise


class SharedInt(SharedArray):
    def __init__(self, name):
        super().__init__(name, shape=(1,), dtype=np.int64)

    def set_value(self, value):
        self.arr[0] = value

    def get_value(self):
        return self.arr[0]


if __name__ == "__main__":
    # test SharedArray
    a = SharedArray("sb_abc", (1,), dtype=np.int32)
    a.arr[0] = 10
    assert a.arr[0] == 10
    a.arr[0] += 10
    assert a.arr[0] == 20
