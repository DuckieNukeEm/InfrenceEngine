import numpy as np


class data_types:
    def __init__(self):
        self.int = [
            np.int_,
            np.intc,
            np.int8,
            np.int16,
            np.int32,
            np.int64,
            np.uint8,
            np.uint16,
            np.uint32,
            np.uint64,
        ]
        self.float = [np.float_, np.float_16, np.float32, np.float64]
        self.numeric = self.int + self.float
        self.bool = np.bool
        self.complex = [np.complex64, np.complex_, np.complex128]
        self.byte = [np.byte, np.ubyte]

    def is_array(self, obj) -> bool:
        """checks if X is a numpy array"""
        return isinstance(obj, np.ndarray)

    def is_numeric(self, obj) -> bool:
        """checks if numeric"""
        if self.is_array(obj) and obj.dtypes in self.numeric:
            return True
        else:
            return False

    def is_int(self, obj) -> bool:
        """checks if numeric"""
        if self.is_array(obj) and obj.dtypes in self.int:
            return True
        else:
            return False

    def is_float(self, obj) -> bool:
        """checks if numeric"""
        if self.is_array(obj) and obj.dtypes in self.float:
            return True
        else:
            return False

    def is_bool(self, obj) -> bool:
        """checks if numeric"""
        if self.is_array(obj) and obj.dtypes in self.bool:
            return True
        else:
            return False

    def is_complex(self, obj) -> bool:
        """checks if numeric"""
        if self.is_array(obj) and obj.dtypes in self.complex:
            return True
        else:
            return False

    def is_byte(self, obj) -> bool:
        """checks if numeric"""
        if self.is_array(obj) and obj.dtypes in self.byte:
            return True
        else:
            return False
