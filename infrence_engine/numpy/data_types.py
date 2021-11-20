import numpy as np
from infrence_engine.data_type import data_types as base_data_types


class data_types(base_data_types):
    def __init__(self):
        self.base_python_types()
        self.int = ["i", "u"]
        self.float = ["f"]
        self.bool = ["b"]
        self.complex = ["c"]
        self.byte = ["S"]
        self.char = ["O", "U"]
        self.datetime = ["M"]
        self.timedelta = ["m"]
        self.other = ["V"]
        self.init_level2_types()

    def is_array(self, obj) -> bool:
        """checks if X is a numpy array"""
        return isinstance(obj, np.ndarray)

    def char_type(self, obj) -> str:
        """returns the dtype char of the numpy type"""
        if self.is_array(obj) is True:
            return obj.dtype.kind
        else:
            return ""

    def is_numeric(self, obj) -> bool:
        """checks if numpy numeric"""
        return self.char_type(obj) in tuple(self.numeric)

    def is_int(self, obj) -> bool:
        """checks if numpy integer"""
        return self.char_type(obj) in tuple(self.int)

    def is_float(self, obj) -> bool:
        """checks if numpy float"""
        return self.char_type(obj) in tuple(self.float)

    def is_bool(self, obj) -> bool:
        """checks if numpy bool"""
        return self.char_type(obj) in tuple(self.bool)

    def is_complex(self, obj) -> bool:
        """checks if numpy complex"""
        return self.char_type(obj) in tuple(self.complex)

    def is_byte(self, obj) -> bool:
        """checks if numpy byte"""
        return self.char_type(obj) in tuple(self.byte)

    def is_char(self, obj) -> bool:
        """checks if numpy char"""
        return self.char_type(obj) in tuple(self.char)

    def is_timedelta(self, obj) -> bool:
        """checks if numpy timedelta"""
        return self.char_type(obj) in tuple(self.timedelta)

    def is_datetime(self, obj) -> bool:
        """checks if numpy datetime"""
        return self.char_type(obj) in tuple(self.datetime)

    def is_other(self, obj) -> bool:
        """checks if numpy other"""
        return self.char_type(obj) in tuple(self.other)


class data_types_extended:
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
