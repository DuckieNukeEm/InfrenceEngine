import os


class data_types:
    def __init__(self):
        self.int = int()
        self.float = float()

    def is_int(self, obj) -> bool:
        """checks if integer"""
        return isinstance(obj, int)

    def is_float(self, obj) -> bool:
        """checks if float"""
        return isinstance(obj, float)

    def is_complex(self, obj) -> bool:
        """checks if complex"""
        return isinstance(obj, complex)

    def is_numeric(self, obj) -> bool:
        """checks if object is numeric"""
        return self.is_int(obj) or self.is_float(obj) or self.is_complex(obj)

    def is_bool(self, obj) -> bool:
        """checks if object is boolean"""
        return isinstance(obj, bool)

    def is_string(self, obj) -> bool:
        """checks if object is a string"""
        return isinstance(obj, str)

    def is_list(self, obj) -> bool:
        """checks if object is a list"""
        return isinstance(obj, list)

    def is_dict(self, obj) -> bool:
        """checks if object is a dictionary"""
        return isinstance(obj, dict)

    def is_tuple(self, obj) -> bool:
        """checks if object is a tuple"""
        return isinstance(obj, tuple)

    def is_file(self, obj) -> bool:
        """checks if object is a file"""
        return os.path.isfile(obj)

    def is_dir(self, obj) -> bool:
        """checks if object is a file"""
        return os.path.isdir(obj)

    def is_set(self, obj) -> bool:
        """checks if object is a set"""
        return isinstance(obj, set)

    def is_byte(self, obj) -> bool:
        """checks if object is a byte"""
        return isinstance(obj, bytes)
