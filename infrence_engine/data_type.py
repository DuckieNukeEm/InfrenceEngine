import os
from datetime import date, datetime, timedelta, time, tzinfo, timezone


def unpack(*args):
    """adds all elements of each arg to a list

    Details:
        if a elemnt of args is not iterable, it will make it iterable so it can add it to a list
        if it is iterable, it will iterate through each one

    """
    end_list = []
    for a in args:
        try:
            end_list = end_list + [x for x in a]
        except TypeError:
            end_list = end_list + [x for x in [a]]
    return end_list


class data_types:
    def __init__(self):
        self.base_python_types()
        self.init_level2_types()

    def base_python_types(self):
        """sets all the base level python types as attributes"""
        self.int = int
        self.float = float
        self.complex = complex
        self.bool = bool
        self.str = str
        self.tuple = tuple
        self.list = list
        self.range = range
        self.dict = dict
        self.set = set
        self.bytes = bytes
        self.bytearray = bytearray
        self.memoryview = memoryview
        self.datetime = datetime
        self.date = date
        self.timedelta = timedelta
        self.time = time
        self.tzinfo = tzinfo
        self.timezone = timezone

    def init_level2_types(
        self,
        numeric_type: list = [],
        sequence_type: list = [],
        mapping_type: list = [],
        byte_type: list = [],
        temporal_type: list = [],
    ):
        """joins attomic tpys together into commong gorups

        Args:
            numeric_type (list, optional): addition numeric types to add the numeric group. Defaults to [].
            sequence_type (list, optional): additional sequence types to add to the sequence group. Defaults to [].
            mapping_types (list, optional): addtional mapping types to add to the mapping group. Defaults to [].
            byte_type (list, optional): addtional byte type to add to the byte group. Defaults to [].
            temporal_type (list, optional): additional time/date based tpyes tot add to the temporal group. Defaults to [].

        Details:
            Here are how the following groups are defined
                numeric = int, float, complex, + numeric_type from args
                sequence = tuple, list, range + sequence_type from args
                mapping = dict + mapping_type from args
                binary = bytes, bytearray, memoryview + byte_type from args
                tempoarl = datetime, timerange + tempoarl_list from args
        """
        self.numeric = unpack(self.int, self.float, self.complex, numeric_type)
        self.sequence = unpack(self.tuple, self.list, self.range, sequence_type)
        self.mapping = unpack(self.dict, mapping_type)
        self.binary = unpack(self.bytes, self.bytearray, self.memoryview, byte_type)
        self.temporal = unpack(
            self.datetime,
            self.date,
            self.timedelta,
            self.time,
            self.tzinfo,
            self.timezone,
            temporal_type,
        )

    # Numeric types
    def is_int(self, obj) -> bool:
        """checks if integer"""
        return isinstance(obj, tuple(self.int))

    def is_float(self, obj) -> bool:
        """checks if float"""
        return isinstance(obj, tuple(self.float))

    def is_complex(self, obj) -> bool:
        """checks if complex"""
        return isinstance(obj, tuple(self.complex))

    def is_numeric(self, obj) -> bool:
        """checks if object is numeric"""
        return isinstance(obj, tuple(self.numeric))

    # bool types
    def is_bool(self, obj) -> bool:
        """checks if object is boolean"""
        return isinstance(obj, tuple(self.bool))

    # Sequence Types
    def is_list(self, obj) -> bool:
        """checks if object is a list"""
        return isinstance(obj, tuple(self.list))

    def is_tuple(self, obj) -> bool:
        """checks if object is a tuple"""
        return isinstance(obj, tuple(self.tuple))

    def is_range(self, obj) -> bool:
        """checks if object is a range"""
        return isinstance(obj, tuple(self.range))

    def is_sequence(self, obj) -> bool:
        """checks if object is a sequence tpye"""
        return isinstance(obj, tuple(self.sequence))

    # mapping types
    def is_dict(self, obj) -> bool:
        """checks if object is a dictionary"""
        return isinstance(obj, tuple(dict))

    def is_map(self, obj) -> bool:
        """checks if object is a mappng type"""
        return isinstance(obj, tuple(self.mapping))

    # set

    def is_set(self, obj) -> bool:
        """checks if object is a set"""
        return isinstance(obj, set)

    # str related
    def is_string(self, obj) -> bool:
        """checks if object is a string"""
        return isinstance(obj, tuple(self.str))

    def is_byte(self, obj) -> bool:
        """checks if object is a byte"""
        return isinstance(obj, tuple(self.bytes))

    # byte related
    def is_bytearray(self, obj) -> bool:
        """checks if object is bytearray"""
        return isinstance(obj, tuple(self.bytearray))

    def is_memoryview(self, obj) -> bool:
        """checks if object is memoryvew"""
        return isinstance(obj, tuple(self.memoryview))

    def is_binary(self, obj) -> bool:
        """checks if object is in the binary group"""
        return isinstance(obj, tuple(self.binary))

    # time/date related
    def is_datetime(self, obj) -> bool:
        """checks if object is datetime"""
        return isinstance(obj, tuple(self.datetime))

    def is_date(self, obj) -> bool:
        """checks if object is date"""
        return isinstance(obj, tuple(self.date))

    def is_timedelta(self, obj) -> bool:
        """checks if object is timedelta"""
        return isinstance(obj, tuple(self.timedelta))

    def is_time(self, obj) -> bool:
        """checks if object is time"""
        return isinstance(obj, tuple(self.time))

    def is_tzinfo(self, obj) -> bool:
        """checks if object is tzinfo"""
        return isinstance(obj, tuple(self.tzinfo))

    def is_timezone(self, obj) -> bool:
        """checks if object is timezone"""
        return isinstance(obj, tuple(self.timezone))

    def is_temporal(self, obj) -> bool:
        """checks if object is in the temporal group"""
        return isinstance(obj, tuple(self.temporal))

    # os related
    def is_file(self, obj) -> bool:
        """checks if object is a file"""
        return os.path.isfile(obj)

    def is_dir(self, obj) -> bool:
        """checks if object is a file"""
        return os.path.isdir(obj)
