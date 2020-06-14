cdef class Observation:
    property player_id:
        def __get__(self):
            return self.__player_id
