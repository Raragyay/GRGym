from enum import IntEnum


class ActionResult(IntEnum):
    LOST_MATCH = -2
    LOST_HAND = -1
    NO_CHANGE = 0
    WON_HAND = 1
    WON_MATCH = 2
