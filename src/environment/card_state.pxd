cpdef enum CardState:
    UNKNOWN = 0
    MINE_FROM_DECK = 1
    MINE_FROM_DISCARD = 2
    # no THEIRS_FROM_DECK, player will never know what opponent drew from deck
    THEIRS_FROM_DISCARD = 3
    DISCARD_MINE_TOP = 4
    DISCARD_MINE = 5
    DISCARD_THEIRS_TOP = 6
    DISCARD_THEIRS = 7
