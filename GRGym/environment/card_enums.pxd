cpdef enum Rank:
    ACE = 0
    TWO = 1
    THREE = 2
    FOUR = 3
    FIVE = 4
    SIX = 5
    SEVEN = 6
    EIGHT = 7
    NINE = 8
    TEN = 9
    JACK = 10
    QUEEN = 11
    KING = 12

cpdef enum Suit:
    DIAMONDS = 0
    CLUBS = 1
    HEARTS = 2
    SPADES = 3

cdef list suit_symbols()
cdef list suit_names()
cdef list rank_symbols()
cdef list rank_names()
