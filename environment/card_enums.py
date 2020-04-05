from enum import IntEnum


class Rank(IntEnum):
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


class Suit(IntEnum):
    DIAMONDS = 0
    CLUBS = 1
    HEARTS = 2
    SPADES = 3
