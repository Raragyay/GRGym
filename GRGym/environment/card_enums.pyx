cdef list _suit_symbols = ['D', 'C', 'H', 'S']
cdef list _suit_names = ['Diamonds', 'Clubs', 'Hearts', 'Spades']
cdef list _rank_symbols = ['A', 2, 3, 4, 5, 6, 7, 8, 9, 10, 'J', 'Q', 'K']
cdef list _rank_names = ['Ace', 2, 3, 4, 5, 6, 7, 8, 9, 10, 'Jack', 'Queen', 'King']

cdef list suit_symbols():
    return _suit_symbols
cdef list suit_names():
    return _suit_names
cdef list rank_symbols():
    return _rank_symbols
cdef list rank_names():
    return _rank_names
