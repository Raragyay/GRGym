from GRGym.environment.deadwood_counter cimport DeadwoodCounter
cimport numpy as np

def cytest_deadwood_counter_class():
    return DeadwoodCounter

def cytest_deadwood(DeadwoodCounter counter, int expected_deadwood):
    assert counter.deadwood() == expected_deadwood

def cytest_remaining_cards(DeadwoodCounter counter, np.ndarray[int, ndim=1] expected_remaining_cards):
    assert counter.remaining_cards() == set(expected_remaining_cards)

def cytest_melds(DeadwoodCounter counter, set expected_melds):
    assert counter.melds() == expected_melds

def cytest_deadwood_val(deadwood_counter_class, rank, expected_deadwood):
    assert DeadwoodCounter.deadwood_val(rank) == expected_deadwood
