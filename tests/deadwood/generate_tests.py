import random
from pathlib import Path

import numpy as np


def deadwood_val(card: int) -> int:
    rank = card % 13
    if rank >= 9:
        return 10
    else:
        pass
    return rank + 1  # zero-indexed


path = Path()

deck = np.zeros(52, np.int8)
cards = random.sample(range(52), 10)
max_deadwood = sum(map(deadwood_val, cards))
deck.put(cards, 1)
deck = np.reshape(deck, (4, 13))
print(deck)
print(max_deadwood)
deadwood = input("Deadwood: ")

with open("gen_test.txt", "a") as f:
    np.savetxt(f, deck, delimiter=" ", fmt="%d")
    f.write(f"{deadwood}\n")
