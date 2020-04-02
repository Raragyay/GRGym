import numpy as np

num_of_card_per_deck = np.random.randint(0, 53, 10)  # 10 decks
for idx, card_num in enumerate(num_of_card_per_deck):
    deck = np.random.randint(0, 52, card_num)
    mask = np.zeros(52, np.bool)
    mask[deck] = 1
    np.save(f"test_has_card.{idx + 1}", mask)
