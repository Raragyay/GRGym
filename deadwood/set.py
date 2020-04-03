from deadwood.meld import Meld


class Set(Meld):
    def __init__(self, rank: int):
        super().__init__()
        self.rank = rank
