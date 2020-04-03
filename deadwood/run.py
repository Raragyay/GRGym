from deadwood.meld import Meld


class Run(Meld):
    def __init__(self, start: int, end: int):
        super().__init__()
        self.start = start
        self.end = end
        self.suit = start // 13
