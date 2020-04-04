class Meld:
    def __init__(self):
        pass

    def connectable_cards(self):
        raise NotImplementedError()

    def __hash__(self):
        raise NotImplementedError()

    def __eq__(self, other):
        raise NotImplementedError()
