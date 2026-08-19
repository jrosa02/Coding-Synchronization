from coding_synchronization.StageABC import StageABC

class DecoderStageABC(StageABC):
    def __init__(self, seed: int = 42) -> None:
        super().__init__(seed)