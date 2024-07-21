from .BaseTrainerCL import BaseTrainerCL

class MTLTrainer(BaseTrainerCL):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
