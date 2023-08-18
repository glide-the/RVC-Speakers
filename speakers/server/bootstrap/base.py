
class Bootstrap:

    def __init__(self):
        self._version = "v0.0.1"

    @classmethod
    def from_config(cls, cfg=None):
        return cls()

    @classmethod
    async def run(cls):
        raise NotImplementedError
