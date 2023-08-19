
class Bootstrap:

    """Used by web module to decide which secret for securing"""
    _NONCE: str = ''

    def __init__(self):
        self._version = "v0.0.1"

    @classmethod
    def from_config(cls, cfg=None):
        return cls()

    @property
    def nonce(self) -> str:
        return self._NONCE

    def set_nonce(self, nonce: str):
        self._NONCE = nonce

    @classmethod
    async def run(cls):
        raise NotImplementedError
