import lightning as L
from mdutils.mdutils import MdUtils


class Callback(L.Callback):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
    def make_markdown_report(self, md: MdUtils, base_level: int = 1):
        """
        Make a markdown report presenting a summary of this callback.

        The markdown report root will be created at the experiment root and embedded
        document path should be relative to this folder.
        """
        pass