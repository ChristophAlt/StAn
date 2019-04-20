from typing import List, Dict, Any, Optional


class Instance:

    def __init__(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        self.text = text
        self.metadata = metadata


class DatasetReader:

    def read(self, path: str) -> List[Instance]:
        raise NotImplementedError()
