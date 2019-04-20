from typing import List, Dict, Optional, Any

from joblib import Parallel, delayed
from tqdm import tqdm

from stan.dataset_readers import Instance


class AnnotatedInstance:
    def __init__(
            self,
            tokens: List[str],
            annotations: Optional[Dict[str, Any]] = None,
            metadata: Optional[Dict[str, Any]] = None,
            ) -> None:
        self.tokens = tokens
        self.annotations = annotations
        self.metadata = metadata


class Annotator:

    def __init__(self, n_jobs: int) -> None:
        self.n_jobs = n_jobs

    def annotate_instances(self, instances: List[Instance]) -> List[AnnotatedInstance]:
        annotated_instances = Parallel(n_jobs=self.n_jobs, prefer="threads")(
            delayed(self.annotate)(instance) for instance in tqdm(instances)
        )
        return annotated_instances

    def annotate(self, instance: Instance) -> AnnotatedInstance:
        raise NotImplementedError()

    def cleanup(self) -> None:
        pass
