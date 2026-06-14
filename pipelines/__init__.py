from .utils import get_pipeline
from .config import get_config

from .tasks.causal_language_modeling import (
    CausalLanguageModelingPipeline,
    CausalLanguageModelingNanoPipeline,
)
from .tasks.image_classification import ImageClassificationPipeline
