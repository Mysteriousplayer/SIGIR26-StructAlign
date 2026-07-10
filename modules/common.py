import os
import pickle
from contextlib import contextmanager
from comet_ml import Experiment


class _NullExperiment:
    """Fallback experiment object used when Comet logging is disabled."""

    def set_name(self, *args, **kwargs):
        return None

    def log_metric(self, *args, **kwargs):
        return None

    def log_metrics(self, *args, **kwargs):
        return None

    @contextmanager
    def validate(self):
        yield self


def initialize_experiment(config):
    """Initialize an optional CometML experiment."""
    api_key = getattr(config, 'api_key', None)
    project_name = getattr(config, 'project_name', None)
    workspace = getattr(config, 'workspace', None)

    placeholder_values = {None, '', 'YOUR API KEY', 'YOUR WORKSPACE', 'YOUR PROJECT NAME'}
    if api_key in placeholder_values:
        return _NullExperiment()

    experiment_kwargs = {'api_key': api_key}
    if project_name not in placeholder_values:
        experiment_kwargs['project_name'] = project_name
    if workspace not in placeholder_values:
        experiment_kwargs['workspace'] = workspace

    return Experiment(**experiment_kwargs)


def load_dataset(path: str):
    """Load dataset from a pickle file (generic function)."""
    with open(path, 'rb') as handle:
        data = pickle.load(handle)
    return data


def get_tokenizer(config):
    """Get an appropriate tokenizer based on configuration (generic function)."""
    os.environ['TOKENIZERS_PARALLELISM'] = "false"

    if config.huggingface:
        from transformers import CLIPTokenizer
        return CLIPTokenizer.from_pretrained(
            "/mnt/data/wang_shaokun/CTVR/clip-vit-base-patch32",
            TOKENIZERS_PARALLELISM=False,
            clean_up_tokenization_spaces=True
        )
    else:
        from modules.tokenization_clip import SimpleTokenizer
        return SimpleTokenizer()
