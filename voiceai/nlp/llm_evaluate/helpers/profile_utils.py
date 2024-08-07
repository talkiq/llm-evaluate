import os
import pathlib
import warnings
from typing import Optional

from hydra import compose
from hydra import initialize

from ..datasets import load_dataset_catalog
from ..metrics import load_metric_catalog
from ..models import load_model_catalog
from ..models import load_model_spec
from ..parsers import load_parser_catalog
from ..processing.benchmark import load_benchmark_catalog
from .profile import Profile


def load_profile(profile_path: Optional[str]) -> Profile:
    if not (profile_path and os.path.exists(profile_path)):
        raise ValueError('Profile not specified or file does not exist!')
    default_load_path = 'voiceai/nlp/llm_evaluate'
    path = pathlib.Path(os.path.abspath(profile_path))
    profile_path = path.name.replace('.yaml', '')
    custom_load_path = os.path.relpath(
        path.parents[0], pathlib.Path(__file__).parents[0])

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=UserWarning)
        with initialize(
            version_base=None,
            config_path=custom_load_path,
            job_name='benchmark',
        ):
            profile = compose(
                config_name=profile_path,
                overrides=[f'hydra.searchpath=[pkg://{default_load_path}]'],
            )

    return Profile(
        extensions=profile.get('extensions', {}),
        parsers=load_parser_catalog(profile['parsers']),
        benchmarks=load_benchmark_catalog(profile['benchmarks']),
        datasets=load_dataset_catalog(profile['datasets']),
        metrics=load_metric_catalog(profile['metrics']),
        model_loaders=load_model_catalog(profile['model_loaders']),
        model=load_model_spec(
            profile['model'], load_model_catalog(
                profile['model_loaders'])),
    )
