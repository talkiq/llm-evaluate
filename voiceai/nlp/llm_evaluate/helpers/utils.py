import importlib
import logging
import re
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Type
from typing import Union


def camel_to_snake(name: str) -> str:
    name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', name).lower()


def try_parse_numeric(val: str) -> Optional[Union[int, float]]:
    try:
        numeric = float(val)
        if numeric.is_integer():
            return int(numeric)
        return numeric
    except Exception:
        return None


def parse_kwargs(options: Union[List[str], Dict[str, Any]]) -> Dict[str, Any]:
    if isinstance(options, list):
        options = dict(opt.split('=') for opt in options)

    parsed = {}
    for key, val in options.items():
        numeric_val = try_parse_numeric(val)
        if isinstance(val, (bool, float, int)):
            parsed[key] = val
        elif val.lower() in {'true', 'false'}:
            parsed[key] = bool(val.lower() == 'true')
        elif numeric_val is not None:
            parsed[key] = numeric_val
        elif ',' in val:
            parsed[key] = val.split(',')
        else:
            parsed[key] = val
    return parsed


def dynamic_load_class(filepath: str, class_name: str,
                       module_name: str) -> Type:
    logging.debug('loading class %s into module %s from %s',
                  class_name, module_name, filepath)
    spec = importlib.util.spec_from_file_location(module_name, filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, class_name)
