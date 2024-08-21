import logging
import re
from importlib import util
from typing import Any


def camel_to_snake(name: str) -> str:
    name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', name).lower()


def try_parse_numeric(val: str) -> int | float | None:
    try:
        numeric = float(val)
        if numeric.is_integer():
            return int(numeric)
        return numeric
    except Exception:
        return None


def parse_kwargs(options: list[str] | dict[str, Any]) -> dict[str, Any]:
    if isinstance(options, list):
        options = dict(opt.split('=') for opt in options)

    parsed: dict[str, Any] = {}
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


def dynamic_load_class(
    filepath: str, class_name: str,
    module_name: str,
) -> Any:
    logging.debug(
        'loading class %s into module %s from %s',
        class_name, module_name, filepath,
    )
    spec = util.spec_from_file_location(module_name, filepath)
    if not spec:
        raise ImportError(f'Cannot load module {module_name} '
                          f'from {filepath}')
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, class_name)
