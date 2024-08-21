import pytest

from llm_evaluate.parsers.default_parser import DefaultParser


@pytest.mark.parametrize(
    'raw, expected', [
        ('Hello, how are you?', 'Hello, how are you?'),
        ('  12345, 12345!  ', '12345, 12345!'),
        (
            ' xxxxXKDL<LS aasdfk ,129 34kdaf 9dsakfsjklas asdp--sapdf- saf ',
            'xxxxXKDL<LS aasdfk ,129 34kdaf 9dsakfsjklas asdp--sapdf- saf',
        ),
    ],
)
def test_default_parser(raw: str, expected: str) -> None:
    parser = DefaultParser(none_value='', case_sensitive=True, multi_y=False)
    assert expected == parser.parse(prompt='', output=raw)
