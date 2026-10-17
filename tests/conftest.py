from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


@pytest.fixture
def formulaic() -> str:
    return (EXAMPLES / "formulaic.md").read_text()


@pytest.fixture
def specific() -> str:
    return (EXAMPLES / "specific.md").read_text()
