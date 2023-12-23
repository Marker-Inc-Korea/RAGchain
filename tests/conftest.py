import logging

import pytest

from RAGchain import linker

logger = logging.getLogger(__name__)


@pytest.hookimpl(tryfirst=True)
def pytest_sessionfinish(session, exitstatus):
    linker.flush_db()
    logger.info("Pytest Session End. Flushing linker DB")
