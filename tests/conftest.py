import logging

import pytest

from RAGchain import linker

logger = logging.getLogger(__name__)


@pytest.hookimpl(tryfirst=True)
def pytest_sessionfinish(session, exitstatus):
    try:
        linker.flush_db()
    except FileNotFoundError:
        logger.debug("Json Linker already flushed")
    logger.info("Pytest Session End. Flushing linker DB")
