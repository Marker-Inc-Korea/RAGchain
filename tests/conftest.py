import logging

import pytest

from RAGchain.utils.linker import RedisDBSingleton

logger = logging.getLogger(__name__)


@pytest.hookimpl(tryfirst=True)
def pytest_sessionfinish(session, exitstatus):
    r = RedisDBSingleton()
    r.flush_db()
    logger.info("Pytest Session End. Flushing Redis DB")
