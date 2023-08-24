import pytest
from dotenv import load_dotenv


@pytest.fixture(autouse=True, scope="session")
def setting():
    load_dotenv(dotenv_path='./.env')
