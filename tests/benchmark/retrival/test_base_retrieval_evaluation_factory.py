import os
import pytest
import yaml
from freezegun import freeze_time


class Testbase_retrieval_evaluation_factory:
    @pytest.fixture(scope="function", autouse=True)
    def setup_teardown(self, request):
        test_folder = os.path.dirname(os.path.realpath(__file__))
        self.test_conf_file = os.path.join(test_folder, "resources", "config.yml")

        with open(self.test_conf_file, 'r', encoding='utf-8') as yaml_file:
            self.yaml_dict = yaml.load(yaml_file, Loader=yaml.Loader)

        def teardown():
            print('Function Tear down - - - - ')

        request.addfinalizer(teardown)

    def test_MAPFactory():
        assert