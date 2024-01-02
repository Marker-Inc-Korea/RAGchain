import logging
import os
import warnings
from typing import Union, List
from uuid import UUID

import boto3
from botocore.exceptions import NoCredentialsError, ClientError

from RAGchain.utils.linker.base import BaseLinker, NoIdWarning, NoDataWarning

logger = logging.getLogger(__name__)


class DynamoLinker(BaseLinker):
    """
    DynamoDBSingleton is a singleton class that manages DynamoDB.
    """

    def __init__(self):
        aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
        aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        region_name = os.getenv("AWS_REGION")
        table_name = os.getenv("DYNAMODB_TABLE_NAME")

        if aws_access_key_id is None:
            raise ValueError("Please set AWS_ACCESS_KEY_ID to environment variable")
        if aws_secret_access_key is None:
            raise ValueError("Please set AWS_SECRET_ACCESS_KEY to environment variable")
        if region_name is None:
            raise ValueError("Please set AWS_REGION to environment variable")
        if table_name is None:
            raise ValueError("Please set DYNAMODB_TABLE_NAME to environment variable")

        try:
            self.dynamodb = boto3.resource(
                'dynamodb',
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                region_name=region_name
            )
        except NoCredentialsError:
            raise ValueError("Invalid AWS credentials")

        self.table = None
        self.table_name = table_name
        self.create_or_load_table(table_name)

    def create_table(self, table_name):
        """
        Create a table in DynamoDB table that can be used to store DB origin.
        """
        if table_name in self.dynamodb.meta.client.list_tables()['TableNames']:
            raise ValueError(f'{table_name} already exists')

        try:
            self.table = self.dynamodb.create_table(
                TableName=table_name,
                KeySchema=[{'AttributeName': 'id', 'KeyType': 'HASH'}],
                AttributeDefinitions=[{'AttributeName': 'id', 'AttributeType': 'S'}],
                ProvisionedThroughput={'ReadCapacityUnits': 5, 'WriteCapacityUnits': 5})
            self.table.wait_until_exists()
        except ClientError as err:
            self.handle_error(err, table_name, "create")
        else:
            return self.table

    def load_table(self, table_name):
        """
        Load a table in DynamoDB table that can be used to store DB origin.
        """
        if table_name not in self.dynamodb.meta.client.list_tables()['TableNames']:
            raise ValueError(f'{table_name} does not exist')
        try:
            table = self.dynamodb.Table(table_name)
            table.load()
            self.table = table
        except ClientError as err:
            self.handle_error(err, table_name, "load")
        else:
            return self.table

    def create_or_load_table(self, table_name):
        """
        Create a table if it does not exist, otherwise load it.
        """
        if table_name in self.dynamodb.meta.client.list_tables()['TableNames']:
            self.load_table(table_name)
        else:
            self.create_table(table_name)

    @staticmethod
    def handle_error(err, table_name, operation):
        logger.error(
            "Couldn't %s table %s. Here's why: %s: %s",
            operation,
            table_name,
            err.response["Error"]["Code"],
            err.response["Error"]["Message"],
        )
        raise

    def get_json(self, ids: List[Union[UUID, str]]):
        str_ids = [str(find_id) for find_id in ids]
        keys = [{'id': _id} for _id in str_ids]
        response = self.dynamodb.batch_get_item(
            RequestItems={
                self.table_name: {
                    'Keys': keys
                }
            }
        )
        final_response_list = response['Responses'][f'{self.table_name}']
        id_to_result = {result['id']: result for result in final_response_list}
        results = []
        for _id in str_ids:
            if _id not in id_to_result:
                warnings.warn(f"ID {_id} not found in Linker", NoIdWarning)
                results.append(None)
            else:
                results.append(id_to_result[_id]['data'])
                if id_to_result[_id]['data'] is None:
                    warnings.warn(f"Data {_id} not found in Linker", NoDataWarning)
        return results

    def flush_db(self):
        self.table.delete()

    def put_json(self, ids: List[Union[UUID, str]], json_data_list: List[dict]):
        assert len(ids) == len(json_data_list), "ids and json_data_list must have the same length"
        items = [{
            'PutRequest': {
                'Item': {
                    'id': str(_id),
                    'data': json_data
                }
            }
        } for _id, json_data in zip(ids, json_data_list)]
        request_items = {self.table_name: items}
        self.dynamodb.batch_write_item(RequestItems=request_items)

    def delete_json(self, ids: List[Union[UUID, str]]):
        str_ids = [str(_id) for _id in ids]
        items = [{
            'DeleteRequest': {
                'Key': {
                    'id': _id
                }
            }
        } for _id in str_ids]
        request_items = {self.table_name: items}
        self.dynamodb.batch_write_item(RequestItems=request_items)
