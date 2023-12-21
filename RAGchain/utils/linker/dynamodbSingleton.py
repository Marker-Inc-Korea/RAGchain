import os
from typing import Union
from uuid import UUID
import logging

import boto3
from botocore.exceptions import NoCredentialsError, ClientError

logger = logging.getLogger(__name__)


class DynamoDBSingleton:
    """
    DynamoDBSingleton is a singleton class that manages DynamoDB.
    """
    __instance = None
    _is_initialized = False

    def __new__(cls, *args, **kwargs):
        if not cls.__instance:
            cls.__instance = super().__new__(cls)
        return cls.__instance

    def __init__(self):
        if self._is_initialized:
            return

        aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
        aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        region_name = os.getenv("AWS_REGION")
        table_name = os.getenv("DYNAMODB_TABLE_NAME")

        if not all([aws_access_key_id, aws_secret_access_key, region_name, table_name]):
            raise ValueError("Please set AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION, "
                             "and DYNAMODB_TABLE_NAME to environment variables")

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
        self.create_or_load_table(table_name)
        self._is_initialized = True

    def create_table(self, table_name):
        """
        Create a table in DynamoDB table that can be used to store DB origin.
        """
        if table_name in self.dynamodb.meta.client.list_tables()['TableNames']:
            raise ValueError(f'{table_name} already exists')

        try:
            self.table = self.dynamodb.create_table(
                TableName=table_name,
                KeySchema=[{'AttributeName': 'id', 'KeyType': 'STRING'}],
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

    def get_json(self, ids: list[Union[UUID, str]]):
        str_ids = [str(find_id) for find_id in ids]
        return [self.table.get_item(Key={'id': find_id})['db_origin'] for find_id in str_ids]

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

    def flush_db(self):
        self.table.delete()

    def __del__(self):
        self.dynamodb.close()
