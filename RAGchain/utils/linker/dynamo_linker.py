import os
from typing import Union
from uuid import UUID
import logging
import warnings

import boto3
from botocore.exceptions import NoCredentialsError, ClientError
from dotenv import load_dotenv

from RAGchain.utils.linker.base import BaseLinker, NoIdWarning, NoDataWarning

logger = logging.getLogger(__name__)
load_dotenv()


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

    def get_json(self, ids: list[Union[UUID, str]]):
        str_ids = [str(find_id) for find_id in ids]
        data_list = []
        for find_id in str_ids:
            # Check if id exists in dynamo linker
            data = self.table.get_item(Key={'id': find_id})
            if 'Item' not in data:
                warnings.warn(f"ID {find_id} not found in Linker", NoIdWarning)
            else:
                db_origin = data['Item']['db_origin']
                # Check if data exists in dynamo linker
                if db_origin is None:
                    warnings.warn(f"Data {find_id} not found in Linker", NoDataWarning)
                    data_list.append(None)
                else:
                    data_list.append(db_origin)
        return data_list

    def flush_db(self):
        self.table.delete()

    def put_json(self, id: Union[UUID, str], json_data: dict):
        self.table.put_item(
            Item={
                'id': str(id),
                'db_origin': json_data
            }
        )

    def delete_json(self, id: Union[UUID, str]):
        self.table.delete_item(Key={'id': str(id)})
