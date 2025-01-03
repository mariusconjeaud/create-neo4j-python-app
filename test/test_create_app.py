import pytest
from unittest.mock import patch, mock_open
import requests
import os


# Test get_oauth_token function
@pytest.mark.usefixtures("requests_mock")
def test_get_oauth_token_success(requests_mock):
    from create_neo4j_python_app.create_app import get_oauth_token

    requests_mock.post(
        "https://api.neo4j.io/oauth/token",
        json={"access_token": "test_token"},
        status_code=200,
    )

    # Call the function
    get_oauth_token("client_id", "client_secret")
    from create_neo4j_python_app.create_app import ACCESS_TOKEN

    assert ACCESS_TOKEN == "test_token"


# Test create_neo4j_aura_instance function
@pytest.mark.usefixtures("requests_mock")
def test_create_neo4j_aura_instance(requests_mock):
    from create_neo4j_python_app.create_app import create_neo4j_aura_instance

    requests_mock.post(
        "https://api.neo4j.io/oauth/token",
        json={"access_token": "test_token"},
        status_code=200,
    )
    requests_mock.get(
        "https://api.neo4j.io/v1/tenants",
        json={"data": [{"id": "tenant-id-123"}]},
        status_code=200,
    )
    requests_mock.post(
        "https://api.neo4j.io/v1/instances",
        json={
            "data": {
                "connection_url": "bolt://localhost:7687",
                "username": "neo4j",
                "password": "test_password",
                "id": "instance-id-123",
            }
        },
        status_code=202,
    )

    # Execute instance creation
    create_neo4j_aura_instance("client_id", "client_secret")

    from create_neo4j_python_app.create_app import (
        TENANT_ID,
        INSTANCE_URI,
        INSTANCE_USERNAME,
        INSTANCE_PASSWORD,
        INSTANCE_ID,
    )

    # Asserts to verify proper assignment
    assert TENANT_ID == "tenant-id-123"
    assert INSTANCE_URI == "bolt://localhost:7687"
    assert INSTANCE_USERNAME == "neo4j"
    assert INSTANCE_PASSWORD == "test_password"
    assert INSTANCE_ID == "instance-id-123"


# Test create_folder_structure function
@patch("os.makedirs")
@patch("builtins.open", new_callable=mock_open)
def test_create_folder_structure(mock_file, mock_makedirs):
    from create_neo4j_python_app.create_app import create_folder_structure

    create_folder_structure()

    # Check if directory creation was attempted
    mock_makedirs.assert_called()

    all_writes = mock_file().write.call_args_list
    # Check if files were opened and written to
    # requirements.txt
    mock_file().write.assert_any_call(
        "fastapi\nuvicorn[standard]\nneomodel\nrequests\npython-dotenv\n",
    )
    # main.py
    assert any(
        call[0][0].startswith("        from fastapi import FastAPI\n")
        for call in all_writes
    )
    assert any(
        call[0][0].endswith("# Include routers dynamically later\n        ")
        for call in all_writes
    )
    # .env
    assert any(
        call[0][0].startswith("NEO4J_URI=") and not call[0][0].startswith("NEO4J_URI=<")
        for call in all_writes
    )
    # .env.example
    assert any(call[0][0].startswith("NEO4J_URI=<") for call in all_writes)
    # models/__init__.py
    mock_file().write.assert_any_call(
        "# Package init",
    )
    # models/models.py
    mock_file().write.assert_any_call("# Models will be generated here\n")
    # routers/__init__.py
    mock_file().write.assert_any_call(
        "# Routers import\n",
    )


# More tests can be added here for other functions


if __name__ == "__main__":
    pytest.main()
