import pytest
from unittest.mock import patch, mock_open
import json


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
    assert any("requirements.txt" in call[0] for call in mock_file.call_args_list)
    mock_file().write.assert_any_call(
        "fastapi\nuvicorn[standard]\nneomodel\nrequests\npython-dotenv\n",
    )
    # main.py
    assert any("app/main.py" in call[0] for call in mock_file.call_args_list)
    assert any(
        call[0][0].startswith("        from fastapi import FastAPI\n")
        for call in all_writes
    )
    assert any(
        call[0][0].endswith("# Include routers dynamically later\n        ")
        for call in all_writes
    )
    # .gitignore
    assert any(".gitignore" in call[0] for call in mock_file.call_args_list)
    # .env
    assert any(".env" in call[0] for call in mock_file.call_args_list)
    assert any(
        call[0][0].startswith("NEO4J_URI=") and not call[0][0].startswith("NEO4J_URI=<")
        for call in all_writes
    )
    # .env.example
    assert any(".env.example" in call[0] for call in mock_file.call_args_list)
    assert any(call[0][0].startswith("NEO4J_URI=<") for call in all_writes)
    # models/__init__.py
    assert any("app/models/__init__.py" in call[0] for call in mock_file.call_args_list)
    # models/models.py
    mock_file().write.assert_any_call("# Models will be generated here\n")
    # routers/__init__.py
    assert any(
        "app/routers/__init__.py" in call[0] for call in mock_file.call_args_list
    )


def test_generate_models_from_workspace_json():
    from create_neo4j_python_app.create_app import generate_models_from_workspace_json

    with open("test/example_model.json") as f:
        json_data = f.read()

    with open("test/expected_models_py") as f:
        expected_models_py = f.read()

    with patch("builtins.open", new_callable=mock_open) as mock_file:
        # Mock read method
        mock_file().read = lambda: json_data
        node_labels = generate_models_from_workspace_json("test/example_model.json")
        labels_list = [label["token"] for label in node_labels]
        assert len(labels_list) == 3
        assert "FirstLabel" in labels_list
        assert "SecondLabel" in labels_list
        assert "NodeWithNoRelation" in labels_list

        # Check if files were opened and written to and validate content
        # models.py
        assert any("models.py" in call[0] for call in mock_file.call_args_list)
        assert mock_file().write.call_args_list[0][0][0] == expected_models_py


def test_generate_crud_endpoints():
    from create_neo4j_python_app.create_app import generate_crud_endpoints

    with open("test/node_labels.json") as f:
        node_labels = json.load(f)

    with patch("builtins.open", new_callable=mock_open) as mock_file:
        generate_crud_endpoints(node_labels)

        all_writes = mock_file().write.call_args_list

        # Validate router code files
        assert any(
            "app/routers/first_label.py" in call[0] for call in mock_file.call_args_list
        )
        any(
            call[0][0].startswith(
                """
                from fastapi import APIRouter, HTTPException\n
                from models import FirstLabel\n
                from neomodel import db\n
                \n
                router = APIRouter(prefix='/first_labels', tags=['FirstLabel'])
                """
            )
            for call in all_writes
        )
        any(
            "@router.post('/')\ndef create_first_labels():" in call[0][0]
            for call in all_writes
        )
        assert any(
            "app/routers/second_label.py" in call[0]
            for call in mock_file.call_args_list
        )
        any(
            "@router.get('/')\ndef list_second_labels():" in call[0][0]
            for call in all_writes
        )
        any(
            "@router.get('/{uid}')\ndef get_second_label(uid: str):" in call[0][0]
            for call in all_writes
        )
        any(
            "@router.put('/{uid}')\ndef update_second_label(uid: str, payload: dict):"
            in call[0][0]
            for call in all_writes
        )
        any(
            "@router.delete('/{uid}')\ndef delete_second_label(uid: str):" in call[0][0]
            for call in all_writes
        )
        assert any(
            "app/routers/node_with_no_relation.py" in call[0]
            for call in mock_file.call_args_list
        )
        any(
            "@router.get('/')\ndef list_node_with_no_relations():" in call[0][0]
            for call in all_writes
        )

        # Validate router init (imports)
        assert any(
            "app/routers/__init__.py" in call[0] for call in mock_file.call_args_list
        )
        any(
            "from routers.first_label import router as first_label_router\n"
            == call[0][0]
            for call in all_writes
        )
        any(
            "from routers.second_label import router as second_label_router\n"
            == call[0][0]
            for call in all_writes
        )
        any(
            "from routers.node_with_no_relation import router as node_with_no_relation_router\n"
            == call[0][0]
            for call in all_writes
        )

        # Validate main.py (include routers)
        assert any("app/main.py" in call[0] for call in mock_file.call_args_list)
        any(
            "app.include_router(routers.first_label_router)\n" == call[0][0]
            for call in all_writes
        )


if __name__ == "__main__":
    pytest.main()
