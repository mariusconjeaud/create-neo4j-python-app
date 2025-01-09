"""
This script is used to create a new Neo4j Aura Free instance
and scaffold a Python API project using neomodel and FastAPI.
"""

import argparse
import json
import os
import re
import time
from typing import Any

import inflect
import requests
from requests.auth import HTTPBasicAuth

ACCESS_TOKEN = None
TENANT_ID = None
INSTANCE_ID = None
INSTANCE_NAME = None
INSTANCE_URI = None
INSTANCE_USERNAME = None
INSTANCE_PASSWORD = None

ROOT_DIRECTORY = "app"
MAIN_FILE = f"{ROOT_DIRECTORY}/main.py"
MODELS_DIRECTORY = f"{ROOT_DIRECTORY}/models"
MODELS_FILE = f"{MODELS_DIRECTORY}/models.py"
ROUTERS_DIRECTORY = f"{ROOT_DIRECTORY}/routers"

BASE_AURA_URL = "https://api.neo4j.io/v1"

# Used to pluralize node names for the routers
p = inflect.engine()


def get_oauth_token(client_id: str, client_secret: str) -> None:
    """
    Retrieves an OAuth token from Neo4j's authentication service using client credentials.

    This function makes a POST request to Neo4j's OAuth endpoint to obtain an access token
    using the client credentials authentication flow. Upon successful authentication,
    it stores the access token in a global variable ACCESS_TOKEN.

    Args:
        client_id (str): The client ID for authentication
        client_secret (str): The client secret for authentication

    Returns:
        None

    Side Effects:
        - Sets global ACCESS_TOKEN variable when authentication is successful
        - Prints success/error messages to standard output
    """
    global ACCESS_TOKEN
    url = "https://api.neo4j.io/oauth/token"

    data = {"grant_type": "client_credentials"}
    headers = {"Content-Type": "application/x-www-form-urlencoded"}

    response = requests.request(
        "POST",
        url,
        headers=headers,
        data=data,
        auth=HTTPBasicAuth(client_id, client_secret),
        timeout=10,
    )

    if response.status_code == 200:
        ACCESS_TOKEN = response.json().get("access_token")
        print("Successfully authenticated")
    else:
        print(f"Error: {response.status_code}")
        print(response.text)


def set_tenant_id() -> None:
    """
    Retrieves the tenant ID from the Neo4j Aura service.

    This function makes a GET request to the Neo4j Aura service to retrieve the tenant ID
    for the authenticated user. It stores the tenant ID in a global variable TENANT_ID.

    Returns:
        None

    Side Effects:
        - Sets global TENANT_ID variable when the tenant ID is successfully retrieved
        - Prints success/error messages to standard output
    """
    global TENANT_ID
    url = f"{BASE_AURA_URL}/tenants"
    headers = {"Accept": "application/json", "Authorization": f"Bearer {ACCESS_TOKEN}"}

    response = requests.get(url, headers=headers, timeout=10)

    if response.status_code == 200:
        TENANT_ID = response.json()["data"][0]["id"]
        print("Successfully retrieved tenant ID")
    else:
        print(f"Error: {response.status_code}")
        print(response.text)


def create_neo4j_aura_instance(client_id: str, client_secret: str) -> None:
    """Create a Neo4j Aura instance using the Neo4j Aura API.

    This function handles the creation of a Neo4j Aura instance by making API calls to Neo4j's
    Aura service. It sets up global variables for connection details and creates a free-tier
    database instance in the specified region.

    Args:
        client_id (str): The OAuth client ID for authentication with Neo4j Aura API
        client_secret (str): The OAuth client secret for authentication with Neo4j Aura API

    Returns:
        None

    Global Variables Modified:
        ACCESS_TOKEN: The OAuth access token for API requests
        TENANT_ID: The ID of the tenant in Neo4j Aura
        INSTANCE_ID: The ID of the created database instance
        INSTANCE_NAME: The name of the database instance
        INSTANCE_URI: The connection URI for the database
        INSTANCE_USERNAME: The username for database authentication
        INSTANCE_PASSWORD: The password for database authentication

    Notes:
        - Creates a free-tier database ('free-db') with 1GB memory
        - Deploys on Google Cloud Platform in europe-west1 region
        - Uses Neo4j version 5
        - Instance creation may take several minutes to complete
    """
    global ACCESS_TOKEN
    global TENANT_ID
    global INSTANCE_ID
    global INSTANCE_NAME
    global INSTANCE_URI
    global INSTANCE_USERNAME
    global INSTANCE_PASSWORD
    # First, get authorization token
    get_oauth_token(client_id, client_secret)

    set_tenant_id()

    # Create instance
    headers = {"Accept": "application/json", "Authorization": f"Bearer {ACCESS_TOKEN}"}

    payload = {
        "name": INSTANCE_NAME,
        "type": "free-db",
        "tenant_id": TENANT_ID,
        "version": "5",
        "region": "europe-west1",
        "memory": "1GB",
        "cloud_provider": "gcp",
    }

    response = requests.post(
        f"{BASE_AURA_URL}/instances", json=payload, headers=headers, timeout=60
    )

    if response.status_code == 202:
        data = response.json()["data"]
        # Extract connection details: uri, username, password
        INSTANCE_URI = data["connection_url"]
        INSTANCE_USERNAME = data["username"]
        INSTANCE_PASSWORD = data["password"]
        INSTANCE_ID = data["id"]
        print(
            f"Successfully created instance with id {INSTANCE_ID}. Connection details:"
        )
        print(f"URI: {INSTANCE_URI}")
        print(f"Username: {INSTANCE_USERNAME}")
        print(f"Password: {INSTANCE_PASSWORD}")
        print(
            "Instances can take a few minutes to be ready. We will give you a heads up when it's ready."
        )
        print("Alternatively, go to https://console.neo4j.io/ to check the status.")
    else:
        print("Failed to create instance:", response.text)


def wait_for_instance_ready() -> None:
    """Wait for Neo4j AuraDB instance to be in ready state.

    Makes HTTP GET requests to Neo4j AuraDB API to check instance status.
    Will retry up to 30 times with 20 second intervals between attempts.
    Breaks loop when instance is either 'running' or 'failed' and prints the status.

    Returns:
        None
    """
    url = f"{BASE_AURA_URL}/instances/{INSTANCE_ID}"
    headers = {"Accept": "application/json", "Authorization": f"Bearer {ACCESS_TOKEN}"}

    iterations = 0
    while iterations < 30:
        response = requests.get(url, headers=headers, timeout=10)
        data = response.json()["data"]
        status = data["status"]
        if status == "running":
            print("Instance is ready")
            break
        elif status == "failed":
            print("Instance failed to start")
            break
        print(
            f"Waiting for instance to be ready... (this may take a few minutes). Current status: {status}"
        )
        iterations += 1
        time.sleep(20)


def create_folder_structure() -> None:
    """Creates the initial folder structure and files for a FastAPI application with neomodel integration.

    This function sets up the following structure:
    - Root directory (app/)
        - Models directory (for Neo4j models) (models/)
        - Routers directory (for FastAPI routers) (routers/)
        - Main application file (main.py)
    - Requirements file (requirements.txt) with necessary dependencies
    - Environment files (.env and .env.example)
    - Git ignore file (.gitignore)

    The function also initializes basic content in:
    - main.py with FastAPI and Neo4j configuration
    - Empty model and router initialization files
    - Environment files with Neo4j connection details

    Note:
        All directories are created with exist_ok=True to prevent errors if they already exist

    Returns:
        None
    """
    global INSTANCE_URI
    os.makedirs(ROOT_DIRECTORY, exist_ok=True)
    os.makedirs(MODELS_DIRECTORY, exist_ok=True)
    os.makedirs(ROUTERS_DIRECTORY, exist_ok=True)
    with open("requirements.txt", "w", encoding="utf-8") as f:
        f.write("fastapi\nuvicorn[standard]\nneomodel\nrequests\npython-dotenv\n")

    # Create main.py
    main_content = """\
from fastapi import FastAPI
from neomodel import config
import os
from dotenv import load_dotenv
from app import routers

load_dotenv()

app = FastAPI()

config.DATABASE_URL = os.getenv("NEO4J_URI")

# Include routers dynamically later
"""
    with open(MAIN_FILE, "w", encoding="utf-8") as f:
        f.write(main_content)

    # Create .gitignore
    with open(".gitignore", "w", encoding="utf-8") as f:
        f.write("venv\n__pycache__\n*.pyc\n*.pyo\n*.pyd\n*.log\n.DS_Store\n.env\n")
    # Create secret .env and .env.example
    with open(".env", "w", encoding="utf-8") as f:
        if INSTANCE_URI is None:
            INSTANCE_URI = "<protocol>://<hostname>:<port>"
        protocol = INSTANCE_URI.split("://", maxsplit=1)[0]
        uri = INSTANCE_URI.split("://", maxsplit=1)[1]
        neomodel_uri = f"{protocol}://{INSTANCE_USERNAME}:{INSTANCE_PASSWORD}@{uri}"
        f.write(f"NEO4J_URI={neomodel_uri}")
    with open(".env.example", "w", encoding="utf-8") as f:
        f.write("NEO4J_URI=<protocol>://<username>:<password>@<hostname>:<port>")

    # create models structure
    with open(f"{MODELS_DIRECTORY}/__init__.py", "w", encoding="utf-8") as f:
        f.write("# Package init")
    with open(MODELS_FILE, "w", encoding="utf-8") as f:
        f.write("# Models will be generated here\n")

    # create routers structure
    with open(f"{ROUTERS_DIRECTORY}/__init__.py", "w", encoding="utf-8") as f:
        f.write("# Routers import\n")


def generate_models_from_workspace_json(model_path: str) -> list[dict[str, Any]]:
    """Convert a workspace JSON schema into Python Neomodel classes.

    This function takes a JSON schema file path and generates corresponding Python classes
    for Neo4j nodes and relationships using the neomodel ORM framework.

    Args:
        model_path (str): Path to the workspace export JSON schema file

    Returns:
        list[dict[str, Any]]: List of node label definitions from the schema

    The function performs the following:
    1. Reads and parses the JSON schema file
    2. Extracts node labels, relationship types, constraints and indexes
    3. Maps property types to Neomodel property classes
    4. Generates StructuredRel classes for relationships with properties
    5. Generates StructuredNode classes with:
       - Properties (with uniqueness/index constraints)
       - Relationships to other nodes
       - to_dict() method for serialization
    6. Writes generated code to models.py

    The generated classes follow these conventions:
    - Node class names are PascalCase versions of schema labels
    - Relationship classes are named [RelationName]Rel
    - Properties maintain their original names from the schema
    - Relationships are lowercase versions of schema relation types

    Example output file structure:
        from neomodel import ...

        class SomeRelation(StructuredRel):
            prop = StringProperty()

        class NodeA(StructuredNode):
            name = StringProperty(unique_index=True)
            some_relation = RelationshipTo('NodeB', 'SOME_RELATION')
    """
    with open(model_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    graph_schema = data["dataModel"]["graphSchemaRepresentation"]["graphSchema"]

    node_labels = graph_schema.get("nodeLabels", [])  # node label definitions
    rel_types = graph_schema.get(
        "relationshipTypes", []
    )  # relationship type definitions
    node_objects = graph_schema.get("nodeObjectTypes", [])  # node instances mapping
    rel_objects = graph_schema.get(
        "relationshipObjectTypes", []
    )  # relationship instances mapping
    constraints = graph_schema.get("constraints", [])
    indexes = graph_schema.get("indexes", [])

    # Map $id to relationship type
    rel_type_by_id = {rt["$id"]: rt for rt in rel_types}

    # Extract uniqueness constraints
    # node_label_id -> set of property_ids that are unique
    unique_props: dict[str, set] = {}
    for c in constraints:
        if c["constraintType"] == "uniqueness" and c["entityType"] == "node":
            nl_id = c["nodeLabel"]["$ref"].strip("#")
            prop_ids = [p["$ref"].strip("#") for p in c["properties"]]
            unique_props.setdefault(nl_id, set()).update(prop_ids)

    # Extract indexed properties
    # node_label_id -> set of property_ids that are indexed
    indexed_props: dict[str, set] = {}
    for i in indexes:
        if i["entityType"] == "node":
            nl_id = i["nodeLabel"]["$ref"].strip("#")
            prop_ids = [p["$ref"].strip("#") for p in i["properties"]]
            indexed_props.setdefault(nl_id, set()).update(prop_ids)

    # Map property types to neomodel classes
    type_map = {
        "string": "StringProperty",
        "integer": "IntegerProperty",
        "float": "FloatProperty",
        "boolean": "BooleanProperty",
        "datetime": "DateTimeProperty",
    }

    def to_class_name(name):
        # Converts tokens like "NodeA" or "HAS_SOME_RELATION" to a Pythonic class name
        # We'll just capitalize words and remove non-alphanumeric chars.
        # Already "NodeA" is fine, but "HAS_SOME_RELATION" => "HasSomeRelation"
        if not re.search(r"[\W_]", name):
            return name
        parts = re.split(r"[\W_]+", name)
        return "".join(part.capitalize() for part in parts if part)

    # Build prop_id_map for quick lookup of properties
    prop_id_map = {}
    for nl in node_labels:
        for p in nl["properties"]:
            prop_id_map[p["$id"]] = p

    # Map nodeObjectType to nodeLabelId
    node_to_label = {}
    for no in node_objects:
        # assume single label
        nl_ref = no["labels"][0]["$ref"].strip("#")
        node_to_label[no["$id"]] = nl_ref

    # Handle relationships: we need to create relationship classes if they have properties.
    # Also map from a node label to its outgoing relationships.
    # relationships_by_node: {node_label_id: [(rel_token, target_label_id, rel_properties_classname)]}
    relationships_by_node: dict[str, list[tuple]] = {}
    # We'll also store relationship types that have properties to generate StructuredRel classes.
    rel_types_with_props: dict[str, tuple] = {}
    for ro in rel_objects:
        rt_id = ro["type"]["$ref"].strip("#")
        from_n = ro["from"]["$ref"].strip("#")
        to_n = ro["to"]["$ref"].strip("#")

        rel_info = rel_type_by_id[rt_id]
        rel_token = rel_info["token"]
        rel_class_name = None
        rel_props = rel_info["properties"]

        if rel_props:
            # We have relationship properties, we'll create a StructuredRel class
            rel_class_name = to_class_name(rel_token) + "Rel"
            rel_types_with_props[rt_id] = (rel_class_name, rel_props)

        from_label_id = node_to_label[from_n]
        to_label_id = node_to_label[to_n]

        relationships_by_node.setdefault(from_label_id, []).append(
            (rel_token, to_label_id, rt_id)
        )

    # Prepare code generation
    model_code = []
    model_code.append(
        "from neomodel import StructuredNode, StructuredRel, StringProperty, IntegerProperty, FloatProperty, BooleanProperty, DateTimeProperty, UniqueIdProperty, RelationshipTo"
    )
    model_code.append("")
    model_code.append("# Generated Models")
    model_code.append("")

    # Generate StructuredRel classes for relationship types with properties
    for rt_id, (rel_class_name, rel_props) in rel_types_with_props.items():
        model_code.append(f"class {rel_class_name}(StructuredRel):")
        if not rel_props:
            model_code.append("    pass")
        else:
            for pdef in rel_props:
                p_id = pdef["$id"]
                p_name = pdef["token"]
                p_type = pdef["type"]["type"]
                p_nullable = pdef["nullable"]
                prop_class = type_map.get(p_type, "StringProperty")
                kwargs = []
                # For relationship properties, required=True if nullable=False
                if not p_nullable:
                    kwargs.append("required=True")
                kwargs_str = ""
                if kwargs:
                    kwargs_str = "(" + ", ".join(kwargs) + ")"
                model_code.append(f"    {p_name} = {prop_class}{kwargs_str}")
        model_code.append("")

    # label_id_to_class map
    label_id_to_class = {}
    for nl in node_labels:
        class_name = to_class_name(nl["token"])
        label_id_to_class[nl["$id"]] = class_name

    # Generate Node Classes
    for nl in node_labels:
        class_name = label_id_to_class[nl["$id"]]
        props = nl["properties"]
        nl_id = nl["$id"]

        model_code.append(f"class {class_name}(StructuredNode):")
        node_unique = unique_props.get(nl_id, set())
        node_indexed = indexed_props.get(nl_id, set())

        # Add a fallback uid if there's no unique property
        if not props:
            # no properties, check if no relationships too
            if nl_id not in relationships_by_node:
                model_code.append("    pass")
                model_code.append("")
                continue
            else:
                # If no properties but we have relationships, add a default UniqueIdProperty
                model_code.append("    uid = UniqueIdProperty()")

        else:
            # If no explicit unique property, add a fallback unique id
            if not node_unique:
                model_code.append("    uid = UniqueIdProperty()")

            # Add properties
            for pdef in props:
                p_id = pdef["$id"]
                p_name = pdef["token"]
                p_type = pdef["type"]["type"]
                p_nullable = pdef["nullable"]

                prop_class = type_map.get(p_type, "StringProperty")
                kwargs = []
                # unique if property is in node_unique
                if p_id in node_unique:
                    kwargs.append("unique_index=True")
                else:
                    # If indexed but not unique
                    if p_id in node_indexed:
                        kwargs.append("index=True")

                # For node properties, required=True if nullable=False
                if not p_nullable:
                    kwargs.append("required=True")

                kwargs_str = ""
                if kwargs:
                    kwargs_str = "(" + ", ".join(kwargs) + ")"

                model_code.append(f"    {p_name} = {prop_class}{kwargs_str}")

        # Add relationships
        rels = relationships_by_node.get(nl_id, [])
        if rels:
            for rel_token, target_label_id, rt_id in rels:
                target_class = label_id_to_class[target_label_id]
                # Check if we have a StructuredRel class for this relationship type
                rel_class_entry = rel_types_with_props.get(rt_id)
                if rel_class_entry:
                    rel_class_name, _ = rel_class_entry
                    # Use model=rel_class_name
                    model_code.append(
                        f"    {rel_token.lower()} = RelationshipTo('{target_class}', '{rel_token}', model={rel_class_name})"
                    )
                else:
                    # No properties, no model
                    model_code.append(
                        f"    {rel_token.lower()} = RelationshipTo('{target_class}', '{rel_token}')"
                    )

        # Add a to_dict method - will be used by the routers
        model_code.append("")
        model_code.append("    def to_dict(self):")
        model_code.append("        props = {}")
        model_code.append("        for prop_name, _ in self.__all_properties__:")
        model_code.append("            props[prop_name] = getattr(self, prop_name)")
        model_code.append("        return props")
        model_code.append("")

    with open(MODELS_FILE, "a", encoding="utf-8") as f:
        f.write("\n".join(model_code))

    print("Successfully generated models from JSON")
    print(f"Models are saved in {MODELS_FILE}")
    return node_labels


def generate_crud_endpoints(node_labels: list[dict[str, Any]]) -> None:
    """
    Generates CRUD (Create, Read, Update, Delete) endpoint files for each node label.

    This function creates separate router files for each node label with standard CRUD operations
    using FastAPI. For each node label, it generates endpoints for:
    - Creating new nodes
    - Reading all nodes
    - Reading a single node by UID
    - Updating (PUT) a node by UID
    - Deleting a node by UID

    The function also updates the main router initialization files.

    Args:
        node_labels (list[dict[str, Any]]): A list of dictionaries containing node label information.
                                           Each dictionary must have a "token" key with the class name.

    Generated Files:
        - Individual router files in the ROUTERS_DIRECTORY
        - Updated __init__.py in ROUTERS_DIRECTORY with router imports
        - Updated main.py with router inclusions

    Example:
        For a node label {"token": "Person"}, generates:
        - routers/person.py with CRUD endpoints at /people/
        - Updates routers/__init__.py and main.py with router registration

    Note:
        Class names are automatically converted to lowercase with underscores and pluralized
        for endpoint URLs (e.g., "PersonAddress" becomes "/person_addresses/").
    """
    routers = []
    for el in node_labels:
        class_name = el["token"]

        # Convert class name to lowercase with underscores (e.g. MyLabel -> my_label)
        underscored = re.sub(r"(?<!^)(?=[A-Z])", "_", class_name).lower()
        # Then pluralize the underscored name
        plural = p.plural(underscored)
        filename = f"{ROUTERS_DIRECTORY}/{underscored}.py"
        routers.append(underscored)

        router_code = [
            f'"""{class_name} router."""\n',
            "",
            "from fastapi import APIRouter, HTTPException",
            "from app.models.models import " + class_name,
            "",
            f"router = APIRouter(prefix='/{plural}', tags=['{class_name}'])",
            "",
            "# Create",
            "@router.post('/')",
            f"def create_{underscored}(payload: dict):",
            f"    obj = {class_name}(**payload).save()",
            "    return obj",
            "",
            "# Read all",
            "@router.get('/')",
            f"def list_{plural}():",
            f"    return [{underscored}.to_dict() for {underscored} in {class_name}.nodes.all()]",
            "",
            "# Read one",
            "@router.get('/{uid}')",
            f"def get_{underscored}(uid: str):",
            f"    obj = {class_name}.nodes.get_or_none(uid=uid)",
            "    if not obj:",
            "        raise HTTPException(status_code=404, detail='Not found')",
            "    return obj.to_dict()",
            "",
            "# Update",
            "@router.put('/{uid}')",
            f"def update_{underscored}(uid: str, payload: dict):",
            f"    obj = {class_name}.nodes.get_or_none(uid=uid)",
            "    if not obj:",
            "        raise HTTPException(status_code=404, detail='Not found')",
            "    for k, v in payload.items():",
            "        setattr(obj, k, v)",
            "    obj.save()",
            "    return obj.to_dict()",
            "",
            "# Delete",
            "@router.delete('/{uid}')",
            f"def delete_{underscored}(uid: str):",
            f"    obj = {class_name}.nodes.get_or_none(uid=uid)",
            "    if not obj:",
            "        raise HTTPException(status_code=404, detail='Not found')",
            "    obj.delete()",
            "    return {'detail': 'Deleted'}",
        ]

        with open(filename, "w", encoding="utf-8") as f:
            f.write("\n".join(router_code))

    print("Successfully generated routers for your models.")
    print(f"Routers are saved in {ROUTERS_DIRECTORY}")

    with open(f"{ROUTERS_DIRECTORY}/__init__.py", "w", encoding="utf-8") as f:
        for router_name in routers:
            f.write(
                f"from app.routers.{router_name} import router as {router_name}_router\n"
            )

    with open(MAIN_FILE, "a", encoding="utf-8") as f:
        for router_name in routers:
            f.write(f"app.include_router(routers.{router_name}_router)\n")

    print("Routers have been included in the main application file in app/main.py.")


def delete_neo4j_aura_instance(
    client_id: str, client_secret: str, instance_name: str
) -> None:
    """Delete a Neo4j Aura instance using the Neo4j Aura API.

    This function handles the deletion of a Neo4j Aura instance by making API calls to Neo4j's
    Aura service. It deletes the specified database instance using the instance ID.

    Args:
        client_id (str): The OAuth client ID for authentication with Neo4j Aura API
        client_secret (str): The OAuth client secret for authentication with Neo4j Aura API
        instance_name (str): The name of the database instance to delete

    Returns:
        None

    Notes:
        - Instance deletion may take several minutes to complete
    """
    global ACCESS_TOKEN
    global INSTANCE_ID
    global INSTANCE_NAME

    # First, get authorization token
    get_oauth_token(client_id, client_secret)

    headers = {"Accept": "application/json", "Authorization": f"Bearer {ACCESS_TOKEN}"}

    # Get tenant ID
    set_tenant_id()

    # Get instance ID
    instance_response = requests.get(
        f"{BASE_AURA_URL}/instances?tenantId={TENANT_ID}", headers=headers, timeout=10
    )
    instances = instance_response.json()["data"]
    for instance in instances:
        if instance["name"] == instance_name:
            INSTANCE_ID = instance["id"]
            break

    if INSTANCE_ID is None:
        print(f"Instance with name {instance_name} not found.")
        return

    # Delete instance
    response = requests.delete(
        f"{BASE_AURA_URL}/instances/{INSTANCE_ID}", headers=headers, timeout=60
    )

    if response.status_code == 202:
        print(f"Instance {instance_name} with ID {INSTANCE_ID} is being deleted.")
        INSTANCE_ID = None
    else:
        print("Failed to delete instance:", response.text)


def main():
    """
    Setup Neo4j Aura Free instance and scaffold a Python project.

    This function handles two main workflows:
    1. Creating a new Neo4j Aura instance and scaffolding project directories
    2. Generating data models from an imported Neo4j Workspace JSON file

    Command line arguments:
        -i, --api-client-id: Aura API client ID for authentication
        -s, --api-client-secret: Aura API client secret for authentication
        -n, --instance-name: Name for the Neo4j Aura instance (defaults to "my-instance")
        -m, --import-model: Path to the model.json file exported from Neo4j Workspace

    The function either:
    - Creates a new Aura instance using provided credentials and sets up project structure
    - Generates Python models and CRUD endpoints from an imported Workspace JSON model

    Global variables modified:
        INSTANCE_ID: ID of the created Aura instance
        INSTANCE_NAME: Name of the created Aura instance

    Returns:
        None
    """
    global INSTANCE_ID
    global INSTANCE_NAME
    parser = argparse.ArgumentParser(
        description="Setup Neo4j Aura Free instance and scaffold project."
    )
    parser.add_argument("-i", "--api-client-id", required=False, help="Aura API ID")
    parser.add_argument(
        "-s", "--api-client-secret", required=False, help="Aura API Key"
    )
    parser.add_argument(
        "-n",
        "--instance-name",
        required=False,
        help="Name of the Aura instance to create",
    )
    parser.add_argument(
        "-m", "--import-model", required=False, help="Path to model.json"
    )
    parser.add_argument(
        "-D",
        "--delete-instance",
        action="store_true",
        help="Delete the Aura instance. Requires --instance-name, --api-client-id, --api-client-secret",
    )

    args = parser.parse_args()

    if args.delete_instance:
        if (
            not args.instance_name
            or not args.api_client_id
            or not args.api_client_secret
        ):
            print(
                "Please provide the instance's name, as well as your API client id and secret."
            )
            return
        delete_neo4j_aura_instance(
            args.api_client_id, args.api_client_secret, args.instance_name
        )

    elif args.api_client_id and args.api_client_secret:
        # create aura instance, scaffold directories, etc.
        INSTANCE_NAME = args.instance_name or "my-instance"
        create_neo4j_aura_instance(
            args.api_client_id,
            args.api_client_secret,
        )
        if INSTANCE_ID is None:
            return
        create_folder_structure()
        wait_for_instance_ready()
        print("Your Neo4j Aura instance has been created!")
        print(
            "Please go to the Aura console and use the Import tool to define your schema:"
        )
        print("URL: https://workspace.neo4j.io/workspace/import")
        print("After importing your model, download it as JSON and save it locally.")
        print("Then run: create-neo4j-python-app --import-model path/to/model.json")

    elif args.import_model:
        # generate models from JSON
        node_labels = generate_models_from_workspace_json(args.import_model)
        generate_crud_endpoints(node_labels)


if __name__ == "__main__":
    main()
