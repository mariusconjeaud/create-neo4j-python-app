# create_app.py
import argparse
import inflect
import json
import os
import re
import time
from requests.auth import HTTPBasicAuth
import requests
from typing import Any

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

# Used to pluralize node names for the routers
p = inflect.engine()


def get_oauth_token(client_id: str, client_secret: str) -> None:
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


def create_neo4j_aura_instance(client_id: str, client_secret: str) -> None:
    global ACCESS_TOKEN
    global TENANT_ID
    global INSTANCE_ID
    global INSTANCE_NAME
    global INSTANCE_URI
    global INSTANCE_USERNAME
    global INSTANCE_PASSWORD
    # First, get authorization token
    get_oauth_token(client_id, client_secret)

    base_url = "https://api.neo4j.io/v1"
    headers = {"Accept": "application/json", "Authorization": f"Bearer {ACCESS_TOKEN}"}

    # Get tenant ID
    tenant_response = requests.get(f"{base_url}/tenants", headers=headers, timeout=10)
    TENANT_ID = tenant_response.json()["data"][0]["id"]

    # Create instance
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
        f"{base_url}/instances", json=payload, headers=headers, timeout=60
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


def wait_for_instance_ready():
    url = f"https://api.neo4j.io/v1/instances/{INSTANCE_ID}"
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


def create_folder_structure():
    os.makedirs(ROOT_DIRECTORY, exist_ok=True)
    os.makedirs(MODELS_DIRECTORY, exist_ok=True)
    os.makedirs(ROUTERS_DIRECTORY, exist_ok=True)
    with open("requirements.txt", "w") as f:
        f.write("fastapi\nuvicorn[standard]\nneomodel\nrequests\npython-dotenv\n")

    # Create main.py
    main_content = """\
        from fastapi import FastAPI
        from neomodel import config
        import os
        from dotenv import load_dotenv
        from routers import *

        load_dotenv()

        app = FastAPI()

        config.DATABASE_URL = os.getenv("NEO4J_URI")

        # Include routers dynamically later
        """
    with open(MAIN_FILE, "w") as f:
        f.write(main_content)

    # Create .gitignore
    with open(".gitignore", "w") as f:
        f.write("venv\n__pycache__\n*.pyc\n*.pyo\n*.pyd\n*.log\n.DS_Store\n.env\n")
    # Create secret .env and .env.example
    with open(".env", "w") as f:
        f.write(
            f"NEO4J_URI={INSTANCE_URI}\nNEO4J_USERNAME={INSTANCE_USERNAME}\nNEO4J_PASSWORD={INSTANCE_PASSWORD}\n"
        )
    with open(".env.example", "w") as f:
        f.write(
            "NEO4J_URI=<your-neo4j-uri>\nNEO4J_USERNAME=<your-neo4j-username>\nNEO4J_PASSWORD=<your-neo4j-password>\n"
        )

    # create models structure
    with open(f"{MODELS_DIRECTORY}/__init__.py", "w") as f:
        f.write("# Package init")
    with open(MODELS_FILE, "w") as f:
        f.write("# Models will be generated here\n")

    # create routers structure
    with open(f"{ROUTERS_DIRECTORY}/__init__.py", "w") as f:
        f.write("# Routers import\n")


def generate_models_from_workspace_json(model_path: str) -> list[dict[str, Any]]:
    with open(model_path, "r") as f:
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

    # Map $id to node label, relationship type, node object, and rel object
    node_label_by_id = {nl["$id"]: nl for nl in node_labels}
    rel_type_by_id = {rt["$id"]: rt for rt in rel_types}
    node_object_by_id = {no["$id"]: no for no in node_objects}
    rel_object_by_id = {ro["$id"]: ro for ro in rel_objects}

    # Extract uniqueness constraints
    # node_label_id -> set of property_ids that are unique
    unique_props = {}
    for c in constraints:
        if c["constraintType"] == "uniqueness" and c["entityType"] == "node":
            nl_id = c["nodeLabel"]["$ref"].strip("#")
            prop_ids = [p["$ref"].strip("#") for p in c["properties"]]
            unique_props.setdefault(nl_id, set()).update(prop_ids)

    # Extract indexed properties
    # node_label_id -> set of property_ids that are indexed
    indexed_props = {}
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
    relationships_by_node = {}
    # We'll also store relationship types that have properties to generate StructuredRel classes.
    rel_types_with_props = {}
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
        model_code.append("        for prop_name in self.__all_properties__:")
        model_code.append("            props[prop_name] = getattr(self, prop_name)")
        model_code.append("        return props")
        model_code.append("")

    with open("models.py", "w") as f:
        f.write("\n".join(model_code))

    print("Successfully generated models from JSON")
    print(f"Models are saved in {MODELS_FILE}")
    return node_labels


def generate_crud_endpoints(node_labels: list[dict[str, Any]]):
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
            "from fastapi import APIRouter, HTTPException",
            "from models import " + class_name,
            "from neomodel import db",
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
            f"@router.get('/{{uid}}')",
            f"def get_{underscored}(uid: str):",
            f"    obj = {class_name}.nodes.get_or_none(uid=uid)",
            "    if not obj:",
            "        raise HTTPException(status_code=404, detail='Not found')",
            "    return obj.to_dict()",
            "",
            "# Update",
            f"@router.put('/{{uid}}')",
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
            f"@router.delete('/{{uid}}')",
            f"def delete_{underscored}(uid: str):",
            f"    obj = {class_name}.nodes.get_or_none(uid=uid)",
            "    if not obj:",
            "        raise HTTPException(status_code=404, detail='Not found')",
            "    obj.delete()",
            "    return {'detail': 'Deleted'}",
        ]

        with open(filename, "w") as f:
            f.write("\n".join(router_code))

    with open(f"{ROUTERS_DIRECTORY}/__init__.py", "w") as f:
        for router_name in routers:
            f.write(
                f"from routers.{router_name} import router as {router_name}_router\n"
            )

    with open(MAIN_FILE, "w") as f:
        for router_name in routers:
            f.write(f"app.include_router(routers.{router_name}_router)\n")


def main():
    global INSTANCE_ID
    global INSTANCE_NAME
    parser = argparse.ArgumentParser(
        description="Setup Neo4j Aura Free instance and scaffold project."
    )
    parser.add_argument("--api-client-id", required=False, help="Aura API ID")
    parser.add_argument("--api-client-secret", required=False, help="Aura API Key")
    parser.add_argument(
        "--instance-name", required=False, help="Name of the Aura instance to create"
    )
    parser.add_argument("--import-model", required=False, help="Path to model.json")

    args = parser.parse_args()

    if args.api_client_id and args.api_client_secret:
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

    if args.import_model:
        # generate models from JSON
        node_labels = generate_models_from_workspace_json(args.import_model)
        generate_crud_endpoints(node_labels)

    # Additional logic as needed


if __name__ == "__main__":
    main()
