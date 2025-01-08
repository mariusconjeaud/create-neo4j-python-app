# create-neo4j-python-app

This Python script streamlines setting up a new API project using a Neo4j Aura Free instance or importing existing data models from a Neo4j Workspace JSON file. The script provides two main functionalities: creating a new Neo4j Aura instance with necessary scaffolding and generating Python API code from imported models.

## Features

- **Neo4j Aura Instance Creation**: Automatically set up a new Neo4j Aura Free instance with the provided credentials.
- **Project Scaffolding**: Generate a basic project directory structure for your app.
- **Model Import and Generation**: Import data models from a Neo4j Importer diagram to generate Python models (using neomodel) and CRUD endpoints (using FastAPI).

## Usage

### Installation

First, install the script using PyPI :

```bash
pip install create-neo4j-python-app
```

### Command Line Arguments:

- `-i, --api-client-id`: Aura API client ID for authentication.
- `-s, --api-client-secret`: Aura API client secret for authentication.
- `-n, --instance-name`: Optional name for the Neo4j Aura instance (uses the default Aura name if not specified, e.g. `Instance01`).
- `-m, --import-model`: Path to the model.json file exported from Neo4j Workspace.

### Commands

1. **Create and Scaffold a New Neo4j Aura Instance**:

   ```bash
   create-neo4j-python-app -i YOUR_API_CLIENT_ID -s YOUR_API_CLIENT_SECRET -n my-instance
   ```

   This command will create a new Neo4j Aura instance named `my-instance` using the provided client ID and secret. It also sets up the necessary project directories.

   Note you need to [create a Neo4j Aura account](https://console.neo4j.io), and then [create an API Key](https://console-preview.neo4j.io/account/api-keys) to your account. This is what you can use for the above arguments.

2. **Generate Models from an existing Neo4j Import diagram**:

   ```bash
   create-neo4j-python-app -m path/to/model.json
   ```

   Use this command to generate Python models and CRUD operations from an existing JSON file exported from the Neo4j Workspace Import tool.

### Important Notes

- Before running `--import-model`, go to your Neo4j Aura instance console and use the Import tool to define your schema:
  - Go to the [Aura Workspace](https://workspace.neo4j.io/workspace/import)
  - Once your model is drawn in `Graph Models`, download the model as JSON.

## Run your new application

Once you have completed all the scaffolding steps, you can install requirements and run your API like this :

   ```bash
   # Ideally, create a virtual environment with :
   python3 -m venv venv
   source venv/bin/activate

   # Install requirements :
   pip install -r requirements.txt
   uvicorn --host=0.0.0.0 --port=8000 app.main:app --reload
   ```

Finally, navigate to http://localhost:8000/docs and you should see the Swagger documentation for your brand new API !

## Cleanup

You can delete the Aura instance by running the script like this :

   ```bash
   create-neo4j-python-app -i YOUR_API_CLIENT_ID -s YOUR_API_CLIENT_SECRET -n my-instance -D
   ```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Feel free to submit issues or pull requests for improvements and bug fixes. We welcome contributions from the community!

### Extending

To develop and test locally, just run :

```bash
pip install -e '.[dev]'
pytest
```

This repo has pre-commits for sorting imports and prettifying code :
```bash
pre-commit install
```

Enjoy building with Neo4j Aura! 🚀
