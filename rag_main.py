from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import docker
import pymysql
from sqlalchemy import create_engine, inspect, text
import logging
from datetime import datetime
import json
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import re
import subprocess

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Industrial Machine RAG Database Assistant",
    description="RAG-based SQL assistant for multiple industrial machines",
    version="1.0.0"
)

# Pydantic Models
class QueryRequest(BaseModel):
    machine_id: str
    question: str
    execute: bool = False

class SQLModifyRequest(BaseModel):
    machine_id: str
    original_query: str
    modification_instruction: str

class SchemaQueryRequest(BaseModel):
    machine_id: str
    table_name: Optional[str] = None

# RAG Assistant Class
class IndustrialRAGAssistant:
    def __init__(self):
        # Initialize Docker client with error handling
        self.docker_client = None
        
        # First, check if we're running with sudo or have docker access
        import os
        import subprocess
        
        logger.debug("Checking Docker access...")
        try:
            # Try running docker ps as the current user
            result = subprocess.run(['docker', 'ps'], capture_output=True, text=True)
            if result.returncode == 0:
                logger.debug("Direct Docker access available")
            else:
                logger.debug(f"Docker ps failed: {result.stderr}")
        except Exception as e:
            logger.debug(f"Could not run docker ps: {e}")
        
        # Try different methods to connect to Docker
        connection_methods = [
            ('environment', lambda: docker.from_env()),
            ('default socket', lambda: docker.DockerClient(base_url='unix://var/run/docker.sock')),
            ('local socket', lambda: docker.DockerClient(base_url='unix:///var/run/docker.sock')),
            ('tcp', lambda: docker.DockerClient(base_url='tcp://localhost:2375'))
        ]
        
        for method_name, connector in connection_methods:
            try:
                logger.debug(f"Attempting to connect to Docker via {method_name}")
                client = connector()
                # Test the connection
                client.ping()
                self.docker_client = client
                logger.info(f"Successfully connected to Docker via {method_name}")
                break
            except Exception as e:
                logger.debug(f"Failed to connect via {method_name}: {str(e)}")
                continue
        
        if self.docker_client is None:
            logger.error("Failed to connect to Docker daemon using all available methods")
            # Try to get more information about the Docker socket
            import os
            import stat
            socket_path = '/var/run/docker.sock'
            if os.path.exists(socket_path):
                st = os.stat(socket_path)
                logger.debug(f"Docker socket permissions: {stat.filemode(st.st_mode)}")
                logger.debug(f"Docker socket owner: UID={st.st_uid}, GID={st.st_gid}")
            else:
                logger.debug("Docker socket not found at /var/run/docker.sock")
        
        # Initialize embedding model (lightweight)
        logger.info("Loading embedding model...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize ChromaDB for vector storage
        logger.info("Initializing ChromaDB...")
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        
        # Initialize LLM (using a small, efficient model)
        logger.info("Loading LLM model...")
        self.model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        
        # Check if CUDA is available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None
        )
        
        self.llm_pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=512,
            temperature=0.3,
            do_sample=True,
            top_p=0.95
        )
        
        # Cache for machine metadata
        self.machine_cache = {}
        
        logger.info("RAG Assistant initialized successfully")
    
    def get_running_machines(self) -> List[Dict]:
        """Detect all running MySQL containers dynamically"""
        if self.docker_client is None:
            logger.warning("Docker client is not available, trying direct command...")
            try:
                # Try using direct docker command
                result = subprocess.run(['docker', 'ps', '--format', '{{json .}}'], 
                                     capture_output=True, text=True)
                if result.returncode == 0:
                    containers = []
                    for line in result.stdout.strip().split('\n'):
                        if line:
                            container_info = json.loads(line)
                            if 'mysql' in container_info.get('Image', '').lower():
                                # Get container details
                                inspect = subprocess.run(
                                    ['docker', 'inspect', container_info['ID']],
                                    capture_output=True, text=True
                                )
                                if inspect.returncode == 0:
                                    details = json.loads(inspect.stdout)[0]
                                    env_dict = {}
                                    for env in details['Config']['Env']:
                                        if '=' in env:
                                            key, value = env.split('=', 1)
                                            env_dict[key] = value
                                    
                                    ports = details['NetworkSettings']['Ports']
                                    host_port = None
                                    if '3306/tcp' in ports and ports['3306/tcp']:
                                        host_port = ports['3306/tcp'][0]['HostPort']
                                    
                                    containers.append({
                                        'machine_id': details['Name'].lstrip('/'),
                                        'container_id': details['Id'][:12],
                                        'status': details['State']['Status'],
                                        'image': container_info['Image'],
                                        'database': env_dict.get('MYSQL_DATABASE', 'mysql'),
                                        'host': 'localhost',
                                        'port': int(host_port) if host_port else 3306,
                                        'user': env_dict.get('MYSQL_USER', 'root'),
                                        'root_password': env_dict.get('MYSQL_ROOT_PASSWORD', ''),
                                        'password': env_dict.get('MYSQL_PASSWORD', '')
                                    })
                    return containers
            except Exception as e:
                logger.error(f"Failed to get containers using direct command: {e}")
            return []
            
        try:
            logger.debug("Attempting to list Docker containers...")
            try:
                # Test Docker connection first
                self.docker_client.ping()
                logger.debug("Docker daemon is responsive")
            except Exception as e:
                logger.error(f"Docker daemon is not responsive: {str(e)}")
                return []
                
            containers = self.docker_client.containers.list()
            logger.info(f"Found {len(containers)} containers")
            
            for container in containers:
                logger.debug(f"Examining container: {container.name}")
                logger.debug(f"Container image: {container.image.tags[0] if container.image.tags else str(container.image)}")
                logger.debug(f"Container status: {container.status}")
                logger.debug(f"Container ports: {container.ports}")
            
            machines = []
            
            for container in containers:
                # Check if container has MySQL or is a database container
                image = container.image.tags[0] if container.image.tags else str(container.image)
                
                logger.info(f"Checking container: {container.name}, Image: {image}")
                if 'mysql' in image.lower() or 'mariadb' in image.lower():
                    # Extract connection info from container
                    env_vars = container.attrs['Config']['Env']
                    env_dict = {}
                    for env in env_vars:
                        if '=' in env:
                            key, value = env.split('=', 1)
                            env_dict[key] = value
                            logger.info(f"Found environment variable: {key}")
                    
                    # Get network info
                    networks = container.attrs['NetworkSettings']['Networks']
                    network_name = list(networks.keys())[0] if networks else None
                    
                    # Get port mappings
                    ports = container.attrs['NetworkSettings']['Ports']
                    host_port = None
                    if ports and '3306/tcp' in ports and ports['3306/tcp']:
                        host_port = ports['3306/tcp'][0]['HostPort']
                        logger.info(f"Found port mapping: 3306 -> {host_port}")
                    
                    # Extract all relevant environment variables
                    database = env_dict.get('MYSQL_DATABASE', 
                                         env_dict.get('MARIADB_DATABASE', 'mysql'))
                    
                    # Log environment variables (excluding passwords)
                    for key in ['MYSQL_DATABASE', 'MYSQL_USER', 'MARIADB_DATABASE', 'MARIADB_USER']:
                        if key in env_dict:
                            logger.info(f"Found {key}={env_dict[key]}")
                    
                    machine_info = {
                        'machine_id': container.name,
                        'container_id': container.id[:12],
                        'status': container.status,
                        'image': image,
                        'database': database,
                        'host': 'localhost',  # Always use localhost for port-mapped containers
                        'port': int(host_port) if host_port else 3306,
                        'user': env_dict.get('MYSQL_USER', env_dict.get('MARIADB_USER', 'root')),
                        'root_password': env_dict.get('MYSQL_ROOT_PASSWORD', 
                                                    env_dict.get('MARIADB_ROOT_PASSWORD', '')),
                        'password': env_dict.get('MYSQL_PASSWORD',
                                               env_dict.get('MARIADB_PASSWORD', '')),
                        'network': network_name,
                        'started_at': container.attrs['State']['StartedAt']
                    }
                    
                    machines.append(machine_info)
                    
                    # Update cache
                    self.machine_cache[container.name] = machine_info
            
            logger.info(f"Found {len(machines)} running database machines")
            return machines
            
        except Exception as e:
            logger.error(f"Error detecting machines: {e}")
            raise
    
    def get_db_connection(self, machine_id: str):
        """Get database connection for a specific machine"""
        if machine_id not in self.machine_cache:
            logger.info(f"Machine {machine_id} not in cache, refreshing machines...")
            self.get_running_machines()
        
        if machine_id not in self.machine_cache:
            available_machines = list(self.machine_cache.keys())
            raise ValueError(f"Machine {machine_id} not found or not running. Available machines: {available_machines}")
        
        machine = self.machine_cache[machine_id]
        
        logger.info(f"Attempting to connect to MySQL on {machine['host']}:{machine['port']}")
        logger.info(f"Connection details: database={machine['database']}, user={machine['user']}")
        
        try:
            # Try with root user first
            logger.info("Attempting connection with root user...")
            connection = pymysql.connect(
                host=machine['host'],
                port=machine['port'],
                user='root',
                password=machine['root_password'],
                database=machine['database'],
                connect_timeout=10
            )
            logger.info("Successfully connected with root user")
            return connection
        except Exception as e:
            logger.warning(f"Root connection failed: {str(e)}")
            # Try with app user
            try:
                logger.info("Attempting connection with app user...")
                connection = pymysql.connect(
                    host=machine['host'],
                    port=machine['port'],
                    user=machine['user'],
                    password=machine['password'],  # Use the correct app user password
                    database=machine['database'],
                    connect_timeout=10
                )
                logger.info(f"Successfully connected with app user to {machine['database']}")
                return connection
            except Exception as e2:
                logger.error(f"Failed to connect to {machine_id}: {e2}")
                raise
    
    def get_database_schema(self, machine_id: str, table_name: Optional[str] = None) -> Dict:
        """Extract complete database schema with metadata"""
        conn = self.get_db_connection(machine_id)
        cursor = conn.cursor()
        
        schema_info = {
            'database': self.machine_cache[machine_id]['database'],
            'tables': {}
        }
        
        try:
            # Get all tables or specific table
            if table_name:
                cursor.execute(f"SHOW TABLES LIKE '{table_name}'")
            else:
                cursor.execute("SHOW TABLES")
            
            tables = [row[0] for row in cursor.fetchall()]
            
            for table in tables:
                # Get column information
                cursor.execute(f"DESCRIBE {table}")
                columns = []
                for col in cursor.fetchall():
                    columns.append({
                        'name': col[0],
                        'type': col[1],
                        'null': col[2],
                        'key': col[3],
                        'default': col[4],
                        'extra': col[5]
                    })
                
                # Get row count
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                row_count = cursor.fetchone()[0]
                
                # Get sample data (first 3 rows)
                cursor.execute(f"SELECT * FROM {table} LIMIT 3")
                sample_rows = cursor.fetchall()
                
                # Get indexes
                cursor.execute(f"SHOW INDEXES FROM {table}")
                indexes = []
                for idx in cursor.fetchall():
                    indexes.append({
                        'name': idx[2],
                        'column': idx[4],
                        'unique': not idx[1]
                    })
                
                schema_info['tables'][table] = {
                    'columns': columns,
                    'row_count': row_count,
                    'sample_data': [list(row) for row in sample_rows[:3]],
                    'indexes': indexes
                }
            
            cursor.close()
            conn.close()
            
            return schema_info
            
        except Exception as e:
            cursor.close()
            conn.close()
            raise
    
    def store_schema_in_vectordb(self, machine_id: str):
        """Store schema information in vector database for RAG"""
        schema = self.get_database_schema(machine_id)
        
        # Create or get collection for this machine
        collection_name = f"machine_{machine_id.replace('-', '_')}"
        
        try:
            collection = self.chroma_client.get_collection(collection_name)
            # Delete existing collection to refresh
            self.chroma_client.delete_collection(collection_name)
        except:
            pass
        
        collection = self.chroma_client.create_collection(
            name=collection_name,
            metadata={"machine_id": machine_id}
        )
        
        # Prepare documents for embedding
        documents = []
        metadatas = []
        ids = []
        
        idx = 0
        for table_name, table_info in schema['tables'].items():
            # Table overview document
            table_doc = f"Table: {table_name}\n"
            table_doc += f"Row count: {table_info['row_count']}\n"
            table_doc += f"Columns: {', '.join([col['name'] for col in table_info['columns']])}\n"
            
            documents.append(table_doc)
            metadatas.append({
                'type': 'table_overview',
                'table': table_name,
                'machine_id': machine_id
            })
            ids.append(f"{machine_id}_{table_name}_{idx}")
            idx += 1
            
            # Column details document
            for col in table_info['columns']:
                col_doc = f"Table: {table_name}, Column: {col['name']}\n"
                col_doc += f"Type: {col['type']}, Nullable: {col['null']}, "
                col_doc += f"Key: {col['key']}, Default: {col['default']}\n"
                
                documents.append(col_doc)
                metadatas.append({
                    'type': 'column_detail',
                    'table': table_name,
                    'column': col['name'],
                    'machine_id': machine_id
                })
                ids.append(f"{machine_id}_{table_name}_{col['name']}_{idx}")
                idx += 1
            
            # Sample data document
            if table_info['sample_data']:
                sample_doc = f"Table: {table_name} - Sample Data:\n"
                col_names = [col['name'] for col in table_info['columns']]
                sample_doc += f"Columns: {', '.join(col_names)}\n"
                sample_doc += f"Sample rows: {table_info['sample_data']}\n"
                
                documents.append(sample_doc)
                metadatas.append({
                    'type': 'sample_data',
                    'table': table_name,
                    'machine_id': machine_id
                })
                ids.append(f"{machine_id}_{table_name}_sample_{idx}")
                idx += 1
        
        # Add documents to collection
        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        
        logger.info(f"Stored {len(documents)} schema documents for {machine_id}")
        return len(documents)
    
    def retrieve_relevant_context(self, machine_id: str, question: str, n_results: int = 5) -> str:
        """Retrieve relevant schema context using RAG"""
        collection_name = f"machine_{machine_id.replace('-', '_')}"
        
        try:
            collection = self.chroma_client.get_collection(collection_name)
        except:
            # If collection doesn't exist, create it
            logger.info(f"Collection not found for {machine_id}, creating...")
            self.store_schema_in_vectordb(machine_id)
            collection = self.chroma_client.get_collection(collection_name)
        
        # Query the collection
        results = collection.query(
            query_texts=[question],
            n_results=n_results
        )
        
        # Combine retrieved documents
        context = "\n\n".join(results['documents'][0])
        return context
    
    def generate_sql_query(self, machine_id: str, question: str) -> Dict:
        """Generate SQL query using RAG + LLM"""
        # Get relevant context
        context = self.retrieve_relevant_context(machine_id, question)
        
        # Get database name
        db_name = self.machine_cache[machine_id]['database']
        
        # Construct prompt
        prompt = f"""<|system|>
You are a SQL expert. Generate a MySQL query based on the user's question and database schema.
Ensure the query is syntactically correct and matches the schema. Only output the SQL query.
</|system|>

<|user|>
Database: {db_name}

Schema Context:
{context}

Question: {question}

Generate only the SQL query:
</|user|>

<|assistant|>
"""
        
        # Generate response
        response = self.llm_pipeline(
            prompt,
            max_new_tokens=256,
            temperature=0.1,
            do_sample=False,  # Disable sampling for deterministic output
            return_full_text=False
        )[0]['generated_text']
        
        # Extract and clean SQL query
        sql_query = response.strip()
        # Remove any unintended text (e.g., Markdown formatting or explanations)
        sql_query = re.sub(r'```.*?```', '', sql_query, flags=re.DOTALL).strip()
        sql_query = re.sub(r'[^;]*;', lambda m: m.group(0).strip(), sql_query)  # Ensure query ends with a semicolon
        sql_query = re.sub(r'\s+', ' ', sql_query)  # Normalize whitespace
        
        if not sql_query.lower().startswith(('select', 'insert', 'update', 'delete')):
            raise ValueError(f"Generated query is invalid: {sql_query}")
        
        return {
            'query': sql_query,
            'context_used': context,
            'explanation': self.explain_query(sql_query, context)
        }
    
    def explain_query(self, query: str, context: str) -> str:
        """Explain what a SQL query does in simple terms"""
        prompt = f"""<|system|>
You are a SQL expert. Explain the following SQL query in simple terms for someone with no SQL knowledge.
Focus on what the query does and what kind of data it retrieves.
</|system|>

<|user|>
Database Schema:
{context}

SQL Query:
{query}

Explain this query in simple terms:
</|user|>

<|assistant|>"""
        
        response = self.llm_pipeline(
            prompt,
            max_new_tokens=200,
            temperature=0.3,
            do_sample=True,
            return_full_text=False
        )[0]['generated_text']
        
        return response.strip()
    
    def execute_query(self, machine_id: str, query: str) -> Dict:
        """Execute SQL query and return results"""
        conn = self.get_db_connection(machine_id)
        cursor = conn.cursor()
        
        try:
            cursor.execute(query)
            
            # For SELECT queries
            if query.strip().upper().startswith('SELECT'):
                results = cursor.fetchall()
                columns = [desc[0] for desc in cursor.description]
                
                return {
                    'success': True,
                    'columns': columns,
                    'rows': [list(row) for row in results],
                    'row_count': len(results)
                }
            else:
                # For INSERT, UPDATE, DELETE
                conn.commit()
                return {
                    'success': True,
                    'affected_rows': cursor.rowcount,
                    'message': f"Query executed successfully. {cursor.rowcount} rows affected."
                }
                
        except Exception as e:
            conn.rollback()
            return {
                'success': False,
                'error': str(e)
            }
        finally:
            cursor.close()
            conn.close()
    
    def modify_query(self, machine_id: str, original_query: str, modification: str) -> str:
        """Modify an existing SQL query based on instructions"""
        context = self.retrieve_relevant_context(machine_id, modification)
        
        prompt = f"""<|system|>
You are a SQL expert. Modify the given SQL query according to the user's instruction.
Only output the modified SQL query, nothing else.
</|system|>

<|user|>
Database Schema:
{context}

Original Query:
{original_query}

Modification Requested:
{modification}

Generate the modified SQL query:
</|user|>

<|assistant|>"""
        
        response = self.llm_pipeline(
            prompt,
            max_new_tokens=256,
            temperature=0.1,
            do_sample=True,
            return_full_text=False
        )[0]['generated_text']
        
        # Clean up
        modified_query = response.strip()
        if ';' not in modified_query:
            modified_query += ';'
        
        return modified_query

# Initialize assistant
assistant = IndustrialRAGAssistant()

# API Endpoints
@app.get("/")
async def root():
    return {
        "message": "Industrial Machine RAG Database Assistant",
        "version": "1.0.0",
        "model": assistant.model_name,
        "device": assistant.device
    }

@app.get("/machines")
async def list_machines():
    """List all running database machines"""
    try:
        logger.debug("Endpoint /machines called")
        if assistant.docker_client is None:
            logger.error("Docker client is not initialized")
            return {
                "total_machines": 0,
                "machines": [],
                "error": "Docker client is not initialized"
            }
        
        machines = assistant.get_running_machines()
        logger.debug(f"Found {len(machines)} machines")
        return {
            "total_machines": len(machines),
            "machines": machines
        }
    except Exception as e:
        logger.exception("Error in list_machines endpoint")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/machines/{machine_id}/schema")
async def get_machine_schema(
    machine_id: str,
    table_name: Optional[str] = None
):
    """Get database schema for a specific machine"""
    try:
        schema = assistant.get_database_schema(machine_id, table_name)
        return schema
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/machines/{machine_id}/index-schema")
async def index_machine_schema(machine_id: str):
    """Index machine schema into vector database"""
    try:
        doc_count = assistant.store_schema_in_vectordb(machine_id)
        return {
            "message": f"Schema indexed successfully for {machine_id}",
            "documents_stored": doc_count
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
async def query_database(request: QueryRequest):
    """Ask a natural language question and get SQL query + results"""
    try:
        # Generate SQL query
        query_info = assistant.generate_sql_query(
            request.machine_id,
            request.question
        )
        
        result = {
            "machine_id": request.machine_id,
            "question": request.question,
            "generated_query": query_info['query'],
            "explanation": query_info['explanation'],
            "context_used": query_info['context_used']
        }
        
        # Execute if requested
        if request.execute:
            execution_result = assistant.execute_query(
                request.machine_id,
                query_info['query']
            )
            result['execution_result'] = execution_result
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/modify-query")
async def modify_sql_query(request: SQLModifyRequest):
    """Modify an existing SQL query"""
    try:
        modified_query = assistant.modify_query(
            request.machine_id,
            request.original_query,
            request.modification_instruction
        )
        
        return {
            "machine_id": request.machine_id,
            "original_query": request.original_query,
            "modification_instruction": request.modification_instruction,
            "modified_query": modified_query
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/execute")
async def execute_sql(machine_id: str, query: str):
    """Execute a SQL query directly"""
    try:
        result = assistant.execute_query(machine_id, query)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Check system health"""
    docker_status = False
    docker_error = None
    
    try:
        if assistant.docker_client is not None:
            docker_status = assistant.docker_client.ping()
    except Exception as e:
        docker_error = str(e)
        logger.error(f"Docker health check failed: {e}")

    return {
        "status": "healthy",
        "model_loaded": assistant.model is not None,
        "embedding_model_loaded": assistant.embedding_model is not None,
        "device": assistant.device,
        "docker": {
            "connected": docker_status,
            "error": docker_error,
            "client_initialized": assistant.docker_client is not None
        }
    }

if __name__ == "__main__":
    import uvicorn
    import os
    
    # Set logging to debug level
    logging.basicConfig(level=logging.DEBUG)
    logger.setLevel(logging.DEBUG)
    
    # Log Docker socket path
    docker_socket = os.environ.get('DOCKER_HOST', 'unix://var/run/docker.sock')
    logger.debug(f"Docker socket path: {docker_socket}")
    
    logger.info("Starting RAG Assistant on port 8001...")
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="debug")