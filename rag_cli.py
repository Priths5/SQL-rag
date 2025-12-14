#!/usr/bin/env python3
"""
Industrial Machine RAG Assistant - Command Line Interface
A powerful CLI for interacting with the RAG-based database assistant
"""

import requests
import json
import sys
from typing import Optional, List, Dict
from tabulate import tabulate
import argparse
from colorama import init, Fore, Back, Style
import time

# Initialize colorama for cross-platform colored output
init(autoreset=True)

class RAGClient:
    def __init__(self, base_url: str = "http://localhost:8001"):
        self.base_url = base_url
        self.current_machine = None
        
    def check_connection(self) -> bool:
        """Check if RAG service is available"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def list_machines(self) -> List[Dict]:
        """List all running database machines"""
        try:
            response = requests.get(f"{self.base_url}/machines")
            response.raise_for_status()
            return response.json()['machines']
        except Exception as e:
            print(f"{Fore.RED}Error listing machines: {e}")
            return []
    
    def get_schema(self, machine_id: str, table_name: Optional[str] = None) -> Dict:
        """Get database schema"""
        try:
            url = f"{self.base_url}/machines/{machine_id}/schema"
            if table_name:
                url += f"?table_name={table_name}"
            response = requests.get(url)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"{Fore.RED}Error getting schema: {e}")
            return {}
    
    def index_schema(self, machine_id: str) -> Dict:
        """Index schema for RAG"""
        try:
            response = requests.post(f"{self.base_url}/machines/{machine_id}/index-schema")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"{Fore.RED}Error indexing schema: {e}")
            return {}
    
    def query(self, machine_id: str, question: str, execute: bool = False) -> Dict:
        """Ask a natural language question"""
        try:
            response = requests.post(
                f"{self.base_url}/query",
                json={
                    "machine_id": machine_id,
                    "question": question,
                    "execute": execute
                }
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"{Fore.RED}Error querying: {e}")
            return {}
    
    def execute_sql(self, machine_id: str, query: str) -> Dict:
        """Execute SQL directly"""
        try:
            # Ensure the query starts with SELECT or other valid SQL keywords
            query = query.strip()
            if not query.lower().startswith(('select', 'insert', 'update', 'delete')):
                raise ValueError(f"Invalid SQL query: {query}")
            
            response = requests.post(
                f"{self.base_url}/execute",
                params={"machine_id": machine_id, "query": query}
            )
            response.raise_for_status()
            return response.json()
        except ValueError as ve:
            print(f"{Fore.RED}Error: {ve}")
            return {}
        except Exception as e:
            print(f"{Fore.RED}Error executing SQL: {e}")
            return {}
    
    def modify_query(self, machine_id: str, original_query: str, instruction: str) -> Dict:
        """Modify an existing query"""
        try:
            response = requests.post(
                f"{self.base_url}/modify-query",
                json={
                    "machine_id": machine_id,
                    "original_query": original_query,
                    "modification_instruction": instruction
                }
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"{Fore.RED}Error modifying query: {e}")
            return {}


def print_header(text: str):
    """Print a formatted header"""
    print(f"\n{Fore.CYAN}{Style.BRIGHT}{'='*70}")
    print(f"{Fore.CYAN}{Style.BRIGHT}{text.center(70)}")
    print(f"{Fore.CYAN}{Style.BRIGHT}{'='*70}\n")


def print_machines(machines: List[Dict]):
    """Print machines in a formatted table"""
    if not machines:
        print(f"{Fore.YELLOW}No machines found")
        return
    
    table_data = []
    for machine in machines:
        table_data.append([
            machine['machine_id'],
            machine['database'],
            f"{machine['host']}:{machine['port']}",
            machine['status']
        ])
    
    print(tabulate(
        table_data,
        headers=['Machine ID', 'Database', 'Host:Port', 'Status'],
        tablefmt='grid'
    ))


def print_schema(schema: Dict):
    """Print database schema in a formatted way"""
    if not schema or 'tables' not in schema:
        print(f"{Fore.YELLOW}No schema data available")
        return
    
    print(f"{Fore.GREEN}Database: {Style.BRIGHT}{schema['database']}\n")
    
    for table_name, table_info in schema['tables'].items():
        print(f"{Fore.YELLOW}{Style.BRIGHT}📁 Table: {table_name} {Fore.WHITE}({table_info['row_count']:,} rows)")
        
        # Print columns
        column_data = []
        for col in table_info['columns']:
            key_indicator = ''
            if col['key'] == 'PRI':
                key_indicator = f"{Fore.RED}🔑"
            elif col['key'] == 'MUL':
                key_indicator = f"{Fore.BLUE}🔗"
            
            column_data.append([
                f"{key_indicator} {col['name']}",
                col['type'],
                'YES' if col['null'] == 'YES' else 'NO',
                col['default'] if col['default'] else '-',
                col['extra'] if col['extra'] else '-'
            ])
        
        print(tabulate(
            column_data,
            headers=['Column', 'Type', 'Null', 'Default', 'Extra'],
            tablefmt='simple'
        ))
        print()


def print_query_result(result: Dict):
    """Print query results in a formatted way"""
    if not result:
        return
    
    # Print generated query
    if 'generated_query' in result:
        print(f"{Fore.CYAN}{Style.BRIGHT}Generated SQL Query:")
        print(f"{Fore.WHITE}{Back.BLACK}{result['generated_query']}{Style.RESET_ALL}\n")
    
    # Print explanation
    if 'explanation' in result:
        print(f"{Fore.GREEN}{Style.BRIGHT}Explanation:")
        print(f"{Fore.WHITE}{result['explanation']}\n")
    
    # Print execution results
    if 'execution_result' in result:
        exec_result = result['execution_result']
        
        if exec_result.get('success'):
            if 'columns' in exec_result and 'rows' in exec_result:
                print(f"{Fore.GREEN}{Style.BRIGHT}Results:")
                
                if exec_result['rows']:
                    print(tabulate(
                        exec_result['rows'],
                        headers=exec_result['columns'],
                        tablefmt='grid'
                    ))
                    print(f"\n{Fore.CYAN}Total rows: {exec_result['row_count']}")
                else:
                    print(f"{Fore.YELLOW}No results found")
            
            elif 'message' in exec_result:
                print(f"{Fore.GREEN}✓ {exec_result['message']}")
        else:
            print(f"{Fore.RED}✗ Error: {exec_result.get('error', 'Unknown error')}")


def interactive_mode(client: RAGClient):
    """Run in interactive mode"""
    print_header("🤖 Industrial Machine RAG Assistant - Interactive Mode")
    print(f"{Fore.CYAN}Type 'help' for available commands, 'exit' to quit\n")
    
    # Select machine
    machines = client.list_machines()
    if not machines:
        print(f"{Fore.RED}No machines available. Please start some database machines first.")
        return
    
    print(f"{Fore.GREEN}Available machines:")
    print_machines(machines)
    
    while True:
        machine_choice = input(f"\n{Fore.CYAN}Select machine (enter machine_id): {Fore.WHITE}").strip()
        if machine_choice in [m['machine_id'] for m in machines]:
            client.current_machine = machine_choice
            break
        print(f"{Fore.RED}Invalid machine ID. Please try again.")
    
    print(f"\n{Fore.GREEN}✓ Selected machine: {Style.BRIGHT}{client.current_machine}\n")
    
    # Main interactive loop
    while True:
        try:
            command = input(f"{Fore.YELLOW}rag> {Fore.WHITE}").strip()
            
            if not command:
                continue
            
            if command.lower() in ['exit', 'quit', 'q']:
                print(f"{Fore.CYAN}Goodbye! 👋")
                break
            
            elif command.lower() == 'help':
                print(f"""
{Fore.CYAN}{Style.BRIGHT}Available Commands:
{Fore.WHITE}
  machines              - List all running machines
  schema [table]        - Show database schema (optionally for specific table)
  index                 - Index schema for RAG
  ask <question>        - Ask a question (generates SQL)
  exec <question>       - Ask and execute query
  sql <query>           - Execute SQL directly
  modify <instruction>  - Modify the last generated query
  switch <machine_id>   - Switch to a different machine
  clear                 - Clear screen
  help                  - Show this help
  exit                  - Exit interactive mode
                """)
            
            elif command.lower() == 'machines':
                machines = client.list_machines()
                print_machines(machines)
            
            elif command.lower().startswith('schema'):
                parts = command.split(maxsplit=1)
                table_name = parts[1] if len(parts) > 1 else None
                schema = client.get_schema(client.current_machine, table_name)
                print_schema(schema)
            
            elif command.lower() == 'index':
                print(f"{Fore.YELLOW}Indexing schema... This may take a moment.")
                result = client.index_schema(client.current_machine)
                if result:
                    print(f"{Fore.GREEN}✓ {result.get('message', 'Indexing complete')}")
                    print(f"{Fore.CYAN}Documents stored: {result.get('documents_stored', 0)}")
            
            elif command.lower().startswith('ask '):
                question = command[4:].strip()
                print(f"{Fore.YELLOW}Generating query...")
                result = client.query(client.current_machine, question, execute=False)
                print_query_result(result)
                
                # Store for modify command
                if 'generated_query' in result:
                    client.last_query = result['generated_query']
            
            elif command.lower().startswith('exec '):
                question = command[5:].strip()
                print(f"{Fore.YELLOW}Generating query...")
                result = client.query(client.current_machine, question, execute=False)
                
                if 'generated_query' in result:
                    print(f"{Fore.CYAN}{Style.BRIGHT}RAG-Generated SQL Query:")
                    print(f"{Fore.WHITE}{Back.BLACK}{result['generated_query']}{Style.RESET_ALL}\n")
                    
                    # Ask the user whether to use the RAG-generated query or the provided question
                    choice = input(f"{Fore.YELLOW}Use RAG-generated query? (y/n): {Fore.WHITE}").strip().lower()
                    if choice == 'y':
                        print(f"{Fore.YELLOW}Executing RAG-generated query...")
                        exec_result = client.query(client.current_machine, question, execute=True)
                        print_query_result(exec_result)
                    else:
                        print(f"{Fore.YELLOW}Executing provided SQL query...")
                        exec_result = client.execute_sql(client.current_machine, question)
                        print_query_result({'execution_result': exec_result})
                else:
                    print(f"{Fore.RED}Failed to generate a query. Executing provided SQL query...")
                    exec_result = client.execute_sql(client.current_machine, question)
                    print_query_result({'execution_result': exec_result})
            
            elif command.lower().startswith('sql '):
                query = command[4:].strip()
                print(f"{Fore.YELLOW}Executing query...")
                result = client.execute_sql(client.current_machine, query)
                print_query_result({'execution_result': result})
            
            elif command.lower().startswith('modify '):
                if not hasattr(client, 'last_query'):
                    print(f"{Fore.RED}No previous query to modify. Generate a query first using 'ask' or 'exec'.")
                    continue
                
                instruction = command[7:].strip()
                print(f"{Fore.YELLOW}Modifying query...")
                result = client.modify_query(client.current_machine, client.last_query, instruction)
                
                if result and 'modified_query' in result:
                    print(f"{Fore.CYAN}{Style.BRIGHT}Original Query:")
                    print(f"{Fore.WHITE}{Back.BLACK}{result['original_query']}{Style.RESET_ALL}\n")
                    print(f"{Fore.CYAN}{Style.BRIGHT}Modified Query:")
                    print(f"{Fore.WHITE}{Back.BLACK}{result['modified_query']}{Style.RESET_ALL}\n")
                    
                    # Update last query
                    client.last_query = result['modified_query']
                    
                    # Ask if they want to execute
                    execute = input(f"{Fore.YELLOW}Execute modified query? (y/n): {Fore.WHITE}").strip().lower()
                    if execute == 'y':
                        exec_result = client.execute_sql(client.current_machine, result['modified_query'])
                        print_query_result({'execution_result': exec_result})
            
            elif command.lower().startswith('switch '):
                machine_id = command[7:].strip()
                machines = client.list_machines()
                if machine_id in [m['machine_id'] for m in machines]:
                    client.current_machine = machine_id
                    print(f"{Fore.GREEN}✓ Switched to machine: {Style.BRIGHT}{machine_id}")
                else:
                    print(f"{Fore.RED}Machine not found: {machine_id}")
            
            elif command.lower() == 'clear':
                import os
                os.system('cls' if os.name == 'nt' else 'clear')
            
            else:
                print(f"{Fore.RED}Unknown command. Type 'help' for available commands.")
        
        except KeyboardInterrupt:
            print(f"\n{Fore.CYAN}Use 'exit' to quit")
        except Exception as e:
            print(f"{Fore.RED}Error: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='Industrial Machine RAG Assistant CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python rag_cli.py -i
  
  # List machines
  python rag_cli.py --list-machines
  
  # Show schema
  python rag_cli.py -m mysql_db --schema
  
  # Ask a question
  python rag_cli.py -m mysql_db --ask "Show sensors needing maintenance"
  
  # Execute a question
  python rag_cli.py -m mysql_db --exec "What is the average temperature?"
  
  # Execute SQL directly
  python rag_cli.py -m mysql_db --sql "SELECT COUNT(*) FROM sensor_data"
  
  # Index schema
  python rag_cli.py -m mysql_db --index
        """
    )
    
    parser.add_argument('--url', default='http://localhost:8001', help='RAG service URL')
    parser.add_argument('-i', '--interactive', action='store_true', help='Start interactive mode')
    parser.add_argument('-m', '--machine', help='Machine ID to query')
    parser.add_argument('--list-machines', action='store_true', help='List all running machines')
    parser.add_argument('--schema', nargs='?', const='', help='Show database schema (optionally specify table)')
    parser.add_argument('--index', action='store_true', help='Index schema for RAG')
    parser.add_argument('--ask', help='Ask a natural language question')
    parser.add_argument('--exec', dest='execute_question', help='Ask and execute a question')
    parser.add_argument('--sql', help='Execute SQL query directly')
    parser.add_argument('--modify', nargs=2, metavar=('QUERY', 'INSTRUCTION'), 
                       help='Modify a query with an instruction')
    parser.add_argument('--json', action='store_true', help='Output results as JSON')
    
    args = parser.parse_args()
    
    # Initialize client
    client = RAGClient(args.url)
    
    # Check connection
    print(f"{Fore.CYAN}Connecting to RAG service at {args.url}...", end=' ')
    if not client.check_connection():
        print(f"{Fore.RED}✗ Failed")
        print(f"{Fore.RED}Make sure the RAG service is running at {args.url}")
        sys.exit(1)
    print(f"{Fore.GREEN}✓ Connected\n")
    
    # Interactive mode
    if args.interactive:
        interactive_mode(client)
        return
    
    # List machines
    if args.list_machines:
        print_header("Running Machines")
        machines = client.list_machines()
        
        if args.json:
            print(json.dumps(machines, indent=2))
        else:
            print_machines(machines)
        return
    
    # All other commands require a machine to be specified
    if not args.machine and not args.list_machines:
        print(f"{Fore.RED}Error: Machine ID required (-m/--machine)")
        print(f"{Fore.YELLOW}Use --list-machines to see available machines")
        print(f"{Fore.YELLOW}Or use -i for interactive mode")
        sys.exit(1)
    
    client.current_machine = args.machine
    
    # Show schema
    if args.schema is not None:
        print_header(f"Schema for {args.machine}")
        table_name = args.schema if args.schema else None
        schema = client.get_schema(args.machine, table_name)
        
        if args.json:
            print(json.dumps(schema, indent=2))
        else:
            print_schema(schema)
        return
    
    # Index schema
    if args.index:
        print_header(f"Indexing Schema for {args.machine}")
        print(f"{Fore.YELLOW}Indexing... This may take a moment.\n")
        result = client.index_schema(args.machine)
        
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            if result:
                print(f"{Fore.GREEN}✓ {result.get('message', 'Indexing complete')}")
                print(f"{Fore.CYAN}Documents stored: {result.get('documents_stored', 0)}")
        return
    
    # Ask question (without execution)
    if args.ask:
        print_header(f"Query for {args.machine}")
        result = client.query(args.machine, args.ask, execute=False)
        
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print_query_result(result)
        return
    
    # Execute question
    if args.execute_question:
        print_header(f"Execute Query for {args.machine}")
        result = client.query(args.machine, args.execute_question, execute=True)
        
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print_query_result(result)
        return
    
    # Execute SQL
    if args.sql:
        print_header(f"Execute SQL on {args.machine}")
        result = client.execute_sql(args.machine, args.sql)
        
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print_query_result({'execution_result': result})
        return
    
    # Modify query
    if args.modify:
        original_query, instruction = args.modify
        print_header(f"Modify Query for {args.machine}")
        result = client.modify_query(args.machine, original_query, instruction)
        
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            if result and 'modified_query' in result:
                print(f"{Fore.CYAN}{Style.BRIGHT}Original Query:")
                print(f"{Fore.WHITE}{Back.BLACK}{result['original_query']}{Style.RESET_ALL}\n")
                print(f"{Fore.CYAN}{Style.BRIGHT}Modified Query:")
                print(f"{Fore.WHITE}{Back.BLACK}{result['modified_query']}{Style.RESET_ALL}")
        return
    
    # If no specific command, show help
    parser.print_help()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{Fore.CYAN}Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"{Fore.RED}Error: {e}")
        sys.exit(1)