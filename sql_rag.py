#!/usr/bin/env python3
"""
SQL RAG - Natural Language Database Interface
Connects to Docker-based MySQL backends and provides AI-powered database interaction.
Schema is introspected live from the chosen backend — no hardcoded table definitions.
"""

import subprocess
import json
import re
import os
import sys
from typing import Optional
import pymysql
import requests

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

OLLAMA_HOST  = os.getenv("OLLAMA_HOST", "http://localhost:11434")
MISTRAL_MODEL = os.getenv("MISTRAL_MODEL", "mistral")

# ─────────────────────────────────────────────
# DYNAMIC SCHEMA BUILDER
# ─────────────────────────────────────────────

CLASSIFY_PROMPT = """
You are a query classifier for a database assistant.

Given the user's message, classify it as ONE of:
  - QUERY_READ   : User wants to view/explore/understand data or schema
  - QUERY_WRITE  : User wants to modify/insert/delete data
  - EXPLAIN      : User wants an explanation of schema, columns, or how the system works
  - UNCLEAR      : The intent is ambiguous or not database-related

Respond with ONLY the classification word, nothing else.
"""

SQL_GENERATION_SUFFIX = """

== YOUR TASK ==
The user will give you a request in plain English.
You must respond with a JSON object in this exact format:

{
  "sql": "<the SQL query>",
  "explanation": "<one sentence explaining what this query does in plain English>",
  "is_destructive": <true if it modifies/deletes data, false otherwise>,
  "warning": "<optional warning message if destructive, else null>"
}

Rules:
- sql must be a single valid MySQL query
- Do NOT wrap sql in markdown code blocks
- Do NOT include semicolons at the end
- For SELECT queries, always add LIMIT 100 unless user specifies otherwise
- Never SELECT columns whose type contains 'blob' or 'binary' — use LENGTH() on those instead
- Always output valid JSON only, no prose before or after
"""

EXPLAIN_SUFFIX = """

== YOUR TASK ==
The user has a question about the database schema, table design, or system purpose.
Explain clearly and helpfully in plain English. Be concise but thorough.
Use simple language — imagine explaining to a non-technical stakeholder.
Format your answer in readable paragraphs. You may use bullet points for lists of columns.
"""

RESULT_SUMMARIZE_SUFFIX = """

== YOUR TASK ==
The user ran a database query. Here are the results.
Summarize the results in plain English — highlight key insights, patterns, or notable values.
Keep it concise (3-5 sentences max unless data is complex).
If the result set is empty, say so clearly.
"""


def build_schema_context(db: "DatabaseConnection") -> str:
    """
    Introspect the live database and build a rich schema context string.
    Fetches: tables, columns (name/type/nullable/key/default), row counts,
    foreign key relationships, and a few sample values per column.
    """
    lines = []
    lines.append("You are an expert database assistant with deep knowledge of the following live database.")
    lines.append("")
    lines.append(f"== DATABASE: {db.database}  (connected to {db.host}:{db.port}) ==")
    lines.append("")

    try:
        schema = db.get_full_schema_details()
    except Exception as e:
        lines.append(f"(Could not introspect schema: {e})")
        return "\n".join(lines)

    blob_columns: list[str] = []

    for table, info in schema.items():
        row_count = info.get("row_count", "?")
        lines.append(f"TABLE: {table}  ({row_count} rows)")

        for col in info["columns"]:
            name = col["Field"]
            col_type = col["Type"]
            nullable = "nullable" if col.get("Null") == "YES" else "required"

            key = col.get("Key", "")
            key_str = ""
            if key == "PRI":
                key_str = " [PRIMARY KEY]"
            elif key == "MUL":
                key_str = " [INDEXED]"
            elif key == "UNI":
                key_str = " [UNIQUE]"

            default = col.get("Default")
            default_str = f", default={default}" if default is not None else ""

            samples = info["samples"].get(name, [])
            sample_str = ""
            if samples:
                sample_str = f"  — e.g. {', '.join(str(s) for s in samples[:3])}"

            if "blob" in col_type.lower() or "binary" in col_type.lower():
                blob_columns.append(f"{table}.{name}")
                sample_str = "  — ⚠️ BINARY/BLOB: do not SELECT directly, use LENGTH()"

            lines.append(
                f"  - {name:<35} {col_type:<22} {nullable}{key_str}{default_str}{sample_str}"
            )

        fks = info.get("foreign_keys", [])
        for fk in fks:
            lines.append(f"  → FOREIGN KEY: {fk['column']} → {fk['ref_table']}.{fk['ref_column']}")

        lines.append("")

    lines.append("== IMPORTANT RULES FOR SQL GENERATION ==")
    lines.append("- Always use safe practices: prefer SELECT before UPDATE/DELETE when exploring")
    lines.append("- For DELETE or UPDATE, always include a WHERE clause — never modify all rows blindly")
    lines.append("- Column names with spaces or reserved words should be backtick-quoted")
    lines.append("- Dates should be in 'YYYY-MM-DD HH:MM:SS' format")
    lines.append("- tinyint(1) boolean columns: use 1/0, not TRUE/FALSE, for compatibility")
    if blob_columns:
        lines.append(
            f"- These columns are BLOB/BINARY — NEVER SELECT them, use LENGTH() if size is needed: "
            + ", ".join(blob_columns)
        )

    return "\n".join(lines)


# ─────────────────────────────────────────────
# DOCKER DISCOVERY
# ─────────────────────────────────────────────

def discover_docker_backends() -> list[dict]:
    """Discover running MySQL containers via Docker."""
    try:
        result = subprocess.run(
            ["docker", "ps", "--format", "{{json .}}"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode != 0:
            return []

        containers = []
        for line in result.stdout.strip().split("\n"):
            if not line.strip():
                continue
            try:
                c = json.loads(line)
                image = c.get("Image", "").lower()
                name = c.get("Names", "")
                ports = c.get("Ports", "")

                if "mysql" in image:
                    host_port = None
                    for part in ports.split(","):
                        match = re.search(r":(\d+)->3306", part.strip())
                        if match:
                            host_port = int(match.group(1))
                            break

                    containers.append({
                        "container_name": name,
                        "image": c.get("Image"),
                        "host_port": host_port,
                        "status": c.get("Status"),
                        "raw_ports": ports,
                    })
            except json.JSONDecodeError:
                continue

        return containers

    except (subprocess.TimeoutExpired, FileNotFoundError):
        return []


def get_backend_config_from_compose(compose_path: str = "docker-compose.yml") -> Optional[dict]:
    """Read connection info from a docker-compose.yml file."""
    try:
        import yaml
        with open(compose_path) as f:
            data = yaml.safe_load(f)

        services = data.get("services", {})
        for svc_name, svc in services.items():
            image = svc.get("image", "")
            if "mysql" in image.lower():
                env = svc.get("environment", {})
                ports = svc.get("ports", [])
                host_port = None
                for p in ports:
                    match = re.match(r"(\d+):3306", str(p))
                    if match:
                        host_port = int(match.group(1))
                        break
                return {
                    "host": "127.0.0.1",
                    "port": host_port or 3306,
                    "user": "root",
                    "password": env.get("MYSQL_ROOT_PASSWORD", "password"),
                    "database": env.get("MYSQL_DATABASE", ""),
                }
    except Exception:
        pass
    return None


# ─────────────────────────────────────────────
# DATABASE CONNECTION
# ─────────────────────────────────────────────

class DatabaseConnection:
    def __init__(self, host: str, port: int, user: str, password: str, database: str, label: str = ""):
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.database = database
        self.label = label or f"{host}:{port}/{database}"
        self._conn = None

    def connect(self):
        self._conn = pymysql.connect(
            host=self.host,
            port=self.port,
            user=self.user,
            password=self.password,
            database=self.database,
            connect_timeout=10,
            cursorclass=pymysql.cursors.DictCursor,
        )
        return self

    def _ensure_connected(self):
        if not self._conn or not self._conn.open:
            self.connect()

    def execute(self, sql: str) -> tuple[list, list]:
        """Execute SQL and return (rows, columns)."""
        self._ensure_connected()
        with self._conn.cursor() as cursor:
            cursor.execute(sql)
            if cursor.description:
                columns = [d[0] for d in cursor.description]
                rows = cursor.fetchmany(200)
                return rows, columns
            else:
                self._conn.commit()
                return [{"affected_rows": cursor.rowcount}], ["affected_rows"]

    def get_schema(self) -> dict:
        """Return basic schema dict {table: [column dicts]} for display."""
        self._ensure_connected()
        schema = {}
        with self._conn.cursor() as cursor:
            cursor.execute("SHOW TABLES")
            tables = [list(r.values())[0] for r in cursor.fetchall()]
            for table in tables:
                cursor.execute(f"DESCRIBE `{table}`")
                schema[table] = cursor.fetchall()
        return schema

    def get_full_schema_details(self) -> dict:
        """
        Introspect the live database deeply.
        Returns {table: {columns, row_count, samples, foreign_keys}} for every table.
        """
        self._ensure_connected()
        result = {}

        with self._conn.cursor() as cursor:
            # List all tables
            cursor.execute("SHOW TABLES")
            tables = [list(r.values())[0] for r in cursor.fetchall()]

            for table in tables:
                # Column definitions
                cursor.execute(f"DESCRIBE `{table}`")
                columns = cursor.fetchall()

                # Row count
                try:
                    cursor.execute(f"SELECT COUNT(*) as cnt FROM `{table}`")
                    row_count = cursor.fetchone()["cnt"]
                except Exception:
                    row_count = "?"

                # Sample values per column (skip BLOBs)
                samples: dict[str, list] = {}
                for col in columns:
                    col_name = col["Field"]
                    col_type = col["Type"].lower()
                    if "blob" in col_type or "binary" in col_type:
                        samples[col_name] = []
                        continue
                    try:
                        cursor.execute(
                            f"SELECT DISTINCT `{col_name}` FROM `{table}` "
                            f"WHERE `{col_name}` IS NOT NULL LIMIT 4"
                        )
                        samples[col_name] = [list(r.values())[0] for r in cursor.fetchall()]
                    except Exception:
                        samples[col_name] = []

                # Foreign keys
                fks = []
                try:
                    cursor.execute(f"""
                        SELECT
                            kcu.COLUMN_NAME        AS `column`,
                            kcu.REFERENCED_TABLE_NAME  AS ref_table,
                            kcu.REFERENCED_COLUMN_NAME AS ref_column
                        FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE kcu
                        JOIN INFORMATION_SCHEMA.TABLE_CONSTRAINTS tc
                            ON tc.CONSTRAINT_NAME = kcu.CONSTRAINT_NAME
                           AND tc.TABLE_SCHEMA    = kcu.TABLE_SCHEMA
                        WHERE tc.CONSTRAINT_TYPE = 'FOREIGN KEY'
                          AND kcu.TABLE_SCHEMA   = %s
                          AND kcu.TABLE_NAME     = %s
                    """, (self.database, table))
                    fks = [r for r in cursor.fetchall() if r.get("ref_table")]
                except Exception:
                    fks = []

                result[table] = {
                    "columns": columns,
                    "row_count": row_count,
                    "samples": samples,
                    "foreign_keys": fks,
                }

        return result

    def close(self):
        if self._conn:
            self._conn.close()


# ─────────────────────────────────────────────
# RAG ENGINE
# ─────────────────────────────────────────────

class SQLRAG:
    def __init__(self, db: DatabaseConnection):
        self.db = db
        self.history = []  # list of {"role": "user"|"assistant", "content": str}

        # Verify Ollama is reachable before doing anything else
        self._check_ollama()

        # Build prompts from live schema at init time
        print("  📡 Introspecting database schema...")
        self._schema_context   = build_schema_context(db)
        self._sql_prompt       = self._schema_context + SQL_GENERATION_SUFFIX
        self._explain_prompt   = self._schema_context + EXPLAIN_SUFFIX
        self._summarize_prompt = self._schema_context + RESULT_SUMMARIZE_SUFFIX
        print("  ✅ Schema loaded into AI context.\n")

    def _check_ollama(self):
        """Confirm Ollama is running and the model is available."""
        try:
            r = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
            r.raise_for_status()
            models = [m["name"].split(":")[0] for m in r.json().get("models", [])]
            if MISTRAL_MODEL not in models:
                print(f"\n⚠️  Model '{MISTRAL_MODEL}' not found in Ollama.")
                print(f"   Pull it with:  ollama pull {MISTRAL_MODEL}\n")
                print(f"   Available models: {', '.join(models) or '(none)'}\n")
                sys.exit(1)
            print(f"  🤖 Using local model: {MISTRAL_MODEL} via {OLLAMA_HOST}\n")
        except requests.exceptions.ConnectionError:
            print(f"\n❌  Cannot reach Ollama at {OLLAMA_HOST}")
            print("   Make sure Ollama is running:  ollama serve")
            print(f"   Or set a custom host:         export OLLAMA_HOST=http://your-host:11434\n")
            sys.exit(1)

    def refresh_schema(self):
        """Re-introspect the DB and rebuild all prompts (call after schema changes)."""
        print("  📡 Refreshing schema context...")
        self._schema_context   = build_schema_context(self.db)
        self._sql_prompt       = self._schema_context + SQL_GENERATION_SUFFIX
        self._explain_prompt   = self._schema_context + EXPLAIN_SUFFIX
        self._summarize_prompt = self._schema_context + RESULT_SUMMARIZE_SUFFIX
        self.history.clear()
        print("  ✅ Schema refreshed.\n")

    def _chat(self, system: str, user_message: str, use_history: bool = False) -> str:
        """
        Send a message to Mistral via Ollama's /api/chat endpoint.
        Uses the OpenAI-compatible messages format with a system prompt.
        """
        messages = [{"role": "system", "content": system}]

        if use_history:
            messages += self.history  # inject prior turns

        messages.append({"role": "user", "content": user_message})

        payload = {
            "model": MISTRAL_MODEL,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": 0.1,   # low temp = consistent SQL / JSON
                "num_predict": 2048,
            },
        }

        try:
            response = requests.post(
                f"{OLLAMA_HOST}/api/chat",
                json=payload,
                timeout=120,
            )
            response.raise_for_status()
        except requests.exceptions.Timeout:
            raise RuntimeError("Ollama request timed out — model may still be loading, try again.")
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Ollama request failed: {e}")

        reply = response.json()["message"]["content"]

        if use_history:
            self.history.append({"role": "user",      "content": user_message})
            self.history.append({"role": "assistant",  "content": reply})

        return reply

    def classify(self, user_input: str) -> str:
        result = self._chat(CLASSIFY_PROMPT, user_input)
        return result.strip().upper()

    def generate_sql(self, user_input: str) -> dict:
        raw = self._chat(self._sql_prompt, user_input, use_history=True)
        raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
        return json.loads(raw)

    def explain(self, user_input: str) -> str:
        return self._chat(self._explain_prompt, user_input, use_history=True)

    def summarize_results(self, user_input: str, rows: list, columns: list) -> str:
        preview = rows[:20]
        context = (
            f"User asked: {user_input}\n\n"
            f"Query returned {len(rows)} row(s).\n"
            f"Columns: {columns}\n\n"
            f"Data (first {len(preview)} rows):\n{json.dumps(preview, default=str, indent=2)}"
        )
        return self._chat(self._summarize_prompt, context)


# ─────────────────────────────────────────────
# DISPLAY HELPERS
# ─────────────────────────────────────────────

def print_banner():
    print("\n" + "═" * 65)
    print("  🔍  SQL RAG — Natural Language Database Assistant")
    print("═" * 65)

def print_table(rows: list, columns: list, max_rows: int = 20):
    if not rows:
        print("  (no rows returned)")
        return

    display = rows[:max_rows]
    widths = {col: len(str(col)) for col in columns}
    for row in display:
        for col in columns:
            val = str(row.get(col, ""))[:60]
            widths[col] = max(widths[col], len(val))

    sep = "+-" + "-+-".join("-" * widths[c] for c in columns) + "-+"
    header = "| " + " | ".join(str(c).ljust(widths[c]) for c in columns) + " |"

    print(sep)
    print(header)
    print(sep)
    for row in display:
        line = "| " + " | ".join(str(row.get(c, ""))[:60].ljust(widths[c]) for c in columns) + " |"
        print(line)
    print(sep)

    if len(rows) > max_rows:
        print(f"  ... showing {max_rows} of {len(rows)} rows")

def print_section(title: str, content: str):
    print(f"\n{'─' * 55}")
    print(f"  {title}")
    print(f"{'─' * 55}")
    print(content)

def select_backend() -> Optional[DatabaseConnection]:
    """Interactive backend selector."""
    print("\n🐳 Discovering Docker MySQL backends...\n")
    containers = discover_docker_backends()

    options = []
    for c in containers:
        if c["host_port"]:
            options.append({
                "label": f"[Docker] {c['container_name']} (port {c['host_port']})",
                "host": "127.0.0.1",
                "port": c["host_port"],
                "user": "root",
                "password": "password",
                "database": "",   # will prompt
            })

    options.append({"label": "[Manual] Enter connection details"})

    print("Available backends:")
    for i, opt in enumerate(options, 1):
        print(f"  {i}. {opt['label']}")

    choice = input("\nSelect backend (number): ").strip()
    try:
        idx = int(choice) - 1
        selected = options[idx]
    except (ValueError, IndexError):
        print("Invalid choice.")
        return None

    if selected["label"].startswith("[Manual]") or not selected.get("database"):
        if not selected["label"].startswith("[Manual]"):
            # Docker auto-detected but database name unknown — just ask for it
            print(f"  Host: {selected['host']}  Port: {selected['port']}")
            selected["database"] = input("Database name: ").strip() or "maintenance_data"
        else:
            selected["host"]     = input("Host [127.0.0.1]: ").strip() or "127.0.0.1"
            selected["port"]     = int(input("Port [3306]: ").strip() or "3306")
            selected["user"]     = input("User [root]: ").strip() or "root"
            selected["password"] = input("Password [password]: ").strip() or "password"
            selected["database"] = input("Database: ").strip() or "maintenance_data"
            selected["label"]    = f"{selected['host']}:{selected['port']}/{selected['database']}"

    print(f"\n  Connecting to {selected['label']}...")
    try:
        db = DatabaseConnection(
            host=selected["host"],
            port=selected["port"],
            user=selected["user"],
            password=selected["password"],
            database=selected["database"],
            label=selected["label"],
        )
        db.connect()
        print("  ✅ Connected!\n")
        return db
    except Exception as e:
        print(f"  ❌ Connection failed: {e}")
        return None


# ─────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────

def main():
    print_banner()

    print(f"  Ollama host : {OLLAMA_HOST}")
    print(f"  Model       : {MISTRAL_MODEL}\n")

    db = None
    while not db:
        db = select_backend()
        if not db:
            retry = input("Retry? (y/n): ").strip().lower()
            if retry != "y":
                sys.exit(0)

    rag = SQLRAG(db)

    print("═" * 65)
    print("  Connected to:", db.label)
    print("  Ask anything about your database in plain English.")
    print("  Commands: 'schema' · 'refresh' · 'switch' · 'history' · 'clear' · 'exit'")
    print("═" * 65 + "\n")

    while True:
        try:
            user_input = input("You > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nGoodbye!")
            break

        if not user_input:
            continue

        # ── Built-in commands ──────────────────────────
        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        elif user_input.lower() == "clear":
            rag.history.clear()
            print("  ✅ Conversation history cleared.")
            continue

        elif user_input.lower() == "refresh":
            rag.refresh_schema()
            print("  ✅ Schema re-introspected and AI context updated.")
            continue

        elif user_input.lower() == "history":
            if not rag.history:
                print("  (no history)")
            else:
                for msg in rag.history:
                    role = "You" if msg["role"] == "user" else "AI"
                    print(f"  [{role}] {msg['content'][:120]}{'...' if len(msg['content']) > 120 else ''}")
            continue

        elif user_input.lower() == "switch":
            db.close()
            db = None
            while not db:
                db = select_backend()
                if not db:
                    break
            if db:
                rag = SQLRAG(db)
                print("  ✅ Switched backend.")
            continue

        elif user_input.lower() == "schema":
            try:
                schema = db.get_schema()
                print_section("DATABASE SCHEMA", "")
                for table, cols in schema.items():
                    print(f"\n  📋 {table}")
                    for col in cols:
                        nullable = "nullable" if col.get("Null") == "YES" else "required"
                        print(f"     {col['Field']:<30} {col['Type']:<20} {nullable}")
            except Exception as e:
                print(f"  ❌ Error: {e}")
            continue

        # ── AI Processing ──────────────────────────────
        print("  🤔 Thinking...\n")

        try:
            intent = rag.classify(user_input)
        except Exception as e:
            print(f"  ❌ Classification error: {e}")
            continue

        if intent == "EXPLAIN":
            try:
                response = rag.explain(user_input)
                print_section("📖 EXPLANATION", response)
            except Exception as e:
                print(f"  ❌ Error: {e}")
            continue

        if intent == "UNCLEAR":
            try:
                response = rag.explain(user_input)
                print_section("💬 RESPONSE", response)
            except Exception as e:
                print(f"  ❌ Error: {e}")
            continue

        # QUERY_READ or QUERY_WRITE
        try:
            sql_result = rag.generate_sql(user_input)
        except json.JSONDecodeError as e:
            print(f"  ❌ Failed to parse AI response as JSON: {e}")
            continue
        except Exception as e:
            print(f"  ❌ SQL generation error: {e}")
            continue

        sql = sql_result.get("sql", "").strip()
        explanation = sql_result.get("explanation", "")
        is_destructive = sql_result.get("is_destructive", False)
        warning = sql_result.get("warning")

        print_section("🧾 GENERATED SQL", sql)
        if explanation:
            print(f"\n  ℹ️  {explanation}")

        if is_destructive:
            print(f"\n  ⚠️  WARNING: {warning or 'This query will modify or delete data.'}")
            confirm = input("  Proceed? (yes/no): ").strip().lower()
            if confirm not in ("yes", "y"):
                print("  ❌ Cancelled.")
                continue

        try:
            rows, columns = db.execute(sql)
            print_section(f"📊 RESULTS ({len(rows)} row{'s' if len(rows) != 1 else ''})", "")
            print_table(rows, columns)

            if rows and columns != ["affected_rows"]:
                print("\n  🤖 AI Summary:")
                summary = rag.summarize_results(user_input, rows, columns)
                print(f"  {summary}")

        except Exception as e:
            print(f"\n  ❌ Query execution error: {e}")
            print("  You can rephrase your question or type 'schema' to see table structures.")

        print()


if __name__ == "__main__":
    main()