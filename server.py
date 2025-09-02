import os
from dataclasses import dataclass
from contextlib import asynccontextmanager
from typing import Any, Dict, List

from starlette.applications import Starlette
from starlette.responses import JSONResponse, PlainTextResponse
from starlette.routing import Route, Mount

from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool
from mcp.server.fastmcp import FastMCP, Context
from mcp.server.session import ServerSession

# ---------- Config ----------
APP_NAME = "postgres-mcp"
DEFAULT_PORT = int(os.getenv("PORT", "8000"))
DSN = os.getenv("DATABASE_URL", "")  # e.g., postgresql://user:pass@host:5432/db?sslmode=require

# Optional pool sizing
POOL_MIN = int(os.getenv("PG_POOL_MIN", "1"))
POOL_MAX = int(os.getenv("PG_POOL_MAX", "5"))

# Global handle (used by /healthz)
_pool: AsyncConnectionPool | None = None

# ---------- MCP lifespan (opens/closes the pool) ----------
@dataclass
class AppCtx:
    pool: AsyncConnectionPool

@asynccontextmanager
async def lifespan(_: FastMCP):
    global _pool
    if not DSN:
        raise RuntimeError("DATABASE_URL is not set. Provide it via env or Key Vault.")
    _pool = AsyncConnectionPool(
        conninfo=DSN, min_size=POOL_MIN, max_size=POOL_MAX, open=False
    )
    await _pool.open()
    try:
        yield AppCtx(pool=_pool)
    finally:
        await _pool.close()
        _pool = None

# ---------- MCP server ----------
mcp = FastMCP(APP_NAME, lifespan=lifespan, stateless_http=True)

def _is_select_only(sql: str) -> bool:
    head = sql.lstrip().lower()
    return head.startswith(("select", "with", "show", "explain"))

async def _run_query(sql: str, params: Dict[str, Any] | None, limit: int) -> Dict[str, Any]:
    if not _is_select_only(sql):
        raise ValueError("Only read-only queries are allowed (SELECT/CTE/SHOW/EXPLAIN).")
    app_ctx: AppCtx = mcp.session_manager.lifespan_context  # type: ignore[attr-defined]
    async with app_ctx.pool.connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(sql, params or {})
            rows: List[Dict[str, Any]] = await cur.fetchmany(limit)
            return {"rowcount": len(rows), "rows": rows}

@mcp.tool()
async def ping(ctx: Context[ServerSession, AppCtx]) -> str:
    """Quick connectivity check (SELECT 1)."""
    app_ctx = ctx.request_context.lifespan_context
    async with app_ctx.pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute("SELECT 1;")
            await cur.fetchone()
    return "ok"

@mcp.tool()
async def list_tables(schema: str | None = "public") -> Dict[str, Any]:
    """List tables in a schema (default: public)."""
    sql = """
    SELECT table_schema, table_name
    FROM information_schema.tables
    WHERE table_type='BASE TABLE'
      AND (%(schema)s IS NULL OR table_schema = %(schema)s)
    ORDER BY table_schema, table_name;
    """
    return await _run_query(sql, {"schema": schema}, 10_000)

@mcp.tool()
async def describe_table(table: str, schema: str = "public") -> Dict[str, Any]:
    """Describe columns for a table."""
    sql = """
    SELECT column_name, data_type, is_nullable, character_maximum_length, numeric_precision, numeric_scale
    FROM information_schema.columns
    WHERE table_schema = %(schema)s AND table_name = %(table)s
    ORDER BY ordinal_position;
    """
    return await _run_query(sql, {"schema": schema, "table": table}, 10_000)

@mcp.tool()
async def query(sql: str, max_rows: int = 1000) -> Dict[str, Any]:
    """
    Execute a read-only SQL (SELECT/CTE/SHOW/EXPLAIN).
    """
    return await _run_query(sql, None, max_rows)

@mcp.tool()
async def params_query(sql: str, params: Dict[str, Any], max_rows: int = 1000) -> Dict[str, Any]:
    """
    Execute a parameterized read-only SQL.
    Use psycopg named placeholders like: SELECT * FROM t WHERE id = %(id)s
    """
    return await _run_query(sql, params, max_rows)

# ---------- Starlette app: mount MCP + health ----------
async def healthz(_req):
    # liveness: app is up; readiness: DB reachable
    if not DSN:
        return JSONResponse({"status": "error", "reason": "no DATABASE_URL"}, status_code=500)
    try:
        if _pool is None:
            return JSONResponse({"status": "starting", "reason": "pool not ready"}, status_code=503)
        async with _pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute("SELECT 1;")
                await cur.fetchone()
        return JSONResponse({"status": "ok"})
    except Exception as e:
        return JSONResponse({"status": "error", "reason": str(e)}, status_code=500)

# Initialize the pool when the Starlette app starts
async def startup():
    global _pool
    if not DSN:
        raise RuntimeError("DATABASE_URL is not set. Provide it via env or Key Vault.")
    if _pool is None:
        _pool = AsyncConnectionPool(
            conninfo=DSN, min_size=POOL_MIN, max_size=POOL_MAX, open=False
        )
        await _pool.open()

async def shutdown():
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None

# Create the MCP app
mcp_app = mcp.streamable_http_app()

app = Starlette(
    routes=[
        Route("/healthz", healthz, methods=["GET"]),
        # MCP will be available under /mcp (e.g. https://your.host/mcp)
        Mount("/mcp", app=mcp_app),
    ],
    on_startup=[startup],
    on_shutdown=[shutdown],
)
# For local dev: uvicorn server:app --port 8000 --host 0.0.0.0
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
    "server:app",
    host="0.0.0.0",
    port=int(os.getenv("PORT", "8000")),
    proxy_headers=True,
    forwarded_allow_ips="*",
)
