"""Tool registry and the single choke-point (`execute`) through which every tool call must pass.

Risk levels
  read          - no external or persistent side effects (fetch/read/calc). Allowed in every mode.
  write         - creates files in the user's own workspace. Not available in 'advise' mode.
  consequential - irreversible or leaves the system (delete, email). Requires a matching, single-use,
                  user-approved approval record, verified HERE in the backend, not in the UI.
"""
from __future__ import annotations

import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutTimeout
from dataclasses import dataclass, field
from typing import Any, Callable

import jsonschema

from ..db import Database, new_id

MODES = ("advise", "auto", "execute")
_pool = ThreadPoolExecutor(max_workers=16, thread_name_prefix="tool")


@dataclass
class ToolContext:
    user_id: str
    run_id: str
    conversation_id: str
    mode: str
    db: Database
    settings: Any
    files: Any


@dataclass
class Tool:
    name: str
    description: str
    input_schema: dict
    risk: str  # read | write | consequential
    func: Callable[[ToolContext, dict], str]
    summarize: Callable[[dict], str] = lambda a: json.dumps(a)[:200]
    available: Callable[[Any], tuple[bool, str]] = lambda settings: (True, "")

    def spec(self) -> dict:
        return {"name": self.name, "description": self.description, "input_schema": self.input_schema}


class ToolError(Exception):
    """Expected, user-presentable failure (bad input, blocked URL, unsupported file...)."""


class Registry:
    def __init__(self):
        self.tools: dict[str, Tool] = {}

    def register(self, t: Tool):
        self.tools[t.name] = t

    def for_mode(self, mode: str, settings) -> list[Tool]:
        out = []
        for t in self.tools.values():
            if not t.available(settings)[0]:
                continue
            if mode == "advise" and t.risk != "read":
                continue
            out.append(t)
        return out

    def describe(self, settings) -> list[dict]:
        return [{"name": t.name, "risk": t.risk, "description": t.description.split(".")[0],
                 "available": t.available(settings)[0], "reason": t.available(settings)[1]}
                for t in self.tools.values()]


def args_hash(name: str, args: dict) -> str:
    return hashlib.sha256((name + json.dumps(args, sort_keys=True, default=str)).encode()).hexdigest()


# ---- approvals ---------------------------------------------------------------------------------
def request_approval(db: Database, ctx: ToolContext, tool: Tool, tool_use_id: str, args: dict) -> str:
    aid = new_id()
    db.run("INSERT INTO approvals(id,run_id,user_id,tool_use_id,tool_name,args,args_hash,summary,status,created_at) "
           "VALUES(?,?,?,?,?,?,?,?,?,?)",
           (aid, ctx.run_id, ctx.user_id, tool_use_id, tool.name, json.dumps(args), args_hash(tool.name, args),
            tool.summarize(args), "pending", time.time()))
    return aid


def decide_approval(db: Database, user_id: str, approval_id: str, approve: bool) -> bool:
    n = db.run("UPDATE approvals SET status=?, decided_at=? WHERE id=? AND user_id=? AND status='pending'",
               ("approved" if approve else "denied", time.time(), approval_id, user_id))
    return n == 1


def _consume_approval(db: Database, ctx: ToolContext, tool: Tool, tool_use_id: str, args: dict) -> bool:
    """Atomically flips approved -> consumed. Fails unless the approval matches this exact run, user,
    tool call and arguments. A second attempt with the same approval always fails."""
    n = db.run("UPDATE approvals SET status='consumed' WHERE run_id=? AND user_id=? AND tool_use_id=? "
               "AND tool_name=? AND args_hash=? AND status='approved'",
               (ctx.run_id, ctx.user_id, tool_use_id, tool.name, args_hash(tool.name, args)))
    return n == 1


# ---- execution -----------------------------------------------------------------------------------
@dataclass
class ToolResult:
    content: str
    is_error: bool = False


def execute(registry: Registry, ctx: ToolContext, name: str, tool_use_id: str, args: Any, timeout: float) -> ToolResult:
    tool = registry.tools.get(name)
    if tool is None:
        return ToolResult(f"Unknown tool '{name}'.", True)
    ok, why = tool.available(ctx.settings)
    if not ok:
        return ToolResult(f"Tool '{name}' is unavailable: {why}", True)
    if ctx.mode not in MODES:
        return ToolResult("Invalid mode.", True)
    if ctx.mode == "advise" and tool.risk != "read":
        return ToolResult(f"'{name}' changes things and is disabled because this request is in advice-only mode. "
                          "Explain what you would do instead and let the user switch modes if they want it done.", True)
    if not isinstance(args, dict):
        return ToolResult("Tool arguments must be a JSON object.", True)
    try:
        jsonschema.validate(args, tool.input_schema)
    except jsonschema.ValidationError as e:
        return ToolResult(f"Invalid arguments for '{name}': {e.message}", True)
    if tool.risk == "consequential" and not _consume_approval(ctx.db, ctx, tool, tool_use_id, args):
        return ToolResult(f"'{name}' was NOT executed: no valid user approval exists for this exact action.", True)
    fut = _pool.submit(tool.func, ctx, args)
    try:
        return ToolResult(fut.result(timeout=timeout))
    except FutTimeout:
        fut.cancel()
        extra = " It may still be finishing in the background; verify before retrying." if tool.risk != "read" else ""
        return ToolResult(f"'{name}' timed out after {int(timeout)}s.{extra}", True)
    except ToolError as e:
        return ToolResult(str(e), True)
    except Exception as e:  # unexpected: report type + message, never a traceback with paths
        return ToolResult(f"'{name}' failed unexpectedly: {type(e).__name__}: {str(e)[:300]}", True)
