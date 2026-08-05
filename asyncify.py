"""Move blocking tool bodies off the event loop (RFC-060 P3a).

WHY THIS EXISTS
FastMCP decides dispatch from the DECORATED function. A sync `def` tool runs
INLINE ON THE EVENT LOOP, so while it runs the whole broker stops answering —
all 26 tools, every client, on any transport. Every tool here wraps a blocking
`subprocess.run` (ssh/scp to anteframe), so this is not theoretical.

Measured on ct102, two identical servers differing only in how the slow tool was
declared, each with a trivial `ping` canary:

    def slow()                    -> ping during a 25s call = 22.00s  (FROZEN)
    async def + asyncio.to_thread -> ping during a 25s call =  0.01s  (fine)

The slow tool itself took 25.0s in BOTH arms, so this costs no throughput.

WHY A DECORATOR AND NOT 26 REWRITES
Converting each tool to `async def` by hand means touching every signature and
every call site. Wrapping instead keeps the bodies untouched and provably
preserves the tool contract: `functools.wraps` sets `__wrapped__`, and
`inspect.signature` follows it, so FastMCP derives the SAME schema. Verified —
name, description and inputSchema compare equal between a plain sync tool and
an `@aio`-wrapped one, and the wrapped tool round-trips a real call.

    @mcp.tool()
    @aio
    def af_run_command(command: str) -> str: ...

Order matters: `@mcp.tool()` must sit ABOVE `@aio` so it registers the async
wrapper, not the sync original.

⚠️ CONCURRENCY IS NOW REAL — that is the point, and it has a cost.
The event loop's serialisation WAS this codebase's only mutual-exclusion
primitive. Anything sharing a fixed remote path must now lock explicitly, and
because these bodies run in WORKER THREADS the correct primitive is
`threading.Lock`, NOT `asyncio.Lock`. See `tools/desk_bridge.py`.
"""

import asyncio
import functools


def aio(fn):
    """Return an async wrapper that runs `fn` in a worker thread."""

    @functools.wraps(fn)
    async def wrapper(*args, **kwargs):
        return await asyncio.to_thread(fn, *args, **kwargs)

    return wrapper
