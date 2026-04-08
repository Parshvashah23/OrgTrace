"""
OrgTrace — Email Triage Environment Wrapper
Alias for OrgMemoryEnv to support deployment tutorial naming conventions.
"""

from .org_memory_env import OrgMemoryEnv as OrgTraceEnv

# Optional: Add a factory if specific initialization is needed for the Space
def create_env(data_dir: str = "data/generated/", seed: int = 42) -> OrgTraceEnv:
    return OrgTraceEnv(data_dir=data_dir, seed=seed)
