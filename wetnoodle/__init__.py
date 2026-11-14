"""wetnoodle — a local memory archive for Hermes Agent.

Modules are entry points first and library second: each of ingest, ingest_skills,
query, doctor, eval_recall, export_store and import_store runs as `python -m
wetnoodle.<name>`. `rag_core` holds the store and retrieval logic that all of them
share, and `render` decides how much retrieved text is shown.
"""

__all__ = ["rag_core", "render"]
