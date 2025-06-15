# tests/test_memory.py

import os
import sqlite3
import pytest

from src.memory import (
    initialize_memory_store,
    add_memory,
    get_recent,
    decay_memories,
    flag_for_summarization,
    reembed_memories
)

@pytest.fixture
def db_path(tmp_path):
    """
    Create a fresh SQLite database in a temporary directory.
    The directory is safe for file creation and will be cleaned up automatically.
    """
    # tmp_path is a pathlib.Path pointing to a unique temp directory
    db_file = tmp_path / "test_remi.db"
    # Initialize the database schema in this file
    initialize_memory_store(str(db_file))
    return str(db_file)

def test_add_and_get_recent(db_path):
    # Add two memories and verify retrieval order
    add_memory(db_path, "Hello world", category="episodic", emotions={})
    add_memory(db_path, "Favorite color = blue", category="semantic", emotions={})
    recent = [m.content for m in get_recent(db_path, n=2)]
    assert recent == ["Favorite color = blue", "Hello world"]

def test_decay_and_flag(db_path):
    # Insert an “old” memory manually with low strength
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    old_ts = "2000-01-01T00:00:00"
    cur.execute("""
        INSERT INTO memories
          (timestamp, category, content, strength,
           emotion_score, emotions_json, tags_json,
           is_archived, superseded_by, model_name)
        VALUES (?, 'episodic', 'Old event', 0.05, 0.0, '{}', '[]', 0, NULL, 'test-model')
    """, (old_ts,))
    conn.commit()
    conn.close()

    # Run decay and flagging logic
    decay_memories(db_path)
    flag_for_summarization(db_path)

    # Confirm the "Old event" is flagged for summarization (is_archived == 2)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT is_archived FROM memories WHERE content = 'Old event'")
    flag_value = cur.fetchone()[0]
    conn.close()
    assert flag_value == 2

def test_reembed(db_path):
    # Verify embeddings table is initially empty
    conn = sqlite3.connect(db_path)
    initial_count = conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
    conn.close()
    assert initial_count == 0

    # Perform re-embedding of active memories
    reembed_memories(db_path, "all-MiniLM-L6-v2")

    # Confirm embeddings now match active memories count
    conn = sqlite3.connect(db_path)
    active_count = conn.execute("SELECT COUNT(*) FROM memories WHERE is_archived = 0").fetchone()[0]
    embed_count = conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
    conn.close()
    assert embed_count == active_count
