# Full memory.py with data/memory folder integration

import sqlite3
import json
import os
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple

import torch
import numpy as np
from sentence_transformers import SentenceTransformer

# ------------------------------------------------------------------------------
# Configuration flags and constants
# ------------------------------------------------------------------------------
EMOTION_FLOOR_ENABLED = False      # Gate emotion-floor logic
STRENGTH_FLOOR = 0.3               # Minimum strength for high-emotion memories
SHORT_HALF_LIFE_HOURS = 12         # Short-phase half-life
LONG_HALF_LIFE_DAYS = 14           # Long-phase half-life

RECENCY_WEIGHT = 0.4               # Weight for recency in scoring
SEMANTIC_WEIGHT = 0.5              # Weight for semantic similarity
EMOTION_WEIGHT = 0.1               # Weight for emotion congruence
MIN_REHEARSE_SIM = 0.7             # Minimum similarity to rehearse

DEFAULT_DB_PATH = "data/memory/remi_memory.db"

# ------------------------------------------------------------------------------
# Database schema
# ------------------------------------------------------------------------------
DB_SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp        TEXT    NOT NULL,
    category         TEXT    NOT NULL,    -- episodic, semantic, procedural, summary
    content          TEXT    NOT NULL,
    strength         REAL    NOT NULL,
    emotion_score    REAL    NOT NULL,
    emotions_json    TEXT    NOT NULL,
    tags_json        TEXT,
    is_archived      INTEGER NOT NULL,    -- 0=active,1=archived,2=flagged for summarization
    superseded_by    INTEGER,
    model_name       TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS embeddings (
    memory_id  INTEGER PRIMARY KEY,       -- foreign key to memories.id
    vector     BLOB    NOT NULL           -- serialized float32 embedding
);
"""

# ------------------------------------------------------------------------------
# MemoryEntry data class
# ------------------------------------------------------------------------------
class MemoryEntry:
    """
    Represents a single memory record from the database.
    """
    def __init__(self, row: Tuple):
        (self.id, ts, self.category, self.content,
         self.strength, self.emotion_score, emo_json,
         tags_json, self.is_archived, self.superseded_by,
         self.model_name) = row
        self.timestamp = datetime.fromisoformat(ts)
        self.emotions = json.loads(emo_json)
        self.tags = json.loads(tags_json or "[]")

# ------------------------------------------------------------------------------
# Initialization
# ------------------------------------------------------------------------------
def initialize_memory_store(db_path: str = DEFAULT_DB_PATH) -> None:
    """
    Create/connect to the SQLite DB at db_path and ensure tables exist.
    """
    parent = os.path.dirname(db_path)
    if parent:                      # only make dirs if there's a parent folder
        os.makedirs(parent, exist_ok=True)

    conn = sqlite3.connect(db_path)
    conn.executescript(DB_SCHEMA)
    conn.commit()
    conn.close()
    
# ------------------------------------------------------------------------------
# Embedding Encoder Loader
# ------------------------------------------------------------------------------
def load_encoder() -> SentenceTransformer:
    """
    Auto-select a SentenceTransformer encoder based on hardware availability.
    """
    if torch.cuda.is_available():
        model_name = "all-mpnet-base-v2"
    else:
        model_name = "all-MiniLM-L6-v2"
    return SentenceTransformer(model_name)

# ------------------------------------------------------------------------------
# Memory API functions
# ------------------------------------------------------------------------------
def add_memory(
    db_path: str,
    content: str,
    category: str,
    emotions: Dict[str, float],
    tags: Optional[List[str]] = None,
    supersedes: Optional[int] = None,
    model_name: Optional[str] = None
) -> int:
    """
    Insert a new memory record with initial strength=1.0 and emotion fields.
    Returns the new memory's id.
    """
    # Timestamp in ISO format
    ts = datetime.utcnow().isoformat()
    # Initial strength
    strength = 1.0
    # Determine emotion_score
    emotion_score = max(emotions.values()) if emotions else 0.0
    # JSON-encode emotion vector
    # Ensure all 12 emotions keys exist (if needed, fill zeros)
    emo_json = json.dumps(emotions)
    # Tags
    tags_json = json.dumps(tags or [])
    # Model name for embeddings
    model_name = model_name or "all-MiniLM-L6-v2"
    # Insert into DB
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO memories
          (timestamp, category, content, strength,
           emotion_score, emotions_json, tags_json,
           is_archived, superseded_by, model_name)
        VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?, ?)
    """, (ts, category, content, strength,
          emotion_score, emo_json, tags_json,
          supersedes, model_name))
    mem_id = cur.lastrowid
    conn.commit()
    conn.close()
    return mem_id

def get_recent(db_path: str, n: int = 5) -> List[MemoryEntry]:
    """
    Fetch the last n active, unsuperseded memories ordered by timestamp descending.
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""
        SELECT * FROM memories
        WHERE is_archived = 0 AND superseded_by IS NULL
        ORDER BY timestamp DESC
        LIMIT ?
    """, (n,))
    rows = cur.fetchall()
    conn.close()
    return [MemoryEntry(r) for r in rows]

def query_memories(
    db_path: str,
    query: str,
    top_k: int = 5,
    min_score: float = 0.5,
    include_archived: bool = False,
    historical: bool = False
) -> List[Dict]:
    """
    Retrieve relevant memories scored by recency, semantic similarity,
    and emotion congruence. Returns a list of dicts:
    { "entry": MemoryEntry, "score": float }.
    """
    # 1. Encode query
    encoder = load_encoder()
    q_emb = encoder.encode([query])[0].astype(np.float32)

    # 2. Fetch candidate memories
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    archived_clause = "" if include_archived else "AND is_archived = 0"
    superseded_clause = "" if historical else "AND superseded_by IS NULL"
    cur.execute(f"""
        SELECT m.id, m.timestamp, m.category, m.content,
               m.strength, m.emotion_score, m.emotions_json,
               m.tags_json, m.is_archived, m.superseded_by,
               m.model_name, e.vector
        FROM memories m
        JOIN embeddings e ON m.id = e.memory_id
        WHERE 1=1 {archived_clause} {superseded_clause}
    """)
    rows = cur.fetchall()
    conn.close()

    # 3. Score each memory
    candidates = []
    now = datetime.utcnow()
    mood_vec = getattr(query_memories, "current_mood_vector", None)
    for row in rows:
        # Split row into memory fields and embedding blob
        mem_fields, emb_blob = row[:-1], row[-1]
        entry = MemoryEntry(mem_fields)
        m_emb = np.frombuffer(emb_blob, dtype=np.float32)

        # Recency score
        age = now - entry.timestamp
        tau = (SHORT_HALF_LIFE_HOURS if age < timedelta(hours=48)
               else LONG_HALF_LIFE_DAYS * 24)
        recency_score = np.exp(-age.total_seconds() / (3600 * tau))

        # Semantic similarity
        sim = float(np.dot(q_emb, m_emb) /
                    (np.linalg.norm(q_emb) * np.linalg.norm(m_emb) + 1e-8))

        # Emotion congruence
        if mood_vec is not None:
            emo_vec = np.array(list(entry.emotions.values()), dtype=np.float32)
            emo_score = float(np.dot(mood_vec, emo_vec) /
                              (np.linalg.norm(mood_vec) * (np.linalg.norm(emo_vec)+1e-8)))
        else:
            emo_score = 0.0

        # Combined score
        combined = (RECENCY_WEIGHT * recency_score +
                    SEMANTIC_WEIGHT * sim +
                    EMOTION_WEIGHT * emo_score)

        candidates.append((entry, combined, sim))

    # 4. Filter and sort
    filtered = [t for t in candidates if t[1] >= min_score]
    filtered.sort(key=lambda x: x[1], reverse=True)
    selected = filtered[:top_k]

    # 5. Auto-rehearse high-sim entries
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    for entry, combined, sim in selected:
        if sim >= MIN_REHEARSE_SIM:
            new_strength = entry.strength + 0.1
            if EMOTION_FLOOR_ENABLED and entry.emotion_score >= STRENGTH_FLOOR:
                new_strength = max(new_strength, STRENGTH_FLOOR)
            entry.strength = min(1.0, new_strength)
            cur.execute("UPDATE memories SET strength = ? WHERE id = ?",
                        (entry.strength, entry.id))
    conn.commit()
    conn.close()

    # Return structured results
    return [{"entry": e, "score": s} for e, s, _ in selected]

def decay_memories(db_path: str):
    """
    Apply two-phase exponential decay to all active memories.
    """
    now = datetime.utcnow()
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT id, timestamp, strength, emotion_score FROM memories WHERE is_archived = 0")
    rows = cur.fetchall()
    for mid, ts, strength, emo_score in rows:
        age = now - datetime.fromisoformat(ts)
        if age < timedelta(hours=48):
            tau = SHORT_HALF_LIFE_HOURS
        else:
            tau = LONG_HALF_LIFE_DAYS * 24
        decay = 0.5 ** (age.total_seconds() / (3600 * tau))
        new_strength = strength * decay
        if EMOTION_FLOOR_ENABLED and emo_score >= STRENGTH_FLOOR:
            new_strength = max(new_strength, STRENGTH_FLOOR)
        cur.execute("UPDATE memories SET strength = ? WHERE id = ?", (new_strength, mid))
    conn.commit()
    conn.close()

def flag_for_summarization(db_path: str):
    """
    Flag memories for summarization: low strength and low emotion.
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""
        UPDATE memories
        SET is_archived = 2
        WHERE strength < 0.1 AND emotion_score < 0.8 AND is_archived = 0
    """)
    conn.commit()
    conn.close()

def local_summarize(texts: List[str]) -> str:
    """
    Summarize a list of memory contents via the local LLM.
    """
    from run_robot_brain import ask_model
    prompt = "Please summarize these memories succinctly:\n" + "\n".join(texts)
    return ask_model(prompt)

def summarize_flagged(db_path: str) -> Optional[int]:
    """
    Batch-summarize flagged memories and archive originals.
    Returns the summary memory's id, or None if none flagged.
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT id, content, emotions_json FROM memories WHERE is_archived = 2")
    flagged = cur.fetchall()
    if not flagged:
        conn.close()
        return None

    ids, contents, emos = zip(*flagged)
    summary_text = local_summarize(list(contents))

    # Average emotion vectors
    emos_list = [json.loads(e) for e in emos]
    avg_emos = {k: float(np.mean([em[k] for em in emos_list])) for k in emos_list[0]}

    # Insert summary memory
    summary_id = add_memory(
        db_path=db_path,
        content=summary_text,
        category="summary",
        emotions=avg_emos,
        tags=["auto_summary"],
        supersedes=None,
        model_name=load_encoder().name_or_path
    )

    # Archive originals
    cur.execute(f"UPDATE memories SET is_archived = 1 WHERE id IN ({','.join('?'*len(ids))})", ids)
    conn.commit()
    conn.close()
    return summary_id

def reembed_memories(db_path: str, model_name: str):
    """
    Recompute embeddings for all active memories with a new encoder.
    """
    encoder = SentenceTransformer(model_name)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Update model_name for all
    cur.execute("UPDATE memories SET model_name = ?", (model_name,))

    # Fetch active memories
    cur.execute("SELECT id, content FROM memories WHERE is_archived = 0")
    rows = cur.fetchall()
    for mid, content in rows:
        vec = encoder.encode([content])[0].astype(np.float32)
        blob = vec.tobytes()
        cur.execute("""
            INSERT OR REPLACE INTO embeddings(memory_id, vector)
            VALUES (?, ?)
        """, (mid, blob))

    conn.commit()
    conn.close()

# End of memory.py file

