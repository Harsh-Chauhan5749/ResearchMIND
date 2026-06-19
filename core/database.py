"""
ResearchMind AI – Database Layer (SQLite)
Stores paper metadata, chat sessions, messages, and comparison results.
"""

import sqlite3
import json
from datetime import datetime
from typing import List, Dict, Optional, Any

from core.config import DB_PATH


# ─── Connection ───────────────────────────────────────────────────────────────

def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")   # concurrent reads
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db():
    """Create tables on first run."""
    conn = get_connection()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS papers (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            filename      TEXT    NOT NULL,
            title         TEXT,
            authors       TEXT,
            abstract      TEXT,
            total_pages   INTEGER DEFAULT 0,
            total_chunks  INTEGER DEFAULT 0,
            file_size     INTEGER DEFAULT 0,
            file_path     TEXT,
            upload_date   TEXT    DEFAULT (datetime('now')),
            summary       TEXT,
            keywords      TEXT,    -- JSON array
            citations     TEXT,    -- JSON array
            year          INTEGER,
            status        TEXT    DEFAULT 'processing'
        );

        CREATE TABLE IF NOT EXISTS chat_sessions (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT    NOT NULL UNIQUE,
            title      TEXT,
            paper_ids  TEXT,   -- JSON array of int
            created_at TEXT    DEFAULT (datetime('now')),
            updated_at TEXT    DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS chat_messages (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id       TEXT    NOT NULL,
            role             TEXT    NOT NULL,
            content          TEXT    NOT NULL,
            sources          TEXT,  -- JSON array
            confidence_score REAL,
            created_at       TEXT   DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS paper_comparisons (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            paper_ids   TEXT NOT NULL,   -- JSON array
            result      TEXT,
            created_at  TEXT DEFAULT (datetime('now'))
        );

        CREATE INDEX IF NOT EXISTS idx_messages_session
            ON chat_messages(session_id);
        CREATE INDEX IF NOT EXISTS idx_papers_status
            ON papers(status);
    """)
    conn.commit()
    conn.close()


# ─── Papers ───────────────────────────────────────────────────────────────────

def add_paper(
    filename: str,
    file_path: str,
    file_size: int = 0,
    title: str = "",
    authors: str = "",
    abstract: str = "",
    total_pages: int = 0,
    keywords: Optional[List[str]] = None,
    citations: Optional[List[str]] = None,
    year: Optional[int] = None,
) -> int:
    conn = get_connection()
    
    # Ensure citations column exists (schema migration if needed)
    try:
        conn.execute("ALTER TABLE papers ADD COLUMN citations TEXT")
    except sqlite3.OperationalError:
        pass
        
    cur = conn.execute(
        """INSERT INTO papers
           (filename, file_path, file_size, title, authors, abstract,
            total_pages, keywords, citations, year)
           VALUES (?,?,?,?,?,?,?,?,?,?)""",
        (filename, file_path, file_size, title, authors, abstract,
         total_pages, json.dumps(keywords or []), json.dumps(citations or []), year),
    )
    paper_id = cur.lastrowid
    conn.commit()
    conn.close()
    return paper_id


def update_paper(
    paper_id: int,
    total_chunks: int = 0,
    summary: str = "",
    status: str = "ready",
    title: str = "",
    keywords: Optional[List[str]] = None,
):
    conn = get_connection()
    conn.execute(
        """UPDATE papers
           SET total_chunks=?, summary=?, status=?, title=COALESCE(NULLIF(?,''), title),
               keywords=COALESCE(NULLIF(?,'null'), keywords)
           WHERE id=?""",
        (total_chunks, summary, status, title,
         json.dumps(keywords) if keywords else None, paper_id),
    )
    conn.commit()
    conn.close()


def get_all_papers(status: str = "ready") -> List[Dict]:
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM papers WHERE status=? ORDER BY upload_date DESC", (status,)
    ).fetchall()
    conn.close()
    papers = [dict(r) for r in rows]
    for p in papers:
        try:
            p["keywords"] = json.loads(p.get("keywords") or "[]")
        except Exception:
            p["keywords"] = []
        try:
            p["citations"] = json.loads(p.get("citations") or "[]")
        except Exception:
            p["citations"] = []
    return papers


def get_paper_by_id(paper_id: int) -> Optional[Dict]:
    conn = get_connection()
    row = conn.execute("SELECT * FROM papers WHERE id=?", (paper_id,)).fetchone()
    conn.close()
    if not row:
        return None
    p = dict(row)
    try:
        p["keywords"] = json.loads(p.get("keywords") or "[]")
    except Exception:
        p["keywords"] = []
    try:
        p["citations"] = json.loads(p.get("citations") or "[]")
    except Exception:
        p["citations"] = []
    return p


def delete_paper(paper_id: int):
    conn = get_connection()
    conn.execute("DELETE FROM papers WHERE id=?", (paper_id,))
    conn.commit()
    conn.close()


def get_paper_stats() -> Dict[str, Any]:
    conn = get_connection()

    def scalar(sql, *args):
        row = conn.execute(sql, args).fetchone()
        return (row[0] or 0) if row else 0

    stats = {
        "total_papers":        scalar("SELECT COUNT(*) FROM papers WHERE status='ready'"),
        "total_pages":         scalar("SELECT SUM(total_pages) FROM papers WHERE status='ready'"),
        "total_chunks":        scalar("SELECT SUM(total_chunks) FROM papers WHERE status='ready'"),
        "total_chat_sessions": scalar("SELECT COUNT(DISTINCT session_id) FROM chat_messages"),
        "total_messages":      scalar("SELECT COUNT(*) FROM chat_messages"),
    }
    conn.close()
    return stats


# ─── Chat ─────────────────────────────────────────────────────────────────────

def ensure_session(session_id: str, title: str = "", paper_ids: Optional[List[int]] = None):
    conn = get_connection()
    conn.execute(
        "INSERT OR IGNORE INTO chat_sessions (session_id, title, paper_ids) VALUES (?,?,?)",
        (session_id, title or f"Chat {datetime.now():%Y-%m-%d %H:%M}",
         json.dumps(paper_ids or [])),
    )
    conn.execute(
        "UPDATE chat_sessions SET updated_at=datetime('now') WHERE session_id=?",
        (session_id,),
    )
    conn.commit()
    conn.close()


def save_message(
    session_id: str,
    role: str,
    content: str,
    sources: Optional[List[Dict]] = None,
    confidence: Optional[float] = None,
):
    conn = get_connection()
    conn.execute(
        """INSERT INTO chat_messages (session_id, role, content, sources, confidence_score)
           VALUES (?,?,?,?,?)""",
        (session_id, role, content, json.dumps(sources or []), confidence),
    )
    conn.commit()
    conn.close()


def get_chat_history(session_id: str) -> List[Dict]:
    conn = get_connection()
    rows = conn.execute(
        """SELECT role, content, sources, confidence_score, created_at
           FROM chat_messages WHERE session_id=? ORDER BY created_at""",
        (session_id,),
    ).fetchall()
    conn.close()
    msgs = []
    for r in rows:
        m = dict(r)
        try:
            m["sources"] = json.loads(m.get("sources") or "[]")
        except Exception:
            m["sources"] = []
        msgs.append(m)
    return msgs


def get_all_sessions() -> List[Dict]:
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM chat_sessions ORDER BY updated_at DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]
