"""
ComplyTrade Pilot V2 — Database Setup
"""

import psycopg2
from psycopg2.extras import Json
from .settings import DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASS


def get_connection():
    """Get a database connection."""
    return psycopg2.connect(
        host=DB_HOST, port=DB_PORT,
        dbname=DB_NAME, user=DB_USER, password=DB_PASS,
    )


def init_database():
    """Initialize database tables."""
    conn = get_connection()
    conn.autocommit = True
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS jobs (
        job_id TEXT PRIMARY KEY,
        filename TEXT NOT NULL,
        status TEXT DEFAULT 'uploaded',
        current_step INTEGER DEFAULT 0,
        total_pages INTEGER DEFAULT 0,
        created_at TIMESTAMP DEFAULT NOW(),
        completed_at TIMESTAMP,
        elapsed_seconds REAL,
        error_message TEXT
    );

    CREATE TABLE IF NOT EXISTS pages (
        id SERIAL PRIMARY KEY,
        job_id TEXT REFERENCES jobs(job_id),
        page_number INTEGER NOT NULL,
        raw_text TEXT,
        cleaned_text TEXT,
        refined_text TEXT,
        final_text TEXT,
        confidence REAL DEFAULT 0.0,
        ocr_method TEXT DEFAULT 'glm-ocr',
        has_stamps BOOLEAN DEFAULT FALSE,
        has_signatures BOOLEAN DEFAULT FALSE,
        corrections JSONB DEFAULT '[]',
        created_at TIMESTAMP DEFAULT NOW(),
        UNIQUE(job_id, page_number)
    );

    CREATE TABLE IF NOT EXISTS packets (
        id SERIAL PRIMARY KEY,
        job_id TEXT REFERENCES jobs(job_id),
        packet_id TEXT NOT NULL,
        page_numbers INTEGER[] NOT NULL,
        packet_type TEXT,  -- 'mt' or 'shipping'
        mt_type TEXT,      -- 'MT700', 'MT707', etc.
        document_type TEXT,
        classification_confidence REAL DEFAULT 0.0,
        boundary_confidence REAL DEFAULT 0.0,
        is_alien BOOLEAN DEFAULT FALSE,
        human_override TEXT,
        created_at TIMESTAMP DEFAULT NOW(),
        UNIQUE(job_id, packet_id)
    );

    CREATE TABLE IF NOT EXISTS final_lc (
        id SERIAL PRIMARY KEY,
        job_id TEXT REFERENCES jobs(job_id) UNIQUE,
        dc_number TEXT,
        date_of_issue TEXT,
        expiry_date TEXT,
        amount TEXT,
        currency TEXT,
        applicant TEXT,
        beneficiary TEXT,
        issuing_bank TEXT,
        consolidated_fields JSONB DEFAULT '{}',
        clauses JSONB DEFAULT '[]',
        required_documents JSONB DEFAULT '[]',
        amendment_log JSONB DEFAULT '[]',
        created_at TIMESTAMP DEFAULT NOW()
    );

    CREATE TABLE IF NOT EXISTS verification_rows (
        id SERIAL PRIMARY KEY,
        job_id TEXT REFERENCES jobs(job_id),
        clause_ref TEXT NOT NULL,
        clause_text TEXT,
        condition_text TEXT,
        document_checked TEXT,
        found_text TEXT,
        result TEXT,
        compliance TEXT,  -- 'pass', 'fail', 'review', 'informational'
        confidence REAL DEFAULT 0.0,
        review_reason TEXT,
        dependency_note TEXT,
        source_page INTEGER,
        created_at TIMESTAMP DEFAULT NOW()
    );

    CREATE TABLE IF NOT EXISTS human_reviews (
        id SERIAL PRIMARY KEY,
        job_id TEXT REFERENCES jobs(job_id),
        action TEXT NOT NULL,  -- 'approve', 'reclassify', 'mark_alien', 'unmark_alien'
        packet_id TEXT,
        old_value TEXT,
        new_value TEXT,
        reviewer TEXT DEFAULT 'officer',
        reviewed_at TIMESTAMP DEFAULT NOW()
    );

    CREATE TABLE IF NOT EXISTS sanctions (
        id SERIAL PRIMARY KEY,
        category TEXT NOT NULL,  -- 'banks', 'countries', 'goods'
        value TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT NOW()
    );

    CREATE INDEX IF NOT EXISTS idx_pages_job ON pages(job_id);
    CREATE INDEX IF NOT EXISTS idx_packets_job ON packets(job_id);
    CREATE INDEX IF NOT EXISTS idx_verification_job ON verification_rows(job_id);
    CREATE INDEX IF NOT EXISTS idx_reviews_job ON human_reviews(job_id);
    CREATE INDEX IF NOT EXISTS idx_sanctions_cat ON sanctions(category);
    """)

    cur.close()
    conn.close()
    print("[DB] Database trade_finance_pilot initialized")


if __name__ == '__main__':
    init_database()
