import os
import psycopg2
from psycopg2.extras import RealDictCursor
from contextlib import contextmanager

DB_CONFIG = {
    "host":            os.getenv("DB_HOST",     "localhost"),
    "database":        os.getenv("DB_NAME",     "agent"),
    "user":            os.getenv("DB_USER",     "agent_user"),
    "password":        os.getenv("DB_PASSWORD", "agent_pass"),
    "connect_timeout": 5,
}


@contextmanager
def get_conn():
    conn = psycopg2.connect(**DB_CONFIG)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def check_db():
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
