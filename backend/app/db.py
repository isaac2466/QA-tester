from contextlib import contextmanager

import mysql.connector
from mysql.connector import Error

from app.core.config import MYSQL_CONFIG


@contextmanager
def connection():
    conn = mysql.connector.connect(**MYSQL_CONFIG)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def fetch_one(query: str, params: tuple = ()):
    with connection() as conn:
        cursor = conn.cursor(dictionary=True)
        try:
            cursor.execute(query, params)
            return cursor.fetchone()
        finally:
            cursor.close()


def fetch_all(query: str, params: tuple = ()):
    with connection() as conn:
        cursor = conn.cursor(dictionary=True)
        try:
            cursor.execute(query, params)
            return cursor.fetchall()
        finally:
            cursor.close()


def execute(query: str, params: tuple = ()) -> int:
    with connection() as conn:
        cursor = conn.cursor()
        try:
            cursor.execute(query, params)
            return cursor.lastrowid
        finally:
            cursor.close()


def is_database_error(error: Exception) -> bool:
    return isinstance(error, Error)
