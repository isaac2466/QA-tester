import hashlib
import hmac
import secrets
from datetime import datetime, timedelta, timezone

import mysql.connector
from mysql.connector import Error

from app.core.config import MYSQL_CONFIG


def _connect():
    return mysql.connector.connect(**MYSQL_CONFIG)


def initialize_database() -> None:
    try:
        connection = _connect()
        cursor = connection.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
                email VARCHAR(254) NOT NULL UNIQUE,
                password_hash VARCHAR(255) NOT NULL,
                created_at DATETIME(6) NOT NULL
            ) ENGINE=InnoDB
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                token_hash CHAR(64) PRIMARY KEY,
                user_id INT UNSIGNED NOT NULL,
                expires_at DATETIME(6) NOT NULL,
                CONSTRAINT fk_sessions_user
                    FOREIGN KEY (user_id) REFERENCES users(id)
                    ON DELETE CASCADE
            ) ENGINE=InnoDB
            """
        )
        connection.commit()
    except Error as error:
        raise RuntimeError(
            "No se pudo conectar a MySQL. Revisa backend/.env y que MySQL Server esté iniciado."
        ) from error
    finally:
        if "cursor" in locals():
            cursor.close()
        if "connection" in locals() and connection.is_connected():
            connection.close()


def create_user(email: str, password: str) -> dict:
    salt = secrets.token_bytes(16)
    password_hash = _hash_password(password, salt)
    created_at = datetime.now(timezone.utc).replace(tzinfo=None)
    connection = _connect()
    try:
        cursor = connection.cursor()
        cursor.execute(
            "INSERT INTO users (email, password_hash, created_at) VALUES (%s, %s, %s)",
            (email, f"{salt.hex()}:{password_hash}", created_at),
        )
        connection.commit()
        return {"id": cursor.lastrowid, "email": email, "created_at": created_at.isoformat()}
    finally:
        cursor.close()
        connection.close()


def find_user_by_email(email: str) -> dict | None:
    connection = _connect()
    try:
        cursor = connection.cursor(dictionary=True)
        cursor.execute(
            "SELECT id, email, password_hash, created_at FROM users WHERE email = %s",
            (email,),
        )
        return cursor.fetchone()
    finally:
        cursor.close()
        connection.close()


def verify_password(password: str, stored_hash: str) -> bool:
    salt_hex, password_hash = stored_hash.split(":", maxsplit=1)
    candidate = _hash_password(password, bytes.fromhex(salt_hex))
    return hmac.compare_digest(candidate, password_hash)


def create_session(user_id: int) -> tuple[str, datetime]:
    token = secrets.token_urlsafe(32)
    expires_at = datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(days=7)
    connection = _connect()
    try:
        cursor = connection.cursor()
        cursor.execute(
            "INSERT INTO sessions (token_hash, user_id, expires_at) VALUES (%s, %s, %s)",
            (_token_hash(token), user_id, expires_at),
        )
        connection.commit()
        return token, expires_at.replace(tzinfo=timezone.utc)
    finally:
        cursor.close()
        connection.close()


def get_user_by_token(token: str) -> dict | None:
    connection = _connect()
    try:
        cursor = connection.cursor(dictionary=True)
        cursor.execute(
            """
            SELECT users.id, users.email, users.created_at, sessions.expires_at
            FROM sessions JOIN users ON users.id = sessions.user_id
            WHERE sessions.token_hash = %s
            """,
            (_token_hash(token),),
        )
        row = cursor.fetchone()
    finally:
        cursor.close()
        connection.close()
    if not row:
        return None
    expires_at = row["expires_at"].replace(tzinfo=timezone.utc)
    if expires_at <= datetime.now(timezone.utc):
        return None
    return row


def delete_session(token: str) -> None:
    connection = _connect()
    try:
        cursor = connection.cursor()
        cursor.execute("DELETE FROM sessions WHERE token_hash = %s", (_token_hash(token),))
        connection.commit()
    finally:
        cursor.close()
        connection.close()


def _hash_password(password: str, salt: bytes) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 310_000).hex()


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()
