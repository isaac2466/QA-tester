"""Crea la base de datos qa_tester y sus tablas (usuarios, sesiones, historial, proyectos).

    cd backend
    python crear_bd.py

Usa los datos de MySQL de backend/.env (copia .env.example como .env y pon tu contraseña).
Se puede correr varias veces: no borra nada que ya exista.
"""
from pathlib import Path

import mysql.connector

from app.core.config import MYSQL_CONFIG

SQL = Path(__file__).resolve().parents[1] / "data" / "qa_tester.sql"

cfg = {k: v for k, v in MYSQL_CONFIG.items() if k != "database"}  # la base todavía puede no existir
try:
    conn = mysql.connector.connect(**cfg)
except mysql.connector.Error as e:
    raise SystemExit(f"No se pudo conectar a MySQL como '{cfg['user']}': {e.msg}\n"
                     "Revisa MYSQL_USER y MYSQL_PASSWORD en backend/.env")

cur = conn.cursor()
for stmt in SQL.read_text(encoding="utf-8").split(";"):
    if stmt.strip():
        cur.execute(stmt)
conn.commit()
from app.schema import ensure_schema  # noqa: E402  (proyectos y columnas nuevas del historial)

changes = ensure_schema()
if changes:
    print("Se agregó:", ", ".join(changes))
cur.execute(f"SHOW TABLES FROM {MYSQL_CONFIG['database']}")
print("Base de datos lista. Tablas:", ", ".join(t[0] for t in cur.fetchall()))
conn.close()
