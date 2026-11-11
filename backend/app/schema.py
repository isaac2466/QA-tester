"""Cambios a la base de datos que se aplican solos al arrancar (y con crear_bd.py).

Así quien ya tenía la base creada no tiene que borrar nada: sólo se agrega lo que falta.
"""
from app.db import connection

PROJECTS = """
CREATE TABLE IF NOT EXISTS projects (
    id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    user_id BIGINT UNSIGNED NOT NULL,
    name VARCHAR(120) NOT NULL,
    url VARCHAR(2048) NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id),
    KEY idx_projects_user (user_id),
    CONSTRAINT fk_projects_user FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
) ENGINE=InnoDB
"""

# columna → cómo agregarla a query_history si no existe
HISTORY_COLUMNS = {
    "project_id": "ADD COLUMN project_id BIGINT UNSIGNED NULL AFTER user_id, "
                  "ADD KEY idx_history_project (project_id, created_at), "
                  "ADD CONSTRAINT fk_history_project FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE SET NULL",
    "viewport": "ADD COLUMN viewport VARCHAR(20) NULL AFTER query_type",
    "source_url": "ADD COLUMN source_url VARCHAR(2048) NULL AFTER viewport",
}


def ensure_schema() -> list[str]:
    """Crea la tabla de proyectos y agrega columnas nuevas al historial. Devuelve lo que cambió."""
    changes = []
    with connection() as conn:
        cur = conn.cursor()
        cur.execute("SHOW TABLES LIKE 'projects'")
        if not cur.fetchone():
            cur.execute(PROJECTS)
            changes.append("tabla projects")
        cur.execute("SELECT COLUMN_NAME FROM information_schema.COLUMNS "
                    "WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'query_history'")
        have = {r[0] for r in cur.fetchall()}
        for col, ddl in HISTORY_COLUMNS.items():
            if col not in have:
                cur.execute(f"ALTER TABLE query_history {ddl}")
                changes.append(f"query_history.{col}")
        cur.close()
    return changes
