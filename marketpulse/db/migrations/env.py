# marketpulse/db/migrations/env.py
# Alembic environment script — runs on every 'alembic' command.
# Connects to the database, sets target_metadata, and executes migrations.

from __future__ import annotations

import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

# Import Base so Alembic can read our model definitions for --autogenerate
from marketpulse.db.models import Base

# Alembic Config object — access values from alembic.ini
config = context.config

# Set up logging from alembic.ini [loggers] section (if config file is set)
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# target_metadata: what Alembic compares against when autogenerating migrations.
# Must point to the MetaData object that contains ALL table definitions.
target_metadata = Base.metadata

# Override the database URL from environment — takes precedence over alembic.ini.
# This allows CI, test, and production to all use different databases
# without changing any committed file.
_db_url = os.environ.get("DATABASE_URL")
if _db_url:
    config.set_main_option("sqlalchemy.url", _db_url)


def run_migrations_offline() -> None:
    """
    Run migrations in 'offline' mode.

    Generates SQL statements without connecting to the database.
    Useful for production: a DBA can review the SQL before applying it.
    Run with: alembic upgrade head --sql
    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        # Include schemas and column type changes in autogenerate
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """
    Run migrations in 'online' mode.

    Connects to the live database and applies pending migrations.
    This is the default mode for: alembic upgrade head
    """
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        # NullPool: no persistent connection pool — migration tools connect once
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,   # detect column type changes in autogenerate
        )
        with context.begin_transaction():
            context.run_migrations()


# Alembic calls this script and checks which mode to run
if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
