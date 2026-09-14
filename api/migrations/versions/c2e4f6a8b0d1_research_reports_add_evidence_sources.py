"""research_reports 增加内部证据来源

Revision ID: c2e4f6a8b0d1
Revises: a14c9f0e3b21
Create Date: 2026-09-15 00:00:00.000000
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "c2e4f6a8b0d1"
down_revision: Union[str, None] = "a14c9f0e3b21"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "research_reports",
        sa.Column("evidence_sources", postgresql.JSONB(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("research_reports", "evidence_sources")
