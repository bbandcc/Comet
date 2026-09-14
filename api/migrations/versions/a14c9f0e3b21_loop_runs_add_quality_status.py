"""loop_runs 增加独立质量状态

Revision ID: a14c9f0e3b21
Revises: 6727223d45f9
Create Date: 2026-09-14 00:00:00.000000
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "a14c9f0e3b21"
down_revision: Union[str, None] = "6727223d45f9"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "loop_runs",
        sa.Column("quality_status", sa.String(length=32), nullable=True),
    )
    # 旧 status 无法可靠推导质量状态：旧 passed 包含 Judge 异常 fail-open，
    # 旧 failed 也可能是非 Judge 的执行错误。历史记录保留 NULL 表示未知。


def downgrade() -> None:
    op.drop_column("loop_runs", "quality_status")
