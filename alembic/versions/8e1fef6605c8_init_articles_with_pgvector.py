"""init articles with pgvector

Revision ID: 8e1fef6605c8
Revises: 
Create Date: 2025-08-27 11:13:14.459017

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '8e1fef6605c8'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade():
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute("""
        CREATE TABLE IF NOT EXISTS articles (
          id UUID PRIMARY KEY,
          title TEXT NOT NULL,
          content TEXT NOT NULL,
          url TEXT UNIQUE,
          source VARCHAR(120),
          published_at TIMESTAMPTZ,
          lang VARCHAR(5),
          embedding VECTOR(384),
          created_at TIMESTAMPTZ DEFAULT NOW(),
          updated_at TIMESTAMPTZ
        )
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS articles_embedding_hnsw
        ON articles USING hnsw (embedding vector_l2_ops)
    """)
    
def downgrade():
    op.execute("DROP TABLE IF EXISTS articles")
    op.execute("DROP EXTENSION IF EXISTS vector")