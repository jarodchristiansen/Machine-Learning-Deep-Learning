"""empty message

Revision ID: 66f9bb6abef7
Revises: 80649f4a863f
Create Date: 2020-07-17 16:39:55.874309

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '66f9bb6abef7'
down_revision = '80649f4a863f'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_constraint(None, 'roles_users', type_='foreignkey')
    op.create_foreign_key(None, 'roles_users', 'users', ['user_id'], ['password_hash'])
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_constraint(None, 'roles_users', type_='foreignkey')
    op.create_foreign_key(None, 'roles_users', 'users', ['user_id'], ['id'])
    # ### end Alembic commands ###
