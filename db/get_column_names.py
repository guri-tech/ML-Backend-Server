import pandas as pd
import sqlalchemy
from sqlalchemy import create_engine

def get_column_names():

    url = "postgresql://postgres:postgres@localhost:5432/hoteldb"
    engine = create_engine(url)

    sql = "select column_name from information_schema.columns where table_catalog = 'hoteldb' and table_name = 'hoteltb';"
    df = pd.read_sql(sql, con=engine)
    column_names = df['column_name'].values.tolist()
    return column_names
