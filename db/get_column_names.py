import os
import pandas as pd
from sqlalchemy import create_engine


def get_column_names():

    engine = create_engine(os.getenv("DATABASE_URL"))

    sql = "select column_name from information_schema.columns where table_catalog = 'hoteldb' and table_name = 'hoteltb';"
    df = pd.read_sql(sql, con=engine)
    column_names = df["column_name"].values.tolist()
    return column_names
