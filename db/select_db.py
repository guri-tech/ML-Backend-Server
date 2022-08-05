import pandas as pd
import sqlalchemy
from sqlalchemy import create_engine

def get_db():

    url = "postgresql://postgres:postgres@localhost:5432/hoteldb"
    engine = create_engine(url)

    sql = 'SELECT * FROM hoteltb;'
    df = pd.read_sql(sql, con=engine)

    return df
