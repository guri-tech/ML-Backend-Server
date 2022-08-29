import os
import pandas as pd
from sqlalchemy import create_engine


def get_db():

    engine = create_engine(os.getenv("DATABASE_URL"))

    sql = "SELECT * FROM hoteltb;"
    df = pd.read_sql(sql, con=engine)

    return df
