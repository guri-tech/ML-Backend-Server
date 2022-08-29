import os
from sqlalchemy import create_engine


def insert_db(df):
    df["is_canceled"] = None
    engine = create_engine(os.getenv("DATABASE_URL"))

    df.to_sql(name="bookingtb", con=engine, if_exists="append", index=False)

    return "Success"
