import os
import pandas as pd
from sqlalchemy import create_engine

BASE_DIR = os.path.dirname(__file__)
filename = "bookings.csv"
file_loc = os.path.join(f"{BASE_DIR}/../data/", filename)
df = pd.read_csv(file_loc)

engine = create_engine(os.getenv("DATABASE_URL"))
df.to_sql(name="bookingtb", con=engine, if_exists="append", index=False)
