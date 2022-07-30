import os, sys
import pandas as pd
import sqlalchemy
from sqlalchemy import create_engine

BASE_DIR = os.path.dirname(__file__)
filename = 'hotel_2_sort_columns.csv'
file_loc = os.path.join(f'{BASE_DIR}/../data/', filename)
df = pd.read_csv(file_loc)

url = "postgresql://postgres:postgres@localhost:5432/hoteldb"
engine = create_engine(url)

df.to_sql(name ='hoteltb',
          con  = engine,
          if_exists = 'append',
          index = False)
