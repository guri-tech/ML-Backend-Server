import os, sys
import pandas as pd
import sqlalchemy
from sqlalchemy import create_engine

BASE_DIR = os.path.dirname(__file__)
filename = 'hotel_train.csv'
file_loc = os.path.join(f'{BASE_DIR}/../data/', filename)
df = pd.read_csv(file_loc)

# 타켓 컬럼을 맨 끝으로 보내고 나머지 컬럼은 컬럼명 순으로 재정렬
colnames = list(df.columns)
colnames.sort()
colnames.remove('is_canceled')
x_columns = colnames
x = df[x_columns]
y = df['is_canceled'].to_frame()

new_columns = x_columns.copy()
new_columns.append('is_canceled')
df_columns = new_columns
df_new = df[df_columns]
df_new.to_csv(f'{BASE_DIR}/../data/hotel_train_sort_columns.csv', index=None)


url = "postgresql://postgres:postgres@localhost:5432/hoteldb"
engine = create_engine(url)

df_new.to_sql(name ='hoteltb',
          con  = engine,
          if_exists = 'append',
          index = False)
