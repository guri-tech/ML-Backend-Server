import os
import pandas as pd
from sqlalchemy import create_engine

BASE_DIR = os.path.dirname(__file__)
filename = "bookings.csv"
file_loc = os.path.join(f"{BASE_DIR}/../data/", filename)
df = pd.read_csv(file_loc)

# df_train = df.iloc[:90000, :]  # 90,000건 학습용으로 db에 입력
# df_test = df.iloc[90001:, :]  # 29389건 재학습용으로 csv파일에 저장

# df_train.to_csv(f"{BASE_DIR}/../data/bookings_train.csv", index=None)
# df_test.to_csv(f"{BASE_DIR}/../data/bookings_test.csv", index=None)

engine = create_engine(os.getenv("DATABASE_URL"))
df_train.to_sql(name="bookingtb", con=engine, if_exists="append", index=False)
