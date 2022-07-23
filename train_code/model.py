import pandas as pd
from arange_colums import arangeColumns
from del_columns import deleteColumns
from load_csv import loadCSV
from save_model import saveModel
from train import trianModel
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
import joblib

df = loadCSV('hotel_bookings.csv')
target = 'is_canceled'
df, x, y = arangeColumns(df, target)

# 결측치가 많거나 불필요해 보이는 컬럼 삭제
del_col_names = ['agent','company','country','reservation_status_date','reservation_status', ]
x = deleteColumns(x, del_col_names)

# 수치형 변수: SimpleImputer 결측치는 중앙값으로 대치 & 정규화
numeric_features = ['adr','adults','arrival_date_day_of_month','arrival_date_week_number',
                     'arrival_date_year','babies','booking_changes','children',
                     'days_in_waiting_list','is_repeated_guest','lead_time',
                     'previous_bookings_not_canceled','previous_cancellations',
                     'required_car_parking_spaces','stays_in_week_nights',
                     'stays_in_weekend_nights','total_of_special_requests']

# 범주형 변수: 원핫인코딩
categorical_features = ['arrival_date_month','assigned_room_type','customer_type',
                        'deposit_type','distribution_channel','hotel','market_segment',
                        'meal','reserved_room_type']

# xgb = XGBClassifier(booster='gbtree', eval_metric='logloss', use_label_encoder=False,
#                    max_depth=6, learning_rate=0.1, n_estimators=100, n_jobs=-1)
xgb = XGBClassifier()
clf = trianModel(xgb, x, y, numeric_features, categorical_features)
preint(x.columns)
# logi = LogisticRegression()
# clf = trianModel(logi, x, y, numeric_features, categorical_features)


# 모델 저장
# saveModel(clf, 'xgb.pkl')
# saveModel(clf, 'logistic.pkl')