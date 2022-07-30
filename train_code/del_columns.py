import pandas as pd
import os

BASE_DIR = os.path.dirname(__file__)

def deleteColumns(df, col_names):
    '''
    데이터 프레임과 삭제할 컬럼을 입력받아 

    '''    
    # 결측치 확인
    # print( df.isna().sum() )

    df = df.drop(col_names, axis=1)

    # csv로 저장
    df.to_csv(f'{BASE_DIR}/../data/hotel_3_removeColumns.csv', index=None)

    return df


