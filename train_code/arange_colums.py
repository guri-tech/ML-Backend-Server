import pandas as pd
import os

BASE_DIR = os.path.dirname(__file__)

def arangeColumns(df, target): 
    '''
    데이터프레임과 타겟컬럼명을 입력받아
    컬럼을 알파벳순으로 정렬하고, 타겟 컬럼은 맨 뒤로 보낸 뒤 
    csv 파일로 저장
    df, x, y 데이터프레임으로 반환
    
    '''

    colnames = list(df.columns)
    colnames.sort()
    colnames.remove(target)
    x_columns = colnames
    x = df[x_columns]
    y = df[target].to_frame()

    new_columns = x_columns.copy()
    new_columns.append(target)
    df_columns = new_columns
    df_new = df[df_columns]
    df_new.to_csv(f'{BASE_DIR}/../data/hotel_2_sort_columns.csv', index=None)

    return df_new, x, y
