import pandas as pd
import os

def loadCSV(filename):
    '''
    파일명을 입력받으면 data 폴더에서 csv 파일을 찾아 데이터프레임으로 반환
    '''
    try:
        BASE_DIR = os.path.dirname(__file__)
        file_loc = os.path.join(f'{BASE_DIR}/../data/', filename)
        df = pd.read_csv(file_loc)
        return df
    except Exception as e:
        print(e)
