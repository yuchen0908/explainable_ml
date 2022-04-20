from sklearn.preprocessing import OrdinalEncoder
import pandas as pd
import numpy as np

def process_bike(file_dir):
    """
    :Args:
    - file_dir (string), the file directory of bike dataset
    :Returns:
    - cols_df (Dataframe), column details of processed dataframe
    - bike_df (Dataframe), the processed dataframe
    """
    bike_df = pd.read_csv(file_dir)
    if bike_df.yr.dtype != pd.StringDtype:
        bike_df['yr'] = bike_df.yr.astype(str)
        print('convert yr to string')
    else:
        print("yr is converted alredy")

    # converting categorical variables into numerical ones
    encode_map = {'season':['SPRING','SUMMER','FALL','WINTER']
        , 'yr':['2011','2012']
        , 'mnth':['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC']
        , 'holiday':['NO HOLIDAY','HOLIDAY']
        , 'weekday':['MON','TUE','WED','THU','FRI','SAT','SUN']
        , 'workingday':['NO WORKING DAY','WORKING DAY']
        , 'weathersit':['MISTY','GOOD','RAIN/SNOW/STORM']
    }

    for k in encode_map.keys():
        bike_df[f'{k}_encoded'] = OrdinalEncoder(categories=[encode_map.get(k)]).fit_transform(np.asarray(bike_df[k]).reshape(-1,1)).reshape(-1)
    
    # set up blueprint
    cols_df = pd.DataFrame(bike_df.dtypes)
    cols_df.reset_index(drop=False, inplace=True)
    cols_df.rename(columns={"index":"cols",0:'data_types'}, inplace=True)
    cols_df['input'] = cols_df.data_types.apply(lambda x: "cat" if x == 'object' else "num")
    cols_df.loc[cols_df.cols == 'cnt', 'input'] = 'resp'

    return cols_df, bike_df
    