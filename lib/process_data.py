from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler
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

def process_cancer(file_dir):
    """
    :Args:
        - file_dir (string), the file directory of cancer dataset
    :Returns:
        - cols_cat (dict), the categories of columns from cancer dataset
        - cancer_df (dataframe), the pre-processed cancer data for modelling
    """
    cancer_df = pd.read_csv(file_dir)
    # replace column name 
    cancer_df.columns = ['age','num_of_sex_partners','first_sex','num_of_pregn'\
        ,'smokes','smokes_yrs','hormonal_contrpt','hormonal_contrpt_yrs','iud'\
        ,'iud_yrs','stds','stds_num','stds_num_diagn','stds_time_since_1diagn'\
        ,'stds_time_since_ldiagn','biopsy']
    
    # organise our columns into different category
    cols_cat = {
        'num':['age','num_of_sex_partners','first_sex','num_of_pregn','smokes_yrs','hormonal_contrpt_yrs','iud_yrs','stds_time_since_1diagn','stds_time_since_ldiagn']\
        ,'ordinal':['smokes','hormonal_contrpt','iud','stds','stds_num','stds_num_diagn']\
        ,'cat':[]
        ,'resp':['resp']}
    # add response 
    cancer_df['resp'] = cancer_df['biopsy'].apply(lambda x: 1 if x=='Cancer' else 0)
    # standarise 
    std_cols = [c + "_std" for c in cols_cat.get('num')]
    x_scaler = MinMaxScaler().fit(cancer_df[cols_cat.get('num')].to_numpy())
    scaler_df = pd.DataFrame(x_scaler.transform(cancer_df[cols_cat.get('num')].to_numpy()), columns = std_cols)
    cols_cat['num_std'] = std_cols
    cancer_df = pd.concat([cancer_df, scaler_df], axis=1)
    return cols_cat, cancer_df
