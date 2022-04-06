# imports
import pandas as pd
import numpy as np
import os
import scipy.stats as stats
from pydataset import data

# acquire
from env import host, user, password

# visualize
import seaborn as sns
import matplotlib.pyplot as plt

# import splitting and imputing functions
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import sklearn.preprocessing


# Acquire data

def get_zillow17_data(use_cache=True):
    filename = "zillow.csv"
    if os.path.isfile(filename) and use_cache:
        print("Let me get that for you...")
        return pd.read_csv(filename)
    print("Sorry, nothing on file, let me create one for you...")
    data = 'zillow'
    url = f'mysql+pymysql://{user}:{password}@{host}/{data}'
    query = '''
    SELECT prop.*,
           pred.logerror,
           pred.transactiondate,
           air.airconditioningdesc,
           arch.architecturalstyledesc,
           build.buildingclassdesc,
           heat.heatingorsystemdesc,
           landuse.propertylandusedesc,
           story.storydesc,
           construct.typeconstructiondesc
    FROM   properties_2017 prop
           INNER JOIN (SELECT parcelid,
                       Max(transactiondate) transactiondate
                       FROM   predictions_2017
                       GROUP  BY parcelid) pred
                   USING (parcelid)
                            JOIN predictions_2017 as pred USING (parcelid, transactiondate)
           LEFT JOIN airconditioningtype air USING (airconditioningtypeid)
           LEFT JOIN architecturalstyletype arch USING (architecturalstyletypeid)
           LEFT JOIN buildingclasstype build USING (buildingclasstypeid)
           LEFT JOIN heatingorsystemtype heat USING (heatingorsystemtypeid)
           LEFT JOIN propertylandusetype landuse USING (propertylandusetypeid)
           LEFT JOIN storytype story USING (storytypeid)
           LEFT JOIN typeconstructiontype construct USING (typeconstructiontypeid)
    WHERE  prop.latitude IS NOT NULL
           AND prop.longitude IS NOT NULL
    '''
    zillow17_data = pd.read_sql(query, url)
    zillow17_data.to_csv(filename)
    return zillow17_data

# prep data
def remove_outliers(df, k, col_list):
    ''' remove outliers from a list of columns in a dataframe 
        and return that dataframe
    '''
    for col in col_list:
        q1, q3 = df[f'{col}'].quantile([.25, .75])  # get quartiles
        iqr = q3 - q1   # calculate interquartile range
        upper_bound = q3 + k * iqr   # get upper bound
        lower_bound = q1 - k * iqr   # get lower bound
        # return dataframe without outliers
        return df[(df[f'{col}'] > lower_bound) & (df[f'{col}'] < upper_bound)]  
def handle_missing_values(df, prop_required_column = .5, prop_required_row = .5):
    ''' 
        take in a dataframe and a proportion for columns and rows
        return dataframe with columns and rows not meeting proportions dropped
    '''
    col_thresh = int(round(prop_required_column*df.shape[0],0)) # calc column threshold
    df.dropna(axis=1, thresh=col_thresh, inplace=True) # drop columns with non-nulls less than threshold
    row_thresh = int(round(prop_required_row*df.shape[1],0))  # calc row threshhold
    df.dropna(axis=0, thresh=row_thresh, inplace=True) # drop columns with non-nulls less than threshold
    return df    
def impute(df, my_strategy, column_list):
    ''' take in a df strategy and cloumn list
        return df with listed columns imputed using input stratagy
    '''
    imputer = SimpleImputer(strategy=my_strategy)  # build imputer
    df[column_list] = imputer.fit_transform(df[column_list]) # fit/transform selected columns
    return df
def prepare_zillow(df):
    ''' Prepare Zillow Data'''
    # Restrict propertylandusedesc to those of single unit
    df = df[(df.propertylandusedesc == 'Single Family Residential') |
          (df.propertylandusedesc == 'Mobile Home') |
          (df.propertylandusedesc == 'Manufactured, Modular, Prefabricated Homes') |
          (df.propertylandusedesc == 'Townhouse') |
          (df.propertylandusedesc == 'Cluster Home')
          |(df.propertylandusedesc == 'Condominium') 
          ]
    # remove outliers in bed count, bath count, and area to better target single unit properties
    df = remove_outliers(df, 1.5, ['calculatedfinishedsquarefeet', 'bedroomcnt', 'bathroomcnt'])
    # dropping cols/rows where more than half of the values are null
    df = handle_missing_values(df, prop_required_column = .5, prop_required_row = .5)
    # dropping the columns with 17K missing values too much to fill/impute/drop rows
    df = df.drop(columns=['heatingorsystemtypeid', 'buildingqualitytypeid', 'propertyzoningdesc', 'unitcnt', 'heatingorsystemdesc', 'finishedsquarefeet12', 'calculatedbathnbr', 'Unnamed: 0'])
    # imputing descreet columns with most frequent value
    df = impute(df, 'most_frequent', ['fullbathcnt', 'regionidcity', 'regionidzip', 'yearbuilt', 'censustractandblock'])
    # imputing continuous columns with median value
    df = impute(df, 'median', ['lotsizesquarefeet', 'structuretaxvaluedollarcnt', 'taxvaluedollarcnt', 'landtaxvaluedollarcnt', 'taxamount'])
    # Creating a new column that coverts fips code to county name '6111.0' = 'Ventura', '6037.0' = 'Los Angeles', '6059.0' = 'Orange'
    # convert fips to a string
    df['fips'] = df['fips'].astype(str)
    # create new column with county name
    df['county'] = df['fips'].replace({'6111.0': 'Ventura', '6037.0': 'Los Angeles', '6059.0': 'Orange'})
    # convert fips to a float
    df['fips'] = df['fips'].astype(float)
    # select only the properties with atleast 1 bedroom and 500 sqft
    df = df[(df.bedroomcnt > 0) & (df.calculatedfinishedsquarefeet > 500)]
    return df
# feature engineering
def feature_engineering(df):
    '''
    Feature engineering
    '''
    df['sqft_per_bed'] = df['calculatedfinishedsquarefeet'] / df['bedroomcnt']
    df['sqft_per_bath'] = df['calculatedfinishedsquarefeet'] / df['bathroomcnt']
    df['total_rooms'] = df['bedroomcnt'] + df['bathroomcnt']
    df['bed_bath_rooms_per_sqft'] = df['total_rooms'] / df['calculatedfinishedsquarefeet']
    df['age_in_years'] = 2017 - df['yearbuilt']
    df['taxrate'] = df['taxamount'] / df['taxvaluedollarcnt']
    df['dollars_per_sqft'] = df['taxvaluedollarcnt'] / df['calculatedfinishedsquarefeet']
    return df
# split data
def split_zillow(df, target):
    ''' split zillow data into training, validate, and test sets
        then splits for X(features) and y(target) 
    '''
    # split into 20% test and 80% training_validate
    train_validate, test = train_test_split(df, test_size=.2, random_state=42)
    # split training_validate into 70% train and 30% validation
    train, validate = train_test_split(train_validate, test_size=.3, random_state=42)
    # split train into X(features) and y(target)
    X_train, y_train = train.drop(columns=[target]), train[target]
    X_validate, y_validate = validate.drop(columns=[target]), validate[target]
    X_test, y_test = test.drop(columns=[target]), test[target]
    return train, validate, test, X_train, y_train, X_validate, y_validate, X_test, y_test