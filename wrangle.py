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
def get_connection(database, user=user, host=host, password=password):
    '''get URL with user, host, and password from env '''
    
    return f"mysql+pymysql://{user}:{password}@{host}/{database}"
    
    
def cache_sql_data(df, database):
    '''write dataframe to csv with title database_query.csv'''
    
    df.to_csv(f'{database}_query.csv',index = False)

def get_sql_data(database,query):
    ''' check if csv exists for the queried database
        if it does read from the csv
        if it does not create the csv then read from the csv  
    '''
    if os.path.isfile(f'{database}_query.csv') == False:   # check for the file
        df = pd.read_sql(query, get_connection(database))  # create file 
        cache_sql_data(df, database) # cache file 
    return pd.read_csv(f'{database}_query.csv') # return contents of file
def get_zillow_data():
    ''' acquire zillow data'''
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
    database = "zillow"
    # create/read csv for query
    df = get_sql_data(database,query) 
    # drop duplicate parcelids keeping the latest
    df = df.sort_values('transactiondate').drop_duplicates('parcelid',keep='last') 
    return df 

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