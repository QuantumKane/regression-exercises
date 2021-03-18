
# imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from env import host, user, password

def get_connection(db, user=user, host=host, password=password):
    '''
    This function uses my info from my env file to
    create a connection url to access the Codeup db.
    '''
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'
    

def get_telco_data():
    '''
    This function uses the SQL query from below and specifies the database to use
    '''
    # SQL query that joins all of the tables together from the 'telco_churn' database     
    sql_query = """
                SELECT customer_id, monthly_charges, tenure, total_charges 
                FROM customers 
                WHERE contract_type_id = 3;               
                """
    return pd.read_sql(sql_query,get_connection('telco_churn'))


def split(df, stratify_by=None):
    """
    Train, validate, test split
    To stratify, send in a column name
    """
    
    if stratify_by == None:
        train, test = train_test_split(df, test_size=.2, random_state=123)
        train, validate = train_test_split(train, test_size=.3, random_state=123)
    else:
        train, test = train_test_split(df, test_size=.2, random_state=123, stratify=df[stratify_by])
        train, validate = train_test_split(train, test_size=.3, random_state=123, stratify=train[stratify_by])
    return train, validate, test


def clean_telco(df):
    '''
    clean the two year contract data down to 
    monthly charges, total charges, tenure, and customer id,
    replace whitespace values in total charges with zeros where appropriate
    '''
    df = df[['customer_id', 'monthly_charges', 'tenure', 'total_charges']]
    df['total_charges'] = df['total_charges'].replace(r'^\s*$', np.nan, regex=True)
    df = df.fillna(0)
    df['total_charges'] = df['total_charges'].astype('float')
    return df
    

def wrangle_telco():
    '''
    wrangle_telco will read in our telco data for two year contract customers via a sql query, clean the data down to 
    monthly charges, total charges, tenure, and customer id,
    replace whitespace values in total charges with zeros where appropriate
    and then split the data
    
    return: train, validate, and test sets of telco data
    '''
    df = clean_telco(get_telco_data())
    return split(df)

def get_zillow_data():
    '''
    This function uses the SQL query from below and specifies the database to use
    '''
    # SQL query that joins all of the tables together from the 'telco_churn' database     
    sql_query = """
                SELECT * 
                FROM properties_2017
                JOIN predictions_2017 USING(parcelid)
                WHERE transactiondate BETWEEN "2017-05-01" AND "2017-06-30"
                AND propertylandusetypeid = 261
                LIMIT 5000;
                """
    return pd.read_sql(sql_query,get_connection('zillow'))


def clean_zillow(df):
    '''
    This function takes in a dataframe, and performs the following:
    - renames columns to make them understandable
    - sets parcelid as the index
    - drops null/NAN rows
    - removes outliers from tax_value and square_feet
    '''
    df = df.rename(columns={"bedroomcnt": "bedrooms", "bathroomcnt": "bathrooms", "calculatedfinishedsquarefeet": "square_feet", 
                        "taxamount": "taxes", "regionidzip": "zip_code", "taxvaluedollarcnt": "tax_value", 
                        "yearbuilt": "year_built", "regionidcounty": "county"})
    df = df.set_index("parcelid")
    df = df.dropna()
    
    upper_bound, lower_bound = remove_outlier(df, "tax_value")
    df = df[df.tax_value < upper_bound]
    
    upper_bound, lower_bound = remove_outlier(df, "square_feet")
    df = df[df.square_feet < upper_bound]
    
    return df


def remove_outlier(df, feature):
    '''
    This function takes in a dataframe's features and performs the following:
    - calculates its 1st and 3rd quartiles
    - uses their diference to calculate the IQR
    - uses the IQR to determine upper and lower bounds
    '''
    
    q1 = df[feature].quantile(.25)
    q3 = df[feature].quantile(.75)
    
    iqr = q3 - q1
    
    upper_bound = q3 + (1.5 * iqr)
    lower_bound = q1 - (1.5 * iqr)
    
    return upper_bound, lower_bound