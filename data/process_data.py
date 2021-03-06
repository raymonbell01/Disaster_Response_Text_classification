import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import matplotlib.pyplot as plt


def load_data(messages_filepath, categories_filepath):
    """
    This function load the csv file for categories and messages, merged the data and create a proper header for the resulting dataframe
    
    arg: 
    messages_filepath: The file path for the messages csv
    categories_filepath: The file path for the categories csv
    
    result:
    df :  dataframe with proper label and column header
    """
    
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    
    # Merge categories and messages data set on 'id'
    df = pd.merge(messages,categories, on='id')
    
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(pat =";",n=36,expand=True)
    
    # select the first row of the categories dataframe
    row = categories.iloc[0]

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = list(row.str.split(pat = "-",expand =True)[0])
    
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = [x[-1] for x in categories[column]]
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
    
    # drop the original categories column from `df`
    df.drop('categories', axis=1, inplace=True)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories],axis=1)
    df.head()
    
    #Change values of related to 1 where it's erronously inputed as 2
    df['related'] = df['related'].replace(2,1)
    
    return df


def clean_data(df):
    """
    This function cleaned the dataset created from load dataset function. It looks for deuplicate values and correct annomalies in the          dataframe

    arg:
    df: dataframe from the load_data set function

    result:
    df: cleaned dataframe
    """
    # check number of duplicates
    print("#number of duplicate of dataset# {}".format(print(df[df.duplicated(subset = 'message')].shape)))
    # drop duplicates
    df = df.drop_duplicates(subset = 'message')
    # check number of duplicates
    df[df.duplicated(subset = 'message')].shape

    #child alone also has just one variable meaning, none of the message is related to child alone. We are dropping this        column.
    #we are dropiing original and id column because the are not useful in our model
    df = df.drop(['child_alone','original','id'], axis =1)
    
    return df


def save_data(df, database_filename):
    """
    This funtion save the dataframe returned by clean_data funtion into an sql datafrma: complete ETL
    
    arg:
    df: Cleaned dataframe
    database_filename: file path to save the dataframe
    
    """
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('messages_category', engine, if_exists = 'replace', index=False)


def main():
    """
    This funtion calls all other funtions that loads our data, clean and save our data to database
    """
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()