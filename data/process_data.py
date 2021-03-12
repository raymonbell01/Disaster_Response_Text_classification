import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import matplotlib.pyplot as plt


def load_data(messages_filepath, categories_filepath):
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
    
    return df


def clean_data(df):
    # check number of duplicates
    print("#shape of dataset# "{}.format(print(df[df.duplicated(subset = 'message')].shape))
    # drop duplicates
    df = df.drop_duplicates(subset = 'message')
    # check number of duplicates
    df[df.duplicated(subset = 'message')].shape
    
    # create plot to show each column
    #df.hist(layout=(8,5), figsize=(30,40))
    #plt.show()
    
    #At quick glance we could see some row in request column was categorise as 2. We prefer to drop this row
    df = df[df['related']!=2]

    #child alone also has just one variable meaning, none of the message is related to child alone. We are dropping this        column.
    #we are dropiing original and id column because the are not useful in our model
    df.drop(['child_alone','original','id'], inplace = True, axis =1)
    
    return df


def save_data(df, database_filename):
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('messages_category', engine, if_exists = 'replace', index=False)


def main():
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