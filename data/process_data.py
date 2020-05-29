import os
import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    Loads and merges data frames with messages and information about categories.
    
        Parameters:
            messages_filepath (string): csv file with information about disaster messages
            categories_filepath (string): csv file with information about disaster categories
            
        Returns:
            df (data frame): merged pandas dataframe with disaster messages and categories for the messages 
    
    '''
    
    messages = pd.read_csv(messages_filepath) 
    categories = pd.read_csv(categories_filepath)
    
    df = pd.merge(messages, categories, on = "id")
    
    return df

def clean_data(df):
    '''
    Preprocesses the disaster response data for the machine-learning pipeline. 
        
        Parameters:
            df (data frame): input pandas data frame to be cleaned
        Returns:
            df_final(data frame): cleaned pandas data frame
            
    
    '''
    
    
    # Create a dataframe of the individual category columns.
    categories = df["categories"].str.split(pat=";", expand=True)
    
    # Select the first row of the categories column and use this row to extract a list of new column names for categories.
    row = categories.iloc[1,:]
    category_colnames = row.str.split("-").str.get(0)
    
    # Rename the newly extracted columns for the former categories column
    categories.columns = category_colnames
    
    # Convert category values to just numbers 0 or 1. Iterate through the category columns in df to keep only the last character of 
    # each string (the 1 or 0). For example, related-0 becomes 0, related-1 becomes 1. Convert the string to a numeric value.
    
    for column in categories:
    # Set each value to be the last character of the string.
        categories[column] = categories[column].str.split("-").str.get(1)    
    # Convert column from string to numeric.
        categories[column] = pd.to_numeric(categories[column])
    
    # Replace categories column in df with newly extracted category columns.
    df.drop(columns="categories", inplace=True)
    df_final = pd.concat([df, categories], axis = 1)
    
    # Some of the columns have category 2, which is incorrect. Replace them with 0s.     
    df_final.replace(2,0,inplace=True)
    
    # Remove duplicates.
    df_final.drop_duplicates(inplace=True)
    
    return df_final
    
    
def save_data(df, database_filename):
    '''
    Saves cleaned data to the sqlite database.
    
        Parameters: 
            df (data frame): pandas data frame with cleaned data
            database_filename (string): string with relative path and name of the sqlite database in which to save cleaned data
        Returns:
            None
    '''
    database_file_path = "sqlite:///" + database_filename
    engine = create_engine(database_file_path)
    
    df.to_sql('messages_cleaned', engine, index=False)


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