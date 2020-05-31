
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.functions import tokenize

from sqlalchemy import create_engine
import pandas as pd
import re



from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import pickle

def load_data(database_filepath):
    '''
    Reads cleaned data frame from sqlite database. 
        Parameteres:
            database_filepath (string): sqlite database from which to read the cleaned data
        Returns:
            X (data frame): pandas dataframe with predictor variable message
            Y (data frame): pandas dataframe with response variables, disaster response categories
            category_names(list): list of category names for the response variables
            
    '''
    # Load data from database.
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table("messages_cleaned", engine)
    
    # Assign predictor and response variables
    X = df.iloc[:,1]
    Y = df.iloc[:,4:]    
    
    #Drop "child alone" column which has only one class
    Y.drop("child_alone", inplace=True, axis = 1)
    # Assign category names
    category_names = Y.columns
    
    return (X, Y, category_names)
    
def build_model():
    '''
    Builds machine learning pipeline for natural language processing. First words corpus is 
    analyzed with Tfidf transformer and this is fed into random forest classificator.
    Hypeparameter tuning is performed with grid search.
        
        Parameters:
            None
        Returns:
            scikit model
    '''
    # Text processing and model pipeline.
    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)), 
    ('tfidf', TfidfTransformer()),
    ('cmo', MultiOutputClassifier(LinearSVC(random_state=1, class_weight='balanced')))
     ])
    
    # Define parameters for GridSearchCV.
    parameters = {
        'vect__max_df' : (0.1, 0.4)
    }
    # Create gridsearch object and return as final model pipeline.
   
    model = GridSearchCV(pipeline, param_grid=parameters)
    
    return model

def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Evaluates model in therms of precision, recall and F1 statistics.
        Parameters: 
            model : machine learning model to be evaluated
            X_test (data frame): Predictor variables for test set
            Y_test (data frame): Response variables for test set
            category_name (list): List of names for response variables
        
        Returns: 
            None
    '''
    Y_pred = model.predict(X_test)
    for i, category in enumerate (category_names):
        print(classification_report(Y_test[category], Y_pred.T[i], labels = [0, 1]))


def save_model(model, model_filepath):
    '''
    Export model as a pickle file.
        Parameters:
            model: model to export
            model_filepath (string): pickle file for saving            
        Returns:
            None
    '''
    
    # Export model as a pickle file.
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()