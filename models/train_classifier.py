# import libraries
import sys
import pandas as pd
import numpy as np
import re
import pickle

from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

import nltk
nltk.download(['punkt','wordnet'])

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix


#see https://docs.python.org/3/library/argparse.html on how to use argparse for similar task: a command line 


def load_data(database_filepath):
    '''
    This function load data from the database and split the relevant data into X and Y
    
    arg: 
    database_filepath: the path to the database
    
    return: 
    X: independent varibale 
    y: dependent varibale
    columnnames: column header for the y columns
    
    '''
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('messages_category', engine)  
    X = df['message'].values
    #multiple MultiOutputClassifier will return error because if genre column is object. We can recode the 3 category to number
    df["genre"].replace(
        {"news": 0, "direct": 1,"social":2}, 
        inplace=True)
    y = df.drop('message', axis=1)
    columnnames = list(y.columns)
    y = df.drop('message', axis=1).values
    return X,y, columnnames


def tokenize(text):
    '''
    This funtion tokenize text and clean up the test for further processing
    
    arg: 
    text : this is the text we want to tokenize
    
    retunr: 
    clean_tokens : this is the cleaned and tokenized text
    
    '''
    tokens = re.sub(r"[^a-zA-Z0-9]"," ",text)
    tokens = word_tokenize(tokens)
    
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    
    return clean_tokens


def build_model():
    '''
    This funtion build our model using CountVectorizer and TfidfTransformer for our text data
    
    arg:
    params: this is the list of parameters from our parameter funtion
    
    return:
    cv: GridSearchCV object
    
    '''
    pipeline = Pipeline([
            ('vect',CountVectorizer(tokenizer = tokenize)),
            ('tfidf', TfidfTransformer()),
            ('clf', MultiOutputClassifier(RandomForestClassifier()))
            
        ])
    
    param = {
        'clf__estimator__max_depth': [5,10,None],
        'clf__estimator__max_leaf_nodes':[5,10,None],
        'clf__estimator__n_jobs':[-1],
        'clf__estimator__max_features':['auto', 'sqrt', 'log2']
    }
    
    cv = GridSearchCV(estimator = pipeline, param_grid = param)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    This funtion evaluate our model and print out result of our cross validation
    
    arg:
    model: this is our GridSearchcv output
    X_test: this is our text data
    Y_test: this is the value of our expected result
    category_names: this is the list of our Y columns
    '''
       
    df = pd.DataFrame.from_dict(model.cv_results_)
    
    print("#Result of cross validation#")
    print("Best score:{}".format(model.best_score_))
    print("Best parameters set:{}".format(model.best_estimator_.get_params()["clf"]))

    print("#Scoring on test#")
    
    y_pred = model.predict(X_test)
    
    print('#Classification Report#')
    
    try:
        print(y_pred.shape)
        print(Y_test.shape)
        print(category_names)
        for i, var in enumerate(category_names):
            print('Predictions for {}'.format(var))
            print(classification_report(Y_test[:,i], y_pred[:,i]))
            print("Confusion Matrix: {}".format(confusion_matrix(Y_test[:,i],y_pred[:,i])))

    except:
        print('done with this')
    
    

def save_model(model, model_filepath):
    '''
    This funtion save the result of our model as a pickle file
    
    arg:
    model: this is our gridsearchCV object
    model_filepath: Pickle file destination
    '''
    with open(model_filepath, "wb") as f:
        pickle.dump(model, f)


def main():
    '''
    This funtion calls all other funtions that loads our data from database, initiate parameters, build model, train model
    evaluate model and save model
    '''
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