# Disaster Response Pipeline Project

This is Assignment project for the Udacity Data Scientist Nanodegree programme. In this project, main goal is to develop machine learning pipeline which processes real messages that were sent during disaster events, and categorizes them according to the content. The web app for the project is available at https://disasterresponse-app.herokuapp.com/ 

## Methodology
1. Data processing. 

    The raw data contains data from Figure Eight (https://appen.com), and is stored in disaster_categories.csv and disaster_messages.csv in the data folder. In the data proceesing step, messages and categories datasets were merged into one dateset, the categories column were splitted into separate, clearly named columns, values converted to binary, and duplicates were dropped. The processed dataset was stored in a SQLite database (DisasterResponse.db). 
2. Machine learning pipeline 

    For the machine learning pipeline, I first dropped predictors which had only one category (e.g child-alone), and tokenized the messages - normalized cases, lemmatized words, removed english stop words and tokenize text. Afterwards TF-IDF was applied to the text and used for the classifiction. For the classification, I tried three different alghorithms - Random Forests, Logistic Regression and Support Vector Classifiers. Grid search was uses for hyperparameter tuning, however for the sake of feasible time scale, I tried limited number of hyperparameters. The main problem with the dataset were unbalanced predictior category, where one category (0) is much more abundant for certain predictors. Therefore, I used hyperparamter to assign class weights in the alghorithm. I also accounted for corpus specific stopwords and tried different values for corpus-specific tresholds. Support vector classifier outperformed other two methods in therms of recall for the less abundant category. Therefore, it was included in the final model and experted as a pickle file (classifier.pkl) 
3. Deployment of web app
 
    The simple web app was built (disaster_app) where user  inputs a message into the app, the app returns classification results for 35 categories (without child-alone). The app was deployed on heroku server, however, it can also be deployed locally. 
 
## Summary
Support vector classifier outperformed logistic regression and random forests in therms of recall for the less abundant category, and was therefore selected for the final model. However, finer hyperparameter- tuning would be desirable in the future, and is currently limited by computer power. 

## How to run app?

### Prequisites 
1. In order to be sure that everything runs smoothly, first install virtual environment
    `python3 -m venv disasstervenv`
    
2. Activate virtual environment.
    - In Linux system 
    `source disasstervenv/bin/activate`
    - In Windows system 
    `disasterenv\Scripts\activate.bat`    
3. Install prequistites 
    - From the requirements file in the project's root directory. 
     `pip install requirements.txt`     
     - Install nltk text files `python -m nltk.downloader wordnet punkt stopwords`  
    

### Run the scripts and app:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. To run the webapp localy, in project's root directory:
    
    - Uncomment designated line in disaster_app.py, depending on if you want to run it on Udacity Workspace or locally on your computer
    - Run the following command in the app's directory to run your web app
    `python disaster_app.py`

3. Go to http://localhost:5000/ if running locally, or https://view<WORKSPACEID>-3001.udacity-student-workspaces.com/
