import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.functions import tokenize
from disaster_app import app
import json
import plotly
import pandas as pd

from disaster_app import app
from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import pickle
from sqlalchemy import create_engine




# load data
engine = create_engine('sqlite:///./data/DisasterResponse.db')
df = pd.read_sql_table('messages_cleaned', engine)
#drop child alone column, since it has only one class
df.drop("child_alone", inplace=True, axis = 1)

# load model
model = pickle.load(open("./models/classifier.pkl", "rb" ))


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    category_counts = df.iloc[:,4:].sum()
    category_names = list(df.iloc[:,4:].columns)
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=category_names,
                    y=category_counts
                )
            ],

            'layout': {
                'autosize': True,
                'title': 'Distribution of Disaster Message Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category"
                    },
                'margin' : {
                    'l': 50,
                    'r': 50,
                    'b':170
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )
