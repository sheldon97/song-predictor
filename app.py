from music import *
from flask import Flask
from flask import request

app = Flask(__name__)

@app.route("/")
def home():
    render_template('search.html')

@app.route("/search", methods=['POST', 'GET'])
def search():
    data =  print_artist_recommendations(request.form['music'], wide_artist_data_zero_one,   model_nn_binary, k = 10)

    render_template('results.html', data=data)
