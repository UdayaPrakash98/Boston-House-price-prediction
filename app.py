import pickle

import flask
import json
import numpy as np
#from sklearn.externals import joblib
import pandas as pd
from flask import Flask, render_template, request
#from keras.models import model_from_json

app = Flask(__name__)
pickle= pickle.load(open('Boston.pkl','rb'))



@app.route("/")
@app.route("/bostonindex")
def index():
    return flask.render_template('index.html')


@app.route("/predict", methods=['POST'])
def make_predictions():
    if request.method == 'POST':
        a = request.form.get('crim')
        b = request.form.get('zn')
        c = request.form.get('indus')
        d = request.form.get('chas')
        e = request.form.get('nox')
        f = request.form.get('rm')
        g = request.form.get('age')
        h = request.form.get('dis')
        i = request.form.get('rad')
        j = request.form.get('tax')
        k = request.form.get('ptratio')
        l = request.form.get('b')
        m = request.form.get('lstat')

        new=pd.DataFrame({'a':[a],'b':[b], 'c':[c],'d':[d],'e':[e],'f':[f],'g':[g],'h':[h],'i':[i],'j':[j],'k':[k],'l':[l],'m':[m]})
        pred = 1000*(pickle.predict(new))
        if pred < 0:
            return render_template('index.html', response="Sorry you cannot sell this House")
        else:
            return render_template('predicting.html', response=pred)
if __name__=="__main__":
    app.run(debug=True)
