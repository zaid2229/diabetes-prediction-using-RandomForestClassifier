import numpy as np
import pickle
from flask import Flask, render_template, request, jsonify
import sklearn
import joblib

app = Flask(__name__)

@app.route('/')
def Home():
    return render_template('index.html')

@app.route('/predict', methods=['POST','GET'])
def results():
    num_preg  = int(request.form['num_preg'])
    glucose_conc = int(request.form['glucose_conc'])
    diastolic_bp = int(request.form['diastolic_bp'])
    thickness   = int(request.form['thickness'])
    insulin = int(request.form['insulin'])
    bmi   = float(request.form['bmi'])
    diab_pred    = float(request.form['diab_pred'])
    age  = int(request.form['age'])
    skin    = float(request.form['skin'])
    
    X = np.array([[num_preg,glucose_conc,diastolic_bp,thickness,insulin,bmi,diab_pred,age,skin]])
    model = joblib.load('model_joblib_rf1')
    Y_predict = model.predict(X)
    return jsonify({'Prediction': int(Y_predict)})


if __name__ == '__main__':
    app.run(debug = True, port = 1010)