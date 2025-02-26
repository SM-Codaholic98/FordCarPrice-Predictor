from flask import Flask, request, render_template
from flask_cors import cross_origin
from sklearn.preprocessing import PowerTransformer
import pickle, numpy as np

app = Flask(__name__)

with open('FordCarPrice_Predictor.pkl', 'rb') as file:
    data = pickle.load(file)
    Model = data['Model']
    transformer = data['Transformer']
    scaler = data['Scaler']


@app.route("/")
@cross_origin()
def home():
    return render_template("WebApp.html")


@app.route("/predict", methods = ["GET", "POST"])
@cross_origin()
def predict():
    if request.method == "POST":            
        model = request.form['model']
        if model == 'B-MAX':
            model = 0
        elif model == 'C-MAx':
            model = 1
        elif model == 'EcoSport':
            model = 2
        elif model == 'Edge':
            model = 3
        elif model == 'Escort':
            model = 4
        elif model == 'Fiesta':
            model = 5
        elif model == 'Focus':
            model = 6
        elif model == 'Fusion':
            model = 7
        elif model == 'Galaxy':
            model = 8
        elif model == 'Grand C-MAX':
            model = 9
        elif model == 'Grand Tourneo Connect':
            model = 10
        elif model == 'KA':
            model = 11
        elif model == 'Ka+':
            model = 12
        elif model == 'Kuga':
            model = 13
        elif model == 'Mondeo':
            model = 14
        elif model == 'Mustang':
            model = 15
        elif model == 'Puma':
            model = 16
        elif model == 'Ranger':
            model = 17
        elif model == 'S-MAX':
            model = 18
        elif model == 'Streetka':
            model = 19
        elif model == 'Tourneo Connect':
            model = 20
        elif model == 'Tourneo Custom':
            model = 21
        else:
            model = 22
            
        transmission = request.form['transmission']
        if transmission == 'Automatic':
            transmission = 0
        elif transmission == 'Manual':
            transmission = 1
        else:
            transmission = 2
            
        fuelType = request.form['fuelType']
        if fuelType == 'Diesel':
            fuelType = 0
        elif fuelType == 'Electric':
            fuelType = 1
        elif fuelType == 'Hybrid':
            fuelType = 2
        else:
            fuelType = 3
            
        year = int(request.form['year'])
        mileage = int(request.form['mileage'])
        tax = int(request.form['tax'])
        mpg = float(request.form['mpg'])
        engineSize = float(request.form['engineSize'])
        
        transformed_data = transformer.transform([[year, mileage, tax, mpg, engineSize]])
        scaled_data = scaler.transform(transformed_data)
        predicted_price = Model.predict([[model, transmission, fuelType] + list(scaled_data[0])])

        return render_template('WebApp.html', prediction_text=f'${predicted_price[0]:.3f} & INR {predicted_price[0] * 85.48:.3f}')

    return render_template("WebApp.html")


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)