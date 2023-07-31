from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

with open('rain_prediction_model.pkl', 'rb') as f:
    model = pickle.load(f)
    
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    location = int(request.form['location'])
    min_temp = float(request.form['min_temp'])
    max_temp = float(request.form['max_temp'])
    wind_dir = float(request.form['wind_dir'])
    wind_speed = float(request.form['wind_speed'])
    humidity = float(request.form['humidity'])
    pressure = float(request.form['pressure'])
    cloud = float(request.form['clouds'])
    temp = float(request.form['temp'])
    today_rain = int(request.form['rain'])
    
    input_values = np.array([[location, min_temp, max_temp, wind_dir, wind_speed, humidity, pressure, cloud, temp, today_rain]])
    
    prediction = model.predict(input_values)[0]
    
    if prediction == 0:
        result = 'It will not rain'
    else:
        result = 'It will rain'
        
    return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)