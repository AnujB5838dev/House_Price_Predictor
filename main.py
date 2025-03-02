from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)


data = pd.read_csv('Modal_HPP/Cleaned_data.csv')
pipe = pickle.load(open('Modal_HPP/RidgeModal.pkl', 'rb'))

@app.route('/')
def index():
    locations = sorted(data['location'].unique())
    return render_template('index.html', locations=locations)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        
        location = request.form.get('location')
        bhk = int(request.form.get('bhk'))
        bath = int(request.form.get('bath'))
        sqft = float(request.form.get('sqft'))

        
        input_data = pd.DataFrame([[location, sqft, bath, bhk]], 
                                  columns=['location', 'total_sqft', 'bath', 'bhk'])
        
        
        prediction = pipe.predict(input_data)[0] * 1e5

        
        return str(np.round(prediction,2))

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    app.run(debug=True, port=5001)
