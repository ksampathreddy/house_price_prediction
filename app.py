from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get form data
        features = [
            float(request.form['MedInc']),
            float(request.form['HouseAge']),
            float(request.form['AveRooms']),
            float(request.form['AveBedrms']),
            float(request.form['Population']),
            float(request.form['AveOccup']),
            float(request.form['Latitude']),
            float(request.form['Longitude'])
        ]
        
        # Predict
        prediction = model.predict([features])[0] * 100000
        return render_template('index.html', 
                            prediction_text=f"Predicted Price: ${prediction:,.2f}")
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)