# Import required libraries
import pandas as pd
from prophet import Prophet
from flask import Flask, render_template, request, jsonify
import time

# Initialize Flask app
app = Flask(__name__)

# Define the route to the homepage
@app.route('/')
def home():
	return render_template('index.html')

# Define the route to the API
@app.route('/predict', methods=['POST'])
def predict():
	# Get the uploaded CSV file
	file = request.files['file']

	# Load the CSV file into a Pandas DataFrame
	df = pd.read_csv(file)

	# Parse the periods input
	periods = int(request.form['periods'])

	# Fit the Prophet model
	model = Prophet()
	model.fit(df)

	# Create the future dataframe for prediction
	future_df = model.make_future_dataframe(periods=periods)

	# Make predictions
	start_time = time.time()
	forecast = model.predict(future_df)
	end_time = time.time()

	# Return the forecast as a JSON response
	return jsonify((forecast[['ds', 'yhat']].tail(periods)).to_dict(orient='records'))

# Run the app
if __name__ == '__main__':
	app.run(host='0.0.0.0',port=5001, debug=True)

