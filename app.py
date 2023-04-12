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

	#Global definitions
	global start_time
	global end_time
	global execution_time
	execution_time = 0

	# Make predictions
	start_time = time.time()
	forecast = model.predict(future_df)
	end_time = time.time()

	execution_time = end_time - start_time

	# Return the forecast as a JSON response
	return jsonify((forecast[['ds', 'yhat']].tail(periods)).to_dict(orient='records'))

# Define a new function for a new page
@app.route('/execute')
def execute():
	about_text = "Time taken to complete "+ str(execution_time)
	return render_template('execute.html', about_text=about_text)

# Run the app
if __name__ == '__main__':
	app.run(host='0.0.0.0',port=5001, debug=True)

