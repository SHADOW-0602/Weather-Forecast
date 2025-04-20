from flask import Flask, render_template, request, flash, redirect, url_for
from weather_forecast import WeatherForecastSystem
import threading
import schedule
import time
import logging
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'dev-secret-key')

# Initialize the weather forecast system
forecast_system = WeatherForecastSystem()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_scheduler():
    """Run the model update scheduler in a separate thread"""
    schedule.every(6).hours.do(forecast_system.update_model)
    while True:
        schedule.run_pending()
        time.sleep(60)

# Start the scheduler in a separate thread
scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
scheduler_thread.start()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        location = request.form.get('location', '').strip()
        if not location:
            flash('Location cannot be empty', 'error')
        else:
            try:
                logger.info(f"Attempting to set location: {location}")
                if forecast_system.set_location(location):
                    forecast_system.update_model()
                    flash('Location set and models updated successfully!', 'success')
                    return redirect(url_for('predict'))
                else:
                    logger.error(f"Failed to set location: {location} - Nominatim could not find coordinates")
                    flash('Invalid location. Please try a different city name (e.g., "London" or "New York").', 'error')
            except Exception as e:
                logger.error(f"Error updating models for location {location}: {str(e)}", exc_info=True)
                flash(f'Failed to update weather models: {str(e)}', 'error')
    return render_template('index.html', now=datetime.now(), location=request.form.get('location', ''))

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    prediction = None
    if request.method == 'POST':
        date_str = request.form.get('date')
        try:
            prediction_date = (datetime.strptime(date_str, '%Y-%m-%d') 
                              if date_str 
                              else datetime.now() + timedelta(days=1))
            
            if prediction_date.date() < datetime.now().date():
                flash('Prediction date must be today or in the future', 'error')
            else:
                prediction = forecast_system.predict_weather(prediction_date)
                if prediction.get('success'):
                    flash('Prediction generated successfully!', 'success')
                else:
                    flash(prediction.get('error', 'Prediction failed'), 'error')
        except ValueError:
            flash('Invalid date format. Please use YYYY-MM-DD', 'error')
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}", exc_info=True)
            flash('Error generating prediction', 'error')
    return render_template('predict.html', prediction=prediction, now=datetime.now())

@app.route('/historical', methods=['GET', 'POST'])
def historical():
    stats = None
    if request.method == 'POST':
        try:
            start_date = request.form.get('start_date')
            end_date = request.form.get('end_date')
            metrics = request.form.getlist('metrics') or ['temperature']  # Default metric
            
            # Validate dates
            if not start_date or not end_date:
                flash('Please provide both start and end dates', 'error')
                return render_template('historical.html', stats=None, now=datetime.now())
            try:
                start_date = datetime.strptime(start_date, '%Y-%m-%d')
                end_date = datetime.strptime(end_date, '%Y-%m-%d')
                if start_date > end_date:
                    flash('Start date must be before end date', 'error')
                    return render_template('historical.html', stats=None, now=datetime.now())
            except ValueError:
                flash('Invalid date format. Please use YYYY-MM-DD', 'error')
                return render_template('historical.html', stats=None, now=datetime.now())
                
            # Validate at least one metric selected
            if not metrics:
                flash('Please select at least one metric', 'error')
                return render_template('historical.html', stats=None, now=datetime.now())
                
            stats = forecast_system.view_historical_trends(start_date, end_date, metrics)
            if stats:
                flash('Historical statistics generated successfully!', 'success')
            else:
                flash('No data available for the selected period', 'warning')
        except Exception as e:
            logger.error(f"Historical data error: {str(e)}", exc_info=True)
            flash(f"Error generating historical data: {str(e)}", 'error')
    return render_template('historical.html', stats=stats, now=datetime.now())

@app.route('/explore', methods=['GET'])
def explore():
    try:
        stats = forecast_system.explore_data()
        if not stats or not all(key in stats for key in ['temperature', 'humidity', 'precipitation', 'weather_conditions']):
            flash('Incomplete data available for exploration', 'warning')
            return render_template('explore.html', stats=None, now=datetime.now())
        return render_template('explore.html', stats=stats, now=datetime.now())
    except Exception as e:
        logger.error(f"Data exploration error: {str(e)}", exc_info=True)
        flash(f"Error exploring data: {str(e)}", 'error')
        return render_template('explore.html', stats=None, now=datetime.now())
    
@app.route('/config')
def get_config():
    return {
        'geoapifyApiKey': os.getenv('GEOAPIFY_API_KEY')
    }    

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)