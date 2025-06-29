# Weather Forecast Application

## Overview
This is a Flask-based web application that provides weather forecasting and historical weather data analysis. It uses machine learning models to predict temperature and weather conditions, integrates with the Open-Meteo API for weather data, and Geoapify for location autocomplete functionality.

## Features
- **Location-based Weather Forecasting**: Users can input a city name to get weather predictions.
- **Historical Weather Trends**: View historical weather statistics for a specified date range and metrics.
- **Data Exploration**: Analyze temperature, humidity, precipitation, and weather conditions.
- **Autocomplete Location Input**: Real-time location suggestions using the Geoapify API.
- **Automated Model Updates**: Machine learning models are updated every 6 hours using a scheduler.

## Requirements
The application dependencies are listed in `requirements.txt`. Key dependencies include:
- Flask (2.3.3): Web framework
- Pandas (2.2.2): Data manipulation
- Scikit-learn (1.5.1): Machine learning models
- Geopy (2.4.1): Geocoding for location data
- Requests (2.32.3): API calls
- Plotly (5.24.1): Data visualization (optional for future enhancements)
- Gunicorn (21.2.0): Production server

To install dependencies, run:
```bash
pip install -r requirements.txt
```

## Setup
1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Environment Variables**:
   Create a `.env` file in the root directory with the following:
   ```plaintext
   FLASK_SECRET_KEY=your-secret-key
   GEOAPIFY_API_KEY=your-geoapify-api-key
   ```
   - Obtain a Geoapify API key from [Geoapify](https://www.geoapify.com/).
   - Generate a secure `FLASK_SECRET_KEY` for session management.

3. **Run the Application**:
   For development:
   ```bash
   python application.py
   ```
   For production (using Gunicorn):
   ```bash
   gunicorn -w 4 -b 0.0.0.0:5000 application:app
   ```

4. **Access the Application**:
   Open a browser and navigate to `http://localhost:5000`.

## File Structure
- **application.py**: Main Flask application with routes for index, prediction, historical trends, and data exploration.
- **weather_forecast.py**: Core logic for fetching weather data, training machine learning models, and making predictions.
- **main.js**: Frontend JavaScript for handling autocomplete, date validation, and flash messages.
- **templates/**:
  - `index.html`: Set location with autocomplete input.
  - `predict.html`: Generate weather predictions.
  - `historical.html`: View historical weather trends.
  - `explore.html`: Explore weather data statistics.
- **requirements.txt**: Lists project dependencies.

## Usage
1. **Set Location** (`/`):
   - Enter a city name (e.g., "London") in the location input field.
   - The autocomplete feature suggests locations using Geoapify.
   - Submit to set the location and update models.

2. **Weather Prediction** (`/predict`):
   - Select a future date or leave blank for tomorrow’s prediction.
   - View predicted temperature and weather conditions with probability scores.

3. **Historical Trends** (`/historical`):
   - Select a date range and metrics (e.g., temperature, humidity).
   - View statistical summaries (min, max, mean, median, latest).

4. **Data Exploration** (`/explore`):
   - View aggregated statistics for temperature, humidity, precipitation, and weather conditions.

## Technical Details
- **Backend**:
  - Uses Open-Meteo API for historical and forecast weather data.
  - Implements RandomForest and GradientBoosting models for temperature and weather condition predictions.
  - Features time-series cross-validation and robust error handling.
  - Scheduler updates models every 6 hours using `schedule`.

- **Frontend**:
  - Built with Jinja2 templates extending `base.html`.
  - Uses Geoapify API for location autocomplete in `main.js`.
  - Includes date validation and flash message handling.

- **Machine Learning**:
  - Temperature prediction uses RandomForest, GradientBoosting, or Ridge regression.
  - Weather condition classification uses RandomForestClassifier with balanced class weights.
  - Features include lagged values, rolling averages, and time-based features (hour, day, month).

## Notes
- Ensure a stable internet connection for API calls (Open-Meteo, Geoapify).
- The application assumes the Geoapify API key is valid and has sufficient quota.
- Models require sufficient historical data (90 days by default) for training.
- Flash messages auto-hide after 5 seconds for better user experience.

## Future Improvements
- Add data visualization with Plotly for historical trends.
- Implement caching for API responses to reduce load.
- Enhance model accuracy with additional features or hyperparameter tuning.
- Add user authentication for personalized settings.

## Troubleshooting
- **Location not found**: Ensure the city name is valid and check Geoapify API key.
- **API errors**: Verify internet connectivity and API key validity.
- **Model failures**: Check logs in `application.py` and `weather_forecast.py` for details.

For support, contact the repository maintainer or open an issue.