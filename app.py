import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from prophet import Prophet
import pickle
import yfinance as yf
from datetime import datetime, timedelta

# Set page config
st.set_page_config(
    page_title="Tesla Stock Prediction",
    page_icon="📈",
    layout="wide"
)

# Title and description
st.title("Tesla Stock Price Prediction")
st.markdown("""
This application predicts Tesla stock prices using Facebook Prophet model. 
The model has been trained on historical data and can forecast future stock prices.
""")

@st.cache_resource
def load_model():
    try:
        with open('prophet_tesla_model.pkl', 'rb') as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def get_latest_data():
    try:
        # Get the latest Tesla stock data
        tesla = yf.Ticker("TSLA")
        today = datetime.now()
        start_date = today - timedelta(days=365)  # Get 1 year of historical data
        df = tesla.history(start=start_date, end=today)
        df = df.reset_index()
        df = df[['Date', 'Close']]
        df.columns = ['ds', 'y']
        return df
    except Exception as e:
        st.error(f"Error fetching Tesla stock data: {str(e)}")
        return None

def create_forecast_plot(historical_data, forecast_data):
    fig = go.Figure()

    # Plot historical data
    fig.add_trace(go.Scatter(
        x=historical_data['ds'],
        y=historical_data['y'],
        name='Historical Price',
        line=dict(color='blue')
    ))

    # Plot forecast
    fig.add_trace(go.Scatter(
        x=forecast_data['ds'],
        y=forecast_data['yhat'],
        name='Predicted Price',
        line=dict(color='red')
    ))

    # Add confidence intervals
    fig.add_trace(go.Scatter(
        x=forecast_data['ds'].tolist() + forecast_data['ds'].tolist()[::-1],
        y=forecast_data['yhat_upper'].tolist() + forecast_data['yhat_lower'].tolist()[::-1],
        fill='toself',
        fillcolor='rgba(255,0,0,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='Confidence Interval'
    ))

    fig.update_layout(
        title='Tesla Stock Price Forecast',
        xaxis_title='Date',
        yaxis_title='Stock Price (USD)',
        hovermode='x',
        showlegend=True
    )

    return fig

try:
    # Load the trained model
    model = load_model()

    # Get latest data
    historical_data = get_latest_data()
    
    # Sidebar for forecast parameters
    st.sidebar.header("Forecast Parameters")
    forecast_days = st.sidebar.slider("Forecast Days", 1, 365, 30)

    # Create future dates for prediction
    future_dates = model.make_future_dataframe(periods=forecast_days)
    
    # Make prediction
    forecast = model.predict(future_dates)

    # Display plots
    st.plotly_chart(create_forecast_plot(historical_data, forecast), use_container_width=True)

    # Display forecast data
    st.subheader("Forecast Data")
    
    # Create a DataFrame with the forecast results
    forecast_df = pd.DataFrame({
        'Date': forecast['ds'].dt.date,
        'Predicted Price': forecast['yhat'].round(2),
        'Lower Bound': forecast['yhat_lower'].round(2),
        'Upper Bound': forecast['yhat_upper'].round(2)
    }).tail(forecast_days)
    
    st.dataframe(forecast_df)
    
    # Add download button for forecast data
    csv = forecast_df.to_csv(index=False)
    st.download_button(
        label="Download Forecast Data",
        data=csv,
        file_name="tesla_stock_forecast.csv",
        mime="text/csv"
    )
    
    # Add model metrics
    st.subheader("Model Performance Metrics")
    metrics_col1, metrics_col2 = st.columns(2)
    
    with metrics_col1:
        latest_price = historical_data['y'].iloc[-1]
        st.metric("Latest Stock Price", f"${latest_price:.2f}")
        
    with metrics_col2:
        next_day_forecast = forecast['yhat'].iloc[-forecast_days]
        price_change = next_day_forecast - latest_price
        st.metric("Next Day Forecast", f"${next_day_forecast:.2f}", 
                 delta=f"${price_change:.2f}")
                 
except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    st.info("Please try refreshing the page or contact support if the error persists.")
    forecast_display = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(forecast_days)
    forecast_display.columns = ['Date', 'Predicted Price', 'Lower Bound', 'Upper Bound']
    st.dataframe(forecast_display.round(2))

    # Model components
    st.subheader("Model Components")
    components = model.plot_components(forecast)
    st.pyplot(components)

except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    st.info("Please make sure the model file 'prophet_tesla_model.pkl' exists in the same directory as this script.")