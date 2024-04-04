# install packages: pip install streamlit fbprophet yfinance plotly
import streamlit as st
from datetime import date
from stocknews import StockNews
import datetime
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly

import pandas as pd
import base64
import plotly.graph_objects as go
import sqlite3


# Connect to the Watchlist SQLite database
conn = sqlite3.connect('watchlist.db')
c = conn.cursor()

# Create a table to store watchlist data
c.execute('''CREATE TABLE IF NOT EXISTS watchlist
             (user_id INTEGER, stock TEXT)''')
# Commit the changes and close the connection
conn.commit()
conn.close()


# Connect to the Returns SQLite database
conn = sqlite3.connect('returns.db')
c = conn.cursor()

# Create a table to store calculated returns
c.execute('''CREATE TABLE IF NOT EXISTS returns
             (ticker TEXT, start_date TEXT, end_date TEXT, return_value REAL)''')
# Commit the changes and close the connection
conn.commit()
conn.close()


# Configure Streamlit app
st.set_page_config(
    page_title="Jack's Stock App",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Function to fetch S&P 500 tickers
def fetch_sp500_tickers():
    # Path to the local CSV file
    file_path = 'symbols_csv.csv'
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)
    # Extract the 'Symbol' column and convert to a list
    sp500_tickers = df['Symbol'].tolist()
    return sp500_tickers

# Fetch S&P 500 tickers
sp500_tickers = fetch_sp500_tickers()

# Default stock and date ranges to display
default_stock = "AAPL"
START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

# Set app title and icons on sidebar and main page
st.markdown('<p style="font-size: 40px; margin-top: -60px; margin-bottom: -100px;">📈📈</p>', unsafe_allow_html=True)
st.sidebar.markdown('<p style="font-size: 40px; margin-top: -60px;">💰💸💎💰</p>', unsafe_allow_html=True)
st.title('Stock Forecasting App')

# Add tabs
tabs = ["Home", "Charts", "Prophet Forecasts", "Raw Data", "Highest Returns", "My Watchlist", "News", "About"]
selected_tab = st.sidebar.selectbox("Select Tab", tabs)

# Function to load stock data from Yahoo Finance
@st.cache_data
def load_data(ticker, start_date, end_date):
    data = yf.download(ticker, start_date, end_date)
    data.reset_index(inplace=True)
    return data

# Function to create candlestick chart
@st.cache_data
def create_candlestick_chart(data):
    fig = go.Figure(data=[go.Candlestick(x=data['Date'], open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'])])
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
    fig.update_layout(margin=dict(l=0, r=0, t=40, b=0), width=800)
    fig.layout.update(title_text='Candlestick Chart', title_font=dict(size=24), xaxis_rangeslider_visible=True)
    st.plotly_chart(fig, use_container_width=True)

# Function to create forecast chart

def create_forecast_chart(m, forecast, n_years):
    fig = plot_plotly(m, forecast)
    fig.update_traces(selector=dict(name=fig.data[0].name), line=dict(color='red'))
    fig.update_traces(mode='lines')
    fig.update_layout(xaxis=dict(title=''), yaxis=dict(title=''))
    fig.update_layout(margin=dict(l=0, r=0, t=40, b=0), width=800)
    st.plotly_chart(fig, use_container_width=True)

# Function to allow user to download CSV file
def download_csv(dataframe, filename):
    csv = dataframe.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}" onclick="event.stopPropagation();">Download CSV File</a>'
    return href

# Function to calculate return over a given time period
@st.cache_data
def calculate_return(ticker, start_date, end_date):
    # Connect to the SQLite database
    conn = sqlite3.connect('returns.db')
    c = conn.cursor()

    # Check if return value exists in the database
    c.execute("SELECT return_value FROM returns WHERE ticker=? AND start_date=? AND end_date=?", (ticker, str(start_date), str(end_date)))
    result = c.fetchone()

    if result:
        return_value = result[0]
    else:
        # Calculate the return
        data = yf.download(ticker, start_date, end_date)
        start_price = data.iloc[0]['Close']
        end_price = data.iloc[-1]['Close']
        return_value = (end_price - start_price) / start_price * 100

        # Store the return value in the database
        c.execute("INSERT INTO returns (ticker, start_date, end_date, return_value) VALUES (?, ?, ?, ?)",
                  (ticker, str(start_date), str(end_date), return_value))
        conn.commit()

    # Close the connection
    conn.close()

    return return_value

# Home tab to display info on all the other tabs
if selected_tab == "Home":

    st.subheader("Welcome to Jack's Stock Forecasting App!")
    st.write("This app allows you to predict S&P 500 stock prices using the Meta Prophet library. It utilizes historical S&P 500 stock data to train a model and generates forecasts for the selected stock.")
    st.write("List of Tabs:")
    st.write("- **Charts** - view candlestick charts of selected stocks and explore the forecast charts for a specified number of years into the future. This tab also allows you to see the return of a stock during a given time period.")

    st.write("- **Prophet Forecasts** - you can access and analyze the forecasted stock prices generated by the Prophet model. Adjust the parameters and explore different stocks to help with investment decisions.")

    st.write("- **Raw Data** - provides access to the raw historical data of the selected stock as well as the forecasted data. You can examine the data and perform your own analysis by downloading the csv file.")

    st.write("- **Highest Returns** - displays the stocks with the highest returns over a specific time period. This can help you identify potential investment opportunities.")

    st.write("- **My Watchlist** - you can create and manage your own watchlist of stocks. Stay updated on the latest price movements and forecasts for your favorite stocks.")

    st.write("- **News** - aggregates the latest news articles related to a stock's sector. Stay informed about the latest trends and events that may impact your investment decisions.")
    st.write("- **About** - lists all of the app's features and gives a detailed view on how Prophet works.")

    st.write("For more information about how the Prophet library works, the methodology behind the forecasts, or what specific parameters do, please navigate to the 'About' tab.")

    st.write("Feel free to explore the different tabs and leverage the app's features!")

# Tab to view candlestick and forecast charts
elif selected_tab == "Charts":
    # Select a stock dataset for display and prediction
    selected_stock = st.selectbox('Select a dataset to display', sp500_tickers, index=sp500_tickers.index(default_stock))

    # Choose the number of years for prediction min 1, max 4
    n_years = st.slider('Years of prediction:', 1, 4)
    period = n_years * 365

    # Select the start and end dates for the chart
    start_date = st.sidebar.date_input('Chart Start Date', value=date(2015, 1, 1), min_value=date(2010, 1, 1), max_value=date.today())
    end_date = st.sidebar.date_input('Chart End Date', value=date.today(), min_value=date(2010, 1, 1), max_value=date.today())
    st.sidebar.write("")
    
    # Load the selected stock data
    data = load_data(selected_stock, start_date, end_date)

    # Create and display the candlestick chart
    create_candlestick_chart(data)
    
    # Select start and end dates to calculate the return (seperate from chart dates)
    start_investment_date = st.sidebar.date_input('Start date to calculate % return', value=start_date, min_value=start_date, max_value=end_date)
    end_investment_date = st.sidebar.date_input('End date to calculate % return', value=end_date, min_value=start_investment_date, max_value=end_date)
    
    # Load historical stock data
    investment_data = load_data(selected_stock, start_investment_date, end_investment_date)
    
    # Calculate overall return
    start_price = investment_data.iloc[0]['Close']
    end_price = investment_data.iloc[-1]['Close']
    overall_return = (end_price - start_price) / start_price * 100

    # Display overall return
    st.subheader('Overall Return')
    st.write(f"Return for {selected_stock} (Period: {start_investment_date} - {end_investment_date}): {overall_return:.2f}%")
    st.write("")
    st.write("")
    
    # Select the 'Date' and 'Close' columns from the data DataFrame
    df_train = data[['Date','Close']]
    # Rename the columns to match the required format for Prophet model (ds for date and y for target variable)
    df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

    # Initialize and fit the Prophet model (different data format)
    m = Prophet()
    m.fit(df_train)
    
    # Make future predictions over period
    future = m.make_future_dataframe(periods=period)
    forecast = m.predict(future)

    # Display the forecast chart
    st.write(f'Forecast chart for {n_years} year(s) into the future')
    create_forecast_chart(m, forecast, n_years)

    # Calculate forecasted return
    forecast_start_date = date.today().strftime("%Y-%m-%d")
    forecast_end_date = (date.today() + datetime.timedelta(days=365*n_years)).strftime("%Y-%m-%d")
    forecasted_data = forecast[(forecast['ds'] >= forecast_start_date) & (forecast['ds'] <= forecast_end_date)]
    forecasted_start_price = forecasted_data.iloc[0]['yhat']
    forecasted_end_price = forecasted_data.iloc[-1]['yhat']
    forecasted_return = (forecasted_end_price - forecasted_start_price) / forecasted_start_price * 100

    # Display forecasted return
    st.subheader('Forecasted Return')
    st.write(f"Forecasted return for {selected_stock} (Period: {forecast_start_date} - {forecast_end_date}): {forecasted_return:.2f}%")
    st.write("")
    st.write("")

# Tab to experiment with different parameters in the Prophet model
elif selected_tab == "Prophet Forecasts":
    # Select a stock dataset for display and prediction
    selected_stock = st.selectbox('Select a dataset for prediction', sp500_tickers, index=sp500_tickers.index(default_stock))

    # Choose the number of years for prediction min 1, max 4
    n_years = st.slider('Years of prediction:', 1, 4)
    period = n_years * 365

    # Select the start and end dates for the chart
    start_date = st.sidebar.date_input('Chart Start Date', value=date(2015, 1, 1), min_value=date(2010, 1, 1), max_value=date.today())
    end_date = st.sidebar.date_input('Chart End Date', value=date.today(), min_value=date(2010, 1, 1), max_value=date.today())
    st.sidebar.write("")
    
    # Load the selected stock data
    data = load_data(selected_stock, start_date, end_date) 
    
    # Select the 'Date' and 'Close' columns from the data DataFrame
    df_train = data[['Date','Close']]
    # Rename the columns to match the required format for Prophet model (ds for date and y for target variable)
    df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

    # Initialize and fit the Prophet model (different data format)
    m = Prophet()
    m.fit(df_train)
    # Make future predictions over period
    future = m.make_future_dataframe(periods=period)
    forecast = m.predict(future)

    # Display the forecast chart
    st.write(f'Forecast chart for {n_years} year(s) into the future')
    create_forecast_chart(m, forecast, n_years)

    # Calculate forecasted return
    forecast_start_date = date.today().strftime("%Y-%m-%d")
    forecast_end_date = (date.today() + datetime.timedelta(days=365*n_years)).strftime("%Y-%m-%d")
    forecasted_data = forecast[(forecast['ds'] >= forecast_start_date) & (forecast['ds'] <= forecast_end_date)]
    forecasted_start_price = forecasted_data.iloc[0]['yhat']
    forecasted_end_price = forecasted_data.iloc[-1]['yhat']
    forecasted_return = (forecasted_end_price - forecasted_start_price) / forecasted_start_price * 100

    # Display forecasted return
    st.subheader('Forecasted Return')
    st.write(f"Forecasted return for {selected_stock} (Period: {forecast_start_date} - {forecast_end_date}): {forecasted_return:.2f}%")
    st.write("")
    st.write("")

    # Display forcasted seasonality components
    st.write("Forecast components")
    fig2 = m.plot_components(forecast)
    st.write(fig2)  
    
    # Create Volatility graph based off forecast
    def create_prediction_interval_plot(forecast):
        st.subheader("Volatility Graph")
        fig = go.Figure()

        # Add traces for lower bound, upper bound, and forecast
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], mode='lines', name='Lower Bound'))
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], mode='lines', name='Upper Bound'))
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecast'))

        # Add sliders for adjusting variability
        variability_lower = st.slider("Lower Bound Variability", min_value=0.0, max_value=1.0, value=0.2, step=0.01)
        variability_upper = st.slider("Upper Bound Variability", min_value=0.0, max_value=1.0, value=0.2, step=0.01)

        # Update the y-values of lower bound and upper bound traces based on variability
        fig.data[0].y = forecast['yhat_lower'] * (1 - variability_lower)
        fig.data[1].y = forecast['yhat_upper'] * (1 + variability_upper)

        # Update layout and display the chart
        fig.update_layout(xaxis=dict(title=''), yaxis=dict(title='Price'), showlegend=True)
        fig.update_layout(margin=dict(l=0, r=0, t=30, b=0), width=800)
        st.plotly_chart(fig)

    create_prediction_interval_plot(forecast)
    
    # Create actual vs forecast chart
    def create_actual_vs_forecast_chart(data, forecast):
        st.subheader("Actual vs. Forecast")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='Actual'))
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='Forecast'))
        fig.update_layout(xaxis_title='Date', yaxis_title='Stock Price', margin=dict(t=30), width=800, height=500)
        st.plotly_chart(fig)
    create_actual_vs_forecast_chart(data, forecast)
    
    # Calculate mae, mse, rmse, r2
    def calculate_evaluation_metrics(data, forecast):
        residuals = data['Close'] - forecast['yhat']
        mae = residuals.abs().mean()
        mse = (residuals ** 2).mean()
        rmse = mse ** 0.5
        r2 = 1 - (mse / data['Close'].var())
        return mae, mse, rmse, r2
    mae, mse, rmse, r2 = calculate_evaluation_metrics(data, forecast)
    
    # Display evaluation metrics
    st.subheader("Forecast Evaluation Metrics")
    st.write("Mean Absolute Error (MAE):", mae)
    st.write("Mean Squared Error (MSE):", mse)
    st.write("Root Mean Squared Error (RMSE):", rmse)
    st.write("R-squared (R2):", r2)
    st.write("")
    
    # Error Analysis (residual plot and distribution)
    residuals = data['Close'] - forecast['yhat']
    st.subheader("Error Analysis")
    fig_res = go.Figure()
    fig_res.add_trace(go.Scatter(x=data['Date'], y=residuals, mode='lines', name='Residuals'))
    fig_res.update_layout(title='Residual Plot', xaxis_title='Date', yaxis_title='Residuals')
    st.plotly_chart(fig_res)

    fig_dist = go.Figure()
    fig_dist.add_trace(go.Histogram(x=residuals, nbinsx=30, name='Residuals'))
    fig_dist.update_layout(title='Residual Distribution', xaxis_title='Residuals', yaxis_title='Count')
    st.plotly_chart(fig_dist)
    
    # Model hyperparameter optimization
    st.subheader("Model Hyperparameter Optimization")
    st.write("For more info on each parameter, navigate to the about tab")
    
    # Define hyperparameter inputs
    seasonality_mode = st.selectbox("Seasonality Mode", ["additive", "multiplicative"], key="seasonality_mode")
    changepoint_prior_scale = st.slider("Changepoint Prior Scale", 0.001, 0.5, 0.05, key="changepoint_prior_scale")
    st.write("Seasonality Options:")
    yearly_seasonality = st.checkbox("Yearly Seasonality", value=True, key="yearly_seasonality")
    weekly_seasonality = st.checkbox("Weekly Seasonality", value=True, key="weekly_seasonality")
    daily_seasonality = st.checkbox("Daily Seasonality", value=True, key="daily_seasonality")
    
    # Fit the model with updated hyperparameters
    m_opt = Prophet(seasonality_mode=seasonality_mode,
                changepoint_prior_scale=changepoint_prior_scale)
    if yearly_seasonality:
        m_opt.add_seasonality(name='yearly', period=365, fourier_order=10)
    if weekly_seasonality:
        m_opt.add_seasonality(name='weekly', period=7, fourier_order=3)
    if daily_seasonality:
        m_opt.add_seasonality(name='daily', period=1, fourier_order=2)
    m_opt.fit(df_train)
    forecast_opt = m_opt.predict(future)

    st.write(f'Optimized Forecast chart for {n_years} year(s) into the future')
    create_forecast_chart(m_opt, forecast_opt, n_years)

    # Calculate optimized model forecasted return
    forecasted_data_opt = forecast_opt[(forecast_opt['ds'] >= forecast_start_date) & (forecast_opt['ds'] <= forecast_end_date)]
    forecasted_start_price_opt = forecasted_data_opt.iloc[0]['yhat']
    forecasted_end_price_opt = forecasted_data_opt.iloc[-1]['yhat']
    forecasted_return_opt = (forecasted_end_price_opt - forecasted_start_price_opt) / forecasted_start_price_opt * 100

    # Display optimized forecasted return
    st.subheader('Optimized Forecasted Return')
    st.write(f"Optimized forecasted return for {selected_stock} (Period: {forecast_start_date} - {forecast_end_date}): {forecasted_return_opt:.2f}%")
    st.write("")
    st.write("")

    # Error Analysis of optimized model
    residuals_opt = data['Close'] - forecast_opt['yhat']
    st.subheader("Error Analysis (Optimized Model)")
    fig_res_opt = go.Figure()
    fig_res_opt.add_trace(go.Scatter(x=data['Date'], y=residuals_opt, mode='lines', name='Residuals (Optimized Model)'))
    fig_res_opt.update_layout(title='Residual Plot (Optimized Model)', xaxis_title='Date', yaxis_title='Residuals')
    st.plotly_chart(fig_res_opt)

    fig_dist_opt = go.Figure()
    fig_dist_opt.add_trace(go.Histogram(x=residuals_opt, nbinsx=30, name='Residuals (Optimized Model)'))
    fig_dist_opt.update_layout(title='Residual Distribution (Optimized Model)', xaxis_title='Residuals', yaxis_title='Count')
    st.plotly_chart(fig_dist_opt)
        
# Tab to view actual and forecast data (optional csv download)
elif selected_tab == "Raw Data":
    # Select a stock dataset for display and prediction
    selected_stock = st.selectbox('Select a dataset to display', sp500_tickers, index=sp500_tickers.index(default_stock))

    # Choose the number of years for prediction min 1, max 4
    n_years = st.slider('Years of prediction:', 1, 4)
    period = n_years * 365
    
    # Select the start and end dates for the chart
    start_date = st.sidebar.date_input('Start Date', value=date(2015, 1, 1), min_value=date(2010, 1, 1), max_value=date.today())
    end_date = st.sidebar.date_input('End Date', value=date.today(), min_value=date(2010, 1, 1), max_value=date.today())

    # Load the selected stock data
    data = load_data(selected_stock, start_date, end_date)
    
    st.subheader('Raw Data')
    st.write(data)
    
    # Generate filename with stock ticker and date range
    filename = f"{selected_stock}_{start_date}_{end_date}_raw_data.csv"
    
    # Add download button for CSV file
    if st.button("Download CSV File"):
        st.markdown(download_csv(data, filename), unsafe_allow_html=True)
    
    # Select the 'Date' and 'Close' columns from the data DataFrame 
    df_train = data[['Date','Close']]
    # Rename the columns to match the required format for Prophet model (ds for date and y for target variable)
    df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

    # Initialize and fit the Prophet model (different data format)
    m = Prophet()
    m.fit(df_train)
    # Make future predictions over period
    future = m.make_future_dataframe(periods=period)
    forecast = m.predict(future)
    
    st.subheader('Forecast data')
    st.write(forecast)
    
    # Generate filename with date range
    filenamef = f"{selected_stock}_{start_date}_{end_date}_forecast_data.csv"
    
    # Add download button for CSV file
    if st.button("Download Forecast CSV File"):
        st.markdown(download_csv(forecast, filenamef), unsafe_allow_html=True)
    
# Tab to view stocks with highest returns over a given time frame
elif selected_tab == "Highest Returns":
    
    # Define the range of years for the dropdown menu
    years = list(range(2010, date.today().year + 1))
    
    # Select the start and end years for the chart using dropdown menus
    start_year = st.sidebar.selectbox('Start Year', years, index=5)
    end_year = st.sidebar.selectbox('End Year', years, index=len(years) - 1)
    
    # Convert the start and end years to date objects
    start_date = date(start_year, 1, 1)
    # Check if the end year is the current year
    if end_year == date.today().year:
    # If the end year is the current year, set the end_date to January 1st of the current year
        end_date = date(date.today().year, 1, 1)
    else:
    # If the end year is not the current year, set the end_date to December 31st of the selected end year
        end_date = date(end_year, 12, 31)
    
    # Create an empty dictionary to store the calculated returns for each stock
    returns = {}
    
    # Iterate through each stock and calculate return and store in dict with ticker as key
    for ticker in sp500_tickers:
        try:
            return_value = calculate_return(ticker, start_date, end_date)
            returns[ticker] = return_value
        except IndexError:
            pass
        
    # Sort the returns dictionary by values in descending order
    sorted_returns = sorted(returns.items(), key=lambda x: x[1], reverse=True)
    
    st.subheader(f"Best Performing Stocks from ({start_date} to {end_date})")
    
    # Iterate over the top 30 stocks with the highest returns and display (allow user to view chart as well)
    if len(sorted_returns) > 0:
        for i, (ticker, return_value) in enumerate(sorted_returns[:30]):
            col1, col2 = st.columns([1, 3]) # column aspect ratio 1:3
            with col1:
                st.write(f"{i+1}. {ticker}: {return_value:.2f}%")
            with col2:
                view_candlestick = st.checkbox("View Candlestick Graph", key=f"candlestick_{i}")
                if view_candlestick: # checkbox is clicked
                    data = load_data(ticker, start_date, end_date)
                    create_candlestick_chart(data)
                    st.write("", unsafe_allow_html=True)
    else:
        st.write("No data available for any stock within the specified date range.")
               
# Tab to select and view multiple stocks
elif selected_tab == "My Watchlist":
    st.subheader('My Watchlist')

    # Get the user ID (default value until login functionality added)
    user_id = 1

    # Connect to the SQLite database
    conn = sqlite3.connect('watchlist.db')
    c = conn.cursor()

    # Load stocks from the database for the current user
    c.execute("SELECT stock FROM watchlist WHERE user_id=?", (user_id,))
    saved_stocks = [row[0] for row in c.fetchall()]

    # Close the connection
    conn.close()

    # Allow multiple stocks to be selected
    portfolio_stocks = st.multiselect('Select stocks to add to your watchlist', sp500_tickers, default=saved_stocks)

    # Connect to the SQLite database
    conn = sqlite3.connect('watchlist.db')
    c = conn.cursor()

    # Save selected stocks to the database
    for stock in portfolio_stocks:
        if stock not in saved_stocks:
            c.execute("INSERT INTO watchlist (user_id, stock) VALUES (?, ?)", (user_id, stock))

    # Remove deselected stocks from the database
    for stock in saved_stocks:
        if stock not in portfolio_stocks:
            c.execute("DELETE FROM watchlist WHERE user_id=? AND stock=?", (user_id, stock))

    # Commit the changes
    conn.commit()

    # Close the connection
    conn.close()

    # Choose the number of years for prediction min 1, max 4
    n_years = st.slider('Years of prediction:', 1, 4)
    period = n_years * 365

    # Select the start and end dates for the chart
    start_date = st.sidebar.date_input('Start Date', value=date(2015, 1, 1), min_value=date(2010, 1, 1), max_value=date.today())
    end_date = st.sidebar.date_input('End Date', value=date.today(), min_value=date(2010, 1, 1), max_value=date.today())

    # For each selected stock, load data and display candlestick and forecast charts
    for stock in portfolio_stocks:
        data = load_data(stock, start_date, end_date)

        st.subheader(f"Stock: {stock}")
        create_candlestick_chart(data)

# Tab to explain more about the app and how the Prophet model / parameters works
elif selected_tab == "About":
    st.subheader('How Prophet Works')
    st.write("Prophet is a forecasting library developed by Meta's Core Data Science team. It is specifically designed for time series forecasting and provides a flexible and intuitive framework for predicting future values based on historical data. The Prophet library combines several forecasting components, including trend, seasonality, and holiday effects, to try and generate accurate forecasts. Prophet utilizes an additive model, where non-linear trends are fit with yearly, weekly, and daily seasonality, along with additional regressors. It is capable of handling missing data and outliers and can automatically detect changepoints in a given time series. The library is built on top of the Stan probabilistic programming language, which allows for efficient computation of posterior predictive distributions and uncertainty estimation.")
    st.write("**Prophet Parameters:**")
    st.write("Changepoint Prior Scale: The changepoint_prior_scale parameter controls the flexibility of the changepoints in the model. It determines how easily the model detects changes in the rate of growth or trend. Adjusting the value can help find the right balance between capturing changepoints and avoiding overfitting.")
    st.write("Additive Seasonality: In this mode, the seasonal component is added to the trend component to generate the forecast. It assumes a constant seasonal effect regardless of the trend magnitude, resulting in consistent seasonal fluctuations. Multiplicative Seasonality: In this mode, the seasonal component is multiplied by the trend component to generate the forecast. It assumes a proportional relationship between the seasonal effect and the trend magnitude, resulting in seasonal fluctuations that increase or decrease with the trend.")
    st.write("Seasonality: Refers to regular patterns or fluctuations that occur at fixed intervals within a time series. It can occur on different time scales, such as daily, weekly, monthly, or yearly. Yearly Seasonality: Represents patterns that repeat on a yearly basis, such as holidays, seasons, or annual events. Weekly Seasonality: Represents patterns that repeat on a weekly basis, such as higher sales on weekends or changes in website traffic based on weekdays. Daily Seasonality: Represents patterns that repeat within a day, such as higher activity during certain hours or recurring events within a 24-hour period. Including these seasonality components in the Prophet model captures and incorporates repeating patterns into the forecasts, improving forecast accuracy by accounting for cyclicality in the data.")
    st.write("")
    st.subheader('List of Features / How the App Works')
    st.write("- **User Interface**: Built using Streamlit, a frontend-focused Python library, providing a multipage interface for a seamless user experience.")
    st.write("- **Data Retrieval**: Connects to Yahoo Finance via yfinance to pull stock data, which is then formatted and displayed for users to view and manipulate.")
    st.write("- **News Aggregation**: Utilizes the StockNews Python module to fetch and display the latest news articles relevant to the selected stocks.")
    st.write("- **Forecasting with Prophet**: Utilizes Facebook Prophet to train a Prophet model on historical stock data, enabling the ability of stock price forecasts. Also allows users to adjust parameters in the Prophet model and view various analysis charts on the forecast.")
    st.write("- **Candlestick Charts**: Displays interactive candlestick charts of selected stocks, allowing users to analyze price patterns and trends. The tab also allows users to calculate the return of a stock during a given time period")
    st.write("- **Raw Data Access**: Provides access to the raw historical data of selected stocks, as well as the forecasted data, allowing users to perform their own analysis. Customized CSV file download with the format: (stockName_startDate_endDate_rawData or forecast)")
    st.write("- **Highest Returns**: Lists the best performing stocks over a given time period, helping users identify potential investment opportunities.")
    st.write("- **Personalized Watchlist**: Allows users to pick multiple stocks to add to their watchlist which displays the candlestick and forecast charts. Will eventually be able to allow the watchlist to persist across multiple sessions for each unique user (once a database is incorporated). This will also massively help with load times specifically for the 'Highest Returns' tab.")
    st.write("- **Built with Python and Anaconda**: This app is developed using the Python programming language and the Anaconda distribution, which provides a comprehensive environment for scientific computing and data analysis.")
    st.write("I hope you find this app helpful or at least somewhat interesting. New features will be coming soon such as implementing a database to have the watchlist persist between sessions for multiple users and it will greatly improve load times, especially for the highest returns tab. There will also be more Prophet parameters to have further control over the forecasts and more markets such as crypto, commodities, and different stocks.")

# Tab to display news related to a specific stock
elif selected_tab == "News":
    # Select a stock dataset
    selected_stock = st.selectbox('Select a dataset for prediction', sp500_tickers, index=sp500_tickers.index(default_stock))

    st.header(f'News for {selected_stock}')
    
    # Create a StockNews object for the selected stock
    sn = StockNews(selected_stock , save_news=False)
    
    # Read the RSS feed for the selected stock and get the news data as a DataFrame
    df_news = sn.read_rss()
    
    # Iterate over the first 10 (most relevant news articles) and display certain attributes
    for i in range(10):
        st.subheader(f'News Article #{i+1}')
        published_date = df_news['published'][i]
        parsed_date = datetime.datetime.strptime(published_date, "%a, %d %b %Y %H:%M:%S %z")
        formatted_date = parsed_date.strftime("%m-%d-%Y")
        st.write("Date Published: ", formatted_date)
        st.write("Title: ", df_news['title'][i])
        st.write("Summary: ", df_news['summary'][i])
        title_sentiment = df_news['sentiment_title'][i]
        st.write(f'Title Sentiment: {title_sentiment}')
        news_sentiment = df_news['sentiment_summary'][i]
        st.write(f'News Sentiment: {news_sentiment}')
