# pip install streamlit prophet yfinance plotly
import streamlit as st
from datetime import date

import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as plt

START = "2010-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title('Stock Prediction App')

selected_stock = st.text_input('Enter a stock ticker for prediction', 'GOOGL').upper()

no_of_years = st.slider('Years of prediction:', 1, 4)
period = no_of_years * 365


@st.cache_resource
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

	
data = load_data(selected_stock)

st.subheader('Current data')
st.write(data.tail())

# Plot raw data
def plot_raw_data():
	fig = plt.Figure()
	fig.add_trace(plt.Scatter(x=data['Date'], y=data['Open'], name="stock opening price", line=dict(color='red')))
	fig.add_trace(plt.Scatter(x=data['Date'], y=data['Close'], name="stock closing price", line=dict(color='green')))
	fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
	st.plotly_chart(fig)
	
plot_raw_data()

# Predict closing price of stock with Prophet.
training_data = data[['Date','Close']]
training_data = training_data.rename(columns={"Date": "ds", "Close": "y"}) # Rename columns to meet the standard of Prophet

model = Prophet()
model.fit(training_data)
future_closing_price = model.make_future_dataframe(periods=period)
forecast = model.predict(future_closing_price)

# Show and plot forecast
st.subheader('Predicted data')
st.write(forecast.tail())

# generate the chart
fig1 = plot_plotly(model, forecast)

# Customize the line colors
# Change the color of the 'Predicted' line
fig1.data[2]['line']['color'] = 'green'

# Change the color of the 'Actual' line
fig1.data[0]['marker']['color'] = 'red'

fig1.update_layout(title=f'Predicted closing price for the next {no_of_years} year(s)', xaxis_title="Date", yaxis_title="Closing Price")

st.plotly_chart(fig1)

st.write("Components of predicted data")
fig2 = model.plot_components(forecast)
st.write(fig2)