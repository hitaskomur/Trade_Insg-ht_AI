import streamlit as st
import yfinance as yf   
import pandas as pd 
import plotly.graph_objects as go
import google.generativeai as genai
import tempfile
import os
import json
from datetime import datetime, timedelta

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)


MODEL_NAME = "gemini-2.5-flash"
gen_model = genai.GenerativeModel(model_name=MODEL_NAME)

st.set_page_config(page_title="Trader AI", page_icon=":guardsman:", layout="wide")
st.title("AI-Powered Technical Stock Analysis Dashboard :guardsman:")
st.sidebar.header("Configurations")

tickers_input = st.sidebar.text_input("Enter stock tickers (comma-separated)", "AAPL, MSFT, GOOGL")
tickers = [ticker.strip().upper() for ticker in tickers_input.split(",") if ticker.strip()]

end_date_default = datetime.today()
start_date_default = end_date_default - timedelta(days=365)
start_date = st.sidebar.date_input("Start Date", value=start_date_default)
end_date = st.sidebar.date_input("End Date", value=end_date_default)

st.sidebar.subheader("Technical Indicators")
indicators = st.sidebar.multiselect(
    "Select Technical Indicators", options=[
    "20-Day SMA", "20-Day EMA", "RSI", "MACD", "20-Day Bollinger Bands"], 
    default=["20-Day SMA"]
    )

if st.sidebar.button("Fetch Data"):
    stock_data = {}
    for ticker in tickers:
        data = yf.download(ticker, start=start_date, end=end_date)
        if not data.empty:
            stock_data[ticker] = data
        else:
            st.warning(f"No data found for {ticker}")
    
    st.session_state["stock_data"] = stock_data
    st.success("Stock data fetched successfully for:" + ", ".join(stock_data.keys()))

if "stock_data" in st.session_state and st.session_state["stock_data"]:
    
    def analyze_ticker(ticker,data):
        fig = go.Figure(data=[
            go.Candlestick(x=data.index,
                           open=data['Open'],
                           high=data['High'],
                           low=data['Low'],
                           close=data['Close'],
                           name="Candlestick")
        ])
    
        def add_indicator(indicator):
            if indicator == "20-Day SMA":
                data['SMA20'] = data['Close'].rolling(window=20).mean()
                fig.add_trace(go.Scatter(x=data.index, y=data['SMA20'], mode='lines', name='20-Day SMA'))
            elif indicator == "20-Day EMA":
                data['EMA20'] = data['Close'].ewm(span=20, adjust=False).mean()
                fig.add_trace(go.Scatter(x=data.index, y=data['EMA20'], mode='lines', name='20-Day EMA'))
            elif indicator == "20-Day Bollinger Bands":
                sma = data['Close'].rolling(window=20).mean()
                std = data['Close'].rolling(window=20).std()    
                bb_upper = sma + 2 * std
                bb_lower = sma - 2 * std
                fig.add_trace(go.Scatter(x=data.index, y=bb_upper, mode='lines', name='BB Upper'))
                fig.add_trace(go.Scatter(x=data.index, y=bb_lower, mode='lines', name='BB Lower'))
            elif indicator == "VWAP":
                vwap = (data['Close'] * data['Volume']).cumsum() / data['Volume'].cumsum()
                fig.add_trace(go.Scatter(x=data.index, y=vwap, mode='lines', name='VWAP'))
        for ind in indicators:
            add_indicator(ind)
        fig.update_layout(xaxis_rangeslider_visible=False)

        
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
            fig.write_image(tmpfile.name)
            tmpfile_path = tmpfile.name
        with open(tmpfile_path, "rb") as f:
            image_bytes = f.read()
        os.remove(tmpfile_path)

        image_part = {
            "data":image_bytes,
            "mime_type":"image/png"
        }

        analysis_prompt = (
        f"You are a Stock Trader specialist in Technical Analysis at a top financial institution."
        f"Analyze the stock data for {ticker} based on its candlestick chart and the displayed technical indicators."
        f"Provide a detailed justification for your analysis, explain what patterns, signals, and trends you observe."
        f"Then, based solely on the chart, provide a recommendation from the following options:"
        f"1. Strong Buy"
        f"2. Buy"
        f"3. Weak Buy"
        f"4. Hold"
        f"5. Sell"
        f"6. Strong Sell"
        f"Return your output as a JSON object with two keys: 'action' and 'justification'."
        )

        contents = [
            {"role":"user","parts": [analysis_prompt]},
            {"role":"user","parts": [image_part]}
        ]

        response = gen_model.generate_content(
            contents=contents
        )

        try:
            result_text = response.text

            json_start_index = result_text.find('{')
            json_end_index = result_text.rfind('}') + 1
            if json_start_index != -1 and json_end_index > json_start_index:
                json_str = result_text[json_start_index:json_end_index]
                result_json = json.loads(json_str)
            else:
                raise ValueError("Invalid JSON response")
        except json.JSONDecodeError as e:
            result_json = {"action": "error", "justification": f"JSON Parsing Error: {str(e)}"}
        except ValueError as ve:
            result_json = {"action": "error", "justification": f"Value Error: {str(ve)}"}
        except Exception as ex:
            result_json = {"action": "error", "justification": f"Unexpected Error: {str(ex)}"}
        
        return fig, result_json
    
    tab_names = ["Overall Summary"]+ list(st.session_state["stock_data"].keys())
    tabs = st.tabs(tab_names)

    overall_results = []

    for i,ticker in enumerate(st.session_state["stock_data"]):
        data = st.session_state["stock_data"][ticker]
        fig, result_json = analyze_ticker(ticker)
        overall_results.append({"Stock": ticker, "Recommendation": result_json.get("action", "N/A")})

        with tabs[i+1]:
            st.subheader(f"Technical Analysis for {ticker}")
            st.plotly_chart(fig)
            st.write("**Detailed Justification**")
            st.write(result_json.get("justification", "No justification provided."))

        with tabs[0]:
            st.subheader("Overall Structured Recommendation")
            overall_df = pd.DataFrame(overall_results)
            st.table(overall_df)

else:
    st.info("Enter stock tickers and click 'Fetch Data' to begin analysis.")