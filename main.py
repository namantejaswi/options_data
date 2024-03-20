import yfinance as yf
import pandas as pd
import streamlit as st
from scipy.stats import percentileofscore
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
from datetime import datetime, date
import numpy as np
from scipy.stats import norm


OG_Tickers = ["^VIX", "AAPL", "NVDA", "AMD", "TSLA", "MSFT", "GOOGL", "AMZN", "INTC", "QCOM", "ADBE","^IXIC","^GSPC"]

# Function to get price history for a given ticker and period
def get_price_history(ticker, period="10y"):
    stock = yf.Ticker(ticker)
    return stock.history(period=period)

# Function to plot the stock price
def plot_ticker(ticker, period,is_candle):
    df = get_price_history(ticker, period)
    st.write( "###", yf.Ticker(ticker).info['longName'])
    
    
    if not is_candle:  # st.line_chart(df['Close'])
        with st.expander('Show Price Line Chart'):
    
            fig = px.line(df, x=df.index, y='Close', title=f'{ticker} Stock Price')
            st.plotly_chart(fig, use_container_width=True)

    elif is_candle:
        with st.spinner('Show Candlestick Chart'):
            plot_candlestick_chart(df)


    with st.expander('Show Volume'):
        st.bar_chart(df['Volume'])


def plot_candlestick_chart(df):
    fig = go.Figure(data=[go.Candlestick(x=df.index,
                                         open=df['Open'],
                                         high=df['High'],
                                         low=df['Low'],
                                         close=df['Close'],
                                         increasing_line_color='green',
                                         decreasing_line_color='red')])

    fig.update_layout(title=f'Candlestick Chart for {df.index[0].strftime("%B %d, %Y")} to {df.index[-1].strftime("%B %d, %Y")}',
                      xaxis_title='Date',
                      yaxis_title='Stock Price',
                      xaxis_rangeslider_visible=True)

    st.plotly_chart(fig, use_container_width=True)


def calculate_vix_percentile(period="1y"):
    vix_ticker = yf.Ticker("^VIX")
    vix_data = vix_ticker.history(period)
    current_vix = vix_data["Close"].iloc[-1]
    percentile = percentileofscore(vix_data["Close"], current_vix)
    return percentile


#fallback to not get none during consecutive holidays
def get_last_price(ticker, fallback_period="5d"):
    stock = yf.Ticker(ticker)
    history_data = stock.history(period=fallback_period)
    
    if not history_data.empty:
        last_close_price = history_data["Close"].iloc[-1]
        if pd.notna(last_close_price):
            return last_close_price
        else:
            # If last close price is None, get the second-to-last close price
            second_to_last_close_price = history_data["Close"].iloc[-2]
            return second_to_last_close_price
    
    else: return "N/A"


def financials(ticker):  
    df = yf.Ticker(ticker)

    balance_sheet = df.get_balance_sheet()
    cash_flow = df.get_cash_flow()



    #print(balance_sheet, cash_flow)

    with st.expander('Show Financials'):
        st.write("Balance Sheet",balance_sheet)
       
        if "marketCap" in df.info:
            market_cap_dollars = df.info["marketCap"]
            # Convert market cap to billions
            market_cap_trillions = market_cap_dollars / 1e12

            # Format market cap in billions with commas as thousands separators
            formatted_market_cap = "{:,.4f}".format(market_cap_trillions)

            # Display the formatted market cap in billions
            st.write("Market Cap", market_cap_dollars,"USD  i.e. ", formatted_market_cap, " Trilion USD")



        df = balance_sheet
        financial_data = pd.DataFrame(df)    

        # Selecting rows to plot
        st.session_state.selected_rows = st.multiselect('Select Financial metrics to Plot', financial_data.index,key="financials"+ticker)
        if st.session_state.selected_rows:
            
            # Plot the selected rows
            fig = go.Figure()
            for row in st.session_state.selected_rows:
                fig.add_trace(go.Scatter(x=financial_data.columns, y=financial_data.loc[row], mode='lines', name=row))
            fig.update_layout(title='Financial Metrics', xaxis_title='Metrics', yaxis_title='Values')
            st.plotly_chart(fig)

        st.write("cash flow",cash_flow)


def holders(ticker):

    df = yf.Ticker(ticker)

    with st.expander('Show Institutional Holders'):
        st.write(df.institutional_holders)
        

    with st.expander('Show Major Holders'):

        df = pd.DataFrame(df.major_holders)
        
        if df.empty:
            st.write("No major holders data available")
            return
        
        
        x =[df.iloc[0, 0],df.iloc[1, 0]]
        y = [df.index[0],df.index[1]]
       

        # Combine data into a DataFrame
        x.append(1-sum(x))
        
        y= [value for value in y]
        y.append('Open Market')

        data = {'Percentage': x, 'Label': y}
        df_share = pd.DataFrame(data)
            # Create a pie chart using Plotly Express
        fig = px.pie(df_share, values='Percentage', names='Label', title='Pie Chart of share holding', color_discrete_sequence=["#FF5733", "#28B463", "#2B66E3"])

        # Customize the layout if needed
        fig.update_layout(
            margin=dict(l=0, r=0, t=30, b=0),
            showlegend=True
        )

        # Display the pie chart
        st.plotly_chart(fig, use_container_width=True)



def get_calendar_events(ticker):
    df = yf.Ticker(ticker)
    calender = df.calendar


    with st.expander('Show Calendar Events'):
        for key, value in calender.items():
            st.write(f"{key}: {value}")


def get_actions(ticker):
    df = yf.Ticker(ticker)
    actions = df.actions

    with st.expander('Show Corporate Actions'):
        st.write(actions)


def get_info(ticker):           
    df = yf.Ticker(ticker)
    info = df.info
    
    with st.expander('Show Company Information'):
        for key, value in info.items():
            st.write(f"{key}: {value}")


def ratios(ticker):
    df = yf.Ticker(ticker)
    
    ratios = {}

    for key, value in df.info.items():
        if key in ['ebitda', 'trailingPE', 'forwardPE', 'priceToBook', 'pegRatio', 'dividendYield', 'payoutRatio',
                   'trailingEps', 'forwardEps', 'debtToEquity', 'returnOnEquity', 'revenuePerShare', 'returnOnEquity',
                   'freeCashflow', 'operatingCashflow', 'totalCash', 'totalCashPerShare', 'quickRatio', 'currentRatio',
                   'bookValue', 'enterpriseValue', 'earningsGrowth', 'revenueGrowth', 'grossMargins', 'ebitdaMargins',
                   'operatingMargins', 'fiftyDayAverage', 'twoHundredDayAverage']:
            ratios[key] = value

    formatted_ratios = {key: f"{value:.2f}" if isinstance(value, (int, float)) else value for key, value in ratios.items()}

    with st.expander('Financial Ratios'):
        st.table(pd.DataFrame(formatted_ratios.items(), columns=['Ratio', 'Value']))



def options(ticker):

    tkr = yf.Ticker(ticker)
    
    with st.sidebar:
        st.title('Options Data' )
        
        dates = tkr.options

        st.write("###", yf.Ticker(ticker).info['longName'])
        st.session_state.expiry_date=st.selectbox("Choose your expiry",dates, index=0,key="expiry_date"+ticker)
        
        st.session_state.opt = tkr.option_chain(st.session_state.expiry_date)
        

        last_ticker_price = get_last_price(ticker)

        #find at the money call and put options

        st.session_state.at_the_money_call = st.session_state.opt.calls[st.session_state.opt.calls['strike'] > last_ticker_price].iloc[0]

        #closest put less than last ticker price
        st.session_state.at_the_money_put = st.session_state.opt.puts[st.session_state.opt.puts['strike'] < last_ticker_price].iloc[-1]
        
        st.write("#### At the money put ")
        st.dataframe(st.session_state.at_the_money_put,width=400) 
        st.write("#### At the money call ")
        st.dataframe(st.session_state.at_the_money_call,width=400)


        st.write('#### Last Price',(round(last_ticker_price,2)), ' Call Strike',st.session_state.at_the_money_call['strike'],' Put Strike',st.session_state.at_the_money_put['strike'])

        st.write('#### Difference between call atm strike and last price',round(st.session_state.at_the_money_call['strike']-last_ticker_price,4),round((st.session_state.at_the_money_call['strike']-last_ticker_price)/last_ticker_price*100,4), "%  price")
        st.write('#### Difference between put atm strike and last price',round(last_ticker_price-st.session_state.at_the_money_put['strike'],4),round((last_ticker_price-st.session_state.at_the_money_put['strike'])/last_ticker_price*100,4), "% price")



        st.session_state.iv_c=(round(st.session_state.at_the_money_call['impliedVolatility'],4))
        st.write('#### Implied Volatility atm call', st.session_state.iv_c)
        st.session_state.iv_p=(round(st.session_state.at_the_money_put['impliedVolatility'],4))
        st.write('#### Implied Volatility atm put', st.session_state.iv_p) 

        st.session_state.iv = (st.session_state.iv_c+st.session_state.iv_p)/2

        st.write('#### Straddle Price', round(st.session_state.at_the_money_call['lastPrice']+st.session_state.at_the_money_put['lastPrice'],4))

        date_1 = st.session_state.expiry_date
        date_obj1 = datetime.strptime(date_1, "%Y-%m-%d")
        date_str2 = date.today().strftime("%Y-%m-%d")
        date_obj2 = datetime.strptime(date_str2, "%Y-%m-%d")

        diff = date_obj1 - date_obj2

        days_to_expiration = diff.days

        calculate_approx_expected_move(ticker,get_last_price(ticker),st.session_state.iv,days_to_expiration)

        #st.write("Atm call bid ask spread",round(st.session_state.at_the_money_call['ask']-st.session_state.at_the_money_call['bid'],4))
        #st.write("Atm put bid ask spread",round(st.session_state.at_the_money_put['ask']-st.session_state.at_the_money_put['bid'],4))

        st.session_state.plot_gaussian = st.toggle("Visualize Gaussian", False,key="gaussian"+ticker+st.session_state.expiry_date)
        if st.session_state.plot_gaussian:
            plot_gaussian(last_ticker_price, st.session_state.one_sd)


        st.session_state.plot_past_daily_returns = st.toggle("Visualize Past Daily Returns", False,key="past_daily_returns"+ticker+st.session_state.expiry_date)
        
        if st.session_state.plot_past_daily_returns:
            plot_past_daily_returns(ticker, "10y")

        st.write("#### Call Options",st.session_state.opt.calls)
        st.write("#### Put Options",st.session_state.opt.puts)




def calculate_approx_expected_move(ticker,last_price, iv,days_to_expiration):
    
    #expected move = last price * iv * sqrt(days to expiration/365)
    st.write("#### Days to Expiration", days_to_expiration)

    st.session_state.one_sd = round(last_price*iv* ((days_to_expiration/365)**0.5),2)
    st.write("#### Expected move 1 Standard Deviation", st.session_state.one_sd)
    st.write("#### Expected move +-2 SD", 2*st.session_state.one_sd, "+- 3 SD", 3*st.session_state.one_sd)
    

def plot_gaussian(mu, sigma):
    x = np.linspace(mu - 4*sigma, mu + 4*sigma, 100)
    y = norm.pdf(x, mu, sigma)
    
    fig, ax = plt.subplots()
    ax.plot(x, y, label='Gaussian Distribution')

    # Define colors for ±1, ±2, and ±3 standard deviations
    colors = ['green', 'orange', 'red']
    

    # Plot lines for ±1, ±2, and ±3 standard deviations with different colors
    for i, color in zip(range(1, 4), colors):
        ax.axvline(mu + i*sigma, color=color, linestyle='--')
        ax.axvline(mu - i*sigma, color=color, linestyle='--')
        ax.text(mu + i*sigma, 0.005, f'+{i} SD', color=color, ha='center')
        ax.text(mu - i*sigma, 0.005, f'-{i} SD', color=color, ha='center')

    # Calculate percentages within ±1, ±2, and ±3 standard deviations
    percentages = [norm.cdf(mu + i*sigma, mu, sigma) - norm.cdf(mu - i*sigma, mu, sigma) for i in range(1, 4)]

    # Display percentages and stock prices on the plot
    for i, (percent, sd) in enumerate(zip(percentages, range(1, 4)), 1):
        stock_price = mu + sd * sigma
        ax.text(stock_price, 0.02, f'{percent*100:.2f}%', color=colors[i-1], ha='center')
        ax.text(stock_price, -0.02, f'${stock_price:.2f}', color=colors[i-1], ha='center', va='top')

    # Display percentages and stock prices on the left side
    for i, (percent, sd) in enumerate(zip(percentages, range(1, 4)), 1):
        stock_price = mu - sd * sigma
        ax.text(stock_price, 0.02, f'{percent*100:.2f}%', color=colors[i-1], ha='center')
        ax.text(stock_price, -0.02, f'${stock_price:.2f}', color=colors[i-1], ha='center', va='top')

    ax.set_title('Gaussian Distribution with ±1, ±2, and ±3 SD')
    ax.set_xlabel('Stock Price (X-axis)')
    ax.set_ylabel('Probability Density Function')

    # Display the plot using Streamlit
    st.pyplot(fig)


#we plot the daily percentage change of stock over last 10 years to see if it resembles a normal distribution
def plot_past_daily_returns(ticker, period="10y"):
    df = get_price_history(ticker, period)
    df['Percent Change'] = df['Close'].pct_change()*100

        # Create a Plotly histogram
    fig = px.histogram(df, x='Percent Change', nbins=50,
                       title=f'Frequency of Daily Percent Change 10 years - {ticker}',
                       labels={'Percent Change': 'Daily Percent Change', 'count': 'Frequency'},
                       marginal='box', color_discrete_sequence=['blue'])

    # Overlay a perfect Gaussian distribution curve
    mean_val = 0 
    std_val = df['Percent Change'].std()
    
    x_values = np.linspace(df['Percent Change'].min(), df['Percent Change'].max(), len(df['Percent Change'])-1)
    st.write("total x values",len(x_values))
    bin_width = (df['Percent Change'].max() - df['Percent Change'].min()) / 50
    y_values = norm.pdf(x_values, mean_val, std_val) * len(df) * bin_width  # Scale by bin width

    # Add Gaussian distribution curve using Scatter trace
    fig.add_trace(go.Scatter(x=x_values, y=y_values, mode='lines', line=dict(color='red'), name='Gaussian Distribution'))

    # Display the plot using Streamlit
    st.plotly_chart(fig)



def main():


    st.write("### Current Volatility Index COBE (^VIX) is",get_last_price("^VIX").round(2)  )

    vix_periods = ["5d", "10d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y"]
    st.session_state.vix_time = st.selectbox("Select Time Frame",vix_periods,index=5)

    st.write("### Current VIX Percentile is or IVR", calculate_vix_percentile(st.session_state.vix_time).round(2))
    
    #fstring to 2 decimal places
    st.write(f"##### This means that the vix is higher than {calculate_vix_percentile(st.session_state.vix_time):.2f}% of the time in the past {st.session_state.vix_time} ")


    with st.expander('Show VIX Chart', expanded=False):
        vix_df = get_price_history("^VIX",st.session_state.vix_time)
        vix_df.drop(columns=['Dividends','Stock Splits','Volume'],inplace=True)
        st.line_chart(vix_df)

    st.session_state.is_candle =st.toggle("Use candlestick chart", False,key="candlestick")

    # Allow user to select time frame
    periods = ["5d", "10d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y"]
    # Allow user to select multiple stock symbols
    st.session_state.selected_tickers = st.multiselect("Select Stock Symbols", OG_Tickers,key="tickers")
    st.session_state.selected_period = st.selectbox("Select Time Frame", periods, key="period",index=5)


    for selected_ticker in st.session_state.selected_tickers:
       

        plot_ticker(selected_ticker, st.session_state.selected_period,st.session_state.is_candle)
        financials(selected_ticker)

        
        #st.write("Word on the street is",df.get_news())
        ratios(selected_ticker)

        holders(selected_ticker)
        get_calendar_events(selected_ticker)
        get_actions(selected_ticker)
        #get_info(selected_ticker)
        options(selected_ticker)


    custom_ticker = st.text_input("Enter Custom Ticker", "SPY", key="custom_ticker")
    plot_ticker(custom_ticker, st.session_state.selected_period,st.session_state.is_candle)
    financials(custom_ticker)
    ratios(custom_ticker)
    holders(custom_ticker)
    get_calendar_events(custom_ticker)
    get_actions(custom_ticker)
    options(custom_ticker)



if __name__ == "__main__":
    main()
