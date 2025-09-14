import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import plotly.express as px

st.set_page_config(page_title="Portfolio Dashboard", layout="wide")
st.title("📊 Advanced Portfolio & Stock Dashboard")


st.sidebar.header("Portfolio Inputs")
tickers_input = st.sidebar.text_input("Tickers (comma separated)", "AAPL,MSFT,GOOGL")
years = st.sidebar.number_input("Years of Data", min_value=1, max_value=15, value=5)
rf = st.sidebar.number_input("Risk Free Rate (decimal)", min_value=0.0, max_value=0.2, value=0.05, step=0.01)
Tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

if len(Tickers) > 0:

    data = yf.download(Tickers, period=f"{years}y", auto_adjust=True, threads=True)
    if isinstance(data.columns, pd.MultiIndex):
        adj_close = data['Close']
    else:
        adj_close = data.to_frame(name=Tickers[0])

    monthly_returns = adj_close.resample("M").last().pct_change().dropna()
    annual_returns = ((1 + monthly_returns.mean()) ** 12) - 1
    cov_matrix = monthly_returns.cov()
    annual_volatility = monthly_returns.std() * np.sqrt(12)
    sharpe_ratio = (annual_returns - rf) / annual_volatility

   ####################################### Max Sharpe Portfolio (Monte Carlo)
    num_portfolios = 5000
    results = np.zeros((3, num_portfolios))
    weights_record = []
    for i in range(num_portfolios):
        w = np.random.random(len(Tickers))
        w /= np.sum(w)
        weights_record.append(w)
        port_return = np.sum(w * annual_returns)
        port_vol = np.sqrt(np.dot(w.T, np.dot(cov_matrix * 12, w)))
        results[0, i] = port_vol
        results[1, i] = port_return
        results[2, i] = (port_return - rf) / port_vol

    max_sharpe_idx = np.argmax(results[2])
    max_sharpe_weights = weights_record[max_sharpe_idx]
    max_sharpe_return = results[1, max_sharpe_idx]
    max_sharpe_vol = results[0, max_sharpe_idx]
#########################################
 ############################################################# Logistic Regression for each stock
    prob_up_dict = {}
    trend_dict = {}
    for ticker in Tickers:
        df = adj_close[[ticker]].copy()
        df['return'] = df[ticker].pct_change()
        df['ma3'] = df[ticker].rolling(3).mean().pct_change()
        df['ma6'] = df[ticker].rolling(6).mean().pct_change()
        df['volatility'] = df['return'].rolling(3).std()
        df['target'] = (df['return'].shift(-1) > 0).astype(int)
        df.dropna(inplace=True)

        X = df[['return', 'ma3', 'ma6', 'volatility']]
        y = df['target']
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)
        model = LogisticRegression(solver="lbfgs", max_iter=1000)
        model.fit(X_train, y_train)

        latest_scaled = scaler.transform(X.iloc[[-1]])
        prob_up = model.predict_proba(latest_scaled)[0][1]
        prob_up_dict[ticker] = prob_up
        trend_dict[ticker] = df['return'].apply(lambda x: 1 if x > 0 else 0).rolling(3).mean()
###############################################################
    tab1, tab2, tab3, tab4 = st.tabs(["Stock Metrics", "Max Sharpe Portfolio", "ML Predictions", "Backtest"])

    with tab1:
        st.subheader("📊 Individual Stock Metrics")
        metrics_df = pd.DataFrame({
            "Expected Annual Return (%)": (annual_returns * 100).round(2),
            "Annual Volatility (%)": (annual_volatility * 100).round(2),
            "Sharpe Ratio": sharpe_ratio.round(2)
        })
        st.dataframe(metrics_df.style.highlight_max(subset=["Sharpe Ratio"], color="lightgreen"))

    with tab2:
        st.subheader("📈 Efficient Frontier & Portfolio Allocation")
        frontier_df = pd.DataFrame({
            "Volatility": results[0, :],
            "Return": results[1, :],
            "Sharpe": results[2, :]
        })
        fig = px.scatter(frontier_df, x="Volatility", y="Return", color="Sharpe",
                         title="Efficient Frontier",
                         labels={"Volatility": "Annualized Volatility", "Return": "Expected Return"})
        fig.add_scatter(x=[max_sharpe_vol], y=[max_sharpe_return], mode="markers",
                        marker=dict(size=15, color="red"), name="Max Sharpe")

    ################################### Highlight high probability stocks
        for i, ticker in enumerate(Tickers):
            if prob_up_dict[ticker] > 0.7:
                fig.add_annotation(x=annual_volatility[ticker], y=annual_returns[ticker],
                                   text=f"{ticker} ↑", showarrow=True, arrowhead=3,
                                   arrowsize=1, arrowcolor="green")
        st.plotly_chart(fig, use_container_width=True)

     ######################################
        alloc_df = pd.DataFrame({"Ticker": Tickers, "Weight": max_sharpe_weights})
        fig2 = px.pie(alloc_df, values='Weight', names='Ticker',
                      title='Max Sharpe Portfolio Allocation')
        st.plotly_chart(fig2)

    with tab3:
        st.subheader("📈 Stock Movement Probability Trends")
        for ticker in Tickers:
            st.line_chart(trend_dict[ticker], use_container_width=True, height=200)

        prob_df = pd.DataFrame({
            "Ticker": Tickers,
            "Weight (%)": [round(w * 100, 2) for w in max_sharpe_weights],
            "Prob Up Next Month (%)": [round(prob_up_dict[t] * 100, 2) for t in Tickers]
        })
        st.dataframe(prob_df)

    with tab4:
        st.subheader("📊 Historical Backtest")
        port_rets = adj_close.pct_change().dropna() @ max_sharpe_weights
        cumulative_returns = (1 + port_rets).cumprod()

        st.line_chart(cumulative_returns, use_container_width=True)
        st.metric("Expected Portfolio Return (%)", round(max_sharpe_return * 100, 2))
        st.metric("Portfolio Volatility (%)", round(max_sharpe_vol * 100, 2))
        st.metric("Portfolio Sharpe Ratio", round(results[2, max_sharpe_idx], 2))
