%%writefile app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema

st.set_page_config(page_title="股票支撐壓力分析器", page_icon="📈", layout="centered")

st.title("📈 股票支撐與壓力關卡判斷系統")
st.caption("自動抓取歷史 K 線數據，計算成交量密集區、波段極值與均線共振點")

input_symbol = st.text_input("請輸入台股代碼（例如 2330 或 2454）或美股代碼（例如 TSLA）:", value="2330")

def get_ticker_symbol(symbol_str):
    symbol_str = symbol_str.strip().upper()
    if symbol_str.isdigit() and len(symbol_str) == 4:
        return f"{symbol_str}.TW"
    return symbol_str

if st.button("開始計算支撐壓力", type="primary"):
    ticker = get_ticker_symbol(input_symbol)
    
    with st.spinner('正在下載股票數據並計算中...'):
        stock = yf.Ticker(ticker)
        df = stock.history(period="1y")

    if df.empty:
        st.error(f"無法取得 [{ticker}] 的歷史數據，請確認代碼是否正確（台股請填寫 4 位數字）。")
    else:
        current_price = df['Close'].iloc[-1]
        
        df['MA120'] = df['Close'].rolling(window=120).mean()
        df['20D_Low'] = df['Low'].rolling(window=20).min()
        df['Vol_MA20'] = df['Volume'].rolling(window=20).mean()
        
        high_vol_closes = df[df['Volume'] > df['Vol_MA20'] * 1.8]['Close'].tolist()
        
        order = 5
        local_mins = df.iloc[argrelextrema(df['Low'].values, np.less_equal, order=order)[0]]['Low'].tolist()
        local_maxs = df.iloc[argrelextrema(df['High'].values, np.greater_equal, order=order)[0]]['High'].tolist()

        support_pool = [df['MA120'].iloc[-1], df['20D_Low'].iloc[-1]] + high_vol_closes + local_mins
        support_pool = [x for x in support_pool if not pd.isna(x) and x < current_price]
        
        resistance_pool = high_vol_closes + local_maxs
        resistance_pool = [x for x in resistance_pool if not pd.isna(x) and x > current_price]

        def get_clusters(pool, threshold=0.015):
            pool = sorted(pool)
            if not pool: return []
            clusters, current = [], [pool[0]]
            for price in pool[1:]:
                if price <= current[-1] * (1 + threshold):
                    current.append(price)
                else:
                    clusters.append(np.mean(current))
                    current = [price]
            clusters.append(np.mean(current))
            return sorted(list(set(clusters)))

        sup_clusters = get_clusters(support_pool)
        res_clusters = get_clusters(resistance_pool)

        near_sup = max([x for x in sup_clusters if x < current_price], default=None)
        first_res = min([x for x in res_clusters if current_price < x <= current_price * 1.10], default=None)
        
        second_res = None
        if first_res:
            second_res_candidates = [x for x in res_clusters if x > first_res * 1.03]
            if second_res_candidates:
                second_res = min(second_res_candidates)

        st.write("---")
        st.subheader(f"標的：{ticker} ｜ 現價：**{current_price:.2f}**")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if near_sup:
                diff = ((near_sup - current_price) / current_price) * 100
                st.metric("🛡️ 近端支撐", f"{near_sup:.2f}", f"{diff:.1f}%")
            else:
                st.metric("🛡️ 近端支撐", "查無數據")
                
        with col2:
            if first_res:
                diff = ((first_res - current_price) / current_price) * 100
                st.metric("⚡ 第一壓力", f"{first_res:.2f}", f"+{diff:.1f}%")
            else:
                st.metric("⚡ 第一壓力", "查無數據")
                
        with col3:
            if second_res:
                diff = ((second_res - current_price) / current_price) * 100
                st.metric("🔥 第二壓力", f"{second_res:.2f}", f"+{diff:.1f}%")
            else:
                st.metric("🔥 第二壓力", "查無數據")