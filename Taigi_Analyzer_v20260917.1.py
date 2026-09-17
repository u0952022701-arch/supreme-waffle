import sys
import pandas as pd
import numpy as np

def run_app():
    import streamlit as st
    import yfinance as yf
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
            
            # =========================================================================
            # 技術指標與結構候選池計算
            # =========================================================================
            
            # 1. MA120 (半年線)
            df['MA120'] = df['Close'].rolling(window=120).mean()
            
            # 2. 近 20 日低點
            df['20D_Low'] = df['Low'].rolling(window=20).min()
            
            # 3. 爆量 K 收盤 (成交量大於 20 日均量 1.8 倍之收盤價)
            df['Vol_MA20'] = df['Volume'].rolling(window=20).mean()
            high_vol_closes = df[df['Volume'] > df['Vol_MA20'] * 1.8]['Close'].tolist()
            
            # 4. 波段極值 (前高轉支撐 / 波段低點 / 波段高點)
            order = 5
            local_mins = df.iloc[argrelextrema(df['Low'].values, np.less_equal, order=order)[0]]['Low'].tolist()
            local_maxs = df.iloc[argrelextrema(df['High'].values, np.greater_equal, order=order)[0]]['High'].tolist()

            # -------------------------------------------------------------------------
            # 【近端支撐 候選池】
            # 包含：成交量密集區 / 爆量 K 收盤 / 前高轉支撐 / MA120 / 波段低點 / 近 20 日低點
            # -------------------------------------------------------------------------
            support_pool = [x for x in [df['MA120'].iloc[-1], df['20D_Low'].iloc[-1]] + high_vol_closes + local_mins if not pd.isna(x) and x < current_price]
            
            # -------------------------------------------------------------------------
            # 【壓力 候選池】
            # -------------------------------------------------------------------------
            resistance_pool = [x for x in high_vol_closes + local_maxs if not pd.isna(x) and x > current_price]

            # 結構共振區分群演算法 (將相近價位聚合為代表性區塊)
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

            # =========================================================================
            # 支撐與壓力核心判斷邏輯
            # =========================================================================
            
            # 【近端支撐】
            # 成交量密集區 / 爆量 K 收盤 / 前高轉支撐 / MA120 / 波段低點 / 近 20 日低點
            # 現價之下最接近的結構價位；跌破後結構轉弱。距離以現價為基準計算。
            near_sup = max([x for x in sup_clusters if x < current_price], default=None)
            
            # 【第一壓力】
            # 現價上方 10% 以內、結構共振最具代表性的第一個壓力區。
            first_res = min([x for x in res_clusters if current_price < x <= current_price * 1.10], default=None)
            
            # 【第二壓力】
            # 第一壓力之上、真正獨立的下一個壓力共振區。
            second_res = min([x for x in res_clusters if x > (first_res * 1.03 if first_res else current_price)], default=None)

            # =========================================================================
            # UI 卡片結果顯示
            # =========================================================================
            st.write("---")
            st.subheader(f"標的：{ticker} ｜ 現價：**{current_price:.2f}**")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    label="🛡️ 近端支撐", 
                    value=f"{near_sup:.2f}" if near_sup else "查無數據", 
                    delta=f"{((near_sup-current_price)/current_price)*100:.1f}%" if near_sup else "",
                    help="成交量密集區 / 爆量 K 收盤 / 前高轉支撐 / MA120 / 波段低點 / 近 20 日低點\n\n現價之下最接近的結構價位；跌破後結構轉弱。距離以現價為基準計算。"
                )
                st.caption("現價之下最接近的結構價位；跌破後結構轉弱。")
                
            with col2:
                st.metric(
                    label="⚡ 第一壓力", 
                    value=f"{first_res:.2f}" if first_res else "查無數據", 
                    delta=f"+{((first_res-current_price)/current_price)*100:.1f}%" if first_res else "",
                    help="現價上方 10% 以內、結構共振最具代表性的第一個壓力區。"
                )
                st.caption("現價上方 10% 以內、結構共振最具代表性。")
                
            with col3:
                st.metric(
                    label="🔥 第二壓力", 
                    value=f"{second_res:.2f}" if second_res else "查無數據", 
                    delta=f"+{((second_res-current_price)/current_price)*100:.1f}%" if second_res else "",
                    help="第一壓力之上、真正獨立的下一個壓力共振區。"
                )
                st.caption("第一壓力之上、真正獨立的下一個壓力區。")

if __name__ == "__main__":
    import streamlit as st
    from streamlit.web import cli as stcli

    if not st.runtime.exists():
        sys.argv = ["streamlit", "run", __file__]
        sys.exit(stcli.main())
    else:
        run_app()