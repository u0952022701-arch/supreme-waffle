import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema

# 頁面基本設定
st.set_page_config(page_title="股票支撐壓力分析器", layout="centered")

# --- 樣式設定 (美化為圖片中的暗色風格) ---
st.markdown("""
<style>
    .main { background-color: #0E1117; }
    .card-support {
        background-color: #1A2234;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 15px;
        border: 1px solid #2B3A55;
    }
    .card-resistance1 {
        background-color: #1A2234;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 15px;
        border: 1px solid #2B3A55;
    }
    .card-resistance2 {
        background-color: #1A2234;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 15px;
        border: 1px solid #2B3A55;
    }
    .card-title { font-size: 18px; font-weight: bold; color: #E0E0E0; }
    .card-subtitle { font-size: 13px; color: #8C9BAE; margin-bottom: 10px; }
    .price-value { font-size: 32px; font-weight: bold; color: #FFFFFF; text-align: right; }
    .price-diff { font-size: 14px; text-align: right; }
    .diff-neg { color: #4CAF50; } /* 綠色為負或跌，可依習慣調整 */
    .diff-pos { color: #FF5252; } /* 紅色為正或漲 */
</style>
""", unsafe_allow_html=True)

st.title("📈 股票支撐壓力分析系統")


# --- 自動處理台股與美股代號轉換 ---
def fix_stock_symbol(symbol):
    symbol = symbol.strip().upper()
    # 如果純數字，預設加上台股上市 .TW (若無資料後續可切換 .TWO)
    if symbol.isdigit():
        return f"{symbol}.TW"
    return symbol


# --- 核心計算邏輯 ---
def calculate_levels(df, current_price):
    df['MA120'] = df['Close'].rolling(window=120).mean()
    df['20D_Low'] = df['Low'].rolling(window=20).min()
    df['Vol_MA20'] = df['Volume'].rolling(window=20).mean()

    # 爆量K線價位 (成交量 > 20日均量 2 倍)
    high_vol_closes = df[df['Volume'] > df['Vol_MA20'] * 2]['Close'].tolist()

    # 波段高低點
    order = 5
    local_mins = df.iloc[argrelextrema(df['Low'].values, np.less_equal, order=order)[0]]['Low'].tolist()
    local_maxs = df.iloc[argrelextrema(df['High'].values, np.greater_equal, order=order)[0]]['High'].tolist()

    # 支撐池與壓力池
    support_pool = [df['MA120'].iloc[-1], df['20D_Low'].iloc[-1]] + high_vol_closes + local_mins
    support_pool = [x for x in support_pool if not pd.isna(x) and x < current_price]

    resistance_pool = high_vol_closes + local_maxs
    resistance_pool = [x for x in resistance_pool if not pd.isna(x) and x > current_price]

    # 共振區聚合
    def get_resonance(pool, threshold=0.015):
        pool = sorted(pool)
        if not pool: return []
        clusters, curr = [], [pool[0]]
        for p in pool[1:]:
            if p <= curr[-1] * (1 + threshold):
                curr.append(p)
            else:
                clusters.append(np.mean(curr))
                curr = [p]
        clusters.append(np.mean(curr))
        return sorted(list(set(clusters)))

    resonated_supports = get_resonance(support_pool)
    resonated_resistances = get_resonance(resistance_pool)

    # 計算近端支撐、第一壓力、第二壓力
    near_supp = max([x for x in resonated_supports if x < current_price], default=None)
    first_res = min([x for x in resonated_resistances if current_price < x <= current_price * 1.10], default=None)

    second_res = None
    if first_res:
        second_candidates = [x for x in resonated_resistances if x > first_res * 1.03]
        if second_candidates:
            second_res = min(second_candidates)

    return near_supp, first_res, second_res


# --- 介面輸入區 ---
input_symbol = st.text_input("請輸入股票代號（例如：2330、2454、AAPL）：", value="2330")

if input_symbol:
    symbol = fix_stock_symbol(input_symbol)

    # 抓取 K 線資料
    data = yf.download(symbol, period="1y", progress=False)

    # 如果 .TW 抓不到，嘗試 上櫃 .TWO
    if data.empty and symbol.endswith(".TW"):
        symbol = symbol.replace(".TW", ".TWO")
        data = yf.download(symbol, period="1y", progress=False)

    if data.empty:
        st.error(f"查無股票代號 `{input_symbol}` 的資料，請確認後重試。")
    else:
        # 處理多重索引 (yfinance 近期版本問題)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        current_price = float(data['Close'].iloc[-1])
        near_supp, first_res, second_res = calculate_levels(data, current_price)

        st.markdown(f"### 現價：**{current_price:.2f}**")
        st.markdown("---")

        # 1. 近端支撐卡片
        if near_supp:
            diff_supp = ((near_supp - current_price) / current_price) * 100
            st.markdown(f"""
            <div class="card-support">
                <div style="display: flex; justify-content: space-between;">
                    <div>
                        <div class="card-title">🛡️ 支撐觀察</div>
                        <div style="font-weight: bold; color:#4A90E2; margin-top:5px;">近端支撐</div>
                        <div class="card-subtitle">成交量密集區 / 爆量 K 收盤 / 前高轉支撐 / MA120 / 波段低點 / 近 20 日低點</div>
                    </div>
                    <div>
                        <div class="price-value">{near_supp:.2f}</div>
                        <div class="price-diff diff-neg">距現價 {diff_supp:+.1f}%</div>
                    </div>
                </div>
                <div class="card-subtitle" style="margin-bottom:0;">現價之下最接近的結構價位；跌破後結構轉弱。距離以現價為基準計算。</div>
            </div>
            """, unsafe_allow_html=True)

        # 2. 第一壓力卡片
        if first_res:
            diff_res1 = ((first_res - current_price) / current_price) * 100
            st.markdown(f"""
            <div class="card-resistance1">
                <div style="display: flex; justify-content: space-between;">
                    <div>
                        <div class="card-title">⚡ 第一壓力</div>
                        <div style="font-weight: bold; color:#E2A04A; margin-top:5px;">第一壓力</div>
                    </div>
                    <div>
                        <div class="price-value">{first_res:.2f}</div>
                        <div class="price-diff diff-pos">距現價 {diff_res1:+.1f}%</div>
                    </div>
                </div>
                <div class="card-subtitle" style="margin-top:10px; margin-bottom:0;">現價上方 10% 以內、結構共振最具代表性的第一個壓力區。</div>
            </div>
            """, unsafe_allow_html=True)

        # 3. 第二壓力卡片
        if second_res:
            diff_res2 = ((second_res - current_price) / current_price) * 100
            st.markdown(f"""
            <div class="card-resistance2">
                <div style="display: flex; justify-content: space-between;">
                    <div>
                        <div class="card-title">🔥 第二壓力</div>
                        <div style="font-weight: bold; color:#E2554A; margin-top:5px;">第二壓力</div>
                    </div>
                    <div>
                        <div class="price-value">{second_res:.2f}</div>
                        <div class="price-diff diff-pos">距現價 {diff_res2:+.1f}%</div>
                    </div>
                </div>
                <div class="card-subtitle" style="margin-top:10px; margin-bottom:0;">第一壓力之上、真正獨立的下一個壓力共振區。</div>
            </div>
            """, unsafe_allow_html=True)