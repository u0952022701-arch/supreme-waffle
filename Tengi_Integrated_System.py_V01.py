import sys
import time
import logging
import datetime
import pandas as pd
import numpy as np
import requests
import urllib3
import yfinance as yf
import streamlit as st
from scipy.signal import argrelextrema
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 關閉 SSL 警告訊息
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# 屏蔽預載 Log
logging.getLogger("streamlit.runtime.scriptrunner_utils.script_run_context").setLevel(logging.ERROR)
logging.getLogger("streamlit.runtime.caching.cache_data_api").setLevel(logging.ERROR)

# 統一 Request Headers
HTTP_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}


# =========================================================================
# 1. 全域資料快取 (對照表與雙保險籌碼資料)
# =========================================================================
@st.cache_data(ttl=86400)
def load_taiwan_stock_mapping():
    """載入全台股名稱與代碼字典"""
    mapping = {}
    try:
        url_twse = "https://openapi.twse.com.tw/v1/exchangeReport/BWIBBU_ALL"
        res = requests.get(url_twse, headers=HTTP_HEADERS, timeout=6, verify=False)
        if res.status_code == 200:
            for row in res.json():
                code = str(row.get('Code', '')).strip()
                name = str(row.get('Name', '')).strip()
                if code.isdigit() and len(code) == 4 and not code.startswith("00"):
                    mapping[f"{code} {name}"] = f"{code}.TW"
    except Exception:
        pass

    try:
        url_tpex = "https://www.tpex.org.tw/openapi/v1/mopsfin_t187ap03_O"
        res = requests.get(url_tpex, headers=HTTP_HEADERS, timeout=6, verify=False)
        if res.status_code == 200:
            for row in res.json():
                code = str(row.get('SecuritiesCompanyCode', row.get('CompanyCode', ''))).strip()
                name = str(row.get('CompanyName', '')).strip()
                if code.isdigit() and len(code) == 4 and not code.startswith("00"):
                    mapping[f"{code} {name}"] = f"{code}.TWO"
    except Exception:
        pass

    default_stocks = {
        "2330 台積電": "2330.TW", "2317 鴻海": "2317.TW", "2454 聯發科": "2454.TW",
        "2308 台達電": "2308.TW", "2382 廣達": "2382.TW", "3231 緯創": "3231.TW",
        "2603 長榮": "2603.TW", "2881 富邦金": "2881.TW", "3008 大立光": "3008.TW"
    }
    for k, v in default_stocks.items():
        if k not in mapping:
            mapping[k] = v

    return mapping


def fetch_single_stock_finmind_chip(stock_code):
    """【第一保險】利用 FinMind 免費 API 抓取單股近 5 日三大法人買賣超 (張數) - 僅供頁籤一使用"""
    try:
        start_date = (datetime.date.today() - datetime.timedelta(days=10)).strftime("%Y-%m-%d")
        url = f"https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockInstitutionalInvestorsBuySell&data_id={stock_code}&start_date={start_date}"
        res = requests.get(url, headers=HTTP_HEADERS, timeout=6)
        if res.status_code == 200:
            data = res.json().get("data", [])
            if data:
                df = pd.DataFrame(data)
                latest_date = df['date'].max()
                df_latest = df[df['date'] == latest_date]

                foreign = 0
                trust = 0
                dealer = 0

                for _, row in df_latest.iterrows():
                    name = row.get("name", "")
                    diff_zhang = int((row.get("buy", 0) - row.get("sell", 0)) / 1000)
                    if "Foreign" in name:
                        foreign += diff_zhang
                    elif "Investment_Trust" in name:
                        trust += diff_zhang
                    elif "Dealer" in name:
                        dealer += diff_zhang

                return {
                    "Foreign_張": foreign,
                    "Trust_張": trust,
                    "Dealer_張": dealer,
                    "Total_3Inst_張": foreign + trust + dealer
                }, f"{latest_date} (最新交易日)"
    except Exception:
        pass
    return None, ""


@st.cache_data(ttl=1800)
def get_latest_institutional_data():
    """【第二保險】抓取全市場三大法人買賣超資料 (官方 API 備援) - 僅供頁籤一使用"""
    chip_map = {}
    data_date_str = ""

    def parse_to_zhang(val):
        try:
            val_str = str(val).replace(',', '').strip()
            return int(int(val_str) / 1000)
        except Exception:
            return 0

    try:
        url_twse = "https://openapi.twse.com.tw/v1/fund/T86"
        res_twse = requests.get(url_twse, headers=HTTP_HEADERS, timeout=6, verify=False).json()

        if isinstance(res_twse, list) and len(res_twse) > 0:
            for item in res_twse:
                code = str(item.get("Code", "")).strip()
                if code and len(code) == 4 and code.isdigit():
                    foreign = parse_to_zhang(item.get("ForeignInvestorsDifference", 0))
                    trust = parse_to_zhang(item.get("InvestmentTrustDifference", 0))
                    dealer = parse_to_zhang(item.get("DealerSelfDifference", 0)) + parse_to_zhang(
                        item.get("DealerHedgingDifference", 0))
                    total = parse_to_zhang(item.get("TotalDifference", 0))
                    chip_map[code] = {
                        "Foreign_張": foreign, "Trust_張": trust,
                        "Dealer_張": dealer, "Total_3Inst_張": total
                    }

            url_tpex = "https://www.tpex.org.tw/openapi/v1/tpex_mainboard_3insti"
            res_tpex = requests.get(url_tpex, headers=HTTP_HEADERS, timeout=6, verify=False).json()
            if isinstance(res_tpex, list) and len(res_tpex) > 0:
                for item in res_tpex:
                    code = str(
                        item.get("SecuritiesCompanyCode", item.get("CompanyCode", item.get("StkNo", "")))).strip()
                    if code and len(code) == 4 and code.isdigit():
                        foreign = parse_to_zhang(item.get("ForeignBuySellNet", 0))
                        trust = parse_to_zhang(item.get("InvestmentTrustBuySellNet", 0))
                        dealer = parse_to_zhang(item.get("DealerBuySellNet", 0))
                        total = parse_to_zhang(item.get("ThreeInstitutionsBuySellNetTotal", 0))
                        chip_map[code] = {
                            "Foreign_張": foreign, "Trust_張": trust,
                            "Dealer_張": dealer, "Total_3Inst_張": total
                        }

            if len(chip_map) > 0:
                return chip_map, "最新盤後開放資料"
    except Exception:
        pass

    today = datetime.date.today()
    for i in range(7):
        target_date = today - datetime.timedelta(days=i)
        date_str_twse = target_date.strftime("%Y%m%d")
        roc_year = target_date.year - 1911
        roc_date_str = f"{roc_year}/{target_date.strftime('%m/%d')}"

        twse_web_url = f"https://www.twse.com.tw/rwd/zh/fund/T86?response=json&date={date_str_twse}"
        try:
            r_twse = requests.get(twse_web_url, headers=HTTP_HEADERS, timeout=5, verify=False).json()
            if r_twse.get("stat") == "OK" and "data" in r_twse:
                for row in r_twse["data"]:
                    code = str(row[0]).strip()
                    if len(code) == 4 and code.isdigit():
                        foreign = parse_to_zhang(row[4]) if len(row) > 4 else 0
                        trust = parse_to_zhang(row[10]) if len(row) > 10 else 0
                        dealer = parse_to_zhang(row[11]) if len(row) > 11 else 0
                        total = parse_to_zhang(row[-1]) if len(row) > 18 else 0
                        chip_map[code] = {
                            "Foreign_張": foreign, "Trust_張": trust,
                            "Dealer_張": dealer, "Total_3Inst_張": total
                        }

                tpex_web_url = f"https://www.tpex.org.tw/web/stock/33insti/daily_trade/33insti_result.php?l=zh-tw&o=json&se=EW&t=D&d={roc_date_str}"
                try:
                    r_tpex = requests.get(tpex_web_url, headers=HTTP_HEADERS, timeout=5, verify=False).json()
                    if "aaData" in r_tpex:
                        for row in r_tpex["aaData"]:
                            code = str(row[0]).strip()
                            if len(code) == 4 and code.isdigit():
                                foreign = parse_to_zhang(row[8]) if len(row) > 8 else 0
                                trust = parse_to_zhang(row[11]) if len(row) > 11 else 0
                                dealer = parse_to_zhang(row[14]) if len(row) > 14 else 0
                                total = parse_to_zhang(row[17]) if len(row) > 17 else 0
                                chip_map[code] = {
                                    "Foreign_張": foreign, "Trust_張": trust,
                                    "Dealer_張": dealer, "Total_3Inst_張": total
                                }
                except Exception:
                    pass

                suffix = "(今日)" if i == 0 else "(最新交易日)"
                data_date_str = f"{target_date.strftime('%Y-%m-%d')} {suffix}"
                return chip_map, data_date_str
        except Exception:
            continue

    return chip_map, "暫無籌碼數據"


@st.cache_data(ttl=86400)
def get_all_taiwan_tickers():
    """取得台股全市場代碼清單"""
    stocks = []
    try:
        twse_url = "https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL"
        res = requests.get(twse_url, headers=HTTP_HEADERS, timeout=6, verify=False).json()
        for item in res:
            code = str(item.get("Code", "")).strip()
            name = str(item.get("Name", "")).strip()
            if len(code) == 4 and code.isdigit() and not code.startswith("00"):
                stocks.append({"Yahoo_Symbol": f"{code}.TW", "Code": code, "Name": name, "Market": "上市"})
    except Exception:
        pass

    try:
        tpex_url = "https://www.tpex.org.tw/openapi/v1/tpex_mainboard_daily_close_quotes"
        res = requests.get(tpex_url, headers=HTTP_HEADERS, timeout=6, verify=False).json()
        for item in res:
            code = str(item.get("SecuritiesCompanyCode", "")).strip()
            name = str(item.get("CompanyName", "")).strip()
            if len(code) == 4 and code.isdigit() and not code.startswith("00"):
                stocks.append({"Yahoo_Symbol": f"{code}.TWO", "Code": code, "Name": name, "Market": "上櫃"})
    except Exception:
        pass

    return pd.DataFrame(stocks)


# =========================================================================
# 2. 核心邏輯 A：全市場掃描算價模組 (不含籌碼數據)
# =========================================================================
def analyze_single_ticker_for_scan(ticker, stock_name, market, df, min_turnover_wan=3000):
    try:
        code_str = ticker.replace(".TW", "").replace(".TWO", "")
        if code_str.startswith("00") or len(code_str) != 4 or df.empty or len(df) < 60:
            return None

        close, open_p, high, low = df['Close'], df['Open'], df['High'], df['Low']
        volume = df['Volume'] / 1000

        curr_price, curr_open = float(close.iloc[-1]), float(open_p.iloc[-1])
        curr_vol, prev_close = float(volume.iloc[-1]), float(close.iloc[-2])
        daily_change = ((curr_price - prev_close) / prev_close) * 100

        turnover_amount_wan = int((curr_price * curr_vol * 1000) / 10000)
        if turnover_amount_wan < min_turnover_wan:
            return None

        ma5, ma10, ma20, ma60 = close.rolling(5).mean(), close.rolling(10).mean(), close.rolling(
            20).mean(), close.rolling(60).mean()
        vol_ma5, vol_ma20 = volume.rolling(5).mean(), volume.rolling(20).mean()

        std20 = close.rolling(20).std()
        bb_upper, bb_lower = ma20 + (std20 * 2), ma20 - (std20 * 2)
        bb_width = (bb_upper - bb_lower) / ma20

        is_red_k = curr_price > curr_open
        high_20d = close.iloc[-21:-1].max() if len(close) >= 21 else close.iloc[:-1].max()
        is_20d_high = curr_price >= high_20d
        c_price_above_all = curr_price > max(ma5.iloc[-1], ma10.iloc[-1], ma20.iloc[-1], ma60.iloc[-1])
        c_change_valid = (1.0 <= daily_change <= 6.5)
        p1_solid_red_k = is_red_k and is_20d_high and c_price_above_all and c_change_valid

        p2_bullish_ma = (ma5.iloc[-1] > ma10.iloc[-1] > ma20.iloc[-1] > ma60.iloc[-1]) and (ma5.iloc[-1] > ma5.iloc[-2])

        ma_today = [ma5.iloc[-1], ma10.iloc[-1], ma20.iloc[-1], ma60.iloc[-1]]
        p3_ma_squeeze = ((max(ma_today) - min(ma_today)) / ma20.iloc[-1]) <= 0.035
        p4_bb_squeeze = bb_width.iloc[-1] <= (bb_width.iloc[-60:].min() * 1.25)
        p5_vol_burst = (curr_vol >= vol_ma5.iloc[-1] * 1.3) and (curr_vol >= vol_ma20.iloc[-1] * 1.5)

        score = sum([p1_solid_red_k, p2_bullish_ma, p3_ma_squeeze, p4_bb_squeeze, p5_vol_burst])
        near_sup = round(float(ma20.iloc[-1] * 0.98), 2)
        stop_loss = round(float(min(curr_price * 0.95, near_sup)), 2)

        return {
            "Code": code_str, "Name": stock_name, "Market": market,
            "Price": round(curr_price, 2), "Daily_Change_%": round(daily_change, 2),
            "Volume_張": int(curr_vol), "Turnover_萬": turnover_amount_wan,
            "Pre_Launch_Score": score, "Solid_Red_K": "✅" if p1_solid_red_k else "❌",
            "Bullish_MA": "✅" if p2_bullish_ma else "❌", "MA_Squeeze": "✅" if p3_ma_squeeze else "❌",
            "BB_Squeeze": "✅" if p4_bb_squeeze else "❌", "Vol_Burst": "✅" if p5_vol_burst else "❌",
            "Near_Sup": near_sup, "Stop_Loss": stop_loss
        }
    except Exception:
        return None


# =========================================================================
# 3. 頁籤一 UI：單股深度診斷視圖
# =========================================================================
def render_single_stock_analyzer(stock_map, chip_map, chip_date_str):
    if "symbol_text" not in st.session_state:
        st.session_state["symbol_text"] = "2330"

    def update_symbol_from_select():
        chosen = st.session_state.get("stock_lookup_select")
        if chosen and chosen != "-- 請選擇或輸入名稱過濾 --":
            full_ticker = stock_map[chosen]
            st.session_state["symbol_text"] = full_ticker.split('.')[0]

    with st.expander("💡 不知道代碼？點此用「股票名稱」尋找台股代碼", expanded=False):
        st.selectbox(
            "請選擇或直接輸入股票名稱過濾：",
            options=["-- 請選擇或輸入名稱過濾 --"] + list(stock_map.keys()),
            key="stock_lookup_select",
            on_change=update_symbol_from_select
        )

    input_symbol = st.text_input("請輸入台股代碼（如 2330）或美股代碼（如 TSLA）:", key="symbol_text")

    if st.button("開始診斷個股", type="primary"):
        clean_code = input_symbol.strip().upper().replace(".TW", "").replace(".TWO", "")

        with st.spinner('正在獲取個股歷史數據與計算關卡價位...'):
            df = pd.DataFrame()
            ticker_used = clean_code

            if clean_code.isdigit() and len(clean_code) == 4:
                cand_tw, cand_two = f"{clean_code}.TW", f"{clean_code}.TWO"
                all_tickers = set(stock_map.values())
                candidates = [cand_tw] if cand_tw in all_tickers else (
                    [cand_two] if cand_two in all_tickers else [cand_tw, cand_two])

                for cand in candidates:
                    stk = yf.Ticker(cand)
                    temp_df = stk.history(period="1y")
                    if not temp_df.empty:
                        df = temp_df
                        ticker_used = cand
                        break
            else:
                stk = yf.Ticker(clean_code)
                df = stk.history(period="1y")
                ticker_used = clean_code

        if df.empty or len(df) < 60:
            st.error(f"無法取得 [{clean_code}] 足夠的歷史數據，請確認代碼是否正確。")
            return

        current_price = df['Close'].iloc[-1]
        prev_close = df['Close'].iloc[-2] if len(df) >= 2 else current_price
        latest_k_date = df.index[-1].strftime('%Y-%m-%d')
        stock_display_name = next((k for k, v in stock_map.items() if v == ticker_used), clean_code)

        # 指標計算
        df['MA5'] = df['Close'].rolling(5).mean()
        df['MA10'] = df['Close'].rolling(10).mean()
        df['MA20'] = df['Close'].rolling(20).mean()
        df['MA60'] = df['Close'].rolling(60).mean()
        df['MA120'] = df['Close'].rolling(120).mean()
        df['20D_Low'] = df['Low'].rolling(20).min()
        df['Vol_MA5'] = df['Volume'].rolling(5).mean()
        df['Vol_MA20'] = df['Volume'].rolling(20).mean()

        df['BB_Mid'] = df['MA20']
        df['BB_Std'] = df['Close'].rolling(20).std()
        df['BB_Upper'] = df['BB_Mid'] + (df['BB_Std'] * 2)
        df['BB_Lower'] = df['BB_Mid'] - (df['BB_Std'] * 2)
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Mid']

        high_vol_closes = df[df['Volume'] > df['Vol_MA20'] * 1.8]['Close'].tolist()
        order = 5
        local_mins = df.iloc[argrelextrema(df['Low'].values, np.less_equal, order=order)[0]]['Low'].tolist()
        local_maxs = df.iloc[argrelextrema(df['High'].values, np.greater_equal, order=order)[0]]['High'].tolist()

        support_pool = [x for x in [df['MA120'].iloc[-1], df['20D_Low'].iloc[-1]] + high_vol_closes + local_mins if
                        not pd.isna(x) and x < current_price]
        resistance_pool = [x for x in high_vol_closes + local_maxs if not pd.isna(x) and x > current_price]

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
        second_res = min([x for x in res_clusters if x > (first_res * 1.03 if first_res else current_price)],
                         default=None)

        # 評分
        daily_change = df['Close'].pct_change().iloc[-1] * 100
        high_20d = df['Close'].iloc[-21:-1].max() if len(df) >= 21 else df['Close'].iloc[:-1].max()

        ma_list = [df['MA5'].iloc[-1], df['MA10'].iloc[-1], df['MA20'].iloc[-1], df['MA60'].iloc[-1]]
        ma_max, ma_min = max(ma_list), min(ma_list)

        is_red_k = df['Close'].iloc[-1] > df['Open'].iloc[-1]
        is_20d_high = df['Close'].iloc[-1] >= high_20d
        above_all_ma = df['Close'].iloc[-1] > ma_max
        c_change_valid = (1.0 <= daily_change <= 6.5)
        p1_solid_red_k = is_red_k and is_20d_high and above_all_ma and c_change_valid

        p2_bullish_ma = (df['MA5'].iloc[-1] > df['MA10'].iloc[-1] > df['MA20'].iloc[-1] > df['MA60'].iloc[-1]) and (
                    df['MA5'].iloc[-1] > df['MA5'].iloc[-2])
        p3_ma_squeeze = ((ma_max - ma_min) / df['MA20'].iloc[-1]) <= 0.035
        p4_bband_squeeze = df['BB_Width'].iloc[-1] <= (df['BB_Width'].iloc[-60:].min() * 1.25)
        p5_vol_burst = (df['Volume'].iloc[-1] >= df['Vol_MA5'].iloc[-1] * 1.3) and (
                    df['Volume'].iloc[-1] >= df['Vol_MA20'].iloc[-1] * 1.5)

        pre_launch_score = sum([p1_solid_red_k, p2_bullish_ma, p3_ma_squeeze, p4_bband_squeeze, p5_vol_burst])

        c1_ma = (df['Close'].iloc[-1] > df['MA5'].iloc[-1] > df['MA20'].iloc[-1] > df['MA60'].iloc[-1])
        c2_vol = (df['Volume'].iloc[-1] > df['Vol_MA20'].iloc[-1] * 1.5)
        c3_breakout = (df['Close'].iloc[-1] >= high_20d)
        c4_momentum = (daily_change >= 3.0)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rsi_series = 100 - (100 / (1 + (gain / loss)))
        c5_rsi = (rsi_series.iloc[-1] >= 55)
        rocket_score = sum([c1_ma, c2_vol, c3_breakout, c4_momentum, c5_rsi])

        ma20_val = df['MA20'].iloc[-1]
        dip_buy_low = near_sup if near_sup else ma20_val * 0.98
        dip_buy_high = min(current_price, max(ma20_val, dip_buy_low * 1.02))
        breakout_buy = first_res if first_res else high_20d
        stop_loss = (near_sup * 0.98) if near_sup else (ma20_val * 0.97)

        # 畫面輸出
        st.write("---")
        st.markdown(
            f"### 📌 標的：{stock_display_name}\n"
            f"### 現價：**{current_price:.2f}** （單日：{daily_change:+.2f}%）\n"
            f"<div style='font-size: 13px; color: #888888; margin-top: -12px; margin-bottom: 15px;'>昨收價：{prev_close:.2f} ｜ K線資料日期：{latest_k_date}</div>",
            unsafe_allow_html=True
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🛡️ 近端支撐區", f"{near_sup:.2f}" if near_sup else "查無數據")
        with col2:
            st.metric("⚡ 第一壓力區", f"{first_res:.2f}" if first_res else "查無數據")
        with col3:
            st.metric("🔥 第二壓力區", f"{second_res:.2f}" if second_res else "查無數據")

        # 三大法人籌碼流向展示區 (單股診斷保留)
        st.write("---")
        chip_info, source_date_str = None, ""
        if clean_code.isdigit() and len(clean_code) == 4:
            chip_info, source_date_str = fetch_single_stock_finmind_chip(clean_code)

        if not chip_info:
            chip_info = chip_map.get(clean_code, None)
            source_date_str = chip_date_str

        st.subheader(f"🏛️ 三大法人籌碼流向（{source_date_str if source_date_str else '查詢中'}）")

        if chip_info:
            c_f, c_t, c_d, c_tot = st.columns(4)
            f_val = chip_info.get("Foreign_張", 0)
            t_val = chip_info.get("Trust_張", 0)
            d_val = chip_info.get("Dealer_張", 0)
            tot_val = chip_info.get("Total_3Inst_張", 0)

            with c_f:
                st.metric("外資買賣超", f"{f_val:+,} 張")
            with c_t:
                st.metric("投信買賣超", f"{t_val:+,} 張")
            with c_d:
                st.metric("自營商買賣超", f"{d_val:+,} 張")
            with c_tot:
                st.metric("三大法人合計", f"{tot_val:+,} 張")
        else:
            st.info("ℹ️ 暫無三大法人籌碼數據，或該標的非台股上市櫃個股。")

        st.write("---")
        st.subheader("💡 最佳進場點與風控價位")
        in_col1, in_col2, in_col3 = st.columns(3)
        with in_col1:
            st.warning(f"🟢 **壓低佈局區（低吸）**\n\n**{dip_buy_low:.2f} ~ {dip_buy_high:.2f}**")
        with in_col2:
            st.error(f"🚀 **突破加碼點（追擊）**\n\n**{breakout_buy:.2f}** (帶量突破時)")
        with in_col3:
            st.info(f"🛑 **關鍵防守價（停損）**\n\n**{stop_loss:.2f}** (-2% 防守)")

        st.write("---")
        st.subheader("🎯 飆股型態與攻擊動能雙評核")
        score_col1, score_col2 = st.columns(2)
        with score_col1:
            st.markdown(f"**起漲發動指數：{pre_launch_score} / 5 分**")
            st.write(f"{'✅' if p1_solid_red_k else '❌'} 強勢創高紅K (+1%~6.5% 創20日高)")
            st.write(f"{'✅' if p2_bullish_ma else '❌'} 均線多頭排列 (5MA向上)")
            st.write(f"{'✅' if p3_ma_squeeze else '❌'} 均線高度糾結 (價差≤3.5%)")
            st.write(f"{'✅' if p4_bband_squeeze else '❌'} 布林通道極致壓縮 (≤60日低位1.25倍)")
            st.write(f"{'✅' if p5_vol_burst else '❌'} 資金爆量試盤 (≥5日均1.3x & ≥20日均1.5x)")

        with score_col2:
            st.markdown(f"**飆股攻擊分數：{rocket_score} / 5 分**")
            st.write(f"{'✅' if c1_ma else '❌'} 均線多頭排列")
            st.write(f"{'✅' if c2_vol else '❌'} 當日成交爆量")
            st.write(f"{'✅' if c3_breakout else '❌'} 創 20 日新高")
            st.write(f"{'✅' if c4_momentum else '❌'} 單日強勢攻擊 (≥3%)")
            st.write(f"{'✅' if c5_rsi else '❌'} RSI 強勢區 (≥55)")

        # K線圖繪製
        st.write("---")
        st.markdown("#### 📊 近半年日 K 線、均線與布林通道圖")
        df_chart = df.tail(120).copy()

        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=('日 K 線圖、均線與布林通道', '成交量'),
            row_heights=[0.7, 0.3]
        )

        if 'BB_Upper' in df_chart.columns and 'BB_Lower' in df_chart.columns:
            fig.add_trace(
                go.Scatter(x=df_chart.index.strftime('%Y-%m-%d'), y=df_chart['BB_Upper'], mode='lines', name='布林上軌',
                           line=dict(width=1, color='rgba(160, 160, 160, 0.6)', dash='dash')), row=1, col=1)
            fig.add_trace(
                go.Scatter(x=df_chart.index.strftime('%Y-%m-%d'), y=df_chart['BB_Lower'], mode='lines', name='布林下軌',
                           line=dict(width=1, color='rgba(160, 160, 160, 0.6)', dash='dash'), fill='tonexty',
                           fillcolor='rgba(180, 180, 200, 0.12)'), row=1, col=1)

        fig.add_trace(
            go.Candlestick(x=df_chart.index.strftime('%Y-%m-%d'), open=df_chart['Open'], high=df_chart['High'],
                           low=df_chart['Low'], close=df_chart['Close'], name='K線', increasing_line_color='#e74c3c',
                           increasing_fillcolor='#e74c3c', decreasing_line_color='#2ecc71',
                           decreasing_fillcolor='#2ecc71'), row=1, col=1)

        ma_colors = {'MA5': '#f39c12', 'MA10': '#3498db', 'MA20': '#9b59b6', 'MA60': '#e67e22'}
        for ma_key, color in ma_colors.items():
            if ma_key in df_chart.columns:
                fig.add_trace(
                    go.Scatter(x=df_chart.index.strftime('%Y-%m-%d'), y=df_chart[ma_key], mode='lines', name=ma_key,
                               line=dict(width=1.5, color=color)), row=1, col=1)

        vol_colors = ['#e74c3c' if c >= o else '#2ecc71' for c, o in zip(df_chart['Close'], df_chart['Open'])]
        fig.add_trace(
            go.Bar(x=df_chart.index.strftime('%Y-%m-%d'), y=df_chart['Volume'], name='成交量', marker_color=vol_colors),
            row=2, col=1)

        fig.update_layout(xaxis_rangeslider_visible=False, height=500, margin=dict(l=10, r=10, t=30, b=10),
                          template="plotly_white")
        fig.update_xaxes(type='category')
        st.plotly_chart(fig, width="stretch")


# =========================================================================
# 4. 頁籤二 UI：全市場飆股自動掃描視圖 (完全移除籌碼資訊)
# =========================================================================
def render_market_scanner():
    #st.subheader("🚀 市場飆股型態自動掃描")

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        selected_market = st.selectbox("市場範疇", ["全部", "上市", "上櫃"], index=0)
    with col2:
        min_score = st.slider("最低起漲指數", 1, 5, 3)
    with col3:
        min_volume = st.number_input("最低成交量 (張)", value=300, step=100)
    with col4:
        min_turnover = st.number_input("最低成交額 (萬元)", value=3000, step=500)
    with col5:
        max_scan_limit = st.selectbox("掃描數量限制", [150, 300, 600, 1000, "全部個股"], index=1)

    if st.button("🚀 開始掃描行情", type="primary", key="scan_btn"):
        with st.spinner("正在讀取交易所全股票對照表..."):
            stock_df = get_all_taiwan_tickers()

        if stock_df.empty:
            st.error("❌ 無法取得股票清單。")
            return

        filtered_df = stock_df.copy()
        if selected_market != "全部":
            filtered_df = filtered_df[filtered_df['Market'] == selected_market]

        scan_list = filtered_df.to_dict('records')
        if max_scan_limit != "全部個股":
            scan_list = scan_list[:int(max_scan_limit)]

        tickers = [x["Yahoo_Symbol"] for x in scan_list]
        symbol_to_info = {x["Yahoo_Symbol"]: x for x in scan_list}

        CHUNK_SIZE = 150
        ticker_chunks = [tickers[i:i + CHUNK_SIZE] for i in range(0, len(tickers), CHUNK_SIZE)]

        st.info(f"📊 正在分析 **{len(tickers)}** 檔股票，分 **{len(ticker_chunks)}** 批次下載數據...")

        progress_bar = st.progress(0)
        status_text = st.empty()
        all_results = []

        for idx, chunk in enumerate(ticker_chunks):
            status_text.text(f"⏳ 下載第 {idx + 1}/{len(ticker_chunks)} 批次數據 (共 {len(chunk)} 檔)...")
            try:
                data = yf.download(chunk, period="1y", interval="1d", group_by="ticker", auto_adjust=True, threads=True,
                                   progress=False)

                for ticker in chunk:
                    try:
                        df = data.dropna() if len(chunk) == 1 else data[ticker].dropna()
                        info = symbol_to_info.get(ticker, {})

                        res = analyze_single_ticker_for_scan(
                            ticker, info.get("Name", "未知"), info.get("Market", "未知"), df,
                            min_turnover_wan=min_turnover
                        )

                        if res and res['Pre_Launch_Score'] >= min_score and res['Volume_張'] >= min_volume:
                            if res['Solid_Red_K'] == "✅" or res['MA_Squeeze'] == "✅" or res['Bullish_MA'] == "✅":
                                all_results.append(res)
                    except Exception:
                        continue
            except Exception as e:
                st.warning(f"⚠️ 第 {idx + 1} 批次略過: {e}")

            progress_bar.progress((idx + 1) / len(ticker_chunks))
            if idx + 1 < len(ticker_chunks):
                time.sleep(1.2)

        status_text.text("✅ 行情掃描完成！")

        if all_results:
            res_df = pd.DataFrame(all_results).sort_values(by=["Pre_Launch_Score", "Turnover_萬"],
                                                           ascending=[False, False])
            st.success(f"🎯 成功找出 **{len(res_df)}** 檔「飆股蓄勢發動」潛力標的！")

            display_cols = {
                "Code": "股票代碼", "Name": "股票名稱", "Market": "市場", "Price": "現價",
                "Daily_Change_%": "漲幅(%)", "Volume_張": "成交量(張)", "Turnover_萬": "成交額(萬)",
                "Pre_Launch_Score": "起漲指數", "Solid_Red_K": "創高紅K", "Bullish_MA": "多頭排列",
                "MA_Squeeze": "均線糾結", "BB_Squeeze": "布林壓縮", "Vol_Burst": "量能爆發",
                "Near_Sup": "關鍵支撐", "Stop_Loss": "建議停損"
            }

            st.dataframe(res_df[list(display_cols.keys())].rename(columns=display_cols), width="stretch",
                         hide_index=True)

            csv = res_df[list(display_cols.keys())].rename(columns=display_cols).to_csv(index=False).encode('utf-8-sig')
            st.download_button("📥 下載篩選結果 CSV 報表", csv,
                               f"breakout_stocks_{datetime.datetime.now().strftime('%Y%m%d')}.csv", "text/csv")
        else:
            st.warning("⚠️ 暫無符合條件標的，建議放寬門檻條件。")


# =========================================================================
# 5. Streamlit 主程式進入點
# =========================================================================
def run_app():
    st.set_page_config(page_title="TENGI 飆股掃描與個股診斷小工具", page_icon="📈", layout="wide")
    st.title("📈 TENGI 飆股掃描與個股診斷小工具")
    st.caption("Copyright © 2026 DUKE All rights reserved.")

    stock_map = load_taiwan_stock_mapping()
    chip_map, chip_date_str = get_latest_institutional_data()

    tab1, tab2 = st.tabs(["🔍 個股診斷分析", "🚀 市場飆股掃描"])

    with tab1:
        render_single_stock_analyzer(stock_map, chip_map, chip_date_str)

    with tab2:
        render_market_scanner()


if __name__ == "__main__":
    from streamlit.web import cli as stcli

    if not st.runtime.exists():
        sys.argv = ["streamlit", "run", __file__]
        sys.exit(stcli.main())
    else:
        run_app()
