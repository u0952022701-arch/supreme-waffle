import sys
import datetime
import requests
import pandas as pd
import numpy as np
import streamlit as st
import yfinance as yf
from scipy.signal import argrelextrema
from streamlit.web import cli as stcli
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# =========================================================================
# 0. 自動載入與快取台股名稱與代碼對照表 (上市/上櫃)
# =========================================================================
@st.cache_data(ttl=86400)
def load_taiwan_stock_mapping():
    mapping = {}
    # 1. 下載 TWSE 上市股票清單
    try:
        url_twse = "https://openapi.twse.com.tw/v1/exchangeReport/BWIBBU_ALL"
        df_twse = pd.read_json(url_twse)
        for _, row in df_twse.iterrows():
            code = str(row.get('Code', '')).strip()
            name = str(row.get('Name', '')).strip()
            if code.isdigit() and len(code) == 4:
                mapping[f"{code} {name}"] = f"{code}.TW"
    except Exception:
        pass

    # 2. 下載 TPEX 上櫃股票清單
    try:
        url_tpex = "https://www.tpex.org.tw/openapi/v1/mopsfin_t187ap03_O"
        df_tpex = pd.read_json(url_tpex)
        for _, row in df_tpex.iterrows():
            code = str(row.get('SecuritiesCompanyCode', row.get('CompanyCode', ''))).strip()
            name = str(row.get('CompanyName', '')).strip()
            if code.isdigit() and len(code) == 4:
                mapping[f"{code} {name}"] = f"{code}.TWO"
    except Exception:
        pass

    # 3. 備用常見熱門股票清單
    default_stocks = {
        "2330 台積電": "2330.TW",
        "2317 鴻海": "2317.TW",
        "2454 聯發科": "2454.TW",
        "2308 台達電": "2308.TW",
        "2382 廣達": "2382.TW",
        "3231 緯創": "3231.TW",
        "2603 長榮": "2603.TW",
        "2881 富邦金": "2881.TW",
        "2882 國泰金": "2882.TW",
        "3008 大立光": "3008.TW",
        "3289 宜特": "3289.TWO",
        "8069 元太": "8069.TWO",
        "3374 精材": "3374.TWO",
        "3017 奇鋐": "3017.TW",
        "6669 緯穎": "6669.TW",
        "2368 金像電": "2368.TW",
    }
    for k, v in default_stocks.items():
        if k not in mapping:
            mapping[k] = v

    return mapping


# =========================================================================
# 0-1. 自動抓取三大法人最新買賣超數據 (包含當天無資料時自動回溯前一日)
# =========================================================================
@st.cache_data(ttl=3600)
def fetch_taiwan_institutional_data():
    inst_data = {}
    data_date_str = ""

    def parse_val(v):
        try:
            return int(str(v).replace(',', ''))
        except Exception:
            return 0

    # 1. 優先透過 OpenAPI 抓取（通常自動為最新交易日資料）
    try:
        url_twse = "https://openapi.twse.com.tw/v1/fund/T86"
        df_twse = pd.read_json(url_twse)
        if not df_twse.empty and 'Code' in df_twse.columns:
            if 'Date' in df_twse.columns and not df_twse['Date'].dropna().empty:
                data_date_str = str(df_twse['Date'].iloc[0])

            for _, row in df_twse.iterrows():
                code = str(row.get('Code', '')).strip()
                if code:
                    foreign = parse_val(row.get('ForeignInvestorsBuySell', row.get('ForeignInvestorsBuy', 0)))
                    trust = parse_val(row.get('InvestmentTrustBuySell', 0))
                    dealer = parse_val(row.get('DealerBuySell', row.get('DealerProprietaryBuySell', 0)))
                    total = parse_val(row.get('TotalBuySell', 0))

                    inst_data[code] = {
                        'foreign': foreign // 1000,
                        'trust': trust // 1000,
                        'dealer': dealer // 1000,
                        'total': (foreign + trust + dealer) // 1000 if total == 0 else total // 1000,
                        'date': data_date_str
                    }
    except Exception:
        pass

    # 2. 若 OpenAPI 無回應或空值，向前回溯尋找最近一個有數據的交易日 (最多回溯 7 天)
    if not inst_data:
        today = datetime.date.today()
        for i in range(7):
            target_date = today - datetime.timedelta(days=i)
            date_str = target_date.strftime("%Y%m%d")
            url = f"https://www.twse.com.tw/rwd/zh/fund/T86?date={date_str}&selectType=ALLBUT0999&response=json"
            try:
                resp = requests.get(url, timeout=5)
                if resp.status_code == 200:
                    res_json = resp.json()
                    if res_json.get('stat') == 'OK' and 'data' in res_json and len(res_json['data']) > 0:
                        data_date_str = target_date.strftime("%Y-%m-%d")
                        for row in res_json['data']:
                            code = str(row[0]).strip()
                            if code and code.isdigit() and len(code) == 4:
                                foreign = parse_val(row[4] if len(row) > 4 else 0)
                                trust = parse_val(row[7] if len(row) > 7 else 0)
                                dealer = parse_val(row[10] if len(row) > 10 else 0)
                                total = parse_val(row[11] if len(row) > 11 else 0)
                                inst_data[code] = {
                                    'foreign': foreign // 1000,
                                    'trust': trust // 1000,
                                    'dealer': dealer // 1000,
                                    'total': total // 1000,
                                    'date': data_date_str
                                }
                        break
            except Exception:
                continue

    # 3. TPEX 上櫃三大法人買賣超
    try:
        url_tpex = "https://www.tpex.org.tw/openapi/v1/tpex_mainboard_33_summary"
        df_tpex = pd.read_json(url_tpex)
        if not df_tpex.empty and ('SecuritiesCompanyCode' in df_tpex.columns or 'CompanyCode' in df_tpex.columns):
            for _, row in df_tpex.iterrows():
                code = str(row.get('SecuritiesCompanyCode', row.get('CompanyCode', ''))).strip()
                if code and code not in inst_data:
                    foreign = parse_val(row.get('ForeignInvestorBuySell', row.get('ForeignInvestorsBuySell', 0)))
                    trust = parse_val(row.get('InvestmentTrustBuySell', 0))
                    dealer = parse_val(row.get('DealerBuySell', 0))
                    total = parse_val(row.get('TotalBuySell', 0))

                    inst_data[code] = {
                        'foreign': foreign // 1000,
                        'trust': trust // 1000,
                        'dealer': dealer // 1000,
                        'total': (foreign + trust + dealer) // 1000 if total == 0 else total // 1000,
                        'date': data_date_str
                    }
    except Exception:
        pass

    return inst_data


def run_app():
    st.set_page_config(page_title="TENGI 股票診斷小工具", page_icon="📈", layout="centered")

    st.title("📈 TENGI 股票診斷小工具")
    st.caption("自動計算關卡價位，診斷「起漲蓄勢訊號」與「進行中飆股動能」並精算「最佳進場與停損點」")

    # 載入台股名稱與代碼對照
    stock_map = load_taiwan_stock_mapping()

    # 初始化預設輸入值
    if "symbol_text" not in st.session_state:
        st.session_state["symbol_text"] = "2330"

    def update_symbol_from_select():
        chosen = st.session_state.get("stock_lookup_select")
        if chosen and chosen != "-- 請選擇或輸入名稱過濾 --":
            full_ticker = stock_map[chosen]
            st.session_state["symbol_text"] = full_ticker.split('.')[0]

    # =========================================================================
    # 功能：股票名稱尋找台股代碼
    # =========================================================================
    with st.expander("🔍 **不知道代碼？點此用「股票名稱」尋找台股代碼**", expanded=False):
        st.selectbox(
            "請選擇或直接輸入股票名稱過濾（例如：台積電、聯發科、長榮）：",
            options=["-- 請選擇或輸入名稱過濾 --"] + list(stock_map.keys()),
            key="stock_lookup_select",
            on_change=update_symbol_from_select
        )
        current_choice = st.session_state.get("stock_lookup_select")
        if current_choice and current_choice != "-- 請選擇或輸入名稱過濾 --":
            clean_code = stock_map[current_choice].split('.')[0]
            st.success(f"🎯 已將 **{current_choice}** 代碼 (`{clean_code}`) 自動帶入下方診斷框！")

    input_symbol = st.text_input(
        "請輸入台股代碼（例如 2330 或 3289）或美股代碼（例如 TSLA）:",
        key="symbol_text"
    )

    if st.button("開始診斷", type="primary"):
        clean_code = input_symbol.strip().upper().replace(".TW", "").replace(".TWO", "")

        with st.spinner('正在自動尋找股票數據與三大法人籌碼資料中...'):
            df = pd.DataFrame()
            ticker_used = clean_code

            # 抓取三大法人資料
            inst_dict = fetch_taiwan_institutional_data()
            inst_info = inst_dict.get(clean_code, None)

            # 若為 4 位數純數字，自動判斷上市 (.TW) 或上櫃 (.TWO)
            if clean_code.isdigit() and len(clean_code) == 4:
                cand_tw = f"{clean_code}.TW"
                cand_two = f"{clean_code}.TWO"
                all_tickers = set(stock_map.values())

                if cand_tw in all_tickers:
                    candidates = [cand_tw]
                elif cand_two in all_tickers:
                    candidates = [cand_two]
                else:
                    candidates = [cand_tw, cand_two]

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
            st.error(f"無法取得 [{clean_code}] 足夠的歷史數據（需至少 60 個交易日），請確認代碼是否正確。")
        else:
            current_price = df['Close'].iloc[-1]
            prev_close = df['Close'].iloc[-2] if len(df) >= 2 else current_price

            stock_display_name = next((k for k, v in stock_map.items() if v == ticker_used), clean_code)

            # =========================================================================
            # 1. 技術指標與結構計算
            # =========================================================================
            df['MA5'] = df['Close'].rolling(window=5).mean()
            df['MA10'] = df['Close'].rolling(window=10).mean()
            df['MA20'] = df['Close'].rolling(window=20).mean()
            df['MA60'] = df['Close'].rolling(window=60).mean()
            df['MA120'] = df['Close'].rolling(window=120).mean()

            df['20D_Low'] = df['Low'].rolling(window=20).min()
            df['Vol_MA5'] = df['Volume'].rolling(window=5).mean()
            df['Vol_MA20'] = df['Volume'].rolling(window=20).mean()

            # 布林通道 (Bollinger Bands)
            df['BB_Mid'] = df['MA20']
            df['BB_Std'] = df['Close'].rolling(window=20).std()
            df['BB_Upper'] = df['BB_Mid'] + (df['BB_Std'] * 2)
            df['BB_Lower'] = df['BB_Mid'] - (df['BB_Std'] * 2)
            df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Mid']

            # 爆量 K 收盤
            high_vol_closes = df[df['Volume'] > df['Vol_MA20'] * 1.8]['Close'].tolist()

            # 波段極值
            order = 5
            local_mins = df.iloc[argrelextrema(df['Low'].values, np.less_equal, order=order)[0]]['Low'].tolist()
            local_maxs = df.iloc[argrelextrema(df['High'].values, np.greater_equal, order=order)[0]]['High'].tolist()

            # 支撐與壓力候選池
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

            # =========================================================================
            # 2. 策略 A：進行中飆股動能
            # =========================================================================
            c1_ma = (df['Close'].iloc[-1] > df['MA5'].iloc[-1] > df['MA20'].iloc[-1] > df['MA60'].iloc[-1])
            c2_vol = (df['Volume'].iloc[-1] > df['Vol_MA20'].iloc[-1] * 1.5)
            high_20d = df['Close'].iloc[-21:-1].max() if len(df) >= 21 else df['Close'].max()
            c3_breakout = (df['Close'].iloc[-1] >= high_20d)
            daily_change = df['Close'].pct_change().iloc[-1] * 100
            c4_momentum = (daily_change >= 3.0)

            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi_series = 100 - (100 / (1 + rs))
            latest_rsi = rsi_series.iloc[-1]
            c5_rsi = (latest_rsi >= 55)

            rocket_score = sum([c1_ma, c2_vol, c3_breakout, c4_momentum, c5_rsi])

            # =========================================================================
            # 3. 策略 B：準備飆漲（起漲前夕 / 潛力蓄勢特徵檢測）
            # =========================================================================
            ma_list = [df['MA5'].iloc[-1], df['MA10'].iloc[-1], df['MA20'].iloc[-1], df['MA60'].iloc[-1]]
            ma_max, ma_min = max(ma_list), min(ma_list)
            p1_ma_squeeze = ((ma_max - ma_min) / df['MA20'].iloc[-1]) <= 0.035

            bband_width_min = df['BB_Width'].iloc[-60:].min()
            p2_bband_squeeze = df['BB_Width'].iloc[-1] <= (bband_width_min * 1.25)
            p3_vol_pickup = df['Volume'].iloc[-1] >= (df['Vol_MA5'].iloc[-1] * 1.3)
            p4_turning_strong = (df['Close'].iloc[-1] > df['MA20'].iloc[-1]) and (0.5 <= daily_change <= 4.0)

            pre_launch_score = sum([p1_ma_squeeze, p2_bband_squeeze, p3_vol_pickup, p4_turning_strong])

            # =========================================================================
            # 4. 最佳進場與風控價位精算邏輯
            # =========================================================================
            ma20_val = df['MA20'].iloc[-1]
            dip_buy_low = near_sup if near_sup else ma20_val * 0.98
            dip_buy_high = min(current_price, max(ma20_val, dip_buy_low * 1.02))
            breakout_buy = first_res if first_res else high_20d
            stop_loss = (near_sup * 0.98) if near_sup else (ma20_val * 0.97)

            # =========================================================================
            # UI 顯示區：標的、現價、昨收價與支撐壓力卡片 (改為紅漲綠跌 inverse 模式)
            # =========================================================================
            st.write("---")
            st.markdown(
                f"### 📌 標的：{stock_display_name}\n"
                f"### 現價：**{current_price:.2f}** （單日：{daily_change:+.2f}%）\n"
                f"<div style='font-size: 14px; color: #888888; margin-top: -12px; margin-bottom: 15px;'>昨收價：{prev_close:.2f}</div>",
                unsafe_allow_html=True
            )

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    label="🛡️ 近端支撐區",
                    value=f"{near_sup:.2f}" if near_sup else "查無數據",
                    delta=f"{((near_sup - current_price) / current_price) * 100:.1f}%" if near_sup else "",
                    delta_color="inverse"
                )
                st.caption("最接近的支撐價位；跌破後結構轉弱。")
            with col2:
                st.metric(
                    label="⚡ 第一壓力區",
                    value=f"{first_res:.2f}" if first_res else "查無數據",
                    delta=f"+{((first_res - current_price) / current_price) * 100:.1f}%" if first_res else "",
                    delta_color="inverse"
                )
                st.caption("現價上方 10% 以內；第一個壓力區。")
            with col3:
                st.metric(
                    label="🔥 第二壓力區",
                    value=f"{second_res:.2f}" if second_res else "查無數據",
                    delta=f"+{((second_res - current_price) / current_price) * 100:.1f}%" if second_res else "",
                    delta_color="inverse"
                )
                st.caption("第一壓力之上；第二壓力區。")

            # =========================================================================
            # UI 顯示區：三大法人最新籌碼流向 (紅漲綠跌 inverse 模式)
            # =========================================================================
            st.write("---")
            data_date_title = f" ({inst_info['date']})" if (inst_info and inst_info.get('date')) else ""
            st.subheader(f"🏛️ 三大法人最新籌碼動向 (單位：張){data_date_title}")

            if inst_info:
                ic1, ic2, ic3, ic4 = st.columns(4)
                with ic1:
                    f_val = inst_info['foreign']
                    st.metric(
                        label="外資買賣超",
                        value=f"{f_val:+,} ",
                        delta="買超" if f_val >= 0 else "-賣超",
                        delta_color="inverse"
                    )
                with ic2:
                    t_val = inst_info['trust']
                    st.metric(
                        label="投信買賣超",
                        value=f"{t_val:+,} ",
                        delta="買超" if t_val >= 0 else "-賣超",
                        delta_color="inverse"
                    )
                with ic3:
                    d_val = inst_info['dealer']
                    st.metric(
                        label="自營商買賣超",
                        value=f"{d_val:+,} ",
                        delta="買超" if d_val >= 0 else "-賣超",
                        delta_color="inverse"
                    )
                with ic4:
                    tot_val = inst_info['total']
                    st.metric(
                        label="三大法人合計",
                        value=f"{tot_val:+,} ",
                        delta="淨買超" if tot_val >= 0 else "-淨賣超",
                        delta_color="inverse"
                    )
            else:
                st.info("ℹ️ 暫無三大法人籌碼數據（美股標的或交易所當日盤後資料更新中）。")

            # =========================================================================
            # UI 顯示區：準備飆漲（起漲前夕診斷）
            # =========================================================================
            st.write("---")
            st.subheader("🎯 準備飆漲（起漲蓄勢）診斷")

            if pre_launch_score >= 3:
                st.success(f"🔥 **起漲發動指數：{pre_launch_score} / 4 分 (極高潛力發動型態！)**")
                st.info("💡 **發動評價**：籌碼高度沉澱，均線與布林通道極度壓縮完畢，資金出現初次試盤！極可能於近期噴發突破。")
            elif pre_launch_score == 2:
                st.warning(f"⚡ **起漲發動指數：{pre_launch_score} / 4 分 (持續蓄勢中)**")
                st.info("💡 **發動評價**：波動已逐步壓縮完畢，正等待量能進一步釋放突破關鍵價位。")
            else:
                st.error(f"⚪ **起漲發動指數：{pre_launch_score} / 4 分 (尚未進入發動期)**")
                st.info("💡 **發動評價**：目前未出現爆發前夕的壓縮型態，建議觀望或選擇高指數標的。")

            st.markdown("#### 📋 起漲 4 大特徵檢核")
            st.write(
                f"{'✅' if p1_ma_squeeze else '❌'} **均線高度糾結**：5, 10, 20, 60 日線糾結度 {((ma_max - ma_min) / df['MA20'].iloc[-1]) * 100:.2f}% (<= 3.5% 為壓縮完畢)")
            st.write(
                f"{'✅' if p2_bband_squeeze else '❌'} **布林極度變窄**：目前寬度 ({df['BB_Width'].iloc[-1]:.3f}) 處於近 60 日相對低檔區")
            st.write(
                f"{'✅' if p3_vol_pickup else '❌'} **資金初次試盤**：當日量比近 5 日均量放量 1.3 倍以上 ({int(df['Volume'].iloc[-1]):,})")
            st.write(
                f"{'✅' if p4_turning_strong else '❌'} **站上月線小漲**：股價站上 MA20，且單日小幅攻擊收紅 (+0.5% ~ +4.0%)")

            # =========================================================================
            # UI 顯示區：最佳進場與風控策略區
            # =========================================================================
            st.write("---")
            st.subheader("💡 最佳進場點與風控價位")

            in_col1, in_col2, in_col3 = st.columns(3)

            with in_col1:
                st.warning("🟢 **1. 壓低佈局區（低吸）**")
                st.write(f"**{dip_buy_low:.2f} ~ {dip_buy_high:.2f}**")
                st.caption("分批建立基本持股，靠近支撐/月線附近風險最低。")

            with in_col2:
                st.error("🚀 **2. 突破加碼點（追擊）**")
                st.write(f"**{breakout_buy:.2f}** (帶量突破時)")
                st.caption("當盤中放量強勢攻克第一壓力或前高時順勢加碼。")

            with in_col3:
                st.info("🛑 **3. 關鍵防守價（停損）**")
                st.write(f"**{stop_loss:.2f}** (-2% 防守)")
                st.caption("跌破此價位代表結構破壞或洗盤失敗，應果斷停損。")

            # =========================================================================
            # UI 顯示區：進行中飆股動能
            # =========================================================================
            st.write("---")
            st.subheader("🚀 進行中飆股動能檢測 (5大指標)")

            if rocket_score >= 4:
                st.success(f"🔥 **飆股攻擊分數：{rocket_score} / 5 分 (強烈攻擊動能！)**")
            elif rocket_score >= 2:
                st.warning(f"⚡ **飆股攻擊分數：{rocket_score} / 5 分 (動能溫和/累積中)**")
            else:
                st.error(f"⚪ **飆股攻擊分數：{rocket_score} / 5 分 (尚無明顯多頭攻擊力道)**")

            st.write(f"{'✅' if c1_ma else '❌'} **均線多頭排列**：現價 > MA5 > MA20 > MA60")
            st.write(f"{'✅' if c2_vol else '❌'} **成交量爆量**：大於 20 日均量 1.5 倍以上")
            st.write(f"{'✅' if c3_breakout else '❌'} **創波段新高**：收盤價突破近 20 日高點")
            st.write(f"{'✅' if c4_momentum else '❌'} **單日強勢攻擊**：單日漲幅 >= +3.0%")
            st.write(f"{'✅' if c5_rsi else '❌'} **RSI 強勢區塊**：RSI(14) >= 55")

            # =========================================================================
            # UI 顯示區：近半年日 K 線圖 (含布林通道、均線與成交量)
            # =========================================================================
            st.write("---")
            st.markdown("#### 📊 近半年日 K 線圖 (含布林通道、均線與成交量)")
            df_chart = df.tail(120).copy()

            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                subplot_titles=('日 K 線圖、布林通道與均線', '成交量'),
                row_heights=[0.7, 0.3]
            )

            # 1. K 線圖 (台股習慣：紅漲綠跌)
            fig.add_trace(go.Candlestick(
                x=df_chart.index.strftime('%Y-%m-%d'),
                open=df_chart['Open'],
                high=df_chart['High'],
                low=df_chart['Low'],
                close=df_chart['Close'],
                name='K線',
                increasing_line_color='#e74c3c',
                increasing_fillcolor='#e74c3c',
                decreasing_line_color='#2ecc71',
                decreasing_fillcolor='#2ecc71'
            ), row=1, col=1)

            # 2. 布林通道 (Upper / Lower + 填滿陰影)
            fig.add_trace(go.Scatter(
                x=df_chart.index.strftime('%Y-%m-%d'),
                y=df_chart['BB_Upper'],
                mode='lines',
                name='布林上軌 (+2σ)',
                line=dict(width=1, color='rgba(150, 150, 150, 0.6)', dash='dash')
            ), row=1, col=1)

            fig.add_trace(go.Scatter(
                x=df_chart.index.strftime('%Y-%m-%d'),
                y=df_chart['BB_Lower'],
                mode='lines',
                name='布林下軌 (-2σ)',
                line=dict(width=1, color='rgba(150, 150, 150, 0.6)', dash='dash'),
                fill='tonexty',
                fillcolor='rgba(200, 200, 200, 0.15)'
            ), row=1, col=1)

            # 3. 均線
            ma_colors = {'MA5': '#f39c12', 'MA10': '#3498db', 'MA20': '#9b59b6', 'MA60': '#e67e22'}
            for ma_key, color in ma_colors.items():
                if ma_key in df_chart.columns:
                    fig.add_trace(go.Scatter(
                        x=df_chart.index.strftime('%Y-%m-%d'),
                        y=df_chart[ma_key],
                        mode='lines',
                        name=ma_key,
                        line=dict(width=1.5, color=color)
                    ), row=1, col=1)

            # 4. 成交量圖 (紅漲綠跌)
            vol_colors = ['#e74c3c' if c >= o else '#2ecc71' for c, o in zip(df_chart['Close'], df_chart['Open'])]
            fig.add_trace(go.Bar(
                x=df_chart.index.strftime('%Y-%m-%d'),
                y=df_chart['Volume'],
                name='成交量',
                marker_color=vol_colors
            ), row=2, col=1)

            fig.update_layout(
                xaxis_rangeslider_visible=False,
                height=580,
                margin=dict(l=10, r=10, t=30, b=10),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                template="plotly_white"
            )

            # 隱藏非交易日空白間隔
            fig.update_xaxes(type='category')

            st.plotly_chart(fig, use_container_width=True)

            st.caption("Copyright © 2026 DUKE All rights reserved.")


if __name__ == "__main__":
    if not st.runtime.exists():
        sys.argv = ["streamlit", "run", __file__]
        sys.exit(stcli.main())
    else:
        run_app()
