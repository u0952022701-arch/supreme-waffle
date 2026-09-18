import sys
import pandas as pd
import numpy as np


# =========================================================================
# 0. 自動載入與快取台股名稱與代碼對照表 (上市/上櫃)
# =========================================================================
def run_app():
    import streamlit as st
    import yfinance as yf
    from scipy.signal import argrelextrema

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

        # 3. 備用常見熱門股票清單（避免網路阻擋時無法使用）
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

    st.set_page_config(page_title="Tingi股票診斷小工具", page_icon="📈", layout="centered")

    st.title("📈 Tingi股票診斷小工具")
    st.caption("自動計算關卡價位，診斷「起漲蓄勢訊號」與「進行中飆股動能」並精算「最佳進場與停損點」")

    # 載入台股名稱與代碼對照
    stock_map = load_taiwan_stock_mapping()

    # 初始化預設輸入值
    if "symbol_text" not in st.session_state:
        st.session_state["symbol_text"] = "2330.TW"

    # 選取選單項目時的自動帶入觸發函式
    def update_symbol_from_select():
        chosen = st.session_state.get("stock_lookup_select")
        if chosen and chosen != "-- 請選擇或輸入名稱過濾 --":
            st.session_state["symbol_text"] = stock_map[chosen]

    # =========================================================================
    # 功能新增：股票名稱尋找台股代碼
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
            st.success(f"🎯 已將 **{current_choice}** 代碼 (`{stock_map[current_choice]}`) 自動帶入下方診斷框！")

    input_symbol = st.text_input(
        "請輸入台股代碼（例如 2330.TW 或 3289.TWO）或美股代碼（例如 TSLA）:",
        key="symbol_text"
    )

    def get_ticker_symbol(symbol_str):
        symbol_str = symbol_str.strip().upper()
        if symbol_str.isdigit() and len(symbol_str) == 4:
            return f"{symbol_str}.TW"
        return symbol_str

    if st.button("開始診斷", type="primary"):
        ticker = get_ticker_symbol(input_symbol)

        with st.spinner('正在下載股票數據並計算中...'):
            stock = yf.Ticker(ticker)
            df = stock.history(period="1y")

        if df.empty or len(df) < 60:
            st.error(f"無法取得 [{ticker}] 足夠的歷史數據（需至少 60 個交易日），請確認代碼是否正確。")
        else:
            current_price = df['Close'].iloc[-1]

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

            # -------------------------------------------------------------------------
            # 支撐與壓力候選池
            # -------------------------------------------------------------------------
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

            # 壓低買進區間：近端支撐價 ~ 月線價 (或現價下方 1%~3%)
            dip_buy_low = near_sup if near_sup else ma20_val * 0.98
            dip_buy_high = min(current_price, max(ma20_val, dip_buy_low * 1.02))

            # 突破加碼點：第一壓力價，若無第一壓力則設為近20日高價
            breakout_buy = first_res if first_res else high_20d

            # 防守停損價：跌破近端支撐 2%，或跌破月線 2%
            stop_loss = (near_sup * 0.98) if near_sup else (ma20_val * 0.97)

            # =========================================================================
            # UI 顯示區：支撐壓力卡片
            # =========================================================================
            st.write("---")
            st.subheader(f"📌 標的：{ticker} ｜ 現價：**{current_price:.2f}** （單日：{daily_change:+.2f}%）")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    label="🛡️ 近端支撐區",
                    value=f"{near_sup:.2f}" if near_sup else "查無數據",
                    delta=f"{((near_sup - current_price) / current_price) * 100:.1f}%" if near_sup else "",
                )
                st.caption("最接近的支撐價位；跌破後結構轉弱。")
            with col2:
                st.metric(
                    label="⚡ 第一壓力區",
                    value=f"{first_res:.2f}" if first_res else "查無數據",
                    delta=f"+{((first_res - current_price) / current_price) * 100:.1f}%" if first_res else "",
                )
                st.caption("現價上方 10% 以內；第一個壓力區。")
            with col3:
                st.metric(
                    label="🔥 第二壓力區",
                    value=f"{second_res:.2f}" if second_res else "查無數據",
                    delta=f"+{((second_res - current_price) / current_price) * 100:.1f}%" if second_res else "",
                )
                st.caption("第一壓力之上；第二壓力區。")

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
            # UI 顯示區：進行中飆股動能（直接展示）
            # =========================================================================
            st.write("---")
            st.subheader("🚀 飆股動能進行中檢測 (5大指標)")

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

            st.caption("此小工具是DUKE設計所有，禁止轉載轉傳感謝配合!!")
if __name__ == "__main__":
    import streamlit as st
    from streamlit.web import cli as stcli

    if not st.runtime.exists():
        sys.argv = ["streamlit", "run", __file__]
        sys.exit(stcli.main())
    else:
        run_app()