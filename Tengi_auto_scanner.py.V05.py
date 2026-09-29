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
# 1. 全域資料與 API 模組
# =========================================================================
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
# 2. 盤中 1 分鐘 K 線微觀起漲動能模組 (驗證 1分鐘急漲、萬張大單、外盤吃單)
# =========================================================================
def analyze_intraday_micro_momentum(ticker):
    """
    透過 1 分鐘 K 線解析盤中微觀起漲訊號：
    - 1 分鐘內急漲 >= 2%
    - 連續 4 筆/分 爆量大單 (以張數與相對均量綜合評估)
    - 連續外盤大單 (連續強勢外盤吃單收最高)
    """
    try:
        ticker_obj = yf.Ticker(ticker)
        df_1m = ticker_obj.history(period="1d", interval="1m")
        if df_1m.empty or len(df_1m) < 5:
            return {"Is_1m_Surge_2Pct": False, "Is_4x_Large_Vol": False, "Is_Consecutive_Ask_Buy": False}

        close_1m = df_1m['Close']
        open_1m = df_1m['Open']
        high_1m = df_1m['High']
        low_1m = df_1m['Low']
        vol_1m = df_1m['Volume'] / 1000  # 轉張數

        # 1. 一分鐘內漲 2% 以上 (計算單一分鐘棒線高低點或近 2 分鐘內最大漲幅)
        m1_returns = (high_1m - low_1m) / low_1m
        m1_price_change = (close_1m - open_1m) / open_1m
        is_1m_surge = bool((m1_returns.iloc[-5:] >= 0.02).any() or (m1_price_change.iloc[-5:] >= 0.02).any())

        # 2. 連續 4 筆/分 萬張以上或相對極致大單 (單分量 >= 10,000張 或 為1m均量 5 倍以上)
        avg_vol_1m = vol_1m.mean()
        is_large_vol_bar = (vol_1m >= 10000) | (vol_1m >= avg_vol_1m * 5)
        # 檢查近 10 分鐘內是否有連續 4 棒大單
        is_4x_large_vol = bool((is_large_vol_bar.rolling(4).sum().iloc[-10:] >= 4).any())

        # 3. 連續外盤大單 (收盤價接近最高價 + 紅棒，代表買方強勢外盤成交)
        is_ask_buying = (close_1m > open_1m) & ((high_1m - close_1m) <= (high_1m - low_1m) * 0.2)
        is_consecutive_ask_buy = bool((is_ask_buying.rolling(4).sum().iloc[-10:] >= 4).any())

        return {
            "Is_1m_Surge_2Pct": is_1m_surge,
            "Is_4x_Large_Vol": is_4x_large_vol,
            "Is_Consecutive_Ask_Buy": is_consecutive_ask_buy
        }
    except Exception:
        return {"Is_1m_Surge_2Pct": False, "Is_4x_Large_Vol": False, "Is_Consecutive_Ask_Buy": False}


# =========================================================================
# 3. 核心算價與進場型態掃描模組
# =========================================================================
def analyze_single_ticker_for_scan(ticker, stock_name, market, df, min_turnover_wan=3000, enable_intraday_check=True):
    """結合 Tengi_auto_scanner 強勢型態、進場策略與 4 大微觀起漲條件"""
    try:
        code_str = ticker.replace(".TW", "").replace(".TWO", "")
        if code_str.startswith("00") or len(code_str) != 4 or df.empty or len(df) < 60:
            return None

        close, open_p, high, low = df['Close'], df['Open'], df['High'], df['Low']
        volume = df['Volume'] / 1000  # 轉為張數

        curr_price = float(close.iloc[-1])
        curr_open = float(open_p.iloc[-1])
        curr_high = float(high.iloc[-1])
        curr_low = float(low.iloc[-1])
        curr_vol = float(volume.iloc[-1])
        prev_close = float(close.iloc[-2])
        daily_change = ((curr_price - prev_close) / prev_close) * 100

        # 成交金額 (萬元)
        turnover_amount_wan = int((curr_price * curr_vol * 1000) / 10000)
        if turnover_amount_wan < min_turnover_wan:
            return None

        # -----------------------------------------------------------------
        # ★ 新增技術判斷 1: 股價創 5 日新高
        # -----------------------------------------------------------------
        high_5d_prev = high.iloc[-6:-1].max() if len(high) >= 6 else high.iloc[:-1].max()
        is_5d_high = curr_price >= high_5d_prev

        # -----------------------------------------------------------------
        # K 棒實體與上影線過濾
        # -----------------------------------------------------------------
        k_range = curr_high - curr_low
        k_body = curr_price - curr_open
        is_red_k = curr_price > curr_open
        body_ratio = (k_body / k_range) if k_range > 0 else 0
        upper_shadow = (curr_high - max(curr_price, curr_open)) / k_range if k_range > 0 else 0

        # 實體紅K: 收紅 + 實體比率 >= 45% + 上影線 <= 30%
        is_solid_red_k = is_red_k and (body_ratio >= 0.45) and (upper_shadow <= 0.30)

        # 均線與布林通道計算
        ma5 = close.rolling(5).mean()
        ma10 = close.rolling(10).mean()
        ma20 = close.rolling(20).mean()
        ma60 = close.rolling(60).mean()
        vol_ma5 = volume.rolling(5).mean()
        vol_ma20 = volume.rolling(20).mean()

        std20 = close.rolling(20).std()
        bb_upper, bb_lower = ma20 + (std20 * 2), ma20 - (std20 * 2)
        bb_width = (bb_upper - bb_lower) / ma20

        # 1. 均線高度糾結
        ma_today = [ma5.iloc[-1], ma10.iloc[-1], ma20.iloc[-1], ma60.iloc[-1]]
        ma_prev = [ma5.iloc[-2], ma10.iloc[-2], ma20.iloc[-2], ma60.iloc[-2]]
        squeeze_today = ((max(ma_today) - min(ma_today)) / ma20.iloc[-1]) <= 0.035
        squeeze_prev = ((max(ma_prev) - min(ma_prev)) / ma20.iloc[-2]) <= 0.035
        p3_ma_squeeze = squeeze_today or squeeze_prev

        # 2. 布林通道極致壓縮
        p4_bb_squeeze = bb_width.iloc[-1] <= (bb_width.iloc[-60:].min() * 1.25)

        # 3. 資金爆量試盤
        p5_vol_burst = (curr_vol >= vol_ma5.iloc[-1] * 1.3) and (curr_vol >= vol_ma20.iloc[-1] * 1.5)

        # 4. 強勢起漲突破 (實體紅K + 創近 20 日高價 + 收盤站上所有均線)
        is_20d_high = curr_price >= close.iloc[-20:].max()
        c_price_above_all = curr_price > max(ma_today)
        c_change_valid = (1.0 <= daily_change <= 6.5)
        p1_solid_red_k = is_solid_red_k and c_price_above_all and is_20d_high and c_change_valid

        # 5. 均線多頭排列
        ma_bullish = (ma5.iloc[-1] > ma10.iloc[-1] > ma20.iloc[-1] > ma60.iloc[-1])
        ma5_up = ma5.iloc[-1] > ma5.iloc[-2]
        p2_bullish_ma = ma_bullish and ma5_up

        score = sum([p1_solid_red_k, p2_bullish_ma, p3_ma_squeeze, p4_bb_squeeze, p5_vol_burst, is_5d_high])

        # -----------------------------------------------------------------
        # ★ 新增技術判斷 2, 3, 4: 盤中分時微觀數據驗證
        # -----------------------------------------------------------------
        intraday_res = {"Is_1m_Surge_2Pct": False, "Is_4x_Large_Vol": False, "Is_Consecutive_Ask_Buy": False}
        if enable_intraday_check and (p1_solid_red_k or p5_vol_burst or is_5d_high):
            intraday_res = analyze_intraday_micro_momentum(ticker)

        near_sup = round(float(ma20.iloc[-1] * 0.98), 2)
        stop_loss = round(float(min(curr_price * 0.95, near_sup)), 2)
        target_price = round(curr_price * 1.15, 2)

        # 自動匹配進場策略邏輯
        if p1_solid_red_k and p5_vol_burst and intraday_res["Is_1m_Surge_2Pct"]:
            entry_strategy = "⚡ 極速追擊 (分時強攻)"
            entry_min = round(curr_price * 0.995, 2)
            entry_max = round(curr_price * 1.015, 2)
        elif p1_solid_red_k and p5_vol_burst:
            entry_strategy = "⚡ 突破追擊 (帶量強攻)"
            entry_min = round(curr_price * 0.995, 2)
            entry_max = round(curr_price * 1.015, 2)
        elif p3_ma_squeeze and p4_bb_squeeze:
            entry_strategy = "🎯 壓縮低吸 (變盤前夕)"
            entry_min = round(min(ma5.iloc[-1], ma20.iloc[-1]), 2)
            entry_max = round(curr_price, 2)
        elif p2_bullish_ma and p5_vol_burst:
            entry_strategy = "🔥 順勢加碼 (多頭點火)"
            entry_min = round(ma5.iloc[-1], 2)
            entry_max = round(curr_price, 2)
        else:
            entry_strategy = "👀 守株待兔 (回測支撐)"
            entry_min = round(near_sup, 2)
            entry_max = round(ma5.iloc[-1], 2)

        entry_range_str = f"{entry_min} ~ {entry_max}"

        # 計算風報比
        risk = curr_price - stop_loss
        reward = target_price - curr_price
        rr_ratio = round(reward / risk, 2) if risk > 0 else 0.0

        return {
            "Code": code_str, "Name": stock_name, "Market": market,
            "Price": round(curr_price, 2), "Daily_Change_%": round(daily_change, 2),
            "Volume_張": int(curr_vol), "Turnover_萬": turnover_amount_wan,
            "Pre_Launch_Score": score,
            "High_5D": "✅" if is_5d_high else "❌",
            "M1_Surge_2%": "✅" if intraday_res["Is_1m_Surge_2Pct"] else "❌",
            "4x_Large_Vol": "✅" if intraday_res["Is_4x_Large_Vol"] else "❌",
            "Ask_Buy_Series": "✅" if intraday_res["Is_Consecutive_Ask_Buy"] else "❌",
            "Entry_Strategy": entry_strategy,
            "Entry_Range": entry_range_str,
            "Stop_Loss": stop_loss,
            "Target_Price": target_price,
            "RR_Ratio": rr_ratio,
            "Solid_Red_K": "✅" if p1_solid_red_k else "❌",
            "Bullish_MA": "✅" if p2_bullish_ma else "❌",
            "MA_Squeeze": "✅" if p3_ma_squeeze else "❌",
            "BB_Squeeze": "✅" if p4_bb_squeeze else "❌",
            "Vol_Burst": "✅" if p5_vol_burst else "❌"
        }
    except Exception:
        return None


# =========================================================================
# 4. UI 介面：市場飆股自動掃描
# =========================================================================
def render_market_scanner():
    st.subheader("🚀 市場飆股型態與微觀起漲自動掃描")

    with st.expander("💡 查看 4 大新增微觀起漲條件與算價說明"):
        st.markdown("""
        ### 🔥 新增 4 大微觀起漲條件
        1. **創 5 日新高**：當日收盤/現價突破過去 5 個交易日最高價。
        2. **1 分鐘內漲 ≥ 2%**：盤中分時 K 線出現單分鐘內拉升 2% 以上之攻擊波。
        3. **連續萬張/特大單 4 筆**：分時 K 線連續爆出特大成交量（單分 $\ge 10,000$ 張或達均量 5 倍以上）。
        4. **連續外盤大單**：買方以賣價（外盤）連續吃單，推升分時棒線高檔收亮紅 K。

        ---
        ### 🎯 策略擬定與風報比
        * **風報比 (RR Ratio)**：$(\text{目標價} - \text{現價}) / (\text{現價} - \text{停損價})$，建議選擇 **RR Ratio $\ge 2.0$** 之標的。
        """)

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        selected_market = st.selectbox("市場範疇", ["全部", "上市", "上櫃"], index=0)
    with col2:
        min_score = st.slider("最低起漲指數", 1, 6, 3)
    with col3:
        min_volume = st.number_input("最低成交量 (張)", value=300, step=100)
    with col4:
        min_turnover = st.number_input("最低成交額 (萬元)", value=3000, step=500)
    with col5:
        max_scan_limit = st.selectbox("掃描數量限制", [150, 300, 600, 1000, "全部個股"], index=1)

    enable_intraday = st.checkbox("⚡ 啟用 1 分鐘 K 線微觀動能檢測 (急漲 / 大單 / 外盤吃單)", value=True)

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
                            min_turnover_wan=min_turnover,
                            enable_intraday_check=enable_intraday
                        )

                        if res and res['Pre_Launch_Score'] >= min_score and res['Volume_張'] >= min_volume:
                            if res['Solid_Red_K'] == "✅" or res['MA_Squeeze'] == "✅" or res['High_5D'] == "✅":
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
            res_df = pd.DataFrame(all_results).sort_values(
                by=["Pre_Launch_Score", "RR_Ratio", "Turnover_萬"],
                ascending=[False, False, False]
            )
            st.success(f"🎯 成功找出 **{len(res_df)}** 檔「微觀發動」潛力標的！")

            display_cols = {
                "Code": "股票代碼", "Name": "股票名稱", "Market": "市場", "Price": "現價",
                "Daily_Change_%": "漲幅(%)", "Volume_張": "成交量(張)", "Turnover_萬": "成交額(萬)",
                "Pre_Launch_Score": "起漲指數", "High_5D": "創5日高", "M1_Surge_2%": "1分急漲2%",
                "4x_Large_Vol": "連續特大單", "Ask_Buy_Series": "外盤連續吃單",
                "Entry_Strategy": "進場策略", "Entry_Range": "建議買點區間",
                "Stop_Loss": "建議停損", "Target_Price": "目標價(+15%)", "RR_Ratio": "風報比"
            }

            st.dataframe(res_df[list(display_cols.keys())].rename(columns=display_cols), width="stretch",
                         hide_index=True)

            csv = res_df[list(display_cols.keys())].rename(columns=display_cols).to_csv(index=False).encode('utf-8-sig')
            st.download_button("📥 下載篩選結果 CSV 報表", csv,
                               f"micro_breakout_stocks_{datetime.datetime.now().strftime('%Y%m%d')}.csv", "text/csv")
        else:
            st.warning("⚠️ 暫無符合條件標的，建議放寬門檻條件。")


# =========================================================================
# 5. Streamlit 主程式進入點
# =========================================================================
def run_app():
    st.set_page_config(page_title="TENGI 飆股型態與微觀起漲系統", page_icon="📈", layout="wide")
    st.title("📈 TENGI 飆股型態與微觀起漲系統")
    st.caption("Copyright © 2026 DUKE All rights reserved.")

    render_market_scanner()


if __name__ == "__main__":
    from streamlit.web import cli as stcli

    if not st.runtime.exists():
        sys.argv = ["streamlit", "run", __file__]
        sys.exit(stcli.main())
    else:
        run_app()