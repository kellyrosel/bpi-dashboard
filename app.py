# if want to push updated code to github, run in terminal: git add .
# git commit -m "Updated chart logic"
# git push  

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import yfinance as yf
import streamlit as st

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Bullish Percent Index Dashboard",
    layout="wide"
)

# ─────────────────────────────────────────────────────────────────────────────
# BUILT-IN INDEX TICKER UNIVERSES
# ─────────────────────────────────────────────────────────────────────────────

_BUILTIN_UNIVERSES = {
    "DOW": [
        "AAPL","AMGN","AXP","BA","CAT","CRM","CSCO","CVX","DIS","DOW",
        "GS","HD","HON","IBM","JNJ","JPM","KO","MCD","MMM","MRK",
        "MSFT","NKE","NVDA","PG","SHW","TRV","UNH","V","VZ","WMT",
    ],
    "NDX": [
        "AAPL","MSFT","NVDA","AMZN","META","GOOG","GOOGL","TSLA","AVGO","COST",
        "NFLX","TMUS","AMD","PEP","ADBE","TXN","QCOM","INTU","AMGN","AMAT",
        "ISRG","MU","LRCX","KLAC","PANW","ADI","REGN","SNPS","CDNS","MELI",
        "ORLY","FTNT","MAR","CTAS","MNST","WDAY","ABNB","TEAM","NXPI","CRWD",
        "MRVL","AEP","PAYX","FAST","ROST","IDXX","ODFL","PCAR","GEHC","TTD",
    ],
    "XLK": [
        "AAPL","MSFT","NVDA","AVGO","ORCL","CRM","AMD","QCOM","TXN","AMAT",
        "INTU","ADI","LRCX","MU","KLAC","SNPS","CDNS","PANW","CSCO","ACN",
        "IBM","FTNT","NOW","MRVL","MPWR","TER","HPE","GLW","KEYS","ZBRA",
    ],
    "XLF": [
        "BRK-B","JPM","V","MA","BAC","WFC","GS","MS","SPGI","BLK",
        "AXP","USB","PNC","TFC","COF","CB","ICE","CME","AON","MMC",
        "MET","PRU","ALL","HIG","SCHW","STT","BK","NTRS","RF","FITB",
    ],
    "XLE": [
        "XOM","CVX","COP","EOG","SLB","MPC","VLO","PSX","OXY","HAL",
        "BKR","DVN","CTRA","APA","TRGP","EQT","WMB","OKE","KMI","PR",
    ],
    "XLV": [
        "LLY","UNH","JNJ","ABBV","MRK","TMO","ABT","DHR","BMY","AMGN",
        "PFE","ISRG","SYK","GILD","MDT","VRTX","REGN","ZTS","CI","HCA",
        "ELV","CVS","BSX","MCK","IQV","DXCM","IDXX","RMD","EW","WST",
    ],
    "XLI": [
        "GE","CAT","RTX","HON","UPS","DE","LMT","NOC","GD","BA",
        "MMM","EMR","ETN","PH","ROK","AME","CTAS","VRSK","FAST","XYL",
        "PWR","CARR","OTIS","IR","TT","FTV","J","SWK","HWM","GNRC",
    ],
    "XLY": [
        "AMZN","TSLA","HD","MCD","NKE","LOW","SBUX","TJX","BKNG","CMG",
        "MAR","YUM","ORLY","AZO","ROST","DHI","PHM","LEN","F","GM",
        "HLT","EXPE","LVS","MGM","WYNN","GRMN","RL","TPR","NVR","DECK",
    ],
    "XLP": [
        "PG","KO","PEP","COST","WMT","PM","MO","CL","MDLZ","EL",
        "STZ","GIS","KHC","KMB","SYY","HSY","MKC","CHD","CLX","HRL",
        "CAG","TSN","LW","COTY","NWL","INGR","POST","SFM","BF-B","TAP",
    ],
    "NYSE": [
        "JPM","BAC","WFC","GS","MS","BLK","AXP","USB","PNC","TFC",
        "COF","CB","ICE","CME","AON","MMC","MET","PRU","ALL","SCHW",
        "XOM","CVX","COP","EOG","SLB","MPC","VLO","PSX","OXY","HAL",
        "BKR","DVN","TRGP","EQT","WMB","OKE","KMI","APA","CTRA","PR",
        "JNJ","UNH","LLY","ABBV","MRK","PFE","TMO","ABT","DHR","BMY",
        "ISRG","SYK","GILD","MDT","VRTX","REGN","ZTS","CI","HCA","ELV",
        "HD","WMT","COST","TGT","LOW","MCD","SBUX","NKE","YUM","TJX",
        "PG","KO","PEP","CL","MO","PM","GIS","KMB","SYY","HSY",
        "CAT","DE","HON","MMM","GE","BA","LMT","RTX","NOC","GD",
        "UPS","EMR","ETN","PH","CTAS","FAST","VRSK","AME","ROK","XYL",
        "NEE","DUK","SO","AEP","EXC","XEL","ED","ES","D","PCG",
        "AMT","PLD","CCI","EQIX","SPG","PSA","O","ARE","VTR","WY",
    ],
}


@st.cache_data(ttl=3600)
def fetch_sp500_tickers():
    try:
        tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        tickers = tables[0]["Symbol"].tolist()
        return [t.replace(".", "-") for t in tickers]
    except Exception:
        return _BUILTIN_UNIVERSES["NDX"]


# ─────────────────────────────────────────────────────────────────────────────
# CACHED CHUNKED DOWNLOADER  ← NEW
# Splits large ticker lists into chunks of 100, downloads in parallel,
# then concatenates. Results are cached for 1 hour so re-runs are instant.
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner="Downloading price data…")
def fetch_price_data(tickers: tuple, start: str, end: str = None) -> pd.DataFrame:
    """
    Download adjusted close prices for a list of tickers.
    - Splits into chunks of 100 to avoid yfinance timeouts on large universes.
    - Results are cached for 1 hour; repeat runs with the same inputs are instant.
    - tickers must be a sorted tuple so the cache key is stable regardless of order.
    """
    chunk_size = 100
    ticker_list = list(tickers)
    chunks = [ticker_list[i:i + chunk_size] for i in range(0, len(ticker_list), chunk_size)]

    frames = []
    for chunk in chunks:
        try:
            raw = yf.download(
                chunk,
                start=start,
                end=end,
                progress=False,
                auto_adjust=True,
                threads=True,
            )
            if isinstance(raw.columns, pd.MultiIndex):
                frames.append(raw["Close"])
            else:
                frames.append(raw[["Close"]] if "Close" in raw.columns else raw)
        except Exception:
            continue  # skip failed chunks rather than crashing the whole run

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, axis=1)
    combined = combined.loc[:, ~combined.columns.duplicated()]
    return combined.dropna(axis=1, thresh=50)


def pf_has_buy_signal(prices: pd.Series, box_pct: float = 2.0, reversal: int = 3) -> bool:
    prices = prices.dropna()
    if len(prices) < 10:
        return False

    def box(p): return p * (box_pct / 100.0)
    def rd(p): return np.floor(p / box(p)) * box(p)

    p0 = float(prices.iloc[0])
    cur = rd(p0)
    direction = None
    columns = []
    cur_boxes = []

    for price in prices:
        price = float(price)
        bx = box(max(cur, 1e-6))

        if direction is None:
            r = rd(price)
            if r > cur:
                direction = "X"
                b = cur + bx
                while b <= r + 1e-9:
                    cur_boxes.append(b)
                    b += box(max(b, 1e-6))
                cur = r
            elif r < cur:
                direction = "O"
                b = cur - bx
                while b >= r - 1e-9:
                    cur_boxes.append(b)
                    b -= box(max(b, 1e-6))
                cur = r
            continue

        bx = box(max(cur, 1e-6))
        if direction == "X":
            if price >= cur + bx:
                while cur + box(max(cur, 1e-6)) <= price + 1e-9:
                    cur += box(max(cur, 1e-6))
                    cur_boxes.append(cur)
            elif price <= cur - reversal * bx:
                columns.append(("X", list(cur_boxes)))
                cur_boxes = []
                direction = "O"
                step = cur
                while step - box(max(step, 1e-6)) >= price - 1e-9:
                    step -= box(max(step, 1e-6))
                    cur_boxes.append(step)
                cur = step
        else:
            if price <= cur - bx:
                while cur - box(max(cur, 1e-6)) >= price - 1e-9:
                    cur -= box(max(cur, 1e-6))
                    cur_boxes.append(cur)
            elif price >= cur + reversal * bx:
                columns.append(("O", list(cur_boxes)))
                cur_boxes = []
                direction = "X"
                step = cur
                while step + box(max(step, 1e-6)) <= price + 1e-9:
                    step += box(max(step, 1e-6))
                    cur_boxes.append(step)
                cur = step

    if cur_boxes:
        columns.append((direction or "X", cur_boxes))

    x_cols = [(i, c) for i, (t, c) in enumerate(columns) if t == "X" and c]
    if len(x_cols) < 2:
        return False

    last_x_high = max(x_cols[-1][1])
    prev_x_high = max(x_cols[-2][1])
    return last_x_high > prev_x_high


def compute_bpi_history(price_data: pd.DataFrame, box_pct: float = 2.0, reversal: int = 3, freq: str = "W") -> pd.Series:
    prices = price_data.resample(freq).last().dropna(how="all")
    dates = prices.index
    bpi_vals = []

    for i, date in enumerate(dates):
        if i < 5:
            bpi_vals.append(np.nan)
            continue

        window = price_data.loc[:date]
        n_buy = 0
        n_valid = 0

        for ticker in window.columns:
            series = window[ticker].dropna()
            if len(series) < 10:
                continue
            n_valid += 1
            if pf_has_buy_signal(series, box_pct=box_pct, reversal=reversal):
                n_buy += 1

        bpi_vals.append(round(n_buy / n_valid * 100, 2) if n_valid else np.nan)

    return pd.Series(bpi_vals, index=dates, name="BPI").dropna()


def build_bpi_pf_columns(bpi_series: pd.Series, box_size: float = 2.0, reversal: int = 6):
    prices = bpi_series.dropna()
    dates = prices.index
    box = box_size

    def rd(p): return np.floor(p / box) * box

    p0 = float(prices.iloc[0])
    cur = rd(p0)
    direction = None
    columns = []
    cur_boxes = []
    cur_dates = []

    for date, price in zip(dates, prices):
        price = float(price)
        if direction is None:
            r = rd(price)
            if r > cur:
                direction = "X"
                b = cur + box
                while b <= r + 1e-9:
                    cur_boxes.append(round(b, 4))
                    cur_dates.append(date)
                    b += box
                cur = r
            elif r < cur:
                direction = "O"
                b = cur - box
                while b >= r - 1e-9:
                    cur_boxes.append(round(b, 4))
                    cur_dates.append(date)
                    b -= box
                cur = r
            continue

        if direction == "X":
            if price >= cur + box:
                while cur + box <= price + 1e-9:
                    cur += box
                    cur_boxes.append(round(cur, 4))
                    cur_dates.append(date)
            elif price <= cur - reversal * box:
                columns.append({"type": "X", "boxes": list(cur_boxes), "dates": list(cur_dates)})
                cur_boxes, cur_dates = [], []
                direction = "O"
                step = cur - box
                while step >= price - 1e-9:
                    cur_boxes.append(round(step, 4))
                    cur_dates.append(date)
                    step -= box
                cur = step + box
        else:
            if price <= cur - box:
                while cur - box >= price - 1e-9:
                    cur -= box
                    cur_boxes.append(round(cur, 4))
                    cur_dates.append(date)
            elif price >= cur + reversal * box:
                columns.append({"type": "O", "boxes": list(cur_boxes), "dates": list(cur_dates)})
                cur_boxes, cur_dates = [], []
                direction = "X"
                step = cur + box
                while step <= price + 1e-9:
                    cur_boxes.append(round(step, 4))
                    cur_dates.append(date)
                    step += box
                cur = step - box

    if cur_boxes:
        columns.append({"type": direction or "X", "boxes": cur_boxes, "dates": cur_dates})

    return columns


def bpi_market_status(columns: list, current_bpi: float) -> tuple:
    if not columns:
        return "Unknown", "#8b949e"

    last_col = columns[-1]
    direction = last_col["type"]

    if direction == "X":
        if current_bpi >= 50:
            return "Bull Confirmed", "#39ff14"
        elif current_bpi >= 30:
            return "Bear Correction", "#7ee8fa"
        else:
            return "Bull Alert", "#ffd700"
    else:
        if current_bpi <= 50:
            return "Bear Confirmed", "#ff4d6d"
        elif current_bpi <= 70:
            return "Bear Alert", "#ff9500"
        else:
            return "Bull Correction", "#c084fc"


class BullishPercentIndex:
    def __init__(
        self,
        index: str = "SP500",
        tickers: list = None,
        name: str = None,
        start: str = "2019-01-01",
        end: str = None,
        stock_box_pct: float = 2.0,
        stock_reversal: int = 3,
        bpi_box_size: float = 2.0,
        bpi_reversal: int = 6,
        freq: str = "W",
        figsize: tuple = (16, 10),
    ):
        self.figsize = figsize
        self.bpi_box_size = bpi_box_size
        self.bpi_reversal = bpi_reversal
        self.freq = freq

        if tickers is not None and len(tickers) > 0:
            self.tickers = tickers
            self.name = name or "Custom"
        else:
            idx = (index or "SP500").upper()
            if idx == "SP500":
                self.tickers = fetch_sp500_tickers()
            else:
                self.tickers = _BUILTIN_UNIVERSES.get(idx, _BUILTIN_UNIVERSES["NDX"])
            self.name = name or idx

        # ── UPDATED: use cached chunked downloader instead of raw yf.download ──
        self.price_data = fetch_price_data(
            tuple(sorted(self.tickers)),   # sorted tuple = stable cache key
            start=start,
            end=end,
        )
        # ────────────────────────────────────────────────────────────────────────

        self.bpi_series = compute_bpi_history(
            self.price_data,
            box_pct=stock_box_pct,
            reversal=stock_reversal,
            freq=freq,
        )

        self.columns = build_bpi_pf_columns(
            self.bpi_series,
            box_size=bpi_box_size,
            reversal=bpi_reversal,
        )

        self.current_bpi = float(self.bpi_series.iloc[-1]) if len(self.bpi_series) else np.nan
        self.status, self.status_color = bpi_market_status(self.columns, self.current_bpi)

    def _snap(self, val: float) -> float:
        return round(round(val / self.bpi_box_size) * self.bpi_box_size, 4)

    def summary(self) -> pd.DataFrame:
        last_col_type = self.columns[-1]["type"] if self.columns else "?"
        return pd.DataFrame([{
            "Index": self.name,
            "Current BPI": f"{self.current_bpi:.1f}%" if pd.notna(self.current_bpi) else "N/A",
            "Column": "X (Rising)" if last_col_type == "X" else "O (Falling)",
            "Status": self.status,
            "# Stocks": len(self.price_data.columns),
            "# P&F Columns": len(self.columns),
        }])

    def plot_figure(self, title: str = None):
        cols = self.columns
        if not cols:
            return None

        fig, (ax, ax_line) = plt.subplots(
            2, 1,
            figsize=self.figsize,
            gridspec_kw={"height_ratios": [5, 1.2]},
            facecolor="#0d1117"
        )
        ax.set_facecolor("#0d1117")
        ax_line.set_facecolor("#0d1117")

        all_boxes = [b for c in cols for b in c["boxes"]]
        bpi_min = max(0, min(all_boxes) - self.bpi_box_size * 2)
        bpi_max = min(100, max(all_boxes) + self.bpi_box_size * 2)
        price_levels = np.arange(
            self._snap(bpi_min),
            self._snap(bpi_max) + self.bpi_box_size,
            self.bpi_box_size
        ).tolist()
        price_levels = [round(p, 4) for p in price_levels]
        level_to_y = {p: i for i, p in enumerate(price_levels)}
        n_levels = len(price_levels)
        n_cols = len(cols)

        zones = [
            (70, 100, "#39ff14", 0.06, "Overbought (>70)"),
            (30, 70, "#ffffff", 0.02, "Neutral (30–70)"),
            (0, 30, "#ff4d6d", 0.06, "Oversold (<30)"),
        ]
        for z_low, z_high, z_color, z_alpha, _ in zones:
            y0 = level_to_y.get(self._snap(z_low), 0)
            y1 = level_to_y.get(self._snap(z_high), n_levels - 1)
            ax.axhspan(y0, y1, facecolor=z_color, alpha=z_alpha, zorder=0)

        for ref_val, ref_label, ref_color in [
            (30, "30%", "#ff4d6d"),
            (50, "50%", "#8b949e"),
            (70, "70%", "#39ff14"),
        ]:
            y = level_to_y.get(self._snap(ref_val))
            if y is not None:
                ax.axhline(y, color=ref_color, linewidth=0.7, linestyle="--", alpha=0.5, zorder=1)
                ax.text(n_cols + 0.3, y, ref_label, color=ref_color, fontsize=6.5,
                        fontfamily="monospace", va="center")

        for col_idx, col in enumerate(cols):
            ctype = col["type"]
            color = "#00e5ff" if ctype == "X" else "#ff4d6d"

            col_max = max(col["boxes"]) if col["boxes"] else 0
            col_min = min(col["boxes"]) if col["boxes"] else 0
            if ctype == "X" and col_max > 50 >= col_min:
                color = "#39ff14"
            elif ctype == "O" and col_min < 50 <= col_max:
                color = "#ff6b00"

            for price in col["boxes"]:
                y = level_to_y.get(round(price, 4))
                if y is None:
                    continue
                ax.text(col_idx, y, ctype, color=color, fontsize=7.5,
                        fontfamily="monospace", fontweight="bold",
                        ha="center", va="center", zorder=3)

        step = max(1, n_levels // 15)
        ax.set_yticks(range(0, n_levels, step))
        ax.set_yticklabels(
            [f"{price_levels[i]:.0f}%" for i in range(0, n_levels, step)],
            color="#8b949e", fontsize=7, fontfamily="monospace"
        )

        xt_pos, xt_lbl, last_qtr = [], [], None
        for i, col in enumerate(cols):
            if col["dates"]:
                d = pd.Timestamp(col["dates"][0])
                qtr = (d.year, d.quarter)
                if qtr != last_qtr:
                    xt_pos.append(i)
                    xt_lbl.append(d.strftime("%b\n%y"))
                    last_qtr = qtr

        ax.set_xticks(xt_pos)
        ax.set_xticklabels(xt_lbl, color="#8b949e", fontsize=6.5, fontfamily="monospace")
        ax.grid(True, color="#21262d", linewidth=0.35, linestyle="--", alpha=0.5, zorder=0)
        ax.set_xlim(-1, n_cols + 2)
        ax.set_ylim(-1, n_levels + 1)

        badge_text = f"  {self.status}  \n  BPI: {self.current_bpi:.1f}%  "
        ax.text(
            0.985, 0.97, badge_text,
            transform=ax.transAxes,
            ha="right", va="top",
            fontsize=9, fontfamily="monospace", fontweight="bold",
            color="#0d1117",
            bbox=dict(boxstyle="round,pad=0.4", facecolor=self.status_color,
                      edgecolor=self.status_color, alpha=0.9)
        )

        bv = self.bpi_series.values
        xt = range(len(bv))
        ax_line.plot(xt, bv, color="#7ee8fa", linewidth=1.0, alpha=0.9)
        ax_line.axhline(50, color="#8b949e", linewidth=0.6, linestyle="--")
        ax_line.axhspan(70, 100, facecolor="#39ff14", alpha=0.07)
        ax_line.axhspan(0, 30, facecolor="#ff4d6d", alpha=0.07)
        ax_line.fill_between(xt, bv, 50, where=np.array(bv) >= 50,
                             facecolor="#39ff1425", interpolate=True)
        ax_line.fill_between(xt, bv, 50, where=np.array(bv) < 50,
                             facecolor="#ff4d6d25", interpolate=True)
        ax_line.set_xlim(0, len(bv))
        ax_line.set_ylim(0, 100)
        ax_line.set_ylabel("BPI %", color="#8b949e", fontsize=6.5, fontfamily="monospace")
        ax_line.yaxis.set_tick_params(labelcolor="#8b949e", labelsize=6)
        ax_line.xaxis.set_tick_params(labelbottom=False)
        ax_line.grid(True, color="#21262d", linewidth=0.3, linestyle="--", alpha=0.5)

        for sp in ax_line.spines.values():
            sp.set_edgecolor("#30363d")

        freq_label = {"W": "Weekly", "M": "Monthly", "D": "Daily"}.get(self.freq, self.freq)
        chart_title = title or f"{self.name} Bullish Percent Index"
        subtitle = (
            f"Box: {self.bpi_box_size:.0f}pt  |  "
            f"Reversal: {self.bpi_reversal}-Box  |  "
            f"Sampling: {freq_label}  |  "
            f"Universe: {len(self.price_data.columns)} stocks  |  "
            f"Dorsey Wright Method"
        )
        ax.set_title(chart_title, color="#e6edf3", fontsize=13,
                     fontfamily="monospace", fontweight="bold", pad=14)
        ax.text(0.5, 1.012, subtitle, transform=ax.transAxes,
                ha="center", fontsize=7, color="#8b949e", fontfamily="monospace")

        legend_elements = [
            Line2D([0],[0], marker="$X$", color="w", markerfacecolor="#00e5ff",
                   markersize=9, label="BPI Rising (X)", linestyle="None"),
            Line2D([0],[0], marker="$O$", color="w", markerfacecolor="#ff4d6d",
                   markersize=9, label="BPI Falling (O)", linestyle="None"),
            Line2D([0],[0], marker="$X$", color="w", markerfacecolor="#39ff14",
                   markersize=9, label="Bull Confirmed Crossover", linestyle="None"),
            Line2D([0],[0], marker="$O$", color="w", markerfacecolor="#ff6b00",
                   markersize=9, label="Bear Confirmed Crossover", linestyle="None"),
            mpatches.Patch(facecolor="#39ff14", alpha=0.25, label="Overbought zone (>70%)"),
            mpatches.Patch(facecolor="#ff4d6d", alpha=0.25, label="Oversold zone (<30%)"),
        ]
        ax.legend(handles=legend_elements, loc="upper left",
                  framealpha=0.25, facecolor="#161b22",
                  labelcolor="#e6edf3", fontsize=7, edgecolor="#30363d")

        for sp in ax.spines.values():
            sp.set_edgecolor("#30363d")

        plt.tight_layout(h_pad=0.4)
        return fig


def plot_bpi_dashboard_figure(bpi_objects: list):
    n = len(bpi_objects)
    if n == 0:
        return None

    cols_grid = min(n, 3)
    rows_grid = (n + cols_grid - 1) // cols_grid

    fig, axes = plt.subplots(
        rows_grid, cols_grid,
        figsize=(7 * cols_grid, 6 * rows_grid),
        facecolor="#0d1117",
        squeeze=False
    )
    fig.suptitle(
        "Bullish Percent Index Dashboard — Dorsey Wright Style",
        color="#e6edf3",
        fontsize=14,
        fontfamily="monospace",
        fontweight="bold",
        y=1.01
    )

    for idx, bpi in enumerate(bpi_objects):
        row, col = divmod(idx, cols_grid)
        ax = axes[row][col]
        ax.set_facecolor("#0d1117")

        cols_data = bpi.columns
        all_boxes = [b for c in cols_data for b in c["boxes"]]
        if not all_boxes:
            ax.set_title(f"{bpi.name}\n(insufficient data)", color="#8b949e", fontsize=9)
            continue

        bpi_min = max(0, min(all_boxes) - bpi.bpi_box_size * 2)
        bpi_max = min(100, max(all_boxes) + bpi.bpi_box_size * 2)
        price_levels = np.arange(
            bpi._snap(bpi_min),
            bpi._snap(bpi_max) + bpi.bpi_box_size,
            bpi.bpi_box_size
        ).tolist()
        price_levels = [round(p, 4) for p in price_levels]
        level_to_y = {p: i for i, p in enumerate(price_levels)}
        n_levels = len(price_levels)
        n_cols_pf = len(cols_data)

        for z_low, z_high, z_color, z_alpha in [
            (70, 100, "#39ff14", 0.07),
            (0, 30, "#ff4d6d", 0.07),
        ]:
            y0 = level_to_y.get(bpi._snap(z_low), 0)
            y1 = level_to_y.get(bpi._snap(z_high), n_levels - 1)
            ax.axhspan(y0, y1, facecolor=z_color, alpha=z_alpha)

        for ref_val, ref_color in [(30, "#ff4d6d"), (50, "#8b949e"), (70, "#39ff14")]:
            y = level_to_y.get(bpi._snap(ref_val))
            if y is not None:
                ax.axhline(y, color=ref_color, linewidth=0.6, linestyle="--", alpha=0.5)

        for ci, c in enumerate(cols_data):
            ctype = c["type"]
            color = "#00e5ff" if ctype == "X" else "#ff4d6d"
            col_max = max(c["boxes"]) if c["boxes"] else 0
            col_min = min(c["boxes"]) if c["boxes"] else 0

            if ctype == "X" and col_max > 50 >= col_min:
                color = "#39ff14"
            elif ctype == "O" and col_min < 50 <= col_max:
                color = "#ff6b00"

            for price in c["boxes"]:
                y = level_to_y.get(round(price, 4))
                if y is None:
                    continue
                ax.text(ci, y, ctype, color=color, fontsize=6,
                        fontfamily="monospace", fontweight="bold",
                        ha="center", va="center")

        step = max(1, n_levels // 10)
        ax.set_yticks(range(0, n_levels, step))
        ax.set_yticklabels(
            [f"{price_levels[i]:.0f}%" for i in range(0, n_levels, step)],
            color="#8b949e", fontsize=6, fontfamily="monospace"
        )

        ax.grid(True, color="#21262d", linewidth=0.3, linestyle="--", alpha=0.5)
        ax.set_xlim(-1, n_cols_pf + 1)
        ax.set_ylim(-1, n_levels + 1)

        for sp in ax.spines.values():
            sp.set_edgecolor("#30363d")

        ax.text(
            0.97, 0.97,
            f" {bpi.status} \n BPI: {bpi.current_bpi:.1f}% ",
            transform=ax.transAxes,
            ha="right", va="top",
            fontsize=7.5, fontfamily="monospace", fontweight="bold",
            color="#0d1117",
            bbox=dict(boxstyle="round,pad=0.3",
                      facecolor=bpi.status_color,
                      edgecolor=bpi.status_color,
                      alpha=0.9)
        )

        ax.set_title(
            f"{bpi.name} BPI",
            color="#e6edf3",
            fontsize=10,
            fontfamily="monospace",
            fontweight="bold"
        )

    for idx in range(n, rows_grid * cols_grid):
        r, c = divmod(idx, cols_grid)
        axes[r][c].set_visible(False)

    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# STREAMLIT UI
# ─────────────────────────────────────────────────────────────────────────────

st.title("Bullish Percent Index Dashboard")
st.caption("Run your BPI charts in the browser and view them cleanly on your phone.")

with st.sidebar:
    st.header("Inputs")

    mode = st.radio("Universe Type", ["Built-in Index", "Custom Tickers"])

    if mode == "Built-in Index":
        selected_index = st.selectbox(
            "Choose universe",
            ["SP500", "NDX", "DOW", "NYSE", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP"]
        )
        custom_tickers = None
        custom_name = selected_index
    else:
        custom_text = st.text_area(
            "Enter tickers separated by commas",
            value="AAPL,MSFT,NVDA,AMZN,META,GOOG"
        )
        custom_tickers = [t.strip().upper() for t in custom_text.split(",") if t.strip()]
        custom_name = st.text_input("Custom universe name", value="Custom Basket")
        selected_index = None

    start_date = st.date_input("Start date", value=pd.to_datetime("2020-01-01"))
    freq = st.selectbox("Sampling frequency", ["W", "M", "D"], index=0)
    stock_box_pct = st.number_input("Stock P&F box %", value=2.0, step=0.5)
    stock_reversal = st.number_input("Stock P&F reversal", value=3, step=1)
    bpi_box_size = st.number_input("BPI box size", value=2.0, step=1.0)
    bpi_reversal = st.number_input("BPI reversal", value=6, step=1)

    run_single = st.button("Generate Single BPI Chart", use_container_width=True)
    run_dashboard = st.button("Generate Sector Dashboard", use_container_width=True)

if run_single:
    with st.spinner("Building chart..."):
        try:
            bpi = BullishPercentIndex(
                index=selected_index if mode == "Built-in Index" else None,
                tickers=custom_tickers,
                name=custom_name,
                start=str(start_date),
                freq=freq,
                stock_box_pct=stock_box_pct,
                stock_reversal=int(stock_reversal),
                bpi_box_size=bpi_box_size,
                bpi_reversal=int(bpi_reversal),
                figsize=(16, 10),
            )

            c1, c2, c3 = st.columns(3)
            c1.metric("Current BPI", f"{bpi.current_bpi:.1f}%")
            c2.metric("Status", bpi.status)
            c3.metric("Stocks Used", len(bpi.price_data.columns))

            st.dataframe(bpi.summary(), use_container_width=True)

            fig = bpi.plot_figure()
            st.pyplot(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Error building chart: {e}")

if run_dashboard:
    with st.spinner("Building dashboard..."):
        try:
            names = ["NDX", "XLK", "XLF", "XLE"]
            objects = [
                BullishPercentIndex(name_, start=str(start_date), freq=freq, figsize=(10, 6))
                for name_ in names
            ]

            summary_df = pd.concat([obj.summary() for obj in objects], ignore_index=True)
            st.dataframe(summary_df, use_container_width=True)

            fig = plot_bpi_dashboard_figure(objects)
            st.pyplot(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Error building dashboard: {e}")