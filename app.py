import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import io
import base64
from datetime import datetime, date
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model as keras_load_model

app = Flask(__name__, template_folder='templates')

# ─── Load Model & Config ─────────────────────────────────────────
MODEL_DIR = 'model'
DATA_PATH = 'data/idn_air_quality_daily.csv'

try:
    with open(os.path.join(MODEL_DIR, 'config.json')) as f:
        CFG = json.load(f)
    MODEL = keras_load_model(os.path.join(MODEL_DIR, 'lstm_pm25_best.keras'))
    SCALER_FEAT   = joblib.load(os.path.join(MODEL_DIR, 'scaler_features.pkl'))
    SCALER_TARGET = joblib.load(os.path.join(MODEL_DIR, 'scaler_target.pkl'))
    MODEL_LOADED  = True
    print("[OK] Model berhasil dimuat.")
except Exception as e:
    print(f"[WARN] Gagal memuat model: {e}")
    MODEL_LOADED = False
    CFG = {}

RAW_COLS  = ['pm25_nasional', 'pm25_max', 'pm10', 'aqi', 'pct_melebihi_who']
TIMESTEPS = 30


# ─── Helper Functions ────────────────────────────────────────────

def buat_fitur(raw: pd.DataFrame) -> pd.DataFrame:
    """Log-transform + encoding musiman (doy_sin, doy_cos)."""
    f = raw.copy()
    for c in ['pm25_nasional', 'pm25_max', 'pm10', 'aqi']:
        f[c] = np.log1p(f[c])
    doy = f.index.day_of_year
    f['doy_sin'] = np.sin(2 * np.pi * doy / 365)
    f['doy_cos'] = np.cos(2 * np.pi * doy / 365)
    return f[CFG.get('features', RAW_COLS + ['doy_sin', 'doy_cos'])]


def kategori_pm25(v: float) -> dict:
    if v <= 9.0:
        return {'label': 'Baik',              'color': '#16a34a',
                'bg': 'rgba(22,163,74,0.15)',  'border': 'rgba(22,163,74,0.5)',
                'icon': 'fa-circle-check',     'level': 1}
    if v <= 35.4:
        return {'label': 'Sedang',            'color': '#d97706',
                'bg': 'rgba(217,119,6,0.15)', 'border': 'rgba(217,119,6,0.5)',
                'icon': 'fa-circle-half-stroke','level': 2}
    if v <= 55.4:
        return {'label': 'Tdk Sehat (Sensitif)', 'color': '#ea580c',
                'bg': 'rgba(234,88,12,0.15)',  'border': 'rgba(234,88,12,0.5)',
                'icon': 'fa-triangle-exclamation','level': 3}
    if v <= 125.4:
        return {'label': 'Tidak Sehat',       'color': '#dc2626',
                'bg': 'rgba(220,38,38,0.15)', 'border': 'rgba(220,38,38,0.5)',
                'icon': 'fa-biohazard',        'level': 4}
    if v <= 225.4:
        return {'label': 'Sgt Tidak Sehat',   'color': '#9333ea',
                'bg': 'rgba(147,51,234,0.15)','border': 'rgba(147,51,234,0.5)',
                'icon': 'fa-skull-crossbones', 'level': 5}
    return     {'label': 'Berbahaya',          'color': '#7f1d1d',
                'bg': 'rgba(127,29,29,0.15)', 'border': 'rgba(127,29,29,0.5)',
                'icon': 'fa-radiation',        'level': 6}


def load_daily_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, parse_dates=['date'])
    daily = df.groupby('date').agg(
        pm25_nasional=('pm2_5_mean',      'mean'),
        pm25_max=     ('pm2_5_max',       'mean'),
        pm10=         ('pm10_mean',       'mean'),
        aqi=          ('us_aqi_mean',     'mean'),
        pct_melebihi_who=('exceeds_who_pm25','mean'),
    ).sort_index()
    daily = daily.asfreq('D').interpolate(method='linear').ffill().bfill()
    daily['pct_melebihi_who'] *= 100
    return daily


def _satu_langkah(window: pd.DataFrame) -> float:
    """Prediksi satu hari ke depan dari window 30 hari terakhir."""
    w_raw  = window[RAW_COLS].tail(TIMESTEPS).copy()
    w_feat = buat_fitur(w_raw)
    x      = SCALER_FEAT.transform(w_feat).reshape(1, TIMESTEPS, len(CFG['features']))
    delta_s = float(MODEL.predict(x, verbose=0)[0, 0])
    delta   = float(SCALER_TARGET.inverse_transform([[delta_s]])[0, 0])
    pm25_prev = float(window['pm25_nasional'].iloc[-1])
    return max(0.0, float(np.expm1(np.log1p(pm25_prev) + delta)))


def bridge_to_today(daily: pd.DataFrame) -> pd.DataFrame:
    """
    Jika data terakhir < hari ini, isi gap dengan prediksi autoregresif
    sehingga window selalu up-to-date per hari ini.
    """
    today      = pd.Timestamp(date.today())
    last_date  = daily.index[-1]
    window     = daily[RAW_COLS].copy()

    cur = last_date
    while cur < today:
        nxt  = cur + pd.Timedelta(days=1)
        val  = _satu_langkah(window)
        new  = pd.DataFrame({
            'pm25_nasional':   [val],
            'pm25_max':        [val * 1.1],
            'pm10':            [window['pm10'].iloc[-1]],
            'aqi':             [window['aqi'].iloc[-1]],
            'pct_melebihi_who':[window['pct_melebihi_who'].iloc[-1]],
        }, index=[nxt])
        window = pd.concat([window, new])
        cur    = nxt

    return window


def prediksi_n_hari(window: pd.DataFrame, n_days: int = 7) -> tuple:
    """
    Prediksi N hari ke depan dari window yang sudah di-bridge ke hari ini.
    Mengembalikan (tanggal_prediksi, nilai_prediksi).
    """
    today = pd.Timestamp(date.today())
    results_dates  = []
    results_values = []

    for i in range(n_days):
        val  = _satu_langkah(window)
        nxt  = today + pd.Timedelta(days=i + 1)
        results_dates.append(nxt)
        results_values.append(val)

        new = pd.DataFrame({
            'pm25_nasional':   [val],
            'pm25_max':        [val * 1.1],
            'pm10':            [window['pm10'].iloc[-1]],
            'aqi':             [window['aqi'].iloc[-1]],
            'pct_melebihi_who':[window['pct_melebihi_who'].iloc[-1]],
        }, index=[nxt])
        window = pd.concat([window, new])

    return results_dates, results_values


# ─── Chart Helpers ───────────────────────────────────────────────

# Warna chart: tema putih-biru bersih
BG_CHART  = '#f8faff'   # background chart (putih kebiruan)
BG_FIG    = '#ffffff'   # background figure
GRID_CLR  = '#e2e8f0'   # grid lines abu muda
TICK_CLR  = '#64748b'   # tick labels
LABEL_CLR = '#475569'   # axis labels
TEXT_CLR  = '#1e293b'   # annotation text
BLUE_LINE = '#2563eb'   # warna historis (biru)
ORNG_LINE = '#f97316'   # warna prediksi (oranye)

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.spines.top':   False,
    'axes.spines.right': False,
})


def fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=140, bbox_inches='tight',
                facecolor=BG_FIG, edgecolor='none')
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return b64


def _style_ax(ax):
    """Gaya putih-biru bersih untuk semua axis."""
    ax.set_facecolor(BG_CHART)
    ax.tick_params(colors=TICK_CLR, labelsize=9)
    ax.yaxis.label.set_color(LABEL_CLR)
    ax.xaxis.label.set_color(LABEL_CLR)
    for spine in ['bottom', 'left']:
        ax.spines[spine].set_edgecolor('#cbd5e1')
        ax.spines[spine].set_linewidth(1)
    ax.grid(axis='y', color=GRID_CLR, alpha=1.0,
            linestyle='--', linewidth=0.8)
    ax.grid(axis='x', visible=False)


def buat_chart_prediksi(daily_hist: pd.DataFrame,
                        pred_dates: list, pred_vals: list) -> str:
    """Kurva historis 60 hari + prediksi N hari ke depan."""
    hist = daily_hist['pm25_nasional'].tail(60)

    fig, ax = plt.subplots(figsize=(9.5, 3.8))
    fig.patch.set_color(BG_FIG)

    # Historis — biru
    ax.plot(hist.index, hist.values,
            color=BLUE_LINE, lw=2.2,
            label='Historis (60 hari)', zorder=3)
    ax.fill_between(hist.index, hist.values,
                    alpha=0.08, color=BLUE_LINE, zorder=2)

    # Prediksi — oranye dashed
    conn_dates = [hist.index[-1]] + pred_dates
    conn_vals  = [hist.values[-1]] + pred_vals
    ax.plot(conn_dates, conn_vals,
            color=ORNG_LINE, lw=2.2, linestyle='--',
            label='Prediksi', marker='o', markersize=5.5,
            markerfacecolor=ORNG_LINE, markeredgecolor='white',
            markeredgewidth=1.5, zorder=4)
    ax.fill_between(conn_dates, conn_vals,
                    alpha=0.08, color=ORNG_LINE, zorder=2)

    # Ambang WHO — merah dashed
    ax.axhline(15, color='#dc2626', lw=1.3, ls=(0,(6,4)),
               alpha=0.75, label='Ambang WHO 15 µg/m³', zorder=3)

    # Shading zona prediksi
    ax.axvspan(pred_dates[0], pred_dates[-1],
               color=ORNG_LINE, alpha=0.05, zorder=1)

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    plt.setp(ax.xaxis.get_majorticklabels(),
             rotation=22, ha='right', fontsize=9, color=TICK_CLR)
    ax.set_ylabel('PM2.5 (µg/m³)', fontsize=9.5, labelpad=8)
    _style_ax(ax)

    ax.legend(fontsize=9, frameon=True, framealpha=0.92,
              facecolor='white', edgecolor='#e2e8f0',
              labelcolor=TEXT_CLR, loc='upper left',
              borderpad=0.6)
    plt.tight_layout(pad=1.0)
    return fig_to_base64(fig)


def buat_chart_bar(pred_dates: list, pred_vals: list) -> str:
    """Bar chart nilai PM2.5 per hari prediksi."""
    fig, ax = plt.subplots(figsize=(9.5, 3.2))
    fig.patch.set_color(BG_FIG)

    labels     = [d.strftime('%a\n%d %b') for d in pred_dates]
    bar_colors = [kategori_pm25(v)['color'] for v in pred_vals]

    bars = ax.bar(labels, pred_vals, color=bar_colors, alpha=0.85,
                  edgecolor='white', linewidth=1.5,
                  zorder=3, width=0.58)

    for bar, val in zip(bars, pred_vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.22,
                f'{val:.1f}', ha='center', va='bottom',
                fontsize=9, color=TEXT_CLR, fontweight='700')

    ax.axhline(15, color='#dc2626', lw=1.3, ls=(0,(6,4)),
               alpha=0.75, label='Ambang WHO 15 µg/m³', zorder=3)
    ax.set_ylabel('PM2.5 (µg/m³)', fontsize=9.5, labelpad=8)
    _style_ax(ax)
    ax.legend(fontsize=9, frameon=True, framealpha=0.92,
              facecolor='white', edgecolor='#e2e8f0',
              labelcolor=TEXT_CLR)
    plt.tight_layout(pad=1.0)
    return fig_to_base64(fig)


def buat_chart_gauge(pm25_val: float, kategori: dict) -> str:
    """Gauge setengah lingkaran — tema putih bersih."""
    fig, ax = plt.subplots(figsize=(4.0, 2.2),
                           subplot_kw={'projection': 'polar'})
    fig.patch.set_color(BG_FIG)
    ax.set_facecolor(BG_FIG)

    breakpoints = [0, 9, 35.4, 55.4, 125.4, 225.4, 300]
    colors_seg  = ['#16a34a', '#d97706', '#ea580c',
                   '#dc2626', '#9333ea', '#7f1d1d']
    max_val = 300.0

    # Track ring abu muda (background segmen)
    for i in range(len(breakpoints) - 1):
        s = np.pi - (breakpoints[i]     / max_val) * np.pi
        e = np.pi - (breakpoints[i + 1] / max_val) * np.pi
        theta = np.linspace(s, e, 60)
        ax.fill_between(theta, 0.58, 1.0, color='#f1f5f9', alpha=1.0)
        ax.fill_between(theta, 0.60, 0.98, color=colors_seg[i], alpha=0.80)

    # Jarum
    needle = np.pi - (min(pm25_val, max_val) / max_val) * np.pi
    ax.annotate('', xy=(needle, 0.90), xytext=(needle, 0.14),
                arrowprops=dict(arrowstyle='->', color=kategori['color'],
                                lw=2.8, mutation_scale=14))
    ax.plot(needle, 0.14, 'o', color=kategori['color'],
            markersize=5.5, zorder=5)

    ax.set_xlim([0, np.pi]); ax.set_ylim([0, 1.15])
    ax.set_theta_zero_location('W')
    ax.set_theta_direction(-1)
    ax.set_thetamin(0); ax.set_thetamax(180)
    ax.axis('off')

    ax.text(np.pi / 2, -0.20, f'{pm25_val:.1f}',
            ha='center', va='center', fontsize=21,
            fontweight='900', color=kategori['color'],
            transform=ax.transData)
    ax.text(np.pi / 2, -0.52, 'µg/m³',
            ha='center', va='center', fontsize=9,
            color=LABEL_CLR, transform=ax.transData)

    plt.tight_layout(pad=0)
    return fig_to_base64(fig)


# ─── Routes ──────────────────────────────────────────────────────

@app.route('/')
def home():
    if not MODEL_LOADED:
        return "<h2>Model belum dimuat. Jalankan notebook terlebih dahulu.</h2>"

    try:
        daily      = load_daily_data()
        today_str  = datetime.now().strftime('%d %b %Y')
        today_ts   = pd.Timestamp(date.today())

        last_data_date = daily.index[-1]
        data_date_str  = last_data_date.strftime('%d %b %Y')

        # PM2.5 "hari ini" — bisa dari data asli atau estimasi bridge
        if last_data_date >= today_ts:
            pm25_today = round(float(daily['pm25_nasional'].loc[today_ts]), 2)
        else:
            # Bridge ke hari ini dan ambil nilai terakhir sebagai estimasi
            window_now = bridge_to_today(daily)
            pm25_today = round(float(window_now['pm25_nasional'].iloc[-1]), 2)

        kat_today = kategori_pm25(pm25_today)

        stats = {
            'today_str':    today_str,
            'data_date_str':data_date_str,
            'pm25_today':   pm25_today,
            'kat_today':    kat_today,
            'r2':           round(CFG.get('metrics_test', {}).get('R2', 0) * 100, 1),
            'mape':         round(CFG.get('metrics_test', {}).get('MAPE (%)', 0), 2),
            'rmse':         round(CFG.get('metrics_test', {}).get('RMSE', 0), 2),
        }
    except Exception as e:
        import traceback; traceback.print_exc()
        stats = {'today_str': datetime.now().strftime('%d %b %Y'),
                 'pm25_today': '—', 'kat_today': {'label':'—','color':'#fff'}}

    return render_template('index.html', stats=stats)


@app.route('/predict', methods=['POST'])
def predict():
    if not MODEL_LOADED:
        return "<h3>Model tidak tersedia</h3><a href='/'>Kembali</a>"

    try:
        n_days = int(request.form.get('n_days', 7))
        n_days = max(1, min(n_days, 14))

        daily  = load_daily_data()
        window = bridge_to_today(daily)   # <-- selalu bridge ke hari ini dulu

        pred_dates, pred_vals = prediksi_n_hari(window, n_days)

        # Charts
        chart_line  = buat_chart_prediksi(daily, pred_dates, pred_vals)
        chart_bar   = buat_chart_bar(pred_dates, pred_vals)

        # Tabel
        tabel = []
        for d, v in zip(pred_dates, pred_vals):
            kat = kategori_pm25(v)
            tabel.append({
                'tanggal':  d.strftime('%a, %d %b %Y'),
                'pm25':     round(v, 2),
                'kategori': kat['label'],
                'color':    kat['color'],
                'bg':       kat['bg'],
                'border':   kat['border'],
                'icon':     kat['icon'],
            })

        today_str     = datetime.now().strftime('%d %b %Y')
        avg_pred      = round(float(np.mean(pred_vals)), 2)
        max_pred      = round(float(np.max(pred_vals)),  2)
        min_pred      = round(float(np.min(pred_vals)),  2)
        kat_avg       = kategori_pm25(avg_pred)
        pm25_now      = round(float(window['pm25_nasional'].iloc[-1]), 2)

        return render_template(
            'result.html',
            n_days     = n_days,
            today_str  = today_str,
            tabel      = tabel,
            avg_pred   = avg_pred,
            max_pred   = max_pred,
            min_pred   = min_pred,
            kat_avg    = kat_avg,
            pm25_now   = pm25_now,
            chart_line = chart_line,
            chart_bar  = chart_bar,
            metrics    = {
                'R2':   round(CFG.get('metrics_test',{}).get('R2',   0) * 100, 1),
                'MAPE': round(CFG.get('metrics_test',{}).get('MAPE (%)', 0), 2),
                'RMSE': round(CFG.get('metrics_test',{}).get('RMSE', 0), 2),
            }
        )

    except Exception as e:
        import traceback; traceback.print_exc()
        return f'<h3>Error: {e}</h3><a href="/">Kembali</a>'


if __name__ == '__main__':
    app.run(debug=True, port=5000)
