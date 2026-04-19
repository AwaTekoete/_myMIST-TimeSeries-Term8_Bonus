# ============================================================
# app.py — Streamlit Forecasting App
# Store 44, Quito, Ecuador — Corporación Favorita
# Champion Model: Random Forest (HyperOpt)
# ============================================================

import os
import warnings
import joblib
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import streamlit as st

warnings.filterwarnings('ignore')

# Farb-Fix Windows 125% Skalierung
matplotlib.rcParams['text.color'] = 'black'
matplotlib.rcParams['axes.labelcolor'] = 'black'
matplotlib.rcParams['xtick.color'] = 'black'
matplotlib.rcParams['ytick.color'] = 'black'
matplotlib.rcParams['figure.facecolor'] = 'white'
matplotlib.rcParams['axes.facecolor'] = 'white'
matplotlib.rcParams['savefig.facecolor'] = 'white'
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

# ============================================================
# KONFIGURATION
# ============================================================
APP_TITLE = "Store 44 — Verkaufsprognose"
MODEL_PATH = "models/champion_model.pkl"
DATA_PROC = "data/processed/"
DATA_RAW = "data/raw/"
HISTORY_DAYS = 60

FEATURE_COLS = [
    'day_of_week', 'day_of_month', 'month', 'week_of_year',
    'is_weekend', 'lag_1', 'lag_7', 'lag_14',
    'rolling_mean_7', 'rolling_mean_14', 'rolling_std_7',
    'rolling_max_7', 'rolling_min_7', 'oil_price', 'is_holiday'
]


# ============================================================
# DATEN & MODELL LADEN (gecacht für Performance)
# ============================================================

@st.cache_resource
def load_model() -> object:
    """Lädt das Champion Model (einmalig, gecacht)."""
    assert os.path.exists(MODEL_PATH), f"Model nicht gefunden: {MODEL_PATH}"
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_data() -> tuple:
    """Lädt alle benötigten Datensätze (einmalig, gecacht)."""
    df = pd.read_csv(
        DATA_PROC + "timeseries_cleaned.csv",
        index_col='date', parse_dates=True
    )
    df_oil = pd.read_csv(
        DATA_PROC + "oil_cleaned.csv",
        index_col='date', parse_dates=True
    )
    df_holiday = pd.read_csv(
        DATA_RAW + "holidays.csv",
        parse_dates=['date']
    )
    return df, df_oil, df_holiday


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def build_features(df: pd.DataFrame,
                   df_oil: pd.DataFrame,
                   df_holiday: pd.DataFrame) -> pd.DataFrame:
    """Erstellt vollständigen Feature-Datensatz aus Rohdaten."""
    df_feat = df.copy()

    # Kalender-Features
    df_feat['day_of_week'] = df_feat.index.dayofweek
    df_feat['day_of_month'] = df_feat.index.day
    df_feat['month'] = df_feat.index.month
    df_feat['week_of_year'] = df_feat.index.isocalendar().week.astype(int)
    df_feat['is_weekend'] = (df_feat.index.dayofweek >= 5).astype(int)

    # Lag-Features
    df_feat['lag_1'] = df_feat['unit_sales'].shift(1)
    df_feat['lag_7'] = df_feat['unit_sales'].shift(7)
    df_feat['lag_14'] = df_feat['unit_sales'].shift(14)

    # Rolling-Features (shift(1) verhindert Data Leakage)
    df_feat['rolling_mean_7'] = df_feat['unit_sales'].shift(1).rolling(7).mean()
    df_feat['rolling_mean_14'] = df_feat['unit_sales'].shift(1).rolling(14).mean()
    df_feat['rolling_std_7'] = df_feat['unit_sales'].shift(1).rolling(7).std()
    df_feat['rolling_max_7'] = df_feat['unit_sales'].shift(1).rolling(7).max()
    df_feat['rolling_min_7'] = df_feat['unit_sales'].shift(1).rolling(7).min()

    # Exogene Features
    df_feat['oil_price'] = df_oil['dcoilwtico']

    # Feiertage: National + Quito (Store 44)
    national = df_holiday[
        df_holiday['locale'] == 'National'
        ]['date'].dt.date.tolist()
    local_quito = df_holiday[
        (df_holiday['locale'] == 'Local') &
        (df_holiday['locale_name'] == 'Quito')
        ]['date'].dt.date.tolist()
    alle_feiertage = list(set(national + local_quito))

    df_feat['is_holiday'] = df_feat.index.date
    df_feat['is_holiday'] = df_feat['is_holiday'].apply(
        lambda x: 1 if x in alle_feiertage else 0
    )

    return df_feat.dropna()


# ============================================================
# PROGNOSE-FUNKTIONEN
# ============================================================

def predict_single_day(model, df_feat: pd.DataFrame,
                       target_date: pd.Timestamp) -> float:
    """Prognose für einen einzelnen Tag."""
    X = df_feat.loc[[target_date], FEATURE_COLS]
    return float(model.predict(X)[0])


def predict_n_days(model, df_feat: pd.DataFrame,
                   start_date: pd.Timestamp,
                   n_days: int,
                   df_oil: pd.DataFrame,
                   df_holiday: pd.DataFrame) -> pd.DataFrame:
    """Autoregressive N-Tages-Prognose ab start_date."""
    df_work = df_feat.copy()

    # Feiertage vorbereiten
    national = df_holiday[
        df_holiday['locale'] == 'National'
        ]['date'].dt.date.tolist()
    local_quito = df_holiday[
        (df_holiday['locale'] == 'Local') &
        (df_holiday['locale_name'] == 'Quito')
        ]['date'].dt.date.tolist()
    alle_feiertage = list(set(national + local_quito))

    predictions = []
    current_date = start_date

    for _ in range(n_days):
        row = {
            'day_of_week': current_date.dayofweek,
            'day_of_month': current_date.day,
            'month': current_date.month,
            'week_of_year': current_date.isocalendar()[1],
            'is_weekend': int(current_date.dayofweek >= 5),
            'lag_1': df_work['unit_sales'].get(
                current_date - pd.Timedelta(days=1), np.nan),
            'lag_7': df_work['unit_sales'].get(
                current_date - pd.Timedelta(days=7), np.nan),
            'lag_14': df_work['unit_sales'].get(
                current_date - pd.Timedelta(days=14), np.nan),
        }

        history_7 = df_work['unit_sales'][df_work.index < current_date].tail(7)
        history_14 = df_work['unit_sales'][df_work.index < current_date].tail(14)

        row['rolling_mean_7'] = history_7.mean()
        row['rolling_mean_14'] = history_14.mean()
        row['rolling_std_7'] = history_7.std()
        row['rolling_max_7'] = history_7.max()
        row['rolling_min_7'] = history_7.min()

        oil_val = df_oil['dcoilwtico'].get(current_date, np.nan)
        if np.isnan(oil_val):
            oil_val = df_oil['dcoilwtico'].dropna().iloc[-1]
        row['oil_price'] = oil_val
        row['is_holiday'] = int(current_date.date() in alle_feiertage)

        pred = max(0.0, float(model.predict(
            pd.DataFrame([row])[FEATURE_COLS]
        )[0]))

        predictions.append({'date': current_date, 'prediction': pred})
        df_work = pd.concat([
            df_work,
            pd.DataFrame({**row, 'unit_sales': pred}, index=[current_date])
        ])
        current_date += pd.Timedelta(days=1)

    return pd.DataFrame(predictions).set_index('date')


# ============================================================
# VISUALISIERUNG
# ============================================================

def plot_forecast(df_feat: pd.DataFrame,
                  forecast: pd.DataFrame,
                  history_days: int = HISTORY_DAYS) -> plt.Figure:
    """Erstellt kombinierten Verlauf + Prognose Plot."""
    history_end = forecast.index.min() - pd.Timedelta(days=1)
    history_start = history_end - pd.Timedelta(days=history_days)
    history = df_feat.loc[history_start:history_end, 'unit_sales']

    fig, axes = plt.subplots(2, 1, figsize=(12, 9))
    fig.suptitle(APP_TITLE, fontsize=15, fontweight='bold')

    # Plot 1: Verlauf + Prognose
    ax1 = axes[0]
    ax1.plot(history.index, history.values,
             color='steelblue', linewidth=1.2,
             label=f'Historisch (letzte {history_days} Tage)')
    ax1.plot(forecast.index, forecast['prediction'].values,
             color='orange', linewidth=2, linestyle='--',
             marker='o', markersize=5, label='Prognose (Champion Model)')
    ax1.axvline(forecast.index.min() - pd.Timedelta(hours=12),
                color='red', linestyle=':', linewidth=1.5,
                label='Prognosestart')
    ax1.set_title('Historischer Verlauf & Prognose', fontsize=12)
    ax1.set_ylabel('Verkaufte Einheiten', fontsize=11)
    ax1.set_xlabel('Datum', fontsize=11)
    ax1.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m.%Y'))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    ax1.legend(loc='upper left', fontsize=10, framealpha=0.9)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Balkendiagramm pro Tag
    ax2 = axes[1]
    colors = ['coral' if d.dayofweek >= 5 else 'steelblue'
              for d in forecast.index]
    bars = ax2.bar(range(len(forecast)),
                   forecast['prediction'].values,
                   color=colors, edgecolor='white', width=0.6)
    for bar, val in zip(bars, forecast['prediction'].values):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 5,
                 f'{val:.0f}',
                 ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax2.set_xticks(range(len(forecast)))
    ax2.set_xticklabels(
        [d.strftime('%a\n%d.%m') for d in forecast.index], fontsize=9
    )
    ax2.set_title('Prognose pro Tag (rot = Wochenende, blau = Werktag)',
                  fontsize=12)
    ax2.set_ylabel('Vorhergesagte Einheiten', fontsize=11)
    ax2.set_ylim(0, forecast['prediction'].max() * 1.2)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout(pad=3.0)
    return fig


def forecast_to_csv(forecast: pd.DataFrame) -> str:
    """Konvertiert Prognose in CSV-String für Download."""
    export = forecast.copy()
    export.index.name = 'Datum'
    export.columns = ['Prognose (Einheiten)']
    export['Wochentag'] = export.index.strftime('%A')
    export['Prognose (Einheiten)'] = export['Prognose (Einheiten)'].round(0).astype(int)
    return export.to_csv(encoding='utf-8-sig')


def forecast_summary(forecast: pd.DataFrame) -> dict:
    """Berechnet KPI-Kennzahlen der Prognose."""
    return {
        'Zeitraum': f"{forecast.index.min().strftime('%d.%m.%Y')} — {forecast.index.max().strftime('%d.%m.%Y')}",
        'Anzahl Tage': len(forecast),
        'Gesamt (Einheiten)': int(forecast['prediction'].sum().round(0)),
        'Durchschnitt/Tag': round(float(forecast['prediction'].mean()), 1),
        'Maximum': int(forecast['prediction'].max().round(0)),
        'Minimum': int(forecast['prediction'].min().round(0)),
        'Wochenende-Anteil': f"{(forecast['prediction'][forecast.index.dayofweek >= 5].sum() / forecast['prediction'].sum() * 100):.1f}%",
    }


# ============================================================
# STREAMLIT UI
# ============================================================

def main():
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="🛒",
        layout="wide"
    )

    # Header
    st.title("🛒 Store 44 — Verkaufsprognose")
    st.markdown("""
    **Corporación Favorita | Quito, Ecuador**

    Tägliche Verkaufsprognosen auf Basis des Champion Models 
    (Random Forest, HyperOpt-optimiert).
    Unterstützt Geschäftsinhaber bei Bestell- und Personalplanung.
    """)
    st.divider()

    # Daten & Modell laden
    with st.spinner("Modell und Daten werden geladen..."):
        model = load_model()
        df, df_oil, df_holiday = load_data()
        df_feat = build_features(df, df_oil, df_holiday)

    # Sidebar — Steuerung
    st.sidebar.header("Prognose-Einstellungen")
    st.sidebar.markdown("**Modell:** Random Forest (HyperOpt)")
    st.sidebar.markdown("**Store:** 44 | Quito, Pichincha")
    st.sidebar.divider()

    prognose_modus = st.sidebar.radio(
        "Prognosemodus",
        ["Einzelner Tag", "Nächste N Tage"]
    )

    if prognose_modus == "Einzelner Tag":
        zieldatum = st.sidebar.date_input(
            "Prognosedatum",
            value=pd.Timestamp('2014-03-31').date(),
            min_value=df_feat.index.min().date(),
            max_value=df_feat.index.max().date()
        )
        n_tage = 1
    else:
        startdatum = st.sidebar.date_input(
            "Startdatum der Prognose",
            value=pd.Timestamp('2014-04-01').date(),
            min_value=df_feat.index.min().date()
        )
        n_tage = st.sidebar.slider(
            "Anzahl Prognosetage", min_value=1, max_value=30, value=7
        )
        history_tage = st.sidebar.slider(
            "Historische Tage im Plot", min_value=14, max_value=90, value=60
        )

    st.sidebar.divider()
    prognose_button = st.sidebar.button("Prognose erstellen", type="primary")

    # Prognose ausführen
    if prognose_button:
        with st.spinner("Prognose wird berechnet..."):

            if prognose_modus == "Einzelner Tag":
                target = pd.Timestamp(zieldatum)
                if target not in df_feat.index:
                    st.error(f"Datum {zieldatum} nicht im Datensatz.")
                    st.stop()

                pred = predict_single_day(model, df_feat, target)
                real_val = df_feat.loc[target, 'unit_sales']
                abweich = abs(pred - real_val)

                # KPI-Anzeige
                col1, col2, col3 = st.columns(3)
                col1.metric("Prognose", f"{pred:.0f} Einheiten")
                col2.metric("Realwert", f"{real_val:.0f} Einheiten")
                col3.metric("Abweichung", f"{abweich:.0f} Einheiten",
                            delta=f"{(abweich / real_val * 100):.1f}%",
                            delta_color="inverse")

                # Einzeltag als DataFrame für Plot
                forecast = pd.DataFrame(
                    {'prediction': [pred]}, index=[target]
                )
                fig = plot_forecast(df_feat, forecast, history_days=60)
                st.pyplot(fig)

            else:
                start = pd.Timestamp(startdatum)
                forecast = predict_n_days(
                    model, df_feat, start, n_tage, df_oil, df_holiday
                )

                # KPI-Kennzahlen
                summary = forecast_summary(forecast)
                cols = st.columns(len(summary))
                for col, (key, val) in zip(cols, summary.items()):
                    col.metric(key, val)

                st.divider()

                # Plot
                fig = plot_forecast(df_feat, forecast,
                                    history_days=history_tage)
                st.pyplot(fig)

                st.divider()

                # Tabelle
                st.subheader("Prognose-Tabelle")
                display_df = pd.DataFrame({
                    'Datum': forecast.index.strftime('%d.%m.%Y'),
                    'Wochentag': forecast.index.strftime('%A'),
                    'Prognose': forecast['prediction'].fillna(0).round(0).astype(int).values
                })
                display_df = display_df.set_index('Datum')
                st.dataframe(display_df, use_container_width=True)

                # CSV Download
                st.download_button(
                    label="Prognose als CSV herunterladen",
                    data=forecast_to_csv(forecast),
                    file_name=f"prognose_{n_tage}tage_ab_{startdatum}.csv",
                    mime="text/csv"
                )

                # MIDI Download (nur für N-Tage Modus)
                if n_tage >= 7:
                    midi_bytes = generate_midi(forecast)
                    st.download_button(
                        label="🎵 Prognose als MIDI herunterladen",
                        data=midi_bytes,
                        file_name=f"prognose_{n_tage}tage_ab_{startdatum}.mid",
                        mime="audio/midi"
                    )

    else:
        # Willkommens-Ansicht
        st.info("Prognose-Einstellungen in der Sidebar auswaehlen und 'Prognose erstellen' klicken.")

        st.subheader("Verfuegbare Daten")
        col1, col2, col3 = st.columns(3)
        col1.metric("Datenpunkte", f"{len(df_feat)} Tage")
        col2.metric("Zeitraum",
                    f"{df_feat.index.min().strftime('%d.%m.%Y')} — {df_feat.index.max().strftime('%d.%m.%Y')}")
        col3.metric("Champion Model", "Random Forest (HyperOpt)")


# ============================================================
# MIDI SONIFICATION FUNKTION
# ============================================================
def generate_midi(forecast: pd.DataFrame) -> bytes:
    """
    Generiert MIDI Datei aus Prognose-DataFrame.
    Modulation: Oktave + Notendauer nach Verkaufswert.
    """
    from midiutil import MIDIFile
    import io

    WOCHENTAG_NOTEN = {
        0: 57, 1: 60, 2: 64, 3: 67,
        4: 69, 5: 72, 6: 76,
    }

    TEMPO   = 120
    TRACK   = 0
    CHANNEL = 0

    q25   = forecast['prediction'].quantile(0.25)
    q75   = forecast['prediction'].quantile(0.75)
    q95   = forecast['prediction'].quantile(0.95)
    min_v = forecast['prediction'].min()
    max_v = forecast['prediction'].max()

    midi = MIDIFile(1)
    midi.addTempo(TRACK, 0, TEMPO)
    beat = 0.0

    for datum, row in forecast.iterrows():
        verkauf   = float(row['prediction'])
        wochentag = datum.dayofweek

        if verkauf == 0:
            beat += 0.5
            continue

        note = WOCHENTAG_NOTEN[wochentag]

        if verkauf > q95:
            note  += 24
            dauer  = 1.5
        elif verkauf > q75:
            note  += 12
            dauer  = 1.25
        elif verkauf < q25:
            note  -= 12
            dauer  = 0.5
        else:
            dauer  = 0.75

        if wochentag >= 5:
            dauer += 0.25

        velocity = max(40, min(120, int(
            40 + (verkauf - min_v) / (max_v - min_v) * 80
        )))

        midi.addNote(TRACK, CHANNEL, note, beat, dauer, velocity)
        beat += dauer

    buffer = io.BytesIO()
    midi.writeFile(buffer)
    return buffer.getvalue()

if __name__ == "__main__":
    main()