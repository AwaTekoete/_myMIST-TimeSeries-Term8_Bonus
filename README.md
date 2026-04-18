# Store 44 — Verkaufsprognose (Corporación Favorita)

## Übersicht

Zeitreihen-Forecasting-Projekt für Store 44 (Quito, Ecuador) des
Einzelhandelsunternehmens Corporación Favorita. Die Anwendung
prognostiziert tägliche Verkaufszahlen auf Basis historischer Daten
und unterstützt Geschäftsinhaber bei Bestell- und Personalplanung.

Entwickelt im Rahmen des MIST-Kurses "Time Series Forecasting" (Term 8).

---

## Funktionen der Web-App

- Einzeltag-Prognose mit Vergleich zum Realwert
- N-Tages-Prognose (autoregressive Mehrtagesprognose, 1–30 Tage)
- Interaktiver Chart: historischer Verlauf + Prognose
- Balkendiagramm pro Tag (Wochenende / Werktag farblich markiert)
- KPI-Dashboard: Gesamt, Durchschnitt, Maximum, Minimum, Wochenende-Anteil
- Prognose-Tabelle mit Wochentag
- CSV-Download der Prognose

---

## Projektstruktur

```
_MyMSIT_TimeSeries_Term8/
├── app.py                          # Streamlit Web-App
├── requirements.txt                # Abhängigkeiten
├── README.md                       # Diese Datei
├── mlflow.db                       # MLflow Datenbank
├── data/
│   ├── raw/                        # Originaldaten (unverändert)
│   │   ├── timeseries.csv
│   │   ├── oil.csv
│   │   ├── holidays.csv
│   │   └── stores.csv
│   ├── processed/                  # Bereinigte Daten
│   │   ├── timeseries_cleaned.csv
│   │   └── oil_cleaned.csv
│   └── plots/                      # Visualisierungen (16 Plots)
├── models/
│   └── champion_model.pkl          # Champion Model (RF HyperOpt)
├── mlartifacts/                    # MLflow Artefakte (automatisch)
├── mlruns/                         # MLflow Runs (automatisch)
└── notebooks/
    ├── week1_eda.ipynb             # Explorative Datenanalyse
    ├── week2_statistical_models.ipynb        # ARIMA, SARIMA, ETS
    ├── week2_feature_engineering_models.ipynb # LR, RF, XGBoost
    └── app_test.ipynb              # App-Logik Test
```
---

## Modell-Übersicht

| Modell | MAE | RMSE | MAPE |
|---|---|---|---|
| ARIMA(1,0,1) | 144.06 | 188.78 | 32.65% |
| SARIMA(1,0,1)(1,0,1,7) | 99.06 | 150.28 | 21.15% |
| ETS (Holt-Winters) | 98.43 | 150.41 | 20.69% |
| Linear Regression | 103.14 | 152.69 | 22.62% |
| Random Forest (Standard) | 104.52 | 155.50 | 22.15% |
| XGBoost (Standard) | 110.34 | 164.45 | 23.67% |
| XGBoost (HyperOpt) | 99.73 | 150.88 | 21.19% |
| **Random Forest (HyperOpt)** | **97.95** | **148.75** | **21.15%** |

**Champion Model:** Random Forest (HyperOpt)
- n_estimators: 200, max_depth: 10
- min_samples_leaf: 5, max_features: log2

---

## Voraussetzungen

- Python 3.14
- Windows / macOS / Linux

Installation:

```bash
python -m pip install -r requirements.txt
```

---

## App starten

```bash
streamlit run app.py
```

Browser öffnet automatisch unter: http://localhost:8501

---

## Wie die Prognose funktioniert

1. **Feature Engineering:** Aus der Verkaufszeitreihe werden 15 Features
   berechnet (Kalender, Lags, Rolling Statistics, Ölpreis, Feiertage)
2. **Einzeltag:** Champion Model gibt direkt eine Vorhersage zurück
3. **N-Tage (autoregressive Schleife):**
   - Tag 1 wird vorhergesagt
   - Vorhersage wird als Eingabe für Tag 2 verwendet
   - Schleife wiederholt sich für N Tage
4. **Ergebnis:** Prognose in Einheiten/Tag

---

## Datenquellen

- **timeseries.csv:** Tägliche Verkaufsdaten Store 44 (2013–2014)
- **oil.csv:** WTI Ölpreise (Quelle: Kaggle Corporación Favorita)
- **holidays.csv:** Nationale und lokale Feiertage Ecuador / Quito
- **stores.csv:** Filialdaten (Store 44: Quito, Pichincha)

Original-Datensatz:
[Kaggle — Corporación Favorita Grocery Sales Forecasting](https://www.kaggle.com/c/favorita-grocery-sales-forecasting)

---

## Fehlerbehebung

| Problem | Lösung |
|---|---|
| `Model nicht gefunden` | `models/champion_model.pkl` muss vorhanden sein |
| `Datum nicht im Datensatz` | Datum zwischen 16.01.2013 und 31.03.2014 wählen |
| Port 8501 belegt | `streamlit run app.py --server.port 8502` |
| MLflow UI nicht erreichbar | `python -m mlflow ui --backend-store-uri sqlite:///mlflow.db` |

---

## Technologie-Stack

| Komponente | Technologie |
|---|---|
| Sprache | Python 3.14 |
| Web-App | Streamlit 1.56 |
| Champion Model | scikit-learn RandomForestRegressor |
| Hyperparameter-Tuning | HyperOpt 0.2.7 (TPE) |
| Experiment Tracking | MLflow 3.11.1 |
| Datenverarbeitung | pandas, numpy |
| Visualisierung | matplotlib, seaborn |

---

## Autor

Awa Tekoete - MIST Student — Term 8
Kurs: Time Series Forecasting
Masterschool MIST