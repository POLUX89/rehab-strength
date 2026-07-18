# Changelog

## [Unreleased]

Rama **Classification** en Models (en progreso: Logistic Regression listo;
Non Linear y Bagging & Boosting aún son stubs).

### Added
- Paquete `app/tabs/models/classification/`: el despachador construye el target
  binario (`Score < 80` = clase 1 = "Bad Sleep") y el **split temporal** train/test
  una sola vez, con gráficos de distribución de clases (total, train y test).
- Sub-rama **Logistic Regression** (`logit.py`): tuning con `GridSearchCV` sobre
  `TimeSeriesSplit` (CV que respeta el orden temporal, sin barajar). Pipeline
  `StandardScaler → [SMOTE] → LogisticRegression(solver="saga")`; la regularización
  se busca con `l1_ratio` (L2/L1/elastic-net), acorde a la deprecación de `penalty`
  en scikit-learn 1.8. Scoring **F2** (prioriza recall de la clase minoritaria).
- Panel de métricas train vs test (accuracy/precision/recall/F1/F2 con deltas),
  classification report y matriz de confusión.
- Toggle **SMOTE** opcional: remuestrea solo dentro de cada fold de CV (sin fuga a
  validación/test) y, cuando está activo, fija `class_weight=None` para no corregir
  el desbalance dos veces.
- `@st.cache_data` en el tuning: no re-entrena si no cambian los datos ni el toggle.

### Changed
- Python fijado a **3.13**: `requires-python = ">=3.13,<3.14"`, ruff
  `target-version = "py313"` y `.python-version`. Paridad con Streamlit Community
  Cloud (tope 3.13) y cierra la puerta a que un resolver elija 3.14.
- `models/__init__.py` enruta el tipo "Classification" a la nueva sub-rama.


## [2.6.0] - 2026/07/18

Modularización del monolito. **La app no cambia de comportamiento** (verificado en
cada paso comparando el multiset de llamadas AST contra el original).

### Changed
- `streamlit_app.py` pasó de **3.217 a 243 líneas**: ahora es solo orquestación
  (setup, sidebar, panel de carga y 7 llamadas `*_tab.render(...)`).
- Helpers extraídos a `app/helpers/`: `stats.py`, `transforms.py`, `plots.py`.
- Las 7 tabs extraídas a `app/tabs/`: home, workouts, sleep, recovery, timeseries,
  stats y el paquete `models/`.
- `models/` subdividido: `__init__` despacha por tipo de algoritmo; `unsupervised.py`
  y el paquete `regression/` (despachador + `ols.py`, `linear.py`, `nonlinear.py`,
  `ensemble.py`). El archivo más grande bajó de 3.217 a 1.137 líneas.
- Los módulos de la app viven en `app/` (hermano del entrypoint), no en
  `src/rehab_strength/`: Streamlit Cloud instala requirements pero no el paquete.
- `app_version` de la UI alineado a V2.6.0.

### Added
- 61 tests nuevos sobre helpers y tabs (34 → 95 en total).
- `per-file-ignores` de E501 para `app/tabs/*.py` (UI migrada con textos largos).


## [2.5.1] - 2026/07/17

### Changed
- La key de Google se movió del Escritorio a `~/.config/rehab-strength/` (permisos
  600). El `.env` apunta a la ruta nueva; `.env.example` documenta la convención.
- `run_pipeline.sh` notifica con `osascript` nativo en vez de depender de
  `NotifyGymPipeline.app`, que vivía en el Escritorio. La automatización ya no
  referencia esa carpeta: se puede borrar sin romper nada.

## [2.5.0] - 2026/07/16

### Added
- `scripts/run_pipeline.sh`: corre la ingesta completa, loguea y notifica por
  macOS. Deriva la raíz del repo de su propia ubicación, sin rutas absolutas.
  Reemplaza al `run_pipeline.sh` que vivía suelto en el Escritorio, fuera de git.
- Notificación también cuando la ingesta **falla**. El script viejo solo avisaba
  al terminar bien: si Google fallaba, no había aviso y los datos quedaban viejos
  en silencio.
- README: cómo enganchar el pipeline a sleepwatcher para que corra al despertar
  el Mac, y enlace a la app desplegada.

### Changed
- Fuente de verdad única: los Excel y el pipeline viven en el repo. La carpeta
  `GYM WORKOUT ANALYSIS PROJECT` del Escritorio queda como archivo histórico.
- `~/.wakeup` (sleepwatcher) apunta al script del repo. El del Escritorio quedó
  como `run_pipeline.sh.RETIRED` para que no corra en paralelo.

## [2.4.2] - 2026/07/16

### Fixed
- `make ingest` fallaba con "No hay credenciales de Google" aunque existiera un
  `.env`: el README mandaba crearlo pero nada lo leía. `config.py` ahora carga
  `.env` con python-dotenv. Las variables ya exportadas en el shell siguen
  teniendo prioridad sobre el archivo.

## [2.4.1] - 2026/07/16

### Fixed
- CI en rojo por el escaneo de secretos. `detect-secrets` y `gitleaks` se pisaban:
  el primero escribía digests SHA1 en `.secrets.baseline` y el segundo los leía
  como API keys por su forma. Ninguno era un secreto real; los dos hallazgos del
  baseline eran un placeholder y la palabra "secrets" en el Makefile.

### Changed
- Un solo escáner de secretos: `gitleaks`, con la misma config (`.gitleaks.toml`)
  en pre-commit y en CI, para que nada pase en local y falle en CI.
- Eliminados `detect-secrets`, `.secrets.baseline` y el hook `detect-private-key`,
  redundantes con `gitleaks` y origen de falsos positivos en cadena.
- La allowlist es deliberadamente estrecha: matchea el placeholder literal, no la
  plantilla entera. Una private key real en `secrets.toml.example` sigue fallando.
- `make check-secrets` ahora corre `gitleaks`.

## [2.4.0] - 2026/07/16

Reestructuración del repositorio. La app no cambió de comportamiento.

### Added
- Pipeline de ingesta publicado como paquete en `src/rehab_strength/`:
  `ingest.sleep` (Google Sheets + Garmin), `ingest.strong` (export de Strong) y
  `ingest.run_all`. Antes vivía fuera del repo con rutas absolutas.
- `config.py`: todas las rutas se resuelven relativas a la raíz del repo y se
  pueden sobrescribir por variables de entorno.
- `gsheets.py`: las credenciales de Google se resuelven en runtime desde el
  entorno o desde los secrets de Streamlit, nunca desde un archivo del repo.
- Carpetas `data/{raw,processed,external}/`, `models/` y `reports/figures/`, cada
  una con su README y todas ignoradas por git.
- Suite de tests (34 casos) sobre parseo de duraciones, scoring de siestas y
  limpieza de workouts.
- CI en GitHub Actions: lint, tests y escaneo de secretos con `gitleaks` sobre
  todo el historial.
- `pre-commit` con `detect-secrets`, `detect-private-key`, `nbstripout` y `ruff`.
- `Makefile`, `pyproject.toml`, `LICENSE` y plantillas `.env.example` y
  `.streamlit/secrets.toml.example`.

### Changed
- `beta.py` renombrado a `streamlit_app.py` (historial preservado con `git mv`).
- `.gitignore` ampliado para bloquear credenciales, datos de salud, modelos y
  notebooks con outputs.
- README: arquitectura, instalación y una sección de privacidad que refleja que
  ahora se publica el código de ingesta pero nunca los datos.

### Fixed
- **Un peso mal capturado borraba el entrenamiento completo.** En la limpieza de
  workouts, `DATE` es el índice y todos los sets de una sesión comparten
  timestamp, así que `drop(index=...)` eliminaba todas las filas de esa fecha en
  vez del set con peso > 900 lbs. Ahora se filtra con una máscara booleana. El
  bug era latente: no hay pesos > 900 lbs en los datos actuales.

## [2.3.4] - 2026/06/07
- Time to compare models
- Non linear models included fit, lc & performance metrics (DT, KNN, SVR)
- Time series analysis tab - Stationary and Autocorrelation (ACF, PACF, ADF, KPSS)
- Ensemble models added (Random Forest, AdaBoost and GradientBoost)
- st.cache_data() added to minimize times of training models when data is not changed
- Learning curve improved

## [2.3.3] -2026/04/26
- Regression Learned models: OLS, Ridge, Lasso, Enet
- Learning curve
- Shap interpretation
- Unsupervised Models: PCA, T-SNE, K-Means

## [2.3.2] -2026/03/15
- Ramsey reset Test on OLS (Linearity)
- Durbin Watson Test (Autocorrelation)
- OLS with polynomial
- Stress_sleep feature gathered and added to the model
- Models included: Elastic Net, Ridge, Lasso, Decision Tree Regressor

## [2.3.1] -2026/02/21
Stats Tab
- Skewness & Kurtosis added
- Outliers values shown
Models Tab
- Model diagnostic (vih, cook`s distance, leverage)

## [2.3.0] -2026/02/04
-Toggle button to change between OLS and Logit
- OLS linear regression training below 150 samples
- OLS frozen at 150 samples
- OLS for deployment after 150 samples
- Deployment and training params
- OLS analysis (metrics, residuals)
- Correlation matrix
- ECDF and complementary ECDF on stats tab
- Hypothesis test for normality improved
- Learning curve in training phase for OLS model
- Stat Tab: Spearman correlation fully deployed

## [2.2.0] - 2026/01/28
-Stats tab
-Location estimates metrics
-CV metric
-Histogram, boxplots & lineplots for features
-Shapiro Wilk test for normality of distributions
-Outliers detection IQR & z-score modified
-Hypothesis testinf using ttest for normal distributions and mwu/correlation for non normal distributions

## [2.1.1] -2026/01/28
-Tag HRV metric on home tab as Excellent, Good or Bad
-Time exercised metric on home tab with a goal of 4 hours
-Recovery plot with nap on home tab
-Home tab recovery charts with insights
-Error handling with naps metrics

## [2.1.0] - 2026/01/20
-Naps added to sleep tab
-Home tab with nap metrics (average, nap days & nap frequency)
-Naps classified based on the duration & the hour it was taken
-Sigmoid recovery function modified by naps
-Home tab with naps summary (sigmoid modified, delta & status)

## [2.0.4] -2026/01/18
-Fixed data governance for better accuracy
-Visuals improved

## [2.0.3] -2026/01/17
- Sliders with moving average up to 11 days added for plot recovery and sleep charts in home tab
-Dark mode always ON
-Standardize y-axis scaling for consistent interpretation


## [2.0.2] - 2026/01/16
- Data freshness badge (workouts / sleep / recovery)
- Home tab data status indicator (slightly delayed / fresh)

## [2.0.1]
- Data integrity added as a visible metric in Home tab

## [2.0.0]
- Fixed major bugs
- Collapsed file upload panel
- Added data integrity checks
- Added pd.merge logic in import_sleep_data.py
- Refactored app architecture

## [1.5.0]
- Added 5 tabs: Home, Workouts, Sleep, Recovery, Correlations
- Introduced sigmoid recovery score (experimental)

## [1.0.0]
- Initial app creatioN
