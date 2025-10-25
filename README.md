# Bot de Trading

Este repositorio contiene un bot de trading de criptomonedas que obtiene datos de mercado desde Binance y ejecuta órdenes en Bitget.
Además soporta la conexión con otros exchanges opcionalmente y dispone de un pequeño panel web para monitorear las operaciones en tiempo real. También pueden enviarse avisos por Telegram y Discord.

## Funcionalidades

- Indicadores técnicos avanzados (RSI, MACD, ATR) con soporte opcional de [TA-Lib](https://ta-lib.org) y soportes/resistencias mediante extremos locales (requiere `scipy`)
- Parámetros de gestión de riesgo
- Tamaño de posición calculado según porcentaje de balance disponible y distancia al stop

- Stop-loss y take-profit con límite diario de pérdidas
- Filtrado de símbolos y ajuste de apalancamiento
- Verificación del apalancamiento aplicado en Bitget tras la configuración
- Ejecución de órdenes en varios exchanges mediante `ccxt`
- Tipos de orden avanzados (market, limit, stop)
- Selección de trades priorizando la mayor probabilidad con ratio beneficio/riesgo >= 2:1
- Panel web en `http://localhost:8000` para monitoreo en tiempo real. Los datos
  de operaciones se consultan desde el endpoint `/api/trades`
- La plantilla HTML del dashboard se encuentra en `trading_bot/templates/index.html` para facilitar su personalizacion.
- Si Flask no está instalado, intentar iniciar el dashboard lanzará un `ImportError`.
- Las operaciones abiertas se registran en `trade_manager` y el dashboard las
  obtiene directamente desde ahí. Cada operación muestra en la tabla su PnL no
  realizado calculado con el precio actual
- Notificaciones por Telegram y Discord al abrir y cerrar operaciones
- Arquitectura modular para facilitar mejoras
- Métricas de Prometheus disponibles en `http://localhost:8001/metrics`
- Análisis del libro de órdenes para zonas de liquidez
- Los símbolos en el dashboard de liquidez muestran la fuente, p.ej.
  `BTC_USDT (Binance)` o `BTC_USDT (Bitget)`
 - La lista de símbolos que analiza el bot se obtiene de forma dinámica según el
  volumen de futuros USDT del exchange. En `TEST_MODE` se usa `MockExchange`,
  que genera 25 pares con volúmenes deterministas para devolver siempre los 15
  más líquidos.
- Sentimiento de mercado usando el ratio de posiciones long/short de Bitget
- Reconciliación automática con posiciones de Bitget al iniciar
- Cancelación de órdenes pendientes no registradas al sincronizar

- Cancelación automática de órdenes límite que exceden cierta antigüedad
- Verificación de balance y llenado de órdenes con registro de slippage
- Historial persistente de operaciones en `trade_history.csv` y archivos JSON
- Endpoints públicos de Binance para ticker, libro de órdenes y velas
- Flujo en tiempo real del order book por `wss://fstream.binance.com`
- Los WebSocket se inician explícitamente con `strategy.start_liquidity()` para evitar conexiones al importar módulos. Sin pasar símbolos se conectará automáticamente a los 15 pares de mayor volumen
- Modelos de machine learning optimizados con `trading_bot.optimizer`
- Entrenamiento de modelos con `python -m trading_bot.train_model`
- Exchange simulado para pruebas sin conexión a Bitget
- Motor de backtesting con `python -m trading_bot.backtest`

## Uso

Instala las dependencias y pon en marcha el bot:

```bash
pip install -r requirements.txt
python -m trading_bot.bot
python -m trading_bot.backtest
python -m trading_bot.train_model miarchivo.csv --target result

Los antiguos envoltorios `run_backtest.py` (cargaba `backtest.yml` + `backtest.csv`
y ejecutaba el motor) y `train_predictive_model.py` (envoltorio CLI sobre
`trading_bot.predictive_model`) se eliminaron para evitar duplicar lógica de
línea de comandos. Utiliza directamente los módulos `trading_bot.backtest` y
`trading_bot.train_model`, que exponen las mismas interfaces desde un único
punto de mantenimiento.

`get_market_data` obtiene hasta 500 velas por defecto usando el endpoint
`/fapi/v1/klines` de Binance. Puedes ajustar el parámetro `limit` (1‑1000)
para cargar más o menos historial. Las velas descargadas se guardan en
`cache/` para poder realizar análisis en modo offline si la API no está
disponible.

### Variables de entorno

Las credenciales de API y el comportamiento del bot se controlan mediante las
variables definidas en `trading_bot/config.py`:

- `BITGET_API_KEY`, `BITGET_API_SECRET`, `BITGET_PASSPHRASE`
- `BINANCE_API_KEY`, `BINANCE_API_SECRET`
- `DEFAULT_EXCHANGE` (default `bitget`)
- `TELEGRAM_TOKEN` / `TELEGRAM_CHAT_ID`
- `DISCORD_WEBHOOK`
- `MAX_OPEN_TRADES` (default 10)
- `DAILY_RISK_LIMIT` (default `-50`)
- `TEST_MODE` set to `1` to use a mock exchange without sending real orders

- `TEST_SYMBOLS` comma separated list of symbols to analyse when `TEST_MODE` is
  enabled
- `MODEL_PATH` path to saved ML model (default `model.pkl`)
- `STOP_ATR_MULT` ATR multiple for stop loss (default `1.5`)
- `RSI_PERIOD` período del RSI (default `14`)
- `MIN_RISK_REWARD` ratio mínimo beneficio/riesgo para abrir (default `2.0`, `1.3` en modo test)
- `DEFAULT_LEVERAGE` apalancamiento por defecto (default `10`)
- `MIN_POSITION_SIZE` tamaño mínimo de posición permitido (default `0.001`, `1e-4` en modo test)
- `RISK_PER_TRADE` cantidad fija en USDT o porcentaje del balance a arriesgar por trade. Si es menor que 1 se interpreta como porcentaje (default `0.01`, es decir 1% del saldo)
- `ORDER_FILL_TIMEOUT` seconds to wait before canceling unfilled limit orders (default `15`)

- `ENABLE_TRADE_HISTORY_LOG` activa el registro detallado de cambios en memoria (default `0`)
- `MAX_TRADE_HISTORY_SIZE` número máximo de eventos en memoria antes de descartar los más antiguos (default `1000`)
- `MAX_TRADES_PER_SYMBOL` límite de operaciones simultáneas por par (default `1`)
- `COOLDOWN_MINUTES` minutos de espera tras cerrar una operación antes de volver a operar el mismo par (default `5`)

- `ORDER_MAX_AGE` seconds after which pending orders are automatically cancelled (default `60`)
- `MAX_SLIPPAGE` maximum allowed difference between target and execution price when closing a trade (default `0.01`)
- `WEBAPP_HOST` dashboard host (default `0.0.0.0`)
- `WEBAPP_PORT` dashboard port (default `8000`)
- `DASHBOARD_GATEWAY_BASE` URL base del gateway FastAPI (déjalo vacío para usar el
  mismo origen que el dashboard; default cadena vacía)
- `ANALYTICS_GRAPHQL_URL` endpoint opcional de analítica GraphQL (default
  `${DASHBOARD_GATEWAY_BASE}/graphql` cuando el gateway está configurado)
- `AI_ASSISTANT_URL` endpoint opcional del asistente IA (default
  `${DASHBOARD_GATEWAY_BASE}/ai/chat` cuando el gateway está configurado)
- `BOT_SERVICE_URL` URL base que utilizará el gateway para reenviar llamadas al
  dashboard Flask del bot (default `http://trading_bot:8000`)

Copia `.env.example` a `.env` y rellena tus claves API para comenzar. El bot
cargará automáticamente ese archivo al iniciarse.

## Auto-Training y filtro de ruido

El bot puede registrar automáticamente operaciones cerradas para re-entrenar el
modelo predictivo en caliente. Para activarlo:

1. Ajusta tu `.env` con los nuevos parámetros:

   ```env
   ENABLE_MODEL=1
   AUTO_TRAIN_ENABLED=1
   AUTO_TRAIN_POLL_SECONDS=30
   MIN_TRAIN_SAMPLE_SIZE=50
   RETRAIN_INTERVAL_TRADES=300
   DATASET_PATH=./data/auto_train_data.csv
   MODEL_DIR=./models
   POST_DEPLOY_MIN_SAMPLES=50
   POST_DEPLOY_MIN_HIT_RATE=0.52
   POST_DEPLOY_MAX_DRIFT=0.08
   NOISE_FILTER_METHOD=ema
   NOISE_FILTER_SPAN=12
   VOL_HIGH_TH=0.015
   VOL_MARGIN_BPS=10
   ```

2. Cada cierre de trade añade una fila etiquetada en `DATASET_PATH` usando un
   *file lock* para evitar corrupciones. El hilo `auto_trainer` valida el
   esquema (con winsorization ligera) antes de entrenar y usa *class
   weighting* balanceado. El repositorio incluye un dataset semilla en
   `data/auto_train_data.csv` con 60 ejemplos para que el primer entrenamiento
   pueda ejecutarse nada más arrancar.

3. Cuando hay suficientes muestras, el modelo se entrena de nuevo. Si supera
   los umbrales offline (`MODEL_MIN_WIN_RATE` y `MODEL_MAX_CALIBRATION_DRIFT`),
   se despliega con swap atómico escribiendo primero `model.pkl.tmp` y luego
   `os.replace`. Cada versión se guarda como `models/model-YYYYMMDD_HHMMSS.pkl`
   y el symlink `models/latest` apunta al modelo activo.

4. Tras desplegar, el bot monitorea una ventana corta de operaciones vivas. Si
   el `MODEL_MONITOR` detecta que el hit-rate cae por debajo de los mínimos o
   el *drift* excede el umbral, se hace **rollback** inmediato al artefacto
   `good_model.pkl` sin interrumpir el proceso.

5. Las nuevas métricas Prometheus (`auto_train_last_duration_seconds`,
   `auto_train_last_status`, `auto_train_samples_total`, `model_version_info`)
   permiten vigilar el pipeline de entrenamiento en Grafana/Alertmanager.

El filtro de ruido aplica un suavizado configurable (EMA o mediana) antes de
calcular indicadores y aumenta automáticamente el umbral de probabilidad cuando
la volatilidad (ATR/precio) supera `VOL_HIGH_TH`. Ajusta `VOL_MARGIN_BPS` para
controlar el margen adicional expresado en *basis points*.
## Dependencias opcionales

Para acelerar el cálculo de indicadores se puede instalar [TA-Lib](https://ta-lib.org).
La detección de soportes y resistencias utiliza `scipy`; si estas bibliotecas no están disponibles, se usarán implementaciones en Python puro.


## Configuración avanzada

| Variable | Descripción | Por defecto |
|----------------------|-------------------------------------------------------------------------------|------------|
| `LOG_LEVEL` | Nivel de verbosidad del log (`DEBUG`, `INFO`, `WARNING`, `ERROR`). | `INFO` |
| `DATA_RETRY_ATTEMPTS`| Número de reintentos al descargar datos antes de usar la caché. | `3` |
| `ORDER_SUBMIT_ATTEMPTS` | Reintentos al enviar o cerrar órdenes. | `3` |
| `TRADE_COOLDOWN` | Tiempo en segundos que debe transcurrir antes de reabrir el mismo símbolo (derivado de `COOLDOWN_MINUTES` si no se especifica). | `300` |
| `MAX_CONCURRENT_REQUESTS` | Número máximo de llamadas simultáneas al exchange. | `5` |
| `CPU_THRESHOLD` | Fracción de uso de CPU a partir de la cual se enviará una alerta (0–1). | `0.8` |
| `MEMORY_THRESHOLD_MB` | Uso de memoria en MB que dispara una alerta. | `500` |
| `LATENCY_THRESHOLD_MS` | Latencia (en milisegundos) considerada excesiva para llamadas de API. | `1000` |

Las métricas del bot se exponen por defecto en `http://localhost:8001/metrics`. Pueden configurarse alertas de uso de CPU y memoria mediante las variables de entorno anteriores.

```mermaid
graph TD
    config --> data
    config --> execution
    data --> strategy
    strategy --> execution
    execution --> trade_manager
    trade_manager --> webapp
    webapp --> notify
```

## Production hardening

El bot incluye utilidades pensadas para operar en producción con resiliencia:


### Selector de modo de arranque

* Ejecuta `python -m trading_bot.bot --mode {normal,shadow,heuristic,hybrid,testnet,backtest,maintenance}` para seleccionar el comportamiento en CLI.
* Alternativamente define `BOT_MODE` en el entorno; si no hay bandera ni variable y existe TTY, se mostrará un menú interactivo.
* Las banderas aplican sobre los knobs existentes (`ENABLE_TRADING`, `SHADOW_MODE`, `ENABLE_MODEL`, etc.) sin romper la compatibilidad.

| Modo | Descripción | Efectos principales |
| --- | --- | --- |
| `normal` / `hybrid` | Trading real combinando heurística + modelo | `ENABLE_TRADING=1`, `ENABLE_MODEL=1` |
| `heuristic` | Solo heurística (modelo deshabilitado) | `ENABLE_TRADING=1`, `ENABLE_MODEL=0`, `MODEL_WEIGHT=0.0` |
| `shadow` | Shadow-mode A/B sin enviar órdenes reales | `ENABLE_TRADING=0`, `SHADOW_MODE=1` |
| `testnet` | Dry-run o testnet con envío simulado | `ENABLE_TRADING=1`, `DRY_RUN=1` |
| `backtest` | Ejecuta el runner de backtest al arrancar y termina | `RUN_BACKTEST_ON_START=1` |
| `maintenance` | Mantiene métricas sin abrir nuevas operaciones | `ENABLE_TRADING=0`, `MAINTENANCE=1` |


### Backtest walk-forward

* Ejecuta `python -m trading_bot.backtest --config backtest.yml --data dataset.csv` para generar reportes en `reports/backtest_*` con KPIs económicos (CAGR, Sharpe, Max Drawdown, Profit Factor, Expectancy).
* El modo `walk_forward` amplía progresivamente la ventana de entrenamiento, mientras que `rolling` usa una ventana móvil (`rolling_train_days` / `rolling_test_days`).
* Los reportes contienen `kpis.json`, `trades.csv` y `config.json` para auditar cada corrida.

### Shadow-mode

* Actívalo con `SHADOW_MODE=1` para comparar heurística vs. blending ML sin enviar órdenes reales.
* Los resultados se almacenan en `shadow_trades/` (`signals.jsonl`, `results.jsonl`) y permiten calcular KPIs de ambos enfoques antes de desplegar.
* Para ejecutar shadow con tamaño fijo y datos exclusivos de Bitget configura tu `.env` con:
  * `BOT_MODE=shadow`, `ENABLE_TRADING=false`, `DRY_RUN=true`.
  * `USE_FIXED_POSITION_SIZE=true`, `FIXED_POSITION_SIZE_USDT=20`, `MIN_POSITION_SIZE_USDT=5`, `ENFORCE_EXCHANGE_MIN_NOTIONAL=false`.
  * `PRIMARY_EXCHANGE=bitget`, `DATA_EXCHANGE=bitget`, `WS_EXCHANGE=bitget`, `ENABLE_BITGET=true`, `ENABLE_BINANCE=false`.

### Latencia y alertas

* Las métricas `trading_bot_latency_ms_*` exponen p95/p99 de feature→predicción, submit→ack y ack→filled.
* `maybe_alert` envía avisos a Telegram/Discord cuando se superan umbrales de pérdida diaria, latencia o drift del modelo.

### Apagado limpio

* El manejador de señales (`trading_bot.shutdown`) detiene nuevas entradas, cancela órdenes pendientes y guarda el estado antes de finalizar el proceso.

### Nuevas variables de entorno

| Variable | Descripción | Por defecto |
| --- | --- | --- |
| `MODEL_WEIGHT` | Peso del modelo predictivo en el blending (0–1). | `0.5` |
| `MIN_PROB_SUCCESS` | Probabilidad mínima aceptada tras fees y blending. | `0.55` |
| `FEE_EST` | Estimación de fees+slippage para los filtros de probabilidad. | `0.0006` |
| `SHADOW_MODE` | Si es `1`, habilita el modo sombra (sin órdenes reales). | `0` |

| `BOT_MODE` | Define el modo de arranque si no se pasa `--mode`. | _vacío_ |
| `DRY_RUN` | Fuerza el envío simulado de órdenes. | `0` |
| `BACKTEST_REPORT_DIR` | Carpeta raíz donde se guardan los reportes de backtest. | `reports` |
| `BACKTEST_CONFIG_PATH` / `BACKTEST_DATA_PATH` | Rutas usadas al iniciar en modo `backtest`. | `backtest.yml` / `backtest.csv` |

| `MAX_API_RETRIES` | Reintentos máximos al contactar con el exchange. | `5` |
| `API_BACKOFF_BASE` | Backoff base (segundos) para reintentos exponenciales. | `0.2` |
| `LATENCY_SLO_MS` | Límite de latencia (ms) que dispara alertas automáticas. | `1000` |
| `RSI_OVERSOLD` / `RSI_OVERBOUGHT` | Banda de RSI configurable para la heurística. | `45` / `55` |

## Pruebas

Para ejecutar todas las pruebas y ver el reporte de cobertura ejecuta:

```bash
pytest
```


## Licencia

Este proyecto se distribuye bajo la licencia MIT. Consulta el archivo [LICENSE](LICENSE) para más informacion.


## Nueva arquitectura basada en microservicios

El bot se ha ampliado para funcionar como una plataforma moderna compuesta por servicios independientes:

- **Trading Engine** (`services/trading_engine/app.py`): API de FastAPI para la gestión de órdenes en memoria.
- **Streaming Service** (`services/streaming_service/app.py`): publica eventos de mercado mediante Redis Streams.
- **Analytics Service** (`services/analytics/app`): expone un endpoint GraphQL con modelos SQLAlchemy para usuarios, ajustes y operaciones.
- **AI Service** (`services/ai_service/app.py`): integra ChatGPT para generar informes y responder preguntas contextuales.
- **Gateway** (`services/gateway/app.py`): punto de entrada único que reenvía peticiones a los servicios anteriores.
- **Frontend** (`services/frontend`): base documentada para una SPA Next.js/SvelteKit conectada al gateway.

Cada servicio se empaqueta mediante Docker y se orquesta con `docker-compose` para facilitar despliegues reproducibles.

### Orquestación con Docker Compose

El archivo `docker-compose.yml` en la raíz levanta Redis, PostgreSQL y los microservicios descritos. Ajusta las variables de entorno
según tus credenciales y la clave `OPENAI_API_KEY` para habilitar la IA.

### Integración de la IA

El Gateway expone `/ai/report` y `/ai/chat`, que a su vez llaman al servicio AI. Puedes invocar estos endpoints desde el frontend o scripts
de automatización para mostrar informes de rendimiento, explicar métricas y resolver dudas operativas.

### Próximos pasos sugeridos

1. Desplegar el frontend siguiendo las instrucciones de `services/frontend/README.md`.
2. Conectar el bot de trading existente a la cola de eventos (`Redis Streams`) para enviar datos en tiempo real al Streaming Service.
3. Alimentar el Analytics Service con los resultados históricos del bot para aprovechar las consultas GraphQL.
4. Definir pipelines de CI/CD que construyan y publiquen las imágenes Docker de cada microservicio.
