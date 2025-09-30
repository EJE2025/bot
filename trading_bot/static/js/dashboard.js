const REFRESH_INTERVAL = 10000;
const MAX_POINTS = 60;

const numberFormatter = new Intl.NumberFormat('en-US', {
  minimumFractionDigits: 2,
  maximumFractionDigits: 2,
});

const priceFormatter = new Intl.NumberFormat('en-US', {
  minimumFractionDigits: 2,
  maximumFractionDigits: 4,
});

const quantityFormatter = new Intl.NumberFormat('en-US', {
  minimumFractionDigits: 2,
  maximumFractionDigits: 4,
});

const percentFormatter = new Intl.NumberFormat('en-US', {
  style: 'percent',
  minimumFractionDigits: 1,
  maximumFractionDigits: 1,
});

const state = {
  trades: [],
  summary: null,
  history: [],
  liquidity: {},
  symbolFilter: '',
  pnlSeries: [],
};

let pnlChart = null;

async function fetchJSON(url) {
  const response = await fetch(url, { cache: 'no-cache' });
  if (!response.ok) {
    throw new Error(`Request failed for ${url}`);
  }
  return response.json();
}

function setStatus(label, variant) {
  const badge = document.getElementById('connectionStatus');
  badge.textContent = label;
  badge.className = `badge badge-status bg-${variant}`;
}

function showAlert(message, variant = 'danger') {
  const container = document.getElementById('globalAlerts');
  if (!container) return;
  container.innerHTML = `
    <div class="alert alert-${variant} alert-dismissible fade show" role="alert">
      ${message}
      <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
    </div>`;
}

function clearAlert() {
  const container = document.getElementById('globalAlerts');
  if (container) {
    container.innerHTML = '';
  }
}

function formatNumber(value, formatter = numberFormatter) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) {
    return '—';
  }
  return formatter.format(numeric);
}

function formatPnL(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) {
    return '—';
  }
  const prefix = numeric > 0 ? '+' : '';
  return `${prefix}${numberFormatter.format(numeric)}`;
}

function formatDate(value) {
  if (!value) return '—';
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return value;
  }
  return date.toLocaleString();
}

function ensureChart() {
  if (pnlChart) {
    return pnlChart;
  }
  const ctx = document.getElementById('pnlChart');
  if (!ctx) {
    return null;
  }
  pnlChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: [],
      datasets: [
        {
          label: 'PnL no realizado',
          data: [],
          borderColor: '#2b6cb0',
          backgroundColor: 'rgba(66, 153, 225, 0.25)',
          tension: 0.35,
          fill: true,
          borderWidth: 2,
          pointRadius: 0,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          display: false,
        },
        tooltip: {
          callbacks: {
            label(context) {
              return formatPnL(context.parsed.y);
            },
          },
        },
      },
      scales: {
        x: {
          ticks: {
            color: '#5b6478',
          },
          grid: {
            color: 'rgba(91, 100, 120, 0.15)',
          },
        },
        y: {
          ticks: {
            color: '#5b6478',
            callback(value) {
              return formatNumber(value);
            },
          },
          grid: {
            color: 'rgba(91, 100, 120, 0.15)',
          },
        },
      },
    },
  });
  return pnlChart;
}

function updatePnlSeries(summary) {
  if (!summary) return;
  const chart = ensureChart();
  if (!chart) return;

  const timestampLabel = new Date(summary.generated_at || Date.now()).toLocaleTimeString();
  const currentValue = Number(summary.unrealized_pnl) || 0;

  state.pnlSeries.push({ label: timestampLabel, value: currentValue });
  if (state.pnlSeries.length > MAX_POINTS) {
    state.pnlSeries.shift();
  }

  chart.data.labels = state.pnlSeries.map((item) => item.label);
  chart.data.datasets[0].data = state.pnlSeries.map((item) => item.value);
  chart.update('none');
}

function renderSummary(summary) {
  if (!summary) return;
  document.getElementById('metricPositions').textContent = summary.total_positions;
  document.getElementById('metricPnL').textContent = formatPnL(summary.unrealized_pnl);
  document.getElementById('metricExposure').textContent = formatNumber(summary.gross_notional);
  document.getElementById('metricWinRate').textContent = formatNumber(summary.win_rate, percentFormatter);
  document.getElementById('lastUpdated').textContent = new Date(summary.generated_at).toLocaleTimeString();

  const list = document.getElementById('symbolBreakdown');
  if (!list) return;
  list.innerHTML = '';
  if (!summary.per_symbol || summary.per_symbol.length === 0) {
    list.innerHTML = '<li class="list-group-item">Sin posiciones abiertas.</li>';
    return;
  }

  summary.per_symbol.forEach((item) => {
    const pnlClass = item.unrealized_pnl >= 0 ? 'text-success' : 'text-danger';
    const exposure = formatNumber(item.exposure, quantityFormatter);
    const pnl = formatPnL(item.unrealized_pnl);
    const notional = formatNumber(item.notional_value);
    const li = document.createElement('li');
    li.className = 'list-group-item d-flex flex-column flex-sm-row align-items-sm-center justify-content-between';
    li.innerHTML = `
      <div class="d-flex align-items-center gap-2 mb-2 mb-sm-0">
        <span class="symbol-badge"><i class="bi bi-graph-up"></i>${item.symbol}</span>
        <span class="badge bg-dark-subtle text-dark">${item.positions} posiciones</span>
      </div>
      <div class="d-flex flex-wrap gap-3">
        <span><strong>Exposición:</strong> ${exposure}</span>
        <span><strong>Notional:</strong> ${notional}</span>
        <span class="${pnlClass}"><strong>PnL:</strong> ${pnl}</span>
      </div>`;
    list.appendChild(li);
  });
}

function renderTrades() {
  const tbody = document.getElementById('tradesTableBody');
  if (!tbody) return;

  const filter = state.symbolFilter.trim().toLowerCase();
  const trades = filter
    ? state.trades.filter((trade) => String(trade.symbol || '').toLowerCase().includes(filter))
    : state.trades;

  if (!trades || trades.length === 0) {
    tbody.innerHTML = '<tr><td colspan="9" class="text-center text-muted py-4">No hay operaciones abiertas.</td></tr>';
    return;
  }

  const rows = trades
    .map((trade) => {
      const side = String(trade.side || '').toLowerCase();
      const sideClass = side === 'buy' ? 'buy' : 'sell';
      const pnlClass = trade.pnl_unrealized >= 0 ? 'pnl-positive' : 'pnl-negative';
      const currentPrice = formatNumber(trade.current_price, priceFormatter);
      const entryPrice = formatNumber(trade.entry_price, priceFormatter);
      const takeProfit = formatNumber(trade.take_profit, priceFormatter);
      const stopLoss = formatNumber(trade.stop_loss, priceFormatter);
      const quantity = formatNumber(trade.quantity, quantityFormatter);
      const pnl = formatPnL(trade.pnl_unrealized);
      return `
        <tr>
          <td><span class="symbol-badge"><i class="bi bi-currency-bitcoin"></i>${trade.symbol}</span></td>
          <td class="trade-side ${sideClass}">${side.toUpperCase()}</td>
          <td>${quantity}</td>
          <td>${entryPrice}</td>
          <td>${currentPrice}</td>
          <td>${takeProfit}</td>
          <td>${stopLoss}</td>
          <td class="${pnlClass}">${pnl}</td>
          <td>${trade.open_time ? formatDate(trade.open_time) : '—'}</td>
        </tr>`;
    })
    .join('');

  tbody.innerHTML = rows;
}

function renderLiquidity() {
  const container = document.getElementById('liquidityContainer');
  if (!container) return;

  const entries = Object.entries(state.liquidity || {});
  if (entries.length === 0) {
    container.innerHTML = '<div class="text-muted text-center py-4">No hay datos de liquidez disponibles.</div>';
    return;
  }

  const html = entries
    .map(([symbol, book]) => {
      const bids = (book.bids || []).slice(0, 5);
      const asks = (book.asks || []).slice(0, 5);
      const rows = [];
      for (let i = 0; i < Math.max(bids.length, asks.length); i += 1) {
        const bid = bids[i] || ['—', '—'];
        const ask = asks[i] || ['—', '—'];
        rows.push(`
          <tr>
            <td class="text-success">${formatNumber(bid[0], priceFormatter)}</td>
            <td class="text-success">${formatNumber(bid[1], quantityFormatter)}</td>
            <td class="text-danger">${formatNumber(ask[0], priceFormatter)}</td>
            <td class="text-danger">${formatNumber(ask[1], quantityFormatter)}</td>
          </tr>`);
      }
      return `
        <div class="orderbook-card">
          <div class="d-flex justify-content-between align-items-center mb-2">
            <span class="symbol-badge"><i class="bi bi-lightning"></i>${symbol}</span>
            <span class="badge bg-secondary-subtle text-dark">Top 5 niveles</span>
          </div>
          <div class="table-responsive">
            <table class="table table-sm mb-0">
              <thead>
                <tr>
                  <th class="text-success">Bid</th>
                  <th class="text-success">Cantidad</th>
                  <th class="text-danger">Ask</th>
                  <th class="text-danger">Cantidad</th>
                </tr>
              </thead>
              <tbody>
                ${rows.join('')}
              </tbody>
            </table>
          </div>
        </div>`;
    })
    .join('');

  container.innerHTML = `<div class="liquidity-grid">${html}</div>`;
}

function renderHistory() {
  const container = document.getElementById('historyList');
  if (!container) return;

  if (!state.history || state.history.length === 0) {
    container.innerHTML = '<div class="text-center text-muted">Sin operaciones cerradas recientes.</div>';
    return;
  }

  const items = state.history
    .map((trade) => {
      const side = String(trade.side || '').toLowerCase();
      const sideLabel = side === 'buy' ? 'Compra' : 'Venta';
      const pnlClass = Number(trade.profit) >= 0 ? 'text-success' : 'text-danger';
      return `
        <div class="history-item">
          <div class="d-flex justify-content-between align-items-center">
            <span class="symbol-badge"><i class="bi bi-clock-history"></i>${trade.symbol || '—'}</span>
            <span class="${side === 'buy' ? 'side-buy' : 'side-sell'}">${sideLabel}</span>
          </div>
          <div class="d-flex flex-wrap gap-3">
            <span><strong>Entrada:</strong> ${formatNumber(trade.entry_price, priceFormatter)}</span>
            <span><strong>Salida:</strong> ${formatNumber(trade.exit_price, priceFormatter)}</span>
            <span class="${pnlClass}"><strong>PnL:</strong> ${formatPnL(trade.profit)}</span>
          </div>
          <div class="text-muted small">${formatDate(trade.open_time)} → ${formatDate(trade.close_time)}</div>
        </div>`;
    })
    .join('');

  container.innerHTML = items;
}

async function refreshDashboard(manual = false) {
  try {
    if (manual) {
      setStatus('Actualizando…', 'info');
    }
    clearAlert();
    const [trades, summary, liquidity, history] = await Promise.all([
      fetchJSON('/api/trades'),
      fetchJSON('/api/summary'),
      fetchJSON('/api/liquidity'),
      fetchJSON('/api/history?limit=15').catch(() => []),
    ]);

    state.trades = trades;
    state.summary = summary;
    state.liquidity = liquidity;
    state.history = history;

    renderTrades();
    renderSummary(summary);
    renderLiquidity();
    renderHistory();
    updatePnlSeries(summary);
    setStatus('En vivo', 'success');
  } catch (error) {
    console.error(error);
    showAlert('No se pudieron sincronizar los datos del bot. Reintentaremos automáticamente.', 'danger');
    setStatus('Desconectado', 'danger');
  }
}

function attachEvents() {
  const filterInput = document.getElementById('symbolFilter');
  if (filterInput) {
    filterInput.addEventListener('input', (event) => {
      state.symbolFilter = event.target.value;
      renderTrades();
    });
  }

  const refreshBtn = document.getElementById('refreshTradesBtn');
  if (refreshBtn) {
    refreshBtn.addEventListener('click', () => refreshDashboard(true));
  }
}

function initialize() {
  ensureChart();
  attachEvents();
  refreshDashboard();
  setStatus('Sincronizando…', 'warning');
  setInterval(refreshDashboard, REFRESH_INTERVAL);
}

document.addEventListener('DOMContentLoaded', initialize);
