const REFRESH_INTERVAL = 10000;
const MAX_VIEW_POINTS = 60;
const MAX_SERIES_POINTS = 500;

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
  symbolFilter: '',
  pnlSeries: [],
  sessionBaselines: {
    realized_balance: 0,
    realized_pnl_total: 0,
    total_pnl: 0,
  },
  tradingActive: true,
  connectionHealthy: true,
  sessionId: null,
  pnlViewportOffset: 0,
  pnlViewportManual: false,
};

let pnlChart = null;
let socket = null;
let partialModal = null;
let partialTradeContext = null;
let pnlScroll = null;

function getTradeId(trade) {
  return trade.trade_id || trade.id || trade.uuid;
}

async function fetchJSON(url) {
  const response = await fetch(url, { cache: 'no-cache' });
  if (!response.ok) {
    throw new Error(`Request failed for ${url}`);
  }
  return response.json();
}

function setStatus(label, variant) {
  const badge = document.getElementById('connectionStatus');
  if (!badge) return;
  badge.textContent = label;
  badge.className = `badge badge-status bg-${variant}`;
}

function updateTradingControls(isActive) {
  const toggleBtn = document.getElementById('toggleTradeBtn');
  if (typeof isActive === 'boolean') {
    state.tradingActive = isActive;
    if (state.connectionHealthy) {
      const label = isActive ? 'En vivo' : 'Pausado';
      const variant = isActive ? 'success' : 'secondary';
      setStatus(label, variant);
    }
    if (toggleBtn) {
      toggleBtn.disabled = !state.connectionHealthy;
      toggleBtn.className = `btn btn-sm ${isActive ? 'btn-warning' : 'btn-success'}`;
      toggleBtn.innerHTML = isActive
        ? '<i class="bi bi-pause-circle"></i> Pausar bot'
        : '<i class="bi bi-play-circle"></i> Reanudar bot';
    }
  }
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

function showToast(message, variant = 'info') {
  const container = document.getElementById('toastContainer');
  if (!container || !window.bootstrap) {
    console.log(`[${variant}] ${message}`);
    return;
  }
  const toastEl = document.createElement('div');
  const variantClass = variant === 'error' ? 'danger' : variant;
  toastEl.className = `toast align-items-center text-bg-${variantClass} border-0`;
  toastEl.setAttribute('role', 'alert');
  toastEl.setAttribute('aria-live', 'assertive');
  toastEl.setAttribute('aria-atomic', 'true');
  toastEl.innerHTML = `
    <div class="d-flex">
      <div class="toast-body">${message}</div>
      <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast" aria-label="Cerrar"></button>
    </div>`;
  container.appendChild(toastEl);
  const toast = new bootstrap.Toast(toastEl, { delay: 3500 });
  toast.show();
  toastEl.addEventListener('hidden.bs.toast', () => {
    toastEl.remove();
  });
}

function setRowBusy(tradeId, busy) {
  const row = document.querySelector(`tr[data-trade-id="${tradeId}"]`);
  if (!row) return;
  row.classList.toggle('table-active', busy);
  row.querySelectorAll('button[data-trade-id]').forEach((btn) => {
    btn.disabled = busy;
  });
}

function formatNumber(value, formatter = numberFormatter) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) {
    return '—';
  }
  return formatter.format(numeric);
}

function formatPnL(value) {
  const raw = Number(value);
  if (!Number.isFinite(raw)) {
    return '—';
  }
  const numeric = normalizeBaselineDelta(raw);
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

function normalizeBaselineDelta(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) {
    return 0;
  }
  return Math.abs(numeric) < 1e-8 ? 0 : numeric;
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
          label: 'PnL total',
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

function resetPnlSeries() {
  state.pnlSeries = [];
  state.pnlViewportOffset = 0;
  state.pnlViewportManual = false;
  const chart = ensureChart();
  if (!chart) return;
  chart.data.labels = [];
  chart.data.datasets[0].data = [];
  chart.update('none');
  updatePnlSlider();
}

function getPnlScroll() {
  if (!pnlScroll) {
    pnlScroll = document.getElementById('pnlScroll');
  }
  return pnlScroll;
}

function updatePnlSlider() {
  const slider = getPnlScroll();
  if (!slider) return;
  const total = state.pnlSeries.length;
  const windowSize = Math.min(MAX_VIEW_POINTS, total);
  const maxOffset = Math.max(0, total - windowSize);
  slider.max = String(maxOffset);
  slider.disabled = maxOffset === 0;
  if (maxOffset === 0) {
    state.pnlViewportOffset = 0;
    state.pnlViewportManual = false;
  }
  const desiredValue = state.pnlViewportManual ? Math.min(state.pnlViewportOffset, maxOffset) : 0;
  if (Number(slider.value) !== desiredValue) {
    slider.value = String(desiredValue);
  }
}

function applyPnlViewport() {
  const chart = ensureChart();
  if (!chart) return;
  const total = state.pnlSeries.length;
  if (total === 0) {
    chart.data.labels = [];
    chart.data.datasets[0].data = [];
    chart.update('none');
    return;
  }
  const windowSize = Math.min(MAX_VIEW_POINTS, total);
  const maxOffset = Math.max(0, total - windowSize);
  const offset = Math.min(state.pnlViewportOffset, maxOffset);
  const startIndex = Math.max(0, total - windowSize - offset);
  const visible = state.pnlSeries.slice(startIndex, startIndex + windowSize);
  chart.data.labels = visible.map((item) => item.label);
  chart.data.datasets[0].data = visible.map((item) => item.value);
  chart.update('none');
}

function handleSession(summary) {
  if (!summary) return;
  const newSessionId = summary.session_id || null;
  const openPositions = Number(summary.total_positions || 0);
  const sessionChanged = newSessionId && newSessionId !== state.sessionId;
  const realizedBaseline = Number(summary.realized_balance ?? 0);
  const realizedTotalBaseline = Number(
    summary.realized_pnl_total ?? summary.realized_pnl ?? 0,
  );
  const totalBaseline = Number(
    summary.total_pnl ?? realizedTotalBaseline + Number(summary.unrealized_pnl ?? 0),
  );
  if (openPositions === 0 && (sessionChanged || state.sessionId === null)) {
    state.sessionBaselines = {
      realized_balance: realizedBaseline,
      realized_pnl_total: realizedTotalBaseline,
      total_pnl: totalBaseline,
    };
    resetPnlSeries();
  }
  if (openPositions > 0 && (state.sessionId === null || sessionChanged)) {
    state.sessionBaselines = {
      realized_balance: realizedBaseline,
      realized_pnl_total: realizedTotalBaseline,
      total_pnl: totalBaseline,
    };
  }
  state.sessionId = newSessionId;
}

function updatePnlSeries(summary) {
  if (!summary) return;
  const chart = ensureChart();
  if (!chart) return;

  const timestampLabel = new Date(summary.generated_at || Date.now()).toLocaleTimeString();
  const realizedRaw = Number(summary.realized_pnl_total ?? summary.realized_pnl ?? 0);
  const unrealized = Number(summary.unrealized_pnl ?? 0);
  const totalRaw = Number(summary.total_pnl ?? realizedRaw + unrealized) || 0;
  const adjustedTotal = normalizeBaselineDelta(
    totalRaw - (state.sessionBaselines.total_pnl || 0),
  );

  state.pnlSeries.push({ label: timestampLabel, value: adjustedTotal });
  if (state.pnlSeries.length > MAX_SERIES_POINTS) {
    state.pnlSeries.shift();
    if (state.pnlViewportManual) {
      const windowSize = Math.min(MAX_VIEW_POINTS, state.pnlSeries.length);
      const maxOffset = Math.max(0, state.pnlSeries.length - windowSize);
      state.pnlViewportOffset = Math.min(state.pnlViewportOffset, maxOffset);
    }
  }

  if (!state.pnlViewportManual) {
    state.pnlViewportOffset = 0;
  }

  updatePnlSlider();
  applyPnlViewport();
}

function renderSummary(summary) {
  if (!summary) return;
  document.getElementById('metricPositions').textContent = summary.total_positions;
  document.getElementById('metricPnL').textContent = formatPnL(summary.unrealized_pnl);
  const totalInvested = Number(summary.total_invested ?? summary.total_exposure ?? 0);
  document.getElementById('metricExposure').textContent = formatNumber(totalInvested);
  document.getElementById('metricWinRate').textContent = formatNumber(summary.win_rate, percentFormatter);
  const realizedBalance = document.getElementById('metricRealizedBalance');
  if (realizedBalance) {
    const baseline = state.sessionBaselines.realized_balance || 0;
    const realizedBalanceValue = normalizeBaselineDelta(
      Number(summary.realized_balance ?? 0) - baseline,
    );
    realizedBalance.textContent = formatNumber(Math.max(0, realizedBalanceValue));
  }
  const realizedPnL = document.getElementById('metricRealizedPnL');
  if (realizedPnL) {
    const baseline = state.sessionBaselines.realized_pnl_total || 0;
    const realizedPnLValue = normalizeBaselineDelta(
      Number(summary.realized_pnl_total ?? summary.realized_pnl ?? 0) - baseline,
    );
    realizedPnL.textContent = formatPnL(realizedPnLValue);
  }
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
    const invested = formatNumber(item.invested_value ?? 0);
    const li = document.createElement('li');
    li.className = 'list-group-item d-flex flex-column flex-sm-row align-items-sm-center justify-content-between';
    li.innerHTML = `
      <div class="d-flex align-items-center gap-2 mb-2 mb-sm-0">
        <span class="symbol-badge"><i class="bi bi-graph-up"></i>${item.symbol}</span>
        <span class="badge bg-dark-subtle text-dark">${item.positions} posiciones</span>
      </div>
      <div class="d-flex flex-wrap gap-3">
        <span><strong>Cantidad:</strong> ${exposure}</span>
        <span><strong>Invertido:</strong> ${invested}</span>
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
  const sortedTrades = trades
    ? [...trades].sort((a, b) => {
        const pnlA = Number(a.pnl_unrealized ?? 0);
        const pnlB = Number(b.pnl_unrealized ?? 0);
        if (!Number.isFinite(pnlA) && !Number.isFinite(pnlB)) return 0;
        if (!Number.isFinite(pnlA)) return 1;
        if (!Number.isFinite(pnlB)) return -1;
        return pnlB - pnlA;
      })
    : [];

  if (!sortedTrades || sortedTrades.length === 0) {
    tbody.innerHTML = '<tr><td colspan="11" class="text-center text-muted py-4">No hay operaciones abiertas.</td></tr>';
    return;
  }

  const rows = sortedTrades
    .map((trade) => {
      const tradeId = getTradeId(trade);
      const side = String(trade.side || '').toLowerCase();
      const sideClass = side === 'buy' ? 'buy' : 'sell';
      const pnlClass = trade.pnl_unrealized >= 0 ? 'pnl-positive' : 'pnl-negative';
      const currentPrice = formatNumber(trade.current_price, priceFormatter);
      const entryPrice = formatNumber(trade.entry_price, priceFormatter);
      const takeProfit = formatNumber(trade.take_profit, priceFormatter);
      const stopLoss = formatNumber(trade.stop_loss, priceFormatter);
      const remaining = trade.quantity_remaining ?? trade.quantity;
      const quantity = formatNumber(remaining, quantityFormatter);
      const pnl = formatPnL(trade.pnl_unrealized);
      const realized = formatPnL(trade.realized_pnl);
      return `
        <tr data-trade-id="${tradeId}">
          <td><span class="symbol-badge"><i class="bi bi-currency-bitcoin"></i>${trade.symbol}</span></td>
          <td class="trade-side ${sideClass}">${side.toUpperCase()}</td>
          <td>${quantity}</td>
          <td>${entryPrice}</td>
          <td>${currentPrice}</td>
          <td>${takeProfit}</td>
          <td>${stopLoss}</td>
          <td class="${pnlClass}">${pnl}</td>
          <td>${realized}</td>
          <td>${trade.open_time ? formatDate(trade.open_time) : '—'}</td>
          <td>
            <div class="btn-group" role="group">
              <button class="btn btn-sm btn-outline-danger" data-trade-id="${tradeId}" data-action="close">Cerrar</button>
              <button class="btn btn-sm btn-outline-warning" data-trade-id="${tradeId}" data-action="partial" data-remaining="${remaining}">Cerrar parcial</button>
            </div>
          </td>
        </tr>`;
    })
    .join('');

  tbody.innerHTML = rows;
  attachTradeRowEvents();
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
    const [trades, summary, history] = await Promise.all([
      fetchJSON('/api/trades'),
      fetchJSON('/api/summary'),
      fetchJSON('/api/history?limit=50').catch(() => []),
    ]);

    state.trades = trades;
    state.summary = summary;
    state.history = history;
    state.connectionHealthy = true;

    const tradingActive =
      summary && Object.prototype.hasOwnProperty.call(summary, 'trading_active')
        ? Boolean(summary.trading_active)
        : state.tradingActive;
    updateTradingControls(tradingActive);

    handleSession(summary);

    renderTrades();
    renderSummary(summary);
    renderHistory();
    updatePnlSeries(summary);
  } catch (error) {
    console.error(error);
    showAlert('No se pudieron sincronizar los datos del bot. Reintentaremos automáticamente.', 'danger');
    state.connectionHealthy = false;
    updateTradingControls(state.tradingActive);
    setStatus('Desconectado', 'danger');
  }
}

function attachEvents() {
  pnlScroll = document.getElementById('pnlScroll');
  if (pnlScroll) {
    pnlScroll.addEventListener('input', (event) => {
      const value = Number(event.target.value || 0);
      if (!Number.isFinite(value) || value < 0) {
        return;
      }
      state.pnlViewportOffset = value;
      state.pnlViewportManual = value !== 0;
      applyPnlViewport();
    });
    pnlScroll.addEventListener('change', (event) => {
      const value = Number(event.target.value || 0);
      if (!Number.isFinite(value) || value < 0) {
        return;
      }
      state.pnlViewportOffset = value;
      state.pnlViewportManual = value !== 0;
      updatePnlSlider();
      applyPnlViewport();
    });
    pnlScroll.addEventListener('dblclick', () => {
      state.pnlViewportOffset = 0;
      state.pnlViewportManual = false;
      updatePnlSlider();
      applyPnlViewport();
    });
  }

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

  const toggleBtn = document.getElementById('toggleTradeBtn');
  if (toggleBtn) {
    toggleBtn.addEventListener('click', async () => {
      toggleBtn.disabled = true;
      try {
        const response = await fetch('/api/toggle-trading', { method: 'POST' });
        const data = await response.json();
        if (!response.ok || !data.ok) {
          throw new Error(data.error || 'No se pudo cambiar el estado del bot');
        }
        state.connectionHealthy = true;
        updateTradingControls(Boolean(data.trading_active));
        showToast(`Trading ${data.trading_active ? 'reanudado' : 'pausado'} correctamente`, 'info');
      } catch (error) {
        console.error(error);
        showToast(error.message || 'Error al cambiar el estado del bot', 'danger');
      } finally {
        updateTradingControls(state.tradingActive);
      }
    });
  }

  const modalElement = document.getElementById('partialCloseModal');
  if (modalElement && window.bootstrap) {
    partialModal = new bootstrap.Modal(modalElement);
  }

  const confirmBtn = document.getElementById('partialConfirmBtn');
  if (confirmBtn) {
    confirmBtn.addEventListener('click', handlePartialConfirm);
  }

  updatePnlSlider();
}

function initialize() {
  ensureChart();
  attachEvents();
  refreshDashboard();
  setStatus('Sincronizando…', 'warning');
  setInterval(refreshDashboard, REFRESH_INTERVAL);
  connectSocket();
}

document.addEventListener('DOMContentLoaded', initialize);

function attachTradeRowEvents() {
  document.querySelectorAll('button[data-action="close"]').forEach((btn) => {
    btn.addEventListener('click', async (event) => {
      const { tradeId } = event.currentTarget.dataset;
      if (!tradeId) return;
      if (!window.confirm('¿Cerrar totalmente esta operación?')) {
        return;
      }
      setRowBusy(tradeId, true);
      try {
        const response = await fetch(`/api/trades/${tradeId}/close`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ reason: 'manual_close_ui' }),
        });
        const data = await response.json();
        if (!response.ok || !data.ok) {
          throw new Error(data.error || 'No se pudo cerrar la operación');
        }
        showToast('Operación cerrada correctamente', 'success');
      } catch (error) {
        console.error(error);
        showToast(error.message || 'Error cerrando la operación', 'danger');
      } finally {
        setRowBusy(tradeId, false);
      }
    });
  });

  document.querySelectorAll('button[data-action="partial"]').forEach((btn) => {
    btn.addEventListener('click', (event) => {
      const { tradeId, remaining } = event.currentTarget.dataset;
      if (!tradeId) return;
      partialTradeContext = { tradeId, remaining: Number(remaining) };
      const info = document.getElementById('partialModalInfo');
      if (info) {
        info.textContent = `Trade ${tradeId} — tamaño restante: ${formatNumber(Number(remaining), quantityFormatter)}`;
      }
      const input = document.getElementById('partialPercent');
      if (input) {
        input.value = '';
        input.focus();
      }
      if (partialModal) {
        partialModal.show();
      }
    });
  });
}

async function handlePartialConfirm() {
  if (!partialTradeContext) {
    return;
  }
  const input = document.getElementById('partialPercent');
  const value = Number(input?.value || 0);
  if (!Number.isFinite(value) || value <= 0 || value > 100) {
    showToast('Porcentaje inválido (1–100)', 'warning');
    return;
  }
  const { tradeId } = partialTradeContext;
  setRowBusy(tradeId, true);
  try {
    const response = await fetch(`/api/trades/${tradeId}/close-partial`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ percent: value, reason: 'manual_partial_ui' }),
    });
    const data = await response.json();
    if (!response.ok || !data.ok) {
      throw new Error(data.error || 'No se pudo cerrar parcialmente');
    }
    showToast(`Cierre parcial del ${value}% ejecutado`, 'success');
    if (partialModal) {
      partialModal.hide();
    }
  } catch (error) {
    console.error(error);
    showToast(error.message || 'Error realizando cierre parcial', 'danger');
  } finally {
    setRowBusy(tradeId, false);
    partialTradeContext = null;
  }
}

function updateTradeInState(updatedTrade) {
  const tradeId = getTradeId(updatedTrade);
  const index = state.trades.findIndex((trade) => getTradeId(trade) === tradeId);
  if (index >= 0) {
    state.trades[index] = { ...state.trades[index], ...updatedTrade };
  } else {
    state.trades.push(updatedTrade);
  }
  renderTrades();
}

function removeTradeFromState(tradeId) {
  const index = state.trades.findIndex((trade) => getTradeId(trade) === tradeId);
  if (index >= 0) {
    state.trades.splice(index, 1);
    renderTrades();
  }
}

function connectSocket() {
  if (!window.io) {
    return;
  }
  socket = window.io('/ws');
  socket.on('connect', () => {
    setStatus('En vivo', 'success');
  });
  socket.on('disconnect', () => {
    setStatus('Desconectado', 'danger');
  });
  socket.on('trades_refresh', (trades) => {
    if (Array.isArray(trades)) {
      state.trades = trades;
      renderTrades();
    }
  });
  socket.on('trade_updated', ({ trade }) => {
    if (trade) {
      updateTradeInState(trade);
    }
  });
  socket.on('trade_closed', ({ trade }) => {
    if (trade) {
      removeTradeFromState(getTradeId(trade));
    }
  });
  socket.on('bot_status', ({ trading_active: tradingActive }) => {
    if (typeof tradingActive === 'boolean') {
      state.connectionHealthy = true;
      updateTradingControls(Boolean(tradingActive));
    }
  });
}
