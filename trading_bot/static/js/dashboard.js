const DEFAULT_REFRESH_INTERVAL = 10000;
const MAX_POINTS = 60;
const DASHBOARD_ORDER_KEY = 'dashboard:layout';
const COLLAPSED_CARDS_KEY = 'dashboard:collapsed';
const NOTIFICATION_REQUESTED_KEY = 'dashboard:notifications-requested';
const THEME_STORAGE_KEY = 'dashboard:theme';
const REFRESH_INTERVAL_STORAGE_KEY = 'dashboard:refresh-interval';
const WIDGET_VISIBILITY_KEY = 'dashboard:widgets';

const APP_CONFIG = window.APP_CONFIG || {};
const GATEWAY_BASE = APP_CONFIG.apiBase || '';
const GRAPHQL_URL = APP_CONFIG.graphqlUrl || '';
const AI_CHAT_URL = APP_CONFIG.aiChatUrl || '';
const AI_REPORT_URL = APP_CONFIG.aiReportUrl || '';

const themeOrder = ['light', 'dark', 'pastel'];

const themePalettes = {
  light: {
    '--background': '#f5f6fa',
    '--surface': 'rgba(255, 255, 255, 0.6)',
    '--surface-strong': '#ffffff',
    '--surface-elevated': '#ffffff',
    '--accent-primary': '#00bcd4',
    '--accent-secondary': '#ff6f61',
    '--accent-tertiary': '#2ecc71',
    '--text-primary': '#2c3e50',
    '--text-secondary': '#95a5a6',
    '--shadow': 'rgba(0, 0, 0, 0.1)',
    '--border-subtle': 'rgba(0, 0, 0, 0.05)',
  },
  dark: {
    '--background': '#1e272e',
    '--surface': 'rgba(47, 54, 64, 0.7)',
    '--surface-strong': 'rgba(47, 54, 64, 0.9)',
    '--surface-elevated': 'rgba(60, 66, 74, 0.92)',
    '--accent-primary': '#00bcd4',
    '--accent-secondary': '#ff6f61',
    '--accent-tertiary': '#2ecc71',
    '--text-primary': '#ecf0f1',
    '--text-secondary': 'rgba(189, 195, 199, 0.8)',
    '--shadow': 'rgba(0, 0, 0, 0.35)',
    '--border-subtle': 'rgba(255, 255, 255, 0.08)',
  },
  pastel: {
    '--background': '#f7f1ff',
    '--surface': 'rgba(255, 255, 255, 0.8)',
    '--surface-strong': 'rgba(255, 255, 255, 0.95)',
    '--surface-elevated': 'rgba(255, 255, 255, 0.95)',
    '--accent-primary': '#a855f7',
    '--accent-secondary': '#f472b6',
    '--accent-tertiary': '#60a5fa',
    '--text-primary': '#4c1d95',
    '--text-secondary': 'rgba(124, 58, 237, 0.6)',
    '--shadow': 'rgba(168, 85, 247, 0.2)',
    '--border-subtle': 'rgba(168, 85, 247, 0.15)',
  },
};

const userLocale = (() => {
  try {
    return localStorage.getItem('dashboard:locale') || navigator.language || 'es-ES';
  } catch (error) {
    return navigator.language || 'es-ES';
  }
})();

const numberFormatter = new Intl.NumberFormat(userLocale, {
  minimumFractionDigits: 2,
  maximumFractionDigits: 2,
});

const priceFormatter = new Intl.NumberFormat(userLocale, {
  minimumFractionDigits: 2,
  maximumFractionDigits: 4,
});

const quantityFormatter = new Intl.NumberFormat(userLocale, {
  minimumFractionDigits: 2,
  maximumFractionDigits: 4,
});

const percentFormatter = new Intl.NumberFormat(userLocale, {
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
  tradingActive: true,
  connectionHealthy: true,
  sessionId: null,
  flashTrades: new Set(),
  priceMemory: new Map(),
  priceDirection: new Map(),
  collapsedCards: new Set(),
  isInitialLoad: true,
  theme: 'light',
  refreshInterval: DEFAULT_REFRESH_INTERVAL,
  widgetVisibility: {},
  activeSection: 'dashboard',
  analytics: {
    trades: [],
    preferences: null,
    loading: false,
    error: null,
    symbolFilter: '',
  },
  chat: {
    history: [],
    sending: false,
  },
};

const widgetSelectors = {
  metrics_primary: '[data-widget="metrics_primary"]',
  metrics_secondary: '[data-widget="metrics_secondary"]',
  pnl: '[data-widget="pnl"]',
  symbols: '[data-widget="symbols"]',
  trades: '[data-widget="trades"]',
  history: '[data-widget="history"]',
};

const SECTION_IDS = ['dashboard', 'analytics', 'ai-assistant', 'services'];

const DEFAULT_WIDGET_VISIBILITY = {
  metrics_primary: true,
  metrics_secondary: true,
  pnl: true,
  symbols: true,
  trades: true,
  history: true,
};

let pnlChart = null;
let socket = null;
let partialModal = null;
let partialTradeContext = null;
let notificationsRequested = false;
let settingsModal = null;
let refreshTimer = null;
let analyticsFilterTimeout = null;
let analyticsInitialized = false;

function getCssVariable(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name)?.trim();
}

function hexToRgb(hex) {
  if (!hex) {
    return { r: 0, g: 0, b: 0 };
  }
  const normalized = hex.replace('#', '');
  const value = normalized.length === 3
    ? normalized
        .split('')
        .map((char) => char + char)
        .join('')
    : normalized;
  const int = Number.parseInt(value, 16);
  if (Number.isNaN(int)) {
    return { r: 0, g: 0, b: 0 };
  }
  return {
    r: (int >> 16) & 255,
    g: (int >> 8) & 255,
    b: int & 255,
  };
}

function colorWithAlpha(color, alpha) {
  if (!color) {
    return `rgba(0, 0, 0, ${alpha})`;
  }
  const trimmed = color.trim();
  if (trimmed.startsWith('rgba')) {
    const [r, g, b] = trimmed
      .slice(trimmed.indexOf('(') + 1, trimmed.lastIndexOf(')'))
      .split(',')
      .map((part) => part.trim())
      .slice(0, 3);
    return `rgba(${r}, ${g}, ${b}, ${alpha})`;
  }
  if (trimmed.startsWith('rgb')) {
    const [r, g, b] = trimmed
      .slice(trimmed.indexOf('(') + 1, trimmed.lastIndexOf(')'))
      .split(',')
      .map((part) => part.trim());
    return `rgba(${r}, ${g}, ${b}, ${alpha})`;
  }
  if (trimmed.startsWith('#')) {
    const { r, g, b } = hexToRgb(trimmed);
    return `rgba(${r}, ${g}, ${b}, ${alpha})`;
  }
  return trimmed;
}

function createPnlGradient(ctx) {
  const gradient = ctx.createLinearGradient(0, 0, 0, 320);
  const accent = getCssVariable('--accent-primary') || '#00bcd4';
  gradient.addColorStop(0, colorWithAlpha(accent, 0.4));
  gradient.addColorStop(1, colorWithAlpha(accent, 0));
  return gradient;
}

function updateChartTheme() {
  if (!pnlChart) {
    return;
  }
  const accent = getCssVariable('--accent-primary') || '#00bcd4';
  const textSecondary = getCssVariable('--text-secondary') || '#95a5a6';
  const gridColor = colorWithAlpha(textSecondary || '#95a5a6', 0.2);
  if (pnlChart.ctx) {
    pnlChart.data.datasets[0].backgroundColor = createPnlGradient(pnlChart.ctx);
  }
  pnlChart.data.datasets[0].borderColor = accent;
  pnlChart.options.scales.x.ticks.color = textSecondary;
  pnlChart.options.scales.y.ticks.color = textSecondary;
  pnlChart.options.scales.x.grid.color = gridColor;
  pnlChart.options.scales.y.grid.color = gridColor;
  pnlChart.update('none');
}

function applyThemeVariables(theme) {
  const palette = themePalettes[theme] || themePalettes.light;
  Object.entries(palette).forEach(([variable, value]) => {
    document.documentElement.style.setProperty(variable, value);
  });
  document.documentElement.dataset.theme = theme;
  if (document.body) {
    document.body.dataset.theme = theme;
  }
  state.theme = theme;
  try {
    localStorage.setItem(THEME_STORAGE_KEY, theme);
  } catch (error) {
    console.debug('No se pudo persistir el tema preferido', error);
  }
  updateThemeToggleButton(theme);
  updateChartTheme();
}

function updateThemeToggleButton(theme) {
  const toggle = document.getElementById('themeToggleBtn');
  if (!toggle) return;
  const iconMap = {
    light: 'bi-moon-stars',
    dark: 'bi-sun',
    pastel: 'bi-palette',
  };
  const nextTheme = getNextTheme(theme);
  toggle.setAttribute('data-theme', theme);
  toggle.setAttribute('aria-pressed', String(theme === 'dark'));
  const labels = {
    light: 'Cambiar a tema oscuro',
    dark: 'Cambiar a tema pastel',
    pastel: 'Cambiar a tema claro',
  };
  const icon = toggle.querySelector('i');
  if (icon) {
    icon.className = `bi ${iconMap[theme] || 'bi-moon-stars'}`;
  }
  const nextLabel = labels[theme] || 'Cambiar de tema';
  toggle.setAttribute('title', nextLabel);
  toggle.setAttribute('aria-label', nextLabel);
  toggle.dataset.nextTheme = nextTheme;
}

function toggleTheme() {
  const next = getNextTheme(state.theme);
  applyThemeVariables(next);
}

function getNextTheme(current) {
  const index = themeOrder.indexOf(current);
  if (index === -1) {
    return themeOrder[0];
  }
  return themeOrder[(index + 1) % themeOrder.length];
}

function initTheme() {
  let storedTheme = null;
  try {
    storedTheme = localStorage.getItem(THEME_STORAGE_KEY);
  } catch (error) {
    storedTheme = null;
  }
  if (!themeOrder.includes(storedTheme)) {
    storedTheme =
      window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches
        ? 'dark'
        : 'light';
  }
  applyThemeVariables(storedTheme);
}

function clearSkeleton(element) {
  if (!element) return;
  element.classList.remove('skeleton');
  element.querySelectorAll('.skeleton').forEach((node) => {
    node.classList.remove('skeleton');
  });
}

function markElementLoaded(elementOrId) {
  const element = typeof elementOrId === 'string' ? document.getElementById(elementOrId) : elementOrId;
  clearSkeleton(element);
}

try {
  notificationsRequested = localStorage.getItem(NOTIFICATION_REQUESTED_KEY) === '1';
} catch (error) {
  notificationsRequested = false;
}

function getTradeId(trade) {
  return trade.trade_id || trade.id || trade.uuid;
}

async function fetchJSON(url, options = {}) {
  const { silent = false, ...fetchOptions } = options;
  try {
    const response = await fetch(url, { cache: 'no-cache', ...fetchOptions });
    const text = await response.text();
    if (!response.ok) {
      const snippet = text ? text.slice(0, 180) : response.statusText;
      throw new Error(
        `Error ${response.status} al llamar a ${url}: ${snippet || 'sin cuerpo de respuesta'}`,
      );
    }
    if (!text) {
      return null;
    }
    try {
      return JSON.parse(text);
    } catch (parseError) {
      throw new Error(`Respuesta JSON inválida desde ${url}`);
    }
  } catch (error) {
    console.error(error);
    if (!silent) {
      showToast(error.message || 'Error de red', 'danger');
    }
    return null;
  }
}

async function readJsonResponse(response, fallbackMessage = 'Respuesta inválida del servidor') {
  const text = await response.text();
  if (!text) {
    if (response.ok) {
      return {};
    }
    throw new Error(fallbackMessage);
  }
  try {
    return JSON.parse(text);
  } catch (error) {
    const snippet = text.length > 180 ? `${text.slice(0, 180)}…` : text;
    throw new Error(`${fallbackMessage}: ${snippet}`);
  }
}

function setStatus(label, variant) {
  const badge = document.getElementById('connectionStatus');
  if (!badge) return;
  badge.textContent = label;
  const variantClass = variant ? `badge-status--${variant}` : '';
  badge.className = ['badge', 'badge-status', variantClass].filter(Boolean).join(' ');
}

function updateTradingControls(isActive) {
  const toggleBtn = document.getElementById('toggleTradeBtn');
  if (typeof isActive === 'boolean') {
    state.tradingActive = isActive;
    if (state.connectionHealthy) {
      const label = isActive ? 'En vivo' : 'Pausado';
      const variant = isActive ? 'success' : 'warning';
      setStatus(label, variant);
    }
  if (toggleBtn) {
    toggleBtn.disabled = !state.connectionHealthy;
    toggleBtn.className = `btn btn-sm ${isActive ? 'btn-secondary' : 'btn-tertiary'}`;
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
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) {
    return '—';
  }
  const prefix = numeric > 0 ? '+' : '';
  return `${prefix}${numberFormatter.format(numeric)}`;
}

function getStoredJSON(key) {
  try {
    const raw = localStorage.getItem(key);
    if (!raw) return null;
    return JSON.parse(raw);
  } catch (error) {
    console.warn(`No se pudo leer ${key} de localStorage`, error);
    return null;
  }
}

function setStoredJSON(key, value) {
  try {
    localStorage.setItem(key, JSON.stringify(value));
  } catch (error) {
    console.warn(`No se pudo almacenar ${key} en localStorage`, error);
  }
}

function loadWidgetPreferences() {
  const stored = getStoredJSON(WIDGET_VISIBILITY_KEY);
  if (stored && typeof stored === 'object') {
    state.widgetVisibility = { ...DEFAULT_WIDGET_VISIBILITY, ...stored };
  } else {
    state.widgetVisibility = { ...DEFAULT_WIDGET_VISIBILITY };
  }
  applyWidgetVisibility();
}

function applyWidgetVisibility() {
  Object.entries(widgetSelectors).forEach(([key, selector]) => {
    const isVisible = state.widgetVisibility[key] !== false;
    document.querySelectorAll(selector).forEach((element) => {
      element.classList.toggle('widget-hidden', !isVisible);
      if (!isVisible) {
        element.setAttribute('aria-hidden', 'true');
      } else {
        element.removeAttribute('aria-hidden');
      }
    });
  });
}

function setWidgetVisibility(key, visible, persist = true) {
  if (!Object.prototype.hasOwnProperty.call(DEFAULT_WIDGET_VISIBILITY, key)) {
    return;
  }
  state.widgetVisibility[key] = Boolean(visible);
  applyWidgetVisibility();
  if (persist) {
    setStoredJSON(WIDGET_VISIBILITY_KEY, state.widgetVisibility);
  }
}

function loadRefreshInterval() {
  let stored = null;
  try {
    stored = localStorage.getItem(REFRESH_INTERVAL_STORAGE_KEY);
  } catch (error) {
    stored = null;
  }
  const numeric = Number(stored);
  if (Number.isFinite(numeric) && numeric > 0) {
    state.refreshInterval = numeric;
  } else {
    state.refreshInterval = DEFAULT_REFRESH_INTERVAL;
  }
  updateRefreshSelect();
}

function setRefreshInterval(value, { persist = true, reschedule = true } = {}) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric) || numeric <= 0) {
    return state.refreshInterval;
  }
  state.refreshInterval = numeric;
  if (persist) {
    try {
      localStorage.setItem(REFRESH_INTERVAL_STORAGE_KEY, String(numeric));
    } catch (error) {
      console.warn('No se pudo guardar el intervalo preferido', error);
    }
  }
  if (reschedule) {
    scheduleNextRefresh();
  }
  updateRefreshSelect();
  return numeric;
}

function scheduleNextRefresh() {
  if (refreshTimer) {
    clearTimeout(refreshTimer);
  }
  if (state.refreshInterval <= 0) {
    return;
  }
  refreshTimer = window.setTimeout(() => {
    refreshDashboard(false);
  }, state.refreshInterval);
}

function updateRefreshSelect() {
  const select = document.getElementById('refreshIntervalSelect');
  if (select) {
    select.value = String(state.refreshInterval);
  }
}

function populateSettingsForm() {
  const form = document.getElementById('dashboardSettingsForm');
  if (!form) {
    return;
  }
  form.querySelectorAll('#widgetList input[type="checkbox"]').forEach((input) => {
    const key = input.value;
    if (!Object.prototype.hasOwnProperty.call(DEFAULT_WIDGET_VISIBILITY, key)) {
      return;
    }
    input.checked = state.widgetVisibility[key] !== false;
  });
  const intervalSelect = form.querySelector('#refreshIntervalSelect');
  if (intervalSelect) {
    intervalSelect.value = String(state.refreshInterval);
  }
}

function setupSettingsModal() {
  const modalElement = document.getElementById('dashboardSettingsModal');
  if (!modalElement || !window.bootstrap) {
    return;
  }
  if (!settingsModal) {
    settingsModal = new bootstrap.Modal(modalElement);
  }
  if (!modalElement.dataset.settingsBound) {
    modalElement.addEventListener('show.bs.modal', populateSettingsForm);
    modalElement.dataset.settingsBound = '1';
  }
}

function openSettingsModal() {
  if (!settingsModal) {
    setupSettingsModal();
  }
  populateSettingsForm();
  if (settingsModal) {
    settingsModal.show();
  }
}

function handleSettingsSubmit(event) {
  event.preventDefault();
  const form = event.currentTarget;
  const visibilityUpdate = { ...state.widgetVisibility };
  form.querySelectorAll('#widgetList input[type="checkbox"]').forEach((input) => {
    const key = input.value;
    if (Object.prototype.hasOwnProperty.call(DEFAULT_WIDGET_VISIBILITY, key)) {
      visibilityUpdate[key] = Boolean(input.checked);
    }
  });
  Object.entries(DEFAULT_WIDGET_VISIBILITY).forEach(([key]) => {
    const visible = visibilityUpdate[key] !== false;
    setWidgetVisibility(key, visible, false);
  });
  setStoredJSON(WIDGET_VISIBILITY_KEY, state.widgetVisibility);
  const intervalSelect = form.querySelector('#refreshIntervalSelect');
  if (intervalSelect) {
    setRefreshInterval(intervalSelect.value, { persist: true, reschedule: true });
  }
  applyWidgetVisibility();
  showToast('Preferencias actualizadas', 'success');
  if (settingsModal) {
    settingsModal.hide();
  }
}

function openExternalService(key) {
  const base = GATEWAY_BASE || '';
  const serviceMap = {
    orders: base ? `${base}/orders` : null,
    docs: base ? `${base}/docs` : null,
    analytics: GRAPHQL_URL || (base ? `${base}/graphql` : null),
  };
  const url = serviceMap[key];
  if (!url) {
    showToast('No se pudo determinar la URL del servicio solicitado.', 'warning');
    return;
  }
  const win = window.open(url, '_blank', 'noopener,noreferrer');
  if (win) {
    win.opener = null;
  } else {
    showToast('El navegador bloqueó la apertura de la pestaña.', 'warning');
  }
}

function handleServiceOpen(event) {
  const key = event.currentTarget.dataset.openService;
  openExternalService(key);
}

function applyDashboardLayout() {
  const grid = document.getElementById('dashboardGrid');
  if (!grid) return;

  const savedOrder = getStoredJSON(DASHBOARD_ORDER_KEY);
  if (Array.isArray(savedOrder) && savedOrder.length) {
    const elements = new Map(Array.from(grid.children).map((child) => [child.dataset.id, child]));
    savedOrder.forEach((id) => {
      const element = elements.get(id);
      if (element) {
        grid.appendChild(element);
        elements.delete(id);
      }
    });
  }

  if (window.Sortable && !grid.sortableInstance) {
    grid.sortableInstance = new Sortable(grid, {
      animation: 150,
      handle: '.card-header',
      onEnd: () => {
        const order = Array.from(grid.children)
          .map((child) => child.dataset.id)
          .filter(Boolean);
        setStoredJSON(DASHBOARD_ORDER_KEY, order);
      },
    });
  }
}

function persistCollapsedCards() {
  setStoredJSON(COLLAPSED_CARDS_KEY, Array.from(state.collapsedCards));
}

function loadCollapsedCards() {
  const stored = getStoredJSON(COLLAPSED_CARDS_KEY);
  if (Array.isArray(stored)) {
    state.collapsedCards = new Set(stored);
  } else {
    state.collapsedCards = new Set();
  }
}

function updateToggleButtons(cardId, collapsed) {
  document.querySelectorAll(`button[data-card-toggle="${cardId}"]`).forEach((button) => {
    button.setAttribute('aria-expanded', String(!collapsed));
    button.classList.remove('btn-outline-secondary', 'btn-outline-primary', 'btn-primary', 'btn-ghost');
    button.classList.add('btn');
    button.classList.add('btn-sm');
    button.classList.add(collapsed ? 'btn-primary' : 'btn-ghost');
    const icon = collapsed ? 'bi-arrows-expand' : 'bi-arrows-collapse';
    button.innerHTML = `<i class="bi ${icon}"></i>`;
    const actionLabel = collapsed ? 'Expandir tarjeta' : 'Colapsar tarjeta';
    button.setAttribute('aria-label', actionLabel);
    button.setAttribute('title', actionLabel);
  });
}

function toggleCardCollapse(card, collapsed, persist = true) {
  if (!card) return;
  const cardId = card.dataset.cardId;
  if (!cardId) return;
  card.classList.toggle('is-collapsed', collapsed);
  if (collapsed) {
    state.collapsedCards.add(cardId);
  } else {
    state.collapsedCards.delete(cardId);
  }
  updateToggleButtons(cardId, collapsed);
  if (persist) {
    persistCollapsedCards();
  }
}

function setupCollapsibles() {
  loadCollapsedCards();
  document.querySelectorAll('.collapsible-card').forEach((card) => {
    const cardId = card.dataset.cardId;
    if (!cardId) return;
    const collapsed = state.collapsedCards.has(cardId);
    toggleCardCollapse(card, collapsed, false);
  });

  document.querySelectorAll('button[data-card-toggle]').forEach((button) => {
    button.addEventListener('click', () => {
      const cardId = button.dataset.cardToggle;
      const card = document.querySelector(`.collapsible-card[data-card-id="${cardId}"]`);
      if (!card) return;
      const collapsed = !card.classList.contains('is-collapsed');
      toggleCardCollapse(card, collapsed, true);
    });
  });
}

function updateNavLinks(activeSection) {
  document.querySelectorAll('[data-section-target]').forEach((link) => {
    const target = link.getAttribute('data-section-target');
    const isActive = target === activeSection;
    link.classList.toggle('active', isActive);
    if (isActive) {
      link.setAttribute('aria-current', 'page');
    } else {
      link.removeAttribute('aria-current');
    }
  });
}

function focusAiInput() {
  const input = document.getElementById('aiMessage');
  if (input) {
    window.requestAnimationFrame(() => {
      input.focus();
    });
  }
}

function switchSection(sectionId) {
  const normalized = SECTION_IDS.includes(sectionId) ? sectionId : 'dashboard';
  state.activeSection = normalized;
  document.querySelectorAll('.app-section').forEach((section) => {
    const isTarget = section.dataset.section === normalized;
    section.classList.toggle('is-active', isTarget);
    if (isTarget) {
      section.classList.remove('d-none');
      section.removeAttribute('hidden');
    } else {
      section.classList.add('d-none');
      section.setAttribute('hidden', 'hidden');
    }
  });
  updateNavLinks(normalized);
  if (normalized === 'analytics') {
    ensureAnalyticsData();
  }
  if (normalized === 'ai-assistant') {
    focusAiInput();
  }
}

function registerServiceWorker() {
  if (!('serviceWorker' in navigator)) {
    return;
  }
  window.addEventListener('load', () => {
    navigator.serviceWorker
      .register('/service-worker.js')
      .catch((error) => console.error('No se pudo registrar el service worker', error));
  });
}

function requestNotificationPermission() {
  if (!('Notification' in window)) {
    notificationsRequested = true;
    return;
  }
  if (Notification.permission === 'granted' || Notification.permission === 'denied') {
    notificationsRequested = true;
    try {
      localStorage.setItem(NOTIFICATION_REQUESTED_KEY, '1');
    } catch (error) {
      // ignore storage errors
    }
    return;
  }
  if (notificationsRequested) {
    return;
  }
  notificationsRequested = true;
  Notification.requestPermission()
    .catch((error) => {
      console.warn('No se pudo solicitar permiso de notificaciones', error);
    })
    .finally(() => {
      try {
        localStorage.setItem(NOTIFICATION_REQUESTED_KEY, '1');
      } catch (error) {
        // ignore storage errors
      }
    });
}

function scheduleNotificationRequest() {
  if (!('Notification' in window)) {
    return;
  }
  if (Notification.permission !== 'default') {
    if (!notificationsRequested) {
      notificationsRequested = true;
      try {
        localStorage.setItem(NOTIFICATION_REQUESTED_KEY, '1');
      } catch (error) {
        // ignore storage errors
      }
    }
    return;
  }
  if (notificationsRequested) {
    return;
  }
  setTimeout(() => {
    requestNotificationPermission();
  }, 1200);
}

function registerHotkeys() {
  document.addEventListener('keydown', (event) => {
    if (!(event.ctrlKey || event.metaKey)) {
      return;
    }
    const target = event.target;
    if (
      target &&
      (target.isContentEditable ||
        ['input', 'textarea', 'select'].includes(String(target.tagName).toLowerCase()))
    ) {
      return;
    }
    const key = String(event.key || '').toLowerCase();
    if (key === 'p') {
      event.preventDefault();
      const toggleBtn = document.getElementById('toggleTradeBtn');
      if (toggleBtn) {
        toggleBtn.click();
      }
    } else if (key === 'r') {
      event.preventDefault();
      refreshDashboard(true);
    }
  });
}

function formatDate(value) {
  if (!value) return '—';
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return value;
  }
  return date.toLocaleString(userLocale);
}

function ensureChart() {
  if (pnlChart) {
    return pnlChart;
  }
  const ctx = document.getElementById('pnlChart');
  if (!ctx) {
    return null;
  }
  const accent = getCssVariable('--accent-primary') || '#00bcd4';
  const textSecondary = getCssVariable('--text-secondary') || '#95a5a6';
  pnlChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: [],
      datasets: [
        {
          label: 'PnL total',
          data: [],
          borderColor: accent,
          backgroundColor: createPnlGradient(ctx),
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
            color: textSecondary,
          },
          grid: {
            color: colorWithAlpha(textSecondary, 0.2),
          },
        },
        y: {
          ticks: {
            color: textSecondary,
            callback(value) {
              return formatNumber(value);
            },
          },
          grid: {
            color: colorWithAlpha(textSecondary, 0.2),
          },
        },
      },
    },
  });
  updateChartTheme();
  return pnlChart;
}

function resetPnlSeries() {
  state.pnlSeries = [];
  const chart = ensureChart();
  if (!chart) return;
  chart.data.labels = [];
  chart.data.datasets[0].data = [];
  chart.update('none');
}

function handleSession(summary) {
  if (!summary) return;
  const newSessionId = summary.session_id || null;
  const openPositions = Number(summary.total_positions || 0);
  const sessionChanged = newSessionId && newSessionId !== state.sessionId;
  if (openPositions === 0 && (sessionChanged || state.sessionId === null)) {
    resetPnlSeries();
  }
  state.sessionId = newSessionId;
}

function updatePnlSeries(summary) {
  if (!summary) return;
  const chart = ensureChart();
  if (!chart) return;

  const timestampLabel = new Date(summary.generated_at || Date.now()).toLocaleTimeString(
    userLocale,
  );
  const realized = Number(summary.realized_pnl_total ?? summary.realized_pnl ?? 0);
  const unrealized = Number(summary.unrealized_pnl ?? 0);
  const currentValue = Number(summary.total_pnl ?? realized + unrealized) || 0;

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
  const positionsEl = document.getElementById('metricPositions');
  if (positionsEl) {
    positionsEl.textContent = Number(summary.total_positions ?? 0);
    markElementLoaded(positionsEl);
  }

  const pnlEl = document.getElementById('metricPnL');
  if (pnlEl) {
    pnlEl.textContent = formatPnL(summary.unrealized_pnl);
    markElementLoaded(pnlEl);
  }

  const totalExposure = Number(summary.total_exposure ?? 0);
  const totalInvested = Number(summary.total_invested ?? 0);
  const exposureEl = document.getElementById('metricExposure');
  if (exposureEl) {
    exposureEl.textContent = formatNumber(totalExposure);
    exposureEl.title = `Capital invertido: ${formatNumber(totalInvested)}`;
    markElementLoaded(exposureEl);
  }

  const winRateEl = document.getElementById('metricWinRate');
  if (winRateEl) {
    const hasValue = typeof summary.win_rate === 'number' && !Number.isNaN(summary.win_rate);
    if (hasValue) {
      winRateEl.textContent = percentFormatter.format(summary.win_rate);
      winRateEl.classList.remove('text-muted');
    } else {
      winRateEl.textContent = '—';
      winRateEl.classList.add('text-muted');
    }
    const samples = Number(summary.win_rate_samples || 0);
    const winners = Number(summary.win_rate_winners || 0);
    const losers = Number(summary.win_rate_losers || 0);
    const tooltip = samples
      ? `${samples} operaciones (${winners} ganadas / ${losers} perdidas)`
      : 'Sin operaciones suficientes para calcular la métrica';
    winRateEl.setAttribute('title', tooltip);
    markElementLoaded(winRateEl);
  }

  const realizedBalance = document.getElementById('metricRealizedBalance');
  if (realizedBalance) {
    realizedBalance.textContent = formatNumber(summary.realized_balance);
    markElementLoaded(realizedBalance);
  }
  const realizedPnL = document.getElementById('metricRealizedPnL');
  if (realizedPnL) {
    const realizedPnLValue = Number(summary.realized_pnl_total ?? summary.realized_pnl ?? 0);
    realizedPnL.textContent = formatPnL(realizedPnLValue);
    markElementLoaded(realizedPnL);
  }
  const lastUpdated = document.getElementById('lastUpdated');
  if (lastUpdated) {
    const generatedAt = summary.generated_at ? new Date(summary.generated_at) : new Date();
    lastUpdated.textContent = generatedAt.toLocaleTimeString(userLocale);
  }

  const tradingMode = String(summary.trading_mode || '').toLowerCase();
  const isDemo = Boolean(tradingMode && tradingMode !== 'live');
  document.body.classList.toggle('demo-mode', isDemo);
  const modeBadge = document.getElementById('modeBadge');
  if (modeBadge) {
    if (isDemo) {
      const label = tradingMode === 'paper' ? 'Demo' : tradingMode.toUpperCase();
      modeBadge.textContent = label;
      modeBadge.classList.remove('d-none');
    } else {
      modeBadge.classList.add('d-none');
    }
  }

  const list = document.getElementById('symbolBreakdown');
  if (!list) return;
  list.innerHTML = '';
  if (!summary.per_symbol || summary.per_symbol.length === 0) {
    list.innerHTML = '<li class="list-group-item">Sin posiciones abiertas.</li>';
    clearSkeleton(list);
    return;
  }

  summary.per_symbol.forEach((item) => {
    const pnlClass = item.unrealized_pnl >= 0 ? 'pnl-positive' : 'pnl-negative';
    const quantity = formatNumber(item.quantity, quantityFormatter);
    const exposure = formatNumber(item.exposure, numberFormatter);
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
        <span><strong>Cantidad:</strong> ${quantity}</span>
        <span><strong>Exposición:</strong> ${exposure}</span>
        <span><strong>Invertido:</strong> ${invested}</span>
        <span class="${pnlClass}"><strong>PnL:</strong> ${pnl}</span>
      </div>`;
    list.appendChild(li);
  });
  clearSkeleton(list);
}

const ANALYTICS_TRADES_QUERY = `
  query AnalyticsTrades($symbol: String) {
    trades(symbol: $symbol) {
      id
      symbol
      side
      quantity
      pnl
      openTime
      closeTime
    }
  }
`;

const ANALYTICS_SETTINGS_QUERY = `
  query AnalyticsUserSettings($userId: Int!) {
    userSettings(userId: $userId) {
      locale
      theme
      maxRisk
      notificationsEnabled
    }
  }
`;

async function postGraphQL(query, variables = {}) {
  if (!GRAPHQL_URL) {
    throw new Error('No se configuró el endpoint de GraphQL.');
  }
  const response = await fetch(GRAPHQL_URL, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query, variables }),
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    const message = payload?.errors?.[0]?.message || response.statusText || 'Error en la petición GraphQL';
    throw new Error(message);
  }
  if (payload.errors && payload.errors.length) {
    throw new Error(payload.errors[0]?.message || 'Error en la consulta GraphQL');
  }
  return payload.data || {};
}

function setAnalyticsStatus(message, variant = 'muted') {
  const status = document.getElementById('analyticsStatus');
  if (!status) return;
  const classMap = {
    success: 'text-success',
    info: 'text-info',
    warning: 'text-warning',
    danger: 'text-danger',
    muted: 'text-muted',
  };
  status.className = `small ${classMap[variant] || 'text-muted'}`;
  status.textContent = message;
}

function getAnalyticsUserId() {
  const input = document.getElementById('analyticsUserId');
  if (!input) {
    return 1;
  }
  const numeric = Number(input.value || input.getAttribute('value') || 1);
  return Number.isFinite(numeric) && numeric > 0 ? Math.floor(numeric) : 1;
}

function normalizeAnalyticsTrade(trade) {
  if (!trade) {
    return null;
  }
  return {
    id: trade.id,
    symbol: trade.symbol,
    side: trade.side,
    quantity: trade.quantity,
    pnl: trade.pnl,
    openTime: trade.openTime || trade.open_time,
    closeTime: trade.closeTime || trade.close_time,
  };
}

function renderAnalyticsTrades() {
  const tbody = document.getElementById('analyticsTradesBody');
  if (!tbody) return;

  const trades = state.analytics.trades;
  if (!Array.isArray(trades) || trades.length === 0) {
    tbody.innerHTML = '<tr><td colspan="6" class="text-muted py-4">No se encontraron operaciones para los filtros seleccionados.</td></tr>';
    return;
  }

  const rows = trades
    .slice(0, 50)
    .map((raw) => {
      const trade = normalizeAnalyticsTrade(raw) || {};
      const side = String(trade.side || '').toUpperCase();
      const pnlValue = Number(trade.pnl);
      const pnlClass = Number.isFinite(pnlValue) && pnlValue >= 0 ? 'pnl-positive' : 'pnl-negative';
      const sideClass = side === 'BUY' ? 'text-success' : side === 'SELL' ? 'text-danger' : 'text-muted';
      const openTime = trade.openTime ? formatDate(trade.openTime) : '—';
      const closeTime = trade.closeTime ? formatDate(trade.closeTime) : '—';
      return `
        <tr>
          <td>${trade.symbol || '—'}</td>
          <td class="${sideClass}">${side || '—'}</td>
          <td>${formatNumber(trade.quantity, quantityFormatter)}</td>
          <td class="${pnlClass}">${formatPnL(trade.pnl)}</td>
          <td>${openTime}</td>
          <td>${closeTime}</td>
        </tr>`;
    })
    .join('');

  tbody.innerHTML = rows;
}

function renderAnalyticsPreferences() {
  const container = document.getElementById('analyticsPreferences');
  if (!container) return;
  const preferences = state.analytics.preferences;
  const localeField = container.querySelector('[data-field="locale"]');
  const themeField = container.querySelector('[data-field="theme"]');
  const maxRiskField = container.querySelector('[data-field="max_risk"]');
  const notificationsField = container.querySelector('[data-field="notifications_enabled"]');
  if (!preferences) {
    if (localeField) localeField.textContent = '—';
    if (themeField) themeField.textContent = '—';
    if (maxRiskField) maxRiskField.textContent = '—';
    if (notificationsField) notificationsField.textContent = '—';
    return;
  }
  const locale = preferences.locale || '—';
  const theme = preferences.theme || preferences.preferredTheme || '—';
  const maxRisk = Number(preferences.maxRisk ?? preferences.max_risk);
  const notificationsEnabled = preferences.notificationsEnabled ?? preferences.notifications_enabled;
  if (localeField) localeField.textContent = locale;
  if (themeField) themeField.textContent = theme;
  if (maxRiskField) {
    maxRiskField.textContent = Number.isFinite(maxRisk) ? formatNumber(maxRisk) : '—';
  }
  if (notificationsField) {
    if (typeof notificationsEnabled === 'boolean') {
      notificationsField.textContent = notificationsEnabled ? 'Activadas' : 'Desactivadas';
    } else {
      notificationsField.textContent = '—';
    }
  }
}

async function loadAnalyticsTrades(symbol = state.analytics.symbolFilter || '') {
  if (!GRAPHQL_URL) {
    setAnalyticsStatus('Configura la URL del gateway para consultar analítica.', 'warning');
    state.analytics.trades = [];
    renderAnalyticsTrades();
    state.analytics.loading = false;
    return;
  }
  state.analytics.loading = true;
  state.analytics.symbolFilter = symbol.trim();
  try {
    setAnalyticsStatus('Consultando operaciones…', 'info');
    const variables = state.analytics.symbolFilter ? { symbol: state.analytics.symbolFilter } : {};
    const data = await postGraphQL(ANALYTICS_TRADES_QUERY, variables);
    const trades = data?.trades || data?.Trades || [];
    state.analytics.trades = Array.isArray(trades) ? trades : [];
    state.analytics.error = null;
    if (state.analytics.trades.length === 0) {
      setAnalyticsStatus('No se encontraron operaciones para los filtros seleccionados.', 'warning');
    } else {
      setAnalyticsStatus(`Se muestran ${state.analytics.trades.length} operaciones.`, 'success');
    }
  } catch (error) {
    console.error(error);
    state.analytics.error = error;
    state.analytics.trades = [];
    setAnalyticsStatus(error.message || 'Error consultando analítica.', 'danger');
  } finally {
    state.analytics.loading = false;
    renderAnalyticsTrades();
  }
}

async function loadAnalyticsPreferences(userId = getAnalyticsUserId()) {
  if (!GRAPHQL_URL) {
    return;
  }
  try {
    const data = await postGraphQL(ANALYTICS_SETTINGS_QUERY, { userId });
    const preferences = data?.userSettings || data?.user_settings || null;
    state.analytics.preferences = preferences;
    renderAnalyticsPreferences();
  } catch (error) {
    console.error(error);
    showToast(error.message || 'No se pudieron obtener las preferencias.', 'warning');
  }
}

async function loadAnalyticsData({ force = false } = {}) {
  if (state.analytics.loading && !force) {
    return;
  }
  await Promise.all([loadAnalyticsTrades(state.analytics.symbolFilter), loadAnalyticsPreferences()]);
}

function ensureAnalyticsData(force = false) {
  if (!GRAPHQL_URL) {
    setAnalyticsStatus('Configura la URL del gateway para consultar analítica.', 'warning');
    return;
  }
  if (force) {
    loadAnalyticsData({ force: true });
    return;
  }
  if (!analyticsInitialized) {
    analyticsInitialized = true;
    loadAnalyticsData({ force: true });
  }
}

function renderTrades() {
  const tbody = document.getElementById('tradesTableBody');
  if (!tbody) return;

  const filter = state.symbolFilter.trim().toLowerCase();
  const trades = filter
    ? state.trades.filter((trade) => String(trade.symbol || '').toLowerCase().includes(filter))
    : state.trades;

  if (!trades || trades.length === 0) {
    tbody.innerHTML = '<tr><td colspan="11" class="text-center text-muted py-4">No hay operaciones abiertas.</td></tr>';
    state.priceDirection.clear();
    clearSkeleton(tbody);
    return;
  }

  state.priceDirection.clear();
  const visibleIds = new Set();

  const rows = trades
    .map((trade) => {
      const tradeId = getTradeId(trade);
      visibleIds.add(tradeId);
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
      const numericPrice = Number(trade.current_price);
      let priceChangeClass = '';
      if (Number.isFinite(numericPrice)) {
        const previous = state.priceMemory.has(tradeId)
          ? Number(state.priceMemory.get(tradeId))
          : undefined;
        if (Number.isFinite(previous) && numericPrice !== previous) {
          priceChangeClass = numericPrice > previous ? 'price-up' : 'price-down';
          state.priceDirection.set(tradeId, priceChangeClass);
        }
        state.priceMemory.set(tradeId, numericPrice);
      } else {
        state.priceMemory.delete(tradeId);
      }
      return `
        <tr data-trade-id="${tradeId}">
          <td><span class="symbol-badge"><i class="bi bi-currency-bitcoin"></i>${trade.symbol}</span></td>
          <td class="trade-side ${sideClass}">${side.toUpperCase()}</td>
          <td>${quantity}</td>
          <td>${entryPrice}</td>
          <td class="${priceChangeClass}" data-field="current_price">${currentPrice}</td>
          <td>${takeProfit}</td>
          <td>${stopLoss}</td>
          <td class="${pnlClass}">${pnl}</td>
          <td>${realized}</td>
          <td>${trade.open_time ? formatDate(trade.open_time) : '—'}</td>
          <td>
            <div class="btn-group" role="group">
              <button class="btn btn-sm btn-secondary" type="button" data-trade-id="${tradeId}" data-action="close">Cerrar</button>
              <button class="btn btn-sm btn-tertiary" type="button" data-trade-id="${tradeId}" data-action="partial" data-remaining="${remaining}">Cerrar parcial</button>
            </div>
          </td>
        </tr>`;
    })
    .join('');

  tbody.innerHTML = rows;
  clearSkeleton(tbody);
  attachTradeRowEvents();
  applyTradeHighlights();

  Array.from(state.priceMemory.keys()).forEach((id) => {
    if (!visibleIds.has(id)) {
      state.priceMemory.delete(id);
    }
  });
}

function applyTradeHighlights() {
  const tbody = document.getElementById('tradesTableBody');
  if (!tbody) return;

  if (state.flashTrades.size > 0) {
    state.flashTrades.forEach((tradeId) => {
      const row = tbody.querySelector(`tr[data-trade-id="${tradeId}"]`);
      if (!row) return;
      row.classList.add('flash-update');
      setTimeout(() => {
        row.classList.remove('flash-update');
      }, 1000);
    });
    state.flashTrades.clear();
  }

  if (state.priceDirection.size > 0) {
    state.priceDirection.forEach((className, tradeId) => {
      if (!className) return;
      const cell = tbody.querySelector(
        `tr[data-trade-id="${tradeId}"] td[data-field="current_price"]`,
      );
      if (!cell) return;
      cell.classList.add(className);
      setTimeout(() => {
        cell.classList.remove('price-up');
        cell.classList.remove('price-down');
      }, 700);
    });
    state.priceDirection.clear();
  }
}

function notifyTradeClosed(trade) {
  if (!trade || !('Notification' in window) || Notification.permission !== 'granted') {
    return;
  }
  const symbol = String(trade.symbol || trade.pair || 'Operación');
  const pnlValue = trade.realized_pnl ?? trade.profit ?? trade.pnl_unrealized ?? 0;
  const body = `PnL: ${formatPnL(pnlValue)}`;
  try {
    new Notification(`Operación cerrada · ${symbol}`, { body });
  } catch (error) {
    console.warn('No se pudo mostrar la notificación', error);
  }
}

function renderHistory() {
  const container = document.getElementById('historyList');
  if (!container) return;

  if (!state.history || state.history.length === 0) {
    container.innerHTML = '<div class="text-center text-muted">Sin operaciones cerradas recientes.</div>';
    clearSkeleton(container);
    return;
  }

  const items = state.history
    .map((trade) => {
      const side = String(trade.side || '').toLowerCase();
      const sideLabel = side === 'buy' ? 'Compra' : 'Venta';
      const pnlClass = Number(trade.profit) >= 0 ? 'pnl-positive' : 'pnl-negative';
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
  clearSkeleton(container);
}

function updateAiStatus(label, variant = 'light') {
  const badge = document.getElementById('aiStatus');
  if (!badge) return;
  const textClassMap = {
    light: 'text-dark',
    info: 'text-dark',
    warning: 'text-dark',
    success: 'text-white',
    primary: 'text-white',
    danger: 'text-white',
  };
  const bgClass = variant ? `bg-${variant}` : 'bg-light';
  const textClass = textClassMap[variant] || 'text-dark';
  badge.className = `badge ${bgClass} ${textClass}`.trim();
  badge.textContent = label;
}

function appendChatMessage(role, content) {
  const container = document.getElementById('aiChatMessages');
  if (!container) return;
  const normalized = role === 'assistant' || role === 'system' ? role : 'user';
  const messageEl = document.createElement('div');
  messageEl.className = `chat-message chat-message--${normalized}`;
  messageEl.textContent = content;
  container.appendChild(messageEl);
  container.scrollTop = container.scrollHeight;
  if (!Array.isArray(state.chat.history)) {
    state.chat.history = [];
  }
  state.chat.history.push({ role: normalized, content });
}

function renderAiReport(text) {
  const container = document.getElementById('aiReportOutput');
  if (container) {
    container.textContent = text;
  }
}

function buildAiSummary() {
  const summary = state.summary || {};
  const pnl = Number(summary.total_pnl ?? summary.unrealized_pnl ?? 0);
  const winRate = Number(summary.win_rate ?? 0);
  const openPositions = Number(summary.total_positions ?? 0);
  const risk = Number(summary.total_exposure ?? 0);
  return {
    pnl,
    win_rate: winRate,
    open_positions: openPositions,
    risk,
  };
}

async function handleAiChatSubmit(event) {
  event.preventDefault();
  if (state.chat.sending) {
    return;
  }
  const form = event.currentTarget;
  const textarea = document.getElementById('aiMessage');
  if (!textarea) {
    return;
  }
  const message = textarea.value.trim();
  if (!message) {
    showToast('Escribe un mensaje para enviarlo al asistente.', 'warning');
    return;
  }
  appendChatMessage('user', message);
  textarea.value = '';
  const submitBtn = form.querySelector('button[type="submit"]');
  if (submitBtn) {
    submitBtn.disabled = true;
  }
  state.chat.sending = true;
  updateAiStatus('Pensando…', 'info');
  try {
    if (!AI_CHAT_URL) {
      throw new Error('No se configuró la ruta del asistente IA en el gateway.');
    }
    const response = await fetch(AI_CHAT_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        message,
        summary: buildAiSummary(),
      }),
    });
    const data = await readJsonResponse(response, 'No se pudo interpretar la respuesta del asistente');
    if (!response.ok) {
      throw new Error(data?.detail || data?.error || 'El asistente devolvió un error.');
    }
    const answer = data?.answer || 'No recibimos respuesta del asistente.';
    appendChatMessage('assistant', answer);
    updateAiStatus('Listo', 'light');
  } catch (error) {
    console.error(error);
    updateAiStatus('Error', 'danger');
    showToast(error.message || 'No se pudo comunicar con el asistente IA.', 'danger');
  } finally {
    state.chat.sending = false;
    if (submitBtn) {
      submitBtn.disabled = false;
    }
  }
}

async function handleAiQuickReport() {
  const button = document.getElementById('aiQuickReportBtn');
  if (!AI_REPORT_URL) {
    showToast('No se configuró la ruta de informes IA.', 'warning');
    return;
  }
  if (!state.summary) {
    showToast('Aún no hay métricas suficientes para generar un informe.', 'warning');
    return;
  }
  if (button) {
    button.disabled = true;
  }
  updateAiStatus('Generando informe…', 'info');
  try {
    const payload = {
      pnl_total: Number(state.summary.total_pnl ?? state.summary.unrealized_pnl ?? 0),
      win_rate: Number(state.summary.win_rate ?? 0),
      open_positions: Number(state.summary.total_positions ?? 0),
      risk: Number(state.summary.total_exposure ?? 0),
    };
    const response = await fetch(AI_REPORT_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const data = await readJsonResponse(response, 'No se pudo generar el informe');
    if (!response.ok) {
      throw new Error(data?.detail || data?.error || 'No se pudo generar el informe');
    }
    const report = data?.report || 'No se recibieron datos del informe.';
    renderAiReport(report);
    updateAiStatus('Informe listo', 'success');
    showToast('Informe generado correctamente.', 'success');
  } catch (error) {
    console.error(error);
    updateAiStatus('Error', 'danger');
    showToast(error.message || 'Error al generar el informe IA.', 'danger');
  } finally {
    if (button) {
      button.disabled = false;
    }
  }
}

async function refreshDashboard(manual = false) {
  try {
    if (manual) {
      setStatus('Actualizando…', 'info');
    }
    clearAlert();

    const trades = (await fetchJSON('/api/trades', { silent: !manual })) ?? [];
    const summarySilent = manual ? false : state.summary !== null;
    const summary = await fetchJSON('/api/summary', { silent: summarySilent });
    const history = (await fetchJSON('/api/history?limit=50', { silent: true })) ?? [];

    state.trades = Array.isArray(trades) ? trades : [];
    state.summary = summary;
    state.history = Array.isArray(history) ? history : [];
    state.connectionHealthy = Boolean(summary);

    const tradingActive =
      summary && Object.prototype.hasOwnProperty.call(summary, 'trading_active')
        ? Boolean(summary.trading_active)
        : state.tradingActive;
    updateTradingControls(tradingActive);

    handleSession(summary);

    renderTrades();
    renderHistory();
    renderSummary(summary);
    updatePnlSeries(summary);

    if (!state.connectionHealthy) {
      showAlert(
        'No se pudieron obtener las métricas principales. Mostramos los últimos datos disponibles.',
        'warning',
      );
      setStatus('Datos incompletos', 'warning');
    }
    state.isInitialLoad = false;
  } catch (error) {
    console.error(error);
    showAlert('No se pudieron sincronizar los datos del bot. Reintentaremos automáticamente.', 'danger');
    state.connectionHealthy = false;
    updateTradingControls(state.tradingActive);
    setStatus('Desconectado', 'danger');
  } finally {
    scheduleNextRefresh();
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

  const themeToggle = document.getElementById('themeToggleBtn');
  if (themeToggle) {
    themeToggle.addEventListener('click', toggleTheme);
  }

  document.querySelectorAll('[data-section-target]').forEach((element) => {
    element.addEventListener('click', (event) => {
      const target = event.currentTarget.getAttribute('data-section-target');
      if (!target) {
        return;
      }
      event.preventDefault();
      switchSection(target);
    });
  });

  const openConfigBtn = document.getElementById('openConfigBtn');
  if (openConfigBtn) {
    openConfigBtn.addEventListener('click', openSettingsModal);
  }

  const settingsForm = document.getElementById('dashboardSettingsForm');
  if (settingsForm) {
    settingsForm.addEventListener('submit', handleSettingsSubmit);
  }

  const analyticsRefreshBtn = document.getElementById('analyticsRefreshBtn');
  if (analyticsRefreshBtn) {
    analyticsRefreshBtn.addEventListener('click', () => {
      if (analyticsFilterTimeout) {
        clearTimeout(analyticsFilterTimeout);
        analyticsFilterTimeout = null;
      }
      ensureAnalyticsData(true);
    });
  }

  const analyticsSymbolFilter = document.getElementById('analyticsSymbolFilter');
  if (analyticsSymbolFilter) {
    analyticsSymbolFilter.addEventListener('input', (event) => {
      const { value } = event.target;
      if (analyticsFilterTimeout) {
        clearTimeout(analyticsFilterTimeout);
      }
      analyticsFilterTimeout = window.setTimeout(() => {
        loadAnalyticsTrades(value.trim());
      }, 400);
    });
  }

  const analyticsSettingsForm = document.getElementById('analyticsSettingsForm');
  if (analyticsSettingsForm) {
    analyticsSettingsForm.addEventListener('submit', (event) => {
      event.preventDefault();
      const userIdInput = document.getElementById('analyticsUserId');
      const numeric = Number(userIdInput?.value || 1);
      const userId = Number.isFinite(numeric) && numeric > 0 ? Math.floor(numeric) : 1;
      loadAnalyticsPreferences(userId);
    });
  }

  const aiChatForm = document.getElementById('aiChatForm');
  if (aiChatForm) {
    aiChatForm.addEventListener('submit', handleAiChatSubmit);
  }

  const aiQuickReportBtn = document.getElementById('aiQuickReportBtn');
  if (aiQuickReportBtn) {
    aiQuickReportBtn.addEventListener('click', handleAiQuickReport);
  }

  document.querySelectorAll('[data-open-service]').forEach((button) => {
    button.addEventListener('click', handleServiceOpen);
  });

  const toggleBtn = document.getElementById('toggleTradeBtn');
  if (toggleBtn) {
    toggleBtn.addEventListener('click', async () => {
      toggleBtn.disabled = true;
      try {
        const response = await fetch('/api/toggle-trading', { method: 'POST' });
        const data = await readJsonResponse(response, 'No se pudo cambiar el estado del bot');
        if (!response.ok || !data.ok) {
          throw new Error(data.error || 'No se pudo cambiar el estado del bot');
        }
        state.connectionHealthy = true;
        updateTradingControls(Boolean(data.trading_active));
        const persistedLabel = data.persisted ? ' (estado guardado)' : '';
        showToast(
          `Trading ${data.trading_active ? 'reanudado' : 'pausado'} correctamente${persistedLabel}`,
          'info',
        );
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
}

function initialize() {
  initTheme();
  loadRefreshInterval();
  loadWidgetPreferences();
  setupSettingsModal();
  registerServiceWorker();
  switchSection(state.activeSection);
  applyDashboardLayout();
  setupCollapsibles();
  registerHotkeys();
  ensureChart();
  attachEvents();
  if (AI_CHAT_URL) {
    updateAiStatus('Listo', 'light');
  } else {
    updateAiStatus('Sin gateway IA', 'warning');
  }
  setStatus('Sincronizando…', 'warning');
  refreshDashboard();
  connectSocket();
  scheduleNotificationRequest();
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
        const data = await readJsonResponse(response, 'No se pudo cerrar la operación');
        if (!response.ok || !data.ok) {
          throw new Error(data.error || 'No se pudo cerrar la operación');
        }
        state.connectionHealthy = true;
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
        info.textContent = `Trade ${tradeId} — tamaño restante: ${formatNumber(
          Number(remaining),
          quantityFormatter,
        )}. Introduce un porcentaje entre 1 y 100.`;
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
    showToast('Porcentaje inválido (usa un valor entre 1 y 100)', 'warning');
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
    const data = await readJsonResponse(response, 'No se pudo cerrar parcialmente');
    if (!response.ok || !data.ok) {
      throw new Error(data.error || 'No se pudo cerrar parcialmente');
    }
    const formattedPercent = percentFormatter.format(value / 100);
    showToast(`Cierre parcial de ${formattedPercent} ejecutado`, 'success');
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
  state.flashTrades.add(tradeId);
  renderTrades();
}

function removeTradeFromState(tradeId) {
  const index = state.trades.findIndex((trade) => getTradeId(trade) === tradeId);
  if (index >= 0) {
    state.trades.splice(index, 1);
    state.priceMemory.delete(tradeId);
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
      const tradeId = getTradeId(trade);
      removeTradeFromState(tradeId);
      const pnlValue = Number(trade.realized_pnl ?? trade.profit ?? 0);
      const pnlLabel = formatPnL(pnlValue);
      const variant = Number.isFinite(pnlValue) && pnlValue >= 0 ? 'success' : 'warning';
      showToast(
        `Operación ${trade.symbol || tradeId} cerrada · PnL ${pnlLabel}`,
        variant,
      );
      notifyTradeClosed(trade);
    }
  });
  socket.on('bot_status', ({ trading_active: tradingActive }) => {
    if (typeof tradingActive === 'boolean') {
      state.connectionHealthy = true;
      updateTradingControls(Boolean(tradingActive));
    }
  });
}
