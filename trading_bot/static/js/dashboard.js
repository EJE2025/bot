const DEFAULT_REFRESH_INTERVAL = 10000;
const MIN_REFRESH_INTERVAL = 5000;
const MAX_REFRESH_INTERVAL = 300000;
const MAX_POINTS = 60;
const DASHBOARD_ORDER_KEY = 'dashboard:layout';
const COLLAPSED_CARDS_KEY = 'dashboard:collapsed';
const NOTIFICATION_REQUESTED_KEY = 'dashboard:notifications-requested';
const THEME_STORAGE_KEY = 'dashboard:theme';
const PREFERENCES_STORAGE_KEY = 'dashboard:preferences';
const AVAILABLE_THEMES = ['light', 'dark', 'pastel'];
const ORDERED_SECTIONS = ['dashboard', 'analytics', 'assistant', 'services'];

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
    '--background': '#fef9ff',
    '--surface': 'rgba(255, 255, 255, 0.78)',
    '--surface-strong': 'rgba(255, 255, 255, 0.92)',
    '--surface-elevated': 'rgba(255, 255, 255, 0.95)',
    '--accent-primary': '#7f67ff',
    '--accent-secondary': '#ff8fb1',
    '--accent-tertiary': '#6dd5ed',
    '--text-primary': '#413659',
    '--text-secondary': 'rgba(65, 54, 89, 0.7)',
    '--shadow': 'rgba(127, 103, 255, 0.2)',
    '--border-subtle': 'rgba(127, 103, 255, 0.12)',
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

const integerFormatter = new Intl.NumberFormat(userLocale, {
  maximumFractionDigits: 0,
});

const ANALYTICS_OVERVIEW_QUERY = `
  query DashboardOverview {
    performanceSummary {
      totalTrades
      winRate
      sharpeRatio
      profitFactor
      bestSymbol
      worstSymbol
    }
    volumeBySymbol(limit: 6) {
      symbol
      volume
      change24h
    }
  }
`;

function initAppConfig() {
  const { dataset } = document.body;
  appConfig.apiBase = (dataset.apiBase || '').replace(/\/$/, '');
  appConfig.analyticsGraphql = (dataset.analyticsGraphql || '').trim();
  appConfig.aiEndpoint = (dataset.aiEndpoint || '').trim();
  try {
    const parsed = dataset.serviceLinks ? JSON.parse(dataset.serviceLinks) : [];
    if (Array.isArray(parsed)) {
      appConfig.serviceLinks = parsed
        .map((item) => {
          if (!item) return null;
          if (typeof item === 'string') {
            return { label: item, url: item };
          }
          const { label, url } = item;
          if (!url) return null;
          return {
            label: label || url,
            url,
          };
        })
        .filter(Boolean);
    } else {
      appConfig.serviceLinks = [];
    }
  } catch (error) {
    console.warn('No se pudieron analizar los enlaces de servicios externos', error);
    appConfig.serviceLinks = [];
  }
}

function resolveApiUrl(path) {
  if (!path) {
    return appConfig.apiBase || '';
  }
  if (/^https?:/i.test(path)) {
    return path;
  }
  const base = appConfig.apiBase;
  if (!base) {
    return path;
  }
  const normalized = path.startsWith('/') ? path : `/${path}`;
  return `${base}${normalized}`;
}

function resolveSocketUrl(path = '/ws') {
  const httpUrl = resolveApiUrl(path);
  if (httpUrl) {
    try {
      const url = new URL(httpUrl, window.location.origin);
      if (url.protocol === 'http:') {
        url.protocol = 'ws:';
      } else if (url.protocol === 'https:') {
        url.protocol = 'wss:';
      }
      return url.toString();
    } catch (error) {
      return httpUrl.replace(/^http:/i, 'ws:').replace(/^https:/i, 'wss:');
    }
  }
  const normalized = path.startsWith('/') ? path : `/${path}`;
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  return `${protocol}//${window.location.host}${normalized}`;
}

function getAnalyticsEndpoint() {
  if (appConfig.analyticsGraphql) {
    return appConfig.analyticsGraphql;
  }
  if (appConfig.apiBase) {
    return `${appConfig.apiBase}/graphql`;
  }
  return '/graphql';
}

function isAiConfigured() {
  return Boolean(appConfig.aiEndpoint);
}

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
  activeSection: 'dashboard',
  preferences: null,
};

let pnlChart = null;
let socket = null;
let partialModal = null;
let partialTradeContext = null;
let notificationsRequested = false;
let preferencesModal = null;
let refreshTimer = null;

const appConfig = {
  apiBase: '',
  analyticsGraphql: '',
  aiEndpoint: '',
  serviceLinks: [],
};

const DEFAULT_PREFERENCES = {
  sections: {
    dashboard: true,
    analytics: true,
    assistant: true,
    services: true,
  },
  widgets: {
    metricsPrimary: true,
    metricsSecondary: true,
    pnlCard: true,
    symbolCard: true,
    tradesCard: true,
    historyCard: true,
  },
  refreshInterval: DEFAULT_REFRESH_INTERVAL,
};

const analyticsState = {
  lastUpdated: null,
  loading: false,
};

const aiState = {
  conversation: [],
  busy: false,
};

let currentServiceUrl = null;

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
  toggle.setAttribute('aria-pressed', String(theme === 'dark'));
  const icon = toggle.querySelector('i');
  if (icon) {
    icon.className = 'bi';
    if (theme === 'dark') {
      icon.classList.add('bi-sun');
    } else if (theme === 'pastel') {
      icon.classList.add('bi-palette');
    } else {
      icon.classList.add('bi-moon-stars');
    }
  }
  let title = 'Cambiar tema';
  if (theme === 'dark') {
    title = 'Cambiar a tema pastel';
  } else if (theme === 'pastel') {
    title = 'Cambiar a tema claro';
  } else {
    title = 'Cambiar a tema oscuro';
  }
  toggle.setAttribute('title', title);
  toggle.setAttribute('aria-label', title);
}

function getNextTheme(current) {
  const index = AVAILABLE_THEMES.indexOf(current);
  const nextIndex = index >= 0 ? (index + 1) % AVAILABLE_THEMES.length : 0;
  return AVAILABLE_THEMES[nextIndex] || 'light';
}

function toggleTheme() {
  applyThemeVariables(getNextTheme(state.theme));
}

function sanitizeRefreshInterval(value) {
  let numeric = Number(value);
  if (!Number.isFinite(numeric) || numeric <= 0) {
    return DEFAULT_REFRESH_INTERVAL;
  }
  if (numeric < MIN_REFRESH_INTERVAL && numeric >= 5) {
    numeric *= 1000;
  }
  numeric = Math.max(MIN_REFRESH_INTERVAL, Math.min(MAX_REFRESH_INTERVAL, numeric));
  return Math.round(numeric);
}

function loadPreferences() {
  const stored = getStoredJSON(PREFERENCES_STORAGE_KEY);
  const preferences = {
    sections: { ...DEFAULT_PREFERENCES.sections },
    widgets: { ...DEFAULT_PREFERENCES.widgets },
    refreshInterval: DEFAULT_PREFERENCES.refreshInterval,
  };
  if (stored && typeof stored === 'object') {
    if (stored.sections && typeof stored.sections === 'object') {
      Object.assign(preferences.sections, stored.sections);
    }
    if (stored.widgets && typeof stored.widgets === 'object') {
      Object.assign(preferences.widgets, stored.widgets);
    }
    if (stored.refreshInterval) {
      preferences.refreshInterval = sanitizeRefreshInterval(stored.refreshInterval);
    }
  }
  preferences.refreshInterval = sanitizeRefreshInterval(preferences.refreshInterval);
  state.preferences = preferences;
  state.refreshInterval = preferences.refreshInterval;
}

function savePreferences() {
  if (!state.preferences) {
    return;
  }
  setStoredJSON(PREFERENCES_STORAGE_KEY, state.preferences);
}

function isSectionEnabled(sectionId) {
  if (!state.preferences) {
    return true;
  }
  const value = state.preferences.sections?.[sectionId];
  return value !== false;
}

function isWidgetEnabled(widgetId) {
  if (!state.preferences) {
    return true;
  }
  const value = state.preferences.widgets?.[widgetId];
  return value !== false;
}

function getFirstEnabledSection() {
  const fallback = document.querySelector('.app-section');
  const preferred = ORDERED_SECTIONS.find((section) => isSectionEnabled(section));
  return preferred || fallback?.dataset.section || 'dashboard';
}

function setActiveSection(sectionId, options = {}) {
  const { force = false } = options;
  if (!force && !isSectionEnabled(sectionId)) {
    return;
  }
  state.activeSection = sectionId;
  document.querySelectorAll('.app-section').forEach((section) => {
    const id = section.dataset.section;
    const enabled = !section.classList.contains('section-hidden');
    const isActive = enabled && id === sectionId;
    section.classList.toggle('active', isActive);
    section.classList.toggle('d-none', !isActive);
  });
  document.querySelectorAll('[data-section-target]').forEach((link) => {
    const target = link.dataset.sectionTarget;
    const enabled = isSectionEnabled(target);
    const isActive = enabled && target === sectionId;
    link.classList.toggle('active', isActive);
    link.setAttribute('aria-current', isActive ? 'page' : 'false');
  });
}

function updateWidgetVisibility() {
  document.querySelectorAll('[data-widget-id]').forEach((element) => {
    const id = element.dataset.widgetId;
    if (!id) return;
    const enabled = isWidgetEnabled(id);
    element.classList.toggle('section-hidden', !enabled);
    element.classList.toggle('d-none', !enabled);
  });
}

function updateSectionVisibility() {
  document.querySelectorAll('.app-section').forEach((section) => {
    const id = section.dataset.section;
    const enabled = isSectionEnabled(id);
    section.classList.toggle('section-hidden', !enabled);
    if (!enabled) {
      section.classList.add('d-none');
      section.classList.remove('active');
    }
  });
  document.querySelectorAll('[data-section-target]').forEach((link) => {
    const target = link.dataset.sectionTarget;
    const enabled = isSectionEnabled(target);
    link.classList.toggle('disabled', !enabled);
    link.setAttribute('aria-disabled', String(!enabled));
    link.tabIndex = enabled ? 0 : -1;
    if (!enabled) {
      link.classList.remove('active');
    }
  });
  const active = state.activeSection;
  const nextSection = isSectionEnabled(active) ? active : getFirstEnabledSection();
  setActiveSection(nextSection, { force: true });
}

function updateRefreshIntervalPreview(intervalMs) {
  const preview = document.getElementById('refreshIntervalPreview');
  if (!preview) return;
  const seconds = Math.round((intervalMs || state.refreshInterval || DEFAULT_REFRESH_INTERVAL) / 1000);
  preview.innerHTML = `Los datos se actualizarán cada <strong>${seconds}</strong> segundos.`;
}

function scheduleAutoRefresh() {
  if (refreshTimer) {
    clearInterval(refreshTimer);
  }
  const interval = sanitizeRefreshInterval(state.refreshInterval || DEFAULT_REFRESH_INTERVAL);
  state.refreshInterval = interval;
  updateRefreshIntervalPreview(interval);
  refreshTimer = window.setInterval(() => {
    if (!document.hidden) {
      refreshDashboard();
      if (isSectionEnabled('analytics')) {
        refreshAnalytics();
      }
    }
  }, interval);
}

function applyPreferences({ persist = true } = {}) {
  if (!state.preferences) {
    loadPreferences();
  }
  updateWidgetVisibility();
  updateSectionVisibility();
  scheduleAutoRefresh();
  if (persist) {
    savePreferences();
  }
  if (isSectionEnabled('analytics')) {
    refreshAnalytics();
  }
}

function populatePreferencesForm() {
  if (!state.preferences) {
    return;
  }
  document.querySelectorAll('[data-pref-section]').forEach((input) => {
    const section = input.dataset.prefSection;
    input.checked = isSectionEnabled(section);
  });
  document.querySelectorAll('[data-pref-widget]').forEach((input) => {
    const widget = input.dataset.prefWidget;
    input.checked = isWidgetEnabled(widget);
  });
  const refreshInput = document.getElementById('prefRefreshInterval');
  if (refreshInput) {
    refreshInput.value = Math.round((state.refreshInterval || DEFAULT_REFRESH_INTERVAL) / 1000);
  }
  updateRefreshIntervalPreview(state.refreshInterval);
}

function openPreferencesModal() {
  if (!preferencesModal && window.bootstrap) {
    const modalElement = document.getElementById('preferencesModal');
    if (modalElement) {
      preferencesModal = new bootstrap.Modal(modalElement);
    }
  }
  populatePreferencesForm();
  preferencesModal?.show();
}

function handlePreferencesSubmit(event) {
  event.preventDefault();
  const nextPreferences = {
    sections: { ...DEFAULT_PREFERENCES.sections },
    widgets: { ...DEFAULT_PREFERENCES.widgets },
    refreshInterval: state.refreshInterval,
  };
  document.querySelectorAll('[data-pref-section]').forEach((input) => {
    const section = input.dataset.prefSection;
    if (!section) return;
    nextPreferences.sections[section] = input.checked;
  });
  document.querySelectorAll('[data-pref-widget]').forEach((input) => {
    const widget = input.dataset.prefWidget;
    if (!widget) return;
    nextPreferences.widgets[widget] = input.checked;
  });
  const refreshInput = document.getElementById('prefRefreshInterval');
  if (refreshInput) {
    const seconds = Number(refreshInput.value);
    nextPreferences.refreshInterval = sanitizeRefreshInterval(seconds * 1000);
  }
  state.preferences = nextPreferences;
  state.refreshInterval = nextPreferences.refreshInterval;
  applyPreferences();
  if (preferencesModal) {
    preferencesModal.hide();
  }
  showToast('Preferencias actualizadas', 'success');
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
  if (!AVAILABLE_THEMES.includes(storedTheme)) {
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

function showAnalyticsStatus(message, variant = 'info') {
  const element = document.getElementById('analyticsStatus');
  if (!element) return;
  if (!message) {
    element.classList.add('d-none');
    element.textContent = '';
    element.classList.remove('alert-warning', 'alert-danger', 'alert-info', 'alert-success');
    return;
  }
  element.textContent = message;
  element.classList.remove('d-none');
  element.classList.remove('alert-warning', 'alert-danger', 'alert-info', 'alert-success');
  element.classList.add(`alert-${variant}`);
}

function renderAnalyticsOverview(overview) {
  const container = document.getElementById('analyticsOverview');
  if (!container) return;
  clearSkeleton(container);
  if (!overview) {
    container.innerHTML = '<p class="text-muted mb-0">Sin datos recientes. Ejecuta el microservicio de analítica para ver estadísticas.</p>';
    return;
  }
  const metrics = [
    { label: 'Operaciones totales', value: formatNumber(overview.totalTrades, integerFormatter) },
    {
      label: 'Win rate',
      value:
        typeof overview.winRate === 'number' && !Number.isNaN(overview.winRate)
          ? percentFormatter.format(overview.winRate)
          : '—',
    },
    { label: 'Sharpe', value: formatNumber(overview.sharpeRatio) },
    { label: 'Profit factor', value: formatNumber(overview.profitFactor) },
    { label: 'Mejor símbolo', value: overview.bestSymbol || '—' },
    { label: 'Peor símbolo', value: overview.worstSymbol || '—' },
  ];
  container.innerHTML = `
    <dl class="row gy-2 gx-1 analytics-overview">
      ${metrics
        .map(
          (metric) => `
            <dt class="col-7 text-muted small">${metric.label}</dt>
            <dd class="col-5 text-end fw-semibold">${metric.value ?? '—'}</dd>
          `,
        )
        .join('')}
    </dl>`;
}

function renderAnalyticsSymbols(items) {
  const list = document.getElementById('analyticsSymbols');
  if (!list) return;
  clearSkeleton(list);
  if (!Array.isArray(items) || items.length === 0) {
    list.innerHTML = `
      <li class="list-group-item d-flex justify-content-between text-muted">
        <span>Sin datos de volumen disponibles.</span>
        <i class="bi bi-broadcast"></i>
      </li>`;
    return;
  }
  const rows = items
    .map((item) => {
      const change = Number(item.change24h);
      const changeLabel = Number.isFinite(change)
        ? `${change >= 0 ? '+' : ''}${numberFormatter.format(change)}%`
        : '—';
      const changeClass = Number.isFinite(change)
        ? change >= 0
          ? 'text-success'
          : 'text-danger'
        : 'text-muted';
      const volumeLabel = formatNumber(item.volume);
      return `
        <li class="list-group-item d-flex justify-content-between align-items-center">
          <span class="fw-semibold">${item.symbol || '—'}</span>
          <span class="d-flex align-items-center gap-2">
            <span class="badge">${volumeLabel}</span>
            <small class="${changeClass}">${changeLabel}</small>
          </span>
        </li>`;
    })
    .join('');
  list.innerHTML = rows;
}

async function refreshAnalytics(manual = false) {
  if (!isSectionEnabled('analytics')) {
    return;
  }
  const endpoint = getAnalyticsEndpoint();
  if (!endpoint) {
    showAnalyticsStatus('Configura ANALYTICS_GRAPHQL_URL para habilitar la analítica.', 'warning');
    return;
  }
  if (analyticsState.loading) {
    return;
  }
  clearSkeleton(document.getElementById('analyticsOverview'));
  clearSkeleton(document.getElementById('analyticsSymbols'));
  analyticsState.loading = true;
  if (manual) {
    showAnalyticsStatus('Actualizando analítica…', 'info');
  }
  try {
    const response = await fetch(endpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query: ANALYTICS_OVERVIEW_QUERY }),
    });
    const payload = await response.json();
    if (payload.errors && payload.errors.length) {
      throw new Error(payload.errors[0]?.message || 'Error del microservicio de analítica');
    }
    if (!response.ok) {
      throw new Error('Respuesta inesperada del microservicio de analítica');
    }
    const data = payload.data || {};
    renderAnalyticsOverview(data.performanceSummary);
    renderAnalyticsSymbols(data.volumeBySymbol);
    analyticsState.lastUpdated = new Date();
    const badge = document.getElementById('analyticsLastUpdated');
    if (badge) {
      badge.textContent = analyticsState.lastUpdated.toLocaleTimeString(userLocale);
    }
    showAnalyticsStatus('');
  } catch (error) {
    console.error(error);
    showAnalyticsStatus(error.message || 'No se pudo cargar la analítica.', 'warning');
  } finally {
    analyticsState.loading = false;
  }
}

function ensureAiContainerReady() {
  const container = document.getElementById('aiChatMessages');
  if (!container) {
    return null;
  }
  clearSkeleton(container);
  const placeholder = container.querySelector('.ai-chat-placeholder');
  if (placeholder) {
    placeholder.remove();
  }
  return container;
}

function updateAiStatus(label, variant = 'secondary') {
  const badge = document.getElementById('aiStatus');
  if (!badge) return;
  badge.textContent = label;
  badge.className = `badge bg-${variant}`;
}

function setAssistantBusy(isBusy) {
  aiState.busy = isBusy;
  const sendBtn = document.getElementById('aiChatSendBtn');
  if (sendBtn) {
    sendBtn.disabled = isBusy || !isAiConfigured();
  }
  const input = document.getElementById('aiMessageInput');
  if (input) {
    input.disabled = !isAiConfigured();
  }
}

function appendAiMessage(role, content) {
  const container = ensureAiContainerReady();
  if (!container) {
    return;
  }
  const message = document.createElement('div');
  message.className = `ai-chat-message ai-chat-message--${role}`;
  const roleSpan = document.createElement('span');
  roleSpan.className = 'ai-chat-message__role';
  roleSpan.textContent = role === 'assistant' ? 'Asistente' : 'Tú';
  const bubble = document.createElement('div');
  bubble.className = 'ai-chat-message__bubble';
  bubble.textContent = content;
  message.append(roleSpan, bubble);
  container.appendChild(message);
  container.scrollTo({ top: container.scrollHeight, behavior: 'smooth' });
}

async function sendAiMessage(message) {
  if (!message.trim()) {
    return;
  }
  if (!isAiConfigured()) {
    showToast('Configura AI_ASSISTANT_URL para habilitar el asistente.', 'warning');
    return;
  }
  setAssistantBusy(true);
  updateAiStatus('Pensando…', 'info');
  appendAiMessage('user', message);
  aiState.conversation.push({ role: 'user', content: message });
  try {
    const response = await fetch(appConfig.aiEndpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        message,
        conversation: aiState.conversation,
      }),
    });
    const data = await readJsonResponse(response, 'No se pudo obtener respuesta del asistente');
    if (!response.ok) {
      throw new Error(data.error || 'El asistente devolvió un error');
    }
    const reply = data.reply || data.answer || data.message;
    if (reply) {
      aiState.conversation.push({ role: 'assistant', content: reply });
      appendAiMessage('assistant', reply);
      updateAiStatus('Listo', 'success');
    } else {
      updateAiStatus('Sin respuesta', 'warning');
      showToast('El asistente no generó respuesta.', 'warning');
    }
  } catch (error) {
    console.error(error);
    updateAiStatus('Error', 'danger');
    showToast(error.message || 'Error al contactar al asistente', 'danger');
  } finally {
    setAssistantBusy(false);
  }
}

function initializeAssistant() {
  const container = document.getElementById('aiChatMessages');
  if (container) {
    clearSkeleton(container);
  }
  const input = document.getElementById('aiMessageInput');
  if (input) {
    input.placeholder = isAiConfigured()
      ? '¿Cómo va el momentum de BTC/USDT?'
      : 'Configura AI_ASSISTANT_URL para habilitar el asistente.';
    input.disabled = !isAiConfigured();
  }
  const sendBtn = document.getElementById('aiChatSendBtn');
  if (sendBtn) {
    sendBtn.disabled = !isAiConfigured();
  }
  updateAiStatus(isAiConfigured() ? 'Listo' : 'Sin configurar', isAiConfigured() ? 'success' : 'warning');
}

function initializeServices() {
  const listContainer = document.getElementById('servicesButtons');
  const emptyAlert = document.getElementById('servicesEmpty');
  const preview = document.getElementById('servicePreview');
  const placeholder = document.getElementById('servicePlaceholder');
  const openBtn = document.getElementById('serviceOpenExternal');
  if (preview) {
    preview.src = 'about:blank';
    preview.classList.remove('is-visible');
  }
  if (placeholder) {
    placeholder.classList.remove('d-none');
  }
  currentServiceUrl = null;
  if (openBtn) {
    openBtn.disabled = true;
  }
  if (!listContainer) {
    return;
  }
  clearSkeleton(listContainer);
  listContainer.innerHTML = '';
  if (!Array.isArray(appConfig.serviceLinks) || appConfig.serviceLinks.length === 0) {
    if (emptyAlert) {
      emptyAlert.classList.remove('d-none');
    }
    return;
  }
  if (emptyAlert) {
    emptyAlert.classList.add('d-none');
  }
  appConfig.serviceLinks.forEach((service) => {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'btn btn-secondary';
    button.textContent = service.label || service.url;
    button.addEventListener('click', () => {
      loadServicePreview(service);
    });
    listContainer.appendChild(button);
  });
  if (appConfig.serviceLinks.length > 0) {
    loadServicePreview(appConfig.serviceLinks[0]);
  }
}

function loadServicePreview(service) {
  const iframe = document.getElementById('servicePreview');
  const placeholder = document.getElementById('servicePlaceholder');
  const openBtn = document.getElementById('serviceOpenExternal');
  if (!iframe || !placeholder || !openBtn) {
    return;
  }
  if (!service) {
    iframe.src = 'about:blank';
    iframe.classList.remove('is-visible');
    placeholder.classList.remove('d-none');
    openBtn.disabled = true;
    currentServiceUrl = null;
    return;
  }
  currentServiceUrl = service.url;
  openBtn.disabled = false;
  placeholder.classList.add('d-none');
  iframe.classList.remove('is-visible');
  iframe.src = service.url;
  iframe.addEventListener(
    'load',
    () => {
      iframe.classList.add('is-visible');
    },
    { once: true },
  );
}

async function refreshDashboard(manual = false) {
  try {
    if (manual) {
      setStatus('Actualizando…', 'info');
    }
    clearAlert();

    const trades = (await fetchJSON(resolveApiUrl('/api/trades'), { silent: !manual })) ?? [];
    const summarySilent = manual ? false : state.summary !== null;
    const summary = await fetchJSON(resolveApiUrl('/api/summary'), { silent: summarySilent });
    const history =
      (await fetchJSON(resolveApiUrl('/api/history?limit=50'), { silent: true })) ?? [];

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
    if (manual && isSectionEnabled('analytics')) {
      refreshAnalytics(true);
    }
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
    refreshBtn.addEventListener('click', () => {
      refreshDashboard(true);
    });
  }

  const themeToggle = document.getElementById('themeToggleBtn');
  if (themeToggle) {
    themeToggle.addEventListener('click', toggleTheme);
  }

  const navCollapse = document.getElementById('navbarSections');
  const collapseInstance =
    navCollapse && window.bootstrap
      ? window.bootstrap.Collapse.getOrCreateInstance(navCollapse, { toggle: false })
      : null;

  document.querySelectorAll('[data-section-target]').forEach((link) => {
    link.addEventListener('click', (event) => {
      event.preventDefault();
      const { sectionTarget } = link.dataset;
      if (!sectionTarget) return;
      if (!isSectionEnabled(sectionTarget)) {
        showToast('Activa la sección desde la configuración para poder visualizarla.', 'info');
        return;
      }
      setActiveSection(sectionTarget);
      if (collapseInstance && navCollapse?.classList.contains('show')) {
        collapseInstance.hide();
      }
    });
  });

  const preferencesBtn = document.getElementById('preferencesBtn');
  if (preferencesBtn) {
    preferencesBtn.addEventListener('click', openPreferencesModal);
  }

  const preferencesForm = document.getElementById('preferencesForm');
  if (preferencesForm) {
    preferencesForm.addEventListener('submit', handlePreferencesSubmit);
  }

  const refreshInput = document.getElementById('prefRefreshInterval');
  if (refreshInput) {
    refreshInput.addEventListener('input', (event) => {
      const seconds = Number(event.target.value);
      updateRefreshIntervalPreview(sanitizeRefreshInterval(seconds * 1000));
    });
  }

  const refreshAnalyticsBtn = document.getElementById('refreshAnalyticsBtn');
  if (refreshAnalyticsBtn) {
    refreshAnalyticsBtn.addEventListener('click', () => refreshAnalytics(true));
  }

  const aiForm = document.getElementById('aiChatForm');
  if (aiForm) {
    aiForm.addEventListener('submit', async (event) => {
      event.preventDefault();
      if (aiState.busy) return;
      const input = document.getElementById('aiMessageInput');
      const message = input?.value?.trim() || '';
      if (!message) {
        showToast('Escribe un mensaje para el asistente.', 'warning');
        return;
      }
      if (input) {
        input.value = '';
      }
      await sendAiMessage(message);
      input?.focus();
    });
  }

  document.querySelectorAll('[data-ai-prompt]').forEach((button) => {
    button.addEventListener('click', () => {
      const prompt = button.dataset.aiPrompt || '';
      const input = document.getElementById('aiMessageInput');
      if (input) {
        input.value = prompt;
        input.focus();
      }
      if (!aiState.busy && isAiConfigured()) {
        sendAiMessage(prompt);
      }
    });
  });

  const openServiceBtn = document.getElementById('serviceOpenExternal');
  if (openServiceBtn) {
    openServiceBtn.addEventListener('click', () => {
      if (currentServiceUrl) {
        window.open(currentServiceUrl, '_blank', 'noopener');
      }
    });
  }

  const toggleBtn = document.getElementById('toggleTradeBtn');
  if (toggleBtn) {
    toggleBtn.addEventListener('click', async () => {
      toggleBtn.disabled = true;
      try {
        const response = await fetch(resolveApiUrl('/api/toggle-trading'), { method: 'POST' });
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
  initAppConfig();
  initTheme();
  loadPreferences();
  registerServiceWorker();
  switchSection(state.activeSection);
  applyDashboardLayout();
  setupCollapsibles();
  registerHotkeys();
  ensureChart();
  applyPreferences({ persist: false });
  attachEvents();
  initializeAssistant();
  initializeServices();
  refreshDashboard();
  refreshAnalytics();
  setStatus('Sincronizando…', 'warning');
  scheduleAutoRefresh();
  connectSocket();
  scheduleNotificationRequest();
  document.addEventListener('visibilitychange', () => {
    if (!document.hidden) {
      refreshDashboard();
      refreshAnalytics();
    }
  });
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
        const response = await fetch(resolveApiUrl(`/api/trades/${tradeId}/close`), {
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
    const response = await fetch(resolveApiUrl(`/api/trades/${tradeId}/close-partial`), {
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
  const socketUrl = resolveSocketUrl('/ws');
  socket = window.io(socketUrl);
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
