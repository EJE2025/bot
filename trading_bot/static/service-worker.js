const CACHE_VERSION = 'v20240615';
const CACHE_NAME = `bot-cache-${CACHE_VERSION}`;
const RESOURCES = ['/', '/static/css/dashboard.css', '/static/js/dashboard.js'];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => cache.addAll(RESOURCES)).catch((error) => {
      console.error('SW install error', error);
    }),
  );
  self.skipWaiting();
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    (async () => {
      const cacheNames = await caches.keys();
      await Promise.all(
        cacheNames
          .filter((name) => name !== CACHE_NAME)
          .map((name) => caches.delete(name)),
      );
      await self.clients.claim();
    })(),
  );
});

self.addEventListener('fetch', (event) => {
  const { request } = event;
  if (request.method !== 'GET') {
    return;
  }

  const acceptHeader = request.headers.get('accept') || '';
  if (acceptHeader.includes('text/event-stream')) {
    return;
  }

  if (request.headers.get('upgrade') === 'websocket') {
    return;
  }

  const url = new URL(request.url);
  const isSameOrigin = url.origin === self.location.origin;

  if (request.mode === 'navigate') {
    event.respondWith(networkFirst(request));
    return;
  }

  if (isSameOrigin && RESOURCES.includes(url.pathname)) {
    event.respondWith(networkFirst(request));
  }
});

async function networkFirst(request) {
  try {
    const response = await fetch(request);
    const cache = await caches.open(CACHE_NAME);
    cache.put(request, response.clone());
    return response;
  } catch (error) {
    const cached = await caches.match(request);
    if (cached) {
      return cached;
    }
    throw error;
  }
}
