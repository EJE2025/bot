const cacheVersionSource =
  self.registration?.installing?.scriptURL ||
  self.registration?.active?.scriptURL ||
  'bot-cache-v20240606';
const CACHE_NAME = `bot-cache-${cacheVersionSource
  .split('/')
  .pop()
  ?.replace(/[^a-z0-9_-]/gi, '-') ?? 'default'}`;
const RESOURCES = ['/', '/static/css/dashboard.css', '/static/js/dashboard.js'];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => cache.addAll(RESOURCES)).catch((error) => {
      console.error('SW install error', error);
    }),
  );
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
  event.respondWith(
    caches.match(event.request).then((response) => response || fetch(event.request)),
  );
});
