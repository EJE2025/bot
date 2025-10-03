const CACHE_NAME = 'bot-cache-v1';
const RESOURCES = ['/', '/static/css/dashboard.css', '/static/js/dashboard.js'];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => cache.addAll(RESOURCES)).catch((error) => {
      console.error('SW install error', error);
    }),
  );
});

self.addEventListener('fetch', (event) => {
  event.respondWith(
    caches.match(event.request).then((response) => response || fetch(event.request)),
  );
});
