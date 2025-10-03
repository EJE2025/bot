# Plataforma modular de trading

Este directorio agrupa los microservicios que amplían el bot tradicional y permiten desplegar una plataforma integral:

- `analytics/`: servicio Flask + GraphQL respaldado por SQLAlchemy.
- `ai_service/`: envoltura sobre ChatGPT para informes y asistencia.
- `gateway/`: API Gateway con FastAPI que centraliza las peticiones.
- `streaming_service/`: publicador de datos en tiempo real usando Redis Streams.
- `trading_engine/`: motor de órdenes ligero con FastAPI.
- `frontend/`: documentación y estructura inicial para un cliente Next.js/SvelteKit.

Cada servicio incluye su `Dockerfile` y puede desplegarse de forma independiente o mediante `docker-compose`.
