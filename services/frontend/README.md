# Frontend Service

Este directorio actúa como punto de partida para una aplicación Next.js o SvelteKit. 
Incluye comandos sugeridos y la estructura recomendada para integrar Apollo Client, 
los widgets inteligentes y el asistente conversacional descritos en el plan arquitectónico.

## Estructura sugerida

```
frontend/
  ├─ app/                # Rutas y componentes con soporte SSR
  ├─ components/
  │    ├─ AiReportCard.tsx
  │    ├─ MetricsPanel.tsx
  │    └─ TradingAssistantChat.tsx
  ├─ lib/
  │    ├─ apollo-client.ts
  │    └─ i18n.ts
  ├─ public/
  └─ package.json
```

## Pasos recomendados

1. Ejecutar `npx create-next-app@latest frontend` para inicializar la base del proyecto.
2. Instalar dependencias clave: `@apollo/client`, `graphql`, `styled-components`, `next-intl`.
3. Configurar Apollo Client apuntando al gateway (`/graphql`).
4. Implementar los widgets descritos en el plan, reutilizando los endpoints del gateway.

> La implementación concreta depende de los requisitos visuales del proyecto, pero este README deja documentado el enfoque y componentes esperados.
