# Plan: Detección de Cobertura Terrestre — Armenia, Quindio

## Objetivo

Programa en Python para clasificar el tipo de cobertura terrestre de Armenia, Quindio (Colombia) usando modelos de IA de código abierto.

## Decisiones del proyecto

| Decisión | Elección |
|----------|----------|
| Municipio | Armenia (capital del Quindio) |
| Fuente de imágenes | Google Earth Engine (Sentinel-2 L2A) |
| Período | Imagen más reciente (<10% nubes) |
| Modelo | ResNet-50 pre-entrenado (torchvision) |
| Resolución | 10 metros |
| Coberturas | 6 clases básicas |
| Salida | Mapa interactivo HTML (folium) |

## Coberturas a detectar

| Color | Cobertura | Descripción |
|-------|-----------|-------------|
| Verde oscuro | Bosque | Zonas con vegetación densa y árboles |
| Verde claro | Pastizal | Potreros, pastos para ganado |
| Amarillo | Agricultura | Cultivos, cafetales, zonas de labranza |
| Gris | Zona urbana | Edificios, calles, infraestructura |
| Azul | Cuerpos de agua | Río, quebradas, embalses |
| Marrón | Suelo desnudo | Tierra expuesta, sin vegetación |

## Flujo de trabajo

1. **Autenticación GEE** — Conectarse a Google Earth Engine
2. **Descarga** — Obtener imagen Sentinel-2 L2A más reciente de Armenia
3. **Preprocesamiento** — Calcular NDVI, crear tiles de 256x256 píxeles
4. **Inferencia** — Clasificar cada tile con ResNet-50
5. **Visualización** — Generar mapa interactivo HTML con folium

## Coordenadas de Armenia

- Latitud: 4.5339° N
- Longitud: -75.6811° W
- Bounding box aproximado:
  - Noroeste: (4.57, -75.72)
  - Sureste: (4.50, -75.64)

## Estructura propuesta del proyecto

```
taller_ia/
├── requirements.txt
├── main.py
├── config.json
├── PLAN.md
├── README.md
└── modules/
    ├── gee_auth.py
    ├── download.py
    ├── preprocess.py
    ├── model.py
    ├── inference.py
    └── visualize.py
```

## Dependencias

```
torch
torchvision
earthengine-api
geemap
folium
numpy
pillow
matplotlib
scikit-learn
```

## Próximos pasos

- [ ] Crear el código del pipeline completo
- [ ] Probar con imágenes reales de Sentinel-2
- [ ] Ajustar el modelo según resultados
- [ ] Documentar uso para principiantes
