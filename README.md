# Cobertura Terrestre del Quindio

Proyecto de detección de cobertura terrestre para el municipio de Armenia, Quindio, Colombia usando inteligencia artificial.

## Descripción

Este proyecto clasifica el tipo de cobertura terrestre (bosque, pastizal, agricultura, zona urbana, cuerpos de agua, suelo desnudo) empleando:

- **Google Earth Engine** para descarga de imágenes Sentinel-2
- **ResNet-50** pre-entrenado para clasificación de imágenes satelitales
- **Folium** para visualización de mapas interactivos

## Configuración del proyecto

| Parámetro | Valor |
|-----------|-------|
| Municipio | Armenia, Quindio |
| Coordenadas | 4.5339° N, 75.6811° W |
| Resolución | 10 metros |
| Período | Imagen más reciente (<10% nubes) |
| Coberturas | 6 clases básicas |
| Salida | Mapa interactivo HTML |

## Estructura del proyecto

```
taller_ia/
├── requirements.txt          # Dependencias del proyecto
├── main.py                   # Script principal (pipeline completo)
├── config.json               # Configuración: coordenadas, clases
├── README.md                 # Este archivo
├── modules/
│   ├── __init__.py
│   ├── gee_auth.py           # Autenticación Google Earth Engine
│   ├── download.py           # Descarga de imágenes Sentinel-2
│   ├── preprocess.py         # Preprocesamiento y tiles
│   ├── model.py              # Definición ResNet-50
│   ├── inference.py          # Clasificación de tiles
│   └── visualize.py          # Mapa interactivo HTML
└── data/
    └── (imágenes descargadas automáticamente)
```

## Instalación

### 1. Crear entorno virtual

```powershell
python -m venv venv
venv\Scripts\activate
```

### 2. Instalar dependencias

```powershell
pip install -r requirements.txt
```

### 3. Autenticar Google Earth Engine

```powershell
python -c "import ee; ee.Authenticate()"
python -c "import ee; ee.Initialize()"
```

## Uso

```powershell
python main.py
```

## Coberturas detectadas

| Color | Cobertura |
|-------|-----------|
| Verde oscuro | Bosque |
| Verde claro | Pastizal |
| Amarillo | Agricultura |
| Gris | Zona urbana |
| Azul | Cuerpos de agua |
| Marrón | Suelo desnudo |

## Estado del proyecto

- [x] Planificación y diseño
- [x] Autenticación Google Earth Engine
- [ ] Implementación del pipeline
- [ ] Descarga de imágenes Sentinel-2
- [ ] Modelo ResNet-50
- [ ] Visualización con folium

## Autor

Gabriela Zea
