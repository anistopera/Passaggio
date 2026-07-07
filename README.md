# IA Coach Vocal - Analizador de Técnica Vocal con Deep Learning

Sistema basado en Inteligencia Artificial diseñado para analizar grabaciones de voz cantada en tiempo real. Utiliza redes neuronales convolucionales (CNN) sobre espectrogramas de audio para clasificar la técnica vocal del usuario y proporcionar retroalimentación técnica y ejercicios de mejora.

## Descripción del Proyecto

Este proyecto (MVP) aborda la clasificación acústica de la voz humana. Dado que el audio crudo unidimensional presenta alta complejidad para la extracción directa de características de timbre, el sistema transforma las ondas de audio en espectrogramas bidimensionales mediante la Transformada Rápida de Fourier (STFT).

El modelo es capaz de clasificar el audio en tres categorías técnicas distintas extraídas del dataset académico VocalSet:
- **Breathy:** Fuga de aire por falta de cierre cordal.
- **Belt:** Emisión potente con alta resonancia de pecho/mix.
- **Straight:** Tono limpio, recto y lírico, sin presencia de vibrato.

## Arquitectura del Sistema

El pipeline del proyecto está compuesto por dos fases principales:

### Preprocesamiento Acústico (Torchaudio):
- Carga de archivos `.wav` y conversión a formato monoaural.
- Remuestreo (Resampling) a 16,000 Hz.
- Homogeneización de la longitud de la muestra a exactamente 2 segundos mediante técnicas de truncado y padding (zero-padding).
- Transformación a espectrograma en escala logarítmica para estabilización de gradientes.

### Modelo de Deep Learning (PyTorch):
- Red Neuronal Convolucional 2D (CNN) diseñada a medida.
- Consta de 2 bloques convolucionales para extracción de características espaciales y de frecuencia.
- Implementación de `BatchNorm2d` para normalización de activaciones y mitigación del desvanecimiento del gradiente.
- Capas `MaxPool2d` para reducción de dimensionalidad espacial.
- Capa lineal final con `Dropout (0.5)` como mecanismo de regularización para evitar el sobreajuste (Overfitting).

## Estructura del Proyecto

```text
Passaggio/
 ├── datos/                  # Carpeta de audios de entrenamiento (Ignorada en Git)
 ├── modelos/                # Almacenamiento de pesos del modelo (.pth)
 ├── static/
 │   └── style.css           # Hoja de estilos minimalista del frontend
 ├── templates/
 │   └── index.html          # Interfaz de usuario (Jinja2)
 ├── uploads/                # Directorio temporal para inferencia
 ├── prepararDatos.ipynb     # Pipeline de extraccion y formateo de datos
 ├── entrenamiento.ipynb     # Bucle de entrenamiento y validacion del modelo
 ├── app.py                  # Servidor Backend y logica de inferencia
 └── README.md               # Documentacion
```

## Requisitos de Instalación

1. **Clonar el repositorio:**
   ```bash
   git clone https://github.com/anistopera/Passaggio.git
   cd Passaggio
   ```

2. **Crear y activar el entorno virtual:**

   * **En Windows (PowerShell/CMD):**
     ```cmd
     python -m venv venv
     venv\Scripts\activate
     ```

   * **En Linux/Debian:**
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```

3. **Instalar dependencias del sistema:**

   * **En Windows:**  
     No se requieren instalaciones adicionales en el sistema; los binarios necesarios vienen incluidos con Python y las librerías pip.
   
   * **En Linux/Debian** (necesarias para el procesamiento de audio):
     ```bash
     sudo apt update
     sudo apt install ffmpeg libsm6 libxext6 -y
     ```

4. **Instalar las dependencias de Python:**
   En este proyecto no hay un archivo de requerimientos, por lo que debes instalar las librerías base manualmente ejecutando:
   
   ```bash
   pip install Flask torch torchaudio soundfile werkzeug
   ```
   *(Nota: La instalación estándar de PyTorch optimizada para CPU será suficiente para el uso de esta aplicación).*

## Uso de la Aplicación

Para desplegar la interfaz web localmente, asegúrese de tener el entorno virtual activado y ejecute el servidor Flask:

```bash
python app.py
```

Abra su navegador web e ingrese a `http://127.0.0.1:5000`. Podrá cargar un archivo de audio `.wav` y el sistema procesará la inferencia, retornando el diagnóstico y el plan de acción sugerido por el modelo.

## Trabajo Futuro

Para iteraciones posteriores del proyecto, se planifica implementar:
- Clasificación de Rango Vocal (Tesituras).
- Recomendación de repertorio musical basado en el perfil acústico del usuario.
- Historial de progreso vocal del usuario utilizando bases de datos relacionales.

## Reconocimientos

Dataset de entrenamiento extraído de **VocalSet**: A Singing Voice Dataset.
