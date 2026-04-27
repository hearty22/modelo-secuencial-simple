

# TP: Entrenamiento de Modelo Secuencial con TensorFlow.js

## Descripcion
Este proyecto consiste en la implementacion de una red neuronal artificial simple (Regresion Lineal) entrenada desde cero en el navegador. El objetivo es que el modelo aprenda la relacion matematica detras de una funcion lineal especifica mediante el ajuste de pesos y sesgos a lo largo de un proceso de entrenamiento iterativo.

## Especificaciones Tecnicas
* Formula Objetivo: y = 2x + 6
* Arquitectura: Modelo secuencial con una unica capa densa (1 neurona).
* Optimizador: Stochastic Gradient Descent (SGD).
* Funcion de Perdida: Mean Squared Error (MSE).
* Dataset de Entrenamiento:
  * Entradas (X): Tensores de forma [9, 1] iniciando en -6 con incremento unitario.
  * Etiquetas (Y): Valores calculados mediante la funcion objetivo correspondientes a cada X.
* Ciclos de Entrenamiento: 350 epocas (Epochs).

## Tecnologias Utilizadas
* React (Vite)
* TypeScript
* TensorFlow.js (@tensorflow/tfjs)

## Caracteristicas de la Implementacion
* Feedback en tiempo real: El sistema utiliza callbacks (onEpochEnd) para informar al usuario el progreso exacto del entrenamiento.
* Gestion de Memoria: Uso de tf.tidy() para la limpieza automatica de tensores durante la fase de prediccion, evitando fugas de memoria en el hardware grafico (GPU/WebGL).
* Arquitectura Desacoplada: El modelo se mantiene en una referencia (useRef) para persistir la red entrenada sin causar re-renderizados innecesarios del DOM.

## Instalacion y Ejecucion

1. Clonar el repositorio o descargar los archivos.
2. Instalar dependencias:
   ```bash
   npm install
   ```
3. Iniciar el servidor de desarrollo:
   ```bash
   npm run dev
   ```
4. Acceder a la aplicacion y presionar el boton "Iniciar Entrenamiento" para generar los pesos del modelo antes de realizar predicciones.

## Ejemplo de Uso
Una vez finalizadas las 350 epocas, si se ingresa un valor X = 10, el modelo deberia predecir un valor Y cercano a 26, demostrando que la red neuronal ha inferido correctamente la logica de la ecuacion lineal.
