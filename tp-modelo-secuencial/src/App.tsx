import { useState, useRef } from "react";
import * as tf from "@tensorflow/tfjs";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from "chart.js";
import { Line } from "react-chartjs-2";

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

function App() {
  const [isTraining, setIsTraining] = useState(false);
  const [isReady, setIsReady] = useState(false);
  const [currentEpoch, setCurrentEpoch] = useState(0);
  const [lossHistory, setLossHistory] = useState<number[]>([]);

  const [inputX, setInputX] = useState<string>("");
  const [predictionResult, setPredictionResult] = useState<number | null>(null);

  const modelRef = useRef<tf.Sequential | null>(null);

  const trainModel = async () => {
    setIsTraining(true);
    setIsReady(false);
    setPredictionResult(null);
    setLossHistory([]);

    const model = tf.sequential();
    model.add(tf.layers.dense({ units: 1, inputShape: [1] }));

    model.compile({ loss: "meanSquaredError", optimizer: "sgd" });

    const xs = tf.tensor2d([-6, -5, -4, -3, -2, -1, 0, 1, 2], [9, 1]);
    const ys = tf.tensor2d([-6, -4, -2, 0, 2, 4, 6, 8, 10], [9, 1]);

    await model.fit(xs, ys, {
      epochs: 600,
      callbacks: {
        onEpochEnd: async (epoch, logs) => {
          setCurrentEpoch(epoch + 1);
          setLossHistory((prev) => [...prev, logs.loss]);
        },
      },
    });

    modelRef.current = model;
    xs.dispose();
    ys.dispose();

    setIsTraining(false);
    setIsReady(true);
  };

  const handlePredict = () => {
    if (!modelRef.current || inputX === "") return;

    const xValue = parseFloat(inputX);
    if (isNaN(xValue)) return;

    const result = tf.tidy(() => {
      const inputTensor = tf.tensor2d([xValue], [1, 1]);
      const outputTensor = modelRef.current!.predict(inputTensor) as tf.Tensor;
      return outputTensor.dataSync()[0];
    });

    setPredictionResult(Math.round(result * 100) / 100);
  };

  const chartData = {
    labels: lossHistory.map((_, index) => index + 1),
    datasets: [
      {
        label: "Pérdida (Loss)",
        data: lossHistory,
        borderColor: "rgb(255, 99, 132)",
        backgroundColor: "rgba(255, 99, 132, 0.5)",
        tension: 0.1,
        pointRadius: 0,
      },
    ],
  };

  const chartOptions = {
    responsive: true,
    plugins: {
      legend: { position: "top" as const },
      title: { display: true, text: "Evolución de la Función de Pérdida" },
    },
    scales: {
      x: { title: { display: true, text: "Épocas" } },
      y: { title: { display: true, text: "Loss" } },
    },
  };

  return (
    <div
      style={{
        maxWidth: "600px",
        margin: "40px auto",
        padding: "20px",
        fontFamily: "system-ui",
      }}
    >
      <h1>TP: Modelo Secuencial Simple</h1>
      <hr />

      <section style={{ marginBottom: "30px" }}>
        <p>
          Entrenamiento de red neuronal para aproximar la función:{" "}
          <strong>y = 2x + 6</strong>
        </p>

        <div
          style={{
            padding: "20px",
            backgroundColor: "#f8f9fa",
            border: "1px solid #ddd",
            borderRadius: "8px",
          }}
        >
          <button
            onClick={trainModel}
            disabled={isTraining}
            style={{
              padding: "10px 20px",
              fontSize: "16px",
              cursor: isTraining ? "not-allowed" : "pointer",
              backgroundColor: isTraining ? "#6c757d" : "#007bff",
              color: "white",
              border: "none",
              borderRadius: "5px",
              width: "100%",
            }}
          >
            {isTraining
              ? `Entrenando... Época ${currentEpoch} / 600`
              : "Iniciar Entrenamiento"}
          </button>

          {isReady && (
            <div
              style={{
                marginTop: "15px",
                padding: "10px",
                backgroundColor: "#d4edda",
                color: "#155724",
                borderRadius: "5px",
                textAlign: "center",
              }}
            >
              <strong>
                ¡Entrenamiento finalizado! El modelo está listo para usarse.
              </strong>
            </div>
          )}
        </div>
      </section>

      {(isTraining || isReady) && (
        <section
          style={{
            marginBottom: "30px",
            padding: "20px",
            backgroundColor: "#fff",
            border: "1px solid #ddd",
            borderRadius: "8px",
            boxShadow: "0 2px 4px rgba(0,0,0,0.1)",
          }}
        >
          <Line data={chartData} options={chartOptions} />
        </section>
      )}

      {isReady && (
        <section
          style={{
            padding: "20px",
            border: "1px solid #b8daff",
            backgroundColor: "#e2e3e5",
            borderRadius: "8px",
          }}
        >
          <h3 style={{ marginTop: 0 }}>Realizar Predicción</h3>
          <div style={{ display: "flex", gap: "10px", alignItems: "center" }}>
            <label htmlFor="inputX">
              Valor de <strong>X</strong>:
            </label>
            <input
              id="inputX"
              type="number"
              value={inputX}
              onChange={(e) => setInputX(e.target.value)}
              style={{
                padding: "8px",
                fontSize: "16px",
                width: "100px",
                borderRadius: "4px",
                border: "1px solid #ccc",
              }}
            />
            <button
              onClick={handlePredict}
              style={{
                padding: "8px 15px",
                fontSize: "16px",
                cursor: "pointer",
                backgroundColor: "#28a745",
                color: "white",
                border: "none",
                borderRadius: "5px",
              }}
            >
              Calcular Y
            </button>
          </div>

          {predictionResult !== null && (
            <div
              style={{
                marginTop: "20px",
                padding: "15px",
                backgroundColor: "#fff",
                borderLeft: "5px solid #28a745",
                fontSize: "18px",
              }}
            >
              Resultado: Si X = {inputX}, entonces{" "}
              <strong>Y = {predictionResult}</strong>
            </div>
          )}
        </section>
      )}
    </div>
  );
}

export default App;
