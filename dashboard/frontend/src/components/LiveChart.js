import React, { useEffect, useState } from "react";
import { Line } from "react-chartjs-2";
import { Chart, LineElement, PointElement, CategoryScale, LinearScale, Legend, Tooltip } from "chart.js";

Chart.register(LineElement, PointElement, CategoryScale, LinearScale, Legend, Tooltip);

const API = "http://localhost:5000/api/live";

export default function LiveChart() {
  const [data, setData] = useState([]);

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 10000);
    return () => clearInterval(interval);
  }, []);

  async function fetchData() {
    try {
      const res = await fetch(API);
      const json = await res.json();
      setData(json);
    } catch (e) {
      console.error("Error fetching live data:", e);
    }
  }

  return (
    <div className="card">
      <h3>RL Agent Live Metrics</h3>
      <div className="chart-container">
        <Line
          data={{
            labels: data.map((d) => d.timestamp),
            datasets: [
              {
                label: "Reward",
                data: data.map((d) => d.reward),
                borderColor: "#22c55e",
                tension: 0.3,
                fill: false,
              },
              {
                label: "Gamma",
                data: data.map((d) => d.gamma),
                borderColor: "#f97316",
                tension: 0.3,
                fill: false,
              },
            ],
          }}
          options={{
            plugins: { legend: { position: "bottom" } },
            maintainAspectRatio: false,
          }}
        />
      </div>
    </div>
  );
}
