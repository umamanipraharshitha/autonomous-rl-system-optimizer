import React, { useEffect, useState } from "react";
import { Line } from "react-chartjs-2";
import { Chart, LineElement, PointElement, CategoryScale, LinearScale, Legend, Tooltip } from "chart.js";

Chart.register(LineElement, PointElement, CategoryScale, LinearScale, Legend, Tooltip);

const API = "http://localhost:5000/api/system";

export default function SystemChart() {
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
      console.error("Error fetching system data:", e);
    }
  }

  return (
    <div className="card">
      <h3>System Metrics</h3>
      <div className="chart-container">
        <Line
          data={{
            labels: data.map((d) => d.timestamp),
            datasets: [
              {
                label: "CPU %",
                data: data.map((d) => d.cpu_percent),
                borderColor: "#ff5c5c",
                tension: 0.3,
                fill: false,
              },
              {
                label: "RAM %",
                data: data.map((d) => d.virtual_memory_percent),
                borderColor: "#4b9eff",
                tension: 0.3,
                fill: false,
              },
              {
                label: "Disk %",
                data: data.map((d) => d.disk_percent),
                borderColor: "#888888",
                tension: 0.3,
                fill: false,
              },
            ],
          }}
          options={{
            plugins: { legend: { position: "bottom" } },
            scales: { y: { beginAtZero: true, max: 100 } },
            maintainAspectRatio: false,
          }}
        />
      </div>
    </div>
  );
}
