import React from "react";
import LiveChart from "./components/LiveChart";
import SystemChart from "./components/SystemChart";
import TrainingChart from "./components/TrainingChart";
import "./Dashboard.css";

function App() {
  return (
    <div>
      <header className="dashboard-header">
        <h1>Smart Monitoring Dashboard</h1>
        <p>Real-time System, Live Agent, and Training Metrics</p>
      </header>

      <div className="dashboard-container">
        <SystemChart />
        <LiveChart />
        <TrainingChart />
      </div>
    </div>
  );
}

export default App;
