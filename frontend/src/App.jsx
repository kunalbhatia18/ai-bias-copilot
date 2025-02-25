import { useState } from "react";
import axios from "axios";
import { Bar } from "react-chartjs-2";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
} from "chart.js";

ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend
);

function App() {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleFileChange = (e) => setFile(e.target.files[0]);

  const handleSubmit = async () => {
    try {
      const formData = new FormData();
      formData.append("file", file);
      const res = await axios.post("http://localhost:5000/analyze", formData);
      console.log("Response:", res.data); // Debug log
      if (res.data.success) {
        setResult(res.data);
        setError(null);
      } else {
        setError(res.data.error || "Analysis failed");
      }
    } catch (e) {
      console.error("Error:", e); // Debug error
      setError("Failed to analyze file: " + e.message);
    }
  };

  const chartData = result
    ? {
        labels: ["Males", "Females"],
        datasets: [
          {
            label: "Before Mitigation",
            data: [result.before.males, result.before.females], // Already percentages
            backgroundColor: "rgba(255, 99, 132, 0.5)",
          },
          {
            label: "After Mitigation",
            data: [result.after.males, result.after.females], // Already percentages
            backgroundColor: "rgba(54, 162, 235, 0.5)",
          },
        ],
      }
    : null;

  return (
    <div style={{ padding: "20px", fontFamily: "Arial" }}>
      <h1>AI Bias Mitigation Co-Pilot</h1>
      <input type="file" onChange={handleFileChange} accept=".csv" />
      <button onClick={handleSubmit} disabled={!file}>
        Analyze
      </button>
      {error && <p style={{ color: "red" }}>{error}</p>}
      {result && (
        <div>
          <h2>Results</h2>
          <p>Before: Impact {result.before.impact.toFixed(4)}</p>
          <p>After: Impact {result.after.impact.toFixed(4)}</p>
          <Bar
            data={chartData}
            options={{
              scales: {
                y: {
                  beginAtZero: true,
                  max: 100,
                  title: { display: true, text: "% Positive" },
                },
              },
              plugins: {
                legend: { position: "top" },
                title: { display: true, text: "Bias Mitigation Results" },
              },
            }}
          />
        </div>
      )}
    </div>
  );
}

export default App;
