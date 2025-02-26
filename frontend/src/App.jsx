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

// Register Chart.js components
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
  const [isLoading, setIsLoading] = useState(false);
  const [reweightedUrl, setReweightedUrl] = useState(null);

  const handleFileChange = (e) => setFile(e.target.files[0]);

  const handleAnalyze = async () => {
    try {
      setIsLoading(true);
      setResult(null);
      setError(null);
      setReweightedUrl(null);

      const formData = new FormData();
      formData.append("file", file);
      const res = await axios.post("http://localhost:5000/analyze", formData);
      console.log("Response:", res.data);

      if (res.data.success) {
        setResult(res.data);
      } else {
        setError(res.data.error || "Analysis failed");
      }
    } catch (e) {
      console.error("Error:", e);
      setError("Failed to analyze file: " + e.message);
    } finally {
      setIsLoading(false);
    }
  };

  const handleMitigate = async () => {
    try {
      setIsLoading(true);
      setError(null);

      const formData = new FormData();
      formData.append("file", file);
      const res = await axios.post("http://localhost:5000/mitigate", formData, {
        responseType: "blob",
      });
      const url = window.URL.createObjectURL(new Blob([res.data]));
      setReweightedUrl(url);
      handleAnalyze(); // Refresh metrics after mitigation
    } catch (e) {
      console.error("Error:", e);
      setError("Failed to mitigate bias: " + e.message);
    } finally {
      setIsLoading(false);
    }
  };

  const chartOptions = {
    scales: {
      y: {
        beginAtZero: true,
        max: 100,
        title: { display: true, text: "% Positive" },
      },
    },
    plugins: {
      legend: { position: "top" },
    },
  };

  const createChartData = (before, after, labels, title, colors) => ({
    labels,
    datasets: [
      {
        label: "Before Mitigation",
        data: before,
        backgroundColor: colors.before,
      },
      {
        label: "After Mitigation",
        data: after,
        backgroundColor: colors.after,
      },
    ],
    options: {
      ...chartOptions,
      plugins: {
        ...chartOptions.plugins,
        title: { display: true, text: title },
      },
    },
  });

  return (
    <div style={styles.container}>
      <style>
        {`
          @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
          }
        `}
      </style>

      <header style={styles.header}>
        <h1>AI Bias Mitigation Co-Pilot</h1>
        <p>
          Upload a CSV file to analyze and mitigate bias in your AI model. Get
          bias metrics and a reweighted dataset.
        </p>
      </header>

      <section style={styles.uploadSection}>
        <input type="file" onChange={handleFileChange} accept=".csv" />
        <button
          onClick={handleAnalyze}
          disabled={!file || isLoading}
          style={styles.button}
        >
          {isLoading ? "Analyzing..." : "Analyze"}
        </button>
        <button
          onClick={handleMitigate}
          disabled={!file || isLoading}
          style={styles.button}
        >
          {isLoading ? "Mitigating..." : "Mitigate Bias"}
        </button>
        {reweightedUrl && (
          <a
            href={reweightedUrl}
            download="reweighted_dataset.csv"
            style={styles.downloadLink}
          >
            Download Reweighted Dataset
          </a>
        )}
      </section>

      {isLoading && (
        <div style={styles.loadingContainer}>
          <div style={styles.spinner}></div>
          <p>Processing... Please wait.</p>
        </div>
      )}

      {error && <p style={styles.errorText}>{error}</p>}

      {result && !isLoading && (
        <section style={styles.resultsSection}>
          <h2>Results</h2>
          <div style={styles.chartsContainer}>
            <div style={styles.chartWrapper}>
              <h3 style={styles.chartTitle}>Gender Bias</h3>
              <p style={styles.chartInfo}>
                Before Mitigation: Your model favors men by{" "}
                {(
                  result.before.gender.males - result.before.gender.females
                ).toFixed(2)}
                %
              </p>
              <p style={styles.chartInfo}>
                After Mitigation: Your model favors men by{" "}
                {(
                  result.after.gender.males - result.after.gender.females
                ).toFixed(2)}
                %
              </p>
              <p style={styles.chartInfo}>
                Before DI: {result.before.gender.impact.toFixed(4)}, After DI:{" "}
                {result.after.gender.impact.toFixed(4)}
              </p>
              <p style={styles.chartInfo}>
                Accuracy: Before {result.before.accuracy.toFixed(4)}, After{" "}
                {result.after.accuracy.toFixed(4)}
              </p>
              <Bar
                data={createChartData(
                  [result.before.gender.males, result.before.gender.females],
                  [result.after.gender.males, result.after.gender.females],
                  ["Males", "Females"],
                  "Gender Bias Mitigation",
                  {
                    before: "rgba(54, 162, 235, 0.7)",
                    after: "rgba(255, 99, 132, 0.7)",
                  }
                )}
              />
            </div>

            <div style={styles.chartWrapper}>
              <h3 style={styles.chartTitle}>Race Bias</h3>
              <p style={styles.chartInfo}>
                Before Mitigation: Your model favors privileged race by{" "}
                {(
                  result.before.race.privileged -
                  result.before.race.unprivileged
                ).toFixed(2)}
                %
              </p>
              <p style={styles.chartInfo}>
                After Mitigation: Your model favors privileged race by{" "}
                {(
                  result.after.race.privileged - result.after.race.unprivileged
                ).toFixed(2)}
                %
              </p>
              <p style={styles.chartInfo}>
                Before DI: {result.before.race.impact.toFixed(4)}, After DI:{" "}
                {result.after.race.impact.toFixed(4)}
              </p>
              <p style={styles.chartInfo}>
                Accuracy: Before {result.before.accuracy.toFixed(4)}, After{" "}
                {result.after.accuracy.toFixed(4)}
              </p>
              <Bar
                data={createChartData(
                  [
                    result.before.race.privileged,
                    result.before.race.unprivileged,
                  ],
                  [
                    result.after.race.privileged,
                    result.after.race.unprivileged,
                  ],
                  ["Privileged", "Unprivileged"],
                  "Race Bias Mitigation",
                  {
                    before: "rgba(139, 69, 19, 0.7)",
                    after: "rgba(222, 184, 135, 0.7)",
                  }
                )}
              />
            </div>

            <div style={styles.chartWrapper}>
              <h3 style={styles.chartTitle}>Age Bias</h3>
              <p style={styles.chartInfo}>
                Before Mitigation: Your model favors older individuals by{" "}
                {(result.before.age.old - result.before.age.young).toFixed(2)}%
              </p>
              <p style={styles.chartInfo}>
                After Mitigation: Your model favors older individuals by{" "}
                {(result.after.age.old - result.after.age.young).toFixed(2)}%
              </p>
              <p style={styles.chartInfo}>
                Before DI: {result.before.age.impact.toFixed(4)}, After DI:{" "}
                {result.after.age.impact.toFixed(4)}
              </p>
              <p style={styles.chartInfo}>
                Accuracy: Before {result.before.accuracy.toFixed(4)}, After{" "}
                {result.after.accuracy.toFixed(4)}
              </p>
              <Bar
                data={createChartData(
                  [result.before.age.old, result.before.age.young],
                  [result.after.age.old, result.after.age.young],
                  ["Old (>=40)", "Young (<40)"],
                  "Age Bias Mitigation",
                  {
                    before: "rgba(255, 165, 0, 0.7)",
                    after: "rgba(50, 205, 50, 0.7)",
                  }
                )}
              />
            </div>
          </div>
        </section>
      )}
    </div>
  );
}

const styles = {
  container: {
    padding: "20px",
    fontFamily: "Arial, sans-serif",
    maxWidth: "1200px",
    margin: "0 auto",
  },
  header: {
    textAlign: "center",
    marginBottom: "30px",
  },
  uploadSection: {
    textAlign: "center",
    marginBottom: "20px",
  },
  button: {
    marginLeft: "10px",
    padding: "8px 16px",
    backgroundColor: "#007bff",
    color: "#fff",
    border: "none",
    borderRadius: "5px",
    cursor: "pointer",
  },
  downloadLink: {
    marginLeft: "10px",
    color: "#007bff",
    textDecoration: "underline",
  },
  errorText: {
    color: "red",
    textAlign: "center",
  },
  resultsSection: {
    marginTop: "40px",
  },
  chartsContainer: {
    display: "flex",
    justifyContent: "space-around",
    gap: "20px",
    flexWrap: "wrap",
  },
  chartWrapper: {
    width: "30%",
    minWidth: "300px",
    padding: "10px",
    border: "1px solid #ddd",
    borderRadius: "8px",
    backgroundColor: "#f9f9f9",
  },
  chartTitle: {
    textAlign: "center",
    marginBottom: "10px",
  },
  chartInfo: {
    textAlign: "center",
    fontSize: "14px",
    margin: "5px 0",
  },
  loadingContainer: {
    textAlign: "center",
    marginTop: "20px",
    fontSize: "18px",
    color: "#007bff",
  },
  spinner: {
    margin: "20px auto",
    width: "50px",
    height: "50px",
    border: "6px solid #f3f3f3",
    borderTop: "6px solid #007bff",
    borderRadius: "50%",
    animation: "spin 1s cubic-bezier(0.4, 0, 0.2, 1) infinite",
  },
};

export default App;
