import { useEffect, useState } from "react";
import axios from "axios";

function App() {
  const [message, setMessage] = useState("");

  useEffect(() => {
    axios
      .get("http://127.0.0.1:5000/")
      .then((response) => setMessage(response.data.message))
      .catch((error) => console.error("API error:", error));
  }, []);

  return (
    <div className="flex flex-col items-center justify-center h-screen bg-gray-100">
      <h1 className="text-2xl font-bold text-gray-800">
        Bias Co-Pilot Frontend
      </h1>
      <p className="mt-4 text-lg text-blue-600">{message}</p>
    </div>
  );
}

export default App;
