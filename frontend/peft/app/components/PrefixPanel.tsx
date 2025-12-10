"use client";

import axios from "axios";
import React, { useState } from "react";

type PrefixFormValues = {
  task_type: string;
  dataset: string;
  prefix_length: string;
  prefix_dropout: string;
  learning_rate: string;
  batch_size: string;
  epoch: string;
};

export default function PrefixPanel() {
  const [formValues, setFormValues] = useState<PrefixFormValues>({
    task_type: "text generation",
    dataset: "tatsu-lab/alpaca",
    prefix_length: "8",
    prefix_dropout: "0.1",
    learning_rate: "1e-5",
    batch_size: "4",
    epoch: "1",
  });

  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleChange = (
    e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>
  ) => {
    setFormValues({ ...formValues, [e.target.name]: e.target.value });
  };


  const predictPrefix = async () => {
    setLoading(true);
    setResults(null);

    try {
      const response = await axios.post(
        "http://localhost:8001/predict/",  // prefix API
        formValues
      );
      setResults(response.data);
    } catch (err) {
      console.error(err);
      alert("Prefix API call failed.");
    }

    setLoading(false);
  };

  return (
    <div className="flex gap-6">
      {/* Form */}
      <div className="flex-1 bg-gray-900 rounded-2xl p-6 shadow-lg">
        <h2 className="text-xl mb-4 text-gray-100">Prefix Hyperparameters</h2>

          {(Object.keys(formValues) as Array<keyof PrefixFormValues>).map((key) => (
            <div key={key}>
              <label>{key}</label>
              <input
                name={key}
                value={formValues[key]}
                onChange={handleChange}
              className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-gray-100"
            />
          </div>
        ))}

        <button
          onClick={predictPrefix}
          className="mt-4 bg-gray-700 text-gray-100 px-6 py-2 rounded-lg hover:bg-gray-600"
        >
          Predict Prefix Performance
        </button>
      </div>

      {/* Results */}
      <div className="flex-1 bg-gray-900 rounded-2xl p-6 shadow-lg">
        <h2 className="text-xl mb-4 text-gray-100">Prefix Results</h2>

        {loading && <p className="text-gray-400">Loading...</p>}

        {results && (
          <pre className="text-gray-300 text-sm">
            {JSON.stringify(results, null, 2)}
          </pre>
        )}
      </div>
    </div>
  );
}
