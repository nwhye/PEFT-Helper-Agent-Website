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
  layers_tuned: string;
  prefix_hidden: string;
  prefix_projection: string;
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
    layers_tuned: "0",
    prefix_hidden: "64",
    prefix_projection: "True",
  });

  const lockedFields = ["prefix_projection", "task_type", "dataset"];

  const [results, setResults] = useState<any>(null);
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
        "http://localhost:8001/predict/", // prefix API
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

        {/* Prefix Length */}
        <div className="mb-4">
          <label className="block text-gray-300 mb-1">Prefix Length</label>
          <input
            type="text"
            name="prefix_length"
            value={formValues.prefix_length}
            onChange={handleChange}
            className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-gray-100"
          />
        </div>

        {/* Prefix Dropout */}
        <div className="mb-4">
          <label className="block text-gray-300 mb-1">Prefix Dropout</label>
          <input
            type="text"
            name="prefix_dropout"
            value={formValues.prefix_dropout}
            onChange={handleChange}
            className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-gray-100"
          />
        </div>

        {/* Learning Rate */}
        <div className="mb-4">
          <label className="block text-gray-300 mb-1">Learning Rate</label>
          <input
            type="text"
            name="learning_rate"
            value={formValues.learning_rate}
            onChange={handleChange}
            className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-gray-100"
          />
        </div>

        {/* Batch Size */}
        <div className="mb-4">
          <label className="block text-gray-300 mb-1">Batch Size</label>
          <input
            type="text"
            name="batch_size"
            value={formValues.batch_size}
            onChange={handleChange}
            className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-gray-100"
          />
        </div>

        {/* Epoch */}
        <div className="mb-4">
          <label className="block text-gray-300 mb-1">Epoch</label>
          <input
            type="text"
            name="epoch"
            value={formValues.epoch}
            onChange={handleChange}
            className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-gray-100"
          />
        </div>

        {/* Layers Tuned */}
        <div className="mb-4">
          <label className="block text-gray-300 mb-1">Layers Tuned</label>
          <select
            name="layers_tuned"
            value={formValues.layers_tuned}
            onChange={handleChange}
            className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-gray-100"
          >
            <option value="0">all</option>
            <option value="1">last_3</option>
            <option value="2">first_3</option>
          </select>
        </div>

        {/* Prefix Hidden */}
        <div className="mb-4">
          <label className="block text-gray-300 mb-1">Prefix Hidden</label>
          <select
            name="prefix_hidden"
            value={formValues.prefix_hidden}
            onChange={handleChange}
            className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-gray-100"
          >
            <option value="64">64</option>
            <option value="128">128</option>
            <option value="256">256</option>
            <option value="0">None</option>
          </select>
        </div>

        {/* Prefix Projection */}
        <div className="mb-4">
          <label className="block text-gray-300 mb-1">Prefix Projection</label>
          <input
            type="text"
            name="prefix_projection"
            value="True"
            disabled
            className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-gray-100 cursor-not-allowed"
          />
        </div>

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
