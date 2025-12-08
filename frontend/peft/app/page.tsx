"use client";

import React, { useState } from "react";
import * as Tabs from "@radix-ui/react-tabs";
import axios from "axios";
import {
  ResponsiveContainer,
  Bar,
  BarChart,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
}
from "recharts";

export default function HomePage() {
  const [selectedTab, setSelectedTab] = useState("lora");
  const [formValues, setFormValues] = useState({
    task_type: "text generation",
    dataset: "tatsu-lab/alpaca ",
    lora_r: "4",
    lora_alpha: "8",
    lora_dropout: "0.05",
    learning_rate: "1e-6",
    batch_size: "4",
    epoch: "1",
    target_modules: "['q','v']",
  });

  const [results, setResults] = useState<any>(null);

  const options: Record<string, string[]> = {
    task_type: ["text generation"],
    dataset: ["tatsu-lab/alpaca "],
    lora_r: ["4", "8", "16", "32"],
    lora_alpha: ["4", "8", "16"],
    lora_dropout: ["0.05", "0.1", "0.2"],
    learning_rate: ["1e-5", "1e-6", "3e-6", "5e-6"],
    batch_size: ["4", "8", "16"],
    epoch: ["1", "3", "5"],
    target_modules: ["['q','v']", "['q','v','o']"],
  };

  const lockedFields = ["task_type", "dataset"];

  const handleChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const { name, value } = e.target;
    if (lockedFields.includes(name)) return;
    setFormValues((prev) => ({ ...prev, [name]: value }));
  };

const handleSubmit = async () => {
  try {
    setLoading(true);
    setResults(null);

    const payload = {
      ...formValues,
      layers_tuned: 0,
    };

    const response = await axios.post("http://localhost:8000/predict/", payload);
    setResults(response.data);
  } catch (err) {
    console.error(err);
    alert("API call failed. Make sure FastAPI is running at localhost:8000.");
  } finally {
    setLoading(false);
  }
};

  const overfitData = results ? [{ name: "Overfit", value: results.predicted_overfit }] : [];
  const efficiencyData = results ? [{ name: "Efficiency", value: results.predicted_efficiency }] : [];
  const genGapData = results ? [{ name: "Gen.Gap", value: results.predicted_generalization_gap }] : [];

  const [loading, setLoading] = useState(false);

  return (
    <div className="min-h-screen bg-gray-950 flex justify-center items-start p-8">
      <div className="w-full max-w-6xl">
        <Tabs.Root value={selectedTab} onValueChange={setSelectedTab}>
          <Tabs.List className="flex border-b border-gray-700 mb-6">
            <Tabs.Trigger
              value="lora"
              className="px-6 py-2 rounded-t-lg bg-gray-900 border border-b-0 border-gray-700 mr-2 text-gray-200 hover:bg-gray-800"
            >
              LoRA
            </Tabs.Trigger>
          </Tabs.List>

          <Tabs.Content value="lora" className="flex gap-6">
            {/* Input Panel */}
            <div className="flex-1 bg-gray-900 rounded-2xl p-6 shadow-lg">
              <h2 className="text-xl font-semibold mb-4 text-gray-100">Hyperparameters</h2>

              {Object.keys(formValues).map((key) => {
                const isLocked = lockedFields.includes(key);

                return (
                  <div key={key} className="mb-4">
                    <label className="block text-sm font-medium mb-1 text-gray-300">
                      {key}
                    </label>

                    <select
                      name={key}
                      value={(formValues as any)[key]}
                      onChange={handleChange}
                      disabled={isLocked}
                      className={`w-full border rounded-lg px-3 py-2 text-gray-100
                        ${isLocked
                          ? "bg-gray-800 border-gray-800 text-gray-500 cursor-not-allowed"
                          : "bg-gray-800 border-gray-700"}
                      `}
                    >
                      {options[key].map((opt) => (
                        <option key={opt} value={opt}>
                          {opt}
                        </option>
                      ))}
                    </select>
                  </div>
                );
              })}

              <button
                onClick={handleSubmit}
                className="mt-4 bg-gray-700 text-gray-100 px-6 py-2 rounded-lg hover:bg-gray-600 transition"
              >
                Predict & Generate Recommendations
              </button>
            </div>

{/* Results Panel */}
<div className="relative flex-1 bg-gray-900 rounded-2xl p-6 shadow-lg overflow-hidden">
  <h2 className="text-xl font-semibold mb-4 text-gray-100">Results & Chart</h2>

  {/* Loading Overlay */}
  {loading && (
    <div className="absolute inset-0 bg-black/60 backdrop-blur-sm flex flex-col items-center justify-center z-50">
      <div className="animate-spin rounded-full h-12 w-12 border-4 border-gray-500 border-t-transparent"></div>
      <p className="text-gray-300 mt-4">Processing…</p>
    </div>
  )}

  {/* Content */}
  {results ? (
    <div className={`space-y-4 text-gray-100 transition-opacity ${loading ? "opacity-20 pointer-events-none" : "opacity-100"}`}>
      {/* Recommendations */}
      <ul className="list-disc list-inside space-y-1">
        {results.final_recommendations.map((rec: string, idx: number) => (
          <li key={idx}>{rec}</li>
        ))}
      </ul>

      {/* Charts */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-4">
        {/* Overfit */}
        <div className="bg-gray-800 rounded-lg p-4">
          <ResponsiveContainer width="100%" height={150}>
            <BarChart data={overfitData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis dataKey="name" stroke="#D1D5DB" />
              <YAxis stroke="#D1D5DB" domain={[-1, 1]} />
              <Tooltip
                contentStyle={{
                  backgroundColor: "#1F2937",
                  border: "none",
                  color: "#F3F4F6",
                }}
                formatter={(value: number) => value.toFixed(2)}
              />
              <Bar dataKey="value" fill="#EF4444" radius={[6, 6, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
          <p className="text-gray-400 mt-5 text-sm">
            The goal: Low (closer to 0). Lower values indicate better generalization.
          </p>
        </div>

        {/* Efficiency */}
        <div className="bg-gray-800 rounded-lg p-4">
          <ResponsiveContainer width="100%" height={150}>
            <BarChart data={efficiencyData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis dataKey="name" stroke="#D1D5DB" />
              <YAxis stroke="#D1D5DB" domain={[-1, 1]} />
              <Tooltip
                contentStyle={{
                  backgroundColor: "#1F2937",
                  border: "none",
                  color: "#F3F4F6",
                }}
                formatter={(value: number) => value.toFixed(2)}
              />
              <Bar dataKey="value" fill="#10B981" radius={[6, 6, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
          <p className="text-gray-400 mt-5 text-sm">
            The goal: High (closer to 1). Higher values mean faster, more efficient training.
          </p>
        </div>

        {/* Generalization Gap */}
        <div className="bg-gray-800 rounded-lg p-4">
          <ResponsiveContainer width="100%" height={150}>
            <BarChart data={genGapData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis dataKey="name" stroke="#D1D5DB" />
              <YAxis stroke="#D1D5DB" domain={[-1, 1]} />
              <Tooltip
                contentStyle={{
                  backgroundColor: "#1F2937",
                  border: "none",
                  color: "#F3F4F6",
                }}
                formatter={(value: number) => value.toFixed(2)}
              />
              <Bar dataKey="value" fill="#3B82F6" radius={[6, 6, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
          <p className="text-gray-400 mt-5 text-sm">
            The goal: Low (closer to 0). Smaller gap indicates better generalization to unseen data.
          </p>
        </div>
      </div>
    </div>
  ) : (
    <p className="text-gray-400">Select hyperparameters and click Predict to see results.</p>
  )}
</div>

          </Tabs.Content>
        </Tabs.Root>
      </div>
    </div>
  );
}
