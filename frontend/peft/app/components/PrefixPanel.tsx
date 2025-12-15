"use client";

import React, { useState } from "react";
import axios from "axios";
import {
  ResponsiveContainer,
  Bar,
  BarChart,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Cell
} from "recharts";

const layersTunedMap: Record<string, number> = {
  all: 0,
  last_3: 1,
  first_3: 2,
};

export default function PrefixPanel() {
  const [formValues, setFormValues] = useState({
    model_name: "google/flan-t5-base",
    task_type: "text generation",
    dataset: "tatsu-lab/alpaca",
    prefix_length: "8",
    prefix_dropout: "0.1",
    learning_rate: "1e-5",
    batch_size: "4",
    epoch: "1",
    layers_tuned: "all",
    prefix_hidden: "64",
    prefix_projection: "True",
  });

  const [results, setResults] = useState<any>(null);
  const [loading, setLoading] = useState(false);

  const options: Record<string, string[] | null> = {
    model_name: ["google/flan-t5-base"],
    task_type: ["text generation"],
    dataset: ["tatsu-lab/alpaca"],
    prefix_length: null,
    prefix_dropout: null,
    learning_rate: null,
    batch_size: null,
    epoch: null,
    layers_tuned: ["all", "last_3", "first_3"],
    prefix_hidden: ["64", "128", "256", "0"],
    prefix_projection: ["True"],
  };

  const lockedFields = ["model_name", "prefix_projection", "task_type", "dataset"];
  const freeInputFields = ["prefix_length", "prefix_dropout", "learning_rate", "batch_size", "epoch"];

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    setFormValues((prev) => ({ ...prev, [e.target.name]: e.target.value }));
  };

  const handleSubmit = async () => {
    try {
      setLoading(true);
      setResults(null);

      const payload = {
        ...formValues,
        prefix_length: Number(formValues.prefix_length),
        prefix_dropout: Number(formValues.prefix_dropout),
        learning_rate: Number(formValues.learning_rate),
        batch_size: Number(formValues.batch_size),
        epoch: Number(formValues.epoch),
        layers_tuned: layersTunedMap[formValues.layers_tuned],
        prefix_hidden: Number(formValues.prefix_hidden),
        prefix_projection: formValues.prefix_projection === "True" ? 1 : 0,
      };

      const res = await axios.post("http://localhost:8001/predict/", payload);
      setResults(res.data);
    } catch (err) {
      console.error(err);
      alert("Prefix API call failed.");
    } finally {
      setLoading(false);
    }
  };

  function getBarColor(metric: string, value: number): string {
    switch (metric) {
      case "Training Speed":
        if (value > 25000) return "#10B981";
        if (value > 8000) return "#FACC15";
        return "#EF4444";
      case "Loss Slope":
        if (value < 0.0005) return "#10B981";
        if (value < 0.0015) return "#FACC15";
        return "#EF4444";
      case "Gradient Norm":
        if (value < 0.35) return "#10B981";
        if (value < 0.43) return "#FACC15";
        return "#EF4444";
      default:
        return "#3B82F6";
    }
  }

  const trainingSpeedData = results ? [{ name: "Training Speed", value: results.training_speed }] : [];
  const lossSlopeData = results ? [{ name: "Loss Slope", value: results.loss_slope }] : [];
  const gradientNormData = results ? [{ name: "Gradient Norm", value: results.gradient_norm }] : [];

  return (
    <div className="min-h-screen bg-gray-950 flex justify-center items-start p-8">
      <div className="w-full max-w-6xl flex gap-6">

        {/* Input Panel */}
        <div className="flex-1 bg-gray-900 rounded-2xl p-6 shadow-lg">
          <h2 className="text-xl font-semibold mb-4 text-gray-100">Prefix Hyperparameters</h2>

          {Object.keys(formValues).map((key) => {
            const isLocked = lockedFields.includes(key);
            const hasOptions = Array.isArray(options[key]) && options[key]?.length;
            const value = (formValues as any)[key];
            let isValid = true;
            let errorMessage = "";

            if (freeInputFields.includes(key)) {
              switch (key) {
                case "prefix_length":
                case "epoch":
                  isValid = Number.isInteger(Number(value)) && Number(value) >= 1;
                  if (!isValid) errorMessage = "Must be integer >= 1";
                  break;
                case "batch_size":
                  isValid = Number.isInteger(Number(value)) && Number(value) > 0;
                  if (!isValid) errorMessage = "Must be positive integer";
                  break;
                case "prefix_dropout":
                  isValid = !isNaN(Number(value)) && Number(value) >= 0 && Number(value) <= 1;
                  if (!isValid) errorMessage = "Must be float 0–1";
                  break;
                case "learning_rate":
                  isValid = !isNaN(Number(value));
                  if (!isValid) errorMessage = "Must be numeric";
                  break;
              }
            }

            return (
              <div key={key} className="mb-4">
                <label className="block text-sm font-medium mb-1 text-gray-300">{key}</label>
                {hasOptions && !freeInputFields.includes(key) ? (
                  <select
                    name={key}
                    value={value}
                    onChange={handleChange}
                    disabled={isLocked}
                    className={`w-full border rounded-lg px-3 py-2 text-gray-100 ${
                      isLocked ? "bg-gray-800 border-gray-800 text-gray-500 cursor-not-allowed" : "bg-gray-800 border-gray-700"
                    }`}
                  >
                    {options[key]?.map((opt) => (
                      <option key={opt} value={opt}>{opt === "0" && key === "prefix_hidden" ? "None" : opt}</option>
                    ))}
                  </select>
                ) : (
                  <input
                    type="text"
                    name={key}
                    value={isValid ? value : ""}
                    onChange={handleChange}
                    disabled={isLocked}
                    placeholder={isValid ? "Enter value" : errorMessage}
                    className={`w-full border rounded-lg px-3 py-2 text-gray-100 ${
                      isLocked ? "bg-gray-800 border-gray-800 text-gray-500 cursor-not-allowed" : isValid ? "bg-gray-800 border-gray-700" : "bg-red-900 border-red-600 text-red-300"
                    }`}
                  />
                )}
              </div>
            );
          })}

          <button
            onClick={handleSubmit}
            disabled={Object.keys(formValues).some((k) => {
              if (!freeInputFields.includes(k)) return false;
              const v = (formValues as any)[k];
              switch (k) {
                case "prefix_length":
                case "epoch":
                  return !Number.isInteger(Number(v)) || Number(v) <= 0;
                case "batch_size":
                  return !Number.isInteger(Number(v)) || Number(v) <= 0;
                case "prefix_dropout":
                  return isNaN(Number(v)) || Number(v) < 0 || Number(v) > 1;
                case "learning_rate":
                  return isNaN(Number(v));
                default:
                  return false;
              }
            })}
            className="mt-4 bg-gray-700 text-gray-100 px-6 py-2 rounded-lg hover:bg-gray-600 transition disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Predict & Generate Recommendations
          </button>
        </div>

        {/* Results Panel */}
        <div className="relative flex-1 bg-gray-900 rounded-2xl p-6 shadow-lg overflow-visible">
          <h2 className="text-xl font-semibold mb-4 text-gray-100">Results & Charts</h2>

          {loading && (
            <div className="absolute inset-0 bg-black/60 backdrop-blur-sm flex flex-col items-center justify-center z-50">
              <div className="animate-spin rounded-full h-12 w-12 border-4 border-gray-500 border-t-transparent"></div>
              <p className="text-gray-300 mt-4">Processing…</p>
            </div>
          )}

          {results ? (
            <div className={`space-y-4 text-gray-100 transition-opacity ${loading ? "opacity-20 pointer-events-none" : ""}`}>
              {/* Recommendations */}
              <ul className="list-disc list-inside space-y-1">
                {results.final_recommendations.map((rec: string, idx: number) => (
                  <li key={idx}>{rec}</li>
                ))}
              </ul>

              {/* Charts */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-4">
                {/* Training Speed */}
                <div className="bg-gray-800 rounded-lg p-4">
                  <ResponsiveContainer width="100%" height={150}>
                    <BarChart data={trainingSpeedData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                      <XAxis dataKey="name" stroke="#D1D5DB" />
                      <YAxis stroke="#D1D5DB" domain={[Math.min(...trainingSpeedData.map(d => d.value), 0), Math.max(...trainingSpeedData.map(d => d.value), 70000)]} />
                      <Tooltip
                          contentStyle={{
                              backgroundColor: "#1F2937",
                              border: "none",
                              color: "#F3F4F6",
                            }}
                            itemStyle={{
                              color: "#F3F4F6",
                            }}
                          wrapperStyle={{ zIndex: 10000 }}
                          formatter={(value: number) => value.toFixed(2)} />
                      <Bar dataKey="value" radius={[6,6,0,0]}>
                        {trainingSpeedData.map((entry, index) => (
                          <Cell key={index} fill={getBarColor("Training Speed", entry.value)} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                  <p className="text-gray-400 mt-2 text-sm">Higher is better. Excellent &gt;25k, Good 8k–25k, Poor &lt;8k.</p>
                </div>

                {/* Loss Slope */}
                <div className="bg-gray-800 rounded-lg p-4">
                  <ResponsiveContainer width="100%" height={150}>
                    <BarChart data={lossSlopeData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                      <XAxis dataKey="name" stroke="#D1D5DB" />
                      <YAxis stroke="#D1D5DB" domain={[Math.min(...lossSlopeData.map(d => d.value), -0.0025), Math.max(...lossSlopeData.map(d => d.value), 0.0025)]} />
                      <Tooltip
                          contentStyle={{
                              backgroundColor: "#1F2937",
                              border: "none",
                              color: "#F3F4F6",
                            }}
                            itemStyle={{
                              color: "#F3F4F6",
                            }}
                          wrapperStyle={{ zIndex: 10000 }}
                          formatter={(value: number) => value.toFixed(6)} />
                      <Bar dataKey="value" radius={[6,6,0,0]}>
                        {lossSlopeData.map((entry, index) => (
                          <Cell key={index} fill={getBarColor("Loss Slope", entry.value)} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                  <p className="text-gray-400 mt-2 text-sm">Closer to 0 is better. Excellent &lt;0.0005, Good 0.0005–0.0015, Poor &gt;0.001.</p>
                </div>

                {/* Gradient Norm */}
                <div className="bg-gray-800 rounded-lg p-4">
                  <ResponsiveContainer width="100%" height={150}>
                    <BarChart data={gradientNormData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                      <XAxis dataKey="name" stroke="#D1D5DB" />
                      <YAxis stroke="#D1D5DB" domain={[Math.min(...gradientNormData.map(d => d.value), 0), Math.max(...gradientNormData.map(d => d.value), 0.35)]} />
                      <Tooltip
                          contentStyle={{
                              backgroundColor: "#1F2937",
                              border: "none",
                              color: "#F3F4F6",
                            }}
                            itemStyle={{
                              color: "#F3F4F6",
                            }}
                          wrapperStyle={{ zIndex: 10000 }}
                          formatter={(value: number) => value.toFixed(3)} />
                      <Bar dataKey="value" radius={[6,6,0,0]}>
                        {gradientNormData.map((entry, index) => (
                          <Cell key={index} fill={getBarColor("Gradient Norm", entry.value)} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                  <p className="text-gray-400 mt-2 text-sm">Lower is safer. Excellent &lt;0.35, Good 0.35–0.43, Poor &gt;0.43.</p>
                </div>
              </div>
            </div>
          ) : (
            <p className="text-gray-400">Select hyperparameters and click Predict to see results.</p>
          )}
        </div>
      </div>
    </div>
  );
}
