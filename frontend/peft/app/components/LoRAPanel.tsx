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
  Cell
  /*Legend,*/
}
from "recharts";

export default function LoRAPanel() {
  const [selectedTab, setSelectedTab] = useState("lora");
  const [formValues, setFormValues] = useState({
    model_name: "google/flan-t5-base",
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

  const options: Record<string, string[] | null> = {
    model_name: ["google/flan-t5-base"],
    task_type: ["text generation"],
    dataset: ["tatsu-lab/alpaca"],
    lora_r: null,
    lora_alpha: null,
    lora_dropout: null,
    learning_rate: null,
    batch_size: null,
    epoch: null,
    target_modules: ["['q','v']", "['q','v','o']"],
  };

  const lockedFields = ["model_name", "task_type", "dataset"];

const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
  const { name, value } = e.target;
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

  function getBarColor(metric: string, value: number): string {
    switch(metric) {
      case "Training Speed":
        if(value > 60000) return "#10B981"; // green
        if(value > 30000) return "#FACC15"; // yellow
        return "#EF4444"; // red
      case "Loss Slope":
        if(Math.abs(value) < 0.0005) return "#10B981"; // stable
        if(Math.abs(value) < 0.001) return "#FACC15"; // moderate
        return "#EF4444"; // unstable
      case "Gradient Norm":
        if(value < 0.2) return "#10B981"; // safe
        if(value < 0.3) return "#FACC15"; // caution
        return "#EF4444"; // high
      default:
        return "#3B82F6";
    }
  }


  const trainingSpeedData = results ? [{ name: "Training Speed", value: results.training_speed }] : [];
  const lossSlopeData     = results ? [{ name: "Loss Slope", value: results.loss_slope }] : [];
  const gradientNormData  = results ? [{ name: "Gradient Norm", value: results.gradient_norm }] : [];


  const [loading, setLoading] = useState(false);

  return (
    <div className="min-h-screen bg-gray-950 flex justify-center items-start p-8">
      <div className="w-full max-w-6xl">
        <Tabs.Root value={selectedTab} onValueChange={setSelectedTab}>


          <Tabs.Content value="lora" className="flex gap-6">
            {/* Input Panel */}
            <div className="flex-1 bg-gray-900 rounded-2xl p-6 shadow-lg">
              <h2 className="text-xl font-semibold mb-4 text-gray-100">LoRA Hyperparameters</h2>

              {/* Define free input fields at the top so both map and button can use */}
              {(() => {
                const freeInputFields = [
                  "lora_r",
                  "lora_alpha",
                  "lora_dropout",
                  "learning_rate",
                  "batch_size",
                  "epoch",
                ];

                return (
                  <>
                    {Object.keys(formValues).map((key) => {
                      const isLocked = lockedFields.includes(key);
                      const isFreeInput = freeInputFields.includes(key);
                      /*const hasOptions = options[key] && Array.isArray(options[key]) && options[key].length > 0;*/

                      // Validation logic
                      const value = (formValues as any)[key];
                      let isValid = true;
                      let errorMessage = "";

                      if (isFreeInput) {
                        switch (key) {
                          case "lora_r":
                          case "lora_alpha":
                          case "epoch":
                            isValid = Number.isInteger(Number(value)) && (key !== "epoch" || Number(value) >= 1);
                            if (!isValid) errorMessage = key === "epoch" ? "Must be integer >= 1" : "Must be integer";
                            break;
                          case "batch_size":
                            isValid = Number.isInteger(Number(value)) && Number(value) > 0;
                            if (!isValid) errorMessage = "Must be positive integer";
                            break;
                          case "lora_dropout":
                            isValid = !isNaN(Number(value)) && Number(value) >= 0 && Number(value) <= 1;
                            if (!isValid) errorMessage = "Must be float between 0 and 1";
                            break;
                          case "learning_rate":
                        }
                      }

                      return (
                        <div key={key} className="mb-4">
                          <label className="block text-sm font-medium mb-1 text-gray-300">{key}</label>

                          {isFreeInput ? (
                            <input
                              type="text"
                              name={key}
                              value={isValid ? value : ""}
                              onChange={handleChange}
                              disabled={isLocked}
                              placeholder={isValid ? "Enter any value" : errorMessage}
                              className={`w-full border rounded-lg px-3 py-2 text-gray-100 ${
                                isLocked
                                  ? "bg-gray-800 border-gray-800 text-gray-500 cursor-not-allowed"
                                  : isValid
                                  ? "bg-gray-800 border-gray-700"
                                  : "bg-red-900 border-red-600 text-red-300"
                              }`}
                            />
                          ) : (
                            <select
                              name={key}
                              value={value}
                              onChange={handleChange}
                              disabled={isLocked}
                              className={`w-full border rounded-lg px-3 py-2 text-gray-100 ${
                                isLocked
                                  ? "bg-gray-800 border-gray-800 text-gray-500 cursor-not-allowed"
                                  : "bg-gray-800 border-gray-700"
                              }`}
                            >
                              {options[key]?.map((opt) => (
                                <option key={opt} value={opt}>
                                  {opt}
                                </option>
                              )) ?? <option value="">No options</option>}

                            </select>
                          )}
                        </div>
                      );
                    })}

                    <button
                      onClick={handleSubmit}
                      disabled={Object.keys(formValues).some((key) => {
                        if (!freeInputFields.includes(key)) return false;
                        const val = (formValues as any)[key];
                        switch (key) {
                          case "lora_r":
                          case "lora_alpha":
                          case "epoch":
                            return !Number.isInteger(Number(val)) || (key === "epoch" && Number(val) <= 0);
                          case "batch_size":
                            return !Number.isInteger(Number(val)) || Number(val) <= 0;
                          case "lora_dropout":
                            return isNaN(Number(val)) || Number(val) < 0 || Number(val) > 1;
                          case "learning_rate":
                            return isNaN(Number(val));
                          default:
                            return false;
                        }
                      })}
                      className="mt-4 bg-gray-700 text-gray-100 px-6 py-2 rounded-lg hover:bg-gray-600 transition disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                      Predict & Generate Recommendations
                    </button>
                  </>
                );
              })()}
            </div>

            {/* Results Panel */}
            <div className="relative flex-1 bg-gray-900 rounded-2xl p-6 shadow-lg overflow-visible">
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
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-4 ">
                    {/* Training Speed */}
                    <div className="bg-gray-800 rounded-lg p-4">
                      <ResponsiveContainer width="100%" height={150}>
                        <BarChart data={trainingSpeedData}>
                          <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                          <XAxis dataKey="name" stroke="#D1D5DB" />
                          <YAxis
                            stroke="#D1D5DB"
                            domain={[
                              Math.min(...trainingSpeedData.map(d => d.value), 0),
                              Math.max(...trainingSpeedData.map(d => d.value), 70000)
                            ]}
                          />
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
                            formatter={(value: number) => value.toFixed(2)}
                          />
                          <Bar dataKey="value" radius={[6, 6, 0, 0]}>
                            {trainingSpeedData.map((entry, index) => (
                              <Cell key={`cell-${index}`} fill={getBarColor("Training Speed", entry.value)} />
                            ))}
                          </Bar>
                        </BarChart>
                      </ResponsiveContainer>
                      <p className="text-gray-400 mt-5 text-sm">
                        Higher is better. Faster, more efficient training. Excellent {'>'} 60k, Good 30k–60k, Poor {'<'} 30k.
                      </p>
                    </div>

                    {/* Loss Slop */}
                    <div className="bg-gray-800 rounded-lg p-4">
                      <ResponsiveContainer width="100%" height={150}>
                        <BarChart data={lossSlopeData}>
                          <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                          <XAxis dataKey="name" stroke="#D1D5DB" />
                          <YAxis
                            stroke="#D1D5DB"
                            domain={[
                              Math.min(...lossSlopeData.map(d => d.value), -0.0025),
                              Math.max(...lossSlopeData.map(d => d.value), 0.0025)
                            ]}
                          />
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
                            formatter={(value: number) => value.toFixed(6)}
                          />
                          <Bar dataKey="value" radius={[6, 6, 0, 0]}>
                            {lossSlopeData.map((entry, index) => (
                              <Cell key={`cell-${index}`} fill={getBarColor("Loss Slope", entry.value)} />
                            ))}
                          </Bar>
                        </BarChart>
                      </ResponsiveContainer>
                      <p className="text-gray-400 mt-5 text-sm">
                        Closer to 0 is better. Stable convergence. Excellent {'<'} 0.0005, Good 0.0005–0.001, Poor {'>'} 0.001."
                      </p>
                    </div>

                    {/* Gradient Norm */}
                    <div className="bg-gray-800 rounded-lg p-4">
                      <ResponsiveContainer width="100%" height={150}>
                        <BarChart data={gradientNormData}>
                          <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                          <XAxis dataKey="name" stroke="#D1D5DB" />
                          <YAxis
                            stroke="#D1D5DB"
                            domain={[
                              Math.min(...gradientNormData.map(d => d.value), -0.35),
                              Math.max(...gradientNormData.map(d => d.value), 0.35)
                            ]}
                          />
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
                            formatter={(value: number) => value.toFixed(3)}
                          />
                          <Bar dataKey="value" radius={[6, 6, 0, 0]}>
                            {gradientNormData.map((entry, index) => (
                              <Cell key={`cell-${index}`} fill={getBarColor("Gradient Norm", entry.value)} />
                            ))}
                          </Bar>
                        </BarChart>
                      </ResponsiveContainer>
                      <p className="text-gray-400 mt-5 text-sm">
                        Lower is safer. Avoid gradient explosion. Excellent {'<'} 0.2, Good 0.2–0.3, Poor {'>'} 0.3.
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
