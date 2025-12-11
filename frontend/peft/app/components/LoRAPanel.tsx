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
  /*Legend,*/
}
from "recharts";

export default function LoRAPanel() {
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

  const options: Record<string, string[] | null> = {
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

  const lockedFields = ["task_type", "dataset"];

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

  const overfitData = results ? [{ name: "Overfit", value: results.predicted_overfit }] : [];
  const efficiencyData = results ? [{ name: "Efficiency", value: results.predicted_efficiency }] : [];
  const genGapData = results ? [{ name: "Gen.Gap", value: results.predicted_generalization_gap }] : [];

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
