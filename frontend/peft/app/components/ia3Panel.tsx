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
} from "recharts";

export default function IA3Panel() {
  const [selectedTab, setSelectedTab] = useState("ia3");
  const [formValues, setFormValues] = useState({
    model_name: "google/flan-t5-base",
    task_type: "text generation",
    dataset: "tatsu-lab/alpaca",
    learning_rate: "1e-5",
    batch_size: "4",
    epoch: "1",
    layers_tuned: "all",
    target_modules: ["q","v"],
  });

  const [results, setResults] = useState<any>(null);
  const [loading, setLoading] = useState(false);

  const lockedFields = ["model_name", "task_type", "dataset"];
  const layerOptions = ["all", "encoder_last_3", "decoder_last_3"];
  const moduleOptions = ["q", "k", "v", "o"];

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value } = e.target;
    setFormValues(prev => ({ ...prev, [name]: value }));
  };

  const handleModuleToggle = (module: string) => {
    setFormValues(prev => {
      const modules = [...prev.target_modules];
      if (modules.includes(module)) {
        return { ...prev, target_modules: modules.filter(m => m !== module) };
      } else {
        return { ...prev, target_modules: [...modules, module] };
      }
    });
  };

  const handleSubmit = async () => {
    try {
      setLoading(true);
      setResults(null);

      const payload = { ...formValues };
      const response = await axios.post("http://localhost:8002/predict/", payload);
      setResults(response.data);
    } catch (err) {
      console.error(err);
      alert("API call failed. Make sure FastAPI is running at localhost:8000.");
    } finally {
      setLoading(false);
    }
  };

  const getBarColor = (metric: string, value: number) => {
    switch(metric) {
      case "Training Speed":
        if(value > 60000) return "#10B981";
        if(value > 30000) return "#FACC15";
        return "#EF4444";
      case "Loss Slope":
        if(Math.abs(value) < 0.0005) return "#10B981";
        if(Math.abs(value) < 0.001) return "#FACC15";
        return "#EF4444";
      case "Gradient Norm":
        if(value < 0.2) return "#10B981";
        if(value < 0.3) return "#FACC15";
        return "#EF4444";
      default:
        return "#3B82F6";
    }
  };

  const trainingSpeedData = results
  ? [{ name: "Training Speed", value: results.predicted_metrics.training_speed }]
  : [];
const lossSlopeData = results
  ? [{ name: "Loss Slope", value: results.predicted_metrics.loss_slope }]
  : [];
const gradientNormData = results
  ? [{ name: "Gradient Norm", value: results.predicted_metrics.gradient_norm }]
  : [];


  return (
    <div className="min-h-screen bg-gray-950 flex justify-center items-start p-8">
      <div className="w-full max-w-6xl">
        <Tabs.Root value={selectedTab} onValueChange={setSelectedTab}>
          <Tabs.Content value="ia3" className="flex gap-6">

            {/* Input Panel */}
            <div className="flex-1 bg-gray-900 rounded-2xl p-6 shadow-lg">
              <h2 className="text-xl font-semibold mb-4 text-gray-100">IA3 Hyperparameters</h2>

            {/* Standard Inputs */}
{Object.keys(formValues).map(key => {
  const isLocked = lockedFields.includes(key);
  const value = (formValues as any)[key];

  return (
    <div key={key} className="mb-4">
      <label className="block text-sm font-medium mb-1 text-gray-300">{key}</label>
      {isLocked ? (
        <select value={value} disabled className="w-full border rounded-lg px-3 py-2 bg-gray-800 border-gray-800 text-gray-500 cursor-not-allowed">
          <option>{value}</option>
        </select>
      ) : (
        <input
          type="text"
          name={key}
          value={value}
          onChange={handleChange}
          className="w-full border rounded-lg px-3 py-2 text-gray-100 bg-gray-800 border-gray-700"
        />
      )}
    </div>
  );
})}


            {/* Layers Tuned Dropdown */}
            <div className="mb-4">
              <label className="block text-sm font-medium mb-1 text-gray-300">layers_tuned</label>
              <select
                name="layers_tuned"
                value={formValues.layers_tuned}
                onChange={handleChange}
                disabled={lockedFields.includes("layers_tuned")} // can remain editable if you prefer
                className={`w-full border rounded-lg px-3 py-2 text-gray-100 ${
                  lockedFields.includes("layers_tuned")
                    ? "bg-gray-800 border-gray-800 text-gray-500 cursor-not-allowed"
                    : "bg-gray-800 border-gray-700"
                }`}
              >
                {layerOptions.map(opt => <option key={opt} value={opt}>{opt}</option>)}
              </select>
            </div>

              {/* Target Modules Multi-select */}
              <div className="mb-4">
                <label className="block text-sm font-medium mb-1 text-gray-300">target_modules</label>
                <div className="flex flex-wrap gap-2">
                  {moduleOptions.map(module => (
                    <button
                      key={module}
                      type="button"
                      onClick={() => handleModuleToggle(module)}
                      className={`px-3 py-1 rounded-lg border ${
                        formValues.target_modules.includes(module)
                          ? "bg-green-600 border-green-600 text-white"
                          : "bg-gray-800 border-gray-700 text-gray-100"
                      }`}
                    >
                      {module}
                    </button>
                  ))}
                </div>
              </div>

              {/* Submit */}
              <button
                onClick={handleSubmit}
                disabled={Object.keys(formValues).some(key => ["learning_rate","batch_size","epoch"].includes(key) && (isNaN(Number((formValues as any)[key])) || Number((formValues as any)[key]) < 0))}
                className="mt-4 bg-gray-700 text-gray-100 px-6 py-2 rounded-lg hover:bg-gray-600 transition disabled:opacity-50 disabled:cursor-not-allowed"
              >
                Predict & Generate Recommendations
              </button>
            </div>

            {/* Results Panel */}
            <div className="relative flex-1 bg-gray-900 rounded-2xl p-6 shadow-lg overflow-visible">
              <h2 className="text-xl font-semibold mb-4 text-gray-100">Results & Chart</h2>

              {loading && (
                <div className="absolute inset-0 bg-black/60 backdrop-blur-sm flex flex-col items-center justify-center z-50">
                  <div className="animate-spin rounded-full h-12 w-12 border-4 border-gray-500 border-t-transparent"></div>
                  <p className="text-gray-300 mt-4">Processing…</p>
                </div>
              )}

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

function ChartBlock({ title, data, getColor, metric }: any) {
  return (
    <div className="bg-gray-800 rounded-lg p-4">
      <ResponsiveContainer width="100%" height={150}>
        <BarChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
          <XAxis dataKey="name" stroke="#D1D5DB" />
          <YAxis stroke="#D1D5DB" />
          <Tooltip contentStyle={{ backgroundColor: "#1F2937", border: "none", color: "#F3F4F6" }} itemStyle={{ color: "#F3F4F6" }} formatter={(value: number) => value.toFixed(3)} />
          <Bar dataKey="value" radius={[6, 6, 0, 0]}>
            {data.map((entry: any, index: number) => <Cell key={index} fill={getColor(metric, entry.value)} />)}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
      <p className="text-gray-400 mt-5 text-sm">{title}</p>
    </div>
  );
}
