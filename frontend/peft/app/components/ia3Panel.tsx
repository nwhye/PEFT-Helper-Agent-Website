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
  Cell,
} from "recharts";

export default function IA3Panel() {
  const [selectedTab, setSelectedTab] = useState("ia3");
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState<any>(null);

  const [formValues, setFormValues] = useState({
    model_name: "google/flan-t5-base",
    task_type: "text generation",
    dataset: "tatsu-lab/alpaca",
    learning_rate: "1e-5",
    batch_size: "4",
    epoch: "1",
    layers_tuned: "all",
    target_modules: ["q", "v"],
  });

  const lockedFields = ["model_name", "task_type", "dataset"];
  const layerOptions = ["all", "encoder_last_3", "decoder_last_3"];
  const moduleOptions = ["q", "k", "v", "o"];

  const handleChange = (
    e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>
  ) => {
    const { name, value } = e.target;
    setFormValues((prev) => ({ ...prev, [name]: value }));
  };

  const handleModuleToggle = (module: string) => {
    setFormValues((prev) => {
      const modules = [...prev.target_modules];
      return modules.includes(module)
        ? { ...prev, target_modules: modules.filter((m) => m !== module) }
        : { ...prev, target_modules: [...modules, module] };
    });
  };

  const handleSubmit = async () => {
    try {
      setLoading(true);
      setResults(null);

      const response = await axios.post(
        "http://localhost:8002/predict/",
        formValues
      );
      setResults(response.data);
    } catch (err) {
      console.error(err);
      alert("API call failed. Make sure FastAPI is running.");
    } finally {
      setLoading(false);
    }
  };


  function getBarColor(metric: string, value: number): string {
    switch (metric) {
      case "Training Speed":
        if (value > 60000) return "#10B981";
        if (value > 30000) return "#FACC15";
        return "#EF4444";
      case "Loss Slope":
        if (Math.abs(value) < 0.0005) return "#10B981";
        if (Math.abs(value) < 0.001) return "#FACC15";
        return "#EF4444";
      case "Gradient Norm":
        if (value < 0.2) return "#10B981";
        if (value < 0.3) return "#FACC15";
        return "#EF4444";
      default:
        return "#3B82F6";
    }
  }

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

            {/* INPUT PANEL */}
            <div className="flex-1 bg-gray-900 rounded-2xl p-6 shadow-lg">
              <h2 className="text-xl font-semibold mb-4 text-gray-100">
                IA3 Hyperparameters
              </h2>

              {Object.keys(formValues)
                .filter(k => !["layers_tuned", "target_modules"].includes(k))
                .map((key) => {
                  const isLocked = lockedFields.includes(key);
                  const value = (formValues as any)[key];

                  return (
                    <div key={key} className="mb-4">
                      <label className="block text-sm font-medium mb-1 text-gray-300">
                        {key}
                      </label>
                      {isLocked ? (
                        <select
                          disabled
                          className="w-full border rounded-lg px-3 py-2 bg-gray-800 border-gray-800 text-gray-500 cursor-not-allowed"
                        >
                          <option>{value}</option>
                        </select>
                      ) : (
                        <input
                          type="text"
                          name={key}
                          value={value}
                          onChange={handleChange}
                          className="w-full border rounded-lg px-3 py-2 bg-gray-800 border-gray-700 text-gray-100"
                        />
                      )}
                    </div>
                  );
                })}

              {/* layers_tuned */}
              <div className="mb-4">
                <label className="block text-sm font-medium mb-1 text-gray-300">
                  layers_tuned
                </label>
                <select
                  name="layers_tuned"
                  value={formValues.layers_tuned}
                  onChange={handleChange}
                  className="w-full border rounded-lg px-3 py-2 bg-gray-800 border-gray-700 text-gray-100"
                >
                  {layerOptions.map(opt => (
                    <option key={opt} value={opt}>{opt}</option>
                  ))}
                </select>
              </div>

              {/* target_modules */}
              <div className="mb-4">
                <label className="block text-sm font-medium mb-1 text-gray-300">
                  target_modules
                </label>
                <div className="flex gap-2 flex-wrap">
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

              <button
                onClick={handleSubmit}
                disabled={loading}
                className="mt-4 bg-gray-700 hover:bg-gray-600 transition px-6 py-2 rounded-lg text-gray-100 disabled:opacity-50"
              >
                Predict & Generate Recommendations
              </button>
            </div>

            {/* RESULTS PANEL */}
            <div className="relative flex-1 bg-gray-900 rounded-2xl p-6 shadow-lg overflow-visible">
              <h2 className="text-xl font-semibold mb-4 text-gray-100">
                Results & Chart
              </h2>

              {loading && (
                <div className="absolute inset-0 bg-black/60 backdrop-blur-sm flex flex-col items-center justify-center z-50">
                  <div className="animate-spin rounded-full h-12 w-12 border-4 border-gray-500 border-t-transparent" />
                  <p className="text-gray-300 mt-4">Processing…</p>
                </div>
              )}

              {results ? (
                <div className={`space-y-4 text-gray-100 transition-opacity ${loading ? "opacity-20 pointer-events-none" : "opacity-100"}`}>
                  <ul className="list-disc list-inside space-y-1">
                    {results.final_recommendations.map((rec: string, i: number) => (
                      <li key={i}>{rec}</li>
                    ))}
                  </ul>

                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-4">
                    <MetricChart
                      title="Training Speed"
                      data={trainingSpeedData}
                      domain={[0, 70000]}
                      formatter={(v: number) => v.toFixed(2)}
                      getColor={getBarColor}
                      description="Higher is better. Faster, more efficient training."
                    />
                    <MetricChart
                      title="Loss Slope"
                      data={lossSlopeData}
                      domain={[-0.0025, 0.0025]}
                      formatter={(v: number) => v.toFixed(6)}
                      getColor={getBarColor}
                      description="Closer to 0 is better. Stable convergence."
                    />
                    <MetricChart
                      title="Gradient Norm"
                      data={gradientNormData}
                      domain={[-0.35, 0.35]}
                      formatter={(v: number) => v.toFixed(3)}
                      getColor={getBarColor}
                      description="Lower is safer. Avoid gradient explosion."
                    />
                  </div>
                </div>
              ) : (
                <p className="text-gray-400">
                  Select hyperparameters and click Predict to see results.
                </p>
              )}
            </div>

          </Tabs.Content>
        </Tabs.Root>
      </div>
    </div>
  );
}

function MetricChart({
  title,
  data,
  domain,
  formatter,
  getColor,
  description,
}: any) {
  return (
    <div className="bg-gray-800 rounded-lg p-4">
      <ResponsiveContainer width="100%" height={150}>
        <BarChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
          <XAxis dataKey="name" stroke="#D1D5DB" />
          <YAxis stroke="#D1D5DB" domain={domain} />
          <Tooltip
            contentStyle={{
              backgroundColor: "#1F2937",
              border: "none",
              color: "#F3F4F6",
            }}
            itemStyle={{ color: "#F3F4F6" }}
            wrapperStyle={{ zIndex: 10000 }}
            formatter={formatter}
          />
          <Bar dataKey="value" radius={[6, 6, 0, 0]}>
            {data.map((d: any, i: number) => (
              <Cell key={i} fill={getColor(title, d.value)} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
      <p className="text-gray-400 mt-5 text-sm">{description}</p>
    </div>
  );
}
