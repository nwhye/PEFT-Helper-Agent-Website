"use client";

import React, { useState } from "react";
import * as Tabs from "@radix-ui/react-tabs";

import LoRAPanel from "./components/LoRAPanel";
import PrefixPanel from "./components/PrefixPanel";

export default function HomePage() {
  const [selectedTab, setSelectedTab] = useState("lora");

  return (
    <div className="min-h-screen bg-gray-950 flex justify-center items-start p-8">
      <div className="w-full max-w-6xl">

        <Tabs.Root value={selectedTab} onValueChange={setSelectedTab}>

          {/* ---------------- Tabs Header ---------------- */}
          <Tabs.List className="flex border-b border-gray-700 mb-6">

            {/* LoRA Tab */}
            <Tabs.Trigger
              value="lora"
              className={`px-6 py-2 rounded-t-lg border border-b-0 mr-2 
                ${
                  selectedTab === "lora"
                    ? "bg-gray-900 border-gray-700 text-gray-200"
                    : "bg-gray-800 border-gray-800 text-gray-400 hover:bg-gray-700"
                }`}
            >
              LoRA
            </Tabs.Trigger>

            {/* Prefix Tab */}
            <Tabs.Trigger
              value="prefix"
              className={`px-6 py-2 rounded-t-lg border border-b-0 
                ${
                  selectedTab === "prefix"
                    ? "bg-gray-900 border-gray-700 text-gray-200"
                    : "bg-gray-800 border-gray-800 text-gray-400 hover:bg-gray-700"
                }`}
            >
              Prefix Tuning
            </Tabs.Trigger>
          </Tabs.List>

          {/* ---------------- Tab Content ---------------- */}
          <Tabs.Content value="lora">
            <LoRAPanel />
          </Tabs.Content>

          <Tabs.Content value="prefix">
            <PrefixPanel />
          </Tabs.Content>

        </Tabs.Root>
      </div>
    </div>
  );
}
