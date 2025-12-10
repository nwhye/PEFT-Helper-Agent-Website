"use client";

import React, { useState } from "react";
import * as Tabs from "@radix-ui/react-tabs";

import LoRAPanel from "./components/LoRAPanel";
import PrefixPanel from "./components/PrefixPanel";


export default function HomePage() {
  const [selectedTab, setSelectedTab] = useState("lora");

  return (
    <Tabs.Root value={selectedTab} onValueChange={setSelectedTab}>
      <Tabs.List className="flex border-b border-gray-700 mb-6">
        <Tabs.Trigger value="lora" className="...">LoRA</Tabs.Trigger>
        <Tabs.Trigger value="prefix" className="...">Prefix Tuning</Tabs.Trigger>
      </Tabs.List>

      <Tabs.Content value="lora">
        <LoRAPanel />
      </Tabs.Content>

      <Tabs.Content value="prefix">
        <PrefixPanel />
      </Tabs.Content>
    </Tabs.Root>
  );
}
