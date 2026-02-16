"use client";

import { motion } from "framer-motion";
import { Scan, Layers, FileText } from "lucide-react";
import { useAppStore } from "@/store/useAppStore";

const OPTIONS = [
  {
    key: "full" as const,
    label: "Full Diagnostic Analysis",
    description: "Classification + Segmentation + Grad-CAM",
    icon: FileText,
    recommended: true,
  },
  {
    key: "classification" as const,
    label: "Classification Only",
    description: "Identify tumor type and confidence",
    icon: Scan,
    recommended: false,
  },
  {
    key: "segmentation" as const,
    label: "Segmentation Only",
    description: "Locate tumor region in the X-ray",
    icon: Layers,
    recommended: false,
  },
];

export default function IntentSelector() {
  const { setIntent, setStep, imageUrl } = useAppStore();

  const handleSelect = (intent: "classification" | "segmentation" | "full") => {
    setIntent(intent);
    setStep("analyzing");
  };

  return (
    <div className="w-full max-w-2xl mx-auto">
      {/* Preview thumbnail */}
      {imageUrl && (
        <div className="flex justify-center mb-8">
          <div className="w-32 h-32 rounded-2xl overflow-hidden border-2 border-slate-700 shadow-sm bg-slate-900">
            <img src={imageUrl} alt="Uploaded" className="w-full h-full object-contain" />
          </div>
        </div>
      )}

      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        className="text-center mb-6"
      >
        <h2 className="text-2xl font-bold text-slate-100">
          What would you like to do with this X-ray?
        </h2>
        <p className="text-slate-400 mt-1">Choose an analysis type</p>
      </motion.div>

      <div className="grid gap-3">
        {OPTIONS.map((opt, i) => {
          const Icon = opt.icon;
          return (
            <motion.button
              key={opt.key}
              initial={{ opacity: 0, y: 15 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: i * 0.1 }}
              onClick={() => handleSelect(opt.key)}
              className="relative w-full flex items-center gap-4 p-5 rounded-2xl border border-slate-700 bg-slate-800 hover:border-blue-500 hover:shadow-lg hover:shadow-blue-500/10 transition-all duration-200 text-left group"
            >
              <div className="w-12 h-12 rounded-xl bg-blue-900/30 flex items-center justify-center group-hover:bg-blue-900/50 transition-colors">
                <Icon className="w-6 h-6 text-blue-400" />
              </div>
              <div className="flex-1">
                <p className="font-semibold text-slate-100">{opt.label}</p>
                <p className="text-sm text-slate-400">{opt.description}</p>
              </div>
              {opt.recommended && (
                <span className="text-[10px] font-bold px-2 py-0.5 rounded-full bg-blue-500 text-white uppercase tracking-wider">
                  Recommended
                </span>
              )}
            </motion.button>
          );
        })}
      </div>
    </div>
  );
}
