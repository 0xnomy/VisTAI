"use client";

import { motion } from "framer-motion";
import { Scan, Layers, FileSearch, Loader2, Check } from "lucide-react";
import { useAppStore } from "@/store/useAppStore";
import { runInference } from "@/lib/api";
import { useState, useEffect } from "react";

const STEPS = [
  { key: "cls", label: "Running Classification", icon: Scan, delay: 0 },
  { key: "seg", label: "Performing Segmentation", icon: Layers, delay: 1.5 },
  { key: "gen", label: "Generating Insights", icon: FileSearch, delay: 3 },
] as const;

export default function AnalysisLoader() {
  const { imageId, intent, setAnalysisResult, setStep } = useAppStore();
  const [activeStep, setActiveStep] = useState(0);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!imageId) return;

    // Animate steps
    const timers = STEPS.map((_, i) =>
      setTimeout(() => setActiveStep(i), STEPS[i].delay * 1000)
    );

    // Run inference
    runInference(imageId, intent)
      .then((result) => {
        setActiveStep(STEPS.length); // all done
        setTimeout(() => {
          setAnalysisResult(result);
          setStep("results");
        }, 800);
      })
      .catch((e) => {
        setError(e instanceof Error ? e.message : "Inference failed");
      });

    return () => timers.forEach(clearTimeout);
  }, [imageId, intent, setAnalysisResult, setStep]);

  return (
    <div className="w-full max-w-md mx-auto">
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        className="bg-slate-800 rounded-2xl border border-slate-700 shadow-sm p-8"
      >
        <h2 className="text-xl font-bold text-slate-100 text-center mb-8">
          Analyzing X-Ray
        </h2>

        <div className="space-y-6">
          {STEPS.map((step, i) => {
            const Icon = step.icon;
            const isDone = activeStep > i;
            const isActive = activeStep === i && !error;

            return (
              <motion.div
                key={step.key}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: step.delay * 0.3 }}
                className="flex items-center gap-4"
              >
                <div
                  className={`w-10 h-10 rounded-full flex items-center justify-center transition-all duration-500
                    ${isDone ? "bg-green-900/30" : isActive ? "bg-blue-900/30" : "bg-slate-700"}
                  `}
                >
                  {isDone ? (
                    <Check className="w-5 h-5 text-green-600" />
                  ) : isActive ? (
                    <Loader2 className="w-5 h-5 text-blue-500 animate-spin" />
                  ) : (
                    <Icon className="w-5 h-5 text-gray-400" />
                  )}
                </div>
                <span
                  className={`text-sm font-medium transition-colors
                    ${isDone ? "text-green-400" : isActive ? "text-blue-400" : "text-slate-500"}
                  `}
                >
                  {step.label}
                  {isDone && " ✓"}
                </span>
              </motion.div>
            );
          })}
        </div>

        {error && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="mt-6 p-3 bg-red-950/40 border border-red-900/50 rounded-xl"
          >
            <p className="text-sm text-red-400">{error}</p>
          </motion.div>
        )}
      </motion.div>
    </div>
  );
}
