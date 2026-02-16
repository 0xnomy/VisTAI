"use client";

import { motion } from "framer-motion";
import { fileUrl } from "@/lib/api";
import type { ClassificationResult } from "@/lib/api";

interface Props {
  classification: ClassificationResult;
  gradcamUrl: string | null;
}

const MALIGNANCY_COLORS: Record<string, string> = {
  malignant: "bg-red-100 text-red-700 border-red-200",
  benign: "bg-green-100 text-green-700 border-green-200",
  "benign (locally aggressive)": "bg-amber-100 text-amber-700 border-amber-200",
  unknown: "bg-gray-100 text-gray-600 border-gray-200",
};

export default function ClassificationPanel({ classification, gradcamUrl }: Props) {
  const malClass = MALIGNANCY_COLORS[classification.malignancy] || MALIGNANCY_COLORS.unknown;

  return (
    <motion.div
      initial={{ opacity: 0, x: 20 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ delay: 0.2 }}
      className="bg-[#111827] rounded-2xl border border-[#1f2937] overflow-hidden"
    >
      {/* Header */}
      <div className="px-6 py-5 border-b border-[#1f2937]">
        <h3 className="text-lg font-semibold tracking-tight text-[#f8fafc]">Tumor Classification</h3>
        <div className="flex items-center gap-2 mt-2">
          <span className={`text-xs font-semibold px-3 py-1 rounded-full border ${malClass}`}>
            {classification.malignancy}
          </span>
        </div>
      </div>

      {/* Primary prediction */}
      <div className="px-6 py-6 border-b border-[#1f2937]">
        <p className="text-xs font-medium uppercase tracking-wider text-[#64748b] mb-2">Primary Diagnosis</p>
        <p className="text-3xl font-bold tracking-tight text-[#f8fafc] capitalize leading-tight">
          {classification.top_class}
        </p>
        <div className="mt-4 flex items-center gap-3">
          <div className="flex-1 h-3 bg-[#1f2937] rounded-full overflow-hidden">
            <motion.div
              initial={{ width: 0 }}
              animate={{ width: `${classification.confidence * 100}%` }}
              transition={{ duration: 0.8, ease: [0.4, 0, 0.2, 1] }}
              className="h-full bg-gradient-to-r from-[#3b82f6] to-[#2563eb] rounded-full shadow-sm"
            />
          </div>
          <span className="text-lg font-bold text-[#3b82f6] tabular-nums">
            {(classification.confidence * 100).toFixed(1)}%
          </span>
        </div>
      </div>

      {/* Top 5 */}
      <div className="px-6 py-6">
        <p className="text-xs font-medium uppercase tracking-wider text-[#64748b] mb-4">Differential Diagnosis</p>
        <div className="space-y-3">
          {classification.top5.map((item, i) => (
            <div key={item.class} className="flex items-center gap-4">
              <span className="text-xs font-medium text-[#64748b] w-5">{i + 1}</span>
              <div className="flex-1 min-w-0">
                <div className="flex items-center justify-between mb-1.5">
                  <span className="text-sm font-medium text-[#cbd5e1] capitalize truncate">{item.class}</span>
                  <span className="text-sm font-semibold text-[#94a3b8] tabular-nums ml-3">
                    {(item.probability * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="h-2 bg-[#1f2937] rounded-full overflow-hidden">
                  <motion.div
                    initial={{ width: 0 }}
                    animate={{ width: `${item.probability * 100}%` }}
                    transition={{ duration: 0.6, delay: i * 0.08, ease: [0.4, 0, 0.2, 1] }}
                    className={`h-full rounded-full ${i === 0 ? "bg-gradient-to-r from-[#3b82f6] to-[#2563eb]" : "bg-[#374151]"}`}
                  />
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Grad-CAM */}
      {gradcamUrl && (
        <div className="px-5 py-4 border-t border-slate-700">
          <p className="text-sm font-medium text-slate-400 mb-2">Classification Attention (Grad-CAM)</p>
          <div className="rounded-xl overflow-hidden border border-slate-700">
            <img
              src={fileUrl(gradcamUrl)}
              alt="Classification Grad-CAM"
              className="w-full object-contain"
            />
          </div>
        </div>
      )}
    </motion.div>
  );
}
