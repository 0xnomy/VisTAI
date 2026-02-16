"use client";

import { motion } from "framer-motion";
import { useState } from "react";
import { fileUrl } from "@/lib/api";
import type { SegmentationResult } from "@/lib/api";

interface Props {
  originalUrl: string;
  segmentation: SegmentationResult;
}

export default function SegmentationViewer({ originalUrl, segmentation }: Props) {
  const [opacity, setOpacity] = useState(0.5);
  const [view, setView] = useState<"overlay" | "mask" | "gradcam">("overlay");

  const views = {
    overlay: { url: segmentation.overlay_url, label: "Mask Overlay" },
    mask: { url: segmentation.mask_url, label: "Binary Mask" },
    gradcam: { url: segmentation.gradcam_url, label: "Grad-CAM" },
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.1 }}
      className="bg-[#111827] rounded-2xl border border-[#1f2937] overflow-hidden"
    >
      {/* Header */}
      <div className="px-6 py-5 border-b border-[#1f2937]">
        <h3 className="text-lg font-semibold tracking-tight text-[#f8fafc]">Tumor Segmentation</h3>
        <div className="flex items-center gap-3 mt-2">
          <p className="text-sm text-[#94a3b8]">
            Coverage: <span className="font-semibold text-[#3b82f6]">{segmentation.tumor_coverage.toFixed(1)}%</span> of image area
          </p>
        </div>
      </div>

      {/* Image */}
      <div className="relative aspect-square bg-black flex items-center justify-center p-6">
        <img
          src={fileUrl(originalUrl)}
          alt="Original X-ray"
          className="absolute inset-0 w-full h-full object-contain p-6"
        />
        <motion.img
          key={view}
          initial={{ opacity: 0 }}
          animate={{ opacity: view === "mask" ? 1 : opacity }}
          transition={{ duration: 0.3 }}
          src={fileUrl(views[view].url)}
          alt={views[view].label}
          className="absolute inset-0 w-full h-full object-contain p-6"
        />
      </div>

      {/* Controls */}
      <div className="px-6 py-5 space-y-4 border-t border-[#1f2937]">
        {/* View tabs */}
        <div>
          <p className="text-xs font-medium uppercase tracking-wider text-[#64748b] mb-3">Visualization Mode</p>
          <div className="flex gap-2 bg-[#0a0e1a] rounded-xl p-1.5 border border-[#1f2937]">
            {(Object.keys(views) as (keyof typeof views)[]).map((key) => (
              <motion.button
                key={key}
                whileHover={{ scale: view === key ? 1 : 1.02 }}
                whileTap={{ scale: 0.98 }}
                onClick={() => setView(key)}
                className={`flex-1 py-2.5 text-sm font-semibold rounded-lg transition-all
                  ${view === key
                    ? "bg-gradient-to-r from-[#3b82f6] to-[#2563eb] text-white shadow-lg shadow-blue-500/20"
                    : "text-[#94a3b8] hover:text-[#cbd5e1] hover:bg-[#1f2937]/50"
                  }`}
              >
                {views[key].label}
              </motion.button>
            ))}
          </div>
        </div>

        {/* Opacity slider */}
        {view !== "mask" && (
          <div>
            <p className="text-xs font-medium uppercase tracking-wider text-[#64748b] mb-3">Overlay Opacity</p>
            <div className="flex items-center gap-4">
              <span className="text-xs font-medium text-[#64748b] w-8">0%</span>
              <input
                type="range"
                min="0"
                max="1"
                step="0.05"
                value={opacity}
                onChange={(e) => setOpacity(parseFloat(e.target.value))}
                className="flex-1 h-2 bg-[#1f2937] rounded-full appearance-none cursor-pointer
                  [&::-webkit-slider-thumb]:appearance-none
                  [&::-webkit-slider-thumb]:w-4
                  [&::-webkit-slider-thumb]:h-4
                  [&::-webkit-slider-thumb]:rounded-full
                  [&::-webkit-slider-thumb]:bg-gradient-to-r
                  [&::-webkit-slider-thumb]:from-[#3b82f6]
                  [&::-webkit-slider-thumb]:to-[#2563eb]
                  [&::-webkit-slider-thumb]:shadow-lg
                  [&::-webkit-slider-thumb]:shadow-blue-500/20
                  [&::-webkit-slider-thumb]:cursor-pointer
                  [&::-webkit-slider-thumb]:transition-all
                  hover:[&::-webkit-slider-thumb]:scale-110"
              />
              <span className="text-sm font-semibold text-[#3b82f6] tabular-nums w-12 text-right">
                {Math.round(opacity * 100)}%
              </span>
            </div>
          </div>
        )}
      </div>
    </motion.div>
  );
}
