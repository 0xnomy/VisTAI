/**
 * Global app store – Zustand
 */

import { create } from "zustand";
import type { InferenceResult, ChatMessage } from "@/lib/api";

export type AppStep = "landing" | "upload" | "intent" | "analyzing" | "results" | "chat" | "report";

interface AppState {
  /* navigation */
  step: AppStep;
  setStep: (s: AppStep) => void;

  /* upload */
  imageId: string | null;
  imageUrl: string | null;
  fileName: string | null;
  setUpload: (id: string, url: string, name: string) => void;

  /* intent */
  intent: "classification" | "segmentation" | "full";
  setIntent: (i: "classification" | "segmentation" | "full") => void;

  /* inference */
  analysisResult: InferenceResult | null;
  setAnalysisResult: (r: InferenceResult) => void;

  /* chat */
  chatHistory: ChatMessage[];
  addChat: (msg: ChatMessage) => void;
  clearChat: () => void;

  /* report */
  report: string | null;
  pdfUrl: string | null;
  setReport: (r: string, pdfUrl?: string | null) => void;

  /* reset */
  resetAll: () => void;
}

export const useAppStore = create<AppState>((set) => ({
  step: "landing",
  setStep: (s) => set({ step: s }),

  imageId: null,
  imageUrl: null,
  fileName: null,
  setUpload: (id, url, name) => set({ imageId: id, imageUrl: url, fileName: name }),

  intent: "full",
  setIntent: (i) => set({ intent: i }),

  analysisResult: null,
  setAnalysisResult: (r) => set({ analysisResult: r }),

  chatHistory: [],
  addChat: (msg) => set((s) => ({ chatHistory: [...s.chatHistory, msg] })),
  clearChat: () => set({ chatHistory: [] }),

  report: null,
  pdfUrl: null,
  setReport: (r, pdfUrl = null) => set({ report: r, pdfUrl }),

  resetAll: () =>
    set({
      step: "landing",
      imageId: null,
      imageUrl: null,
      fileName: null,
      intent: "full",
      analysisResult: null,
      chatHistory: [],
      report: null,
      pdfUrl: null,
    }),
}));
