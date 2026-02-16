"use client";

import React from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Activity,
  Shield,
  ArrowLeft,
  RotateCcw,
  MessageSquare,
  FileText as FileTextIcon,
  Sparkles,
  Microscope,
  Brain,
} from "lucide-react";
import { useAppStore } from "@/store/useAppStore";
import { fileUrl } from "@/lib/api";
import ImageUploader from "@/components/ImageUploader";
import IntentSelector from "@/components/IntentSelector";
import AnalysisLoader from "@/components/AnalysisLoader";
import SegmentationViewer from "@/components/SegmentationViewer";
import ClassificationPanel from "@/components/ClassificationPanel";
import ChatPanel from "@/components/ChatPanel";
import ReportViewer from "@/components/ReportViewer";

/* ── Page wrapper for animated transitions ───────────────────────────── */
function StepWrapper({ children }: { children: React.ReactNode }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      transition={{ duration: 0.4, ease: [0.4, 0, 0.2, 1] }}
      className="w-full"
    >
      {children}
    </motion.div>
  );
}

function BackButton({ onClick, label = "Back" }: { onClick: () => void; label?: string }) {
  return (
    <motion.button
      whileHover={{ x: -2 }}
      onClick={onClick}
      className="group flex items-center gap-2 text-sm font-medium text-slate-400 hover:text-slate-200 transition-colors"
    >
      <ArrowLeft className="w-4 h-4 group-hover:-translate-x-0.5 transition-transform" />
      {label}
    </motion.button>
  );
}

/* ═══════════════════════════════════════════════════════════════════════ */

export default function Home() {
  const store = useAppStore();
  const { step, setStep, resetAll, analysisResult } = store;

  /* ── Landing ──────────────────────────────────────────────────────── */
  const LandingView = () => (
    <StepWrapper>
      <div className="flex flex-col items-center text-center max-w-4xl mx-auto px-6">
        {/* Hero Icon */}
        <motion.div
          initial={{ scale: 0.9, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          transition={{ duration: 0.6, ease: [0.4, 0, 0.2, 1] }}
          className="relative mb-8"
        >
          <div className="absolute inset-0 bg-blue-500/20 blur-3xl rounded-full" />
          <div className="relative w-16 h-16 rounded-2xl bg-gradient-to-br from-blue-500 to-blue-600 flex items-center justify-center shadow-lg shadow-blue-500/30">
            <Activity className="w-8 h-8 text-white" strokeWidth={2.5} />
          </div>
        </motion.div>

        {/* Headline */}
        <motion.div
          initial={{ y: 20, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ delay: 0.1, duration: 0.6 }}
        >
          <h1 className="text-5xl sm:text-6xl font-bold text-white tracking-tight leading-tight">
            Bone Tumor
            <br />
            <span className="bg-gradient-to-r from-blue-400 to-cyan-400 bg-clip-text text-transparent">
              X-Ray Detection
            </span>
          </h1>
        </motion.div>

        {/* Subtitle */}
        <motion.p
          initial={{ y: 20, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ delay: 0.2, duration: 0.6 }}
          className="mt-6 text-lg text-slate-400 max-w-2xl leading-relaxed"
        >
          Clinical-grade AI for automated classification and segmentation of bone tumors
          from radiographic imaging using knowledge-distilled deep learning.
        </motion.p>

        {/* CTA Button */}
        <motion.button
          initial={{ y: 20, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ delay: 0.3, duration: 0.6 }}
          whileHover={{ scale: 1.02, boxShadow: "0 0 30px rgba(59, 130, 246, 0.4)" }}
          whileTap={{ scale: 0.98 }}
          onClick={() => setStep("upload")}
          className="group mt-10 px-8 py-4 rounded-xl bg-blue-600 text-white font-semibold text-base hover:bg-blue-500 shadow-xl shadow-blue-600/30 transition-all flex items-center gap-2"
        >
          <span>Start Analysis</span>
          <Sparkles className="w-4 h-4 group-hover:rotate-12 transition-transform" />
        </motion.button>

        {/* Features Grid */}
        <motion.div
          initial={{ y: 40, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ delay: 0.5, duration: 0.6 }}
          className="grid sm:grid-cols-3 gap-6 mt-20 w-full"
        >
          {[
            { icon: Brain, title: "9 Tumor Classes", desc: "Multi-class classification with confidence scoring" },
            { icon: Microscope, title: "Precise Segmentation", desc: "Pixel-level tumor boundary detection" },
            { icon: MessageSquare, title: "AI Assistant", desc: "Interactive medical Q&A and report generation" },
          ].map((feature, i) => {
            const Icon = feature.icon;
            return (
              <motion.div
                key={feature.title}
                initial={{ y: 20, opacity: 0 }}
                animate={{ y: 0, opacity: 1 }}
                transition={{ delay: 0.6 + i * 0.1 }}
                className="group relative p-6 rounded-2xl bg-[#111827] border border-[#1f2937] hover:border-blue-500/50 transition-all"
              >
                <div className="absolute inset-0 bg-gradient-to-br from-blue-500/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity rounded-2xl" />
                <Icon className="w-6 h-6 text-blue-400 mb-4" strokeWidth={2} />
                <h3 className="font-semibold text-white text-base mb-2">{feature.title}</h3>
                <p className="text-sm text-slate-400 leading-relaxed">{feature.desc}</p>
              </motion.div>
            );
          })}
        </motion.div>

        {/* Disclaimer */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.9, duration: 0.6 }}
          className="mt-16 flex items-start gap-3 p-4 rounded-xl bg-amber-950/30 border border-amber-900/40 text-left max-w-2xl"
        >
          <Shield className="w-5 h-5 text-amber-400 flex-shrink-0 mt-0.5" />
          <p className="text-xs text-amber-200/80 leading-relaxed">
            <strong className="text-amber-300">Medical Disclaimer:</strong> This system is designed for research
            and educational purposes. AI predictions do not constitute medical diagnosis.
            Always consult qualified healthcare professionals for clinical decisions.
          </p>
        </motion.div>
      </div>
    </StepWrapper>
  );

  /* ── Upload ───────────────────────────────────────────────────────── */
  const UploadView = () => (
    <StepWrapper>
      <div className="max-w-2xl mx-auto">
        <div className="mb-8">
          <BackButton onClick={() => setStep("landing")} />
        </div>
        <div className="text-center mb-8">
          <h2 className="text-3xl font-bold text-white mb-2">Upload X-Ray Image</h2>
          <p className="text-slate-400">
            Upload a bone X-ray image for AI-powered tumor analysis
          </p>
        </div>
        <ImageUploader />
      </div>
    </StepWrapper>
  );

  /* ── Intent ───────────────────────────────────────────────────────── */
  const IntentView = () => (
    <StepWrapper>
      <div className="max-w-2xl mx-auto">
        <div className="mb-8">
          <BackButton onClick={() => setStep("upload")} />
        </div>
        <IntentSelector />
      </div>
    </StepWrapper>
  );

  /* ── Analyzing ────────────────────────────────────────────────────── */
  const AnalyzingView = () => (
    <StepWrapper>
      <AnalysisLoader />
    </StepWrapper>
  );

  /* ── Results ──────────────────────────────────────────────────────── */
  const ResultsView = () => {
    if (!analysisResult) return null;
    const [activeTab, setActiveTab] = React.useState<'classification' | 'segmentation'>(
      analysisResult.classification ? 'classification' : 'segmentation'
    );
    const hasClassification = !!analysisResult.classification;
    const hasSegmentation = !!analysisResult.segmentation;
    const [runningMissing, setRunningMissing] = React.useState(false);

    const runMissingAnalysis = async () => {
      setRunningMissing(true);
      try {
        const { runInference } = await import('@/lib/api');
        const newIntent = !hasClassification ? 'classification' : 'segmentation';
        const result = await runInference(store.imageId!, newIntent);
        
        // Merge the new result with existing result
        const merged = {
          ...analysisResult,
          ...result,
          classification: result.classification || analysisResult.classification,
          segmentation: result.segmentation || analysisResult.segmentation,
          cls_gradcam_url: result.cls_gradcam_url || analysisResult.cls_gradcam_url,
        };
        store.setAnalysisResult(merged);
        
        // Switch to the newly added tab
        setActiveTab(newIntent === 'classification' ? 'classification' : 'segmentation');
      } catch (e) {
        console.error('Failed to run missing analysis:', e);
      } finally {
        setRunningMissing(false);
      }
    };

    return (
      <StepWrapper>
        <div className="max-w-7xl mx-auto">
          {/* Top Action Bar */}
          <div className="flex items-center justify-between mb-8">
            <div>
              <h2 className="text-3xl font-bold tracking-tight text-[#f8fafc] mb-1">Analysis Results</h2>
              <p className="text-sm text-[#94a3b8]">AI-powered bone tumor analysis</p>
            </div>
            <div className="flex items-center gap-3">
              <motion.button
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                onClick={() => setStep("chat")}
                className="flex items-center gap-2 px-5 py-2.5 rounded-xl border border-[#1f2937] bg-[#111827] bg-opacity-60 backdrop-blur-sm text-sm font-medium text-[#e2e8f0] hover:bg-[#1f2937] hover:bg-opacity-80 hover:border-[#374151] transition-all"
              >
                <MessageSquare className="w-4 h-4" />
                Chat
              </motion.button>
              <motion.button
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                onClick={() => setStep("report")}
                className="flex items-center gap-2 px-5 py-2.5 rounded-xl border border-[#1f2937] bg-[#111827] bg-opacity-60 backdrop-blur-sm text-sm font-medium text-[#e2e8f0] hover:bg-[#1f2937] hover:bg-opacity-80 hover:border-[#374151] transition-all"
              >
                <FileTextIcon className="w-4 h-4" />
                Report
              </motion.button>
              <motion.button
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                onClick={resetAll}
                className="flex items-center gap-2 px-5 py-2.5 rounded-xl bg-gradient-to-r from-[#3b82f6] to-[#2563eb] text-white text-sm font-semibold shadow-lg shadow-blue-500 shadow-opacity-20 hover:shadow-blue-500 hover:shadow-opacity-30 transition-all"
              >
                <RotateCcw className="w-4 h-4" />
                New Analysis
              </motion.button>
            </div>
          </div>

          {/* Tab Navigation */}
          <div className="flex items-center gap-3 mb-8 p-1.5 bg-[#111827] bg-opacity-60 backdrop-blur-sm rounded-2xl border border-[#1f2937] w-fit">
            <motion.button
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              onClick={() => setActiveTab('classification')}
              className={
                activeTab === 'classification'
                  ? 'px-6 py-2.5 rounded-xl text-sm font-semibold transition-all bg-gradient-to-r from-[#3b82f6] to-[#2563eb] text-white shadow-lg shadow-blue-500 shadow-opacity-20'
                  : hasClassification
                  ? 'px-6 py-2.5 rounded-xl text-sm font-semibold transition-all text-[#cbd5e1] hover:text-[#f8fafc] hover:bg-[#1f2937] hover:bg-opacity-50'
                  : 'px-6 py-2.5 rounded-xl text-sm font-semibold transition-all text-[#94a3b8] hover:text-[#cbd5e1] hover:bg-[#1f2937] hover:bg-opacity-30'
              }
            >
              Classification
              {!hasClassification && (
                <span className="ml-1.5 text-xs opacity-60">●</span>
              )}
            </motion.button>
            <motion.button
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              onClick={() => setActiveTab('segmentation')}
              className={
                activeTab === 'segmentation'
                  ? 'px-6 py-2.5 rounded-xl text-sm font-semibold transition-all bg-gradient-to-r from-[#3b82f6] to-[#2563eb] text-white shadow-lg shadow-blue-500 shadow-opacity-20'
                  : hasSegmentation
                  ? 'px-6 py-2.5 rounded-xl text-sm font-semibold transition-all text-[#cbd5e1] hover:text-[#f8fafc] hover:bg-[#1f2937] hover:bg-opacity-50'
                  : 'px-6 py-2.5 rounded-xl text-sm font-semibold transition-all text-[#94a3b8] hover:text-[#cbd5e1] hover:bg-[#1f2937] hover:bg-opacity-30'
              }
            >
              Segmentation
              {!hasSegmentation && (
                <span className="ml-1.5 text-xs opacity-60">●</span>
              )}
            </motion.button>
          </div>

          <div className="grid lg:grid-cols-2 gap-6">
            {activeTab === 'classification' && hasClassification && (
              <>
                {/* Grad-CAM Image */}
                {analysisResult.cls_gradcam_url && (
                  <motion.div
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: 0.1 }}
                    className="bg-[#111827] rounded-2xl border border-[#1f2937] overflow-hidden"
                  >
                    <div className="px-6 py-4 border-b border-[#1f2937]">
                      <h3 className="text-lg font-semibold tracking-tight text-[#f8fafc]">Attention Map</h3>
                      <p className="text-sm text-[#94a3b8] mt-0.5">Grad-CAM visualization showing model focus</p>
                    </div>
                    <div className="aspect-square bg-black flex items-center justify-center p-4">
                      <img
                        src={fileUrl(analysisResult.cls_gradcam_url)}
                        alt="Grad-CAM Heatmap"
                        className="w-full h-full object-contain"
                      />
                    </div>
                  </motion.div>
                )}
                <ClassificationPanel
                  classification={analysisResult.classification!}
                  gradcamUrl={analysisResult.cls_gradcam_url}
                />
              </>
            )}
            {activeTab === 'classification' && !hasClassification && (
              <div className="lg:col-span-2">
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="p-12 rounded-2xl bg-[#111827] bg-opacity-60 backdrop-blur-sm border border-[#1f2937] text-center"
                >
                  <div className="w-16 h-16 rounded-full bg-[#1f2937] flex items-center justify-center mx-auto mb-4">
                    <Brain className="w-8 h-8 text-[#64748b]" />
                  </div>
                  <p className="text-lg font-medium text-[#cbd5e1] mb-2">Classification not performed</p>
                  <p className="text-sm text-[#94a3b8] mb-6">Run classification analysis to identify tumor type</p>
                  <motion.button
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                    onClick={runMissingAnalysis}
                    disabled={runningMissing}
                    className="px-8 py-3 rounded-xl bg-gradient-to-r from-[#3b82f6] to-[#2563eb] text-white font-semibold shadow-lg shadow-blue-500 shadow-opacity-20 hover:shadow-blue-500 hover:shadow-opacity-30 transition-all disabled:opacity-50 disabled:cursor-not-allowed disabled:scale-100"
                  >
                    {runningMissing ? 'Running...' : 'Run Classification'}
                  </motion.button>
                </motion.div>
              </div>
            )}
            {activeTab === 'segmentation' && hasSegmentation && (
              <div className="lg:col-span-2">
                <SegmentationViewer
                  originalUrl={analysisResult.original_url}
                  segmentation={analysisResult.segmentation!}
                />
              </div>
            )}
            {activeTab === 'segmentation' && !hasSegmentation && (
              <div className="lg:col-span-2">
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="p-12 rounded-2xl bg-[#111827] bg-opacity-60 backdrop-blur-sm border border-[#1f2937] text-center"
                >
                  <div className="w-16 h-16 rounded-full bg-[#1f2937] flex items-center justify-center mx-auto mb-4">
                    <Microscope className="w-8 h-8 text-[#64748b]" />
                  </div>
                  <p className="text-lg font-medium text-[#cbd5e1] mb-2">Segmentation not performed</p>
                  <p className="text-sm text-[#94a3b8] mb-6">Run segmentation analysis to detect tumor boundaries</p>
                  <motion.button
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                    onClick={runMissingAnalysis}
                    disabled={runningMissing}
                    className="px-8 py-3 rounded-xl bg-gradient-to-r from-[#3b82f6] to-[#2563eb] text-white font-semibold shadow-lg shadow-blue-500 shadow-opacity-20 hover:shadow-blue-500 hover:shadow-opacity-30 transition-all disabled:opacity-50 disabled:cursor-not-allowed disabled:scale-100"
                  >
                    {runningMissing ? 'Running...' : 'Run Segmentation'}
                  </motion.button>
                </motion.div>
              </div>
            )}
          </div>
        </div>
      </StepWrapper>
    );
  };

  /* ── Chat ──────────────────────────────────────────────────────────── */
  const ChatView = () => (
    <StepWrapper>
      <div className="max-w-4xl mx-auto">
        <div className="flex items-center justify-between mb-8">
          <div>
            <h2 className="text-3xl font-bold tracking-tight text-[#f8fafc] mb-1">AI Assistant</h2>
            <p className="text-sm text-[#94a3b8]">Ask questions about your analysis</p>
          </div>
          <div className="flex items-center gap-3">
            <BackButton onClick={() => setStep("results")} label="Back to Results" />
            <motion.button
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              onClick={() => setStep("report")}
              className="flex items-center gap-2 px-5 py-2.5 rounded-xl border border-[#1f2937] bg-[#111827] bg-opacity-60 backdrop-blur-sm text-sm font-medium text-[#e2e8f0] hover:bg-[#1f2937] hover:bg-opacity-80 hover:border-[#374151] transition-all"
            >
              <FileTextIcon className="w-4 h-4" />
              View Report
            </motion.button>
          </div>
        </div>
        <ChatPanel />
      </div>
    </StepWrapper>
  );

  /* ── Report ────────────────────────────────────────────────────────── */
  const ReportView = () => (
    <StepWrapper>
      <div className="max-w-4xl mx-auto">
        <div className="flex items-center justify-between mb-8">
          <div>
            <h2 className="text-3xl font-bold tracking-tight text-[#f8fafc] mb-1">Clinical Report</h2>
            <p className="text-sm text-[#94a3b8]">Detailed analysis and recommendations</p>
          </div>
          <div className="flex items-center gap-3">
            <BackButton onClick={() => setStep("results")} label="Back to Results" />
            <motion.button
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              onClick={() => setStep("chat")}
              className="flex items-center gap-2 px-5 py-2.5 rounded-xl border border-[#1f2937] bg-[#111827] bg-opacity-60 backdrop-blur-sm text-sm font-medium text-[#e2e8f0] hover:bg-[#1f2937] hover:bg-opacity-80 hover:border-[#374151] transition-all"
            >
              <MessageSquare className="w-4 h-4" />
              Chat
            </motion.button>
          </div>
        </div>
        <ReportViewer />
      </div>
    </StepWrapper>
  );

  /* ── Render ────────────────────────────────────────────────────────── */
  const stepConfig = {
    landing: { label: "Home", index: 0 },
    upload: { label: "Upload", index: 1 },
    intent: { label: "Configure", index: 2 },
    analyzing: { label: "Processing", index: 3 },
    results: { label: "Results", index: 4 },
    chat: { label: "Chat", index: 5 },
    report: { label: "Report", index: 6 },
  };

  return (
    <div className="min-h-screen bg-[#0a0e1a]">
      {/* Premium Navbar */}
      <nav className="sticky top-0 z-50 border-b border-[#1f2937] bg-[#0a0e1a]/80 backdrop-blur-xl">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          {/* Logo */}
          <button
            onClick={resetAll}
            className="group flex items-center gap-3 hover:opacity-80 transition-opacity"
          >
            <div className="relative">
              <div className="absolute inset-0 bg-blue-500/20 blur-md rounded-lg" />
              <div className="relative w-8 h-8 rounded-lg bg-gradient-to-br from-blue-500 to-blue-600 flex items-center justify-center shadow-lg">
                <Activity className="w-4 h-4 text-white" strokeWidth={2.5} />
              </div>
            </div>
            <div>
              <span className="text-base font-semibold text-white tracking-tight">BTXRD</span>
              <span className="hidden md:inline text-xs text-slate-500 ml-2">Medical AI</span>
            </div>
          </button>

          {/* Breadcrumb Progress */}
          {step !== "landing" && (
            <div className="hidden md:flex items-center gap-2">
              {Object.entries(stepConfig)
                .filter(([key]) => key !== "landing")
                .slice(0, stepConfig[step].index)
                .map(([key, config], i, arr) => (
                  <React.Fragment key={key}>
                    <span
                      className={`text-xs font-medium transition-colors ${
                        stepConfig[step].index > config.index
                          ? "text-slate-400"
                          : stepConfig[step].index === config.index
                          ? "text-blue-400"
                          : "text-slate-600"
                      }`}
                    >
                      {config.label}
                    </span>
                    {i < arr.length - 1 && (
                      <span className="text-slate-700">›</span>
                    )}
                  </React.Fragment>
                ))}
              <span className="text-xs font-semibold text-blue-400">
                {stepConfig[step].label}
              </span>
            </div>
          )}

          {/* Status Badge */}
          <div className="flex items-center gap-3">
            <div className="hidden sm:flex items-center gap-2 px-3 py-1.5 rounded-lg bg-[#111827] border border-[#1f2937]">
              <div className="w-1.5 h-1.5 rounded-full bg-green-500 animate-pulse" />
              <span className="text-xs font-medium text-slate-400">AI Ready</span>
            </div>
          </div>
        </div>
      </nav>

      {/* Main Content with Max Width */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 md:py-16">
        <AnimatePresence mode="wait">
          {step === "landing" && <LandingView key="landing" />}
          {step === "upload" && <UploadView key="upload" />}
          {step === "intent" && <IntentView key="intent" />}
          {step === "analyzing" && <AnalyzingView key="analyzing" />}
          {step === "results" && <ResultsView key="results" />}
          {step === "chat" && <ChatView key="chat" />}
          {step === "report" && <ReportView key="report" />}
        </AnimatePresence>
      </main>
    </div>
  );
}
