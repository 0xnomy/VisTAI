"use client";

import { useState, useMemo } from "react";
import { motion } from "framer-motion";
import { FileText, Download, Loader2, RefreshCw, CheckCircle, AlertTriangle } from "lucide-react";
import { generateReport, fileUrl } from "@/lib/api";
import { useAppStore } from "@/store/useAppStore";

/* ── Minimal Markdown → JSX renderer ─────────────────────────────────── */

function renderMarkdown(md: string) {
  const lines = md.split("\n");
  const elements: React.ReactNode[] = [];
  let key = 0;
  let inTable = false;
  let tableRows: string[][] = [];

  const flushTable = () => {
    if (tableRows.length === 0) return;
    const header = tableRows[0];
    const body = tableRows.slice(1);
    elements.push(
      <div key={key++} className="overflow-x-auto my-4">
        <table className="w-full text-sm border-collapse">
          <thead>
            <tr className="bg-[#1e3a5f]">
              {header.map((cell, i) => (
                <th key={i} className="px-4 py-2.5 text-left text-xs font-semibold text-white border border-[#334155]">
                  {renderInline(cell)}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {body.map((row, ri) => (
              <tr key={ri} className={ri % 2 === 0 ? "bg-[#0f172a]" : "bg-[#1e293b]/50"}>
                {row.map((cell, ci) => (
                  <td key={ci} className="px-4 py-2 text-[#e2e8f0] border border-[#334155]">
                    {renderInline(cell)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    );
    tableRows = [];
  };

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const trimmed = line.trim();

    // Table row
    if (trimmed.startsWith("|") && trimmed.endsWith("|")) {
      // Skip separator rows
      if (/^\|[\s\-:|]+\|$/.test(trimmed)) continue;
      const cells = trimmed.split("|").slice(1, -1).map(c => c.trim());
      tableRows.push(cells);
      inTable = true;
      continue;
    } else if (inTable) {
      flushTable();
      inTable = false;
    }

    // Empty line
    if (!trimmed) {
      elements.push(<div key={key++} className="h-2" />);
      continue;
    }

    // H2
    if (trimmed.startsWith("## ")) {
      const text = trimmed.slice(3);
      elements.push(
        <div key={key++} className="mt-6 mb-3">
          <div className="flex items-center gap-2">
            <div className="w-1 h-6 rounded-full bg-blue-500" />
            <h2 className="text-base font-bold text-[#f8fafc] uppercase tracking-wide">
              {renderInline(text)}
            </h2>
          </div>
          <div className="h-px bg-[#334155] mt-2" />
        </div>
      );
      continue;
    }

    // H1
    if (trimmed.startsWith("# ")) {
      continue; // Skip — the component has its own title
    }

    // Numbered list
    const numMatch = trimmed.match(/^(\d+)\.\s+(.*)/);
    if (numMatch) {
      elements.push(
        <div key={key++} className="flex gap-3 pl-2 py-1">
          <span className="text-blue-400 font-semibold text-sm min-w-[20px]">{numMatch[1]}.</span>
          <p className="text-sm text-[#cbd5e1] leading-relaxed">{renderInline(numMatch[2])}</p>
        </div>
      );
      continue;
    }

    // Bullet list
    if (trimmed.startsWith("- ") || trimmed.startsWith("* ")) {
      const text = trimmed.slice(2);
      elements.push(
        <div key={key++} className="flex gap-3 pl-2 py-0.5">
          <span className="text-blue-400 mt-1.5">•</span>
          <p className="text-sm text-[#cbd5e1] leading-relaxed">{renderInline(text)}</p>
        </div>
      );
      continue;
    }

    // Disclaimer line
    if (trimmed.includes("⚠️") || trimmed.toUpperCase().includes("IMPORTANT")) {
      elements.push(
        <div key={key++} className="my-4 p-4 rounded-xl bg-amber-950/40 border border-amber-800/50">
          <p className="text-sm text-amber-200 leading-relaxed font-medium">
            {renderInline(trimmed)}
          </p>
        </div>
      );
      continue;
    }

    // Horizontal rule
    if (trimmed === "---") {
      elements.push(<div key={key++} className="h-px bg-[#334155] my-4" />);
      continue;
    }

    // Regular paragraph
    elements.push(
      <p key={key++} className="text-sm text-[#cbd5e1] leading-relaxed py-0.5">
        {renderInline(trimmed)}
      </p>
    );
  }

  // Flush any remaining table
  if (inTable) flushTable();

  return elements;
}

function renderInline(text: string): React.ReactNode {
  // Convert **bold** and *italic*
  const parts: React.ReactNode[] = [];
  let remaining = text;
  let k = 0;

  while (remaining.length > 0) {
    // Bold
    const boldMatch = remaining.match(/\*\*(.+?)\*\*/);
    if (boldMatch && boldMatch.index !== undefined) {
      if (boldMatch.index > 0) {
        parts.push(<span key={k++}>{remaining.slice(0, boldMatch.index)}</span>);
      }
      parts.push(<strong key={k++} className="text-[#f8fafc] font-semibold">{boldMatch[1]}</strong>);
      remaining = remaining.slice(boldMatch.index + boldMatch[0].length);
      continue;
    }

    // Italic
    const italicMatch = remaining.match(/\*(.+?)\*/);
    if (italicMatch && italicMatch.index !== undefined) {
      if (italicMatch.index > 0) {
        parts.push(<span key={k++}>{remaining.slice(0, italicMatch.index)}</span>);
      }
      parts.push(<em key={k++} className="text-[#94a3b8]">{italicMatch[1]}</em>);
      remaining = remaining.slice(italicMatch.index + italicMatch[0].length);
      continue;
    }

    // No more matches
    parts.push(<span key={k++}>{remaining}</span>);
    break;
  }

  return parts.length === 1 ? parts[0] : <>{parts}</>;
}

/* ── ReportViewer Component ───────────────────────────────────────────── */

export default function ReportViewer() {
  const { imageId, analysisResult, report, pdfUrl, setReport } = useAppStore();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleGenerate = async () => {
    if (!imageId || !analysisResult) return;
    setLoading(true);
    setError(null);

    try {
      const resp = await generateReport(imageId, analysisResult);
      setReport(resp.report, resp.pdf_url);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Failed to generate report");
    } finally {
      setLoading(false);
    }
  };

  const handleDownloadPdf = () => {
    if (!pdfUrl) return;
    const url = fileUrl(pdfUrl);
    const a = document.createElement("a");
    a.href = url;
    a.download = `BTXRD_Report_${imageId || "case"}.pdf`;
    a.target = "_blank";
    a.click();
  };

  const reportElements = useMemo(() => {
    if (!report) return null;
    return renderMarkdown(report);
  }, [report]);

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="rounded-2xl border border-[#1f2937] overflow-hidden bg-[#0f172a]"
    >
      {/* Header */}
      <div className="px-6 py-4 border-b border-[#1f2937] bg-[#111827] flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-blue-500 to-blue-600 flex items-center justify-center shadow-lg shadow-blue-500/20">
            <FileText className="w-5 h-5 text-white" />
          </div>
          <div>
            <h3 className="text-lg font-semibold text-[#f8fafc]">Radiology Report</h3>
            <p className="text-xs text-[#64748b]">
              {report ? "AI-generated structured clinical report" : "Generate a professional report from AI analysis"}
            </p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          {report && (
            <>
              <motion.button
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                onClick={handleGenerate}
                disabled={loading}
                className="flex items-center gap-1.5 px-3 py-2 rounded-lg border border-[#334155]
                           text-xs font-medium text-[#94a3b8] hover:text-[#e2e8f0] hover:bg-[#1e293b] transition-all
                           disabled:opacity-50"
                title="Regenerate report"
              >
                <RefreshCw className={`w-3.5 h-3.5 ${loading ? "animate-spin" : ""}`} />
                Regenerate
              </motion.button>
              {pdfUrl && (
                <motion.button
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                  onClick={handleDownloadPdf}
                  className="flex items-center gap-1.5 px-4 py-2 rounded-lg bg-gradient-to-r from-blue-500 to-blue-600
                             text-xs font-semibold text-white shadow-lg shadow-blue-500/20 hover:shadow-blue-500/30 transition-all"
                >
                  <Download className="w-3.5 h-3.5" />
                  Download PDF
                </motion.button>
              )}
            </>
          )}
        </div>
      </div>

      {/* Content */}
      <div className="p-6">
        {/* Empty state */}
        {!report && !loading && !error && (
          <div className="text-center py-16">
            <div className="w-16 h-16 rounded-full bg-[#1e293b] flex items-center justify-center mx-auto mb-4">
              <FileText className="w-8 h-8 text-[#475569]" />
            </div>
            <p className="text-base font-medium text-[#cbd5e1] mb-2">No Report Generated</p>
            <p className="text-sm text-[#64748b] mb-6 max-w-md mx-auto">
              Generate a professional, radiologist-style report with findings, impressions, and recommendations based on the AI analysis.
            </p>
            <motion.button
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              onClick={handleGenerate}
              className="px-8 py-3 rounded-xl bg-gradient-to-r from-blue-500 to-blue-600 text-white font-semibold
                         shadow-lg shadow-blue-500/20 hover:shadow-blue-500/30 transition-all"
            >
              Generate Report
            </motion.button>
          </div>
        )}

        {/* Loading */}
        {loading && (
          <div className="text-center py-16">
            <Loader2 className="w-10 h-10 animate-spin text-blue-500 mx-auto mb-4" />
            <p className="text-sm font-medium text-[#cbd5e1]">Generating clinical report...</p>
            <p className="text-xs text-[#64748b] mt-1">This may take a few seconds</p>
          </div>
        )}

        {/* Error */}
        {error && !loading && (
          <div className="text-center py-12">
            <div className="w-12 h-12 rounded-full bg-red-950/50 flex items-center justify-center mx-auto mb-3">
              <AlertTriangle className="w-6 h-6 text-red-400" />
            </div>
            <p className="text-sm font-medium text-red-400 mb-1">Report Generation Failed</p>
            <p className="text-xs text-[#64748b] mb-4">{error}</p>
            <button
              onClick={handleGenerate}
              className="text-sm text-blue-400 hover:text-blue-300 font-medium transition-colors"
            >
              Try Again
            </button>
          </div>
        )}

        {/* Report content */}
        {report && !loading && (
          <div className="space-y-1">
            {/* Status bar */}
            <div className="flex items-center gap-2 mb-6 p-3 rounded-lg bg-emerald-950/30 border border-emerald-800/30">
              <CheckCircle className="w-4 h-4 text-emerald-400 flex-shrink-0" />
              <p className="text-xs text-emerald-300">
                Report generated successfully
                {pdfUrl ? " — PDF ready for download" : ""}
              </p>
            </div>

            {/* Rendered report */}
            <div className="space-y-0.5">
              {reportElements}
            </div>
          </div>
        )}
      </div>

      {/* Bottom disclaimer */}
      {report && (
        <div className="px-6 py-3 bg-amber-950/30 border-t border-amber-900/30">
          <p className="text-xs text-amber-300/80">
            ⚠️ This report is AI-generated for educational and research purposes only. It does NOT constitute a
            medical diagnosis. Always consult a qualified healthcare professional for clinical decision-making.
          </p>
        </div>
      )}
    </motion.div>
  );
}
