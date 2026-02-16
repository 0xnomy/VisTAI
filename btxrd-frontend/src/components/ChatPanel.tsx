"use client";

import React, { useState, useRef, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Send, Bot, User, Loader2 } from "lucide-react";
import { sendChat, type ChatMessage } from "@/lib/api";
import { useAppStore } from "@/store/useAppStore";

/* ── Inline markdown renderer ────────────────────────────────────────── */

function renderInline(text: string): React.ReactNode {
  const parts: React.ReactNode[] = [];
  let remaining = text;
  let k = 0;

  while (remaining.length > 0) {
    // Bold: **text**
    const boldMatch = remaining.match(/\*\*(.+?)\*\*/);
    if (boldMatch && boldMatch.index !== undefined) {
      if (boldMatch.index > 0) {
        parts.push(<span key={k++}>{remaining.slice(0, boldMatch.index)}</span>);
      }
      parts.push(<strong key={k++} className="font-semibold text-white">{boldMatch[1]}</strong>);
      remaining = remaining.slice(boldMatch.index + boldMatch[0].length);
      continue;
    }
    // Italic: *text*
    const italicMatch = remaining.match(/\*(.+?)\*/);
    if (italicMatch && italicMatch.index !== undefined) {
      if (italicMatch.index > 0) {
        parts.push(<span key={k++}>{remaining.slice(0, italicMatch.index)}</span>);
      }
      parts.push(<em key={k++}>{italicMatch[1]}</em>);
      remaining = remaining.slice(italicMatch.index + italicMatch[0].length);
      continue;
    }
    parts.push(<span key={k++}>{remaining}</span>);
    break;
  }
  return parts.length === 1 ? parts[0] : <>{parts}</>;
}

function renderChatMessage(content: string): React.ReactNode {
  const lines = content.split("\n");
  const elements: React.ReactNode[] = [];
  let key = 0;

  for (const line of lines) {
    const trimmed = line.trim();

    if (!trimmed) {
      elements.push(<div key={key++} className="h-1.5" />);
      continue;
    }

    // Numbered list
    const numMatch = trimmed.match(/^(\d+)\.\s+(.*)/);
    if (numMatch) {
      elements.push(
        <div key={key++} className="flex gap-2 py-0.5">
          <span className="text-blue-300 font-medium min-w-[18px]">{numMatch[1]}.</span>
          <span>{renderInline(numMatch[2])}</span>
        </div>
      );
      continue;
    }

    // Bullet list
    if (trimmed.startsWith("• ") || trimmed.startsWith("- ") || trimmed.startsWith("* ")) {
      const text = trimmed.replace(/^[•\-*]\s*/, "");
      elements.push(
        <div key={key++} className="flex gap-2 py-0.5">
          <span className="text-blue-300">•</span>
          <span>{renderInline(text)}</span>
        </div>
      );
      continue;
    }

    // Regular line
    elements.push(<div key={key++} className="py-0.5">{renderInline(trimmed)}</div>);
  }

  return <div className="space-y-0">{elements}</div>;
}

/* ── ChatPanel Component ─────────────────────────────────────────────── */

export default function ChatPanel() {
  const { imageId, analysisResult, chatHistory, addChat } = useAppStore();
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [chatHistory]);

  const handleSend = async () => {
    if (!input.trim() || !imageId || !analysisResult || loading) return;
    const userMsg = input.trim();
    setInput("");

    // Add user message to history
    addChat({ role: "user", content: userMsg });
    setLoading(true);

    try {
      // Send previous history only — the backend appends the current message itself
      const resp = await sendChat(imageId, userMsg, analysisResult, chatHistory);
      addChat({ role: "assistant", content: resp.reply });
    } catch (e: unknown) {
      addChat({
        role: "assistant",
        content: `Error: ${e instanceof Error ? e.message : "Something went wrong. Please try again."}`,
      });
    } finally {
      setLoading(false);
    }
  };

  const handleSuggestion = (text: string) => {
    setInput(text);
  };

  const suggestions = [
    "What type of tumor is this?",
    "Is it malignant or benign?",
    "What does the segmentation show?",
    "How confident is the model?",
    "What are the differential predictions?",
    "What are the recommendations?",
  ];

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="rounded-2xl border border-[#1f2937] overflow-hidden bg-[#0f172a] flex flex-col h-[600px]"
    >
      {/* Header */}
      <div className="px-5 py-4 border-b border-[#1f2937] bg-[#111827]">
        <h3 className="text-lg font-semibold text-[#f8fafc] flex items-center gap-2">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-blue-500 to-blue-600 flex items-center justify-center">
            <Bot className="w-4 h-4 text-white" />
          </div>
          AI Medical Assistant
        </h3>
        <p className="text-xs text-[#64748b] mt-1">Ask questions about the analysis results</p>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-5 py-4 space-y-4">
        {chatHistory.length === 0 && (
          <div className="text-center py-8">
            <div className="w-14 h-14 rounded-full bg-[#1e293b] flex items-center justify-center mx-auto mb-4">
              <Bot className="w-7 h-7 text-[#475569]" />
            </div>
            <p className="text-sm text-[#94a3b8] mb-1 font-medium">How can I help you?</p>
            <p className="text-xs text-[#64748b] mb-5">Ask me anything about the X-ray analysis</p>
            <div className="flex flex-wrap gap-2 justify-center max-w-md mx-auto">
              {suggestions.map((s) => (
                <button
                  key={s}
                  onClick={() => handleSuggestion(s)}
                  className="text-xs px-3 py-1.5 rounded-full border border-[#334155] text-[#94a3b8]
                             hover:bg-blue-500/10 hover:border-blue-500/30 hover:text-blue-300 transition-all"
                >
                  {s}
                </button>
              ))}
            </div>
          </div>
        )}

        <AnimatePresence>
          {chatHistory.map((msg, i) => (
            <motion.div
              key={i}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.2 }}
              className={`flex gap-3 ${msg.role === "user" ? "justify-end" : ""}`}
            >
              {msg.role === "assistant" && (
                <div className="w-7 h-7 rounded-full bg-gradient-to-br from-blue-500 to-blue-600 flex items-center justify-center flex-shrink-0 mt-1">
                  <Bot className="w-3.5 h-3.5 text-white" />
                </div>
              )}
              <div
                className={`max-w-[80%] px-4 py-3 rounded-2xl text-sm leading-relaxed
                  ${msg.role === "user"
                    ? "bg-gradient-to-r from-blue-500 to-blue-600 text-white rounded-br-md"
                    : "bg-[#1e293b] text-[#e2e8f0] rounded-bl-md border border-[#334155]"
                  }`}
              >
                {msg.role === "assistant" ? renderChatMessage(msg.content) : msg.content}
              </div>
              {msg.role === "user" && (
                <div className="w-7 h-7 rounded-full bg-[#334155] flex items-center justify-center flex-shrink-0 mt-1">
                  <User className="w-3.5 h-3.5 text-[#94a3b8]" />
                </div>
              )}
            </motion.div>
          ))}
        </AnimatePresence>

        {loading && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className="flex gap-3"
          >
            <div className="w-7 h-7 rounded-full bg-gradient-to-br from-blue-500 to-blue-600 flex items-center justify-center flex-shrink-0">
              <Bot className="w-3.5 h-3.5 text-white" />
            </div>
            <div className="px-4 py-3 rounded-2xl rounded-bl-md bg-[#1e293b] border border-[#334155]">
              <div className="flex items-center gap-1.5">
                <div className="w-1.5 h-1.5 rounded-full bg-blue-400 animate-bounce" style={{ animationDelay: "0ms" }} />
                <div className="w-1.5 h-1.5 rounded-full bg-blue-400 animate-bounce" style={{ animationDelay: "150ms" }} />
                <div className="w-1.5 h-1.5 rounded-full bg-blue-400 animate-bounce" style={{ animationDelay: "300ms" }} />
              </div>
            </div>
          </motion.div>
        )}

        <div ref={bottomRef} />
      </div>

      {/* Input */}
      <div className="px-4 py-3 border-t border-[#1f2937] bg-[#111827]">
        <div className="flex items-center gap-2">
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && !e.shiftKey && handleSend()}
            placeholder="Ask about the analysis..."
            className="flex-1 px-4 py-2.5 rounded-xl bg-[#0f172a] border border-[#334155]
                       text-sm text-[#f8fafc] placeholder:text-[#475569]
                       focus:outline-none focus:ring-2 focus:ring-blue-500/20 focus:border-blue-500/50 transition-all"
          />
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={handleSend}
            disabled={!input.trim() || loading}
            className="p-2.5 rounded-xl bg-gradient-to-r from-blue-500 to-blue-600 text-white shadow-lg shadow-blue-500/20
                       hover:shadow-blue-500/30 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
          >
            <Send className="w-4 h-4" />
          </motion.button>
        </div>
      </div>
    </motion.div>
  );
}
