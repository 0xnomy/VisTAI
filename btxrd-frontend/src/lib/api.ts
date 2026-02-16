/**
 * API client – all calls to the FastAPI backend.
 */

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

/* ── helpers ────────────────────────────────────────────────────────────── */

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...init?.headers },
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `API error ${res.status}`);
  }
  return res.json();
}

/* ── Upload ─────────────────────────────────────────────────────────────── */

export interface UploadResult {
  image_id: string;
  filename: string;
  url: string;
}

export async function uploadImage(file: File): Promise<UploadResult> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${API_BASE}/api/upload`, { method: "POST", body: form });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || "Upload failed");
  }
  return res.json();
}

/* ── Inference ──────────────────────────────────────────────────────────── */

export interface ClassificationResult {
  top_class: string;
  confidence: number;
  malignancy: string;
  probabilities: Record<string, number>;
  top5: { class: string; probability: number }[];
}

export interface SegmentationResult {
  mask_url: string;
  overlay_url: string;
  gradcam_url: string;
  tumor_coverage: number;
}

export interface InferenceResult {
  image_id: string;
  classification: ClassificationResult | null;
  segmentation: SegmentationResult | null;
  cls_gradcam_url: string | null;
  original_url: string;
}

export async function runInference(
  imageId: string,
  intent: "classification" | "segmentation" | "full" = "full"
): Promise<InferenceResult> {
  return request<InferenceResult>("/api/inference", {
    method: "POST",
    body: JSON.stringify({ image_id: imageId, intent }),
  });
}

/* ── Chat ───────────────────────────────────────────────────────────────── */

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
}

export interface ChatResponse {
  reply: string;
}

export async function sendChat(
  imageId: string,
  message: string,
  analysis: InferenceResult,
  history: ChatMessage[] = []
): Promise<ChatResponse> {
  return request<ChatResponse>("/api/chat", {
    method: "POST",
    body: JSON.stringify({ image_id: imageId, message, analysis, history }),
  });
}

/* ── Report ─────────────────────────────────────────────────────────────── */

export interface ReportResponse {
  report: string;
  pdf_url: string | null;
}

export async function generateReport(
  imageId: string,
  analysis: InferenceResult
): Promise<ReportResponse> {
  return request<ReportResponse>("/api/report", {
    method: "POST",
    body: JSON.stringify({ image_id: imageId, analysis }),
  });
}

/* ── Helpers ────────────────────────────────────────────────────────────── */

export function fileUrl(path: string): string {
  if (path.startsWith("http")) return path;
  return `${API_BASE}${path}`;
}
