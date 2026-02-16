"use client";

import { useCallback, useState } from "react";
import { useDropzone } from "react-dropzone";
import { motion, AnimatePresence } from "framer-motion";
import { Upload, ImageIcon, X, Loader2 } from "lucide-react";
import { uploadImage, fileUrl } from "@/lib/api";
import { useAppStore } from "@/store/useAppStore";

export default function ImageUploader() {
  const { setUpload, setStep } = useAppStore();
  const [preview, setPreview] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const onDrop = useCallback(
    async (accepted: File[]) => {
      const file = accepted[0];
      if (!file) return;
      setError(null);

      // Local preview
      const reader = new FileReader();
      reader.onload = () => setPreview(reader.result as string);
      reader.readAsDataURL(file);

      // Upload to backend
      setUploading(true);
      try {
        const result = await uploadImage(file);
        setUpload(result.image_id, fileUrl(result.url), result.filename);
        setStep("intent");
      } catch (e: unknown) {
        setError(e instanceof Error ? e.message : "Upload failed");
      } finally {
        setUploading(false);
      }
    },
    [setUpload, setStep]
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { "image/png": [], "image/jpeg": [], "image/jpg": [], "image/webp": [] },
    maxFiles: 1,
    maxSize: 20 * 1024 * 1024,
    disabled: uploading,
  });

  const clearPreview = () => {
    setPreview(null);
    setError(null);
  };

  return (
    <div className="w-full max-w-2xl mx-auto">
      <AnimatePresence mode="wait">
        {!preview ? (
          <motion.div
            key="dropzone"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            {...getRootProps()}
            className={
              isDragActive
                ? "relative border-2 border-dashed rounded-2xl p-12 text-center cursor-pointer transition-all duration-300 ease-out border-blue-500 bg-blue-900/20 scale-[1.02]"
                : "relative border-2 border-dashed rounded-2xl p-12 text-center cursor-pointer transition-all duration-300 ease-out border-slate-700 hover:border-blue-500 hover:bg-slate-800/50"
            }
          >
            <input {...getInputProps()} />
            <div className="flex flex-col items-center gap-4">
              <div className={`p-4 rounded-full transition-colors ${isDragActive ? "bg-blue-500/20" : "bg-slate-700"}`}>
                <Upload className={`w-8 h-8 ${isDragActive ? "text-blue-400" : "text-slate-400"}`} />
              </div>
              <div>
                <p className="text-lg font-semibold text-slate-200">
                  {isDragActive ? "Drop your X-ray here" : "Upload X-Ray Image"}
                </p>
                <p className="mt-1 text-sm text-slate-400">
                  Drag & drop or click to browse • PNG, JPG up to 20 MB
                </p>
              </div>
            </div>
          </motion.div>
        ) : (
          <motion.div
            key="preview"
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.95 }}
            className="relative rounded-2xl overflow-hidden border border-slate-700 bg-black/20"
          >
            <button
              onClick={clearPreview}
              className="absolute top-3 right-3 z-10 p-1.5 rounded-full bg-white/90 hover:bg-white shadow-md transition"
            >
              <X className="w-4 h-4 text-gray-600" />
            </button>
            <div className="relative aspect-square max-h-[400px] flex items-center justify-center bg-gray-900">
              <img
                src={preview}
                alt="X-ray preview"
                className="max-w-full max-h-full object-contain"
              />
              {uploading && (
                <div className="absolute inset-0 flex items-center justify-center bg-black/40 backdrop-blur-sm">
                  <div className="flex items-center gap-3 text-white">
                    <Loader2 className="w-6 h-6 animate-spin" />
                    <span className="font-medium">Uploading…</span>
                  </div>
                </div>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {error && (
        <motion.p
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="mt-4 text-center text-sm text-red-500 font-medium"
        >
          {error}
        </motion.p>
      )}
    </div>
  );
}
