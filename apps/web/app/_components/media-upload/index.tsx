"use client";
import React from "react";
import { useMediaUpload } from "../../hooks/use-media-upload";
import UploadForm from "./upload-form";
import MediaPreview from "./media-preview";
import ProcessingIndicator from "./processing-indicator";
import ResultDisplay from "./result-display";

const MediaUpload: React.FC = () => {
  const {
    file,
    status,
    progress,
    uploadedMedia,
    processedMedia,
    selectFile,
    startProcessing,
    reset,
  } = useMediaUpload();

  return (
    <div className="max-w-2xl mx-auto p-5 flex flex-col items-center min-h-[400px]">
      <h2 className="text-2xl font-bold mb-8 text-gray-800 text-center">
        Rust Detection System
      </h2>

      {status === "idle" && !uploadedMedia && (
        <UploadForm onFileSelect={selectFile} />
      )}

      {file && uploadedMedia && status === "idle" && (
        <>
          <MediaPreview
            mediaUrl={uploadedMedia.url}
            mediaType={uploadedMedia.type}
            fileName={uploadedMedia.filename}
          />

          <button
            onClick={startProcessing}
            className="bg-green-500 hover:bg-green-600 text-white font-medium py-3 px-7 rounded transition-colors my-5"
          >
            Start Rust Detection
          </button>
        </>
      )}

      {(status === "uploading" || status === "processing") && (
        <ProcessingIndicator progress={progress} />
      )}

      {status === "complete" && processedMedia && (
        <ResultDisplay processedMedia={processedMedia} onReset={reset} />
      )}

      {status === "error" && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-5 w-full my-5 text-center">
          <h3 className="text-xl text-red-600 mb-2">An error occurred</h3>
          <p className="text-red-800 mb-5">
            There was a problem processing your media. Please try again.
          </p>
          <button
            onClick={reset}
            className="bg-red-600 hover:bg-red-700 text-white font-medium py-2 px-5 rounded transition-colors"
          >
            Try Again
          </button>
        </div>
      )}
    </div>
  );
};

export default MediaUpload;
