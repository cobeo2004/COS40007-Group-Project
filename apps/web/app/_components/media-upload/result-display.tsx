"use client";
import React from "react";
import { ProcessedMedia } from "../../utils/mock-api";
import Image from "next/image";

interface ResultDisplayProps {
  processedMedia: ProcessedMedia;
  onReset: () => void;
}

const ResultDisplay: React.FC<ResultDisplayProps> = ({
  processedMedia,
  onReset,
}) => {
  // Default values if detectionResults is not available
  const severity = processedMedia.detectionResults?.severity || "High";
  const affectedArea = processedMedia.detectionResults?.affectedArea || "~25%";
  const hotspots = processedMedia.detectionResults?.hotspots || [
    { top: "20%", left: "30%", width: "20%", height: "15%" },
    { top: "55%", left: "60%", width: "25%", height: "20%" },
  ];

  return (
    <div className="w-full max-w-lg mx-auto my-5 flex flex-col items-center">
      <h3 className="text-xl font-semibold mb-5 text-gray-800">
        Rust Detection Complete
      </h3>

      <div className="w-full mb-5">
        {processedMedia.type === "image" ? (
          <div className="w-full rounded-lg overflow-hidden shadow-lg">
            <div className="relative w-full">
              {/* In a real app, this would be the processed image with detection overlays */}
              {processedMedia.processedUrl.startsWith("/") ? (
                // Public directory image
                <img
                  src={processedMedia.processedUrl}
                  alt="Processed image with rust detection"
                  className="w-full max-h-[400px] object-contain block"
                />
              ) : (
                // Object URL from client-side file
                <img
                  src={processedMedia.processedUrl}
                  alt="Processed image with rust detection"
                  className="w-full max-h-[400px] object-contain block"
                />
              )}
              <div className="absolute top-0 left-0 w-full h-full pointer-events-none">
                {/* Render rust hotspots from detection results */}
                {hotspots.map((spot, index) => (
                  <div
                    key={index}
                    className="absolute rounded bg-red-400/40 border-2 border-red-500/80 animate-pulse"
                    style={{
                      top: spot.top,
                      left: spot.left,
                      width: spot.width,
                      height: spot.height,
                    }}
                  ></div>
                ))}
              </div>
            </div>
          </div>
        ) : (
          <div className="w-full rounded-lg overflow-hidden shadow-lg">
            <video
              src={processedMedia.processedUrl}
              controls
              className="w-full max-h-[400px] object-contain block"
              controlsList="nodownload"
            />
          </div>
        )}
      </div>

      <div className="bg-gray-100 rounded-lg p-4 w-full my-4">
        <h4 className="text-lg mb-2 text-gray-800">Detection Results:</h4>
        <ul className="list-none p-0">
          <li className="mb-2 text-base">
            Rust detected:{" "}
            <span className="text-red-600 font-semibold">Yes</span>
          </li>
          <li className="mb-2 text-base">
            Affected area: <span>{affectedArea}</span>
          </li>
          <li className="mb-2 text-base">
            Severity:{" "}
            <span
              className={`font-semibold ${
                severity === "High"
                  ? "text-red-600"
                  : severity === "Medium"
                    ? "text-orange-500"
                    : "text-yellow-500"
              }`}
            >
              {severity}
            </span>
          </li>
        </ul>
      </div>

      <button
        onClick={onReset}
        className="bg-blue-500 hover:bg-blue-600 text-white font-medium py-3 px-6 rounded transition-colors mt-4"
      >
        Upload Another
      </button>
    </div>
  );
};

export default ResultDisplay;
