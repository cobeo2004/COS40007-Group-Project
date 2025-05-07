"use client";
import React from "react";

interface ProcessingIndicatorProps {
  progress: number;
}

const ProcessingIndicator: React.FC<ProcessingIndicatorProps> = ({
  progress,
}) => {
  // Round progress to nearest integer
  const roundedProgress = Math.round(progress);

  return (
    <div className="w-full max-w-md mx-auto my-8 flex flex-col items-center">
      <h3 className="text-xl font-semibold mb-5 text-gray-800">
        Detecting Rust...
      </h3>

      <div className="w-full h-3 bg-gray-200 rounded-full overflow-hidden mb-2">
        <div
          className="h-full bg-gradient-to-r from-blue-500 to-green-500 rounded-full transition-all duration-300"
          style={{ width: `${roundedProgress}%` }}
        />
      </div>

      <div className="text-base font-medium text-gray-700 mb-4">
        {roundedProgress}% Complete
      </div>

      <p className="text-sm text-gray-500 text-center animate-pulse">
        Our AI model is analyzing the{" "}
        {roundedProgress < 50 ? "image" : "rust patterns"}...
      </p>
    </div>
  );
};

export default ProcessingIndicator;
