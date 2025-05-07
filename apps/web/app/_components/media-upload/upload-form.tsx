"use client";
import React, { ChangeEvent, useRef } from "react";

interface UploadFormProps {
  onFileSelect: (file: File) => void;
  disabled?: boolean;
  acceptedTypes?: string;
}

const UploadForm: React.FC<UploadFormProps> = ({
  onFileSelect,
  disabled = false,
  acceptedTypes = "image/*, video/*",
}) => {
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      onFileSelect(file);
    }
  };

  const handleButtonClick = () => {
    fileInputRef.current?.click();
  };

  return (
    <div className="flex flex-col items-center w-full max-w-md mx-auto p-5">
      <input
        type="file"
        ref={fileInputRef}
        onChange={handleFileChange}
        accept={acceptedTypes}
        className="hidden"
        disabled={disabled}
      />
      <button
        onClick={handleButtonClick}
        disabled={disabled}
        className={`bg-blue-500 hover:bg-blue-600 text-white font-medium py-3 px-6 rounded transition-colors mb-2 ${
          disabled ? "bg-gray-400 cursor-not-allowed" : ""
        }`}
      >
        Select Image/Video
      </button>
      <p className="text-sm text-gray-500 mt-2">
        Supported formats: JPG, PNG, MP4, MOV
      </p>
    </div>
  );
};

export default UploadForm;
