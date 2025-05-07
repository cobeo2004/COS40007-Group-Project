"use client";
import React from "react";
import { MediaType } from "../../utils/mock-api";
import Image from "next/image";

interface MediaPreviewProps {
  mediaUrl: string;
  mediaType: MediaType;
  fileName: string;
}

const MediaPreview: React.FC<MediaPreviewProps> = ({
  mediaUrl,
  mediaType,
  fileName,
}) => {
  return (
    <div className="my-5 w-full max-w-md flex flex-col items-center">
      <h3 className="text-lg font-medium mb-3 capitalize">
        Selected {mediaType}:
      </h3>
      {mediaType === "image" ? (
        <div className="w-full rounded-lg overflow-hidden shadow-md bg-gray-100">
          <Image
            src={mediaUrl}
            alt={fileName}
            className="w-full max-h-[350px] object-contain block"
          />
        </div>
      ) : (
        <div className="w-full rounded-lg overflow-hidden shadow-md bg-gray-100">
          <video
            src={mediaUrl}
            controls
            className="w-full max-h-[350px] object-contain block"
            controlsList="nodownload"
          />
        </div>
      )}
      <p className="mt-2 text-sm text-gray-600 text-center break-all max-w-full">
        {fileName}
      </p>
    </div>
  );
};

export default MediaPreview;
