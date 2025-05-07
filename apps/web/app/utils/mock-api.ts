// Types
import { RUST_SAMPLE_IMAGES, RUST_DETECTION_RESULTS } from "./sample-data";

export type MediaType = "image" | "video";
export type UploadStatus =
  | "idle"
  | "uploading"
  | "processing"
  | "complete"
  | "error";
export type UploadedMedia = {
  id: string;
  url: string;
  type: MediaType;
  filename: string;
};
export type ProcessedMedia = {
  id: string;
  originalUrl: string;
  processedUrl: string;
  type: MediaType;
  detectionResults?: {
    severity: string;
    affectedArea: string;
    hotspots: Array<{
      top: string;
      left: string;
      width: string;
      height: string;
    }>;
  };
};

// Helper to generate unique IDs
const generateId = () => Math.random().toString(36).substring(2, 9);

// Helper to simulate network delay
const delay = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

/**
 * Mock function to simulate uploading media to server
 * In a real app, this would save the file to public folder
 */
export const uploadMedia = async (file: File): Promise<UploadedMedia> => {
  // Simulate network delay (500-1500ms)
  await delay(500 + Math.random() * 1000);

  // Create object URL for client-side preview
  const previewUrl = URL.createObjectURL(file);

  // Determine if it's an image or video
  const type: MediaType = file.type.startsWith("image/") ? "image" : "video";

  return {
    id: generateId(),
    url: previewUrl,
    type,
    filename: file.name,
  };
};

/**
 * Mock function to simulate processing media for rust detection
 */
export const processMedia = async (
  uploadedMedia: UploadedMedia
): Promise<ProcessedMedia> => {
  // Simulate processing time (2-5 seconds)
  const totalTime = 2000 + Math.random() * 3000;
  const steps = 10;

  // Simulate stepwise progress
  for (let i = 0; i < steps; i++) {
    await delay(totalTime / steps);
  }

  // Default to the original URL
  let processedUrl = uploadedMedia.url;
  let detectionResults = undefined;

  if (uploadedMedia.type === "image") {
    // Select a random sample rust image and detection result
    const randomIndex = Math.floor(Math.random() * RUST_SAMPLE_IMAGES.length);
    // Always ensure we have a valid string URL
    processedUrl = RUST_SAMPLE_IMAGES[randomIndex] || uploadedMedia.url;
    detectionResults = RUST_DETECTION_RESULTS[randomIndex];
  }

  return {
    id: uploadedMedia.id,
    originalUrl: uploadedMedia.url,
    processedUrl, // Will always be a string
    type: uploadedMedia.type,
    detectionResults,
  };
};

/**
 * Mock function to get processing status
 */
export const getProcessingStatus = async (
  id: string,
  progress: number
): Promise<number> => {
  // Simulate getting updated progress
  await delay(300);
  return Math.min(progress + Math.random() * 15, 100);
};
