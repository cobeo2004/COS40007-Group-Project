import { useState, useCallback, useEffect } from "react";
import {
  uploadMedia,
  processMedia,
  getProcessingStatus,
  UploadStatus,
  UploadedMedia,
  ProcessedMedia,
} from "../utils/mock-api";

export interface MediaUploadState {
  file: File | null;
  status: UploadStatus;
  progress: number;
  uploadedMedia: UploadedMedia | null;
  processedMedia: ProcessedMedia | null;
  error: Error | null;
}

export function useMediaUpload() {
  const [state, setState] = useState<MediaUploadState>({
    file: null,
    status: "idle",
    progress: 0,
    uploadedMedia: null,
    processedMedia: null,
    error: null,
  });

  // Reset the state and clean up object URLs
  const reset = useCallback(() => {
    // Clean up object URLs before resetting
    if (state.uploadedMedia?.url && !state.uploadedMedia.url.startsWith("/")) {
      URL.revokeObjectURL(state.uploadedMedia.url);
    }

    setState({
      file: null,
      status: "idle",
      progress: 0,
      uploadedMedia: null,
      processedMedia: null,
      error: null,
    });
  }, [state.uploadedMedia?.url]);

  // Handle file selection
  const selectFile = useCallback((file: File) => {
    setState((prevState) => ({
      ...prevState,
      file,
      status: "idle",
      progress: 0,
      error: null,
    }));
  }, []);

  // Start upload and processing
  const startProcessing = useCallback(async () => {
    if (!state.file) return;

    try {
      // Set status to uploading
      setState((prevState) => ({
        ...prevState,
        status: "uploading",
        progress: 0,
      }));

      // Upload the file
      const uploaded = await uploadMedia(state.file);

      // Set status to processing
      setState((prevState) => ({
        ...prevState,
        status: "processing",
        uploadedMedia: uploaded,
        progress: 0,
      }));

      // Start progress simulation
      let currentProgress = 0;

      // Poll for progress updates
      while (currentProgress < 100) {
        currentProgress = await getProcessingStatus(
          uploaded.id,
          currentProgress
        );

        setState((prevState) => ({
          ...prevState,
          progress: currentProgress,
        }));

        if (currentProgress >= 100) break;
      }

      // Process the media
      const processed = await processMedia(uploaded);

      // Set status to complete
      setState((prevState) => ({
        ...prevState,
        status: "complete",
        processedMedia: processed,
        progress: 100,
      }));
    } catch (err) {
      setState((prevState) => ({
        ...prevState,
        status: "error",
        error: err instanceof Error ? err : new Error("Unknown error occurred"),
      }));
    }
  }, [state.file]);

  // Clean up object URLs when component unmounts
  useEffect(() => {
    return () => {
      // Only revoke object URLs (those that don't start with '/')
      if (
        state.uploadedMedia?.url &&
        !state.uploadedMedia.url.startsWith("/")
      ) {
        URL.revokeObjectURL(state.uploadedMedia.url);
      }
    };
  }, [state.uploadedMedia?.url]);

  return {
    ...state,
    selectFile,
    startProcessing,
    reset,
  };
}
