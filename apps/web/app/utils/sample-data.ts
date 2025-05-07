/**
 * Sample data for rust detection demo
 */

// Sample rust detection image paths (from public folder)
export const RUST_SAMPLE_IMAGES = [
  // These images should be placed in the public folder
  // Example: public/rust-images/rust1.jpg
  "/rust-images/rust1.jpg",
  "/rust-images/rust2.jpg",
  "/rust-images/rust3.jpg",
];

// Sample rust detection results
export const RUST_DETECTION_RESULTS = [
  {
    id: "rust1",
    severity: "High",
    affectedArea: "28%",
    hotspots: [
      { top: "20%", left: "30%", width: "20%", height: "15%" },
      { top: "55%", left: "60%", width: "25%", height: "20%" },
    ],
  },
  {
    id: "rust2",
    severity: "Medium",
    affectedArea: "15%",
    hotspots: [
      { top: "35%", left: "45%", width: "30%", height: "10%" },
      { top: "70%", left: "20%", width: "15%", height: "12%" },
    ],
  },
  {
    id: "rust3",
    severity: "Low",
    affectedArea: "8%",
    hotspots: [
      { top: "15%", left: "55%", width: "12%", height: "8%" },
      { top: "40%", left: "25%", width: "18%", height: "10%" },
    ],
  },
];
