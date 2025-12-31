//! YOLO Object Detection with WASI-NN
//!
//! Complete example showing:
//! 1. Loading images from file or HTTP
//! 2. Image preprocessing (resize, normalize to 640x640)
//! 3. Running YOLO inference via WASI-NN
//! 4. Post-processing with NMS and confidence filtering
//!
//! Model requirement: ONNX format (YOLOv8n.onnx)
//! Runtime: Wasmtime or WasmEdge with WASI-NN support

use std::fs;
use std::path::Path;

// Image decoding
use image::ImageReader;
use image::GenericImageView;

// Import the WASI-NN bindings from the Bazel-generated crate
use crate::bindings::wasi::nn::graph::{self, Graph, GraphEncoding, ExecutionTarget};
use crate::bindings::wasi::nn::tensor::{Tensor, TensorType};
use crate::bindings::wasi::nn::inference::GraphExecutionContext;

// ============================================================================
// Detection Output Structures
// ============================================================================

/// A detected object with bounding box and confidence
#[derive(Debug, Clone)]
pub struct Detection {
    pub class_id: usize,
    pub class_name: String,
    pub confidence: f32,
    /// Pixel coords: (x1, y1, x2, y2)
    pub bbox_pixels: (u32, u32, u32, u32),
}

/// Image data (640x640 RGB)
pub struct Image {
    pub data: Vec<u8>,
    pub width: usize,
    pub height: usize,
}

impl Image {
    /// Load and decode image from file (JPEG, PNG, or raw RGB)
    /// Automatically resizes to 640x640
    pub fn from_file(path: &str) -> Result<Self, String> {
        let path = Path::new(path);
        let ext = path.extension()
            .and_then(|e| e.to_str())
            .map(|e| e.to_lowercase());

        match ext.as_deref() {
            Some("jpg") | Some("jpeg") | Some("png") => {
                Self::decode_image(path)
            }
            Some("rgb") | Some("raw") => {
                Self::load_raw(path)
            }
            _ => {
                // Try to decode as image first, fall back to raw
                Self::decode_image(path)
                    .or_else(|_| Self::load_raw(path))
            }
        }
    }

    /// Decode JPEG/PNG and resize to 640x640
    fn decode_image(path: &Path) -> Result<Self, String> {
        let img = ImageReader::open(path)
            .map_err(|e| format!("Failed to open {}: {}", path.display(), e))?
            .decode()
            .map_err(|e| format!("Failed to decode {}: {}", path.display(), e))?;

        let (orig_w, orig_h) = img.dimensions();
        println!("Loaded image: {}x{} -> resizing to 640x640", orig_w, orig_h);

        // Resize to 640x640
        let resized = img.resize_exact(640, 640, image::imageops::FilterType::Triangle);
        let rgb = resized.to_rgb8();
        let data = rgb.into_raw();

        Ok(Self {
            data,
            width: 640,
            height: 640,
        })
    }

    /// Load raw RGB file (expects 640x640x3 bytes)
    fn load_raw(path: &Path) -> Result<Self, String> {
        let data = fs::read(path)
            .map_err(|e| format!("Failed to read {}: {}", path.display(), e))?;

        println!("Loaded raw image: {} bytes", data.len());

        if data.len() != 640 * 640 * 3 {
            return Err(format!(
                "Raw image must be 640x640 RGB ({} bytes), got {} bytes",
                640 * 640 * 3,
                data.len()
            ));
        }

        Ok(Self {
            data,
            width: 640,
            height: 640,
        })
    }

    /// Create test image (640x640, mid-gray)
    pub fn test_image() -> Self {
        Self {
            data: vec![128u8; 640 * 640 * 3],
            width: 640,
            height: 640,
        }
    }
}

// ============================================================================
// Image Preprocessing
// ============================================================================

/// Convert image to YOLO tensor format
/// Input: RGB image (640x640)
/// Output: 3x640x640 float32 in [0, 1] range, channels-first (CHW) format
pub fn preprocess_image(image: &Image) -> Result<Vec<f32>, String> {
    const TARGET: usize = 640;

    if image.width != TARGET || image.height != TARGET {
        return Err(format!(
            "Image must be {}x{}, got {}x{}",
            TARGET, TARGET, image.width, image.height
        ));
    }

    // Convert RGB to YOLO tensor: 3x640x640
    // YOLO expects float32 in [0, 1] range
    let mut tensor = vec![0.0f32; 3 * 640 * 640];

    // Split channels: R, G, B (separate planes)
    for y in 0..640 {
        for x in 0..640 {
            let pixel_idx = (y * 640 + x) * 3;
            let r = image.data[pixel_idx] as f32 / 255.0;
            let g = image.data[pixel_idx + 1] as f32 / 255.0;
            let b = image.data[pixel_idx + 2] as f32 / 255.0;

            // CHW layout (channels first)
            tensor[0 * 640 * 640 + y * 640 + x] = r;
            tensor[1 * 640 * 640 + y * 640 + x] = g;
            tensor[2 * 640 * 640 + y * 640 + x] = b;
        }
    }

    Ok(tensor)
}

// ============================================================================
// YOLO Post-Processing (NMS + Filtering)
// ============================================================================

pub struct YoloPostProcessor {
    pub confidence_threshold: f32,
    pub nms_threshold: f32,
}

impl Default for YoloPostProcessor {
    fn default() -> Self {
        Self {
            confidence_threshold: 0.45,
            nms_threshold: 0.45,
        }
    }
}

impl YoloPostProcessor {
    /// Parse YOLOv8 output and apply NMS
    /// YOLOv8 output format: [1, 84, 8400] transposed to [84, 8400]
    /// - 8400 detection anchors
    /// - 84 values per anchor: [cx, cy, w, h, class_0..class_79]
    /// - No objectness score - class scores are direct confidences
    /// - Coordinates are in pixel space (0-640)
    pub fn process(
        &self,
        output: &[f32],
        image_width: usize,
        image_height: usize,
    ) -> Result<Vec<Detection>, String> {
        const NUM_ANCHORS: usize = 8400;
        const NUM_CLASSES: usize = 80;
        const NUM_FEATURES: usize = 84; // 4 bbox + 80 classes

        // Verify output size
        if output.len() != NUM_ANCHORS * NUM_FEATURES {
            return Err(format!(
                "Unexpected output size: {} (expected {})",
                output.len(),
                NUM_ANCHORS * NUM_FEATURES
            ));
        }

        let mut detections = Vec::new();

        // YOLOv8 output is transposed: [84, 8400]
        // Row 0: all cx values, Row 1: all cy values, etc.
        for anchor_idx in 0..NUM_ANCHORS {
            // Extract bbox (transposed access)
            let cx = output[0 * NUM_ANCHORS + anchor_idx];
            let cy = output[1 * NUM_ANCHORS + anchor_idx];
            let w = output[2 * NUM_ANCHORS + anchor_idx];
            let h = output[3 * NUM_ANCHORS + anchor_idx];

            // Find best class and its confidence
            let mut best_class_id = 0;
            let mut best_conf = output[4 * NUM_ANCHORS + anchor_idx];

            for class_id in 1..NUM_CLASSES {
                let conf = output[(4 + class_id) * NUM_ANCHORS + anchor_idx];
                if conf > best_conf {
                    best_conf = conf;
                    best_class_id = class_id;
                }
            }

            // Filter by confidence
            if best_conf < self.confidence_threshold {
                continue;
            }

            // Convert center coordinates to corner coordinates
            // YOLOv8 outputs pixel coordinates directly
            let x1 = ((cx - w / 2.0).max(0.0) as u32).min(image_width as u32 - 1);
            let y1 = ((cy - h / 2.0).max(0.0) as u32).min(image_height as u32 - 1);
            let x2 = ((cx + w / 2.0) as u32).min(image_width as u32 - 1);
            let y2 = ((cy + h / 2.0) as u32).min(image_height as u32 - 1);

            // Skip invalid boxes
            if x2 <= x1 || y2 <= y1 {
                continue;
            }

            detections.push(Detection {
                class_id: best_class_id,
                class_name: get_class_name(best_class_id).to_string(),
                confidence: best_conf,
                bbox_pixels: (x1, y1, x2, y2),
            });
        }

        println!("Detected {} objects (pre-NMS)", detections.len());

        // Apply Non-Maximum Suppression
        detections.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());
        let mut keep: Vec<Detection> = Vec::new();

        for det_a in detections.iter() {
            let mut suppress = false;

            for det_b in keep.iter() {
                // Only compare same class
                if det_a.class_id != det_b.class_id {
                    continue;
                }

                let iou = calculate_iou(det_a.bbox_pixels, det_b.bbox_pixels);
                if iou > self.nms_threshold {
                    suppress = true;
                    break;
                }
            }

            if !suppress {
                keep.push(det_a.clone());
            }
        }

        println!("After NMS: {} detections", keep.len());
        Ok(keep)
    }
}

/// Calculate Intersection over Union for two boxes
fn calculate_iou(box1: (u32, u32, u32, u32), box2: (u32, u32, u32, u32)) -> f32 {
    let (x1_min, y1_min, x1_max, y1_max) = box1;
    let (x2_min, y2_min, x2_max, y2_max) = box2;

    let inter_xmin = x1_min.max(x2_min);
    let inter_ymin = y1_min.max(y2_min);
    let inter_xmax = x1_max.min(x2_max);
    let inter_ymax = y1_max.min(y2_max);

    if inter_xmax <= inter_xmin || inter_ymax <= inter_ymin {
        return 0.0;
    }

    let inter_area = ((inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)) as f32;
    let box1_area = ((x1_max - x1_min) * (y1_max - y1_min)) as f32;
    let box2_area = ((x2_max - x2_min) * (y2_max - y2_min)) as f32;
    let union_area = box1_area + box2_area - inter_area;

    inter_area / union_area
}

// ============================================================================
// COCO Class Names (80 classes)
// ============================================================================

const COCO_CLASSES: &[&str] = &[
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
    "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis",
    "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
];

fn get_class_name(class_id: usize) -> &'static str {
    COCO_CLASSES.get(class_id).unwrap_or(&"unknown")
}

// ============================================================================
// Main Detection Function
// ============================================================================

pub fn detect_objects(
    image_source: &str, // file path
    model_path: &str,
) -> Result<Vec<Detection>, String> {
    println!("\n======================================================");
    println!("  YOLO Object Detection with WASI-NN                  ");
    println!("======================================================");

    // Step 1: Load image
    println!("\n[1] Loading image...");
    let image = if Path::new(image_source).exists() {
        println!("  Source: File - {}", image_source);
        Image::from_file(image_source)?
    } else {
        println!("  File not found, using test image");
        Image::test_image()
    };
    println!("  Size: {}x{} RGB", image.width, image.height);

    // Step 2: Load YOLO model using WASI-NN
    println!("\n[2] Loading model...");
    println!("  Path: {}", model_path);
    let model_bytes = fs::read(model_path)
        .map_err(|e| format!("Failed to read model: {}", e))?;
    println!("  Size: {:.2} MB", model_bytes.len() as f32 / 1_000_000.0);

    // Load model using wit-bindgen generated graph::load()
    let graph: Graph = graph::load(&[model_bytes], GraphEncoding::Onnx, ExecutionTarget::Cpu)
        .map_err(|e| format!("WASI-NN load failed: {:?}", e.code()))?;
    println!("  Graph loaded successfully");

    // Step 3: Preprocess image
    println!("\n[3] Preprocessing image...");
    let tensor_data = preprocess_image(&image)?;
    println!("  Tensor: 3x640x640 ({:.2} MB)", tensor_data.len() as f32 * 4.0 / 1_000_000.0);

    // Convert f32 to bytes for tensor
    let tensor_bytes: Vec<u8> = tensor_data
        .iter()
        .flat_map(|f| f.to_le_bytes())
        .collect();

    // Step 4: Create execution context
    println!("\n[4] Creating execution context...");
    let context: GraphExecutionContext = graph.init_execution_context()
        .map_err(|e| format!("Failed to init context: {:?}", e.code()))?;
    println!("  Context created");

    // Step 5: Create input tensor and run inference
    println!("\n[5] Running inference...");

    // Create tensor with dimensions [1, 3, 640, 640] for batch of 1
    let dimensions: Vec<u32> = vec![1, 3, 640, 640];
    let input_tensor = Tensor::new(&dimensions, TensorType::Fp32, &tensor_bytes);

    // Run compute with named tensor input
    let inputs = vec![("images".to_string(), input_tensor)];
    let outputs = context.compute(inputs)
        .map_err(|e| format!("Inference failed: {:?}", e.code()))?;
    println!("  Inference complete");

    // Step 6: Process output
    println!("\n[6] Reading output...");

    // Get the first output tensor (YOLOv8 output)
    let (output_name, output_tensor) = outputs.into_iter()
        .next()
        .ok_or_else(|| "No output tensor found".to_string())?;
    println!("  Output name: {}", output_name);

    let output_bytes = output_tensor.data();
    let output: Vec<f32> = output_bytes
        .chunks(4)
        .map(|chunk| {
            let bytes = [chunk[0], chunk[1], chunk[2], chunk[3]];
            f32::from_le_bytes(bytes)
        })
        .collect();

    println!("  Output: {} values ({:.2} MB)",
             output.len(),
             output.len() as f32 * 4.0 / 1_000_000.0);

    // Step 7: Post-process
    println!("\n[7] Post-processing...");
    let processor = YoloPostProcessor::default();
    let detections = processor.process(&output, image.width, image.height)?;

    // Print results
    println!("\n======================================================");
    println!("  Results: {} detections                              ", detections.len());
    println!("======================================================\n");

    for (idx, det) in detections.iter().enumerate() {
        println!(
            "  [{:2}] {:<20} {:.1}% @ pixel ({:4}, {:4}) -> ({:4}, {:4})",
            idx + 1,
            det.class_name,
            det.confidence * 100.0,
            det.bbox_pixels.0,
            det.bbox_pixels.1,
            det.bbox_pixels.2,
            det.bbox_pixels.3,
        );
    }

    Ok(detections)
}

// ============================================================================
// Entry Point
// ============================================================================

pub fn run_yolo() {
    // Get command line arguments
    let args: Vec<String> = std::env::args().collect();

    let image_path = args.get(1)
        .map(|s| s.as_str())
        .unwrap_or("./test_image.jpg");

    let model_path = args.get(2)
        .map(|s| s.as_str())
        .unwrap_or("./models/yolov8n.onnx");

    println!("YOLO Object Detection Demo");
    println!("---------------------------");
    println!("Image: {}", image_path);
    println!("Model: {}", model_path);
    println!();

    match detect_objects(image_path, model_path) {
        Ok(detections) => {
            println!("\n[OK] Detection complete");
            println!("  Found {} objects", detections.len());

            // Count by class
            let mut class_counts: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
            for det in &detections {
                *class_counts.entry(det.class_name.clone()).or_insert(0) += 1;
            }

            println!("\n  Objects by class:");
            for (class, count) in class_counts {
                println!("    - {}: {}", class, count);
            }
        }
        Err(e) => {
            eprintln!("\n[ERROR] {}\n", e);
        }
    }
}
