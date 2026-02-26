import * as ort from "onnxruntime-web/wasm";
// ort.env.wasm.wasmPaths = "/ort/";

export type Detection = {
  cls: "light_square" | "socket" | "switches" | "TV_socket";
  conf: number;
  xyxy: [number, number, number, number]; // pixels in ORIGINAL PNG (top-left origin)
};

const CLASSES = ["light_square", "socket", "switches", "TV_socket"] as const;

type LetterboxInfo = {
  scale: number;
  padX: number;
  padY: number;
  srcW: number;
  srcH: number;
};

function sigmoid(x: number) {
  return 1 / (1 + Math.exp(-x));
}

// ---- NEW: sha256 cache helper ----
async function sha256OfFile(f: File) {
  const buf = await f.arrayBuffer();
  const hashBuf = await crypto.subtle.digest("SHA-256", buf);
  const hashArr = Array.from(new Uint8Array(hashBuf));
  return hashArr.map((b) => b.toString(16).padStart(2, "0")).join("");
}

// IoU for NMS
function iou(a: [number, number, number, number], b: [number, number, number, number]) {
  const x1 = Math.max(a[0], b[0]);
  const y1 = Math.max(a[1], b[1]);
  const x2 = Math.min(a[2], b[2]);
  const y2 = Math.min(a[3], b[3]);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const areaA = Math.max(0, a[2] - a[0]) * Math.max(0, a[3] - a[1]);
  const areaB = Math.max(0, b[2] - b[0]) * Math.max(0, b[3] - b[1]);
  const union = areaA + areaB - inter;
  return union <= 0 ? 0 : inter / union;
}

function nms(
  boxes: Array<{ box: [number, number, number, number]; score: number; cls: number }>,
  iouThresh: number
) {
  boxes.sort((a, b) => b.score - a.score);
  const keep: typeof boxes = [];

  for (const cand of boxes) {
    let ok = true;
    for (const kept of keep) {
      if (cand.cls !== kept.cls) continue; // class-wise NMS
      if (iou(cand.box, kept.box) > iouThresh) {
        ok = false;
        break;
      }
    }
    if (ok) keep.push(cand);
  }
  return keep;
}

async function fileToImageBitmap(file: File) {
  const blobURL = URL.createObjectURL(file);
  try {
    const img = await createImageBitmap(await fetch(blobURL).then((r) => r.blob()));
    return img;
  } finally {
    URL.revokeObjectURL(blobURL);
  }
}

/**
 * Letterbox resize to square input (e.g. 640x640) using YOLO-style padding.
 * Returns Float32 CHW tensor data normalized 0..1.
 */
function preprocessLetterbox(img: ImageBitmap, inputSize: number) {
  const srcW = img.width;
  const srcH = img.height;

  const scale = Math.min(inputSize / srcW, inputSize / srcH);
  const newW = Math.round(srcW * scale);
  const newH = Math.round(srcH * scale);

  const padX = Math.floor((inputSize - newW) / 2);
  const padY = Math.floor((inputSize - newH) / 2);

  const canvas = document.createElement("canvas");
  canvas.width = inputSize;
  canvas.height = inputSize;
  const ctx = canvas.getContext("2d")!;

  // YOLO typical padding value ~114
  ctx.fillStyle = "rgb(114,114,114)";
  ctx.fillRect(0, 0, inputSize, inputSize);

  ctx.drawImage(img, padX, padY, newW, newH);

  const { data } = ctx.getImageData(0, 0, inputSize, inputSize);

  // CHW float32
  const float = new Float32Array(1 * 3 * inputSize * inputSize);
  const area = inputSize * inputSize;

  for (let i = 0; i < area; i++) {
    const r = data[i * 4 + 0] / 255;
    const g = data[i * 4 + 1] / 255;
    const b = data[i * 4 + 2] / 255;
    float[i] = r; // R
    float[i + area] = g; // G
    float[i + 2 * area] = b; // B
  }

  const letterbox: LetterboxInfo = { scale, padX, padY, srcW, srcH };
  return { float, letterbox };
}

/**
 * Converts box coords from letterboxed input back to ORIGINAL image pixels.
 */
function undoLetterboxXYXY(
  box: [number, number, number, number],
  info: LetterboxInfo
): [number, number, number, number] {
  const [x1, y1, x2, y2] = box;

  const ox1 = (x1 - info.padX) / info.scale;
  const oy1 = (y1 - info.padY) / info.scale;
  const ox2 = (x2 - info.padX) / info.scale;
  const oy2 = (y2 - info.padY) / info.scale;

  // clamp
  const cx1 = Math.max(0, Math.min(info.srcW, ox1));
  const cy1 = Math.max(0, Math.min(info.srcH, oy1));
  const cx2 = Math.max(0, Math.min(info.srcW, ox2));
  const cy2 = Math.max(0, Math.min(info.srcH, oy2));

  return [cx1, cy1, cx2, cy2];
}

export class YoloBrowserDetector {
  private session: ort.InferenceSession | null = null;
  private inputName: string | null = null;
  private outputName: string | null = null;

  constructor(private modelUrl = "/models/electrical.onnx", private inputSize = 640) {}

  async init() {
    if (this.session) return;

    // 💥 THE VITE BYPASS 💥
    ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/";

    this.session = await ort.InferenceSession.create(this.modelUrl, {
      executionProviders: ["wasm"],
    });

    this.inputName = this.session.inputNames[0];
    this.outputName = this.session.outputNames[0];
  }

  /**
   * Detect directly from PNG File. Returns detections in ORIGINAL PNG pixel coords.
   * NEW: caches result per-file SHA256 + modelUrl + thresholds.
   */
  async detectFile(file: File, confThres = 0.25, iouThres = 0.45): Promise<Detection[]> {
    await this.init();
    if (!this.session || !this.inputName || !this.outputName) return [];

    // ---- NEW: cache lookup ----
    const h = await sha256OfFile(file);
    const cacheKey = `yoloBrowser:${this.modelUrl}:size=${this.inputSize}:${h}:conf=${confThres}:iou=${iouThres}`;
    const cached = localStorage.getItem(cacheKey);
    if (cached) {
      try {
        return JSON.parse(cached) as Detection[];
      } catch {
        // ignore parse errors
      }
    }

    const img = await fileToImageBitmap(file);
    const { float, letterbox } = preprocessLetterbox(img, this.inputSize);

    const inputTensor = new ort.Tensor("float32", float, [1, 3, this.inputSize, this.inputSize]);
    const outputs = await this.session.run({ [this.inputName]: inputTensor });
    const out = outputs[this.outputName];

    const data = out.data as Float32Array;
    const dims = out.dims;

    // ---- Embedded NMS output cases ----
    // [1, N, 6] OR [N, 6] -> x1,y1,x2,y2,score,cls
    if ((dims.length === 3 && dims[2] === 6) || (dims.length === 2 && dims[1] === 6)) {
      const N = dims.length === 3 ? dims[1] : dims[0];
      const dets: Detection[] = [];

      for (let i = 0; i < N; i++) {
        const x1 = data[i * 6 + 0];
        const y1 = data[i * 6 + 1];
        const x2 = data[i * 6 + 2];
        const y2 = data[i * 6 + 3];
        const score = data[i * 6 + 4];
        const cls = Math.round(data[i * 6 + 5]);

        if (score < confThres) continue;
        if (cls < 0 || cls >= CLASSES.length) continue;

        const orig = undoLetterboxXYXY([x1, y1, x2, y2], letterbox);
        dets.push({ cls: CLASSES[cls], conf: score, xyxy: orig });
      }

      // ---- NEW: cache store ----
      try {
        localStorage.setItem(cacheKey, JSON.stringify(dets));
      } catch {}

      return dets;
    }

    // ---- Raw YOLO output ----
    // dims could be [1, C, N] or [1, N, C]
    const b = dims[0];
    if (b !== 1) console.warn("Batch != 1, using first batch only");

    let C: number, N: number, layout: "1CN" | "1NC";
    if (dims.length !== 3) {
      console.warn("Unexpected output dims:", dims);

      // cache empty to avoid repeated failures (optional)
      try {
        localStorage.setItem(cacheKey, JSON.stringify([]));
      } catch {}

      return [];
    }

    if (dims[1] <= 64 && dims[2] > dims[1]) {
      C = dims[1];
      N = dims[2];
      layout = "1CN";
    } else {
      N = dims[1];
      C = dims[2];
      layout = "1NC";
    }

    const nc = CLASSES.length;
    const hasObj = C === 5 + nc;
    const expectedNoObj = C === 4 + nc;

    if (!hasObj && !expectedNoObj) {
      console.warn("Unexpected channel count:", C, "expected", 4 + nc, "or", 5 + nc);
      // still attempt: treat first 4 as xywh, rest as class scores
    }

    const candidates: Array<{ box: [number, number, number, number]; score: number; cls: number }> = [];

    function get(i: number, c: number) {
      if (layout === "1CN") return data[c * N + i];
      return data[i * C + c];
    }

    for (let i = 0; i < N; i++) {
      const cx = get(i, 0);
      const cy = get(i, 1);
      const w = get(i, 2);
      const h2 = get(i, 3);

      const x1 = cx - w / 2;
      const y1 = cy - h2 / 2;
      const x2 = cx + w / 2;
      const y2 = cy + h2 / 2;

      let obj = 1.0;
      let clsStart = 4;
      if (hasObj) {
        obj = get(i, 4);
        clsStart = 5;
      }

      let bestScore = -1;
      let bestCls = -1;

      for (let k = 0; k < nc; k++) {
        let p = get(i, clsStart + k);
        // If needed:
        // p = sigmoid(p);
        const score = obj * p;
        if (score > bestScore) {
          bestScore = score;
          bestCls = k;
        }
      }

      if (bestScore < confThres) continue;

      candidates.push({ box: [x1, y1, x2, y2], score: bestScore, cls: bestCls });
    }

    console.log(`Raw candidates before NMS (Conf > ${confThres}):`, candidates.length);
    if (candidates.length > 0) {
      console.log(
        "Top candidate:",
        candidates.reduce((prev, current) => (prev.score > current.score ? prev : current))
      );
    }

    const kept = nms(candidates, iouThres);

    const result: Detection[] = kept.map((k) => ({
      cls: CLASSES[k.cls],
      conf: k.score,
      xyxy: undoLetterboxXYXY(k.box, letterbox),
    }));

    // ---- NEW: cache store ----
    try {
      localStorage.setItem(cacheKey, JSON.stringify(result));
    } catch {}

    return result;
  }
}