// main.ts
import * as THREE from "three";
import * as WEBIFC from "web-ifc";
import * as OBC from "@thatopen/components";
import { YoloBrowserDetector } from "./yolo_browser";
import { renderPdfToPngFiles } from "./pdf_render";
import loadOpenCV from "opencv-js-wasm";
console.log("main.ts loaded ✅");

/**
 * What this app does:
 * 1) Loads an IFC into That Open Engine (IfcLoader -> Fragments -> Three.js scene)
 * 2) Reads BuildingStoreys from IFC using web-ifc
 * 3) Places each PNG (or PDF pages rendered to PNG) as a textured plane at storey elevation
 * 4) Provides a floor selector + opacity slider + calibration (2-point) + persistence
 * 5) Runs YOLO in-browser once per plan, then filters/visualizes without re-running YOLO
 * 6) Adds Tools panel: conf filter, class toggles, search, detections list, export/import, isolate storey clipping, screenshot
 */

type StoreyInfo = {
  expressID: number;
  name: string;
  elevation: number; // "best" elevation we will use
  placementElev?: number; // derived from ObjectPlacement (IFC units)
};

type Detection = {
  cls: "light_square" | "socket" | "switches" | "TV_socket";
  conf: number;
  xyxy: [number, number, number, number]; // pixels x1,y1,x2,y2 (top-left origin)
};

type PlanOverlay = {
  storey?: StoreyInfo;
  mesh: THREE.Mesh;
  material: THREE.MeshBasicMaterial;
  fileName: string;
  worldY: number;

  // hashes (for safe persistence)
  planHash: string;
  ifcHash: string;

  // Base/original placement (so reset can snap back instantly)
  basePos: THREE.Vector3;
  baseRotY: number;
  baseScale: number;

  // needed for pixel->local
  imgW: number;
  imgH: number;
  planeW: number;
  planeD: number;
  rotate90: boolean;

//   // symbols
//   detections: Detection[];
//   symbols: THREE.Group;
//   worldSymbols: THREE.Group;
// };
// symbols
  detections: Detection[];
  symbols: THREE.Group;
  worldSymbols: THREE.Group;
  originalPdfFile?: File; // <--- ADDED
};
type FilterState = {
  minConf: number;
  clsEnabled: Record<Detection["cls"], boolean>;
  search: string;
  isolateStorey: boolean;
  isolateBand: number; // world units (half-band)
};

// ==============================
// OpenCV + ECC Auto-Align Helpers
// ==============================

// Keep a module-scoped cv handle so ALL your existing helpers that call `cv.*` still work.
// let cv: any;

let cvPromise: Promise<any> | null = null;

async function ensureOpenCV(): Promise<any> {
  const w: any = window;

  // already ready as cv
  if (w.cv?.findTransformECC) return w.cv;

  // ready as Module (some builds)
  if (w.Module?.findTransformECC) {
    w.cv = w.Module;
    return w.cv;
  }

  // wait for runtime init
  return new Promise((resolve, reject) => {
    const t0 = performance.now();

    const done = () => {
      if (w.cv?.findTransformECC) return resolve(w.cv);
      if (w.Module?.findTransformECC) {
        w.cv = w.Module;
        return resolve(w.cv);
      }
      reject(new Error("OpenCV initialized but findTransformECC missing"));
    };

    // hook onRuntimeInitialized if present
    if (w.Module) {
      const prev = w.Module.onRuntimeInitialized;
      w.Module.onRuntimeInitialized = () => {
        try {
          prev?.();
        } catch {}
        done();
      };
    }

    const tick = () => {
      if (w.cv?.findTransformECC || w.Module?.findTransformECC) return done();
      if (performance.now() - t0 > 15000) return reject(new Error("OpenCV.js not ready"));
      requestAnimationFrame(tick);
    };

    tick();
  });
}

let CV: any = null;

async function getCV(): Promise<any> {
  if (CV) return CV;
  CV = await ensureOpenCV(); // your ensureOpenCV that resolves Module/cv
  // IMPORTANT: use window property, not bare identifier
  (window as any).cv = (window as any).cv || CV;
  return CV;
}

// 2.2 Utilities: simple 3×3 matrix ops (for chaining transforms)
type Mat3 = number[]; // row-major length 9

function mat3Mul(A: Mat3, B: Mat3): Mat3 {
  const C = new Array(9).fill(0);
  for (let r = 0; r < 3; r++) {
    for (let c = 0; c < 3; c++) {
      C[r * 3 + c] =
        A[r * 3 + 0] * B[0 * 3 + c] +
        A[r * 3 + 1] * B[1 * 3 + c] +
        A[r * 3 + 2] * B[2 * 3 + c];
    }
  }
  return C;
}

function mat3Inv(M: Mat3): Mat3 {
  const a = M[0], b = M[1], c = M[2];
  const d = M[3], e = M[4], f = M[5];
  const g = M[6], h = M[7], i = M[8];

  const A =  (e * i - f * h);
  const B = -(d * i - f * g);
  const C =  (d * h - e * g);
  const D = -(b * i - c * h);
  const E =  (a * i - c * g);
  const F = -(a * h - b * g);
  const G =  (b * f - c * e);
  const H = -(a * f - c * d);
  const I =  (a * e - b * d);

  const det = a * A + b * B + c * C;
  if (!isFinite(det) || Math.abs(det) < 1e-12) throw new Error("Singular mat3");

  const invDet = 1 / det;
  return [
    A * invDet, D * invDet, G * invDet,
    B * invDet, E * invDet, H * invDet,
    C * invDet, F * invDet, I * invDet,
  ];
}

function applyH(T: Mat3, x: number, y: number): { x: number; y: number } {
  const X = T[0] * x + T[1] * y + T[2];
  const Y = T[3] * x + T[4] * y + T[5];
  const W = T[6] * x + T[7] * y + T[8];
  if (!isFinite(W) || Math.abs(W) < 1e-12) return { x: NaN, y: NaN };
  return { x: X / W, y: Y / W };
}

function warp2x3ToMat3(w: any): Mat3 {
  // w is cv.Mat 2x3 float
  const d = w.data32F ?? w.data64F ?? w.data;
  return [
    d[0], d[1], d[2],
    d[3], d[4], d[5],
    0,    0,    1,
  ];
}


// ==============================
// Robust crop-to-content (edges)
// ==============================
function autoCropEdgesToSquare(
  cv: any,
  edgesU8: any,        // CV_8U 0/255, assumed square outSize×outSize
  A_src: Mat3,         // maps source coords -> current out coords
  outSize: number,
  padPx = 12
): { edges: any; A: Mat3; invA: Mat3 } {

  const W = edgesU8.cols;
  const H = edgesU8.rows;
  const S = Math.min(outSize, W, H); // safety

  // 1) binarize to 0/1
  const bin01 = new cv.Mat();
  cv.threshold(edgesU8, bin01, 1, 1, cv.THRESH_BINARY);

  // 2) connected components
  if (!cv.connectedComponentsWithStats) {
    bin01.delete();
    const A = A_src;
    return { edges: edgesU8, A, invA: mat3Inv(A) };
  }

  const labels = new cv.Mat();
  const stats = new cv.Mat();
  const cents = new cv.Mat();

  const n = cv.connectedComponentsWithStats(bin01, labels, stats, cents, 8, cv.CV_32S);

  // Copy stats NOW (don’t keep a view after stats.delete())
  const st = stats.data32S ? new Int32Array(stats.data32S) : null;

  // cleanup mats we no longer need
  bin01.delete();
  labels.delete();
  cents.delete();
  stats.delete();

  if (!st || n <= 1) {
    const A = A_src;
    return { edges: edgesU8, A, invA: mat3Inv(A) };
  }

  // 3) find largest component area
  let maxArea = 0;
  for (let i = 1; i < n; i++) {
    const area = st[i * 5 + 4];
    if (area > maxArea) maxArea = area;
  }

  // 4) keep meaningful components, reject long-thin baselines
  const minKeep = Math.max(120, Math.floor(maxArea * 0.02)); // tweakable
  let xMin = Infinity, yMin = Infinity, xMax = -Infinity, yMax = -Infinity;

  for (let i = 1; i < n; i++) {
    const x = st[i * 5 + 0];
    const y = st[i * 5 + 1];
    const w = st[i * 5 + 2];
    const h = st[i * 5 + 3];
    const area = st[i * 5 + 4];

    if (area < minKeep) continue;

    const ar = w > 0 && h > 0 ? Math.max(w / h, h / w) : 999;
    const longThin = ar > 25 && area < maxArea * 0.6;
    if (longThin) continue;

    xMin = Math.min(xMin, x);
    yMin = Math.min(yMin, y);
    xMax = Math.max(xMax, x + w);
    yMax = Math.max(yMax, y + h);
  }

  // If we rejected too much, fallback to full image
  if (!isFinite(xMin)) {
    xMin = 0; yMin = 0; xMax = W; yMax = H;
  }

  // 5) pad + clamp
  xMin = Math.max(0, Math.floor(xMin - padPx));
  yMin = Math.max(0, Math.floor(yMin - padPx));
  xMax = Math.min(W, Math.ceil(xMax + padPx));
  yMax = Math.min(H, Math.ceil(yMax + padPx));

  const wBox = Math.max(1, xMax - xMin);
  const hBox = Math.max(1, yMax - yMin);

  // 6) square crop around bbox center
  const span = Math.min(Math.max(wBox, hBox), S);
  const cx = xMin + wBox * 0.5;
  const cy = yMin + hBox * 0.5;

  let x0 = Math.round(cx - span * 0.5);
  let y0 = Math.round(cy - span * 0.5);

  x0 = Math.max(0, Math.min(S - span, x0));
  y0 = Math.max(0, Math.min(S - span, y0));

  // 7) warpAffine crop+scale to full outSize
  // dst->src: x_src = x0 + (span/outSize)*x_dst
  const k = span / outSize;
  const M = cv.matFromArray(2, 3, cv.CV_32F, [
    k, 0, x0,
    0, k, y0
  ]);

  const cropped = new cv.Mat();
  cv.warpAffine(
    edgesU8,
    cropped,
    M,
    new cv.Size(outSize, outSize),
    cv.INTER_LINEAR + cv.WARP_INVERSE_MAP,
    cv.BORDER_CONSTANT,
    new cv.Scalar(0)
  );
  M.delete();

  // 8) update transform: newOut = s*(oldOut - [x0,y0]), s=outSize/span
  const s = outSize / span;
  const C: Mat3 = [
    s, 0, -s * x0,
    0, s, -s * y0,
    0, 0, 1,
  ];

  const A = mat3Mul(C, A_src);
  const invA = mat3Inv(A);

  return { edges: cropped, A, invA };
}
// 2.3 Plan edges raster (stand-in for PDF extraction stage)
function removeSmallComponents(cv: any, binU8: any, minArea = 60): any {
  // binU8: CV_8U (0..255)
  const bin01 = new cv.Mat();
  cv.threshold(binU8, bin01, 1, 1, cv.THRESH_BINARY);

  const labels = new cv.Mat();
  const stats = new cv.Mat();
  const cents = new cv.Mat();

  if (!cv.connectedComponentsWithStats) {
    // fallback: simple morphological opening
    labels.delete(); stats.delete(); cents.delete();
    const k = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(3, 3));
    const out = new cv.Mat();
    cv.morphologyEx(binU8, out, cv.MORPH_OPEN, k);
    k.delete(); bin01.delete();
    return out;
  }

  const n = cv.connectedComponentsWithStats(bin01, labels, stats, cents, 8, cv.CV_32S);

  const out = new cv.Mat.zeros(binU8.rows, binU8.cols, cv.CV_8U);

  const st = stats.data32S;      // n x 5: [x,y,w,h,area]
  const lab = labels.data32S;    // rows*cols
  const outData = out.data;      // Uint8Array

  // mark which component ids to keep
  const keep = new Uint8Array(n);
  for (let i = 1; i < n; i++) {
    const area = st[i * 5 + 4];
    if (area >= minArea) keep[i] = 1;
  }

  // write output mask
  for (let k = 0; k < lab.length; k++) {
    const id = lab[k];
    if (keep[id]) outData[k] = 255;
  }

  bin01.delete(); labels.delete(); stats.delete(); cents.delete();
  return out;
}

async function rasterizePlanEdges(
  cv: any,
  overlay: PlanOverlay,
  outSize = 1024,
  margin = 10
): Promise<{ edges: any; A_plan: Mat3; invA_plan: Mat3; outCanvas: HTMLCanvasElement }> {
  const imgEl: any = overlay.material.map?.image;
  if (!imgEl) throw new Error("Plan texture has no image");

  // draw plan image into square canvas with padding (records A_plan)
  const srcW = Number(imgEl.width ?? overlay.imgW ?? 1);
  const srcH = Number(imgEl.height ?? overlay.imgH ?? 1);

  

  const scale = (outSize - 2 * margin) / Math.max(srcW, srcH);
  const drawW = srcW * scale;
  const drawH = srcH * scale;
  const offX = margin + (outSize - 2 * margin - drawW) * 0.5;
  const offY = margin + (outSize - 2 * margin - drawH) * 0.5;

  const c = document.createElement("canvas");
  c.width = outSize;
  c.height = outSize;
  const ctx = c.getContext("2d")!;
  ctx.fillStyle = "black";
  ctx.fillRect(0, 0, outSize, outSize);
  ctx.imageSmoothingEnabled = true;
  ctx.drawImage(imgEl, offX, offY, drawW, drawH);

  const A_plan: Mat3 = [
    scale, 0,     offX,
    0,     scale, offY,
    0,     0,     1,
  ];
  const invA_plan = mat3Inv(A_plan);

  // OpenCV edges
  const rgba = cv.imread(c);
  const gray = new cv.Mat();
  cv.cvtColor(rgba, gray, cv.COLOR_RGBA2GRAY);

  const edges = new cv.Mat();
  cv.Canny(gray, edges, 50, 150);

  // remove small CCs (specks / text noise)
  const cleaned = removeSmallComponents(cv, edges, 60);

  // dilate helps ECC
  const k = cv.Mat.ones(3, 3, cv.CV_8U);
  const dil = new cv.Mat();
  cv.dilate(cleaned, dil, k, new cv.Point(-1, -1), 1);

rgba.delete(); gray.delete(); edges.delete(); cleaned.delete(); k.delete();

const cropped = autoCropEdgesToSquare(cv, dil, A_plan, outSize, 12);

// only delete dil if crop returned a different Mat
if (cropped.edges !== dil) dil.delete();

return { edges: cropped.edges, A_plan: cropped.A, invA_plan: cropped.invA, outCanvas: c };
}
function resizeU8ToFixed(cv: any, movingU8: any, fixedU8: any) {
  // ECC requires same rows/cols. If already same, do nothing.
  if (movingU8.rows === fixedU8.rows && movingU8.cols === fixedU8.cols) {
    return { mat: movingU8, owned: false };
  }

  const resized = new cv.Mat();
  cv.resize(
    movingU8,
    resized,
    new cv.Size(fixedU8.cols, fixedU8.rows),
    0,
    0,
    cv.INTER_LINEAR
  );

  return { mat: resized, owned: true };
}
function eccAlign(cv: any, movingU8: any, fixedU8: any, motion: number, nIter = 2000) {
  let moving = movingU8;
  let ownedMoving = false;

  if (movingU8.rows !== fixedU8.rows || movingU8.cols !== fixedU8.cols) {
    const resized = new cv.Mat();
    cv.resize(movingU8, resized, new cv.Size(fixedU8.cols, fixedU8.rows), 0, 0, cv.INTER_LINEAR);
    moving = resized;
    ownedMoving = true;
  }

const mask01 = new cv.Mat();
cv.threshold(fixedU8, mask01, 1, 255, cv.THRESH_BINARY);

const nz = cv.countNonZero(mask01);
if (nz < 200) {
  // Not enough structure for ECC to lock onto.
  // Return identity warp + a SAME-SIZE aligned image so pipeline doesn't crash.
  const warp = cv.Mat.eye(2, 3, cv.CV_32F);

    // ✅ init warp translation from centroids (helps ECC converge)
  try {
    const movBin = new cv.Mat();
    cv.threshold(moving, movBin, 1, 255, cv.THRESH_BINARY);

    const mFix = cv.moments(mask01, true);
    const mMov = cv.moments(movBin, true);

    if (mFix.m00 && mMov.m00) {
      const cxFix = mFix.m10 / mFix.m00;
      const cyFix = mFix.m01 / mFix.m00;
      const cxMov = mMov.m10 / mMov.m00;
      const cyMov = mMov.m01 / mMov.m00;

      const wd = warp.data32F;
      wd[2] = cxFix - cxMov;
      wd[5] = cyFix - cyMov;
    }

    movBin.delete();
  } catch {}

  const aligned = new cv.Mat();
  if (moving.rows === fixedU8.rows && moving.cols === fixedU8.cols) {
    moving.copyTo(aligned);
  } else {
    cv.resize(moving, aligned, new cv.Size(fixedU8.cols, fixedU8.rows), 0, 0, cv.INTER_LINEAR);
  }

  mask01.delete();
  if (ownedMoving) moving.delete();
  return { cc: -1, warp, aligned };
}

  const kMask = cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(9, 9));
  cv.dilate(mask01, mask01, kMask, new cv.Point(-1, -1), 2);
  kMask.delete();

  const movingF = new cv.Mat();
  const fixedF = new cv.Mat();
  moving.convertTo(movingF, cv.CV_32F, 1 / 255);
  fixedU8.convertTo(fixedF, cv.CV_32F, 1 / 255);

  cv.normalize(movingF, movingF, 0, 1, cv.NORM_MINMAX);
  cv.normalize(fixedF, fixedF, 0, 1, cv.NORM_MINMAX);

  cv.GaussianBlur(movingF, movingF, new cv.Size(5, 5), 0);
  cv.GaussianBlur(fixedF, fixedF, new cv.Size(5, 5), 0);

  const warp = cv.Mat.eye(2, 3, cv.CV_32F);

  const criteria = new cv.TermCriteria(
    cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT,
    nIter,
    1e-6
  );

  let cc: any;
  try {
    cc = cv.findTransformECC(fixedF, movingF, warp, motion, criteria, mask01, 5);
  } finally {
    movingF.delete();
    fixedF.delete();
    mask01.delete();
    // ❌ do NOT delete `moving` here
  }

  const aligned = new cv.Mat();
  const size = new cv.Size(fixedU8.cols, fixedU8.rows);

  cv.warpAffine(
    moving,
    aligned,
    warp,
    size,
    cv.INTER_LINEAR + cv.WARP_INVERSE_MAP,
    cv.BORDER_CONSTANT,
    new cv.Scalar(0)
  );

  if (ownedMoving) moving.delete(); // ✅ delete only after warpAffine
  return { cc, warp, aligned };
}
function cvErrorToString(cv: any, err: any) {
  try {
    // OpenCV.js often throws a number = pointer to cv::Exception
    if (typeof err === "number" && cv?.exceptionFromPtr) {
      const ex = cv.exceptionFromPtr(err);
      const msg =
        ex?.msg
          ? `OpenCV: ${ex.msg} (code=${ex.code}) in ${ex.func} @ ${ex.file}:${ex.line}`
          : `OpenCV exception ptr=${err}`;
      try { ex?.delete?.(); } catch {}
      return msg;
    }
  } catch {
    // fall through
  }
  try {
    if (err?.message) return String(err.message);
  } catch {}
  return String(err);
}
type EccResult = { cc: any; warp: any; aligned: any };

function eccAlignSafe(
  cv: any,
  movingU8: any,
  fixedU8: any,
  motion: number,
  nIter: number,
  stageName: string
): EccResult | null {
  try {
    return eccAlign(cv, movingU8, fixedU8, motion, nIter) as EccResult;
  } catch (e) {
    console.warn(`[ECC ${stageName}]`, cvErrorToString(cv, e), e);
    return null;
  }
}



// 2.5 ECC alignment + trimmed chamfer score
// function eccAlign(cv: any, movingU8: any, fixedU8: any, motion: number, nIter = 2000) {
//   // --- 0) Ensure same size (ECC requirement) ---
//   let moving = movingU8;
//   let ownedMoving = false;

//   if (movingU8.rows !== fixedU8.rows || movingU8.cols !== fixedU8.cols) {
//     const resized = new cv.Mat();
//     cv.resize(movingU8, resized, new cv.Size(fixedU8.cols, fixedU8.rows), 0, 0, cv.INTER_LINEAR);
//     moving = resized;
//     ownedMoving = true;
//   }

//   // --- 1) Build a GOOD mask from fixed edges (not full-image) ---
//   // mask = dilated(fixedU8>0) * 255
//   const mask01 = new cv.Mat();
//   cv.threshold(fixedU8, mask01, 1, 255, cv.THRESH_BINARY);

//   const kMask = cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(9, 9));
//   cv.dilate(mask01, mask01, kMask, new cv.Point(-1, -1), 2);
//   kMask.delete();

//   // --- 2) Convert to float, normalize, blur (ECC likes smooth gradients) ---
//   const movingF = new cv.Mat();
//   const fixedF = new cv.Mat();
//   moving.convertTo(movingF, cv.CV_32F, 1 / 255);
//   fixedU8.convertTo(fixedF, cv.CV_32F, 1 / 255);

//   cv.normalize(movingF, movingF, 0, 1, cv.NORM_MINMAX);
//   cv.normalize(fixedF, fixedF, 0, 1, cv.NORM_MINMAX);

//   cv.GaussianBlur(movingF, movingF, new cv.Size(5, 5), 0);
//   cv.GaussianBlur(fixedF, fixedF, new cv.Size(5, 5), 0);

//   // --- 3) Init warp with centroid translation (huge convergence win) ---
//   // compute centroids on binary edges
//   const mFix = cv.moments(mask01, true); // fixed mask already binary-ish
//   const movBin = new cv.Mat();
//   cv.threshold(moving, movBin, 1, 255, cv.THRESH_BINARY);
//   const mMov = cv.moments(movBin, true);

//   const cxFix = mFix.m00 ? mFix.m10 / mFix.m00 : fixedU8.cols * 0.5;
//   const cyFix = mFix.m00 ? mFix.m01 / mFix.m00 : fixedU8.rows * 0.5;
//   const cxMov = mMov.m00 ? mMov.m10 / mMov.m00 : fixedU8.cols * 0.5;
//   const cyMov = mMov.m00 ? mMov.m01 / mMov.m00 : fixedU8.rows * 0.5;

//   movBin.delete();

//   // dx,dy to bring moving toward fixed
//   const dx = cxFix - cxMov;
//   const dy = cyFix - cyMov;

//   const warp = cv.Mat.eye(2, 3, cv.CV_32F);
//   const wd = warp.data32F;
//   wd[2] = dx;
//   wd[5] = dy;

//   // --- ECC ---
//   const criteria = new cv.TermCriteria(
//     cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT,
//     nIter,
//     1e-6
//   );

//   let cc: any;
//   try {
//     cc = cv.findTransformECC(fixedF, movingF, warp, motion, criteria, mask01, 5);
//   } finally {
//     // clean intermediates even if ECC throws
//     movingF.delete();
//     fixedF.delete();
//     mask01.delete();
//     if (ownedMoving) moving.delete();
//   }

//   // aligned movingU8 in fixed size
//   const aligned = new cv.Mat();
//   const size = new cv.Size(fixedU8.cols, fixedU8.rows);
//   cv.warpAffine(
//     moving,
//     aligned,
//     warp,
//     size,
//     cv.INTER_LINEAR + cv.WARP_INVERSE_MAP,
//     cv.BORDER_CONSTANT,
//     new cv.Scalar(0)
//   );

//   return { cc, warp, aligned };
// }

function ensureSameSizeU8(cv: any, fixedU8: any, movingU8: any) {
  if (fixedU8.rows === movingU8.rows && fixedU8.cols === movingU8.cols) {
    return { fixed: { mat: fixedU8, owned: false }, moving: { mat: movingU8, owned: false } };
  }

  const resized = new cv.Mat();
  cv.resize(movingU8, resized, new cv.Size(fixedU8.cols, fixedU8.rows), 0, 0, cv.INTER_LINEAR);

  return {
    fixed: { mat: fixedU8, owned: false },
    moving: { mat: resized, owned: true },
  };
}

function trimmedChamfer(cv: any, ifcWarpedU8: any, planU8: any, trimQ = 90) {
    // Distance transform on plan edges (fixed)
  const planBin = new cv.Mat();
  cv.threshold(planU8, planBin, 1, 1, cv.THRESH_BINARY);

  const ones = cv.Mat.ones(planBin.rows, planBin.cols, cv.CV_8U);
  const inv = new cv.Mat();
  cv.subtract(ones, planBin, inv);
  ones.delete();

  const dist = new cv.Mat();
  cv.distanceTransform(inv, dist, cv.DIST_L2, 3);

  const ifcBin = new cv.Mat();
  cv.threshold(ifcWarpedU8, ifcBin, 1, 1, cv.THRESH_BINARY);

  // collect distances at IFC edge pixels
  const d: number[] = [];
  for (let y = 0; y < ifcBin.rows; y++) {
    for (let x = 0; x < ifcBin.cols; x++) {
      if (ifcBin.ucharPtr(y, x)[0] > 0) {
        d.push(dist.floatPtr(y, x)[0]);
      }
    }
  }

  planBin.delete(); inv.delete(); dist.delete(); ifcBin.delete();

  if (!d.length) return { mean: Infinity, median: Infinity, p90: Infinity };

  d.sort((a, b) => a - b);
  const cutIdx = Math.floor((trimQ / 100) * (d.length - 1));
  const cutoff = d[cutIdx];
  const kept = d.filter((v) => v <= cutoff);

  const mean = kept.reduce((s, v) => s + v, 0) / kept.length;
  const median = kept[Math.floor(kept.length / 2)];
  const p90 = kept[Math.floor(0.9 * (kept.length - 1))];

  return { mean, median, p90, cutoff, n: d.length, nUsed: kept.length, trimQ };
}

async function boot() {
  
  const container = document.getElementById("container") as HTMLDivElement;
  const calibrateBtn = document.getElementById("calibrate") as HTMLButtonElement;
  const autoAlignBtn = document.getElementById("autoAlign") as HTMLButtonElement;
  const resetPlanBtn = document.getElementById("resetPlan") as HTMLButtonElement;
  const statusEl = document.getElementById("status") as HTMLSpanElement;

  const fileInput = document.getElementById("fileInput") as HTMLInputElement;
  const floorSelect = document.getElementById("floorSelect") as HTMLSelectElement;
  const opacitySlider = document.getElementById("opacity") as HTMLInputElement;
  const focusPlanBtn = document.getElementById("focusPlan") as HTMLButtonElement;

  function setStatus(msg: string) {
    if (statusEl) statusEl.textContent = msg;
    console.log("[STATUS]", msg);
  }

  const detector = new YoloBrowserDetector(import.meta.env.BASE_URL + "models/electrical.onnx", 640);

  // ------------------------
  // Panels (Props + Tools)
  // ------------------------
  function ensurePropsPanel() {
    let el = document.getElementById("propsPanel") as HTMLDivElement | null;
    if (el) return el;

    el = document.createElement("div");
    el.id = "propsPanel";
    el.style.position = "absolute";
    el.style.right = "12px";
    el.style.top = "12px";
    el.style.width = "280px";
    el.style.maxHeight = "60vh";
    el.style.overflow = "auto";
    el.style.background = "rgba(0,0,0,0.75)";
    el.style.color = "white";
    el.style.padding = "10px";
    el.style.border = "1px solid rgba(255,255,255,0.15)";
    el.style.borderRadius = "8px";
    el.style.fontFamily = "system-ui, sans-serif";
    el.style.fontSize = "12px";
    el.style.zIndex = "9999";
    el.style.display = "none";

    container.style.position = "relative";
    container.appendChild(el);

    return el;
  }

  function showSymbolProps(obj: THREE.Object3D) {
    const el = ensurePropsPanel();
    const ud: any = obj.userData;
    if (!ud || ud.kind !== "yoloSymbol") return;

    const det: Detection = ud.det;
    const [x1, y1, x2, y2] = det.xyxy;

    const wp = new THREE.Vector3();
    obj.getWorldPosition(wp);

    el.innerHTML = `
      <div style="display:flex;justify-content:space-between;align-items:center;">
        <b>Detection</b>
        <button id="closeProps" style="background:#222;color:#fff;border:1px solid #444;border-radius:6px;padding:2px 6px;cursor:pointer;">x</button>
      </div>
      <hr style="border:0;border-top:1px solid rgba(255,255,255,0.15);margin:8px 0;" />
      <div><b>Class:</b> ${det.cls}</div>
      <div><b>Confidence:</b> ${det.conf.toFixed(3)}</div>
      <div><b>Plan:</b> ${ud.fileName}</div>
      <div><b>Storey:</b> ${ud.storeyName || "-"}</div>
      <div><b>BBox(px):</b> [${x1.toFixed(1)}, ${y1.toFixed(1)}, ${x2.toFixed(1)}, ${y2.toFixed(1)}]</div>
      <div><b>World pos:</b> (${wp.x.toFixed(2)}, ${wp.y.toFixed(2)}, ${wp.z.toFixed(2)})</div>
      <div><b>Det #:</b> ${ud.detIndex ?? "-"}</div>
    `;
    el.style.display = "block";

    const closeBtn = el.querySelector("#closeProps") as HTMLButtonElement | null;
    if (closeBtn) closeBtn.onclick = () => (el!.style.display = "none");
  }

  // ------------------------
  // That Open Engine setup
  // ------------------------
  const components = new OBC.Components();
  const worlds = components.get(OBC.Worlds);
  const world =
    worlds.create<OBC.SimpleScene, OBC.OrthoPerspectiveCamera, OBC.SimpleRenderer>();

  world.scene = new OBC.SimpleScene(components);
  world.scene.setup();
  world.scene.three.background = new THREE.Color(0x111111);

  world.renderer = new OBC.SimpleRenderer(components, container);
  world.camera = new OBC.OrthoPerspectiveCamera(components);

  // Enable clipping
  (world.renderer!.three as any).localClippingEnabled = true;

  // Initial view (we refocus after loading)
  await world.camera.controls.setLookAt(20, 20, 20, 0, 0, 0);

  components.init();
// Keep a reference so we can hide it during offscreen IFC raster
const grid = components.get(OBC.Grids).create(world) as any;

function getGridObj(g: any): any {
  return g?.three ?? g; // some versions expose .three
}
function getGridVisible(g: any): boolean {
  const obj = getGridObj(g);
  return !!obj?.visible;
}
function setGridVisible(g: any, vis: boolean) {
  const obj = getGridObj(g);
  if (obj && "visible" in obj) obj.visible = vis;
}
  // Fragments worker
  const fragments = components.get(OBC.FragmentsManager);
  fragments.init("/worker.mjs");

  // Track materials so we can clip them reliably
  const allModelMaterials = new Set<any>();

  // Keep fragments LOD updated with camera
  world.camera.controls.addEventListener("update", () => fragments.core.update());

  // Add loaded models to the scene
  fragments.list.onItemSet.add(({ value: model }) => {
    model.useCamera(world.camera.three);
    world.scene.three.add(model.object);
    fragments.core.update(true);
    applyStoreyClipping();
  });

  // Reduce z-fighting + collect materials
  fragments.core.models.materials.list.onItemSet.add(({ value: material }) => {
    allModelMaterials.add(material);

    if (!("isLodMaterial" in material && (material as any).isLodMaterial)) {
      (material as any).polygonOffset = true;
      (material as any).polygonOffsetUnits = 1;
      (material as any).polygonOffsetFactor = Math.random();
    }

    applyStoreyClipping();
  });

  // web-ifc loader
  const ifcLoader = components.get(OBC.IfcLoader);
  await ifcLoader.setup({
  autoSetWasm: false,
 wasm: { path: import.meta.env.BASE_URL + "wasm/", absolute: true },
  webIfc: {
    COORDINATE_TO_ORIGIN: true,
  },
});

  // ------------------------
  // Interaction helpers
  // ------------------------
  const raycaster = new THREE.Raycaster();
  const mouse = new THREE.Vector2();
  // ---- NEW: safe model picking cache (avoids broken fragment meshes) ----
let pickableModelMeshes: THREE.Mesh[] = [];

function collectPickableMeshes(root: THREE.Object3D): THREE.Mesh[] {
  const out: THREE.Mesh[] = [];

  root.traverse((obj) => {
    const m: any = obj;
    if (!m?.isMesh) return;

    const g: any = m.geometry;
    if (!g || !g.isBufferGeometry) return;

    const pos: any = g.getAttribute?.("position") ?? g.attributes?.position;
    const arr: any = pos?.array;

    // Skip broken / zero-length geometries that can crash acceleratedRaycast
    if (!pos || !arr || typeof arr.length !== "number" || arr.length < 9) return;

    out.push(m as THREE.Mesh);
  });

  return out;
}
  type CalibStage =
  | "idle"
  | "pickModel1"
  | "pickModel2"
  | "pickModel3"
  | "pickPlan1"
  | "pickPlan2"
  | "pickPlan3";
  let calibStage: CalibStage = "idle";

  let calibModelPts: THREE.Vector3[] = [];
  let calibPlanLocalPts: THREE.Vector3[] = [];

  let calibOverlayIdx: number | null = null;

  type CalibBackup = {
    overlayIdx: number;
    overlayPos: THREE.Vector3;
    overlayRotY: number;
    overlayScale: number;
    overlaysVisible: boolean[];
    overlaysOpacity: number[];
    modelVisible: boolean;
    floorSelectDisabled: boolean;
    fileInputDisabled: boolean;
  };
  let calibBackup: CalibBackup | null = null;

  // ------------------------
  // Overlay state
  // ------------------------
  // let currentModel: OBC.FragmentsModel | null = null;
// let currentModel: OBC.FragmentsModel | null = null;
  type LoadedModel = { object: THREE.Object3D; useCamera?: (c: any) => void };
let currentModel: LoadedModel | null = null;
let currentIfcFile: File | null = null; // <--- ADDED
  let overlays: PlanOverlay[] = [];
  let lastBBox: THREE.Box3 | null = null;
  let currentIfcName = "";
  let currentIfcHash = "";

  // ------------------------
  // Tools / Filters state
  // ------------------------
  const filter: FilterState = {
    minConf: 0.25,
    clsEnabled: { light_square: true, socket: true, switches: true, TV_socket: true },
    search: "",
    isolateStorey: false,
    isolateBand: 1.5,
  };

  function passesFilter(det: Detection, minConf = filter.minConf) {
    if (det.conf < minConf) return false;
    if (!filter.clsEnabled[det.cls]) return false;
    if (!filter.search.trim()) return true;
    const q = filter.search.trim().toLowerCase();
    return det.cls.toLowerCase().includes(q);
  }

  let toolsPanel: HTMLDivElement | null = null;
  let detListEl: HTMLDivElement | null = null;

  // ---- NEW: tools collapse state ----
  let toolsShowBtn: HTMLButtonElement | null = null;
  let toolsCollapsed = localStorage.getItem("ui:toolsCollapsed") === "1";

  function setToolsCollapsed(v: boolean) {
    toolsCollapsed = v;
    localStorage.setItem("ui:toolsCollapsed", v ? "1" : "0");

    if (toolsPanel) toolsPanel.style.display = v ? "none" : "block";

    // Create floating "Tools" button when hidden
    if (!toolsShowBtn) {
      toolsShowBtn = document.createElement("button");
      toolsShowBtn.textContent = "Tools";
      toolsShowBtn.style.position = "absolute";
      toolsShowBtn.style.left = "12px";
      toolsShowBtn.style.top = "56px";
      toolsShowBtn.style.zIndex = "10001";
      toolsShowBtn.style.background = "rgba(0,0,0,0.75)";
      toolsShowBtn.style.color = "#fff";
      toolsShowBtn.style.border = "1px solid rgba(255,255,255,0.2)";
      toolsShowBtn.style.borderRadius = "8px";
      toolsShowBtn.style.padding = "6px 10px";
      toolsShowBtn.style.cursor = "pointer";
      toolsShowBtn.onclick = () => setToolsCollapsed(false);

      container.style.position = "relative";
      container.appendChild(toolsShowBtn);
    }

    toolsShowBtn.style.display = v ? "block" : "none";
  }

  function ensureToolsPanel() {
    if (toolsPanel) return toolsPanel;

    toolsPanel = document.createElement("div");
    toolsPanel.style.position = "absolute";
    toolsPanel.style.left = "12px";
    toolsPanel.style.top = "56px";
    toolsPanel.style.width = "320px";
    toolsPanel.style.maxHeight = "72vh";
    toolsPanel.style.overflow = "auto";
    toolsPanel.style.background = "rgba(0,0,0,0.75)";
    toolsPanel.style.color = "#fff";
    toolsPanel.style.padding = "10px";
    toolsPanel.style.border = "1px solid rgba(255,255,255,0.15)";
    toolsPanel.style.borderRadius = "8px";
    toolsPanel.style.fontFamily = "system-ui, sans-serif";
    toolsPanel.style.fontSize = "12px";
    toolsPanel.style.zIndex = "9999";

    container.style.position = "relative";
    container.appendChild(toolsPanel);

    toolsPanel.innerHTML = `
      <div style="display:flex;justify-content:space-between;align-items:center;gap:8px;">
        <b>Tools</b>
        <div style="display:flex;gap:8px;align-items:center;">
          <button id="hideToolsBtn" style="background:#222;color:#fff;border:1px solid #444;border-radius:6px;padding:4px 8px;cursor:pointer;">Hide</button>
          <button id="shotBtn" style="background:#222;color:#fff;border:1px solid #444;border-radius:6px;padding:4px 8px;cursor:pointer;">Screenshot</button>
        </div>
      </div>

      <hr style="border:0;border-top:1px solid rgba(255,255,255,0.15);margin:8px 0;" />

      <div><b>Confidence</b> <span id="confVal"></span></div>
      <input id="confSlider" type="range" min="0" max="1" step="0.01" value="${filter.minConf}" style="width:100%;" />

      <div style="margin-top:8px;"><b>Classes</b></div>
      <label style="display:block;"><input type="checkbox" id="cls_light" checked /> light_square</label>
      <label style="display:block;"><input type="checkbox" id="cls_socket" checked /> socket</label>
      <label style="display:block;"><input type="checkbox" id="cls_switches" checked /> switches</label>
      <label style="display:block;"><input type="checkbox" id="cls_tv" checked /> TV_socket</label>

      <div style="margin-top:8px;"><b>Search</b></div>
      <input id="searchBox" placeholder="e.g. socket" style="width:100%;padding:6px;border-radius:6px;border:1px solid #444;background:#111;color:#fff;" />

      <div style="margin-top:10px; display:flex; gap:8px;">
        <button id="exportJsonBtn" style="flex:1;background:#222;color:#fff;border:1px solid #444;border-radius:6px;padding:6px;cursor:pointer;">Export JSON</button>
        <button id="exportCsvBtn" style="flex:1;background:#222;color:#fff;border:1px solid #444;border-radius:6px;padding:6px;cursor:pointer;">Export CSV</button>
      </div>

      <div style="margin-top:8px;">
        <label style="display:block;"><b>Import detections</b></label>
        <input id="importJson" type="file" accept=".json" />
      </div>

      <hr style="border:0;border-top:1px solid rgba(255,255,255,0.15);margin:10px 0;" />

      <label style="display:block;">
        <input type="checkbox" id="isoToggle" /> Isolate storey (clip)
      </label>
      <div style="margin-top:6px;"><b>Band</b> <span id="bandVal"></span></div>
      <input id="bandSlider" type="range" min="0.2" max="10" step="0.1" value="${filter.isolateBand}" style="width:100%;" />

      <hr style="border:0;border-top:1px solid rgba(255,255,255,0.15);margin:10px 0;" />

      <div><b>Detections (current floor)</b></div>
      <div id="detList" style="margin-top:6px;display:flex;flex-direction:column;gap:6px;"></div>
    `;

    detListEl = toolsPanel.querySelector("#detList") as HTMLDivElement;

    const confSlider = toolsPanel.querySelector("#confSlider") as HTMLInputElement;
    const confVal = toolsPanel.querySelector("#confVal") as HTMLSpanElement;

    const bandSlider = toolsPanel.querySelector("#bandSlider") as HTMLInputElement;
    const bandVal = toolsPanel.querySelector("#bandVal") as HTMLSpanElement;

    const setConfText = () => (confVal.textContent = `(${filter.minConf.toFixed(2)})`);
    const setBandText = () => (bandVal.textContent = `(${filter.isolateBand.toFixed(1)})`);
    setConfText();
    setBandText();

    confSlider.oninput = () => {
      filter.minConf = parseFloat(confSlider.value);
      setConfText();
      refreshCurrentFloor(true);
    };

    const bindCls = (id: string, cls: Detection["cls"]) => {
      const el = toolsPanel!.querySelector(id) as HTMLInputElement;
      el.onchange = () => {
        filter.clsEnabled[cls] = el.checked;
        refreshCurrentFloor(true);
      };
    };
    bindCls("#cls_light", "light_square");
    bindCls("#cls_socket", "socket");
    bindCls("#cls_switches", "switches");
    bindCls("#cls_tv", "TV_socket");

    const searchBox = toolsPanel.querySelector("#searchBox") as HTMLInputElement;
    searchBox.oninput = () => {
      filter.search = searchBox.value;
      refreshCurrentFloor(true);
    };

    const exportJsonBtn = toolsPanel.querySelector("#exportJsonBtn") as HTMLButtonElement;
    exportJsonBtn.onclick = () => exportDetections("json");

    const exportCsvBtn = toolsPanel.querySelector("#exportCsvBtn") as HTMLButtonElement;
    exportCsvBtn.onclick = () => exportDetections("csv");

    const importJson = toolsPanel.querySelector("#importJson") as HTMLInputElement;
    importJson.onchange = async () => {
      const f = importJson.files?.[0];
      if (!f) return;
      await importDetectionsFile(f);
      importJson.value = "";
    };

    const isoToggle = toolsPanel.querySelector("#isoToggle") as HTMLInputElement;
    isoToggle.onchange = () => {
      filter.isolateStorey = isoToggle.checked;
      applyStoreyClipping();
    };

    bandSlider.oninput = () => {
      filter.isolateBand = parseFloat(bandSlider.value);
      setBandText();
      applyStoreyClipping();
    };

    const shotBtn = toolsPanel.querySelector("#shotBtn") as HTMLButtonElement;
    shotBtn.onclick = () => downloadScreenshot();

    // ---- NEW: hide button ----
    const hideToolsBtn = toolsPanel.querySelector("#hideToolsBtn") as HTMLButtonElement;
    hideToolsBtn.onclick = () => setToolsCollapsed(true);

    // Apply initial collapsed state
    setToolsCollapsed(toolsCollapsed);
    
    return toolsPanel;
  }
  function focusCameraToModel(pad = 2.5) {
  if (!currentModel) return;

  // While debugging: avoid clipping hiding everything
  filter.isolateStorey = false;
  applyStoreyClipping();

  const obj = currentModel.object;
  obj.updateWorldMatrix(true, true);

  const bb = new THREE.Box3().setFromObject(obj);
  const center = new THREE.Vector3();
  const size = new THREE.Vector3();
  bb.getCenter(center);
  bb.getSize(size);

  const span = Math.max(size.x, size.y, size.z);
  const dist = Math.max(10, span * pad);

  const cam: any = world.camera.three;

  // Fix clipping so you don’t get “black screen”
  cam.near = Math.max(0.01, dist / 1000);
  cam.far = dist * 2000;

  // VERY IMPORTANT for OrthoPerspectiveCamera: reset zoom if ortho
  if (cam.isOrthographicCamera) {
    cam.zoom = 1;
  }
  cam.updateProjectionMatrix?.();

  // Move camera to a diagonal vantage point
  world.camera.controls.setLookAt(
    center.x + dist,
    center.y + dist,
    center.z + dist,
    center.x,
    center.y,
    center.z,
    true
  );
}
  // ------------------------
  // Clipping (storey isolate)
  // ------------------------
  const clipPlanes = [
    new THREE.Plane(new THREE.Vector3(0, 1, 0), 0),
    new THREE.Plane(new THREE.Vector3(0, -1, 0), 0),
  ];

  function applyClippingToAllModelMaterials() {
    allModelMaterials.forEach((mat: any) => {
      mat.clippingPlanes = filter.isolateStorey ? clipPlanes : null;
      mat.clipIntersection = true;
      mat.needsUpdate = true;
    });
  }

  function applyStoreyClipping() {
    if (!filter.isolateStorey) {
      applyClippingToAllModelMaterials();
      return;
    }

    const idx = parseInt(floorSelect.value || "0", 10) || 0;
    const o = overlays[idx];
    if (!o) return;

    const y = o.worldY;
    const band = filter.isolateBand;

    const minY = y - band;
    const maxY = y + band;

    clipPlanes[0].set(new THREE.Vector3(0, 1, 0), -minY); // keep y >= minY
    clipPlanes[1].set(new THREE.Vector3(0, -1, 0), maxY); // keep y <= maxY

    applyClippingToAllModelMaterials();
  }

  // ------------------------
  // Highlight + fly-to
  // ------------------------
  // ------------------------
  // Highlight + fly-to
  // ------------------------
  let highlighted: any = null;

function clearHighlight() {
  if (highlighted == null) return;

  const h = highlighted as any;
  const u: any = h.userData;

  if (u?._outline) {
    h.remove(u._outline);
    u._outline.geometry?.dispose?.();
    u._outline.material?.dispose?.();
    u._outline = null;
  }

  if (u?._baseScale) {
    (h as any).scale.copy(u._baseScale);
  }

  highlighted = null;
}

function setHighlight(obj: THREE.Object3D) {
  clearHighlight();

  highlighted = obj;

  const h = obj as any;
  h.userData = h.userData || {};
  h.userData._baseScale = obj.scale.clone();

  if ((obj as any).geometry) {
    const edges = new THREE.EdgesGeometry((obj as any).geometry);
    const line = new THREE.LineSegments(
      edges,
      new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.9 })
    );
    line.renderOrder = 9999;
    obj.add(line);
    h.userData._outline = line;
  }
}

  function flyToWorldPoint(p: THREE.Vector3) {
    const idx = parseInt(floorSelect.value || "0", 10) || 0;
    const o = overlays[idx];
    const span = o ? Math.max(o.planeW, o.planeD) : 10;
    const dist = span * 0.9 + 5;

    world.camera.controls.setLookAt(
      p.x + dist,
      p.y + dist,
      p.z + dist,
      p.x,
      p.y,
      p.z,
      true
    );
  }

  (function tick() {
    requestAnimationFrame(tick);
    if (!highlighted) return;
    const u: any = highlighted.userData;
    if (!u._baseScale) return;
    const t = performance.now() * 0.006;
    const s = 1.0 + 0.08 * Math.sin(t);
    highlighted.scale.set(u._baseScale.x * s, u._baseScale.y * s, u._baseScale.z * s);
  })();

  // ------------------------
  // UI enable/disable helper
  // ------------------------
  function setControlsEnabled(v: boolean) {
    const c: any = world.camera.controls as any;
    if (c && "enabled" in c) c.enabled = v;
  }

  // ------------------------
  // Click markers (for calibration feedback)
  // ------------------------
  const markers: THREE.Object3D[] = [];
  function clearMarkers() {
    for (const m of markers) {
      world.scene.three.remove(m);
      const mesh = m as any;
      if (mesh?.geometry?.dispose) mesh.geometry.dispose();
      if (mesh?.material) {
        if (Array.isArray(mesh.material)) mesh.material.forEach((mm: any) => mm?.dispose?.());
        else mesh.material?.dispose?.();
      }
    }
    markers.length = 0;
  }

  function addMarker(p: THREE.Vector3) {
    const g = new THREE.SphereGeometry(0.12, 16, 16);
    const m = new THREE.MeshBasicMaterial({ color: 0xff00ff });
    const s = new THREE.Mesh(g, m);
    s.position.copy(p);
    world.scene.three.add(s);
    markers.push(s);
  }

  // ------------------------
  // Overlay buttons state
  // ------------------------
  function updateOverlayButtons() {
    const has = overlays.length > 0;
    calibrateBtn.disabled = !has;
    autoAlignBtn.disabled = !has;
    resetPlanBtn.disabled = !has;
    floorSelect.disabled = !has;
    focusPlanBtn.disabled = !has;
  }

  // ------------------------
  // Hash helper
  // ------------------------
  async function sha256OfFile(f: File) {
    const buf = await f.arrayBuffer();
    const hashBuf = await crypto.subtle.digest("SHA-256", buf);
    const hashArr = Array.from(new Uint8Array(hashBuf));
    return hashArr.map((b) => b.toString(16).padStart(2, "0")).join("");
  }

  // ------------------------
  // web-ifc storey extraction (IMPROVED)
  // ------------------------
  const readFileAsUint8Array = async (f: File) => new Uint8Array(await f.arrayBuffer());

  function toArrayOfIds(ids: any): number[] {
    if (!ids) return [];
    if (Array.isArray(ids)) return ids;
    if (typeof ids.size === "function" && typeof ids.get === "function") {
      const out: number[] = [];
      for (let i = 0; i < ids.size(); i++) out.push(ids.get(i));
      return out;
    }
    return [];
  }

  // ---- NEW: placement helpers ----
  function refId(x: any): number | null {
    if (!x) return null;
    if (typeof x === "number") return x;
    if (typeof x.value === "number") return x.value;
    return null;
  }

  function getCartesianPointCoords(api: any, modelID: number, pointRef: any): number[] {
    const pid = refId(pointRef);
    if (!pid) return [0, 0, 0];

    const pt = api.GetLine(modelID, pid);
    const coords = pt?.Coordinates;
    if (!coords) return [0, 0, 0];

    const arr = Array.isArray(coords) ? coords : coords?.value;
    if (!Array.isArray(arr)) return [0, 0, 0];

    const out = arr
      .map((c: any) => (typeof c?.value === "number" ? c.value : Number(c)))
      .filter((n: number) => isFinite(n));

    return [out[0] ?? 0, out[1] ?? 0, out[2] ?? 0];
  }

  /**
   * Walk IfcLocalPlacement -> PlacementRelTo chain, accumulating Z.
   * IFC vertical axis is usually Z.
   */
  function getPlacementElevZ(api: any, modelID: number, placementRef: any): number {
    let z = 0;
    let cur = placementRef;
    let guard = 0;

    while (cur && guard++ < 30) {
      const plId = refId(cur);
      if (!plId) break;

      const pl = api.GetLine(modelID, plId); // IfcLocalPlacement
      const relId = refId(pl?.RelativePlacement);
      const rel = relId ? api.GetLine(modelID, relId) : null; // IfcAxis2Placement3D/2D

      const locRef = rel?.Location;
      const coords = getCartesianPointCoords(api, modelID, locRef);

      z += coords[2] ?? 0;
      cur = pl?.PlacementRelTo;
    }

    return z;
  }

  async function extractStoreysFromIfc(buffer: Uint8Array): Promise<StoreyInfo[]> {
    const modelID = await ifcLoader.readIfcFile(buffer);
    const api: any = ifcLoader.webIfc as any;

    let rawIds: any;
    if (typeof api.GetLineIDsWithType === "function") {
      rawIds = api.GetLineIDsWithType(modelID, WEBIFC.IFCBUILDINGSTOREY);
    } else if (typeof api.GetAllItemsOfType === "function") {
      rawIds = api.GetAllItemsOfType(modelID, WEBIFC.IFCBUILDINGSTOREY, false);
    } else {
      console.warn("web-ifc API did not expose a storey query method.");
      api.CloseModel?.(modelID);
      return [];
    }

    const ids = toArrayOfIds(rawIds);
    const storeys: StoreyInfo[] = [];

    for (const id of ids) {
      const line: any = api.GetLine(modelID, id);
      const name = line?.Name?.value ?? `Storey ${id}`;

      const elevProp =
        typeof line?.Elevation?.value === "number" && isFinite(line.Elevation.value)
          ? line.Elevation.value
          : NaN;

      const placementElev = getPlacementElevZ(api, modelID, line?.ObjectPlacement);

      const elevation = isFinite(elevProp) ? elevProp : placementElev;

      storeys.push({ expressID: id, name, elevation, placementElev });
    }

    // ---- NEW: range sanity fallback ----
    // If Elevation is same for all storeys, use placementElev instead (if it varies)
    if (storeys.length >= 2) {
      const elevs = storeys.map((s) => s.elevation);
      const minE = Math.min(...elevs);
      const maxE = Math.max(...elevs);
      const range = maxE - minE;

      if (Math.abs(range) < 1e-6) {
        const pe = storeys.map((s) => s.placementElev ?? 0);
        const minP = Math.min(...pe);
        const maxP = Math.max(...pe);
        const pr = maxP - minP;

        if (Math.abs(pr) >= 1e-6) {
          storeys.forEach((s) => (s.elevation = s.placementElev ?? 0));
        }
      }
    }

    api.CloseModel?.(modelID);

    storeys.sort((a, b) => a.elevation - b.elevation);
    return storeys;
  }

  // ------------------------
  // Clear previous load
  // ------------------------
  function clearPrevious() {
    clearHighlight();
    clearMarkers();

    for (const o of overlays) {
      clearSymbols(o);
      clearWorldSymbols(o);

      world.scene.three.remove(o.mesh);
      o.material.map?.dispose();
      o.material.dispose();
      (o.mesh.geometry as THREE.BufferGeometry).dispose();
    }
    overlays = [];

    if (currentModel) {
      world.scene.three.remove(currentModel.object);
    }
    currentModel = null;
    lastBBox = null;

    floorSelect.innerHTML = "";
    updateOverlayButtons();

    updateDetectionsList();
  }

  // ------------------------
  // Hash-based persistence key
  // ------------------------
  function overlayKey(overlayIndex: number) {
    const o = overlays[overlayIndex];
    const storeyTag = o?.storey?.expressID ?? o?.storey?.name ?? String(o?.worldY ?? "unknown");
    return `ifcOverlay:v2:${o.ifcHash}:${o.planHash}:${storeyTag}`;
  }

function saveOverlayTransform(idx: number) {
  const o = overlays[idx];
  const data = {
    posXZ: [o.mesh.position.x, o.mesh.position.z], // ✅ no Y
    rotY: o.mesh.rotation.y,
    scale: o.mesh.scale.x,
    flipX: getPlanFlipX(o),
  };
  localStorage.setItem(overlayKey(idx), JSON.stringify(data));
}

function loadOverlayTransform(idx: number): boolean {
  const o = overlays[idx];
  const raw = localStorage.getItem(overlayKey(idx));
  if (!raw) return false;

  try {
    const data = JSON.parse(raw);

    // Supported formats:
    //  A) { x, z, rotY, scale, flipX }
    //  B) { posXZ:[x,z], rotY, scale, flipX }   <-- preferred new
    //  C) { pos:[x,y,z], rotY, scale, flipX }   <-- legacy

    let x = o.mesh.position.x;
    let z = o.mesh.position.z;

    if (typeof data.x === "number" && isFinite(data.x)) x = data.x;
    if (typeof data.z === "number" && isFinite(data.z)) z = data.z;

    if (Array.isArray(data.posXZ) && data.posXZ.length >= 2) {
      const px = Number(data.posXZ[0]);
      const pz = Number(data.posXZ[1]);
      if (isFinite(px)) x = px;
      if (isFinite(pz)) z = pz;
    } else if (Array.isArray(data.pos) && data.pos.length >= 3) {
      const px = Number(data.pos[0]);
      const pz = Number(data.pos[2]);
      if (isFinite(px)) x = px;
      if (isFinite(pz)) z = pz;
    }

    // ✅ Force Y to computed base (prevents all floors collapsing from old saves)
    o.mesh.position.set(x, o.basePos.y, z);

    o.mesh.rotation.y =
      typeof data.rotY === "number" && isFinite(data.rotY) ? data.rotY : o.baseRotY;

    const s = typeof data.scale === "number" && isFinite(data.scale) ? data.scale : o.baseScale;
    o.mesh.scale.set(s, s, s);

    setPlanFlipX(o, !!data.flipX);
    return true;
  } catch {
    return false;
  }
}

  function resetOverlayTransform(idx: number) {
    const o = overlays[idx];
    localStorage.removeItem(overlayKey(idx));

    o.mesh.position.copy(o.basePos);
    o.mesh.rotation.y = o.baseRotY;
    o.mesh.scale.set(o.baseScale, o.baseScale, o.baseScale);

    setPlanFlipX(o, false);
    // refreshCurrentFloor(true);
    // after applying backend align + refreshCurrentFloor(true)
// flyToWorldPoint(new THREE.Vector3(mc.x, overlay.worldY, mc.z));
  }

  // ------------------------
  // Geometry / overlay helpers
  // ------------------------
  function computeModelBBox(obj: THREE.Object3D): THREE.Box3 {
    const box = new THREE.Box3().setFromObject(obj);
    if (!isFinite(box.min.x) || !isFinite(box.max.x)) {
      return new THREE.Box3(new THREE.Vector3(-10, -1, -10), new THREE.Vector3(10, 5, 10));
    }
    return box;
  }

  function guessPlanRotationToMatchAspect(
    modelWidth: number,
    modelDepth: number,
    imgW: number,
    imgH: number
  ): { w: number; d: number; rotate90: boolean } {
    const bboxAspect = modelWidth / modelDepth;
    const imgAspect = imgW / imgH;

    const diff0 = Math.abs(bboxAspect - imgAspect);
    const diff90 = Math.abs(bboxAspect - 1 / imgAspect);

    if (diff90 < diff0) return { w: modelWidth, d: modelDepth, rotate90: true };
    return { w: modelWidth, d: modelDepth, rotate90: false };
  }

  async function createOverlayPlane(
    png: File,
    center: THREE.Vector3,
    modelWidth: number,
    modelDepth: number,
    worldY: number,
    opacity: number
  ): Promise<PlanOverlay> {
    const url = URL.createObjectURL(png);
    const tex = await new THREE.TextureLoader().loadAsync(url);
    URL.revokeObjectURL(url);

    tex.colorSpace = THREE.SRGBColorSpace;
    tex.anisotropy = 8;
    tex.wrapS = THREE.RepeatWrapping;
    tex.wrapT = THREE.RepeatWrapping;
    tex.repeat.set(1, 1);
    tex.offset.set(0, 0);

    const img: any = tex.image;
    const imgW = Number(img?.width ?? 1);
    const imgH = Number(img?.height ?? 1);

    const { w, d, rotate90 } = guessPlanRotationToMatchAspect(modelWidth, modelDepth, imgW, imgH);

    const geom = new THREE.PlaneGeometry(w, d, 1, 1);
    geom.rotateX(-Math.PI / 2);

    if (rotate90) {
      const uv = geom.attributes.uv as THREE.BufferAttribute;
      for (let i = 0; i < uv.count; i++) {
        const u = uv.getX(i);
        const v = uv.getY(i);
        uv.setXY(i, v, 1.0 - u);
      }
      uv.needsUpdate = true;
    }

    const mat = new THREE.MeshBasicMaterial({
      map: tex,
      transparent: true,
      opacity,
      depthWrite: false,
    });

    const mesh = new THREE.Mesh(geom, mat);
    mesh.position.set(center.x, worldY, center.z);
    mesh.position.y -= 0.01;
    mesh.renderOrder = -1;

    const symbols = new THREE.Group();
    symbols.position.y = 0.02;
    mesh.add(symbols);

    const worldSymbols = new THREE.Group();
    worldSymbols.position.y = 0.15;
    mesh.add(worldSymbols);

    world.scene.three.add(mesh);

    return {
      mesh,
      material: mat,
      fileName: png.name,
      worldY,

      ifcHash: "",
      planHash: "",

      basePos: mesh.position.clone(),
      baseRotY: mesh.rotation.y,
      baseScale: mesh.scale.x,

      imgW,
      imgH,
      planeW: w,
      planeD: d,
      rotate90,

      detections: [],
      symbols,
      worldSymbols,
    };
  }

  function focusCameraToPlan(center: THREE.Vector3, width: number, depth: number, y: number) {
    const dist = Math.max(width, depth) * 1.2;
    world.camera.controls.setLookAt(center.x, y + dist, center.z, center.x, y, center.z, true);
  }

  function moveOverlayAsideForCalibration(overlayIdx: number) {
    if (!currentModel) return;
    const overlay = overlays[overlayIdx];

    const modelBB = lastBBox ?? new THREE.Box3().setFromObject(currentModel.object);
    const modelSize = new THREE.Vector3();
    const modelCenter = new THREE.Vector3();
    modelBB.getSize(modelSize);
    modelBB.getCenter(modelCenter);

    const planBB = new THREE.Box3().setFromObject(overlay.mesh);
    const planSize = new THREE.Vector3();
    planBB.getSize(planSize);

    const gap =
      Math.max(modelSize.x, modelSize.z) * 0.15 + Math.max(planSize.x, planSize.z) * 0.6 + 1;

    overlay.mesh.position.set(modelBB.max.x + gap, overlay.worldY - 0.01, modelCenter.z);
  }

  // ------------------------
  // Calibration (2-point similarity)
  // ------------------------
  function beginCalibration() {
    if (!currentModel || overlays.length === 0) {
      setStatus("Load an IFC + at least one plan first.");
      return;
    }
    if (calibStage !== "idle") return;

    const overlayIdx = Math.max(
      0,
      Math.min(overlays.length - 1, parseInt(floorSelect.value || "0", 10) || 0)
    );
    calibOverlayIdx = overlayIdx;

    const active = overlays[overlayIdx];
    calibBackup = {
      overlayIdx,
      overlayPos: active.mesh.position.clone(),
      overlayRotY: active.mesh.rotation.y,
      overlayScale: active.mesh.scale.x,
      overlaysVisible: overlays.map((o) => o.mesh.visible),
      overlaysOpacity: overlays.map((o) => o.material.opacity),
      modelVisible: currentModel.object.visible,
      floorSelectDisabled: floorSelect.disabled,
      fileInputDisabled: fileInput.disabled,
    };

    overlays.forEach((o, i) => {
      o.mesh.visible = i === overlayIdx;
      o.worldSymbols.visible = i === overlayIdx;
      o.material.opacity = i === overlayIdx ? 1 : o.material.opacity;
      o.material.needsUpdate = true;
    });

    moveOverlayAsideForCalibration(overlayIdx);

    calibModelPts = [];
    calibPlanLocalPts = [];
    clearMarkers();
    setControlsEnabled(false);
    floorSelect.disabled = true;
    fileInput.disabled = true;

    currentModel.object.visible = true;

    // const bb = lastBBox ?? new THREE.Box3().setFromObject(currentModel.object);
    const bb = new THREE.Box3().setFromObject(currentModel.object);
    

    const c = new THREE.Vector3(); 
    bb.getCenter(c);
    console.log("MODEL bbox center", c.x, c.z);
    const size = new THREE.Vector3();
    const center = new THREE.Vector3();
    bb.getSize(size);
    bb.getCenter(center);
    focusCameraToPlan(center, Math.max(10, size.x), Math.max(10, size.z), overlays[overlayIdx].worldY);

    calibStage = "pickModel1";
    setStatus("Calibration: click 1st of 3 points on 3D model (make a triangle)");
  }

  function enterPlanPickMode() {
    if (calibOverlayIdx === null) return;
    const overlay = overlays[calibOverlayIdx];
    if (currentModel) currentModel.object.visible = false;

    const planBB = new THREE.Box3().setFromObject(overlay.mesh);
    const planSize = new THREE.Vector3();
    const planCenter = new THREE.Vector3();
    planBB.getSize(planSize);
    planBB.getCenter(planCenter);

    focusCameraToPlan(planCenter, Math.max(5, planSize.x), Math.max(5, planSize.z), overlay.worldY);
  }

  function endCalibration(restoreOverlayTransform: boolean) {
    if (!calibBackup) return;

    const overlay = overlays[calibBackup.overlayIdx];

    if (restoreOverlayTransform) {
      overlay.mesh.position.copy(calibBackup.overlayPos);
      overlay.mesh.rotation.y = calibBackup.overlayRotY;
      overlay.mesh.scale.set(calibBackup.overlayScale, 1, calibBackup.overlayScale);
    }

    overlays.forEach((o, i) => {
      o.mesh.visible = calibBackup!.overlaysVisible[i] ?? o.mesh.visible;
      o.worldSymbols.visible = calibBackup!.overlaysVisible[i] ?? o.worldSymbols.visible;
      o.material.opacity = calibBackup!.overlaysOpacity[i] ?? o.material.opacity;
      o.material.needsUpdate = true;
    });

    if (currentModel) currentModel.object.visible = calibBackup.modelVisible;

    floorSelect.disabled = calibBackup.floorSelectDisabled;
    fileInput.disabled = calibBackup.fileInputDisabled;

    calibBackup = null;
    calibOverlayIdx = null;
  }

  function getMouseNDC(ev: PointerEvent) {
    const dom = world.renderer!.three.domElement;
    const rect = dom.getBoundingClientRect();
    mouse.x = ((ev.clientX - rect.left) / rect.width) * 2 - 1;
    mouse.y = -(((ev.clientY - rect.top) / rect.height) * 2 - 1);
  }

  function pickOn(objects: THREE.Object3D[], ev: PointerEvent) {
    getMouseNDC(ev);
    raycaster.setFromCamera(mouse, world.camera.three);
    return raycaster.intersectObjects(objects, true);
  }

  function intersectHorizontalPlane(ev: PointerEvent, y: number): THREE.Vector3 | null {
    getMouseNDC(ev);
    raycaster.setFromCamera(mouse, world.camera.three);
    const o = raycaster.ray.origin;
    const d = raycaster.ray.direction;
    if (Math.abs(d.y) < 1e-8) return null;
    const t = (y - o.y) / d.y;
    if (t < 0) return null;
    return new THREE.Vector3(o.x + d.x * t, y, o.z + d.z * t);
  }
function safePickModelPoint(ev: PointerEvent, pickY: number): THREE.Vector3 | null {
  // Try raycasting only safe meshes
  if (pickableModelMeshes.length) {
    try {
      const hits = pickOn(pickableModelMeshes, ev);
      if (hits.length) {
        const p = hits[0].point.clone();
        p.y = pickY; // keep calibration 2D (XZ), lock to storey plane
        return p;
      }
    } catch (e) {
      console.warn("Model raycast failed; falling back to plane.", e);
    }
  }

  // Fallback: horizontal plane
  return intersectHorizontalPlane(ev, pickY);
}
  function setPlanFlipX(overlay: PlanOverlay, flipped: boolean) {
    const map = overlay.material.map;
    if (!map) return;
    const s = Math.abs(map.repeat.x) || 1;
    map.repeat.x = flipped ? -s : s;
    map.offset.x = flipped ? 1 : 0;
    map.needsUpdate = true;
  }

  function getPlanFlipX(overlay: PlanOverlay): boolean {
    const map = overlay.material.map;
    if (!map) return false;
    return map.repeat.x < 0;
  }

  function pixelToLocalOnPlan(overlay: PlanOverlay, px: number, py: number) {
    const { imgW, imgH, planeW, planeD, rotate90 } = overlay;

    const u_img = imgW > 0 ? px / imgW : 0;
    const v_img = imgH > 0 ? py / imgH : 0;

    let u_tex = u_img;
    let v_tex = 1.0 - v_img;

    if (getPlanFlipX(overlay)) u_tex = 1.0 - u_tex;

    let u_orig = u_tex;
    let v_orig = v_tex;
    if (rotate90) {
      u_orig = 1.0 - v_tex;
      v_orig = u_tex;
    }

    const lx = (u_orig - 0.5) * planeW;
    const lz = -(v_orig - 0.5) * planeD;

    return new THREE.Vector3(lx, 0, lz);
  }

  function clearWorldSymbols(overlay: PlanOverlay) {
    for (const child of overlay.worldSymbols.children) {
      const obj: any = child;
      if (obj.geometry?.dispose) obj.geometry.dispose();
      if (obj.material) {
        if (Array.isArray(obj.material)) obj.material.forEach((m: any) => m?.dispose?.());
        else obj.material?.dispose?.();
      }
    }
    overlay.worldSymbols.clear();
  }

  function clearSymbols(overlay: PlanOverlay) {
    for (const child of overlay.symbols.children) {
      const obj: any = child;
      if (obj.geometry?.dispose) obj.geometry.dispose();
      if (obj.material) {
        if (Array.isArray(obj.material)) obj.material.forEach((m: any) => m?.dispose?.());
        else obj.material?.dispose?.();
      }
    }
    overlay.symbols.clear();
  }

  function colorForClass(cls: Detection["cls"]) {
    switch (cls) {
      case "socket":
        return 0x00ffff;
      case "switches":
        return 0xffff00;
      case "TV_socket":
        return 0xff00ff;
      case "light_square":
        return 0x00ff00;
      default:
        return 0xffffff;
    }
  }

  function buildWorldSymbolsForOverlay(overlay: PlanOverlay, overlayIdx: number, minConf = filter.minConf) {
    clearWorldSymbols(overlay);

    for (let di = 0; di < overlay.detections.length; di++) {
      const det = overlay.detections[di];
      if (!passesFilter(det, minConf)) continue;

      const [x1, y1, x2, y2] = det.xyxy;
      const cx = (x1 + x2) * 0.5;
      const cy = (y1 + y2) * 0.5;

      const localCenter = pixelToLocalOnPlan(overlay, cx, cy);

      const localA = pixelToLocalOnPlan(overlay, x1, y1);
      const localB = pixelToLocalOnPlan(overlay, x2, y2);
      const w = Math.max(0.05, Math.abs(localB.x - localA.x));
      const d = Math.max(0.05, Math.abs(localB.z - localA.z));
      const h = Math.max(0.08, Math.max(w, d) * 0.6);

      const geom = new THREE.BoxGeometry(1, 1, 1);
      const mat = new THREE.MeshBasicMaterial({
        color: colorForClass(det.cls),
        transparent: true,
        opacity: 0.85,
        depthTest: true,
        depthWrite: false,
      });

      const cube = new THREE.Mesh(geom, mat);
      cube.position.set(localCenter.x, h * 0.5, localCenter.z);
      cube.scale.set(w, h, d);
      cube.renderOrder = 50;

      cube.userData = {
        kind: "yoloSymbol",
        overlayIdx,
        detIndex: di,
        det,
        fileName: overlay.fileName,
        storeyName: overlay.storey?.name ?? "",
      };

      overlay.worldSymbols.add(cube);
    }
  }

  function addDetectionBox(overlay: PlanOverlay, det: Detection) {
    const [x1, y1, x2, y2] = det.xyxy;

    const pTL = pixelToLocalOnPlan(overlay, x1, y1);
    const pTR = pixelToLocalOnPlan(overlay, x2, y1);
    const pBR = pixelToLocalOnPlan(overlay, x2, y2);
    const pBL = pixelToLocalOnPlan(overlay, x1, y2);

    const pts = [pTL, pTR, pBR, pBL, pTL].map((p) => new THREE.Vector3(p.x, 0, p.z));
    const geom = new THREE.BufferGeometry().setFromPoints(pts);

    const mat = new THREE.LineBasicMaterial({
      color: 0xff0000,
      transparent: true,
      opacity: 0.95,
      depthTest: false,
    });

    const line = new THREE.Line(geom, mat);
    line.renderOrder = 999;
    overlay.symbols.add(line);
  }

  function buildSymbolsForOverlay(overlay: PlanOverlay, minConf = filter.minConf) {
    clearSymbols(overlay);
    for (const det of overlay.detections) {
      if (!passesFilter(det, minConf)) continue;
      addDetectionBox(overlay, det);
    }
  }

  function v2_modelXZ(p: THREE.Vector3) {
    return new THREE.Vector2(p.x, p.z);
  }

  function v2_planXZ(local: THREE.Vector3) {
  // Use the actual geometry local XZ directly
  return new THREE.Vector2(local.x, local.z);
}

  type Similarity2D = { s: number; theta: number; t: THREE.Vector2 };

  function solveSimilarity2D(
    planA: THREE.Vector2,
    planB: THREE.Vector2,
    modelA: THREE.Vector2,
    modelB: THREE.Vector2
  ): Similarity2D | null {
    const vP = planB.clone().sub(planA);
    const vM = modelB.clone().sub(modelA);

    const lenP = vP.length();
    const lenM = vM.length();
    if (lenP < 1e-6 || lenM < 1e-6) return null;

    const s = lenM / lenP;

    const angP = Math.atan2(vP.y, vP.x);
    const angM = Math.atan2(vM.y, vM.x);
    const theta = angM - angP;

    const cos = Math.cos(theta),
      sin = Math.sin(theta);

    const RpA = new THREE.Vector2(
      s * (cos * planA.x - sin * planA.y),
      s * (sin * planA.x + cos * planA.y)
    );
    const t = modelA.clone().sub(RpA);

    return { s, theta, t };
  }

  function centroid2(pts: THREE.Vector2[]) {
  const c = new THREE.Vector2(0, 0);
  for (const p of pts) c.add(p);
  return c.multiplyScalar(1 / Math.max(1, pts.length));
}

function triArea2(a: THREE.Vector2, b: THREE.Vector2, c: THREE.Vector2) {
  return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
}

function solveSimilarity2DLeastSquares(
  planPts: THREE.Vector2[],
  modelPts: THREE.Vector2[]
): Similarity2D | null {
  if (planPts.length !== modelPts.length || planPts.length < 2) return null;

  const n = planPts.length;
  const p0 = centroid2(planPts);
  const m0 = centroid2(modelPts);

  let a = 0;
  let b = 0;
  let denom = 0;

  // compute optimal rotation (theta) from cross/dot sums
  for (let i = 0; i < n; i++) {
    const px = planPts[i].x - p0.x;
    const py = planPts[i].y - p0.y;
    const mx = modelPts[i].x - m0.x;
    const my = modelPts[i].y - m0.y;

    a += px * mx + py * my;         // dot
    b += px * my - py * mx;         // 2D cross (signed)
    denom += px * px + py * py;     // |P|^2
  }

  if (denom < 1e-9) return null;

  const theta = Math.atan2(b, a);
  const cos = Math.cos(theta);
  const sin = Math.sin(theta);

  // optimal scale
  let numer = 0;
  for (let i = 0; i < n; i++) {
    const px = planPts[i].x - p0.x;
    const py = planPts[i].y - p0.y;
    const mx = modelPts[i].x - m0.x;
    const my = modelPts[i].y - m0.y;

    const rpx = cos * px - sin * py;
    const rpy = sin * px + cos * py;

    numer += mx * rpx + my * rpy;
  }

  const s = numer / denom;
  if (!isFinite(s) || Math.abs(s) < 1e-9) return null;

  // translation: m = s*R*p + t  => t = m0 - s*R*p0
  const rp0 = new THREE.Vector2(cos * p0.x - sin * p0.y, sin * p0.x + cos * p0.y);
  const t = m0.clone().sub(rp0.multiplyScalar(s));

  return { s, theta, t };
}

function rmsError(sol: Similarity2D, planPts: THREE.Vector2[], modelPts: THREE.Vector2[]) {
  const cos = Math.cos(sol.theta);
  const sin = Math.sin(sol.theta);
  let sum = 0;

  for (let i = 0; i < planPts.length; i++) {
    const p = planPts[i];
    const mx = modelPts[i].x;
    const my = modelPts[i].y;

    const rx = sol.s * (cos * p.x - sin * p.y) + sol.t.x;
    const ry = sol.s * (sin * p.x + cos * p.y) + sol.t.y;

    const dx = rx - mx;
    const dy = ry - my;
    sum += dx * dx + dy * dy;
  }

  return Math.sqrt(sum / Math.max(1, planPts.length));
}

// mirror local coords in the axis that corresponds to flipX,
// which depends on rotate90 UV mapping
function planLocalTo2D(overlay: PlanOverlay, local: THREE.Vector3, mirrored: boolean) {
  let x = local.x;
  let z = local.z;

  if (mirrored) {
    // flipX mirrors U axis:
    // - if rotate90=false => U aligns with local X
    // - if rotate90=true  => U aligns with local Z
    if (!overlay.rotate90) x = -x;
    else z = -z;
  }

  return new THREE.Vector2(x, z);
}

function solve3ptWithAutoFlip(
  overlay: PlanOverlay,
  planLocals: THREE.Vector3[],
  modelPts: THREE.Vector2[]
): { sol: Similarity2D; shouldToggleFlipX: boolean } | null {
  const currentFlip = getPlanFlipX(overlay);

  // Case A: keep flip as-is (mirrored=false)
  const planA = planLocals.map((p) => planLocalTo2D(overlay, p, false));
  const solA = solveSimilarity2DLeastSquares(planA, modelPts);

  // Case B: toggle flip (mirrored=true)
  const planB = planLocals.map((p) => planLocalTo2D(overlay, p, true));
  const solB = solveSimilarity2DLeastSquares(planB, modelPts);

  if (!solA && !solB) return null;
  if (solA && !solB) return { sol: solA, shouldToggleFlipX: false };
  if (!solA && solB) return { sol: solB, shouldToggleFlipX: true };

  const errA = rmsError(solA!, planA, modelPts);
  const errB = rmsError(solB!, planB, modelPts);

  // choose best fit
  return errB + 1e-9 < errA
    ? { sol: solB!, shouldToggleFlipX: true }
    : { sol: solA!, shouldToggleFlipX: false };
}

function applyCalibrationToOverlay(overlay: PlanOverlay, sol: Similarity2D) {
  const y = overlay.basePos.y;
  const s = overlay.baseScale * sol.s;

  overlay.mesh.position.set(sol.t.x, y, sol.t.y);
  overlay.mesh.rotation.set(0, overlay.baseRotY + sol.theta, 0);
  overlay.mesh.scale.set(s, s, s);
}

  // ------------------------
  // Detections list (Tools panel)
  // ------------------------
  function updateDetectionsList() {
    ensureToolsPanel();
    if (!detListEl) return;

    detListEl.innerHTML = "";

    const idx = parseInt(floorSelect.value || "0", 10) || 0;
    const o = overlays[idx];
    if (!o) return;

    const filtered = o.detections
      .map((d, detIndex) => ({ d, detIndex }))
      .filter(({ d }) => passesFilter(d))
      .sort((a, b) => b.d.conf - a.d.conf);

    for (const item of filtered) {
      const row = document.createElement("button");
      row.style.textAlign = "left";
      row.style.padding = "6px";
      row.style.borderRadius = "6px";
      row.style.border = "1px solid rgba(255,255,255,0.15)";
      row.style.background = "#111";
      row.style.color = "#fff";
      row.style.cursor = "pointer";

      row.textContent = `${item.d.cls}  conf=${item.d.conf.toFixed(2)}  (#${item.detIndex})`;

      row.onclick = () => {
        const hit = o.worldSymbols.children.find((c: any) => c?.userData?.detIndex === item.detIndex) as any;
        if (hit) {
          const wp = new THREE.Vector3();
          hit.getWorldPosition(wp);
          setHighlight(hit);
          flyToWorldPoint(wp);
          showSymbolProps(hit);
        }
      };

      detListEl!.appendChild(row);
    }
  }

  function refreshCurrentFloor(rebuildMeshes = true) {
    ensureToolsPanel();

    const idx = parseInt(floorSelect.value || "0", 10) || 0;
    overlays.forEach((o, i) => {
      const visible = i === idx;
      o.mesh.visible = visible;
      o.worldSymbols.visible = visible;
    });

    if (rebuildMeshes) {
      const o = overlays[idx];
      if (o) {
        buildSymbolsForOverlay(o, filter.minConf);
        buildWorldSymbolsForOverlay(o, idx, filter.minConf);
      }
    }

    updateDetectionsList();
    applyStoreyClipping();
  }


  function downloadCanvas(c: HTMLCanvasElement, name: string) {
  c.toBlob((b) => b && downloadBlob(b, name), "image/png");
}

  // ------------------------
  // Export / import / screenshot
  // ------------------------
  function downloadBlob(blob: Blob, filename: string) {
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(a.href), 500);
  }

  function exportDetections(fmt: "json" | "csv") {
    if (!currentModel || overlays.length === 0) return;

    const payload = {
      ifc: currentIfcName,
      ifcHash: currentIfcHash,
      exportedAt: new Date().toISOString(),
      plans: overlays.map((o, overlayIdx) => {
        const dets = o.detections.map((det, detIndex) => {
          const [x1, y1, x2, y2] = det.xyxy;
          const cx = (x1 + x2) * 0.5;
          const cy = (y1 + y2) * 0.5;
          const local = pixelToLocalOnPlan(o, cx, cy);
          const wp = o.mesh.localToWorld(local.clone());

          return {
            cls: det.cls,
            conf: det.conf,
            xyxy: det.xyxy,
            world: [wp.x, wp.y, wp.z],
            detIndex,
          };
        });

        return {
          image: o.fileName,
          planHash: o.planHash,
          storey: o.storey ? { expressID: o.storey.expressID, name: o.storey.name, elevation: o.storey.elevation } : null,
          overlayIdx,
          detections: dets,
        };
      }),
    };

    if (fmt === "json") {
      const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
      downloadBlob(blob, `detections_${currentIfcName.replace(/\W+/g, "_")}.json`);
      return;
    }

    const header = [
      "ifc",
      "plan",
      "planHash",
      "storeyName",
      "storeyExpressID",
      "cls",
      "conf",
      "x1",
      "y1",
      "x2",
      "y2",
      "worldX",
      "worldY",
      "worldZ",
    ].join(",");

    const lines: string[] = [header];

    for (const p of payload.plans) {
      for (const d of p.detections) {
        const row = [
          payload.ifc,
          p.image,
          p.planHash,
          p.storey?.name ?? "",
          p.storey?.expressID ?? "",
          d.cls,
          d.conf.toFixed(4),
          d.xyxy[0].toFixed(2),
          d.xyxy[1].toFixed(2),
          d.xyxy[2].toFixed(2),
          d.xyxy[3].toFixed(2),
          d.world[0].toFixed(3),
          d.world[1].toFixed(3),
          d.world[2].toFixed(3),
        ].join(",");
        lines.push(row);
      }
    }

    const blob = new Blob([lines.join("\n")], { type: "text/csv" });
    downloadBlob(blob, `detections_${currentIfcName.replace(/\W+/g, "_")}.csv`);
  }

  async function importDetectionsFile(f: File) {
    const txt = await f.text();
    const json = JSON.parse(txt);

    if (json?.plans && Array.isArray(json.plans)) {
      for (const plan of json.plans) {
        const name = String(plan.image ?? "");
        const overlay = overlays.find((o) => o.fileName === name);
        if (!overlay) continue;

        const dets = (plan.detections ?? []).map((d: any) => ({
          cls: d.cls,
          conf: d.conf,
          xyxy: d.xyxy,
        })) as Detection[];

        overlay.detections = dets;
      }
      refreshCurrentFloor(true);
      setStatus(`Imported detections from ${f.name}`);
      return;
    }

    if (json?.image && Array.isArray(json?.detections)) {
      const overlay = overlays.find((o) => o.fileName === json.image);
      if (overlay) {
        overlay.detections = json.detections as Detection[];
        refreshCurrentFloor(true);
        setStatus(`Imported detections for ${json.image}`);
      }
    }
  }

  function downloadScreenshot() {
    const canvas = world.renderer!.three.domElement as HTMLCanvasElement;
    const url = canvas.toDataURL("image/png");
    const a = document.createElement("a");
    a.href = url;
    a.download = `viewer_${Date.now()}.png`;
    document.body.appendChild(a);
    a.click();
    a.remove();
  }

  // ------------------------
  // Auto-match plans ↔ storeys + mapping modal
  // ------------------------
  function norm(s: string) {
    return s.toLowerCase().replace(/\.[a-z0-9]+$/i, "").replace(/[^a-z0-9]+/g, " ").trim();
  }

  function extractFirstNumber(s: string): number | null {
    const m = s.match(/(\d+)/);
    return m ? parseInt(m[1], 10) : null;
  }

  function scoreMatch(planName: string, storeyName: string) {
    const p = norm(planName);
    const st = norm(storeyName);

    const pn = extractFirstNumber(p);
    const sn = extractFirstNumber(st);
    if (pn !== null && sn !== null && pn === sn) return 1.0;

    const pTokens = new Set(p.split(" ").filter(Boolean));
    const sTokens = new Set(st.split(" ").filter(Boolean));

    let inter = 0;
    for (const t of pTokens) if (sTokens.has(t)) inter++;

    const union = pTokens.size + sTokens.size - inter;
    const jacc = union > 0 ? inter / union : 0;

    const containsBoost = st.includes(p) || p.includes(st) ? 0.3 : 0;
    return Math.min(1, jacc + containsBoost);
  }

  function autoAssignPlansToStoreys(plans: File[], storeys: StoreyInfo[]) {
    const used = new Set<number>();
    const out: Array<{ planIdx: number; storeyIdx: number | null; score: number }> = [];

    for (let i = 0; i < plans.length; i++) {
      let best = { storeyIdx: null as number | null, score: 0 };

      for (let j = 0; j < storeys.length; j++) {
        if (used.has(j)) continue;
        const s = scoreMatch(plans[i].name, storeys[j].name);
        if (s > best.score) best = { storeyIdx: j, score: s };
      }

      if (best.storeyIdx !== null && best.score >= 0.25) used.add(best.storeyIdx);
      out.push({ planIdx: i, storeyIdx: best.storeyIdx, score: best.score });
    }

    return out;
  }

  function showPlanStoreyMappingModal(plans: File[], storeys: StoreyInfo[], auto: ReturnType<typeof autoAssignPlansToStoreys>) {
    return new Promise<Array<number | null>>((resolve) => {
      const modal = document.createElement("div");
      modal.style.position = "absolute";
      modal.style.inset = "0";
      modal.style.background = "rgba(0,0,0,0.65)";
      modal.style.zIndex = "10000";
      modal.style.display = "flex";
      modal.style.alignItems = "center";
      modal.style.justifyContent = "center";

      const box = document.createElement("div");
      box.style.width = "720px";
      box.style.maxWidth = "92vw";
      box.style.maxHeight = "80vh";
      box.style.overflow = "auto";
      box.style.background = "#111";
      box.style.border = "1px solid rgba(255,255,255,0.15)";
      box.style.borderRadius = "12px";
      box.style.padding = "12px";
      box.style.color = "#fff";
      box.style.fontFamily = "system-ui, sans-serif";
      box.style.fontSize = "13px";

      box.innerHTML = `
        <div style="display:flex;justify-content:space-between;align-items:center;">
          <b>Map plans to storeys</b>
          <button id="mapOk" style="background:#222;color:#fff;border:1px solid #444;border-radius:8px;padding:6px 10px;cursor:pointer;">Continue</button>
        </div>
        <div style="margin-top:6px;color:#aaa;">
          Auto-match ran. Adjust any mismatches, then Continue.
        </div>
        <hr style="border:0;border-top:1px solid rgba(255,255,255,0.15);margin:10px 0;" />
        <div id="rows" style="display:flex;flex-direction:column;gap:8px;"></div>
      `;

      const rows = box.querySelector("#rows") as HTMLDivElement;
      const selects: HTMLSelectElement[] = [];

      plans.forEach((p, i) => {
        const row = document.createElement("div");
        row.style.display = "grid";
        row.style.gridTemplateColumns = "1fr 260px";
        row.style.gap = "10px";
        row.style.alignItems = "center";

        const left = document.createElement("div");
        const a = auto.find((x) => x.planIdx === i);
        left.innerHTML = `<div><b>${p.name}</b></div><div style="color:#aaa;">auto score: ${a?.score.toFixed(2) ?? "0.00"}</div>`;

        const sel = document.createElement("select");
        sel.style.width = "100%";
        sel.style.padding = "6px";
        sel.style.borderRadius = "8px";
        sel.style.border = "1px solid #444";
        sel.style.background = "#0b0b0b";
        sel.style.color = "#fff";

        const optNone = document.createElement("option");
        optNone.value = "";
        optNone.textContent = "(none)";
        sel.appendChild(optNone);

        storeys.forEach((st, idx) => {
          const opt = document.createElement("option");
          opt.value = String(idx);
          opt.textContent = `${st.name} (elev ${st.elevation})`;
          sel.appendChild(opt);
        });

        if (a?.storeyIdx !== null && a?.storeyIdx !== undefined) sel.value = String(a.storeyIdx);
        selects.push(sel);

        row.appendChild(left);
        row.appendChild(sel);
        rows.appendChild(row);
      });

      modal.appendChild(box);
      container.appendChild(modal);

      const ok = box.querySelector("#mapOk") as HTMLButtonElement;
      ok.onclick = () => {
        const mapping = selects.map((s) => (s.value ? parseInt(s.value, 10) : null));
        modal.remove();
        resolve(mapping);
      };
    });
  }

// ------------------------
  // Main load pipeline (PDF support + mapping + hashes + imported detections)
  // ------------------------
  async function loadEverything(ifcFile: File, planPngs: File[], originalPdfs: Map<string, File>, imported?: Map<string, Detection[]>) {
  clearPrevious();
  setStatus("Loading IFC + plans...");

  currentIfcFile = ifcFile; // <--- ADDED
  const ifcBytes = await readFileAsUint8Array(ifcFile);
  currentIfcName = ifcFile.name;
  currentIfcHash = await sha256OfFile(ifcFile);

  // 1) Read storeys (your improved extractStoreysFromIfc should already be in the file)
  const storeys = await extractStoreysFromIfc(ifcBytes);

  // ---- Debug: storey table ----
  console.table(
    storeys.map((s) => ({
      id: s.expressID,
      name: s.name,
      elev: s.elevation,
      placementElev: (s as any).placementElev ?? null,
    }))
  );

  // 2) Auto-match plans↔storeys + modal if needed
  let mapping: Array<number | null> = planPngs.map(() => null);

  if (storeys.length > 0) {
    const auto = autoAssignPlansToStoreys(planPngs, storeys);

    const needsModal =
      planPngs.length !== storeys.length ||
      auto.some((a) => a.storeyIdx === null || a.score < 0.35);

    mapping = needsModal
      ? await showPlanStoreyMappingModal(planPngs, storeys, auto)
      : auto.map((a) => a.storeyIdx);
  }

  // 3) Load IFC into viewer
  currentModel = await ifcLoader.load(ifcBytes, false, ifcFile.name, {
    processData: {
      progressCallback: (p) => console.log("IFC conversion progress:", p),
    },
  });
  pickableModelMeshes = collectPickableMeshes(currentModel.object);
  console.log("Pickable model meshes:", pickableModelMeshes.length);
  (raycaster as any).firstHitOnly = true;
  // Compute model bbox
  lastBBox = computeModelBBox(currentModel.object);
  const size = new THREE.Vector3();
  const center = new THREE.Vector3();
  lastBBox.getSize(size);
  lastBBox.getCenter(center);

  const modelHeightY = size.y || 1;
  const modelWidth = size.x || 10;
  const modelDepth = size.z || 10;

  // Elevation -> worldY mapping
  let minElev = 0;
  let elevRange = 0;
  if (storeys.length >= 2) {
    minElev = storeys[0].elevation;
    elevRange = storeys[storeys.length - 1].elevation - storeys[0].elevation;
  }
  const elevToWorld = Math.abs(elevRange) >= 1e-6 ? modelHeightY / elevRange : 1;

  const opacity = parseFloat(opacitySlider.value);

  // 4) Build planInputs (pair each plan with its mapped storey), then sort by storey elevation if assigned
  const planInputs = planPngs.map((file, i) => {
    const si = mapping[i];
    return { file, storey: si !== null && si !== undefined ? storeys[si] : undefined };
  });

  planInputs.sort((a, b) => {
    const ea = a.storey?.elevation ?? Number.POSITIVE_INFINITY;
    const eb = b.storey?.elevation ?? Number.POSITIVE_INFINITY;
    if (ea !== eb) return ea - eb;
    return a.file.name.localeCompare(b.file.name);
  });

  overlays = [];

  // 5) Create overlays + detect symbols
  for (let i = 0; i < planInputs.length; i++) {
    const planFile = planInputs[i].file;
    const storey = planInputs[i].storey;

    // ✅ Guaranteed fallback spacing even if storeys missing/broken
    const fallbackStep = (modelHeightY || 6) / Math.max(1, planInputs.length - 1);

    let worldY = lastBBox.min.y + i * fallbackStep; // default: spread by index

    if (storey) {
      if (Math.abs(elevRange) >= 1e-6) {
        // normal case: map IFC storey elevations to model height
        worldY = lastBBox.min.y + (storey.elevation - minElev) * elevToWorld;
      } else if (storeys.length >= 2) {
        // storeys exist but elevation range collapsed → spread by storey index
        const si = storeys.findIndex((s) => s.expressID === storey.expressID);
        worldY =
          lastBBox.min.y +
          Math.max(0, si) * (modelHeightY / Math.max(1, storeys.length - 1));
      }
    }

    const overlay = await createOverlayPlane(
      planFile,
      center,
      modelWidth,
      modelDepth,
      worldY,
      opacity
    );

    overlay.storey = storey;
    overlay.originalPdfFile = originalPdfs.get(planFile.name); // <--- ADDED

    // hashes for persistence
    overlay.ifcHash = currentIfcHash;
    overlay.planHash = await sha256OfFile(planFile);

    // default flip if you want it
    setPlanFlipX(overlay, true);

    // ---- Debug: plan height ----
    console.log("PLAN HEIGHT", {
      i,
      plan: overlay.fileName,
      storey: overlay.storey?.name,
      elev: overlay.storey?.elevation,
      worldY: overlay.worldY,
      meshY: overlay.mesh.position.y,
    });

    // Imported detections (skip YOLO) vs YOLO
    const importedDets = imported?.get(overlay.fileName);

    if (importedDets) {
      overlay.detections = importedDets;
      buildWorldSymbolsForOverlay(overlay, i, filter.minConf);
      buildSymbolsForOverlay(overlay, filter.minConf);
    } else {
      try {
        setStatus(`Detecting symbols for ${overlay.fileName}...`);
        overlay.detections = await detector.detectFile(planFile, 0.05, 0.45);
        buildWorldSymbolsForOverlay(overlay, i, filter.minConf);
        buildSymbolsForOverlay(overlay, filter.minConf);
      } catch (e) {
        console.warn(e);
        setStatus(`Detector failed for ${overlay.fileName}`);
      }
    }

    overlays.push(overlay);
  }

  // 6) Apply saved transforms AFTER overlays are created
  // IMPORTANT: this only works correctly if your loadOverlayTransform does NOT overwrite Y.
  overlays.forEach((o, i) => {
    const ok = loadOverlayTransform(i);
    if (ok) console.log("Applied saved transform for overlay", i);

    // rebuild after load (flip/rotate90 mapping can affect pixel->local)
    buildSymbolsForOverlay(o, filter.minConf);
    buildWorldSymbolsForOverlay(o, i, filter.minConf);
  });

  // 7) Populate floor dropdown
  floorSelect.innerHTML = "";
  overlays.forEach((o, i) => {
    const opt = document.createElement("option");
    const label = o.storey
      ? `${i + 1}: ${o.storey.name} (elev ${o.storey.elevation})`
      : `${i + 1}: ${o.fileName}`;
    opt.value = String(i);
    opt.textContent = label;
    floorSelect.appendChild(opt);
  });

  // 8) Show first overlay using your standard refresh pipeline (also updates list + clipping)
  floorSelect.value = "0";
  ensureToolsPanel();
  refreshCurrentFloor(true);

  // 9) Focus whole model
  world.camera.controls.setLookAt(
    center.x + modelWidth * 0.8,
    center.y + modelHeightY * 0.6,
    center.z + modelDepth * 0.8,
    center.x,
    center.y,
    center.z,
    true
  );

  const drop = document.getElementById("drop");
  if (drop) drop.style.display = "none";

  updateOverlayButtons();
  applyStoreyClipping();
  setStatus("Loaded ✅  Use Floor selector / Opacity / Tools / Calibrate.");

  function focusCameraToModel(opts?: { pad?: number }) {
  if (!currentModel) return;

  // While debugging, disable clipping so you don't "focus into nothing"
  filter.isolateStorey = false;
  applyStoreyClipping();

  const obj = currentModel.object;
  obj.updateWorldMatrix(true, true);

  const bb = new THREE.Box3().setFromObject(obj);
  const center = new THREE.Vector3();
  const size = new THREE.Vector3();
  bb.getCenter(center);
  bb.getSize(size);

  const span = Math.max(size.x, size.y, size.z);
  const pad = opts?.pad ?? 2.0;
  const dist = Math.max(10, span * pad);

  // Fix clipping planes (this prevents black screen from clipping)
  const cam: any = world.camera.three;
  cam.near = Math.max(0.01, dist / 1000);
  cam.far = dist * 2000;
  if (cam.updateProjectionMatrix) cam.updateProjectionMatrix();

  // Reset ortho zoom if you're in ortho mode (common reason for "black screen")
  if (cam.isOrthographicCamera) {
    cam.zoom = 1;
    cam.updateProjectionMatrix();
  }

  // Put camera on diagonal above model and look at model center
  world.camera.controls.setLookAt(
    center.x + dist,
    center.y + dist,
    center.z + dist,
    center.x,
    center.y,
    center.z,
    true
  );
      focusCameraToModel({ pad: 2.5 });

}
}

// ------------------------
// IFC edges raster (feature-edges equivalent)
// ------------------------
function disposeEdgeScene(scene: THREE.Scene) {
  scene.traverse((obj: any) => {
    if (obj?.isLineSegments) {
      obj.geometry?.dispose?.();
      // material may be shared; dispose once later if you track it
    }
  });
  // best effort: dispose unique materials
  const mats = new Set<any>();
  scene.traverse((obj: any) => {
    const m = (obj as any).material;
    if (m) mats.add(m);
  });
  mats.forEach((m: any) => m?.dispose?.());
}

// function buildIfcEdgeScene(thresholdAngleDeg = 25): THREE.Scene {
//   if (!currentModel) throw new Error("No IFC model");

//   const scene = new THREE.Scene();
//   scene.background = new THREE.Color(0x000000);

//   const lineMat = new THREE.LineBasicMaterial({ color: 0xffffff });
//   const cache = new Map<string, THREE.EdgesGeometry>();

//   currentModel.object.updateWorldMatrix(true, true);

//   currentModel.object.traverse((obj: any) => {
//     if (!obj?.isMesh) return;
//     const g: any = obj.geometry;
//     if (!g?.isBufferGeometry) return;

//     // reuse edges geom by geometry uuid
//     let eg = cache.get(g.uuid);
//     if (!eg) {
//       eg = new THREE.EdgesGeometry(g, thresholdAngleDeg);
//       cache.set(g.uuid, eg);
//     }

//     const ls = new THREE.LineSegments(eg, lineMat);
//     ls.matrixAutoUpdate = false;
//     ls.matrix.copy(obj.matrixWorld);
//     scene.add(ls);
//   });

//   return scene;
// }

// async function rasterizeIfcEdgesTopDown(
//   cv: any,
//   overlay: PlanOverlay,
//   outSize = 1024,
//   margin = 10
// ): Promise<{ edges: any; A_ifc: Mat3; invA_ifc: Mat3; outCanvas: HTMLCanvasElement }> {
//   if (!currentModel || !lastBBox) throw new Error("No model/bbox");

//   // Use storey clipping band while rendering (align THIS floor, not whole building)
//   const prevIso = filter.isolateStorey;
//   const prevBand = filter.isolateBand;

//   try {
//     filter.isolateStorey = true;
//     filter.isolateBand = Math.max(1.5, prevBand);
//     applyStoreyClipping();

//     const bb = lastBBox;
//     const center = new THREE.Vector3();
//     bb.getCenter(center);

//     const w = bb.max.x - bb.min.x;
//     const d = bb.max.z - bb.min.z;
//     const span = Math.max(w, d) * 1.10; // padding

//     // World square in XZ
//     const x0 = center.x - span / 2;
//     const z0 = center.z - span / 2;
//     const zMax = center.z + span / 2;

//     const scale = (outSize - 2 * margin) / span;

//     // A_ifc maps world (x,z,1) -> out pixels (px,py,1)
//     // py uses (zMax - z) to flip "up" to pixel-down
//     const A_ifc: Mat3 = [
//       scale,  0,      margin - scale * x0,
//       0,     -scale,  margin + scale * zMax,
//       0,      0,      1,
//     ];
//     const invA_ifc = mat3Inv(A_ifc);

//     // Build edge-only scene
//     const edgeScene = buildIfcEdgeScene(25);

//     // Ortho camera top-down (looking -Y), with Z as "up" in view
//     const cam = new THREE.OrthographicCamera(-span / 2, span / 2, span / 2, -span / 2, 0.1, 1e6);
//     cam.position.set(center.x, bb.max.y + 50, center.z);
//     cam.up.set(0, 0, 1);
//     cam.lookAt(center.x, overlay.worldY ?? center.y, center.z);
//     cam.updateProjectionMatrix();

//     const renderer = world.renderer.three as THREE.WebGLRenderer;
//     const rt = new THREE.WebGLRenderTarget(outSize, outSize, { depthBuffer: false, stencilBuffer: false });

//     renderer.setRenderTarget(rt);
//     renderer.clear(true, true, true);
//     renderer.render(edgeScene, cam);

//     const buf = new Uint8Array(outSize * outSize * 4);
//     renderer.readRenderTargetPixels(rt, 0, 0, outSize, outSize, buf);
//     renderer.setRenderTarget(null);

//     rt.dispose();
//     disposeEdgeScene(edgeScene);

//     // WebGL pixels are bottom-up → flip into canvas top-down
//     const c = document.createElement("canvas");
//     c.width = outSize;
//     c.height = outSize;
//     const ctx = c.getContext("2d")!;
//     const imgData = ctx.createImageData(outSize, outSize);

//     for (let y = 0; y < outSize; y++) {
//       const srcY = outSize - 1 - y;
//       for (let x = 0; x < outSize; x++) {
//         const si = (srcY * outSize + x) * 4;
//         const di = (y * outSize + x) * 4;
//         imgData.data[di + 0] = buf[si + 0];
//         imgData.data[di + 1] = buf[si + 1];
//         imgData.data[di + 2] = buf[si + 2];
//         imgData.data[di + 3] = 255;
//       }
//     }
//     ctx.putImageData(imgData, 0, 0);

//     // OpenCV: grayscale + threshold to get binary edges + dilate
//     const rgba = cv.imread(c);
//     const gray = new cv.Mat();
//     cv.cvtColor(rgba, gray, cv.COLOR_RGBA2GRAY);

//     const bin = new cv.Mat();
//     cv.threshold(gray, bin, 10, 255, cv.THRESH_BINARY);

//     const k = cv.Mat.ones(3, 3, cv.CV_8U);
//     const dil = new cv.Mat();
//     cv.dilate(bin, dil, k, new cv.Point(-1, -1), 1);

//     rgba.delete(); gray.delete(); bin.delete(); k.delete();

//     return { edges: dil, A_ifc, invA_ifc, outCanvas: c };
//   } finally {
//     // restore clipping state
//     filter.isolateStorey = prevIso;
//     filter.isolateBand = prevBand;
//     applyStoreyClipping();
//   }
// }


//   async function rasterizeIfcEdgesTopDownNormals(
//   cv: any,
//   overlay: PlanOverlay,
//   outSize = 1024,
//   margin = 10
// ): Promise<{ edges: any; A_ifc: Mat3; invA_ifc: Mat3; outCanvas: HTMLCanvasElement }> {
//   if (!currentModel || !lastBBox) throw new Error("No model/bbox");

//   // --- hide overlays while rendering IFC (avoid plan leaking into IFC raster) ---
//   const prevVis = overlays.map((o) => o.mesh.visible);
//   overlays.forEach((o) => (o.mesh.visible = false));

//   // --- enable storey clipping while rendering, so we get THIS floor slice ---
//   const prevIso = filter.isolateStorey;
//   const prevBand = filter.isolateBand;
//   filter.isolateStorey = true;
//   filter.isolateBand = Math.max(1.5, prevBand);
//   applyStoreyClipping();

//   const bb = lastBBox;
//   const center = new THREE.Vector3();
//   bb.getCenter(center);

//   const w = bb.max.x - bb.min.x;
//   const d = bb.max.z - bb.min.z;
//   const span = Math.max(w, d) * 1.1; // padding

//   // world square in XZ
//   const x0 = center.x - span / 2;
//   const zMax = center.z + span / 2;
//   const scale = (outSize - 2 * margin) / span;

//   // A_ifc maps world (x,z)-> pixel (px,py)
//   const A_ifc: Mat3 = [
//     scale, 0,     margin - scale * x0,
//     0,     -scale, margin + scale * zMax,
//     0,     0,     1,
//   ];
//   const invA_ifc = mat3Inv(A_ifc);

//   // --- top-down ortho camera ---
//   const cam = new THREE.OrthographicCamera(-span / 2, span / 2, span / 2, -span / 2, 0.1, 1e6);
//   cam.position.set(center.x, bb.max.y + 50, center.z);
//   cam.up.set(0, 0, 1);
//   cam.lookAt(center.x, overlay.worldY ?? center.y, center.z);
//   cam.updateProjectionMatrix();

//   // --- render target ---
//   const renderer = world.renderer!.three as THREE.WebGLRenderer;
//   const rt = new THREE.WebGLRenderTarget(outSize, outSize, { depthBuffer: false, stencilBuffer: false });

//     // override material = normals (stable, no EdgesGeometry)
//   const scene = world.scene.three;
//   const prevOverride = (scene as any).overrideMaterial;

//   const normalMat = new THREE.MeshNormalMaterial({ flatShading: false });

//   // ✅ IMPORTANT: make storey clipping work even with overrideMaterial
//   normalMat.clippingPlanes = filter.isolateStorey ? clipPlanes : null;
//   normalMat.clipIntersection = true;
//   normalMat.needsUpdate = true;

//   (scene as any).overrideMaterial = normalMat;

//   renderer.setRenderTarget(rt);
//   renderer.clear(true, true, true);
//   renderer.render(scene, cam);

//   // read pixels (WebGL is bottom-up)
//   const buf = new Uint8Array(outSize * outSize * 4);
//   renderer.readRenderTargetPixels(rt, 0, 0, outSize, outSize, buf);
//   renderer.setRenderTarget(null);

//   // restore
//   (scene as any).overrideMaterial = prevOverride;
//   normalMat.dispose();
//   rt.dispose();

//   // --- restore clipping + overlays ---
//   filter.isolateStorey = prevIso;
//   filter.isolateBand = prevBand;
//   applyStoreyClipping();

//   overlays.forEach((o, i) => (o.mesh.visible = prevVis[i]));

//   // --- flip into canvas ---
//   const c = document.createElement("canvas");
//   c.width = outSize;
//   c.height = outSize;
//   const ctx = c.getContext("2d")!;
//   const imgData = ctx.createImageData(outSize, outSize);

//   for (let y = 0; y < outSize; y++) {
//     const srcY = outSize - 1 - y;
//     for (let x = 0; x < outSize; x++) {
//       const si = (srcY * outSize + x) * 4;
//       const di = (y * outSize + x) * 4;
//       imgData.data[di + 0] = buf[si + 0];
//       imgData.data[di + 1] = buf[si + 1];
//       imgData.data[di + 2] = buf[si + 2];
//       imgData.data[di + 3] = 255;
//     }
//   }
//   ctx.putImageData(imgData, 0, 0);

//   // --- OpenCV: edges via Canny ---
//   const rgba = cv.imread(c);
//   const gray = new cv.Mat();
//   cv.cvtColor(rgba, gray, cv.COLOR_RGBA2GRAY);

//   const edges = new cv.Mat();
//   cv.Canny(gray, edges, 40, 120);

//   // dilate (helps ECC lock)
//   const k = cv.Mat.ones(3, 3, cv.CV_8U);
//   cv.dilate(edges, edges, k, new cv.Point(-1, -1), 1);

//   rgba.delete(); gray.delete(); k.delete();

// // ✅ NEW: crop IFC edges to content too (stabilizes ECC)
// const cropped = autoCropEdgesToSquare(cv, edges, A_ifc, outSize, 12);
// if (cropped.edges !== edges) edges.delete();

// return { edges: cropped.edges, A_ifc: cropped.A, invA_ifc: cropped.invA, outCanvas: c };
// }


async function rasterizeIfcEdgesTopDownNormals(
  cv: any,
  overlay: PlanOverlay,
  outSize = 1024,
  margin = 10
): Promise<{ edges: any; A_ifc: Mat3; invA_ifc: Mat3; outCanvas: HTMLCanvasElement }> {
  if (!currentModel || !lastBBox) throw new Error("No model/bbox");

  // --- hide overlays while rendering IFC (avoid plan leaking into IFC raster) ---
  const prevVis = overlays.map((o) => o.mesh.visible);
  overlays.forEach((o) => (o.mesh.visible = false));

  // --- hide grid (critical: prevents ECC from locking onto long grid lines) ---
  const prevGridVis = grid ? getGridVisible(grid) : false;
  if (grid) setGridVisible(grid, false);

  // --- enable storey clipping while rendering, so we get THIS floor slice ---
  const prevIso = filter.isolateStorey;
  const prevBand = filter.isolateBand;
  filter.isolateStorey = true;
  filter.isolateBand = Math.max(1.5, prevBand);
  applyStoreyClipping();

  // We will restore everything no matter what
  let rt: THREE.WebGLRenderTarget | null = null;
  let normalMat: THREE.MeshNormalMaterial | null = null;
  const scene = world.scene.three;
  const prevOverride = (scene as any).overrideMaterial;

  try {
    const bb = lastBBox;
    const center = new THREE.Vector3();
    bb.getCenter(center);

    const w = bb.max.x - bb.min.x;
    const d = bb.max.z - bb.min.z;
    const span = Math.max(w, d) * 1.1; // padding

    // world square in XZ
    const x0 = center.x - span / 2;
    const zMax = center.z + span / 2;
    const scale = (outSize - 2 * margin) / span;

    // A_ifc maps world (x,z)-> pixel (px,py)
    const A_ifc: Mat3 = [
      scale, 0,      margin - scale * x0,
      0,     -scale, margin + scale * zMax,
      0,     0,      1,
    ];
    const invA_ifc = mat3Inv(A_ifc);

    // --- top-down ortho camera ---
    const cam = new THREE.OrthographicCamera(-span / 2, span / 2, span / 2, -span / 2, 0.1, 1e6);
    cam.position.set(center.x, bb.max.y + 50, center.z);
    cam.up.set(0, 0, 1);
    cam.lookAt(center.x, overlay.worldY ?? center.y, center.z);
    cam.updateProjectionMatrix();

    // --- render target ---
    const renderer = world.renderer!.three as THREE.WebGLRenderer;
    rt = new THREE.WebGLRenderTarget(outSize, outSize, { depthBuffer: false, stencilBuffer: false });

    // override material = normals (stable)
    normalMat = new THREE.MeshNormalMaterial({ flatShading: false });

    // IMPORTANT: make storey clipping work even with overrideMaterial
    normalMat.clippingPlanes = filter.isolateStorey ? clipPlanes : null;
    normalMat.clipIntersection = true;
    normalMat.needsUpdate = true;

    (scene as any).overrideMaterial = normalMat;

    renderer.setRenderTarget(rt);
    renderer.clear(true, true, true);
    renderer.render(scene, cam);

    // read pixels (WebGL is bottom-up)
    const buf = new Uint8Array(outSize * outSize * 4);
    renderer.readRenderTargetPixels(rt, 0, 0, outSize, outSize, buf);
    renderer.setRenderTarget(null);

    // --- flip into canvas ---
    const c = document.createElement("canvas");
    c.width = outSize;
    c.height = outSize;
    const ctx = c.getContext("2d")!;
    const imgData = ctx.createImageData(outSize, outSize);

    for (let y = 0; y < outSize; y++) {
      const srcY = outSize - 1 - y;
      for (let x = 0; x < outSize; x++) {
        const si = (srcY * outSize + x) * 4;
        const di = (y * outSize + x) * 4;
        imgData.data[di + 0] = buf[si + 0];
        imgData.data[di + 1] = buf[si + 1];
        imgData.data[di + 2] = buf[si + 2];
        imgData.data[di + 3] = 255;
      }
    }
    ctx.putImageData(imgData, 0, 0);

    // --- OpenCV: edges via Canny ---
    const rgba = cv.imread(c);
    const gray = new cv.Mat();
    cv.cvtColor(rgba, gray, cv.COLOR_RGBA2GRAY);

    const edges = new cv.Mat();
    cv.Canny(gray, edges, 40, 120);

    // dilate (helps ECC lock)
    const k = cv.Mat.ones(3, 3, cv.CV_8U);
    cv.dilate(edges, edges, k, new cv.Point(-1, -1), 1);

    rgba.delete(); gray.delete(); k.delete();

    // crop IFC edges to content too (stabilizes ECC)
    const cropped = autoCropEdgesToSquare(cv, edges, A_ifc, outSize, 12);
    if (cropped.edges !== edges) edges.delete();

    return { edges: cropped.edges, A_ifc: cropped.A, invA_ifc: cropped.invA, outCanvas: c };
  } finally {
    // restore override material + GPU resources
    (scene as any).overrideMaterial = prevOverride;
    try { normalMat?.dispose?.(); } catch {}
    try { rt?.dispose?.(); } catch {}

    // restore clipping + overlays + grid
    filter.isolateStorey = prevIso;
    filter.isolateBand = prevBand;
    applyStoreyClipping();

    overlays.forEach((o, i) => (o.mesh.visible = prevVis[i]));
    if (grid) setGridVisible(grid, prevGridVis);
  }
}


// ------------------------
// The one function you actually want: Auto-align current overlay (ECC)
// ------------------------
// async function autoAlignCurrentFloorECC() {
  
//   if (!currentModel || overlays.length === 0) return;

//   const cv = await getCV();
//   console.log("ECC:", typeof cv.findTransformECC);

//   const idx = parseInt(floorSelect.value || "0", 10) || 0;
//   const overlay = overlays[idx];


//     const matsToDelete = new Set<any>();
//   const track = (m: any) => {
//     if (m && typeof m.delete === "function") matsToDelete.add(m);
//     return m;
//   };


// //   setStatus("Auto-align: rasterizing edges...");
// //   const outSize = 1024;

// //   const plan = await rasterizePlanEdges(cv, overlay, outSize, 10);
// //   const ifc = await rasterizeIfcEdgesTopDownNormals(cv, overlay, outSize, 10);
// //   setStatus("Auto-align: ECC (AFFINE → EUCLIDEAN)...");
// // downloadCanvas(plan.outCanvas, "debug_plan.png");
// // downloadCanvas(ifc.outCanvas, "debug_ifc.png");
// //   console.log("ECC sizes", {
// //   fixed: [plan.edges.rows, plan.edges.cols],
// //   moving: [ifc.edges.rows, ifc.edges.cols],
// // });
// // console.log("ECC input sizes", {
// //   plan: [plan.edges.rows, plan.edges.cols],
// //   ifc:  [ifc.edges.rows, ifc.edges.cols],
// //   planNonZero: cv.countNonZero(plan.edges),
// //   ifcNonZero:  cv.countNonZero(ifc.edges),
// // });



// // const t0 = eccAlignSafe(cv, ifc.edges, plan.edges, cv.MOTION_TRANSLATION, 800, "T");
// // const t0Aligned = t0?.aligned ?? ifc.edges;

// // const aff = eccAlignSafe(cv, t0Aligned, plan.edges, cv.MOTION_AFFINE, 1500, "A");
// // const affAligned = aff?.aligned ?? t0Aligned;

// // const rig = eccAlignSafe(cv, affAligned, plan.edges, cv.MOTION_EUCLIDEAN, 1200, "E");

// // if (!rig) {
// //   plan.edges.delete(); ifc.edges.delete();
// //   t0?.warp?.delete?.(); t0?.aligned?.delete?.();
// //   aff?.warp?.delete?.(); aff?.aligned?.delete?.();
// //   setStatus("Auto-align failed: ECC did not converge (see console).");
// //   return;
// // }

// // // clean up previous stages
// // t0?.warp?.delete?.();  t0?.aligned?.delete?.();
// // aff?.warp?.delete?.(); aff?.aligned?.delete?.();

// // // cleanup extra stage
// // t0?.warp?.delete?.();
// // t0?.aligned?.delete?.();

// // aff?.warp?.delete?.();
// // aff?.aligned?.delete?.();
// //   const cham = trimmedChamfer(cv, rig.aligned, plan.edges, 90);

// //   const W = warp2x3ToMat3(rig.warp); // warp in OUT-pixel space
// //   const invW = mat3Inv(W);

// //   // Candidate directions
// //   const T1 = mat3Mul(mat3Mul(ifc.invA_ifc, W), plan.A_plan);
// //   const T2 = mat3Mul(mat3Mul(ifc.invA_ifc, invW), plan.A_plan);

// //   setStatus("Auto-align: sampling edges + solving overlay transform...");

// //   // Sample plan edges: OUT -> plan pixels -> (planLocal, modelXZ)
// //   const planLocals: THREE.Vector3[] = [];
// //   const planPix: { x: number; y: number }[] = [];

// //   const step = 6;
// //   const maxSamples = 500;

// //   for (let y = 0; y < plan.edges.rows; y += step) {
// //     for (let x = 0; x < plan.edges.cols; x += step) {
// //       if (plan.edges.ucharPtr(y, x)[0] === 0) continue;

// //       const pPix = applyH(plan.invA_plan, x, y);
// //       if (!isFinite(pPix.x) || !isFinite(pPix.y)) continue;

// //       // keep only points that land inside original image bounds
// //       if (pPix.x < 0 || pPix.y < 0 || pPix.x > overlay.imgW || pPix.y > overlay.imgH) continue;

// //       const local = pixelToLocalOnPlan(overlay, pPix.x, pPix.y);
// //       planLocals.push(local);
// //       planPix.push(pPix);

// //       if (planLocals.length >= maxSamples) break;
// //     }
// //     if (planLocals.length >= maxSamples) break;
// //   }

// //   if (planLocals.length < 20) {
// //     setStatus("Auto-align failed: not enough edge samples. Try a cleaner plan page.");

// //     // cleanup
// //     plan.edges.delete(); ifc.edges.delete();
// //     aff?.warp?.delete?.();
// // aff?.aligned?.delete?.();
// //     rig.warp.delete(); rig.aligned.delete();
// //     return;
// //   }

// //   function evalCandidate(T: Mat3) {
// //     const modelPts: THREE.Vector2[] = planPix.map((p) => {
// //       const m = applyH(T, p.x, p.y);
// //       return new THREE.Vector2(m.x, m.y); // y is worldZ
// //     });

// //     const res = solve3ptWithAutoFlip(overlay, planLocals, modelPts);
// //     if (!res) return null;

// //     const plan2D = planLocals.map((p) => planLocalTo2D(overlay, p, res.shouldToggleFlipX));
// //     const err = rmsError(res.sol, plan2D, modelPts);
// //     if (!isFinite(err)) return null;

// //     return { res, err };
// //   }

// //   const e1 = evalCandidate(T1);
// //   const e2 = evalCandidate(T2);

// //   const chosen =
// //     e1 && e2 ? (e1.err <= e2.err ? { ...e1, which: "T1" as const } : { ...e2, which: "T2" as const }) :
// //     e1 ? { ...e1, which: "T1" as const } :
// //     e2 ? { ...e2, which: "T2" as const } :
// //     null;

// //   if (!chosen) {
// //     setStatus("Auto-align failed: similarity solve failed (both directions).");

// //     plan.edges.delete(); ifc.edges.delete();
// //     aff?.warp?.delete?.();
// // aff?.aligned?.delete?.();
// //     rig.warp.delete(); rig.aligned.delete();
// //     return;
// //   }

// //   // Apply result
// //   if (chosen.res.shouldToggleFlipX) {
// //     setPlanFlipX(overlay, !getPlanFlipX(overlay));
// //   }

// //   applyCalibrationToOverlay(overlay, chosen.res.sol);
// //   saveOverlayTransform(idx);

// //   setStatus(
// //     `Auto-align saved ✅ (ECC cc=${(rig.cc?.toFixed?.(4) ?? rig.cc)}, chamferMean≈${cham.mean.toFixed(2)}px, rms≈${chosen.err.toFixed(2)} [${chosen.which}])`
// //   );
// //   refreshCurrentFloor(true);

// //   // cleanup cv mats
// //   plan.edges.delete(); ifc.edges.delete();
// //   aff?.warp?.delete?.();
// // aff?.aligned?.delete?.();
// //   rig.warp.delete(); rig.aligned.delete();
// // }


// try {
//   setStatus("Auto-align: rasterizing edges...");
//   const outSize = 1024;

//   const plan = await rasterizePlanEdges(cv, overlay, outSize, 10);
//   const ifc = await rasterizeIfcEdgesTopDownNormals(cv, overlay, outSize, 10);

//   track(plan.edges);
//   track(ifc.edges);

//   setStatus("Auto-align: ECC (AFFINE → EUCLIDEAN)...");

//   console.log("ECC sizes", {
//     fixed: [plan.edges.rows, plan.edges.cols],
//     moving: [ifc.edges.rows, ifc.edges.cols],
//   });
//   console.log("ECC input sizes", {
//     plan: [plan.edges.rows, plan.edges.cols],
//     ifc:  [ifc.edges.rows, ifc.edges.cols],
//     planNonZero: cv.countNonZero(plan.edges),
//     ifcNonZero:  cv.countNonZero(ifc.edges),
//   });

//   const t0 = eccAlignSafe(cv, ifc.edges, plan.edges, cv.MOTION_TRANSLATION, 800, "T");
//   if (t0) { track(t0.warp); track(t0.aligned); }
//   const t0Aligned = t0?.aligned ?? ifc.edges;

//   const aff = eccAlignSafe(cv, t0Aligned, plan.edges, cv.MOTION_AFFINE, 1500, "A");
//   if (aff) { track(aff.warp); track(aff.aligned); }
//   const affAligned = aff?.aligned ?? t0Aligned;

//   const rig = eccAlignSafe(cv, affAligned, plan.edges, cv.MOTION_EUCLIDEAN, 1200, "E");
//   if (rig) { track(rig.warp); track(rig.aligned); }

//   if (!rig) {
//     setStatus("Auto-align failed: ECC did not converge (see console).");
//     return;
//   }

//   const cham = trimmedChamfer(cv, rig.aligned, plan.edges, 90);

//   const W = warp2x3ToMat3(rig.warp);
//   const invW = mat3Inv(W);

//   const T1 = mat3Mul(mat3Mul(ifc.invA_ifc, W), plan.A_plan);
//   const T2 = mat3Mul(mat3Mul(ifc.invA_ifc, invW), plan.A_plan);

//   setStatus("Auto-align: sampling edges + solving overlay transform...");

//   const planLocals: THREE.Vector3[] = [];
//   const planPix: { x: number; y: number }[] = [];

//   const step = 6;
//   const maxSamples = 500;

//   for (let y = 0; y < plan.edges.rows; y += step) {
//     for (let x = 0; x < plan.edges.cols; x += step) {
//       if (plan.edges.ucharPtr(y, x)[0] === 0) continue;

//       const pPix = applyH(plan.invA_plan, x, y);
//       if (!isFinite(pPix.x) || !isFinite(pPix.y)) continue;
//       if (pPix.x < 0 || pPix.y < 0 || pPix.x > overlay.imgW || pPix.y > overlay.imgH) continue;

//       const local = pixelToLocalOnPlan(overlay, pPix.x, pPix.y);
//       planLocals.push(local);
//       planPix.push(pPix);

//       if (planLocals.length >= maxSamples) break;
//     }
//     if (planLocals.length >= maxSamples) break;
//   }

//   if (planLocals.length < 20) {
//     setStatus("Auto-align failed: not enough edge samples. Try a cleaner plan page.");
//     return;
//   }

//   function evalCandidate(T: Mat3) {
//     const modelPts: THREE.Vector2[] = planPix.map((p) => {
//       const m = applyH(T, p.x, p.y);
//       return new THREE.Vector2(m.x, m.y);
//     });

//     const res = solve3ptWithAutoFlip(overlay, planLocals, modelPts);
//     if (!res) return null;

//     const plan2D = planLocals.map((p) => planLocalTo2D(overlay, p, res.shouldToggleFlipX));
//     const err = rmsError(res.sol, plan2D, modelPts);
//     if (!isFinite(err)) return null;

//     return { res, err };
//   }

//   const e1 = evalCandidate(T1);
//   const e2 = evalCandidate(T2);

//   const chosen =
//     e1 && e2 ? (e1.err <= e2.err ? { ...e1, which: "T1" as const } : { ...e2, which: "T2" as const }) :
//     e1 ? { ...e1, which: "T1" as const } :
//     e2 ? { ...e2, which: "T2" as const } :
//     null;

//   if (!chosen) {
//     setStatus("Auto-align failed: similarity solve failed (both directions).");
//     return;
//   }

//   if (chosen.res.shouldToggleFlipX) {
//     setPlanFlipX(overlay, !getPlanFlipX(overlay));
//   }

//   applyCalibrationToOverlay(overlay, chosen.res.sol);
//   saveOverlayTransform(idx);

//   setStatus(
//     `Auto-align saved ✅ (ECC cc=${(rig.cc?.toFixed?.(4) ?? rig.cc)}, chamferMean≈${cham.mean.toFixed(2)}px, rms≈${chosen.err.toFixed(2)} [${chosen.which}])`
//   );

//   refreshCurrentFloor(true);
// } finally {
//   // delete each OpenCV Mat exactly once
//   matsToDelete.forEach((m) => {
//     try { m.delete(); } catch {}
//   });
//   matsToDelete.clear();
// }
// }

// Replace the old OpenCV ECC function with this
// async function autoAlignViaBackend() {
//   const idx = parseInt(floorSelect.value || "0", 10) || 0;
//   const overlay = overlays[idx];
  
//   if (!currentIfcFile || !overlay.originalPdfFile) {
//     setStatus("Auto-align requires the original IFC and PDF files.");
//     return;
//   }

//   setStatus("Auto-align: Sending files to backend for vector alignment...");
  
//   const formData = new FormData();
//   formData.append("ifc", currentIfcFile);     // Keep a global ref to the loaded IFC File
//   formData.append("pdf", overlay.originalPdfFile); // Store the original PDF file on the overlay object

//   try {
//     const response = await fetch("http://localhost:8000/api/align", {
//       method: "POST",
//       body: formData
//     });

//     const data = await response.json();
//     if (!data.success) throw new Error("Alignment failed on backend.");

//     const { tl, tr, br, bl } = data.corners;

//     // Create 3D vectors for the corners (ignoring Y elevation for a moment)
//     const TL = new THREE.Vector3(tl[0], 0, tl[1]);
//     const TR = new THREE.Vector3(tr[0], 0, tr[1]);
//     const BR = new THREE.Vector3(br[0], 0, br[1]);

//     // 1. Reset arbitrary UI flips and aspect ratio guesses
//     setPlanFlipX(overlay, false);
//     overlay.rotate90 = false;

//     // 2. Rebuild the geometry natively (1 pixel = 1 Three.js unit)
//     const tex = overlay.material.map;
//     const imgW = Number(tex?.image?.width ?? 1);
//     const imgH = Number(tex?.image?.height ?? 1);
    
//     overlay.planeW = imgW;
//     overlay.planeD = imgH;

//     overlay.mesh.geometry.dispose();
//     const newGeom = new THREE.PlaneGeometry(imgW, imgH, 1, 1);
//     newGeom.rotateX(-Math.PI / 2);
//     overlay.mesh.geometry = newGeom;

//     // 3. Position the center of the image at the mathematical center
//     const center = new THREE.Vector3().addVectors(TL, BR).multiplyScalar(0.5);
//     overlay.mesh.position.set(center.x, overlay.basePos.y, center.z); // Keep existing storey height!

//     // 4. Scale it to match the exact world distance of the top edge
//     const worldWidth = TL.distanceTo(TR);
//     const scale = worldWidth / imgW;
//     overlay.mesh.scale.set(scale, 1, scale);

//     // 5. Rotate it along the top edge vector
//     const dir = new THREE.Vector3().subVectors(TR, TL).normalize();
//     const angle = Math.atan2(-dir.z, dir.x); // In Three.js XZ plane, this gives the Y rotation
//     overlay.mesh.rotation.y = angle;

//     // Save it so it persists on reload
//     saveOverlayTransform(idx);
//     refreshCurrentFloor(true);
    
//     setStatus("Auto-align complete ✅ (Perfect Match)");

//   } catch (error) {
//     console.error(error);
//     setStatus("Auto-align failed. Check backend logs.");
//   }
// }



async function autoAlignViaBackend() {
  const idx = parseInt(floorSelect.value || "0", 10) || 0;
  const overlay = overlays[idx];

  if (!currentIfcFile || !overlay.originalPdfFile || !currentModel) {
    setStatus("Auto-align requires the original IFC and PDF files.");
    return;
  }

  setStatus("Auto-align: Sending files to backend for vector alignment...");

  const formData = new FormData();
  formData.append("ifc", currentIfcFile);
  formData.append("pdf", overlay.originalPdfFile);

  try {
    const response = await fetch("http://localhost:8000/api/align", {
      method: "POST",
      body: formData,
    });

    const data = await response.json();
    if (!data.success) throw new Error("Alignment failed on backend.");

    const { tl, tr, br, bl } = data.corners;

    // -----------------------------
    // 0) Get model center in VIEWER coordinates (XZ)
    // -----------------------------
    const bb = new THREE.Box3().setFromObject(currentModel.object);
    const mc = new THREE.Vector3();
    bb.getCenter(mc);
    // mc.x, mc.z are what we want

    // -----------------------------
    // 1) Convert backend corners -> Three.js ground plane coords (XZ)
    // Backend returns [x, z] in IFC 2D (actually IFC X,Y on plan)
    // We map: IFC X -> Three X, IFC Y -> Three Z (with a sign flip often needed)
    //
    // Use the same convention you previously had commented:
    // IFC Y -> WebGL -Z
    // -----------------------------
    const TL0 = new THREE.Vector3(tl[0], 0, -tl[1]);
    const TR0 = new THREE.Vector3(tr[0], 0, -tr[1]);
    const BR0 = new THREE.Vector3(br[0], 0, -br[1]);
    const BL0 = new THREE.Vector3(bl[0], 0, -bl[1]);

    // -----------------------------
    // 2) Apply coordinationMatrix if available.
    // Some loaders store points in a "coordinated" local space.
    // We'll try BOTH directions (C and C^-1) and pick the one
    // that places the plan center closer to the model center.
    // -----------------------------
    

    function planCenterXZ(a: THREE.Vector3, c: THREE.Vector3, b?: THREE.Vector3, d?: THREE.Vector3) {
      // average of 4 corners if given; else diagonal midpoint
      if (b && d) {
        return new THREE.Vector3(
          (a.x + b.x + c.x + d.x) * 0.25,
          0,
          (a.z + b.z + c.z + d.z) * 0.25
        );
      }
      return new THREE.Vector3((a.x + c.x) * 0.5, 0, (a.z + c.z) * 0.5);
    }

    function distXZ(p: THREE.Vector3, q: THREE.Vector3) {
      const dx = p.x - q.x;
      const dz = p.z - q.z;
      return Math.hypot(dx, dz);
    }

    let TL = TL0.clone();
    let TR = TR0.clone();
    let BR = BR0.clone();
    let BL = BL0.clone();
const fg: any = currentModel.object;
if (fg?.coordinationMatrix) {
  const C = new THREE.Matrix4().fromArray(fg.coordinationMatrix);
  TL.applyMatrix4(C);
  TR.applyMatrix4(C);
  BR.applyMatrix4(C);
  BL.applyMatrix4(C);
}
    if (fg && fg.coordinationMatrix) {
      const C = new THREE.Matrix4().fromArray(fg.coordinationMatrix);

      // candidate A: apply C
      const TL_A = TL0.clone().applyMatrix4(C);
      const TR_A = TR0.clone().applyMatrix4(C);
      const BR_A = BR0.clone().applyMatrix4(C);
      const BL_A = BL0.clone().applyMatrix4(C);
      const pcA = planCenterXZ(TL_A, BR_A, TR_A, BL_A);
      const dA = distXZ(pcA, mc);

      // candidate B: apply C^-1
      const Cinv = C.clone().invert();
      const TL_B = TL0.clone().applyMatrix4(Cinv);
      const TR_B = TR0.clone().applyMatrix4(Cinv);
      const BR_B = BR0.clone().applyMatrix4(Cinv);
      const BL_B = BL0.clone().applyMatrix4(Cinv);
      const pcB = planCenterXZ(TL_B, BR_B, TR_B, BL_B);
      const dB = distXZ(pcB, mc);

      // pick the closer
      if (dA <= dB) {
        TL = TL_A; TR = TR_A; BR = BR_A; BL = BL_A;
      } else {
        TL = TL_B; TR = TR_B; BR = BR_B; BL = BL_B;
      }
    }

    // -----------------------------
    // 3) RECENTER: bring plan onto the model center in viewer space.
    // This "undoes" global translation so the overlay doesn't land far away.
    // -----------------------------
    const pc = planCenterXZ(TL, BR, TR, BL);
    const dx = mc.x - pc.x;
    const dz = mc.z - pc.z;

    TL.x += dx; TL.z += dz;
    TR.x += dx; TR.z += dz;
    BR.x += dx; BR.z += dz;
    BL.x += dx; BL.z += dz;

    // -----------------------------
    // 4) Reset arbitrary UI flips/rotations (so we don't fight old state)
    // -----------------------------
    setPlanFlipX(overlay, false);
    overlay.rotate90 = false;

    // -----------------------------
    // 5) Rebuild geometry (keep your approach: geometry in pixel units + scale)
    // -----------------------------
    const tex = overlay.material.map;
    const imgW = Number((tex?.image as any)?.width ?? 1);
    const imgH = Number((tex?.image as any)?.height ?? 1);

    overlay.planeW = imgW;
    overlay.planeD = imgH;

    overlay.mesh.geometry.dispose();
    const newGeom = new THREE.PlaneGeometry(imgW, imgH, 1, 1);
    newGeom.rotateX(-Math.PI / 2);
    newGeom.computeBoundingBox();
    newGeom.computeBoundingSphere();
    overlay.mesh.geometry = newGeom;

    // -----------------------------
    // 6) Apply rigid transform from corners:
    // - position = center (XZ), Y = storey elevation
    // - scale = (world width) / imgW
    // - rotation = yaw from TL->TR direction
    // -----------------------------
    const center = new THREE.Vector3().addVectors(TL, BR).multiplyScalar(0.5);

    const worldWidth = TL.distanceTo(TR);
    const scale = worldWidth / Math.max(1e-9, imgW);

    overlay.mesh.scale.set(scale, scale, scale);

    // keep plane at correct storey height
    overlay.mesh.position.set(center.x, overlay.worldY - 0.01, center.z);

    // yaw rotation
    overlay.mesh.rotation.set(0, 0, 0);
    const dir = new THREE.Vector3().subVectors(TR, TL).normalize();
    const angle = Math.atan2(dir.z, dir.x); // yaw in XZ
    overlay.mesh.rotation.y = angle;

    console.log("Applied Backend Align (recentering enabled)", {
      modelCenter: { x: mc.x, z: mc.z },
      planCenterBefore: { x: pc.x, z: pc.z },
      delta: { dx, dz },
      centerAfter: { x: center.x, z: center.z },
      scale,
      angle,
      usedCoordinationMatrix: !!(fg && fg.coordinationMatrix),
    });

    saveOverlayTransform(idx);
    refreshCurrentFloor(true);
    // after applying backend align + refreshCurrentFloor(true)
    focusCameraToModel(2.0);
    setStatus("Auto-align complete ✅ (recentered to model)");
  
  } catch (error) {
    console.error(error);
    setStatus("Auto-align failed. Check backend logs.");
  }
}





  // ------------------------
  // UI events
  // ------------------------
// autoAlignBtn.addEventListener("click", () =>
//   autoAlignCurrentFloorECC().catch(async (e) => {
//     const cv = await getCV().catch(() => null);
//     const msg = cv ? cvErrorToString(cv, e) : String(e);
//     console.error("Auto-align failed:", msg, e);
//     setStatus(`Auto-align error: ${msg}`);
//   })
// );

  // ------------------------
  // UI events
  // ------------------------
autoAlignBtn.addEventListener("click", () => {
  autoAlignViaBackend();
});

  opacitySlider.addEventListener("input", () => {
    const value = parseFloat(opacitySlider.value);
    overlays.forEach((o) => (o.material.opacity = value));
  });

  floorSelect.addEventListener("change", () => {
    clearHighlight();
    refreshCurrentFloor(true);
  });

  focusPlanBtn.addEventListener("click", () => {
    if (!lastBBox) return;
    const idx = parseInt(floorSelect.value || "0", 10);
    const overlay = overlays[idx];
    const matsToDelete = new Set<any>();
const track = (m: any) => {
  if (m && typeof m.delete === "function") matsToDelete.add(m);
  return m;
};
    if (!overlay) return;

    const size = new THREE.Vector3();
    const center = new THREE.Vector3();
    lastBBox.getSize(size);
    lastBBox.getCenter(center);

    focusCameraToPlan(center, size.x || 10, size.z || 10, overlay.worldY);
  });

  calibrateBtn.addEventListener("click", () => beginCalibration());

  resetPlanBtn.addEventListener("click", () => {
    if (calibStage !== "idle") {
      endCalibration(true);
      calibStage = "idle";
      calibModelPts = [];
      calibPlanLocalPts = [];
      clearMarkers();
      setControlsEnabled(true);
      setStatus("Calibration cancelled ✅");
      refreshCurrentFloor(true);
      return;
    }

    if (overlays.length === 0) return;
    const idx = parseInt(floorSelect.value || "0", 10);
    resetOverlayTransform(idx);
    setStatus("Reset ✅  (saved calibration cleared + snapped back)");
  });

  // ESC cancel
  window.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && calibStage !== "idle") {
      endCalibration(true);
      calibStage = "idle";
      calibModelPts = [];
      calibPlanLocalPts = [];
      clearMarkers();
      setControlsEnabled(true);
      setStatus("Calibration cancelled.");
      refreshCurrentFloor(true);
    }
  });

  // Keyboard shortcuts (+/- opacity, F focus, C calibrate, R reset, T toggle tools)
  window.addEventListener("keydown", (e) => {
    // Toggle tools (T) — don’t steal keystrokes while typing in inputs
    if (e.key === "t" || e.key === "T") {
      const ae = document.activeElement as HTMLElement | null;
      const typing =
        ae && (ae.tagName === "INPUT" || ae.tagName === "TEXTAREA" || (ae as any).isContentEditable);
      if (!typing) {
        ensureToolsPanel();
        setToolsCollapsed(!toolsCollapsed);
      }
      return;
    }

    if (e.key === "f" || e.key === "F") focusPlanBtn.click();
    if (e.key === "c" || e.key === "C") calibrateBtn.click();
    if (e.key === "r" || e.key === "R") resetPlanBtn.click();

    if (e.key === "+" || e.key === "=") {
      opacitySlider.value = String(Math.min(1, parseFloat(opacitySlider.value) + 0.05));
      opacitySlider.dispatchEvent(new Event("input"));
    }
    if (e.key === "-" || e.key === "_") {
      opacitySlider.value = String(Math.max(0, parseFloat(opacitySlider.value) - 0.05));
      opacitySlider.dispatchEvent(new Event("input"));
    }
  });

  // Pointer handler on CANVAS
  const canvas = world.renderer!.three.domElement as HTMLCanvasElement;
  canvas.style.touchAction = "none";

  canvas.addEventListener(
    "pointerdown",
    (ev) => {
      // If not calibrating: click symbols
      if (calibStage === "idle") {
        const idx = parseInt(floorSelect.value || "0", 10);
        const overlay = overlays[idx];
        if (overlay) {
          const hits = pickOn([overlay.worldSymbols], ev);
          if (hits.length) {
            let obj: any = hits[0].object;
            while (obj && (!obj.userData || obj.userData.kind !== "yoloSymbol") && obj.parent) obj = obj.parent;
            if (obj?.userData?.kind === "yoloSymbol") {
              ev.preventDefault();
              ev.stopPropagation();
              setHighlight(obj);
              showSymbolProps(obj);
              return;
            }
          }
        }
      }

      if (calibStage === "idle" || !currentModel || overlays.length === 0) return;

      ev.preventDefault();
      ev.stopPropagation();

      const overlayIdx = calibOverlayIdx ?? parseInt(floorSelect.value || "0", 10);
      const overlay = overlays[overlayIdx];
      const pickY = overlay.worldY ?? 0;

      // MODEL POINTS (using horizontal plane)
      // MODEL POINTS (raycast actual IFC first, fallback to horizontal plane)
if (calibStage === "pickModel1" || calibStage === "pickModel2" || calibStage === "pickModel3") {
  const point = intersectHorizontalPlane(ev, pickY); // or your safePickModelPoint if you have it
  if (!point) {
    setStatus("No intersection. Use top-down and click where the model is.");
    return;
  }

  calibModelPts.push(point);
  addMarker(point);

  if (calibStage === "pickModel1") {
    calibStage = "pickModel2";
    setStatus("Calibration: click 2nd of 3 points on 3D model");
  } else if (calibStage === "pickModel2") {
    calibStage = "pickModel3";
    setStatus("Calibration: click 3rd of 3 points on 3D model (not collinear)");
  } else {
    calibStage = "pickPlan1";
    enterPlanPickMode();
    setStatus("Calibration: click 1st of 3 matching points on the PLAN image");
  }
  return;
}

      // PLAN POINTS
      if (calibStage === "pickPlan1" || calibStage === "pickPlan2" || calibStage === "pickPlan3") {
  const hits = pickOn([overlay.mesh], ev);
  if (!hits.length) {
    setStatus("No plan hit. Click directly on the PNG plane.");
    return;
  }

  const local = overlay.mesh.worldToLocal(hits[0].point.clone());
  calibPlanLocalPts.push(local);
  addMarker(hits[0].point.clone());

  if (calibStage === "pickPlan1") {
    calibStage = "pickPlan2";
    setStatus("Calibration: click 2nd of 3 matching points on the PLAN image");
    return;
  }

  if (calibStage === "pickPlan2") {
    calibStage = "pickPlan3";
    setStatus("Calibration: click 3rd of 3 matching points on the PLAN image (not collinear)");
    return;
  }

  // ---- SOLVE (3 point) ----
  const overlayIdx2 = calibOverlayIdx ?? parseInt(floorSelect.value || "0", 10);
  const overlay2 = overlays[overlayIdx2];

  // model points in XZ
  const model2 = calibModelPts.map((p) => new THREE.Vector2(p.x, p.z));

  // collinearity check (fail-proof guard)
  const areaM = Math.abs(triArea2(model2[0], model2[1], model2[2]));
  const plan2raw = calibPlanLocalPts.map((p) => new THREE.Vector2(p.x, p.z));
  const areaP = Math.abs(triArea2(plan2raw[0], plan2raw[1], plan2raw[2]));

  const spanM =
    model2[1].distanceTo(model2[0]) +
    model2[2].distanceTo(model2[1]) +
    model2[0].distanceTo(model2[2]);

  const spanP =
    plan2raw[1].distanceTo(plan2raw[0]) +
    plan2raw[2].distanceTo(plan2raw[1]) +
    plan2raw[0].distanceTo(plan2raw[2]);

  if (areaM < 1e-6 * spanM * spanM || areaP < 1e-6 * spanP * spanP) {
    setStatus("Calibration failed: points are too collinear. Pick 3 points that form a triangle.");
    // stay in plan picking so user can retry easily
    calibStage = "pickPlan1";
    calibModelPts = [];
    calibPlanLocalPts = [];
    clearMarkers();
    enterPlanPickMode();
    return;
  }

  const res = solve3ptWithAutoFlip(overlay2, calibPlanLocalPts, model2);
  if (!res) {
    setStatus("Calibration failed: invalid geometry. Try different points.");
    calibStage = "idle";
    setControlsEnabled(true);
    endCalibration(true);
    return;
  }

  // apply auto flip if needed
  if (res.shouldToggleFlipX) {
    setPlanFlipX(overlay2, !getPlanFlipX(overlay2));
  }

  applyCalibrationToOverlay(overlay2, res.sol);
  saveOverlayTransform(overlayIdx2);

  setControlsEnabled(true);
  setStatus("Calibration saved ✅ (3-point)");
  endCalibration(false);

  calibStage = "idle";
  calibModelPts = [];
  calibPlanLocalPts = [];
  refreshCurrentFloor(true);
  return;
}
    },
    { capture: true }
  );

  // File input
  fileInput.addEventListener("change", async () => {
    if (!fileInput.files || fileInput.files.length === 0) return;
    const files = Array.from(fileInput.files);
    await handleFiles(files);
  });

  // Drag & drop anywhere
  window.addEventListener("dragover", (e) => e.preventDefault());
  window.addEventListener("drop", async (e) => {
    e.preventDefault();
    const files = Array.from(e.dataTransfer?.files ?? []);
    if (files.length === 0) return;
    await handleFiles(files);
  });

  // ------------------------
  // Handle files: IFC + (PNG and/or PDF) + optional detection JSON
  // ------------------------
  async function handleFiles(files: File[]) {
    const ifc = files.find((f) => f.name.toLowerCase().endsWith(".ifc"));

    const pdfs = files.filter((f) => f.type === "application/pdf" || f.name.toLowerCase().endsWith(".pdf"));
    let pngs = files.filter((f) => f.type === "image/png" || f.name.toLowerCase().endsWith(".png"));

    const jsonFiles = files.filter((f) => f.name.toLowerCase().endsWith(".json"));

    if (!ifc) {
      alert("Drop/select an .IFC file too.");
      return;
    }

    const originalPdfMap = new Map<string, File>(); // <--- ADDED

    // PDF → PNG
    if (pdfs.length) {
      setStatus(`Rendering ${pdfs.length} PDF(s) to PNG...`);
      for (const pdf of pdfs) {
        const pages = await renderPdfToPngFiles(pdf, 2);
        pngs = pngs.concat(pages);

        // <--- ADDED: Map generated PNGs back to the original PDF
        for (const pagePng of pages) {
          originalPdfMap.set(pagePng.name, pdf);
        }
      }
    }

    if (pngs.length === 0) {
      alert("Drop/select at least one PNG (or PDF) floor plan too.");
      return;
    }

    pngs.sort((a, b) => a.name.localeCompare(b.name));

    calibStage = "idle";
    calibModelPts = [];
    calibPlanLocalPts = [];
    clearMarkers();
    setControlsEnabled(true);

    // Parse imported detection JSONs into map: imageName -> detections[]
    const importedDetections = new Map<string, Detection[]>();
    for (const jf of jsonFiles) {
      try {
        const txt = await jf.text();
        const j = JSON.parse(txt);

        // support {image, detections} format
        if (j?.image && Array.isArray(j?.detections)) {
          importedDetections.set(String(j.image), j.detections as Detection[]);
        }
      } catch (e) {
        console.warn("Bad detection JSON:", jf.name, e);
      }
    }

    await loadEverything(ifc, pngs, originalPdfMap, importedDetections); // <--- CHANGED
  }

  // Initial UI state
  ensureToolsPanel();
  updateOverlayButtons();
  setStatus("Ready: drop an IFC + PNG plans and/or PDFs. (Optional: drop detections JSON too.)");
}

boot().catch((err) => {
  console.error("BOOT FAILED:", err);
  alert("Boot failed - open DevTools Console for details.");
});
