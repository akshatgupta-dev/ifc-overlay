// src/pdf_render.ts
import * as pdfjs from "pdfjs-dist";
import workerUrl from "pdfjs-dist/build/pdf.worker.min.mjs?url";

(pdfjs as any).GlobalWorkerOptions.workerSrc = workerUrl;

function baseName(name: string) {
  const i = name.lastIndexOf(".");
  return i >= 0 ? name.slice(0, i) : name;
}

/**
 * Render each PDF page to a PNG File (so your existing PNG pipeline keeps working).
 * scale=2 gives a good quality/perf balance for architectural sheets.
 */
export async function renderPdfToPngFiles(pdfFile: File, scale = 2): Promise<File[]> {
  const data = new Uint8Array(await pdfFile.arrayBuffer());
  const doc = await (pdfjs as any).getDocument({ data }).promise;

  const out: File[] = [];
  const pdfBase = baseName(pdfFile.name);

  for (let p = 1; p <= doc.numPages; p++) {
    const page = await doc.getPage(p);
    const viewport = page.getViewport({ scale });

    const canvas = document.createElement("canvas");
    canvas.width = Math.ceil(viewport.width);
    canvas.height = Math.ceil(viewport.height);
    const ctx = canvas.getContext("2d")!;

    await page.render({ canvasContext: ctx, viewport }).promise;

    const blob: Blob = await new Promise((resolve, reject) => {
      canvas.toBlob((b) => (b ? resolve(b) : reject(new Error("toBlob failed"))), "image/png");
    });

    const file = new File([blob], `${pdfBase}_p${String(p).padStart(2, "0")}.png`, { type: "image/png" });
    out.push(file);
  }

  return out;
}