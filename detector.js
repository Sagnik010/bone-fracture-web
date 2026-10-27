// In-browser YOLOv5 detector. Runs best.onnx with onnxruntime-web (WASM). No server.
// Pipeline: letterbox -> CHW float tensor -> ONNX (3 raw heads) -> decode -> class-aware NMS.
(function () {
  const Detector = {
    session: null,
    meta: null,

    async load(onProgress) {
      this.meta = await (await fetch("model/model_meta.json")).json();
      const resp = await fetch("model/best.onnx");
      if (!resp.ok) throw new Error("Could not download the model file (" + resp.status + ")");
      const total = +resp.headers.get("Content-Length") || 0;
      const reader = resp.body.getReader();
      const chunks = []; let got = 0;
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        chunks.push(value); got += value.length;
        if (onProgress && total) onProgress(got / total);
      }
      const buf = new Uint8Array(got); let o = 0;
      for (const c of chunks) { buf.set(c, o); o += c.length; }

      if (!window.ort) throw new Error("The ONNX runtime failed to load (check your internet connection).");
      ort.env.wasm.wasmPaths = window.ORT_BASE;
      ort.env.wasm.numThreads = 1; // multi-threading needs cross-origin isolation headers
      this.session = await ort.InferenceSession.create(buf, { executionProviders: ["wasm"] });
    },

    // img: HTMLImageElement. Returns [{class_id,label,confidence,box:[x1,y1,x2,y2]}] in original pixels.
    async detect(img, minConf = 0.05, iouThr = 0.45) {
      const S = this.meta.imgsz, W = img.naturalWidth, H = img.naturalHeight;
      const r = Math.min(S / H, S / W);
      const nw = Math.round(W * r), nh = Math.round(H * r);
      const px = (S - nw) / 2, py = (S - nh) / 2;

      const cv = document.createElement("canvas"); cv.width = cv.height = S;
      const cx = cv.getContext("2d", { willReadFrequently: true });
      cx.fillStyle = "rgb(114,114,114)"; cx.fillRect(0, 0, S, S);
      cx.imageSmoothingQuality = "high";
      cx.drawImage(img, px, py, nw, nh);
      const rgba = cx.getImageData(0, 0, S, S).data;

      const plane = S * S, input = new Float32Array(3 * plane);
      for (let i = 0; i < plane; i++) {
        input[i] = rgba[i * 4] / 255;
        input[plane + i] = rgba[i * 4 + 1] / 255;
        input[2 * plane + i] = rgba[i * 4 + 2] / 255;
      }
      const out = await this.session.run({ images: new ort.Tensor("float32", input, [1, 3, S, S]) });

      const { strides, anchors_px: anchors, nc, names, outputs } = this.meta;
      const K = 5 + nc, sig = (v) => 1 / (1 + Math.exp(-v));
      const cand = [];
      outputs.forEach((name, i) => {
        const data = out[name].data, [, , ny, nx] = out[name].dims, st = strides[i], area = ny * nx;
        for (let a = 0; a < 3; a++) {
          const [aw, ah] = anchors[i][a], base = a * K * area;
          for (let y = 0; y < ny; y++) for (let x = 0; x < nx; x++) {
            const p = y * nx + x, obj = sig(data[base + 4 * area + p]);
            if (obj < minConf) continue;
            let best = 0, bs = 0;
            for (let c = 0; c < nc; c++) {
              const s = obj * sig(data[base + (5 + c) * area + p]);
              if (s > bs) { bs = s; best = c; }
            }
            if (bs < minConf) continue;
            const bx = (sig(data[base + p]) * 2 - 0.5 + x) * st;
            const by = (sig(data[base + area + p]) * 2 - 0.5 + y) * st;
            const bw = (sig(data[base + 2 * area + p]) * 2) ** 2 * aw;
            const bh = (sig(data[base + 3 * area + p]) * 2) ** 2 * ah;
            cand.push({
              class_id: best, score: bs,
              x1: (bx - bw / 2 - px) / r, y1: (by - bh / 2 - py) / r,
              x2: (bx + bw / 2 - px) / r, y2: (by + bh / 2 - py) / r,
            });
          }
        }
      });

      // class-aware NMS
      cand.sort((a, b) => b.score - a.score);
      const kept = [];
      for (const c of cand) {
        let ok = true;
        for (const k of kept) if (k.class_id === c.class_id && iou(k, c) > iouThr) { ok = false; break; }
        if (ok) kept.push(c);
      }
      const clamp = (v, m) => Math.max(0, Math.min(m, v));
      return kept.map((k) => ({
        class_id: k.class_id, label: names[k.class_id], confidence: +k.score.toFixed(4),
        box: [clamp(k.x1, W), clamp(k.y1, H), clamp(k.x2, W), clamp(k.y2, H)].map((v) => +v.toFixed(1)),
      }));
    },
  };

  function iou(a, b) {
    const iw = Math.max(0, Math.min(a.x2, b.x2) - Math.max(a.x1, b.x1));
    const ih = Math.max(0, Math.min(a.y2, b.y2) - Math.max(a.y1, b.y1));
    const inter = iw * ih;
    return inter / ((a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter + 1e-9);
  }

  window.Detector = Detector;
})();
