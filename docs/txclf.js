// txclf in the browser: the same maths as the Python pipeline, no server.
// clean -> tokens -> stop words -> 1..n-grams -> tf * idf -> l2 -> W x + b -> softmax
// W is the LogReg weights folded through the SVD (see src/txclf/export.py).

export function cleanDescription(text) {
  return String(text ?? "")
    .toLowerCase()
    .replace(/[^a-zA-Z0-9 ]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

export function loadModel(json) {
  const vocab = new Map(json.vocabulary.map((t, i) => [t, i]));
  const [nClasses, nTerms] = json.weights_shape;
  const bin = atob(json.weights_int16_b64);
  const view = new DataView(new ArrayBuffer(bin.length));
  for (let i = 0; i < bin.length; i++) view.setUint8(i, bin.charCodeAt(i));
  const W = new Float32Array(nClasses * nTerms);
  for (let i = 0; i < W.length; i++) W[i] = view.getInt16(i * 2, true) * json.weights_scale;
  return {
    vocab,
    idf: Float64Array.from(json.idf),
    stop: new Set(json.stop_words),
    ngramMax: json.ngram_max,
    categories: json.categories,
    intercept: Float64Array.from(json.intercept),
    W,
    nClasses,
    nTerms,
    metrics: json.metrics || {},
  };
}

export function tfidf(model, text) {
  const tokens = (cleanDescription(text).match(/\b\w\w+\b/g) || []).filter(
    (t) => !model.stop.has(t),
  );
  const counts = new Map();
  for (let n = 1; n <= model.ngramMax; n++) {
    for (let i = 0; i + n <= tokens.length; i++) {
      const j = model.vocab.get(tokens.slice(i, i + n).join(" "));
      if (j !== undefined) counts.set(j, (counts.get(j) || 0) + 1);
    }
  }
  let norm = 0;
  const vec = [];
  for (const [j, c] of counts) {
    const v = c * model.idf[j];
    vec.push([j, v]);
    norm += v * v;
  }
  norm = Math.sqrt(norm) || 1;
  return vec.map(([j, v]) => [j, v / norm]);
}

export function predict(model, text, topK = 3) {
  const x = tfidf(model, text);
  const logits = Array.from(model.intercept);
  for (let c = 0; c < model.nClasses; c++) {
    const row = c * model.nTerms;
    for (const [j, v] of x) logits[c] += model.W[row + j] * v;
  }
  const max = Math.max(...logits);
  const exp = logits.map((z) => Math.exp(z - max));
  const sum = exp.reduce((a, b) => a + b, 0);
  const proba = exp.map((e) => e / sum);
  const order = proba.map((p, i) => [p, i]).sort((a, b) => b[0] - a[0]);
  const top = order.slice(0, topK).map(([p, i]) => ({ category: model.categories[i], probability: p }));
  return {
    category: top[0].category,
    probability: top[0].probability,
    top_k: top,
    known_terms: x.length,
    proba,
  };
}
