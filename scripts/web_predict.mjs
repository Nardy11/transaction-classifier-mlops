// Run the browser model in Node for the parity test.
// usage: node scripts/web_predict.mjs <model.json> <texts.json>   -> JSON on stdout
import { readFileSync } from "node:fs";
import { pathToFileURL } from "node:url";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const { loadModel, predict } = await import(
  pathToFileURL(resolve(here, "..", "docs", "txclf.js")).href
);

const [modelPath, textsPath] = process.argv.slice(2);
const model = loadModel(JSON.parse(readFileSync(modelPath, "utf8")));
const texts = JSON.parse(readFileSync(textsPath, "utf8"));
const out = texts.map((t) => {
  const r = predict(model, t, 3);
  return { category: r.category, proba: r.proba, known_terms: r.known_terms };
});
process.stdout.write(JSON.stringify(out));
