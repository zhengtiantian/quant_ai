// R.1b — measure the news corpus length distribution so chunking is chosen from
// data rather than assumed. A fixed 512-token window is the wrong default if most
// articles are shorter than one window; it is also wrong if a long tail needs
// splitting. Both questions are answered below.
const c = db.getSiblingDB("quant_data").news_articles_company_matched_v2;

const total = c.estimatedDocumentCount();
print("total docs: " + total);

// Exact, not sampled: if a meaningful share of rows carry no body, dense retrieval
// has to be able to fall back to the title, and that changes the indexing design.
const noContent = c.countDocuments({ $or: [
  { content: { $exists: false } }, { content: null }, { content: "" },
]});
print("docs with no usable content: " + noContent +
      "  (" + (100 * noContent / total).toFixed(2) + "%)");

const SAMPLE = 20000;
const rows = c.aggregate([
  { $sample: { size: SAMPLE } },
  { $project: {
      _id: 0,
      clen: { $strLenCP: { $ifNull: ["$content", ""] } },
      tlen: { $strLenCP: { $ifNull: ["$title", ""] } },
  }},
], { allowDiskUse: true }).toArray();

function pct(arr, p) {
  const s = [...arr].sort((a, b) => a - b);
  return s[Math.min(s.length - 1, Math.floor(p / 100 * s.length))];
}

const clen = rows.map(r => r.clen).filter(v => v > 0);
const tlen = rows.map(r => r.tlen);

print("\nsampled: " + rows.length + " docs, " + clen.length + " with content");
print("\ncontent length (characters)");
[1, 5, 10, 25, 50, 75, 90, 95, 99].forEach(p =>
  print("  p" + String(p).padStart(2) + ": " + String(pct(clen, p)).padStart(8)));
print("  max: " + String(Math.max(...clen)).padStart(8));
print("  mean: " + (clen.reduce((a, b) => a + b, 0) / clen.length).toFixed(0));

print("\ntitle length (characters)");
[50, 90, 99].forEach(p => print("  p" + p + ": " + pct(tlen, p)));

// ~4 chars/token is the standard rough English ratio; nomic-embed-text takes 8192
// tokens, so the question is not "does it fit" but "how many articles are so short
// that chunking them is pure overhead".
const TOK = 4;
print("\nestimated tokens (chars/4) vs candidate chunk sizes");
[256, 512, 1024, 2048].forEach(w => {
  const fits = clen.filter(v => v / TOK <= w).length;
  print("  <= " + String(w).padStart(4) + " tokens: " +
        (100 * fits / clen.length).toFixed(1) + "% of articles fit in one chunk");
});

const est = clen.map(v => Math.max(1, Math.ceil((v / TOK) / 512)));
print("\nchunks per article at a 512-token window: mean " +
      (est.reduce((a, b) => a + b, 0) / est.length).toFixed(2) +
      ", implying ~" + Math.round(total * est.reduce((a, b) => a + b, 0) / est.length / 1000) +
      "K vectors for the full corpus");
