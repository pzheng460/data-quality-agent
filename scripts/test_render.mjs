#!/usr/bin/env node
/**
 * Test full markdown rendering pipeline (same stack as dashboard).
 * Uses: remark-parse → remark-gfm → remark-math → rehype-katex → HTML
 *
 * Reports KaTeX parse errors (rendered as red error boxes in the browser).
 *
 * Usage:
 *   node scripts/test_render.mjs /tmp/output.jsonl
 */

import { readFileSync } from 'fs';
import { unified } from '../dashboard/node_modules/unified/index.js';
import remarkParse from '../dashboard/node_modules/remark-parse/index.js';
import remarkGfm from '../dashboard/node_modules/remark-gfm/index.js';
import remarkMath from '../dashboard/node_modules/remark-math/index.js';
import remarkRehype from '../dashboard/node_modules/remark-rehype/index.js';
import rehypeKatex from '../dashboard/node_modules/rehype-katex/index.js';
import rehypeStringify from '../dashboard/node_modules/rehype-stringify/index.js';

async function renderMarkdown(text) {
  const result = await unified()
    .use(remarkParse)
    .use(remarkGfm)
    .use(remarkMath)
    .use(remarkRehype)
    .use(rehypeKatex, { throwOnError: false })
    .use(rehypeStringify)
    .process(text);

  return {
    html: String(result),
    warnings: result.messages,
  };
}

const args = process.argv.slice(2);
if (!args[0]) {
  console.log('Usage: node scripts/test_render.mjs <file.jsonl>');
  process.exit(0);
}

// Read JSONL
const { readFileSync: readFile } = await import('fs');
const content = readFile(args[0], 'utf8');
let text = '';
for (const line of content.trim().split('\n')) {
  try {
    text += JSON.parse(line).text || '';
  } catch {
    text += line;
  }
}

console.log(`Input: ${text.length} chars`);

const { html, warnings } = await renderMarkdown(text);

// Count KaTeX errors in output HTML
const katexErrors = (html.match(/class="katex-error"/g) || []).length;
const warnCount = warnings.filter(w => w.fatal || w.message.includes('error')).length;

console.log(`HTML output: ${html.length} chars`);
console.log(`KaTeX render errors: ${katexErrors}`);
console.log(`Remark warnings: ${warnings.length}`);

if (katexErrors > 0) {
  // Extract error details
  const errorRe = /title="([^"]+)" class="katex-error"/g;
  let m;
  let count = 0;
  while ((m = errorRe.exec(html)) !== null && count < 10) {
    console.log(`  ERROR: ${m[1].slice(0, 80)}`);
    count++;
  }
}

for (const w of warnings.slice(0, 5)) {
  console.log(`  WARN: ${w.message}`);
}

if (katexErrors === 0 && warnCount === 0) {
  console.log('\nAll rendering passed!');
} else {
  console.log(`\n${katexErrors} KaTeX errors`);
}
process.exit(katexErrors > 0 ? 1 : 0);
