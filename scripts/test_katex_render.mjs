#!/usr/bin/env node
/**
 * Test KaTeX rendering on pipeline output.
 *
 * Extracts all math blocks ($...$ and $$...$$) from a JSONL file
 * and tries to render each one with KaTeX. Reports any parse errors.
 *
 * Usage:
 *   node scripts/test_katex_render.mjs /tmp/dq_test/output/stage4_final/shard-00000.jsonl.zst
 *   node scripts/test_katex_render.mjs --text "Some text with $x^2$ and $$E=mc^2$$"
 */

import { createReadStream } from 'fs';
import { createInterface } from 'readline';
import { createRequire } from 'module';

const require = createRequire(import.meta.url);
const katex = require('../dashboard/node_modules/katex');

function extractMath(text) {
  const blocks = [];

  // Extract display math $$...$$
  const displayRe = /\$\$([\s\S]*?)\$\$/g;
  let match;
  while ((match = displayRe.exec(text)) !== null) {
    blocks.push({ type: 'display', latex: match[1], pos: match.index });
  }

  // Remove display math, then extract inline
  const noDisplay = text.replace(displayRegex(), '');
  const inlineRe = /\$([^$\n]+?)\$/g;
  while ((match = inlineRe.exec(noDisplay)) !== null) {
    blocks.push({ type: 'inline', latex: match[1], pos: match.index });
  }

  return blocks;
}

function displayRegex() {
  return /\$\$.*?\$\$/gs;
}

function testKatex(latex, displayMode) {
  try {
    const katexModule = katex;
    katexModule.renderToString(latex, {
      displayMode: displayMode,
      throwOnError: true,
      strict: false,
    });
    return null; // no error
  } catch (e) {
    return e.message;
  }
}

// Main
const args = process.argv.slice(2);
let text = '';

if (args[0] === '--text') {
  text = args.slice(1).join(' ');
} else if (args[0]) {
  // Read from file (plain text or JSONL)
  const fs = require('fs');
  const path = args[0];

  if (path.endsWith('.zst')) {
    console.error('Cannot read .zst directly. Pipe through: zstd -d < file.jsonl.zst | node script.mjs --stdin');
    process.exit(1);
  }

  const content = fs.readFileSync(path, 'utf8');
  // Try as JSONL
  const lines = content.trim().split('\n');
  for (const line of lines) {
    try {
      const doc = JSON.parse(line);
      text += (doc.text || '') + '\n';
    } catch {
      text += line + '\n';
    }
  }
} else {
  console.log('Usage: node scripts/test_katex_render.mjs <file.jsonl> | --text "..."');
  process.exit(0);
}

// Extract and test all math
const blocks = [];
const displayRe = /\$\$([\s\S]*?)\$\$/g;
let m;
while ((m = displayRe.exec(text)) !== null) {
  blocks.push({ type: 'display', latex: m[1].trim(), pos: m.index });
}

const noDisplay = text.replace(/\$\$[\s\S]*?\$\$/g, '');
const inlineRe = /\$([^$\n]+?)\$/g;
while ((m = inlineRe.exec(noDisplay)) !== null) {
  blocks.push({ type: 'inline', latex: m[1].trim(), pos: m.index });
}

console.log(`Found ${blocks.length} math blocks (${blocks.filter(b => b.type === 'display').length} display, ${blocks.filter(b => b.type === 'inline').length} inline)`);

let errors = 0;
for (const block of blocks) {
  const err = testKatex(block.latex, block.type === 'display');
  if (err) {
    errors++;
    const preview = block.latex.slice(0, 60).replace(/\n/g, '\\n');
    console.log(`  ERROR [${block.type}]: ${err}`);
    console.log(`    LaTeX: ${preview}...`);
  }
}

if (errors === 0) {
  console.log('All math blocks render successfully!');
} else {
  console.log(`\n${errors} / ${blocks.length} blocks failed to render.`);
}

process.exit(errors > 0 ? 1 : 0);
