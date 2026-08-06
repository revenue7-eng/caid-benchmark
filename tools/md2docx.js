const {
  Document, Packer, Paragraph, TextRun, HeadingLevel,
  Table, TableRow, TableCell, WidthType, ShadingType, BorderStyle
} = require('docx');
const fs = require('fs');

const FONT = 'Calibri';
const MONO = 'Consolas';
const ACCENT = '000000';
const GREY = '000000';
const CODEBG = 'F2F4F8';

// inline tokenizer: **bold** and `code`
function inline(text, base = {}) {
  const out = [];
  const re = /(\*\*[^*]+\*\*|`[^`]+`)/g;
  let last = 0, m;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) out.push(mk(text.slice(last, m.index), base));
    const tok = m[0];
    if (tok.startsWith('**')) out.push(mk(tok.slice(2, -2), { ...base, bold: true }));
    else out.push(mk(tok.slice(1, -1), { ...base, mono: true }));
    last = m.index + tok.length;
  }
  if (last < text.length) out.push(mk(text.slice(last), base));
  return out.length ? out : [mk('', base)];
}

function mk(t, o = {}) {
  return new TextRun({
    text: t,
    bold: !!o.bold,
    italics: !!o.italics,
    font: o.mono ? MONO : FONT,
    size: o.size || (o.mono ? 20 : 22),
    color: o.color,
    shading: o.mono ? { type: ShadingType.CLEAR, fill: CODEBG } : undefined
  });
}

function splitCells(line) {
  return line.trim().replace(/^\|/, '').replace(/\|$/, '').split('|').map(c => c.trim());
}

function convert(md, link) {
  const lines = md.split('\n');
  const ch = [];
  let i = 0, titleDone = false;

  while (i < lines.length) {
    const line = lines[i];

    // fenced code
    if (/^```/.test(line)) {
      i++;
      const buf = [];
      while (i < lines.length && !/^```/.test(lines[i])) buf.push(lines[i++]);
      i++;
      buf.forEach((b, k) => ch.push(new Paragraph({
        children: [new TextRun({ text: b || ' ', font: MONO, size: 19 })],
        spacing: { before: k === 0 ? 100 : 0, after: k === buf.length - 1 ? 160 : 0, line: 240 },
        shading: { type: ShadingType.CLEAR, fill: CODEBG },
        indent: { left: 220 }
      })));
      continue;
    }

    // table
    if (/^\|/.test(line) && i + 1 < lines.length && /^\|[\s:|-]+\|$/.test(lines[i + 1].trim())) {
      const header = splitCells(line);
      i += 2;
      const body = [];
      while (i < lines.length && /^\|/.test(lines[i])) body.push(splitCells(lines[i++]));
      const n = header.length;
      const total = 9000;
      const widths = n === 2 ? [3200, 5800] : Array(n).fill(Math.floor(total / n));
      widths[widths.length - 1] += total - widths.reduce((a, b) => a + b, 0);
      const hasHeader = header.some(h => h !== '');
      const rows = [];
      if (hasHeader) rows.push(header);
      body.forEach(r => rows.push(r));
      ch.push(new Table({
        columnWidths: widths,
        width: { size: total, type: WidthType.DXA },
        rows: rows.map((r, ri) => new TableRow({
          tableHeader: hasHeader && ri === 0,
          children: r.map((c, ci) => new TableCell({
            width: { size: widths[ci], type: WidthType.DXA },
            shading: (hasHeader && ri === 0) ? { type: ShadingType.CLEAR, fill: 'EDF1F8' } : undefined,
            margins: { top: 70, bottom: 70, left: 110, right: 110 },
            children: [new Paragraph({
              children: inline(c, { size: 20, bold: hasHeader && ri === 0 }),
              spacing: { after: 0 }
            })]
          }))
        }))
      }));
      ch.push(new Paragraph({ text: '', spacing: { after: 180 } }));
      continue;
    }

    // headings
    let m;
    if ((m = line.match(/^#\s+(.*)/)) && !titleDone) {
      titleDone = true;
      ch.push(new Paragraph({
        children: [new TextRun({ text: m[1], font: FONT, size: 44, bold: true, color: ACCENT })],
        spacing: { after: 120 }
      }));
      // subtitle = next non-empty line if it is plain prose
      let j = i + 1;
      while (j < lines.length && lines[j].trim() === '') j++;
      if (j < lines.length && !/^[#|`\-]/.test(lines[j])) {
        ch.push(new Paragraph({
          children: inline(lines[j], { color: GREY, italics: true }),
          spacing: { after: link ? 140 : 240 }
        }));
        if (link) ch.push(new Paragraph({
          children: [mk(link)],
          spacing: { after: 240 }
        }));
        i = j + 1;
        continue;
      }
      if (link) ch.push(new Paragraph({
        children: [mk(link)],
        spacing: { after: 240 }
      }));
      i++;
      continue;
    }
    if ((m = line.match(/^##\s+(.*)/))) {
      ch.push(new Paragraph({
        children: [new TextRun({ text: m[1], font: FONT, size: 30, bold: true, color: ACCENT })],
        heading: HeadingLevel.HEADING_1,
        spacing: { before: 380, after: 150 }
      }));
      i++; continue;
    }
    if ((m = line.match(/^###\s+(.*)/))) {
      ch.push(new Paragraph({
        children: [new TextRun({ text: m[1], font: FONT, size: 24, bold: true, color: ACCENT })],
        heading: HeadingLevel.HEADING_2,
        spacing: { before: 260, after: 110 }
      }));
      i++; continue;
    }

    // blockquote
    if ((m = line.match(/^>\s?(.*)/))) {
      ch.push(new Paragraph({
        children: inline(m[1], { italics: true, color: GREY }),
        spacing: { after: 120, line: 264 },
        indent: { left: 340 }
      }));
      i++; continue;
    }

    // horizontal rule
    if (/^---\s*$/.test(line)) {
      ch.push(new Paragraph({ text: '', spacing: { before: 40, after: 80 } }));
      i++; continue;
    }

    // task list
    if ((m = line.match(/^-\s+\[\s*\]\s+(.*)/))) {
      ch.push(new Paragraph({
        children: [mk('\u2610  '), ...inline(m[1])],
        spacing: { after: 60, line: 264 },
        indent: { left: 300 }
      }));
      i++; continue;
    }
    // bullet
    if ((m = line.match(/^-\s+(.*)/))) {
      ch.push(new Paragraph({
        children: inline(m[1]),
        bullet: { level: 0 },
        spacing: { after: 70, line: 264 }
      }));
      i++; continue;
    }

    if (line.trim() === '') { i++; continue; }

    ch.push(new Paragraph({
      children: inline(line),
      spacing: { after: 150, line: 276 }
    }));
    i++;
  }
  return ch;
}

function build(children) {
  return new Document({
    styles: { default: { document: { run: { font: FONT, size: 22 } } } },
    sections: [{
      properties: { page: { margin: { top: 660, bottom: 620, left: 567, right: 567 } } },
      children
    }]
  });
}

(async () => {
  const argv = process.argv.slice(2);
  let link = null;
  const li = argv.indexOf('--link');
  if (li !== -1) { link = argv[li + 1]; argv.splice(li, 2); }
  const [src, dst] = argv;
  const md = fs.readFileSync(src, 'utf8');
  fs.writeFileSync(dst, await Packer.toBuffer(build(convert(md, link))));
  console.log('written', dst, link ? '(link: ' + link + ')' : '');
})();
