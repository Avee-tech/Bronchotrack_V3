const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType,
  Table, TableRow, TableCell, WidthType, ShadingType, BorderStyle,
  ImageRun, PageBreak, LevelFormat, convertInchesToTwip,
  Header, Footer, PageNumber, TabStopType, LeaderType,
} = require("docx");

const CONTENT = require("./thesis_content.js");
const { SECTIONS: PSEUDOCODE_SECTIONS } = require("./pseudocode_content.js");

const INK = "1A1A1A";
const MUTED = "555555";
const ACCENT = "1F3864";
const RULE = "BFBFBF";
const CODE_BG = "F4F2EC";
const CODE_EDGE = "D8D3C4";

// A4, matching the uploaded Curtin template exactly (converted from its
// own EMU page/margin values): width 11907 / height 16839 twips,
// margins top/bottom 1418, left 2268 (extra for binding), right 1418.
const PAGE = { width: 11907, height: 16839 };
const MARGIN = { top: 1418, bottom: 1418, left: 2268, right: 1418 };
const BODY_WIDTH = PAGE.width - MARGIN.left - MARGIN.right; // 8221 twips

// ---------------------------------------------------------------------
// Paragraph helpers
// ---------------------------------------------------------------------
function chapterHeading(text) {
  return new Paragraph({ heading: HeadingLevel.HEADING_1, alignment: AlignmentType.JUSTIFIED, children: [new TextRun({ text })] });
}
function frontMatterHeading(text) {
  return new Paragraph({ heading: HeadingLevel.HEADING_1, alignment: AlignmentType.JUSTIFIED, children: [new TextRun({ text })] });
}
function subHeading(text) {
  return new Paragraph({ heading: HeadingLevel.HEADING_2, alignment: AlignmentType.JUSTIFIED, children: [new TextRun({ text })] });
}
function p(text, opts = {}) {
  return new Paragraph({
    alignment: opts.alignment || AlignmentType.JUSTIFIED,
    spacing: { after: 200, line: 300 },
    children: [new TextRun({ text, ...opts })],
  });
}
function bulletPara(text) {
  return new Paragraph({
    alignment: AlignmentType.JUSTIFIED,
    spacing: { after: 120, line: 290 },
    numbering: { reference: "bullets", level: 0 },
    children: [new TextRun({ text })],
  });
}
function caption(text) {
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 100, after: 260 },
    children: [new TextRun({ text, italics: true, size: 20, color: MUTED })],
  });
}
function ruleParagraph() {
  return new Paragraph({
    spacing: { before: 80, after: 260 },
    border: { bottom: { color: RULE, space: 4, style: BorderStyle.SINGLE, size: 6 } },
    children: [new TextRun({ text: "" })],
  });
}
function pageBreak() {
  return new Paragraph({ children: [new PageBreak()] });
}
function centered(text, opts = {}) {
  return new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 80 }, children: [new TextRun({ text, ...opts })] });
}
function rightAligned(text, opts = {}) {
  return new Paragraph({ alignment: AlignmentType.RIGHT, spacing: { after: 0 }, children: [new TextRun({ text, ...opts })] });
}
function letterPara(text) {
  return new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { after: 160, line: 276 }, children: [new TextRun({ text })] });
}
// Manually-authored Table of Contents entry with a dotted leader to a
// right-aligned page number -- renders correctly immediately, in any
// viewer, without requiring a Word "update field" step.
function tocEntry(title, pageNum, opts = {}) {
  const indent = opts.sub ? convertInchesToTwip(0.3) : 0;
  return new Paragraph({
    spacing: { after: opts.sub ? 40 : 100 },
    indent: { left: indent },
    tabStops: [{ type: TabStopType.RIGHT, position: BODY_WIDTH, leader: LeaderType.DOT }],
    children: [
      new TextRun({ text: title, bold: !opts.sub, size: opts.sub ? 21 : 22 }),
      new TextRun({ text: `\t${pageNum}`, size: opts.sub ? 21 : 22 }),
    ],
  });
}

// ---- code block (single-cell shaded table, one paragraph per line) ----
function codeBlockTable(code) {
  const lines = code.split("\n");
  const lineParagraphs = lines.map((line) => new Paragraph({
    spacing: { after: 0, line: 220 },
    children: [new TextRun({ text: line.length ? line : " ", font: "Courier New", size: 15, color: INK })],
  }));
  return new Table({
    width: { size: BODY_WIDTH, type: WidthType.DXA },
    columnWidths: [BODY_WIDTH],
    rows: [new TableRow({ children: [new TableCell({
      width: { size: BODY_WIDTH, type: WidthType.DXA },
      shading: { type: ShadingType.CLEAR, color: "auto", fill: CODE_BG },
      margins: { top: 140, bottom: 140, left: 160, right: 160 },
      borders: {
        top: { style: BorderStyle.SINGLE, size: 4, color: CODE_EDGE },
        bottom: { style: BorderStyle.SINGLE, size: 4, color: CODE_EDGE },
        left: { style: BorderStyle.SINGLE, size: 4, color: CODE_EDGE },
        right: { style: BorderStyle.SINGLE, size: 4, color: CODE_EDGE },
      },
      children: lineParagraphs,
    })] })],
  });
}
function appendixFileName(text) {
  return new Paragraph({ spacing: { before: 300, after: 40 }, children: [new TextRun({ text, bold: true, size: 22, color: ACCENT, font: "Courier New" })] });
}
function appendixDesc(text) {
  return new Paragraph({ spacing: { after: 100 }, children: [new TextRun({ text, italics: true, size: 19, color: MUTED })] });
}

// ---- results / test-suite tables ----
function thCell(text, width) {
  return new TableCell({
    width: { size: width, type: WidthType.DXA },
    shading: { type: ShadingType.CLEAR, color: "auto", fill: ACCENT },
    verticalAlign: "center",
    children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text, bold: true, size: 20, color: "FFFFFF" })] })],
  });
}
function tdCell(text, width, opts = {}) {
  return new TableCell({
    width: { size: width, type: WidthType.DXA },
    verticalAlign: "center",
    children: [new Paragraph({ alignment: opts.left ? AlignmentType.LEFT : AlignmentType.CENTER, children: [new TextRun({ text, size: 20 })] })],
  });
}
function resultsTable() {
  const w = [Math.round(BODY_WIDTH * 0.5), Math.round(BODY_WIDTH * 0.25), Math.round(BODY_WIDTH * 0.25)];
  return new Table({
    width: { size: BODY_WIDTH, type: WidthType.DXA },
    columnWidths: w,
    rows: [
      new TableRow({ children: [thCell("Metric", w[0]), thCell("Before", w[1]), thCell("After", w[2])] }),
      new TableRow({ children: [
        tdCell("Frames with a reported location", w[0], { left: true }),
        tdCell("15.3%", w[1]), tdCell("39.9%", w[2]),
      ] }),
      new TableRow({ children: [
        tdCell("Frames with a fully confirmed (geometry-verified) detection", w[0], { left: true }),
        tdCell("14.9%", w[1]), tdCell("14.4%", w[2]),
      ] }),
    ],
  });
}
function testsuiteTable() {
  const w = [Math.round(BODY_WIDTH * 0.6), Math.round(BODY_WIDTH * 0.4)];
  return new Table({
    width: { size: BODY_WIDTH, type: WidthType.DXA },
    columnWidths: w,
    rows: [
      new TableRow({ children: [thCell("Module", w[0]), thCell("Tests", w[1])] }),
      new TableRow({ children: [tdCell("General pipeline", w[0], { left: true }), tdCell("17", w[1])] }),
      new TableRow({ children: [tdCell("Baseline association (incl. Candidate Solution B recovery)", w[0], { left: true }), tdCell("23", w[1])] }),
      new TableRow({ children: [tdCell("Candidate Solution A (multi-signal redesign)", w[0], { left: true }), tdCell("18", w[1])] }),
      new TableRow({ children: [tdCell("Total", w[0], { left: true }), tdCell("58", w[1])] }),
    ],
  });
}

// ---- block renderer (interprets thesis_content.js block arrays) ----
const imgBuf = fs.readFileSync("flowchart_pipeline.png");
const imgW = 520, imgH = 373; // scaled to fit the narrower A4 body width

function renderBlocks(blocks) {
  const out = [];
  for (const b of blocks) {
    if (b.t === "h2") out.push(subHeading(b.x));
    else if (b.t === "p") out.push(p(b.x));
    else if (b.t === "bullet") out.push(bulletPara(b.x));
    else if (b.t === "caption") out.push(caption(b.x));
    else if (b.t === "image") {
      out.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 80, after: 80 },
        children: [new ImageRun({ type: "png", data: imgBuf, transformation: { width: imgW, height: imgH } })] }));
    } else if (b.t === "results_table") { out.push(new Paragraph({ spacing: { before: 60, after: 40 } })); out.push(resultsTable()); }
    else if (b.t === "testsuite_table") { out.push(new Paragraph({ spacing: { before: 60, after: 40 } })); out.push(testsuiteTable()); }
  }
  return out;
}

// ---------------------------------------------------------------------
// Document assembly
// ---------------------------------------------------------------------
const children = [];

// ---- Title page ----
for (let i = 0; i < 5; i++) children.push(new Paragraph({ spacing: { after: 0 } }));
children.push(centered(CONTENT.TITLE, { italics: true, size: 30 }));
for (let i = 0; i < 9; i++) children.push(new Paragraph({ spacing: { after: 0 } }));
children.push(centered(CONTENT.AUTHOR, { italics: true, size: 26 }));
for (let i = 0; i < 9; i++) children.push(new Paragraph({ spacing: { after: 0 } }));
children.push(centered("School of Civil & Mechanical Engineering Project Thesis", { size: 22 }));
children.push(centered("Curtin University", { size: 22 }));
children.push(new Paragraph({ spacing: { after: 0 } }));
children.push(centered("{Date goes here}", { size: 22 }));

children.push(pageBreak());

// ---- Letter of submission ----
children.push(rightAligned("{Your street number and name},"));
children.push(rightAligned("{Your suburb},"));
children.push(rightAligned("WA {Your postcode}"));
children.push(rightAligned(""));
children.push(rightAligned("{Date}"));
children.push(new Paragraph({ spacing: { after: 200 } }));
children.push(letterPara("The Head"));
children.push(letterPara("School of Civil & Mechanical Engineering,"));
children.push(letterPara("Curtin University,"));
children.push(letterPara("Kent Street,"));
children.push(letterPara("Bentley,"));
children.push(letterPara("WA 6102"));
children.push(new Paragraph({ spacing: { after: 200 } }));
children.push(letterPara("Dear Sir,"));
children.push(letterPara(
  `I submit this thesis entitled “${CONTENT.TITLE}”, based on MXEN4000 Mechatronic Engineering ` +
  "Research Project 1 and MXEN4004 Mechatronic Engineering Research Project 2, undertaken by me as " +
  "part-requirement for the degree of B.Eng. in Mechatronic Engineering."
));
children.push(new Paragraph({ spacing: { after: 200 } }));
children.push(letterPara("Yours faithfully,"));
for (let i = 0; i < 3; i++) children.push(new Paragraph({ spacing: { after: 0 } }));
children.push(letterPara(CONTENT.AUTHOR));
children.push(letterPara("{Student ID}"));

children.push(pageBreak());

// ---- Acknowledgments ----
children.push(frontMatterHeading("Acknowledgments"));
CONTENT.ACKNOWLEDGMENTS.forEach((t) => children.push(p(t)));

children.push(pageBreak());

// ---- Abstract ----
children.push(frontMatterHeading("Abstract"));
CONTENT.ABSTRACT.forEach((t) => children.push(p(t)));

children.push(pageBreak());

// ---- Nomenclature ----
children.push(frontMatterHeading("Nomenclature"));
const nomW = [1800, BODY_WIDTH - 1800];
children.push(new Table({
  width: { size: BODY_WIDTH, type: WidthType.DXA },
  columnWidths: nomW,
  rows: CONTENT.NOMENCLATURE.map(([term, def]) => new TableRow({
    children: [
      new TableCell({
        width: { size: nomW[0], type: WidthType.DXA },
        borders: { top: { style: BorderStyle.NONE }, bottom: { style: BorderStyle.NONE }, left: { style: BorderStyle.NONE }, right: { style: BorderStyle.NONE } },
        margins: { bottom: 80 },
        children: [new Paragraph({ children: [new TextRun({ text: term, bold: true, size: 21 })] })],
      }),
      new TableCell({
        width: { size: nomW[1], type: WidthType.DXA },
        borders: { top: { style: BorderStyle.NONE }, bottom: { style: BorderStyle.NONE }, left: { style: BorderStyle.NONE }, right: { style: BorderStyle.NONE } },
        margins: { bottom: 80 },
        children: [new Paragraph({ children: [new TextRun({ text: def, size: 21 })] })],
      }),
    ],
  })),
}));

children.push(pageBreak());

// ---- Table of Contents (manually authored -- correct immediately, no
// "update field" step needed; regenerate this listing if chapter content
// is edited enough to shift page numbers) ----
children.push(new Paragraph({
  spacing: { after: 260 },
  children: [new TextRun({ text: "Table of Contents", bold: true, size: 24 })],
}));
children.push(tocEntry("Acknowledgments", 3));
children.push(tocEntry("Abstract", 4));
children.push(tocEntry("Nomenclature", 5));
children.push(tocEntry("Chapter 1. Introduction", 7));
children.push(tocEntry("1.1 Background and Motivation", 7, { sub: true }));
children.push(tocEntry("1.2 Problem Statement", 7, { sub: true }));
children.push(tocEntry("1.3 Aim and Objectives", 8, { sub: true }));
children.push(tocEntry("1.4 Scope", 8, { sub: true }));
children.push(tocEntry("1.5 Thesis Structure", 8, { sub: true }));
children.push(tocEntry("Chapter 2. Background and Literature Review", 10));
children.push(tocEntry("2.1 Electromagnetic Navigation and Robotic-Assisted Bronchoscopy", 10, { sub: true }));
children.push(tocEntry("2.2 Image-Registration and Landmark-Based Vision Methods", 11, { sub: true }));
children.push(tocEntry("2.3 Detection-and-Association Methods and the BronchoTrack Baseline", 12, { sub: true }));
children.push(tocEntry("2.4 Gap Addressed by This Thesis", 12, { sub: true }));
children.push(tocEntry("Chapter 3. Experimental Procedure", 14));
children.push(tocEntry("3.1 System Architecture", 14, { sub: true }));
children.push(tocEntry("3.2 Materials", 15, { sub: true }));
children.push(tocEntry("3.3 Detection and Tracking", 15, { sub: true }));
children.push(tocEntry("3.4 Baseline Airway Association", 16, { sub: true }));
children.push(tocEntry("3.5 Candidate Solution A: Continuous Multi-Signal Re-Evaluation", 16, { sub: true }));
children.push(tocEntry("3.6 Candidate Solution B: Bounded Anchor Recovery", 17, { sub: true }));
children.push(tocEntry("3.7 Evaluation Methodology", 18, { sub: true }));
children.push(tocEntry("3.8 Visualization", 18, { sub: true }));
children.push(tocEntry("Chapter 4. Results and Discussion", 20));
children.push(tocEntry("4.1 Baseline Anchor-Loss Characterization", 20, { sub: true }));
children.push(tocEntry("4.2 Comparison of Candidate Solutions", 20, { sub: true }));
children.push(tocEntry("4.3 The Confirmed-Detection Metric", 21, { sub: true }));
children.push(tocEntry("4.4 Visualization Improvement", 21, { sub: true }));
children.push(tocEntry("4.5 Summary", 21, { sub: true }));
children.push(tocEntry("Chapter 5. Conclusions", 23));
children.push(tocEntry("Chapter 6. Future Work", 24));
children.push(tocEntry("6.1 Confidence Threshold for Recovered Labels", 24, { sub: true }));
children.push(tocEntry("6.2 Frame-Edge Detection Artifact", 24, { sub: true }));
children.push(tocEntry("6.3 Automating the CT-to-Airway-Graph Pipeline", 24, { sub: true }));
children.push(tocEntry("6.4 Re-examining Candidate Solution A in a Noisier Setting", 24, { sub: true }));
children.push(tocEntry("6.5 Broader Clinical Evaluation", 24, { sub: true }));
children.push(tocEntry("References", 26));
children.push(tocEntry("Appendix A: Candidate Solution A — Implementation Reference", 27));

children.push(pageBreak());

// ---- Chapters ----
children.push(chapterHeading("Chapter 1. Introduction"));
children.push(...renderBlocks(CONTENT.CH1));
children.push(pageBreak());

children.push(chapterHeading("Chapter 2. Background and Literature Review"));
children.push(...renderBlocks(CONTENT.CH2));
children.push(pageBreak());

children.push(chapterHeading("Chapter 3. Experimental Procedure"));
children.push(...renderBlocks(CONTENT.CH3));
children.push(pageBreak());

children.push(chapterHeading("Chapter 4. Results and Discussion"));
children.push(...renderBlocks(CONTENT.CH4));
children.push(pageBreak());

children.push(chapterHeading("Chapter 5. Conclusions"));
children.push(...renderBlocks(CONTENT.CH5));
children.push(pageBreak());

children.push(chapterHeading("Chapter 6. Future Work"));
children.push(...renderBlocks(CONTENT.CH6));
children.push(pageBreak());

// ---- References ----
children.push(chapterHeading("References"));
CONTENT.REFERENCES.forEach((r) => children.push(new Paragraph({
  alignment: AlignmentType.JUSTIFIED,
  spacing: { after: 180, line: 276 },
  indent: { left: convertInchesToTwip(0.3), hanging: convertInchesToTwip(0.3) },
  children: [new TextRun({ text: r })],
})));

children.push(pageBreak());

// ---- Appendix A ----
children.push(chapterHeading("Appendix A: Candidate Solution A — Implementation Reference (Pseudocode)"));
children.push(p(
  "This appendix gives the exact algorithmic detail behind Candidate Solution A, the continuous " +
  "multi-signal re-evaluation redesign discussed in Sections 3.5 and 4.2, which was fully implemented " +
  "and evaluated but not adopted as the recommended system. It is included here as a complete " +
  "reference, since it remains in the project codebase and represents a meaningful part of the " +
  "development and evaluation carried out for this thesis. Each entry below gives one source file's " +
  "role in a line, followed by a functional walkthrough of its classes and functions in the order they " +
  "execute."
));
PSEUDOCODE_SECTIONS.forEach((section) => {
  children.push(appendixFileName(section.name));
  children.push(appendixDesc(section.desc));
  children.push(codeBlockTable(section.code));
  children.push(new Paragraph({ spacing: { after: 60 } }));
});

// ---------------------------------------------------------------------
// Document
// ---------------------------------------------------------------------
const footer = new Footer({
  children: [new Paragraph({
    alignment: AlignmentType.CENTER,
    children: [new TextRun({ children: [PageNumber.CURRENT], size: 20, color: MUTED })],
  })],
});

const doc = new Document({
  creator: CONTENT.AUTHOR,
  title: CONTENT.TITLE,
  styles: {
    default: {
      document: { run: { font: "Times New Roman", size: 24, color: INK } }, // 12pt body, matching the template's Default style
    },
    paragraphStyles: [
      {
        id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: "Times New Roman", size: 24, bold: true, color: INK },
        paragraph: { alignment: AlignmentType.JUSTIFIED, spacing: { before: 380, after: 200 }, outlineLevel: 0 },
      },
      {
        id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: "Times New Roman", size: 24, bold: true, italics: true, color: INK },
        paragraph: { alignment: AlignmentType.JUSTIFIED, spacing: { before: 240, after: 120 }, outlineLevel: 1 },
      },
    ],
  },
  numbering: {
    config: [{
      reference: "bullets",
      levels: [{ level: 0, format: LevelFormat.BULLET, text: "–", alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: convertInchesToTwip(0.3), hanging: convertInchesToTwip(0.18) } } } }],
    }],
  },
  sections: [{
    properties: { page: { size: { width: PAGE.width, height: PAGE.height }, margin: MARGIN } },
    footers: { default: footer },
    children,
  }],
});

Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync("BronchoTrack_V3_Thesis.docx", buf);
  console.log("DOCX built");
});
