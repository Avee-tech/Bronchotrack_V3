const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType,
  Table, TableRow, TableCell, WidthType, ShadingType, BorderStyle,
  ImageRun, PageBreak, LevelFormat, convertInchesToTwip,
} = require("docx");
const { SECTIONS: PSEUDOCODE_SECTIONS } = require("./pseudocode_content.js");

const INK = "1E2A30";
const MUTED = "5B6B70";
const ACCENT = "2C5F68";
const RULE = "D3D8D6";

const LETTER = { width: 12240, height: 15840 }; // DXA, US Letter
const MARGIN = convertInchesToTwip(0.9);

function h1(text) {
  return new Paragraph({
    spacing: { before: 380, after: 200 },
    children: [new TextRun({ text, bold: true, size: 28, color: ACCENT, font: "Calibri" })],
  });
}
function h2(text) {
  return new Paragraph({
    spacing: { before: 240, after: 120 },
    children: [new TextRun({ text, bold: true, size: 23, color: INK, font: "Calibri" })],
  });
}
function p(text, opts = {}) {
  return new Paragraph({
    alignment: AlignmentType.JUSTIFIED,
    spacing: { after: 200, line: 300 },
    children: [new TextRun({ text, size: 22, color: INK, font: "Calibri", ...opts })],
  });
}
function bulletPara(text) {
  return new Paragraph({
    alignment: AlignmentType.JUSTIFIED,
    spacing: { after: 120, line: 290 },
    numbering: { reference: "bullets", level: 0 },
    children: [new TextRun({ text, size: 22, color: INK, font: "Calibri" })],
  });
}
function caption(text) {
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 100, after: 260 },
    children: [new TextRun({ text, italics: true, size: 18, color: MUTED, font: "Calibri" })],
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

// ---- appendix: pseudocode blocks ----
const CODE_BG = "F4F2EC";
const CODE_EDGE = "D8D3C4";

function appendixFileName(text) {
  return new Paragraph({
    spacing: { before: 300, after: 40 },
    children: [new TextRun({ text, bold: true, size: 22, color: ACCENT, font: "Courier New" })],
  });
}
function appendixDesc(text) {
  return new Paragraph({
    spacing: { after: 100 },
    children: [new TextRun({ text, italics: true, size: 18, color: MUTED, font: "Calibri" })],
  });
}
// A single-cell table gives the code block a shaded, bordered "panel" look;
// each code line is its own paragraph inside that one cell.
function codeBlockTable(code) {
  const lines = code.split("\n");
  const lineParagraphs = lines.map((line) => new Paragraph({
    spacing: { after: 0, line: 220 },
    children: [new TextRun({ text: line.length ? line : " ", font: "Courier New", size: 15, color: INK })],
  }));
  return new Table({
    width: { size: 9360, type: WidthType.DXA },
    columnWidths: [9360],
    rows: [
      new TableRow({
        children: [
          new TableCell({
            width: { size: 9360, type: WidthType.DXA },
            shading: { type: ShadingType.CLEAR, color: "auto", fill: CODE_BG },
            margins: { top: 140, bottom: 140, left: 160, right: 160 },
            borders: {
              top: { style: BorderStyle.SINGLE, size: 4, color: CODE_EDGE },
              bottom: { style: BorderStyle.SINGLE, size: 4, color: CODE_EDGE },
              left: { style: BorderStyle.SINGLE, size: 4, color: CODE_EDGE },
              right: { style: BorderStyle.SINGLE, size: 4, color: CODE_EDGE },
            },
            children: lineParagraphs,
          }),
        ],
      }),
    ],
  });
}

// ---- header meta table ----
function metaCell(text) {
  return new TableCell({
    width: { size: 4680, type: WidthType.DXA },
    borders: {
      top: { style: BorderStyle.NONE }, bottom: { style: BorderStyle.NONE },
      left: { style: BorderStyle.NONE }, right: { style: BorderStyle.NONE },
    },
    children: [new Paragraph({ children: [new TextRun({ text, size: 18, color: MUTED, font: "Calibri" })] })],
  });
}
const metaTable = new Table({
  width: { size: 9360, type: WidthType.DXA },
  columnWidths: [4680, 4680],
  rows: [
    new TableRow({ children: [metaCell("Prepared by: Aveesha Nishendra"), metaCell("Date: September 8, 2026")] }),
    new TableRow({ children: [metaCell("Project: BronchoTrack_V3 — bronchoscope airway localization"), metaCell("")] }),
  ],
});

// ---- results table ----
function thCell(text, width) {
  return new TableCell({
    width: { size: width, type: WidthType.DXA },
    shading: { type: ShadingType.CLEAR, color: "auto", fill: ACCENT },
    verticalAlign: "center",
    children: [new Paragraph({ alignment: AlignmentType.CENTER,
      children: [new TextRun({ text, bold: true, size: 19, color: "FFFFFF", font: "Calibri" })] })],
  });
}
function tdCell(text, width, opts = {}) {
  return new TableCell({
    width: { size: width, type: WidthType.DXA },
    verticalAlign: "center",
    children: [new Paragraph({ alignment: opts.left ? AlignmentType.LEFT : AlignmentType.CENTER,
      children: [new TextRun({ text, size: 19, color: INK, font: "Calibri" })] })],
  });
}
const resultsTable = new Table({
  width: { size: 9360, type: WidthType.DXA },
  columnWidths: [4680, 2340, 2340],
  rows: [
    new TableRow({ children: [thCell("Metric", 4680), thCell("Before", 2340), thCell("After", 2340)] }),
    new TableRow({ children: [
      tdCell("Frames with a location reported", 4680, { left: true }),
      tdCell("15.3%", 2340), tdCell("39.9%", 2340),
    ] }),
  ],
});

// ---- image ----
const imgBuf = fs.readFileSync("flowchart_pipeline.png");
const imgW = 605, imgH = 434; // preserves the 1960x1405 source aspect ratio at ~96dpi

const children = [];

// Title block
children.push(new Paragraph({
  spacing: { after: 60 },
  children: [new TextRun({ text: "BronchoTrack_V3 — Progress Update", bold: true, size: 40, color: INK, font: "Calibri" })],
}));
children.push(new Paragraph({
  spacing: { after: 160 },
  children: [new TextRun({ text: "A summary of recent development, prepared for supervisor review", italics: true, size: 22, color: MUTED, font: "Calibri" })],
}));
children.push(metaTable);
children.push(new Paragraph({ spacing: { before: 160, after: 0 } }));
children.push(ruleParagraph());

children.push(p(
  "This report summarizes progress on the bronchoscope tracking project since the last update. It " +
  "covers a limitation identified in the existing system, an initial redesign that was implemented, " +
  "evaluated, and ultimately not adopted, the solution that was implemented in its place, and the " +
  "results of testing that solution on real patient video. A technical appendix at the end of this " +
  "document gives the exact algorithmic detail (pseudocode for every file) behind the redesign " +
  "described in the Limitation Identified section; this main body is intended as an accessible " +
  "overview of the same work."
));
children.push(p(
  "As background: the objective of this line of work is a system that can observe bronchoscope video " +
  "in real time and determine, automatically, which airway branch the camera is currently positioned " +
  "in, using the video feed together with a three-dimensional model of the patient's airway tree built " +
  "in advance from their CT scan. No additional tracking hardware or manual annotation during the " +
  "procedure is required. This report covers one complete development cycle on that system: a " +
  "limitation that was identified, a first attempted fix that was set aside, and the fix that was " +
  "ultimately adopted."
));

// 1. Project overview
children.push(h1("Project Overview"));
children.push(p(
  "The system's objective is to determine, from live bronchoscope video alone, which airway branch the " +
  "camera is currently viewing — the trachea, the left or right main bronchus, or one of their " +
  "sub-branches. In practical terms, this would allow a bronchoscope to indicate the operator's current " +
  "position on a map of the patient's own airways, without requiring separate tracking hardware."
));
children.push(p(
  "This is achieved by combining two sources of information. The first is a pre-operative three-" +
  "dimensional model of the patient's airway tree, built from their CT scan, which provides the known " +
  "shape and branching structure in advance. The second is the live video itself: the system detects " +
  "the openings in the airway wall that lead to each branch, tracks them consistently from frame to " +
  "frame as the camera moves, and compares what is currently visible against what the three-dimensional " +
  "model predicts should be visible from the camera's estimated position. When the two are in agreement, " +
  "the system assigns a label — for example, identifying a given opening as the right upper lobe branch " +
  "— and this becomes the basis for reporting the scope's location."
));
children.push(p(
  "This work is based on a published method (BronchoTrack). The implementation maintains a strict, " +
  "faithful reproduction of that method as a separate baseline, with a small number of additional, " +
  "clearly documented improvements layered on top where the published method is underspecified or where " +
  "practical issues were identified during development. Keeping these separate makes it possible to " +
  "attribute any given result to the published method itself versus an addition made during this " +
  "project."
));
children.push(p(
  "The broader motivation is worth stating explicitly: bronchoscopists currently navigate largely by " +
  "memory and visual landmarks, and disorientation within a small, branching airway is a recognized " +
  "challenge, particularly for less experienced operators or unusually complex anatomy. A reliable, " +
  "camera-only localization system represents a step toward tools that could assist with this in real " +
  "time, without introducing additional imaging hardware into the procedure room."
));

children.push(pageBreak());

// 2. System overview + diagram
children.push(h1("System Overview"));
children.push(p(
  "For each frame of video, the system performs four steps in sequence: it detects the airway openings " +
  "visible in that frame; a tracking stage maintains each opening's identity consistently from one frame " +
  "to the next, so that a given opening is not mistaken for a new one as the camera moves; an " +
  "association stage determines which anatomical branch each currently visible opening most likely " +
  "corresponds to, using the three-dimensional model as a reference; and finally, once openings are " +
  "labeled, the system takes a vote among everything currently visible to determine the scope's overall " +
  "location. The diagram below illustrates this sequence."
));
children.push(new Paragraph({
  alignment: AlignmentType.CENTER,
  spacing: { before: 80, after: 80 },
  children: [new ImageRun({ type: "png", data: imgBuf, transformation: { width: imgW, height: imgH } })],
}));
children.push(caption("Figure 1. How a single video frame moves through the system, from input to output."));
children.push(p(
  "Two aspects of this design are worth noting. First, the detection stage produces a full outline of " +
  "each opening rather than a simple bounding box; the size and shape of an opening, not merely its " +
  "approximate position, provides useful evidence for identifying which branch it is, and this " +
  "information would be lost with a box-only approach. Second, the system includes an independent " +
  "verification step: after an opening is matched to a branch by position, its apparent size at that " +
  "distance is separately checked against what the three-dimensional model predicts. A label is only " +
  "treated as fully confirmed once both checks are in agreement — a stricter standard than it may " +
  "initially appear, and one that becomes relevant again in the results section below."
));

children.push(pageBreak());

// 3. Limitation identified
children.push(h1("Limitation Identified"));
children.push(p(
  "The system was already performing reasonably well, but had a specific limitation: it only labels a " +
  "new opening by comparing it against an opening it is already confident about, referred to as an " +
  "“anchor.” This is a reasonable approach under normal conditions. However, if the video " +
  "encounters a difficult stretch — motion blur, a brief loss of view, or a period in which the " +
  "detector fails to identify anything — and every anchor is lost simultaneously, the system had no " +
  "mechanism to recover. It would stop reporting a location for the remainder of the video, having lost " +
  "anything to compare new detections against."
));
children.push(p(
  "This was not a theoretical concern; it was confirmed on a real patient recording. Across a " +
  "1,084-frame video, the system lost its last anchor roughly a fifth of the way through and reported " +
  "no location for approximately 82% of the remaining video. This is clearly insufficient for a system " +
  "intended to track a scope's position throughout an entire procedure."
));

children.push(h2("Initial Approach: A Redesign That Was Not Adopted"));
children.push(p(
  "The first approach taken was a more substantial redesign. Rather than relying on a persistent anchor " +
  "at all, this version combined three independent signals every frame: the rate at which a visible " +
  "opening's apparent size was increasing, used as an indicator that the scope was approaching it; a " +
  "position-based match against the three-dimensional model; and a shape-based match using the complete " +
  "detected outline of each opening rather than a single reference point. These three signals were " +
  "combined, with smoothing over time, into a single running confidence score per branch. The intent " +
  "was that the system would never depend on any single tracked object persisting from one frame to the " +
  "next, which would eliminate the failure mode described above by construction."
));
children.push(p(
  "This approach was fully implemented, covered by an automated test suite, and evaluated on the same " +
  "real video used to identify the original problem. Upon direct comparison with the existing, simpler " +
  "system, however, it performed worse overall — the added complexity did not translate into a " +
  "corresponding improvement in reliability. Based on this evaluation, the decision was made to set " +
  "this approach aside and return to the original system, addressing the identified limitation with a " +
  "smaller, more targeted change instead."
));

children.push(pageBreak());

children.push(p(
  "This attempt is included in this report rather than omitted, as it represents a meaningful finding " +
  "in its own right: it confirmed that the existing anchor-based approach was fundamentally sound and " +
  "required a bounded correction for one specific failure mode, rather than a complete redesign. That " +
  "is a useful conclusion even though this particular implementation was not adopted as the primary " +
  "system. It remains in the codebase, fully tested and functional, should it prove useful as a " +
  "reference in future work."
));

// 4. Solution
children.push(h1("Solution Implemented"));
children.push(p(
  "The solution that was adopted retains the original anchor-based approach but adds a recovery " +
  "mechanism. The system now retains a snapshot of the most recently confirmed opening, even after that " +
  "opening is no longer visible. If every anchor is lost, the system attempts recovery in two stages " +
  "before giving up:"
));
children.push(bulletPara(
  "First, the system checks whether one of the currently unidentified openings is likely the same " +
  "opening reappearing — that is, the scope's own current position, returning after a brief " +
  "interruption — based on how closely it overlaps with the opening's last known position. If the " +
  "overlap is sufficient, the opening is relabeled directly, without requiring a fresh match."
));
children.push(bulletPara(
  "If no sufficiently close match is found, the system falls back to using the last known position as " +
  "a reference point, allowing newly visible branches nearby to still be identified against the " +
  "three-dimensional model, even though the original opening itself is no longer present."
));
children.push(p(
  "In either case, once an opening is successfully re-identified, the system resumes normal operation " +
  "immediately; this is not a one-time reset, and the mechanism can recur as many times as needed over " +
  "the course of a video. A safeguard is also included: if an excessive amount of time has elapsed " +
  "since an anchor was last confidently observed (approximately three seconds), the system treats the " +
  "retained snapshot as unreliable and withholds recovery until a new anchor is established, rather " +
  "than acting on outdated information."
));

children.push(pageBreak());

// 5. Results
children.push(h1("Results"));
children.push(p(
  "The solution was evaluated on the same real, 1,084-frame patient video used to originally identify " +
  "the limitation, comparing a run with recovery disabled against one with recovery enabled, under " +
  "otherwise identical conditions:"
));
children.push(new Paragraph({ spacing: { before: 60, after: 40 } }));
children.push(resultsTable);
children.push(caption("Table 1. Same 1,084-frame video and settings, before and after the recovery mechanism."));
children.push(p(
  "The proportion of the video with a reported location increased by a factor of approximately 2.6, " +
  "confirming that the system no longer permanently stops reporting a location following this type of " +
  "interruption."
));
children.push(p(
  "One important caveat should be noted. A separate, stricter check governs whether a label is " +
  "confident enough to be displayed on the output video, comparing the label against the three-" +
  "dimensional model's expected geometry rather than tracking alone. This stricter metric remained " +
  "essentially unchanged (14.9% before, 14.4% after). This is not an inconsistency: the recovery " +
  "mechanism is functioning exactly as intended, restoring a label after an interruption, while this " +
  "separate confidence check addresses a different question — how strictly the system should trust a " +
  "given match — and is unaffected by the recovery mechanism itself. Whether this threshold should be " +
  "relaxed specifically for labels restored through recovery, as opposed to those matched under normal " +
  "conditions, is a design question worth discussing further."
));

// 6. Visualization
children.push(h1("Visualization Improvement"));
children.push(p(
  "A secondary improvement was also made to the output video. Previously, each identified opening was " +
  "represented by a small dot at its center. The output now displays the actual detected shape as a " +
  "filled, semi-transparent outline, which makes it considerably easier to visually verify what the " +
  "system is detecting, rather than relying on a single point. This change also surfaced a minor, " +
  "pre-existing artifact: an opening located near the edge of the frame is occasionally detected with " +
  "an irregular, oversized outline, because the detection boundary follows the edge of the frame rather " +
  "than stopping cleanly at the true edge of the opening. This is a property of the raw detection " +
  "output rather than a tracking defect, and has been noted as a candidate for a filtering step if it " +
  "proves distracting in practice."
));

children.push(pageBreak());

// 7. Testing
children.push(h1("Testing and Validation"));
children.push(p(
  "To verify the results described above, I ran the pipeline myself, on my own machine, against the " +
  "same patient video I had originally provided — comparing its behavior with the recovery mechanism " +
  "disabled against its behavior with the recovery mechanism enabled. This is the direct source of the " +
  "before-and-after results reported in the Results section: both runs were produced under my own " +
  "testing, using identical settings apart from the recovery mechanism itself, so the comparison " +
  "reflects a genuine before-and-after on real video rather than a simulated or synthetic case."
));

// 8. Status / next steps
children.push(h1("Current Status and Next Steps"));
children.push(p(
  "The recommended, actively maintained version of the system now includes the recovery mechanism and " +
  "the visualization improvement described above. The earlier three-signal redesign remains in the " +
  "codebase for reference but is not the direction being pursued going forward."
));
children.push(p(
  "Two open questions would benefit from further discussion: whether the stricter confidence check " +
  "should apply a more lenient standard to a label restored through the recovery mechanism, as opposed " +
  "to one matched under normal conditions; and whether the frame-edge detection artifact described " +
  "above warrants a dedicated fix before the next round of testing."
));
children.push(p(
  "The exact algorithmic detail for the redesign discussed above — pseudocode for every file in that " +
  "implementation — is included as a technical appendix at the end of this document. Demonstration " +
  "videos are available and can be provided on request."
));

children.push(new Paragraph({ spacing: { before: 200 } }));
children.push(ruleParagraph());
children.push(new Paragraph({ spacing: { after: 40 }, children: [new TextRun({ text: "Sincerely,", size: 22, color: INK, font: "Calibri" })] }));
children.push(new Paragraph({ children: [new TextRun({ text: "Aveesha Nishendra", size: 22, color: INK, font: "Calibri" })] }));

// ---- Appendix ----
children.push(pageBreak());
children.push(h1("Technical Appendix: Redesign Implementation Reference"));
children.push(p(
  "This appendix gives the exact algorithmic detail behind the three-signal redesign discussed in " +
  "the Limitation Identified section above (the diameter-growth motion model, position-based bearing " +
  "matching, and whole-mask ratio method, combined through a Kalman filter). It is not the recommended, " +
  "actively maintained system — see Current Status and Next Steps — but is included here as a complete " +
  "reference, since it remains in the codebase and represents a meaningful part of this development " +
  "cycle. Each entry below gives one file's role in a line, followed by a functional walkthrough of its " +
  "classes and functions in the order they execute."
));
children.push(new Paragraph({ spacing: { before: 40, after: 200 } }));

PSEUDOCODE_SECTIONS.forEach((section) => {
  children.push(appendixFileName(section.name));
  children.push(appendixDesc(section.desc));
  children.push(codeBlockTable(section.code));
  children.push(new Paragraph({ spacing: { after: 60 } }));
});

const doc = new Document({
  creator: "Aveesha Nishendra",
  title: "BronchoTrack_V3 Progress Update",
  numbering: {
    config: [{
      reference: "bullets",
      levels: [{ level: 0, format: LevelFormat.BULLET, text: "–", alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: convertInchesToTwip(0.3), hanging: convertInchesToTwip(0.18) } } } }],
    }],
  },
  sections: [{
    properties: { page: { size: LETTER, margin: { top: MARGIN, bottom: MARGIN, left: MARGIN, right: MARGIN } } },
    children,
  }],
});

Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync("BronchoTrack_V3_Update_Professional.docx", buf);
  console.log("DOCX built");
});
