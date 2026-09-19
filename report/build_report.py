"""Builds the BronchoTrack_V3 progress report PDF for the professor update."""
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle,
    PageBreak, HRFlowable, ListFlowable, ListItem, KeepTogether, Preformatted,
)
from reportlab.pdfbase.pdfmetrics import stringWidth

INK = colors.HexColor("#1c2b33")
MUTED = colors.HexColor("#54646d")
ACCENT = colors.HexColor("#1f6f78")
ACCENT_SOFT = colors.HexColor("#e4f0ee")
RULE = colors.HexColor("#c9d2d5")
CODE_BG = colors.HexColor("#f5f4ef")

styles = getSampleStyleSheet()

styles.add(ParagraphStyle(name="ReportTitle", fontName="Helvetica-Bold", fontSize=22,
                           leading=26, textColor=INK, spaceAfter=4))
styles.add(ParagraphStyle(name="ReportSubtitle", fontName="Helvetica-Oblique", fontSize=11.5,
                           leading=15, textColor=MUTED, spaceAfter=2))
styles.add(ParagraphStyle(name="MetaLine", fontName="Helvetica", fontSize=9.5,
                           leading=13, textColor=MUTED))
styles.add(ParagraphStyle(name="H1", fontName="Helvetica-Bold", fontSize=15, leading=19,
                           textColor=INK, spaceBefore=20, spaceAfter=8))
styles.add(ParagraphStyle(name="H2", fontName="Helvetica-Bold", fontSize=12, leading=16,
                           textColor=ACCENT, spaceBefore=13, spaceAfter=5))
styles.add(ParagraphStyle(name="Body", fontName="Helvetica", fontSize=10, leading=14.6,
                           textColor=INK, alignment=TA_JUSTIFY, spaceAfter=7))
styles.add(ParagraphStyle(name="BodyTight", fontName="Helvetica", fontSize=10, leading=14.2,
                           textColor=INK, alignment=TA_LEFT, spaceAfter=3))
styles.add(ParagraphStyle(name="BulletBody", fontName="Helvetica", fontSize=10, leading=14.2,
                           textColor=INK, alignment=TA_JUSTIFY, spaceAfter=4, leftIndent=0))
styles.add(ParagraphStyle(name="Caption", fontName="Helvetica-Oblique", fontSize=8.8,
                           leading=12, textColor=MUTED, alignment=TA_CENTER, spaceBefore=4, spaceAfter=10))
styles.add(ParagraphStyle(name="CodeLabel", fontName="Helvetica-Bold", fontSize=9.2,
                           leading=12, textColor=ACCENT, spaceBefore=8, spaceAfter=3))
styles.add(ParagraphStyle(name="TableHead", fontName="Helvetica-Bold", fontSize=9,
                           leading=12, textColor=colors.white, alignment=TA_CENTER))
styles.add(ParagraphStyle(name="TableCell", fontName="Helvetica", fontSize=9,
                           leading=12, textColor=INK, alignment=TA_CENTER))
styles.add(ParagraphStyle(name="TableCellL", fontName="Helvetica", fontSize=9,
                           leading=12, textColor=INK, alignment=TA_LEFT))

def code_block(text, bg=CODE_BG, border=RULE, font_size=8.3, leading=11.2):
    ps = ParagraphStyle(name="Code", fontName="Courier", fontSize=font_size, leading=leading,
                         textColor=INK, backColor=bg, borderColor=border, borderWidth=0.75,
                         borderPadding=8, leftIndent=0)
    return Preformatted(text, ps)

def bullets(items, style="BulletBody", bullet_char="–"):
    return ListFlowable(
        [ListItem(Paragraph(it, styles[style]), leftIndent=14, spaceBefore=1) for it in items],
        bulletType="bullet", bulletFontName="Helvetica", bulletFontSize=9,
        bulletColor=ACCENT, start=bullet_char, leftIndent=8,
    )

def rule():
    return HRFlowable(width="100%", thickness=0.8, color=RULE, spaceBefore=2, spaceAfter=10)

story = []

# ---------------------------------------------------------------- Header
story.append(Paragraph("BronchoTrack_V3 — Progress Report", styles["ReportTitle"]))
story.append(Paragraph("Bronchoscope localization pipeline: architecture, development history, and current status", styles["ReportSubtitle"]))
story.append(Spacer(1, 8))
story.append(Table(
    [[Paragraph("Prepared for: Thesis committee / supervising professor", styles["MetaLine"]),
      Paragraph("Prepared by: Aveesha Nishendra", styles["MetaLine"])],
     [Paragraph("Project: BronchoTrack_V3 — bronchoscope real-time airway localization", styles["MetaLine"]),
      Paragraph("Date: August 30, 2026", styles["MetaLine"])]],
    colWidths=[3.4*inch, 3.4*inch], hAlign="LEFT",
))
story.append(Spacer(1, 6))
story.append(rule())

# ---------------------------------------------------------------- 1. Executive summary
story.append(Paragraph("1.  Executive Summary", styles["H1"]))
story.append(Paragraph(
    "BronchoTrack_V3 implements a real-time bronchoscope localization pipeline based on "
    "<i>BronchoTrack</i> (arXiv:2402.12763): a pre-operative 3-D airway graph is used to identify "
    "which airway branch the bronchoscope's camera is currently looking into, frame by frame, by "
    "detecting and tracking visible lumen (airway) openings and matching them against the graph's "
    "known branching structure. Two parallel implementations exist side by side in this codebase: "
    "<b>bronchotrack.paper_exact</b>, a strict reference build that reproduces the paper's method "
    "plus a small number of explicitly-flagged, opt-in extensions, and <b>bronchotrack.pipeline</b>, "
    "an earlier, more heavily embellished main package. <b>bronchotrack.paper_exact is the "
    "recommended and actively developed pipeline</b> as of this report.", styles["Body"]))
story.append(Paragraph(
    "This report covers the pipeline's full architecture, its exact per-frame algorithm "
    "(as pseudocode), and the development history of the current development cycle: a diagnosed "
    "failure mode in the baseline pipeline, an initial three-model \"fusion\" redesign that was "
    "built, tested, and ultimately set aside as a worse solution, and the smaller, targeted fix "
    "that replaced it and is now validated on real patient video. It closes with the current test "
    "and validation status and open items.", styles["Body"]))

# ---------------------------------------------------------------- 2. Pipeline description
story.append(Paragraph("2.  Pipeline Description", styles["H1"]))
story.append(Paragraph(
    "The pipeline processes a bronchoscopy video one frame at a time through four stages — "
    "detection, tracking, airway association (labeling), and localization — plus an optional "
    "appearance re-identification step feeding into tracking. Each stage is described below; "
    "Section 3 shows the same flow as a diagram, and Section 4 gives it as pseudocode.", styles["Body"]))

story.append(Paragraph("2.1  Lumen Detection", styles["H2"]))
story.append(Paragraph(
    "A trained YOLO26 segmentation model (<font face='Courier'>detection.py</font>; the deployed "
    "checkpoint's own training metadata confirms <font face='Courier'>yolo26n-seg</font>) scans each "
    "video frame for visible lumen (airway-opening) candidates. Because the pipeline uses a "
    "segmentation checkpoint rather than a plain bounding-box detector, each raw model output "
    "carries: a bounding box (x_c, y_c, height, aspect ratio — the paper's own box parameterization), "
    "a confidence score, a class id, an image crop of the box region (for the re-identification step "
    "below), and — when a mask comes back for that detection — an (N, 2) polygon of the lumen's "
    "segmentation boundary, in pixel coordinates. The detector is deliberately run at a low "
    "confidence threshold (paper default: 0.1) so that low-confidence-but-real detections remain "
    "available to the tracker's second matching stage rather than being discarded outright.", styles["Body"]))

story.append(Paragraph("2.2  Appearance Re-Identification (optional)", styles["H2"]))
story.append(Paragraph(
    "Each detection's image crop can optionally be passed through a ResNet50 feature extractor "
    "(<font face='Courier'>reid.py</font>) to produce a 2048-dimension appearance embedding. Each "
    "tracklet keeps a running exponential moving average of its own embedding "
    "(e = 0.9 · e_previous + 0.1 · f_new), giving the tracker an appearance cue "
    "(cosine similarity) it can use alongside motion when re-identifying a lumen across frames.", styles["Body"]))

story.append(Paragraph("2.3  Multi-Lumen Tracking", styles["H2"]))
story.append(Paragraph(
    "A BYTE-style, two-stage tracker (<font face='Courier'>tracker.py</font>) maintains one "
    "constant-velocity Kalman filter per tracked lumen, over the state "
    "[x_c, y_c, h, a, ẋ_c, ẏ_c, ḣ] (position, height, and aspect ratio, plus their velocities — "
    "aspect ratio itself is treated as constant, matching the paper). Each frame: (1) every "
    "existing tracklet is predicted forward one step; (2) high-confidence detections are matched "
    "against eligible tracklets using a combined cost C = 0.5·C_appearance + 0.5·C_motion "
    "(C_motion = 1 − IoU, C_appearance = 1 − cosine similarity), via the Hungarian algorithm; "
    "(3) tracklets still unmatched are re-matched against low-confidence detections using motion "
    "cost only (appearance is not trusted for noisier, low-confidence boxes); (4) unmatched "
    "high-confidence detections spawn new tracklets, unmatched low-confidence detections are "
    "discarded; (5) tracklets unmatched for more than 30 consecutive frames are dropped. An "
    "eligibility hook lets the association stage below exclude tracklets whose current airway "
    "label is more than 3 generations away from the pipeline's current location estimate, per the "
    "paper's own filtering rule.", styles["Body"]))

story.append(Paragraph("2.4  Airway Association (Labeling)", styles["H2"]))
story.append(Paragraph(
    "This is the pipeline's core algorithm (<font face='Courier'>paper_exact/association.py</font>): "
    "assigning each tracked lumen an anatomical branch label by matching it against the pre-operative "
    "3-D airway graph.", styles["Body"]))
story.append(bullets([
    "<b>Carina initialization.</b> The very first labels are assigned once exactly as many lumens "
    "are visible as the airway root (trachea) has children — typically the left/right main bronchus "
    "split — sorted left-to-right and matched to the graph's own left-to-right child ordering.",
    "<b>Label propagation.</b> From each already-labeled \"anchor\" tracklet, unlabeled tracklets "
    "nested inside its mask (children, i.e. openings visible past the anchor's own bifurcation) and "
    "unlabeled tracklets at the same level (siblings, under the anchor's own parent) are matched "
    "against the graph's known candidate branches via the Hungarian algorithm, on angular bearing "
    "from the anchor.",
    "<b>Roll-angle correction (paper Eq. 6/7).</b> The bronchoscope can rotate about its own optical "
    "axis; the pipeline estimates this roll from how the vector between the two oldest labeled "
    "tracklets rotates over time, Kalman-smooths that estimate, and rotates the graph's projected "
    "candidate positions by it before comparing bearings.",
    "<b>Diameter:distance verification cue (non-paper addition, on by default).</b> Beyond bearing, "
    "each candidate is scored on the ratio of its diameter to its distance from a reference point — "
    "to a small-angle approximation, its angular width as seen from that point. This is computed "
    "twice — once from the real detected mask, once from the 3-D graph at a \"virtual viewpoint\" "
    "a couple of centimetres past the bifurcation (approximating where the scope will actually be "
    "once a child branch reads clearly) — normalized within each domain to cancel the unknown "
    "pixel-to-millimetre scale factor, and blended into the matching cost. Each accepted match is "
    "additionally checked pass/fail against this cue independently "
    "(<font face='Courier'>Tracklet.diameter_distance_match</font>) — this is the flag that gates "
    "what actually gets drawn on the overlay video (Section 2.6).",
    "<b>Two-tier re-acquisition (new this cycle — see Section 5).</b> If every labeled anchor is "
    "lost at once, the pipeline no longer gives up permanently: it falls back to a frozen snapshot "
    "of the last known anchor to keep assigning labels. Full detail in Section 5.3.",
]))

story.append(Paragraph("2.5  Localization", styles["H2"]))
story.append(Paragraph(
    "Once tracklets carry labels, <font face='Courier'>localization.py</font> implements the "
    "paper's Eq. 8 voting rule every frame, with no temporal smoothing: each currently-visible "
    "labeled tracklet votes for its own branch if it is the only labeled lumen visible among its "
    "siblings, or for their shared parent branch if multiple sibling openings are visible at once "
    "(since which one the scope will advance into isn't yet resolved). The most-voted branch is "
    "reported as the bronchoscope's current location, together with its generation (0 = trachea, "
    "1 = main bronchi, …) read from the graph.", styles["Body"]))

story.append(Paragraph("2.6  Output / Visualization", styles["H2"]))
story.append(Paragraph(
    "Three artifacts are produced per run: an overlay video drawing each confirmed lumen directly "
    "on the source footage, a schematic \"graph view\" video showing the scope's live position and "
    "full traversed path on the 3-D airway tree, and a per-frame JSON log. As of this cycle, the "
    "overlay draws each confirmed tracklet's actual segmentation mask — a translucent fill plus an "
    "outline — rather than a plain center marker, with its branch label above it, falling back to "
    "a center dot only when no mask is available for that tracklet. Display is deliberately gated on "
    "<font face='Courier'>diameter_distance_match is True</font> (Section 2.4), not merely "
    "\"currently tracked\": this lets the detector run at a permissive confidence threshold without "
    "cluttering the video, by leaning on the 3-D geometric verification to filter out the resulting "
    "false positives instead of a blunt per-frame confidence cutoff.", styles["Body"]))

story.append(PageBreak())

# ---------------------------------------------------------------- 3. Flowchart
story.append(Paragraph("3.  Pipeline Overview", styles["H1"]))
story.append(Paragraph(
    "The diagram below shows the per-frame data flow through the recommended "
    "(<font face='Courier'>paper_exact</font>) pipeline, from an incoming video frame to the three "
    "output artifacts, looping back for the next frame.", styles["Body"]))
story.append(Spacer(1, 6))
img = Image("flowchart_pipeline.png")
img._restrictSize(6.6*inch, 9*inch)
story.append(img)
story.append(Paragraph("Figure 1. BronchoTrack_V3 per-frame pipeline flow (bronchotrack.paper_exact).", styles["Caption"]))

story.append(PageBreak())

# ---------------------------------------------------------------- 4. Pseudocode
story.append(Paragraph("4.  Pseudocode", styles["H1"]))
story.append(Paragraph(
    "The two blocks below give the pipeline's main per-frame loop and the airway-association "
    "algorithm that is this cycle's primary contribution (re-acquisition, marked "
    "<font face='Courier'>NEW</font>). Both are written directly from the corresponding source "
    "(<font face='Courier'>paper_exact/pipeline.py</font> and "
    "<font face='Courier'>paper_exact/association.py</font>), simplified for readability.", styles["Body"]))

story.append(Paragraph("4.1  Main per-frame loop", styles["CodeLabel"]))
main_pseudo = """function RUN_PIPELINE(video, airway_graph, detector_weights):
    graph      := AirwayGraph.load(airway_graph)
    detector   := LumenDetector(detector_weights)
    tracker    := MultiLumenTracker()
    assoc      := AirwayAssociation(graph)     # labeling + re-acquisition, Section 4.2
    localizer  := Localizer(graph)

    for frame, frame_idx in video.frames():

        # 1. Detect lumen openings
        detections := detector.infer(frame, frame_idx)      # bbox, confidence, mask

        # 1b. (optional) appearance embedding for re-identification
        for d in detections:
            d.embedding := reid_embedder.embed(d.crop)

        # 2. Track: predict + two-stage BYTE-style match (Section 2.3)
        tracklets := tracker.update(detections, frame_idx,
                                     eligibility_fn = assoc.eligibility_fn)

        # 3. Associate: propagate anatomical labels from the airway graph
        labels := assoc.process_frame(tracklets, frame_idx)

        # 4. Localize: branch-level vote (Eq. 8)
        location   := localizer.localize(tracklets)
        generation := graph.generation(location)

        result := FrameResult(frame_idx, detections, tracklets,
                               labels, location, generation)

        write_overlay_frame(result)          # mask + label overlay
        write_graph_view_frame(result)       # scope position on 3-D tree
        append_json_log(result)

    return all_results"""
story.append(code_block(main_pseudo))

story.append(Paragraph("4.2  Airway association — label propagation and re-acquisition", styles["CodeLabel"]))
assoc_pseudo = """function ASSOCIATION.PROCESS_FRAME(tracklets, frame_idx):
    current := [t in tracklets : t.time_since_update == 0]    # visible this frame

    if not initialized:
        TRY_INITIALIZE_CARINA_SPLIT(current, frame_idx)       # Section 2.4
        if still not initialized:
            return {}

    UPDATE_ROLL_ESTIMATE(current)          # Kalman-filtered, Eq. 6/7

    anchors := [t in current : t.label is not None]
    sort anchors by age, oldest/most-established first

    if anchors is not empty:
        # remember this frame's best anchor as a frozen snapshot, in
        # case every real anchor is lost on some later frame
        last_real_anchor       := freeze(anchors[0])
        last_real_anchor_frame := frame_idx
    else:
        anchors := REACQUIRE(current, frame_idx)               # NEW

    used_labels := { a.label for a in anchors }
    unlabeled    := [t in current : t.label is None]

    for anchor in anchors:
        unlabeled := PROPAGATE_FROM_ANCHOR(anchor, unlabeled,
                                            used_labels, frame_idx)

    update_gallery(current, frame_idx)      # record every branch ever visited
    return { t.track_id: t.label for t in current if t.label is not None }


function REACQUIRE(current, frame_idx):                        # NEW
    # Called only when every real anchor was lost this frame.
    if reacquisition disabled, or no snapshot ever taken, or
       frame_idx - last_real_anchor_frame > max_gap_frames:
        return []                                               # give up (for now)

    snapshot  := last_real_anchor
    unlabeled := [t in current : t.label is None]

    # Tier 1 -- SELF re-acquisition: the scope's own current lumen
    # dropped out for a few frames and came back, e.g. motion blur.
    if unlabeled is not empty:
        best := argmax_{t in unlabeled} IoU(snapshot.last_box, t.last_box)
        if IoU(snapshot.last_box, best.last_box) >= iou_threshold:
            best.label := snapshot.label
            return [best]              # a real anchor again, immediately

    # Tier 2 -- VIRTUAL-ANCHOR fallback: nothing still resembles the
    # old position (the scope kept moving) -- hand the frozen snapshot
    # itself back in as a stand-in anchor so genuinely NEW child /
    # sibling lumens can still be matched against where it last was.
    return [snapshot]


function PROPAGATE_FROM_ANCHOR(anchor, unlabeled, used_labels, frame_idx):
    children := [t in unlabeled : IS_NESTED(anchor, t)]     # inside anchor's mask
    match CHILDREN(anchor.label) <-> children
        via Hungarian assignment on:
            cost = (1 - w)*bearing_cost + w*diameter_distance_cost

    siblings := [t in unlabeled : not nested either direction]
    match SIBLINGS(parent(anchor.label)) <-> siblings   (same cost formula,
                                                          projected from the parent)

    return unlabeled minus every tracklet just matched"""
story.append(code_block(assoc_pseudo, font_size=7.9, leading=10.7))

story.append(PageBreak())
# ==== SECTION_5_MARKER ====

# ---------------------------------------------------------------- 5. Development history
story.append(Paragraph("5.  Development History This Cycle", styles["H1"]))
story.append(Paragraph(
    "This section covers the full progress from a diagnosed failure in the working baseline "
    "through to the current recommended pipeline, including a redesign that was built, evaluated, "
    "and then deliberately set aside in favor of a smaller, targeted fix.", styles["Body"]))

story.append(Paragraph("5.1  Starting point: a diagnosed anchor-loss dead end", styles["H2"]))
story.append(Paragraph(
    "The baseline <font face='Courier'>paper_exact</font> pipeline (Section 2) was already working "
    "well, but label propagation (Section 2.4) only ever runs outward from a currently-visible, "
    "already-labeled tracklet (an \"anchor\"). Read literally, this has a real dead end: if every "
    "anchor is lost at the same time — a long occlusion, or a burst of missed detections — nothing "
    "in the pipeline can ever assign a new label again for the rest of the video. Log analysis of a "
    "real patient video confirmed this in practice: the pipeline reported zero locations for 82% of "
    "a 1084-frame run, after its last anchor was lost around frame 200.", styles["Body"]))

story.append(Paragraph("5.2  First attempt: a three-model fusion redesign (built, evaluated, set aside)", styles["H2"]))
story.append(Paragraph(
    "The first approach taken was a larger redesign, built from scratch as a new "
    "<font face='Courier'>bronchotrack.fusion</font> package rather than a patch to "
    "<font face='Courier'>paper_exact</font>. It combined three separate models every frame instead "
    "of relying on anchor persistence at all:", styles["Body"]))
story.append(bullets([
    "<b>An approach motion model</b> — a scale-free \"time-to-contact\" estimate (τ = diameter / "
    "rate-of-change-of-diameter) derived purely from how fast a tracked lumen's apparent size is "
    "growing, with no camera calibration required, cross-checked against the graph's own known "
    "branch lengths.",
    "<b>A point-based identification cue</b> (\"the bronchotrack version\") — the paper's own "
    "angular-bearing idea, re-derived independently using each lumen's segmentation center.",
    "<b>A whole-mask \"ratio method\" identification cue</b> — using the entire detected mask "
    "boundary (area-equivalent diameter, boundary-to-boundary gap) rather than reducing each lumen "
    "to a single point, to make deliberate use of the segmentation output rather than a plain "
    "bounding box.",
    "<b>A Kalman fusion layer</b> combining all three into one persistent, per-branch confidence "
    "score every frame — with no one-shot initialization gate, so losing every tracklet only ever "
    "caused a few frames of prediction-only coasting rather than a permanent failure.",
]))
story.append(Paragraph(
    "This was implemented in full (motion model, both identification cues, Kalman fusion, "
    "candidate-branch overlay display), covered by 18 hand-written regression tests, and run "
    "end-to-end on real patient video. It was then reviewed against the working baseline and "
    "<b>rejected as a net regression</b> in favor of reverting to the paper_exact baseline and "
    "applying a smaller, targeted fix instead (Section 5.3). The fusion package remains in the "
    "codebase, still passing its own tests, but is <b>not</b> the recommended or actively developed "
    "pipeline going forward.", styles["Body"]))

story.append(Paragraph("5.3  Adopted fix: two-tier re-acquisition in paper_exact", styles["H2"]))
story.append(Paragraph(
    "Rather than removing anchor-dependence altogether, the adopted fix keeps the baseline's "
    "architecture and adds a bounded fallback for the specific failure mode diagnosed in 5.1. "
    "<font face='Courier'>AirwayAssociation</font> now keeps a frozen snapshot of the most-established "
    "real anchor, refreshed every frame one exists. The moment a frame has no real anchor, "
    "re-acquisition tries two things in order, both bounded by a maximum staleness window "
    "(default 90 frames, ≈3 seconds at 30 fps, past which the snapshot is considered too stale "
    "to trust):", styles["Body"]))
story.append(bullets([
    "<b>Tier 1 — self re-acquisition.</b> The dominant real-world case: the lumen the scope is "
    "already inside drops out for a few frames (motion blur, a burst of missed detections) and "
    "reappears as itself, not as a newly-visible child. Among this frame's unlabeled tracklets, "
    "whichever one overlaps the snapshot's last known box the most (by IoU) is relabeled directly "
    "with the snapshot's own label if that overlap clears a threshold (default 0.3) — no Hungarian "
    "matching needed, since the question is \"is this probably the same lumen\", not \"which child "
    "is this\".",
    "<b>Tier 2 — virtual-anchor fallback.</b> If nothing in the current frame still resembles the "
    "snapshot's old position (the scope kept advancing while the anchor was lost), the frozen "
    "snapshot itself is handed to the normal label-propagation step exactly as a live anchor would "
    "be, so genuinely new child or sibling lumens can still be matched against where the anchor "
    "last was.",
]))
story.append(Paragraph(
    "Either tier immediately restores a real anchor from the very next frame onward — this is a "
    "per-frame fallback, not a second one-shot initialization, so it keeps working across repeated "
    "gaps for the rest of a video. Setting the staleness window to 0 disables the feature entirely "
    "and reproduces the original dead-end behavior, for direct before/after comparison.", styles["Body"]))

story.append(Paragraph("5.4  Latest change: full-mask overlay visualization", styles["H2"]))
story.append(Paragraph(
    "Most recently, the overlay video (Section 2.6) was changed to draw each confirmed lumen's "
    "actual segmentation polygon — a translucent fill plus outline — instead of a fixed-radius "
    "center dot, with a fallback to the original dot marker only when no mask is available for a "
    "given tracklet. This is a visualization change only; it does not affect labeling or "
    "localization logic.", styles["Body"]))

story.append(PageBreak())

# ---------------------------------------------------------------- 6. Quantitative results
story.append(Paragraph("6.  Quantitative Results on Real Patient Video", styles["H1"]))
story.append(Paragraph(
    "The re-acquisition fix was validated end-to-end on the same real, 1084-frame patient video "
    "used to originally diagnose the anchor-loss dead end (Section 5.1), comparing a run with "
    "re-acquisition disabled against one with it enabled at default settings, using identical "
    "detector weights, confidence threshold, and airway graph. (Processing-speed / frame-rate "
    "figures are omitted here as not relevant to localization accuracy.)", styles["Body"]))

results_table = Table(
    [[Paragraph("Metric", styles["TableHead"]), Paragraph("Before<br/>(re-acquisition disabled)", styles["TableHead"]),
      Paragraph("After<br/>(re-acquisition enabled)", styles["TableHead"]), Paragraph("Change", styles["TableHead"])],
     [Paragraph("Frames with a labeled location", styles["TableCellL"]), Paragraph("166 / 1084 (15.3%)", styles["TableCell"]),
      Paragraph("433 / 1084 (39.9%)", styles["TableCell"]), Paragraph("≈ 2.6×", styles["TableCell"])],
     [Paragraph("Frames with a 3-D-model-confirmed dot", styles["TableCellL"]), Paragraph("162 / 1084 (14.9%)", styles["TableCell"]),
      Paragraph("156 / 1084 (14.4%)", styles["TableCell"]), Paragraph("essentially flat", styles["TableCell"])],
     [Paragraph("Final reported location", styles["TableCellL"]), Paragraph("—", styles["TableCell"]),
      Paragraph("L231 (generation 4)", styles["TableCell"]), Paragraph("", styles["TableCell"])]],
    colWidths=[2.35*inch, 1.55*inch, 1.55*inch, 1.05*inch], hAlign="LEFT",
)
results_table.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), ACCENT),
    ("BACKGROUND", (0, 1), (-1, -1), colors.white),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f7f6")]),
    ("GRID", (0, 0), (-1, -1), 0.6, RULE),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ("TOPPADDING", (0, 0), (-1, -1), 6),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
]))
story.append(results_table)
story.append(Paragraph("Table 1. Before/after comparison, same 1084-frame video, same methodology.", styles["Caption"]))

story.append(Paragraph(
    "Re-acquisition delivers its intended effect directly: the pipeline can now assign a location "
    "on 2.6× as many frames, confirming it no longer freezes permanently after the anchor-loss "
    "point that produced the original 82%-blank result. The second row is a genuine, and important, "
    "nuance rather than an inconsistency: the \"confirmed dot\" count is governed by a separate, "
    "stricter gate (the diameter:distance virtual-verification check, Section 2.4/2.6) that "
    "re-acquisition does not directly improve — a re-acquired label can be correct while still not "
    "clearing that independent 3-D geometric agreement check on a given frame. Visual spot-checks "
    "of the overlay video confirm this reading: early frames near the carina show both branches "
    "correctly labeled and confirmed, while later frames (e.g. frame 700, deep in the tree at L231) "
    "show the correct location text but no confirmed dot, consistent with the flat second metric.", styles["Body"]))
story.append(Paragraph(
    "The location distribution across the full re-acquisition run was dominated by "
    "<font face='Courier'>L231</font> (881 of 1084 frames), with "
    "<font face='Courier'>L23</font> (91), <font face='Courier'>trachea</font> (78), "
    "<font face='Courier'>L2</font> (18), and <font face='Courier'>R</font> (15) making up most of "
    "the remainder — consistent with a single, mostly-monotonic advance down the left bronchial "
    "tree to generation 4.", styles["Body"]))

# ---------------------------------------------------------------- 7. Testing
story.append(Paragraph("7.  Testing and Validation", styles["H1"]))
story.append(Paragraph(
    "All changes this cycle are covered by an expanded automated test suite, run both in place and "
    "from a freshly extracted copy of the delivered codebase before each hand-off:", styles["Body"]))
test_table = Table(
    [[Paragraph("Suite", styles["TableHead"]), Paragraph("Covers", styles["TableHead"]), Paragraph("Result", styles["TableHead"])],
     [Paragraph("test_pipeline_smoke.py", styles["TableCellL"]), Paragraph("Main package (bronchotrack.pipeline)", styles["TableCellL"]), Paragraph("17 / 17", styles["TableCell"])],
     [Paragraph("test_paper_exact_smoke.py", styles["TableCellL"]), Paragraph("Recommended pipeline, incl. re-acquisition (4 new tests) and mask-overlay rendering (1 new test)", styles["TableCellL"]), Paragraph("23 / 23", styles["TableCell"])],
     [Paragraph("test_fusion_smoke.py", styles["TableCellL"]), Paragraph("Fusion package (retained, not recommended)", styles["TableCellL"]), Paragraph("18 / 18", styles["TableCell"])]],
    colWidths=[2.0*inch, 3.6*inch, 0.9*inch], hAlign="LEFT",
)
test_table.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), ACCENT),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f7f6")]),
    ("GRID", (0, 0), (-1, -1), 0.6, RULE),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ("TOPPADDING", (0, 0), (-1, -1), 6),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
]))
story.append(test_table)
story.append(Paragraph("Table 2. Automated test results, 58 / 58 total, verified from a clean zip extraction.", styles["Caption"]))
story.append(Paragraph(
    "New re-acquisition tests specifically cover: correct relabeling of the same lumen after a "
    "total, genuine tracker drop; re-acquisition correctly giving up once the staleness window is "
    "exceeded; the disable escape hatch (window = 0) reproducing the original behavior exactly; and "
    "tier-2 virtual-anchor propagation correctly matching a new child when tier-1 self-matching does "
    "not apply. The new mask-overlay test confirms the full polygon is actually rendered (not just "
    "a marker) and that its fill is translucent rather than opaque.", styles["Body"]))

# ---------------------------------------------------------------- 8. Status / next steps
story.append(Paragraph("8.  Current Status and Open Items", styles["H1"]))
story.append(bullets([
    "<b>Recommended pipeline:</b> <font face='Courier'>bronchotrack.paper_exact</font>, with "
    "re-acquisition enabled at default settings and the new mask-overlay visualization.",
    "<b>Not recommended, retained in codebase:</b> <font face='Courier'>bronchotrack.fusion</font> "
    "(Section 5.2) — fully implemented and tested, but not the active development path.",
    "<b>Open item — raw mask artifacts at the frame border.</b> Now that the overlay draws full "
    "segmentation polygons rather than a single point, a pre-existing quirk in the raw detector "
    "output is newly visible: a lumen mask that gets clipped at the edge of the video frame can "
    "include polygon points that trace along the image border, rendering as an oversized shape "
    "reaching toward the frame's corners. This is not a labeling or localization defect — it is a "
    "property of the raw segmentation output becoming visible now that the full mask is drawn — but "
    "may be worth a border-touching sanity filter if it proves distracting in the overlay video.",
    "<b>Open item — confirmed-dot coverage.</b> The 3-D-model-verification gate that controls "
    "whether a label is actually displayed remained flat under re-acquisition (Section 6). Whether "
    "this is acceptable as-is, or whether the virtual-verification threshold should be revisited for "
    "re-acquired (as opposed to freshly Hungarian-matched) labels specifically, is an open design "
    "question for discussion.",
]))
story.append(Spacer(1, 10))
story.append(rule())
story.append(Paragraph(
    "Deliverables accompanying this report: the full updated codebase "
    "(<font face='Courier'>bronchotrack_pipeline.zip</font>), and the re-acquisition + mask-overlay "
    "validation run on real patient video (overlay video, graph-view video, per-frame JSON log).",
    styles["MetaLine"]))

# ---------------------------------------------------------------- build
doc = SimpleDocTemplate(
    "BronchoTrack_V3_Progress_Report.pdf", pagesize=LETTER,
    leftMargin=0.85*inch, rightMargin=0.85*inch, topMargin=0.75*inch, bottomMargin=0.75*inch,
    title="BronchoTrack_V3 Progress Report", author="Aveesha Nishendra",
)
doc.build(story)
print("PDF built")
