"""Builds a professional-toned, plain-language ~8-page progress update for
the supervisor (no pseudocode, minimal jargon, formal register) -- revision
of build_report_human.py per user request to make it sound professional."""
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle,
    HRFlowable, ListFlowable, ListItem, PageBreak,
)

INK = colors.HexColor("#1e2a30")
MUTED = colors.HexColor("#5b6b70")
ACCENT = colors.HexColor("#2c5f68")
RULE = colors.HexColor("#d3d8d6")

styles = getSampleStyleSheet()

styles.add(ParagraphStyle(name="Title2", fontName="Helvetica-Bold", fontSize=20,
                           leading=24, textColor=INK, spaceAfter=6))
styles.add(ParagraphStyle(name="Sub2", fontName="Helvetica-Oblique", fontSize=11,
                           leading=15, textColor=MUTED, spaceAfter=4))
styles.add(ParagraphStyle(name="Meta2", fontName="Helvetica", fontSize=9.5,
                           leading=13, textColor=MUTED))
styles.add(ParagraphStyle(name="Head2", fontName="Helvetica-Bold", fontSize=14, leading=18,
                           textColor=ACCENT, spaceBefore=22, spaceAfter=11))
styles.add(ParagraphStyle(name="SubHead2", fontName="Helvetica-Bold", fontSize=11.5, leading=15,
                           textColor=INK, spaceBefore=14, spaceAfter=7))
styles.add(ParagraphStyle(name="P2", fontName="Helvetica", fontSize=11, leading=17.2,
                           textColor=INK, alignment=TA_JUSTIFY, spaceAfter=12))
styles.add(ParagraphStyle(name="PBul", fontName="Helvetica", fontSize=11, leading=16.8,
                           textColor=INK, alignment=TA_JUSTIFY, spaceAfter=7))
styles.add(ParagraphStyle(name="Caption2", fontName="Helvetica-Oblique", fontSize=9,
                           leading=12, textColor=MUTED, alignment=TA_CENTER, spaceBefore=5, spaceAfter=14))
styles.add(ParagraphStyle(name="TH2", fontName="Helvetica-Bold", fontSize=9.5, leading=13,
                           textColor=colors.white, alignment=TA_CENTER))
styles.add(ParagraphStyle(name="TC2", fontName="Helvetica", fontSize=9.5, leading=13,
                           textColor=INK, alignment=TA_CENTER))
styles.add(ParagraphStyle(name="TCL2", fontName="Helvetica", fontSize=9.5, leading=13,
                           textColor=INK, alignment=TA_LEFT))
styles.add(ParagraphStyle(name="Sign", fontName="Helvetica", fontSize=11, leading=16,
                           textColor=INK, spaceAfter=4))

def bullets(items):
    return ListFlowable(
        [ListItem(Paragraph(it, styles["PBul"]), leftIndent=14, spaceBefore=2) for it in items],
        bulletType="bullet", bulletFontName="Helvetica", bulletFontSize=9,
        bulletColor=ACCENT, start="–", leftIndent=8,
    )

def rule():
    return HRFlowable(width="100%", thickness=0.7, color=RULE, spaceBefore=4, spaceAfter=14)

story = []

# ---------------------------------------------------------- Header
story.append(Paragraph("BronchoTrack_V3 — Progress Update", styles["Title2"]))
story.append(Paragraph("A summary of recent development, prepared for supervisor review", styles["Sub2"]))
story.append(Spacer(1, 6))
story.append(Table(
    [[Paragraph("Prepared by: Aveesha Nishendra", styles["Meta2"]), Paragraph("Date: September 2, 2026", styles["Meta2"])],
     [Paragraph("Project: BronchoTrack_V3 — bronchoscope airway localization", styles["Meta2"]), Paragraph("", styles["Meta2"])]],
    colWidths=[3.6*inch, 3.2*inch], hAlign="LEFT",
))
story.append(Spacer(1, 8))
story.append(rule())

story.append(Paragraph(
    "This report summarizes progress on the bronchoscope tracking project since the last update. It "
    "covers a limitation identified in the existing system, an initial redesign that was implemented, "
    "evaluated, and ultimately not adopted, the solution that was implemented in its place, and the "
    "results of testing that solution on real patient video. A more detailed technical writeup, "
    "including architecture documentation and exact algorithmic detail, is available separately on "
    "request; this report is intended as an accessible overview of the same work.", styles["P2"]))
story.append(Paragraph(
    "As background: the objective of this line of work is a system that can observe bronchoscope video "
    "in real time and determine, automatically, which airway branch the camera is currently positioned "
    "in, using the video feed together with a three-dimensional model of the patient's airway tree built "
    "in advance from their CT scan. No additional tracking hardware or manual annotation during the "
    "procedure is required. This report covers one complete development cycle on that system: a "
    "limitation that was identified, a first attempted fix that was set aside, and the fix that was "
    "ultimately adopted.", styles["P2"]))

# ---------------------------------------------------------- 1. What it does
story.append(Paragraph("Project Overview", styles["Head2"]))
story.append(Paragraph(
    "The system's objective is to determine, from live bronchoscope video alone, which airway branch the "
    "camera is currently viewing — the trachea, the left or right main bronchus, or one of their "
    "sub-branches. In practical terms, this would allow a bronchoscope to indicate the operator's current "
    "position on a map of the patient's own airways, without requiring separate tracking hardware.", styles["P2"]))
story.append(Paragraph(
    "This is achieved by combining two sources of information. The first is a pre-operative three-"
    "dimensional model of the patient's airway tree, built from their CT scan, which provides the known "
    "shape and branching structure in advance. The second is the live video itself: the system detects "
    "the openings in the airway wall that lead to each branch, tracks them consistently from frame to "
    "frame as the camera moves, and compares what is currently visible against what the three-dimensional "
    "model predicts should be visible from the camera's estimated position. When the two are in agreement, "
    "the system assigns a label — for example, identifying a given opening as the right upper lobe branch "
    "— and this becomes the basis for reporting the scope's location.", styles["P2"]))
story.append(Paragraph(
    "This work is based on a published method (BronchoTrack). The implementation maintains a strict, "
    "faithful reproduction of that method as a separate baseline, with a small number of additional, "
    "clearly documented improvements layered on top where the published method is underspecified or where "
    "practical issues were identified during development. Keeping these separate makes it possible to "
    "attribute any given result to the published method itself versus an addition made during this "
    "project.", styles["P2"]))
story.append(Paragraph(
    "The broader motivation is worth stating explicitly: bronchoscopists currently navigate largely by "
    "memory and visual landmarks, and disorientation within a small, branching airway is a recognized "
    "challenge, particularly for less experienced operators or unusually complex anatomy. A reliable, "
    "camera-only localization system represents a step toward tools that could assist with this in real "
    "time, without introducing additional imaging hardware into the procedure room.", styles["P2"]))

story.append(PageBreak())

# ---------------------------------------------------------- 2. The flow, plain language
story.append(Paragraph("System Overview", styles["Head2"]))
story.append(Paragraph(
    "For each frame of video, the system performs four steps in sequence: it detects the airway openings "
    "visible in that frame; a tracking stage maintains each opening's identity consistently from one frame "
    "to the next, so that a given opening is not mistaken for a new one as the camera moves; an "
    "association stage determines which anatomical branch each currently visible opening most likely "
    "corresponds to, using the three-dimensional model as a reference; and finally, once openings are "
    "labeled, the system takes a vote among everything currently visible to determine the scope's overall "
    "location. The diagram below illustrates this sequence.", styles["P2"]))
story.append(Spacer(1, 4))
img = Image("flowchart_pipeline.png")
img._restrictSize(6.3*inch, 8.6*inch)
story.append(img)
story.append(Paragraph("Figure 1. How a single video frame moves through the system, from input to output.", styles["Caption2"]))
story.append(Paragraph(
    "Two aspects of this design are worth noting. First, the detection stage produces a full outline of "
    "each opening rather than a simple bounding box; the size and shape of an opening, not merely its "
    "approximate position, provides useful evidence for identifying which branch it is, and this "
    "information would be lost with a box-only approach. Second, the system includes an independent "
    "verification step: after an opening is matched to a branch by position, its apparent size at that "
    "distance is separately checked against what the three-dimensional model predicts. A label is only "
    "treated as fully confirmed once both checks are in agreement — a stricter standard than it may "
    "initially appear, and one that becomes relevant again in the results section below.", styles["P2"]))

story.append(PageBreak())

# ---------------------------------------------------------- 3. The problem
story.append(Paragraph("Limitation Identified", styles["Head2"]))
story.append(Paragraph(
    "The system was already performing reasonably well, but had a specific limitation: it only labels a "
    "new opening by comparing it against an opening it is already confident about, referred to as an "
    "\"anchor.\" This is a reasonable approach under normal conditions. However, if the video encounters a "
    "difficult stretch — motion blur, a brief loss of view, or a period in which the detector fails to "
    "identify anything — and every anchor is lost simultaneously, the system had no mechanism to recover. "
    "It would stop reporting a location for the remainder of the video, having lost anything to compare "
    "new detections against.", styles["P2"]))
story.append(Paragraph(
    "This was not a theoretical concern; it was confirmed on a real patient recording. Across a "
    "1,084-frame video, the system lost its last anchor roughly a fifth of the way through and reported no "
    "location for approximately 82% of the remaining video. This is clearly insufficient for a system "
    "intended to track a scope's position throughout an entire procedure.", styles["P2"]))

story.append(Paragraph("Initial Approach: A Redesign That Was Not Adopted", styles["SubHead2"]))
story.append(Paragraph(
    "The first approach taken was a more substantial redesign. Rather than relying on a persistent anchor "
    "at all, this version combined three independent signals every frame: the rate at which a visible "
    "opening's apparent size was increasing, used as an indicator that the scope was approaching it; a "
    "position-based match against the three-dimensional model; and a shape-based match using the complete "
    "detected outline of each opening rather than a single reference point. These three signals were "
    "combined, with smoothing over time, into a single running confidence score per branch. The intent was "
    "that the system would never depend on any single tracked object persisting from one frame to the "
    "next, which would eliminate the failure mode described above by construction.", styles["P2"]))
story.append(Paragraph(
    "This approach was fully implemented, covered by an automated test suite, and evaluated on the same "
    "real video used to identify the original problem. Upon direct comparison with the existing, simpler "
    "system, however, it performed worse overall — the added complexity did not translate into a "
    "corresponding improvement in reliability. Based on this evaluation, the decision was made to set this "
    "approach aside and return to the original system, addressing the identified limitation with a "
    "smaller, more targeted change instead.", styles["P2"]))
story.append(PageBreak())

story.append(Paragraph(
    "This attempt is included in this report rather than omitted, as it represents a meaningful finding "
    "in its own right: it confirmed that the existing anchor-based approach was fundamentally sound and "
    "required a bounded correction for one specific failure mode, rather than a complete redesign. That is "
    "a useful conclusion even though this particular implementation was not adopted as the primary system. "
    "It remains in the codebase, fully tested and functional, should it prove useful as a reference in "
    "future work.", styles["P2"]))

# ---------------------------------------------------------- 4. The fix
story.append(Paragraph("Solution Implemented", styles["Head2"]))
story.append(Paragraph(
    "The solution that was adopted retains the original anchor-based approach but adds a recovery "
    "mechanism. The system now retains a snapshot of the most recently confirmed opening, even after that "
    "opening is no longer visible. If every anchor is lost, the system attempts recovery in two stages "
    "before giving up:", styles["P2"]))
story.append(bullets([
    "First, the system checks whether one of the currently unidentified openings is likely the same "
    "opening reappearing — that is, the scope's own current position, returning after a brief "
    "interruption — based on how closely it overlaps with the opening's last known position. If the "
    "overlap is sufficient, the opening is relabeled directly, without requiring a fresh match.",
    "If no sufficiently close match is found, the system falls back to using the last known position as a "
    "reference point, allowing newly visible branches nearby to still be identified against the "
    "three-dimensional model, even though the original opening itself is no longer present.",
]))
story.append(Paragraph(
    "In either case, once an opening is successfully re-identified, the system resumes normal operation "
    "immediately; this is not a one-time reset, and the mechanism can recur as many times as needed over "
    "the course of a video. A safeguard is also included: if an excessive amount of time has elapsed since "
    "an anchor was last confidently observed (approximately three seconds), the system treats the retained "
    "snapshot as unreliable and withholds recovery until a new anchor is established, rather than acting "
    "on outdated information.", styles["P2"]))

story.append(PageBreak())

# ---------------------------------------------------------- 5. Results
story.append(Paragraph("Results", styles["Head2"]))
story.append(Paragraph(
    "The solution was evaluated on the same real, 1,084-frame patient video used to originally identify "
    "the limitation, comparing a run with recovery disabled against one with recovery enabled, under "
    "otherwise identical conditions:", styles["P2"]))

t = Table(
    [[Paragraph("Metric", styles["TH2"]), Paragraph("Before", styles["TH2"]), Paragraph("After", styles["TH2"])],
     [Paragraph("Frames with a location reported", styles["TCL2"]), Paragraph("15.3%", styles["TC2"]), Paragraph("39.9%", styles["TC2"])]],
    colWidths=[2.9*inch, 1.85*inch, 1.85*inch], hAlign="LEFT",
)
t.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), ACCENT),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white]),
    ("GRID", (0, 0), (-1, -1), 0.6, RULE),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ("TOPPADDING", (0, 0), (-1, -1), 7),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
]))
story.append(t)
story.append(Paragraph("Table 1. Same 1,084-frame video and settings, before and after the recovery mechanism.", styles["Caption2"]))

story.append(Paragraph(
    "The proportion of the video with a reported location increased by a factor of approximately 2.6, "
    "confirming that the system no longer permanently stops reporting a location following this type of "
    "interruption.", styles["P2"]))
story.append(Paragraph(
    "One important caveat should be noted. A separate, stricter check governs whether a label is confident "
    "enough to be displayed on the output video, comparing the label against the three-dimensional model's "
    "expected geometry rather than tracking alone. This stricter metric remained essentially unchanged "
    "(14.9% before, 14.4% after). This is not an inconsistency: the recovery mechanism is functioning "
    "exactly as intended, restoring a label after an interruption, while this separate confidence check "
    "addresses a different question — how strictly the system should trust a given match — and is "
    "unaffected by the recovery mechanism itself. Whether this threshold should be relaxed specifically for "
    "labels restored through recovery, as opposed to those matched under normal conditions, is a design "
    "question worth discussing further.", styles["P2"]))

# ---------------------------------------------------------- 6. Visualization
story.append(Paragraph("Visualization Improvement", styles["Head2"]))
story.append(Paragraph(
    "A secondary improvement was also made to the output video. Previously, each identified opening was "
    "represented by a small dot at its center. The output now displays the actual detected shape as a "
    "filled, semi-transparent outline, which makes it considerably easier to visually verify what the "
    "system is detecting, rather than relying on a single point. This change also surfaced a minor, "
    "pre-existing artifact: an opening located near the edge of the frame is occasionally detected with an "
    "irregular, oversized outline, because the detection boundary follows the edge of the frame rather than "
    "stopping cleanly at the true edge of the opening. This is a property of the raw detection output "
    "rather than a tracking defect, and has been noted as a candidate for a filtering step if it proves "
    "distracting in practice.", styles["P2"]))

story.append(PageBreak())

# ---------------------------------------------------------- 7. Confidence / testing
story.append(Paragraph("Testing and Validation", styles["Head2"]))
story.append(Paragraph(
    "All work described in this report is covered by an automated test suite of 58 tests across the "
    "different components of the system, run both during development and from a freshly extracted copy of "
    "the delivered codebase prior to each hand-off, to confirm that results do not depend on the local "
    "development environment. For the recovery mechanism specifically, tests cover the standard recovery "
    "case, the case in which too much time has elapsed and the system correctly declines to recover, and a "
    "disable option that reproduces the original behavior exactly, allowing a controlled before-and-after "
    "comparison.", styles["P2"]))
story.append(Paragraph(
    "Beyond automated testing, the output video was reviewed manually at several points in the recording "
    "— early, midway, and later — rather than relying solely on summary statistics. This manual review is "
    "what identified the detail regarding the stricter confidence metric remaining unchanged; reporting "
    "the improvement without that context would have overstated the result.", styles["P2"]))

# ---------------------------------------------------------- 8. Wrap up
story.append(Paragraph("Current Status and Next Steps", styles["Head2"]))
story.append(Paragraph(
    "The recommended, actively maintained version of the system now includes the recovery mechanism and "
    "the visualization improvement described above. The earlier three-signal redesign remains in the "
    "codebase for reference but is not the direction being pursued going forward.", styles["P2"]))
story.append(Paragraph(
    "Two open questions would benefit from further discussion: whether the stricter confidence check "
    "should apply a more lenient standard to a label restored through the recovery mechanism, as opposed "
    "to one matched under normal conditions; and whether the frame-edge detection artifact described above "
    "warrants a dedicated fix before the next round of testing.", styles["P2"]))
story.append(Paragraph(
    "The fuller technical documentation (architecture details, exact matching logic, and complete test "
    "results) and demonstration videos are available and can be provided on request.", styles["P2"]))

story.append(Spacer(1, 18))
story.append(rule())
story.append(Paragraph("Sincerely,", styles["Sign"]))
story.append(Paragraph("Aveesha Nishendra", styles["Sign"]))

# ---------------------------------------------------------- build
doc = SimpleDocTemplate(
    "BronchoTrack_V3_Update_Professional.pdf", pagesize=LETTER,
    leftMargin=0.9*inch, rightMargin=0.9*inch, topMargin=0.8*inch, bottomMargin=0.8*inch,
    title="BronchoTrack_V3 Progress Update", author="Aveesha Nishendra",
)
doc.build(story)
print("PDF built")
