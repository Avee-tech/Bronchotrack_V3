// Content data for the BronchoTrack_V3 thesis. Kept separate from the
// docx layout/build script so the writing can be edited without touching
// formatting code. Each chapter is an array of "blocks":
//   { t: "p", x: "..." }            -- body paragraph
//   { t: "h2", x: "..." }           -- subheading within the chapter
//   { t: "bullet", x: "..." }       -- bullet list item
//   { t: "image" }                  -- the pipeline flowchart figure
//   { t: "caption", x: "..." }      -- figure/table caption
//   { t: "results_table" }          -- the before/after results table
//   { t: "testsuite_table" }        -- the automated test-suite breakdown table

const TITLE = "Robust Real-Time Airway Branch Localization for Bronchoscopy: Extending BronchoTrack with Anchor Re-Acquisition";

const AUTHOR = "Aveesha Nishendra";

const ABSTRACT = [
  "Flexible bronchoscopy relies on the operator's own visual memory and experience to navigate a " +
  "branching airway tree that offers few distinguishing landmarks, particularly beyond the first few " +
  "generations of the bronchial tree. Camera-only localization systems, which determine a bronchoscope's " +
  "position from the video feed alone by comparing detected airway openings against a patient-specific " +
  "three-dimensional airway model, offer a way to support navigation without the cost, disposable " +
  "hardware, or procedural complications associated with electromagnetic navigation systems.",
  "This thesis presents BronchoTrack_V3, an implementation and extension of a recently published " +
  "branch-level bronchoscopic localization method. The baseline system detects airway lumens with a " +
  "segmentation-based detector, tracks them across frames, and associates each visible lumen with a " +
  "branch label by comparing its position against a semantic airway graph built from the patient's " +
  "computed tomography scan. A specific limitation was identified in this baseline: because new " +
  "detections are only associated relative to a single persistently tracked “anchor” lumen, the " +
  "system permanently stops reporting a location if that anchor is lost and never recovers, even once " +
  "clear airway views return. On a 1,084-frame patient bronchoscopy video, this failure mode left the " +
  "system unable to report a location for approximately 82% of the video following the anchor's loss.",
  "Two candidate solutions were developed and evaluated. The first was a substantial redesign that " +
  "replaced the anchor entirely with a continuous, per-frame re-evaluation combining three independent " +
  "signals — a scale-free diameter-growth motion model, point-based bearing matching, and a whole-mask " +
  "boundary-distance ratio method — fused through a Kalman filter. This redesign was fully implemented and " +
  "tested but performed worse overall than the existing baseline on the same evaluation video, and was " +
  "therefore not adopted. The second, adopted solution instead retains the original anchor-based design " +
  "and adds a bounded, two-stage recovery mechanism: a self re-acquisition check against the last known " +
  "anchor position, followed by a virtual-anchor fallback that continues to allow new associations without " +
  "requiring the original lumen to reappear. On the same evaluation video, this increased the proportion " +
  "of frames with a reported location from 15.3% to 39.9%, a factor of approximately 2.6, while leaving " +
  "the system's stricter, geometry-verified confidence metric essentially unchanged (14.9% to 14.4%), " +
  "consistent with the recovery mechanism and the verification check addressing two different questions.",
  "A secondary contribution replaces the system's point-marker visualization with a full segmentation-mask " +
  "overlay, which improves visual verifiability of detections and incidentally surfaced a minor, " +
  "pre-existing frame-edge detection artifact. The thesis concludes that a bounded, targeted correction to " +
  "an already-sound design outperformed a more ambitious redesign for this failure mode, and outlines " +
  "concrete directions for future work, including relaxing the confidence threshold specifically for " +
  "recovered labels and automating the CT-to-airway-graph pipeline that currently precedes the system.",
];

const ACKNOWLEDGMENTS = [
  "I would like to thank {Supervisor Name} for their guidance and feedback throughout this project, and " +
  "the School of Civil & Mechanical Engineering at Curtin University for the resources made available to " +
  "carry it out. I am also grateful to my family and friends for their support over the course of this " +
  "research project.",
];

const NOMENCLATURE = [
  ["3D", "Three-dimensional"],
  ["CT", "Computed tomography"],
  ["EM", "Electromagnetic"],
  ["EMA", "Exponential moving average"],
  ["EMN", "Electromagnetic navigation bronchoscopy"],
  ["FPS", "Frames per second"],
  ["IoU", "Intersection over Union"],
  ["JSON", "JavaScript Object Notation, the file format used for the airway graph and frame logs"],
  ["MOT", "Multi-object tracking"],
  ["Re-ID", "Re-identification (of a tracked object by visual appearance)"],
  ["SLAM", "Simultaneous localization and mapping"],
  ["SORT", "Simple Online and Realtime Tracking"],
  ["YOLO", "You Only Look Once (a family of real-time object detection/segmentation models)"],
];

// ---------------------------------------------------------------------
// Chapter 1 -- Introduction
// ---------------------------------------------------------------------
const CH1 = [
  { t: "h2", x: "1.1 Background and Motivation" },
  { t: "p", x:
    "Flexible bronchoscopy is a minimally invasive procedure in which a thin, camera-tipped instrument " +
    "is navigated through a patient's airway tree to inspect the airway wall, collect tissue samples, or " +
    "guide the delivery of a therapeutic device. Unlike many other endoscopic procedures, the airway tree " +
    "branches repeatedly — from the trachea into the left and right main bronchi, and from there into " +
    "progressively smaller sub-branches — and offers few visually distinguishing landmarks beyond the " +
    "first two or three generations of branching. In practice, bronchoscopists navigate largely from " +
    "memory, training, and an internally held mental model of the patient's airway anatomy built from " +
    "pre-procedural imaging. Disorientation within this branching structure is a recognized and " +
    "well-documented challenge, and it disproportionately affects less experienced operators and cases " +
    "involving anatomically atypical or distorted airways." },
  { t: "p", x:
    "A number of commercial and research systems exist to assist with this navigation problem, most " +
    "notably electromagnetic navigation bronchoscopy (EMN), which tracks a sensor-tipped instrument " +
    "against a pre-procedural three-dimensional airway map built from the patient's computed tomography " +
    "(CT) scan. EMN systems are effective but require dedicated sensor hardware, disposable single-use " +
    "components, specialized capital equipment, and carry a measurable procedural complication rate. This " +
    "has motivated a separate line of research into camera-only, or vision-based, localization: systems " +
    "that infer the bronchoscope's position purely from the live video feed, without any additional " +
    "tracking hardware in the patient or the procedure room." },
  { t: "p", x:
    "This thesis is concerned with one such vision-based approach, based on a recently published method " +
    "(BronchoTrack) that determines the bronchoscope's current airway branch by detecting the openings " +
    "visible in each video frame, tracking them consistently as the camera moves, and associating each " +
    "tracked opening with a branch of a pre-built three-dimensional airway graph. This project implements " +
    "that method as a faithful baseline, identifies a specific and practically significant limitation in " +
    "it through direct evaluation on real patient video, and develops, evaluates, and compares two " +
    "candidate solutions to that limitation." },

  { t: "h2", x: "1.2 Problem Statement" },
  { t: "p", x:
    "The baseline association method identifies a newly visible airway opening only by comparing it " +
    "against a single, currently trusted, previously identified opening — referred to throughout this " +
    "thesis as the anchor. This is an effective strategy under normal conditions, since the anchor gives " +
    "the system a stable reference point from which to reason about which children in the airway graph a " +
    "newly visible opening could plausibly belong to. However, bronchoscopy video routinely includes " +
    "brief but severe interruptions: motion blur during rapid instrument movement, the camera briefly " +
    "pressing against the airway wall, mucus or fluid obscuring the lens, or short stretches in which the " +
    "detector simply fails to identify any opening at all. If every currently tracked anchor is lost " +
    "during such an interruption, the baseline system has no mechanism to recover. It permanently stops " +
    "reporting a location for the remainder of the video, even once a clear, easily identifiable airway " +
    "view returns moments later." },
  { t: "p", x:
    "This is not a theoretical concern. As documented in Chapter 4, evaluating the baseline system on a " +
    "1,084-frame patient bronchoscopy video showed the system losing its last anchor roughly one-fifth of " +
    "the way through the recording and subsequently reporting no location for approximately 82% of the " +
    "remaining video. A localization system intended to support navigation throughout an entire procedure " +
    "cannot tolerate a single transient interruption silently ending its usefulness for the remainder of " +
    "the case." },

  { t: "h2", x: "1.3 Aim and Objectives" },
  { t: "p", x:
    "The aim of this project is to identify, characterize, and correct this anchor-loss failure mode in " +
    "a working bronchoscopic localization system, without compromising the accuracy the baseline system " +
    "already achieves under normal conditions. This aim is addressed through the following objectives:" },
  { t: "bullet", x: "Implement a faithful baseline reproduction of the published BronchoTrack method, including lumen " +
    "detection, multi-object tracking, and graph-based airway association, as a reference against which " +
    "any modification can be directly compared." },
  { t: "bullet", x: "Characterize the anchor-loss limitation quantitatively on real patient video, rather than by " +
    "inspection of the method alone." },
  { t: "bullet", x: "Design, implement, and evaluate a substantial alternative redesign that removes the anchor " +
    "dependency entirely, to test whether a more ambitious architectural change is warranted." },
  { t: "bullet", x: "Design, implement, and evaluate a smaller, targeted recovery mechanism that retains the original " +
    "anchor-based design, as a more conservative alternative to the redesign." },
  { t: "bullet", x: "Compare both candidates directly on the same evaluation video and adopt the better-performing " +
    "approach as the recommended, maintained system." },
  { t: "bullet", x: "Improve the system's output visualization from a single point marker to a full detected-shape " +
    "overlay, to make the system's behaviour easier to verify visually." },

  { t: "h2", x: "1.4 Scope" },
  { t: "p", x:
    "This project is concerned with branch-level localization — determining which named airway branch " +
    "the bronchoscope currently occupies — rather than continuous six-degree-of-freedom pose estimation " +
    "within a branch. The airway graph itself is assumed to already exist as a labeled three-dimensional " +
    "structure derived from a prior CT scan; the process of segmenting a CT volume and extracting that " +
    "graph is treated as a prerequisite input to this system rather than a contribution of this thesis, " +
    "though its automation is discussed as future work in Chapter 6. Evaluation in this project is " +
    "conducted on recorded patient bronchoscopy video rather than live, in-procedure operation, consistent " +
    "with the retrospective evaluation methodology used in the published baseline method this project " +
    "builds on." },

  { t: "h2", x: "1.5 Thesis Structure" },
  { t: "p", x:
    "Chapter 2 reviews existing approaches to bronchoscopic navigation, situates the baseline method this " +
    "project extends within that literature, and identifies the specific gap this thesis addresses. " +
    "Chapter 3 describes the system architecture, the materials and data used, and the methodology behind " +
    "both candidate solutions to the anchor-loss problem, together with the evaluation methodology used to " +
    "compare them. Chapter 4 presents the quantitative results of that evaluation and discusses their " +
    "implications, including the results of the visualization improvement. Chapter 5 draws conclusions " +
    "from this work, and Chapter 6 outlines directions for future work." },
];

// ---------------------------------------------------------------------
// Chapter 2 -- Background / Literature Review
// ---------------------------------------------------------------------
const CH2 = [
  { t: "p", x:
    "Approaches to assisting bronchoscopic navigation broadly fall into three categories: electromagnetic " +
    "and other external-sensor tracking, image-registration methods that align the live video with a " +
    "pre-built three-dimensional model (sometimes described as virtual bronchoscopy or visual SLAM), and " +
    "detection-and-association methods that reason about discrete anatomical landmarks — principally " +
    "airway bifurcations — visible in the video itself. This chapter reviews representative work in each " +
    "category before introducing the specific method this thesis builds on." },

  { t: "h2", x: "2.1 Electromagnetic Navigation and Robotic-Assisted Bronchoscopy" },
  { t: "p", x:
    "Electromagnetic navigation bronchoscopy (EMN) is the most clinically established approach to " +
    "bronchoscopic guidance. A thin-slice (around 1 mm) CT scan of the chest, typically acquired at full " +
    "inspiration, is used to build a virtual bronchoscopic model of the airway tree, segment a target " +
    "lesion, and plan a pathway from the central airway to that lesion. During the procedure, a sensor " +
    "probe passed through the bronchoscope's working channel is tracked within an externally generated " +
    "electromagnetic field and registered against this pre-built map, allowing the operator to follow the " +
    "planned pathway to the target [1]. Two platforms dominate current clinical use, superDimension " +
    "(Medtronic) and the SPiN system (Olympus) [1], [8]." },
  { t: "p", x:
    "Reported diagnostic yield for EMN varies widely across the literature, from as low as 33% to as high " +
    "as 88% [1], a range Lin, Ho, and Frye attribute largely to an early evidence base of small, " +
    "single-centre studies [8]. The NAVIGATE trial, the largest prospective multi-centre evaluation of EMN " +
    "to date, offers a more generalizable figure: its one-year U.S. cohort of 1,157 patients reported a " +
    "73% diagnostic yield and 69% sensitivity for malignancy at a median lesion size of 20 mm, with a 2.9% " +
    "pneumothorax rate, while a subsequent two-year follow-up expanding to 1,388 patients across the United " +
    "States and Europe reported a somewhat lower 67.8% yield, 62.6% sensitivity, and 4.7% pneumothorax " +
    "rate. A positive radiographic “bronchus sign” — an airway visibly leading into the lesion on the " +
    "planning CT — was present in only about half of cases, and was, alongside operator experience and " +
    "the use of rapid on-site cytological evaluation, one of the strongest predictors of diagnostic success " +
    "[8]." },
  { t: "p", x:
    "Even at this validated level, EMN's yield still falls short of CT-guided transthoracic needle " +
    "aspiration, and at least one meta-analysis found no clear advantage over the substantially cheaper " +
    "combination of conventional bronchoscopy with radial endobronchial ultrasound (r-EBUS) [8]. The " +
    "leading technical explanation for this gap is CT-to-body divergence: the planning CT is acquired at " +
    "full lung inflation, while the procedure itself is performed under general anaesthesia and " +
    "positive-pressure ventilation nearer functional residual capacity, so the lesion's true position at " +
    "the time of sampling can differ from its planned position by several centimetres — an effect reported " +
    "to be more pronounced for lower-lobe lesions and to have shifted lesions by as much as 4 cm in at " +
    "least one study [8]. r-EBUS is frequently added specifically to confirm real-time lesion position " +
    "against this drift, though it depends on a positive bronchus sign to be useful at all — one study " +
    "found diagnostic yield fell from 79% to 31% without one — and has itself been reported to give " +
    "false-positive confirmation against atelectatic tissue that ultrasonically resembles the lesion [8]. " +
    "A further practical concern specific to EMN is the theoretical risk of electromagnetic interference " +
    "with implanted cardiac devices, though the small safety studies conducted to date have not observed " +
    "any actual disruption to device function [8]." },
  { t: "p", x:
    "Robotic-assisted bronchoscopy (RAB) is a more recent, related approach that keeps the same " +
    "CT-planned-pathway strategy while adding continuous robotic control, intended to improve reach and " +
    "stability at the periphery of the lung where a conventional, EMN-guided bronchoscope becomes " +
    "difficult to steer precisely [8]. Three platforms have reached clinical use: the Monarch system " +
    "(Auris Health), controlled with a handheld interface and tracked electromagnetically, which wedges a " +
    "6.0 mm outer sheath at the segmental airways and advances a 4.4 mm inner scope the rest of the way to " +
    "the target; the Ion platform (Intuitive Surgical), controlled with a trackball and wheel and tracked " +
    "instead through fibre-optic shape sensing along a 3.5 mm fully articulating catheter, avoiding an " +
    "external electromagnetic field entirely; and the Galaxy system (Noah Medical), a disposable, " +
    "EMN-guided bronchoscope that adds continuous digital tomosynthesis and augmented fluoroscopy so the " +
    "operator retains real-time visual confirmation throughout sampling itself, rather than relying solely " +
    "on the pre-planned route [8]." },
  { t: "p", x:
    "For the purposes of this thesis, the significance of both EMN and RAB is less their individual " +
    "diagnostic performance than what they hold in common: whatever their differences in tracking " +
    "modality, both remain dependent on dedicated capital equipment, proprietary single-use catheters or " +
    "sheaths, and, for the electromagnetically tracked platforms, an external field generator. RAB " +
    "improves on EMN's reach and stability, but does not remove the cost and hardware burden that motivates " +
    "camera-only localization in the first place (Section 1.1) — it relocates that burden onto a more " +
    "capable, but still equally hardware-dependent, platform. This is the gap the vision-based methods " +
    "reviewed in the remainder of this chapter are aimed at." },

  { t: "h2", x: "2.2 Image-Registration and Landmark-Based Vision Methods" },
  { t: "p", x:
    "A separate line of work uses the bronchoscope's video feed itself, without external sensors, either " +
    "by registering the live view against a virtual rendering of the CT-derived airway model (an approach " +
    "sometimes framed as visual simultaneous localization and mapping, or visual SLAM) or by detecting " +
    "specific anatomical landmarks — principally the bifurcations where one airway branch splits into two " +
    "or more children — and reasoning about the instrument's position relative to those landmarks." },
  { t: "p", x:
    "Shen, Giannarou, Shah, and Yang's BRANCH method is representative of the landmark-based approach: it " +
    "constructs a shape-context airway descriptor specifically to detect and characterize bifurcations in " +
    "the distal airway, aiming to support localization deeper into the bronchial tree than image-appearance " +
    "matching alone typically allows [3]. This reflects a broader theme in the literature: as the airway " +
    "tree branches into smaller and more visually similar sub-branches, appearance-based registration " +
    "against the CT-derived model becomes progressively less reliable, and structural landmarks — " +
    "specifically, the openings that lead to each branch — become a more robust cue than overall scene " +
    "appearance." },
  { t: "p", x:
    "This same insight underlies the detection-and-association family of methods this thesis is most " +
    "directly built on, which treats each airway opening as a discrete, trackable object rather than " +
    "registering the whole visual scene, and is described in the following section." },

  { t: "h2", x: "2.3 Detection-and-Association Methods and the BronchoTrack Baseline" },
  { t: "p", x:
    "Tian, Liao, Huang, Yang, Wu, Chen, Li, and Liu's BronchoTrack method — the method this thesis " +
    "implements and extends — frames branch-level localization as a three-stage pipeline: a lightweight " +
    "detector identifies airway lumens (openings) visible in each frame, a multi-object tracking stage " +
    "maintains each lumen's identity consistently across frames despite rapid camera motion, and a " +
    "training-free association stage matches each currently tracked lumen against a semantic airway graph " +
    "that encodes the known hierarchy and branching structure of the patient's own airway tree [4]. This " +
    "avoids the need to register the full visual scene against a rendered CT model on every frame, and " +
    "instead reduces the problem to matching a small number of discrete, well-defined openings. The " +
    "original work reports 85.64% localization accuracy across nine patient datasets up to the fourth " +
    "airway generation, and successful navigation to the eighth generation in porcine in-vivo testing [4]." },
  { t: "p", x:
    "The same research group's subsequent PANS system reformulates branch-level localization " +
    "probabilistically, aiming for improved robustness under the kind of transient visual ambiguity and " +
    "interruption this thesis is specifically concerned with [5]. This is a useful point of comparison for " +
    "the present work: rather than adopting a full probabilistic reformulation of the underlying " +
    "association method, this thesis instead evaluates whether a smaller, more targeted correction to the " +
    "existing anchor-based design — confined specifically to the anchor-loss failure mode — can achieve " +
    "a comparable practical improvement without the added complexity of a broader redesign." },
  { t: "p", x:
    "The multi-object tracking stage in this family of methods draws on general-purpose tracking " +
    "techniques developed outside the medical imaging literature. Bewley, Ge, Ott, Ramos, and Upcroft's " +
    "SORT (Simple Online and Realtime Tracking) algorithm — which associates detections across frames " +
    "using a Kalman filter for motion prediction combined with the Hungarian algorithm for optimal " +
    "assignment — established the basic motion-based matching pattern that this project's tracking stage, " +
    "and BronchoTrack's original tracking stage, both follow [6]. The lumen detector itself is a " +
    "real-time object detection and segmentation network from the YOLO (You Only Look Once) family, " +
    "originally introduced by Redmon, Divvala, Girshick, and Farhadi as a single-pass alternative to " +
    "slower region-proposal-based detectors [7], and realized in this project using the YOLO26 " +
    "architecture described by Sapkota, Cheppally, Sharda, and Karkee [8]." },

  { t: "h2", x: "2.4 Gap Addressed by This Thesis" },
  { t: "p", x:
    "The methods reviewed above establish detection-and-association against a semantic airway graph as an " +
    "effective, hardware-free approach to branch-level bronchoscopic localization, with BronchoTrack " +
    "specifically as the direct basis for this project's implementation. However, published evaluations of " +
    "this family of methods report overall accuracy figures without isolating the specific behaviour of " +
    "the system during and immediately after a total tracking interruption — a scenario this thesis found, " +
    "through direct evaluation on real patient video, to be a severe and specific failure mode of the " +
    "anchor-dependent association design. This thesis addresses that gap directly: it characterizes the " +
    "anchor-loss failure mode quantitatively, evaluates a substantial redesign intended to eliminate it " +
    "entirely, and evaluates a smaller, targeted recovery mechanism as a more conservative alternative, " +
    "reporting an honest, like-for-like comparison between the two on the same evaluation data." },
];

// ---------------------------------------------------------------------
// Chapter 3 -- Experimental Procedure
// ---------------------------------------------------------------------
const CH3 = [
  { t: "h2", x: "3.1 System Architecture" },
  { t: "p", x:
    "The implemented system processes bronchoscopy video one frame at a time through four sequential " +
    "stages, illustrated in Figure 1: lumen detection identifies the airway openings visible in the " +
    "current frame; multi-lumen tracking maintains each opening's identity consistently from one frame to " +
    "the next; airway association determines which branch of the patient's airway graph each currently " +
    "visible, tracked opening most likely corresponds to; and localization takes a vote among all " +
    "currently labeled openings to report the bronchoscope's overall current branch. Two design choices " +
    "in this architecture are significant enough to warrant discussion before the individual stages are " +
    "described in detail." },
  { t: "p", x:
    "First, the detection stage produces a full segmentation mask for each detected opening, not merely " +
    "a bounding box. This is a deliberate choice: an opening's shape and apparent size carry useful " +
    "evidence for identifying which branch it is (a larger, rounder opening close to the camera reads " +
    "differently from a small, foreshortened one further away), and this information would be discarded " +
    "under a box-only detector. In the implemented system, this mask information is used specifically by " +
    "the association stage's diameter-based verification step, described in Section 3.4; the tracking " +
    "stage itself remains deliberately box-based, following the established motion-and-appearance " +
    "matching pattern used by general-purpose trackers such as SORT [6], since full mask-to-mask matching " +
    "on every frame would add tracking-stage cost without a demonstrated accuracy benefit for this " +
    "system's design. This trade-off is revisited as a future-work item in Chapter 6." },
  { t: "p", x:
    "Second, the system performs an independent verification step after association: once an opening is " +
    "matched to a branch by position, its apparent size at the camera's estimated distance from that " +
    "branch is separately checked against the size the airway graph predicts for that branch at that " +
    "distance. A label is only treated as fully confirmed — and rendered as a confirmed detection in the " +
    "output visualization — once both the positional match and this size check agree. This is a stricter " +
    "standard than it may initially appear, and it becomes directly relevant to interpreting the results " +
    "in Chapter 4, since this verification metric behaves differently from the overall labeling rate " +
    "reported alongside it." },
  { t: "image" },
  { t: "caption", x: "Figure 1. System architecture: how a single video frame moves through the implemented pipeline, from input to output." },

  { t: "h2", x: "3.2 Materials" },
  { t: "p", x:
    "Three inputs are required by the system. The first is bronchoscopy video; the evaluation reported in " +
    "Chapter 4 uses a 1,084-frame recording of a real patient bronchoscopy procedure. The second is a " +
    "patient-specific airway graph: a labeled, hierarchical three-dimensional representation of the " +
    "patient's own airway tree, derived in advance from their CT scan and encoding, for each named branch, " +
    "its parent branch, its generation (depth from the trachea), its centerline, and its physical radius " +
    "at points along that centerline. This graph is treated in this project as a pre-existing input " +
    "artifact rather than a contribution of this thesis; its own construction pipeline, which uses " +
    "surface-mesh network extraction to convert a segmented CT-derived mesh into this labeled graph " +
    "structure, is summarized only briefly here and revisited as a candidate for automation in Chapter 6. " +
    "The third input is the trained lumen detector itself: a YOLO26 segmentation model [8], fine-tuned to " +
    "detect airway lumens and output, for each detection, a bounding box, a confidence score, and a " +
    "segmentation mask polygon outlining the opening's true shape." },

  { t: "h2", x: "3.3 Detection and Tracking" },
  { t: "p", x:
    "Each frame is passed through the YOLO26 segmentation detector, which returns zero or more detections " +
    "above a configured confidence threshold. Each detection carries a bounding box, confidence score, and " +
    "mask polygon. These detections are passed to a multi-object tracking stage responsible for maintaining " +
    "a consistent identity for each physical airway opening across frames, so that the same opening is not " +
    "mistaken for a newly appearing one purely because the camera moved between frames." },
  { t: "p", x:
    "Tracking follows a Kalman-filter-based motion model over each tracked object's bounding box " +
    "(center position, height, and aspect ratio), predicting each tracklet's expected position in the " +
    "current frame before matching it against the current frame's detections — the same basic pattern " +
    "established by SORT [6]. Matching proceeds in two tiers, following a BYTE-style strategy: " +
    "high-confidence detections are matched against existing tracklets using a combination of motion " +
    "(predicted position overlap) and appearance similarity, drawn from a ResNet50 feature embedding of " +
    "each detection's image crop, updated over time with an exponential moving average to remain robust to " +
    "gradual appearance change; low-confidence detections are matched using motion alone, which allows the " +
    "tracker to continue following an opening even through a few frames of degraded detection confidence " +
    "without immediately discarding it. A tracklet that is not matched to any detection for 30 consecutive " +
    "frames is dropped." },

  { t: "h2", x: "3.4 Baseline Airway Association" },
  { t: "p", x:
    "The association stage is responsible for assigning each currently tracked lumen a branch label from " +
    "the airway graph, and is where this project's core contribution is situated. The baseline design, " +
    "following the published BronchoTrack method [4], operates around a single trusted reference " +
    "detection at a time, referred to throughout this thesis as the anchor: the tracklet most recently and " +
    "confidently identified as occupying the airway branch the bronchoscope currently sits in." },
  { t: "p", x:
    "Association proceeds in three steps. First, at the very start of a video, before any anchor exists, " +
    "the system bootstraps by matching the visible lumens at the first bifurcation (the carina, where the " +
    "trachea splits into the left and right main bronchi) directly against the graph's known geometry, " +
    "since no prior reference point exists yet to match relative to. Second, for every subsequent frame, " +
    "each newly visible, currently unlabeled lumen is compared against the anchor's known position and the " +
    "graph's predicted layout of the anchor's own child and sibling branches, correcting for the camera's " +
    "estimated roll angle about its own optical axis, to determine which child branch it most plausibly " +
    "corresponds to. Third, once a candidate label passes this positional matching step, it is subjected to " +
    "the diameter:distance verification described in Section 3.1: the candidate's apparent size in the " +
    "frame is compared against the size the graph predicts for that branch at the camera's current " +
    "estimated distance, and the label is only treated as fully confirmed if both checks agree." },
  { t: "p", x:
    "This anchor-centred design is effective under normal conditions, since it reduces every association " +
    "decision to a comparison against one well-understood, currently visible reference point rather than " +
    "an unconstrained search over the whole graph. Its limitation, characterized quantitatively in Chapter " +
    "4, is that it has no defined behaviour for the case where the anchor itself is lost — through motion " +
    "blur, occlusion, or a period of detector failure — and no other opening has yet been confidently " +
    "identified to replace it." },

  { t: "h2", x: "3.5 Candidate Solution A: Continuous Multi-Signal Re-Evaluation" },
  { t: "p", x:
    "The first candidate solution investigated was a substantial redesign that removes the anchor concept " +
    "entirely. Rather than persistently depending on any single tracked object, this design re-evaluates " +
    "every currently visible lumen against every plausible candidate branch on every single frame, " +
    "combining three independent signals into one fused confidence score per candidate branch." },
  { t: "p", x:
    "The first signal is a diameter-growth motion model: each tracklet's own apparent diameter is " +
    "tracked over time with a Kalman filter, and its smoothed rate of change is used to derive a " +
    "scale-free time-to-contact estimate — the number of frames until the camera reaches that opening — " +
    "without requiring any camera calibration, since the estimate depends only on the ratio of current " +
    "diameter to its own rate of growth. These implied time-to-contact values are cross-checked, though " +
    "never used to calibrate the model itself, against the median closing speed observed across the " +
    "video's own already-completed branch transitions, flagging candidates whose implied approach speed is " +
    "wildly inconsistent with that history as implausible. The second signal is a direct re-derivation of " +
    "the baseline's own positional matching idea: comparing each tracklet's segmentation-mask center " +
    "against the graph's predicted bearing to each candidate child branch. The third signal, intended to " +
    "make genuine use of the detector's mask output rather than reducing it to a single point, compares " +
    "each tracklet's whole mask shape against the graph's predicted branch geometry, using an " +
    "area-equivalent diameter computed from the full mask boundary and the nearest boundary-to-boundary " +
    "gap between two mask polygons, rather than a simple center-to-center distance." },
  { t: "p", x:
    "These three signals are combined per candidate branch through a Hungarian assignment followed by a " +
    "per-branch Kalman filter that smooths the fused confidence over time, and a branch transition is only " +
    "committed once a candidate's smoothed confidence clears a threshold for a run of consecutive frames, " +
    "rather than on a single strong frame. This design was fully implemented, covered by an automated test " +
    "suite (Section 3.7), and evaluated on the same 1,084-frame video used to characterize the baseline's " +
    "anchor-loss limitation." },

  { t: "h2", x: "3.6 Candidate Solution B: Bounded Anchor Recovery" },
  { t: "p", x:
    "The second, more conservative candidate solution retains the baseline's original anchor-centred " +
    "design in full, and adds a bounded recovery mechanism specifically for the case where every anchor is " +
    "lost simultaneously. The system retains a snapshot of the most recently confirmed anchor — its " +
    "position, its branch label, and the frame at which it was last confidently observed — even after that " +
    "opening is no longer visible or trackable." },
  { t: "p", x:
    "If every anchor is lost, recovery proceeds in two stages. First, the system checks whether any " +
    "currently unlabeled opening is likely the same physical opening reappearing — that is, the " +
    "bronchoscope's own current position, returning into view after a brief interruption — by measuring " +
    "how closely its position overlaps the anchor snapshot's last known position (intersection-over-union " +
    "against that frozen reference). If the overlap exceeds a threshold, the opening is relabeled directly " +
    "with the anchor's own former label, without requiring a fresh graph-based match. Second, if no " +
    "sufficiently close match is found, the system falls back to treating the anchor snapshot's last known " +
    "position as a virtual reference point, allowing newly visible openings nearby to still be matched " +
    "against the graph's predicted layout of the anchor's children and siblings, even though the original " +
    "physical opening itself has not reappeared." },
  { t: "p", x:
    "In either case, once an opening is successfully re-identified through this mechanism, the system " +
    "resumes normal anchor-based operation immediately and treats the newly identified opening as the " +
    "current anchor; this is not a one-time reset, and the mechanism can recur as many times as needed " +
    "over the course of a video. A safeguard is included to avoid acting on a stale reference: if more than " +
    "approximately three seconds have elapsed since the retained anchor snapshot was itself confidently " +
    "observed, the system treats it as unreliable and withholds recovery until a new anchor is established " +
    "through the normal bootstrap or matching process, rather than propagating an outdated position " +
    "indefinitely." },

  { t: "h2", x: "3.7 Evaluation Methodology" },
  { t: "p", x:
    "Both candidate solutions were evaluated on the same 1,084-frame real patient bronchoscopy video used " +
    "to originally characterize the baseline's anchor-loss limitation, under otherwise identical detector, " +
    "tracker, and graph configuration, to allow a direct, like-for-like comparison. Two metrics are " +
    "reported for each run: the proportion of frames for which the system reports any location at all " +
    "(reflecting whether the anchor-loss failure mode has been addressed), and the proportion of frames " +
    "for which the stricter, geometry-verified diameter:distance check described in Section 3.1 is " +
    "additionally satisfied (reflecting whether label confidence itself has been affected, independent of " +
    "the recovery behaviour). The quantitative results of this comparison, and the resulting decision to " +
    "adopt Candidate Solution B over Candidate Solution A, are reported in Chapter 4." },
  { t: "p", x:
    "In addition to this evaluation-video comparison, the implementation includes an automated regression " +
    "test suite covering the individual software components of the system, developed and maintained " +
    "throughout the project as each stage was implemented. This suite comprises 58 tests across three " +
    "modules — the general pipeline, the baseline association logic including the recovery mechanism, and " +
    "the Candidate Solution A redesign — summarized in Table 1, and is used during development to catch " +
    "regressions in individual components; it is distinct from, and does not substitute for, the " +
    "end-to-end evaluation-video comparison reported in Chapter 4, which was carried out directly against " +
    "real patient video with the recovery mechanism enabled and disabled under otherwise identical " +
    "settings." },
  { t: "testsuite_table" },
  { t: "caption", x: "Table 1. Automated regression test suite, by module." },

  { t: "h2", x: "3.8 Visualization" },
  { t: "p", x:
    "The system's output video originally marked each identified lumen with a small dot at its " +
    "geometric center. As a secondary improvement, this was replaced with a rendering of the detector's " +
    "actual segmentation mask as a filled, translucent, outlined overlay, gated on the diameter:distance " +
    "verification check described in Section 3.1 so that only fully confirmed detections receive the full " +
    "overlay treatment, with a fallback to the original dot marker where no usable mask is available. This " +
    "change and its incidental findings are discussed in Chapter 4." },
];

// ---------------------------------------------------------------------
// Chapter 4 -- Results and Discussion
// ---------------------------------------------------------------------
const CH4 = [
  { t: "h2", x: "4.1 Baseline Anchor-Loss Characterization" },
  { t: "p", x:
    "Running the unmodified baseline system on the 1,084-frame evaluation video confirmed the failure " +
    "mode motivating this thesis. The system lost its last active anchor roughly one-fifth of the way " +
    "through the recording — during a stretch of rapid camera motion and partial occlusion — and, having " +
    "no mechanism to recover, reported no location for approximately 82% of the remaining video, despite " +
    "the video subsequently returning to clear, easily identifiable airway views on multiple occasions. " +
    "This result, on its own, establishes that the anchor-loss limitation identified in Chapter 1 is not " +
    "an edge case but a severe and practically significant failure mode for this class of system." },

  { t: "h2", x: "4.2 Comparison of Candidate Solutions" },
  { t: "p", x:
    "Candidate Solution A (continuous multi-signal re-evaluation) was fully implemented and evaluated on " +
    "the same video. Upon direct comparison against the baseline, it performed worse overall: the added " +
    "complexity of continuously re-deriving and fusing three independent signals every frame did not " +
    "translate into a corresponding improvement in reliability, and in practice introduced its own new " +
    "sources of noise, particularly in the motion model's diameter-growth estimate during frames where a " +
    "tracked opening's apparent size briefly fluctuated for reasons unrelated to genuine approach or " +
    "recession. Based on this direct comparison, Candidate Solution A was set aside rather than adopted." },
  { t: "p", x:
    "This negative result is nonetheless a meaningful finding in its own right, and is reported here " +
    "rather than omitted. It indicates that the baseline's core anchor-based association strategy was " +
    "fundamentally sound, and that the anchor-loss problem specifically required a bounded, targeted " +
    "correction rather than a wholesale architectural replacement. Candidate Solution A remains in the " +
    "project's codebase, fully implemented and tested, should a future extension of this work find a " +
    "specific use for its continuous re-evaluation design — for instance, a scenario with a much noisier " +
    "detector where no single anchor can be trusted for long — that this project's own evaluation video " +
    "did not present." },
  { t: "p", x:
    "Candidate Solution B (bounded anchor recovery) was evaluated on the same video under otherwise " +
    "identical settings, comparing a run with the recovery mechanism disabled against one with it enabled. " +
    "Table 2 reports the results of this comparison." },
  { t: "results_table" },
  { t: "caption", x: "Table 2. Same 1,084-frame evaluation video and settings, Candidate Solution B, before and after the recovery mechanism." },
  { t: "p", x:
    "The proportion of frames with a reported location increased by a factor of approximately 2.6, " +
    "directly confirming that the system no longer permanently stops reporting a location following a " +
    "total anchor-loss interruption of the kind characterized in Section 4.1. Unlike Candidate Solution A, " +
    "this improvement was achieved without a corresponding regression in overall reliability, since the " +
    "recovery mechanism only activates in the specific circumstance where no anchor currently exists, and " +
    "leaves the system's normal anchor-based matching behaviour, including the diameter:distance " +
    "verification step, entirely unchanged the rest of the time." },

  { t: "h2", x: "4.3 The Confirmed-Detection Metric" },
  { t: "p", x:
    "One result in Table 2 warrants direct discussion: the stricter, geometry-verified confidence metric " +
    "remained essentially unchanged before and after the recovery mechanism was enabled (14.9% to 14.4%). " +
    "This is not an inconsistency, and it does not indicate that the recovery mechanism failed to work as " +
    "intended. The recovery mechanism and the diameter:distance verification check answer two different " +
    "questions: the recovery mechanism determines whether the system can resume reporting a location at " +
    "all following an interruption, while the verification check separately determines how strictly the " +
    "system should trust any given match, independent of how that match came to exist. Because recovery " +
    "restores a label through the same position-matching logic used everywhere else in the baseline " +
    "design, and does not relax the subsequent verification step in any way, a recovered label is held to " +
    "exactly the same strict standard as one identified under normal conditions. Whether this verification " +
    "threshold should be deliberately relaxed specifically for labels restored through recovery, given " +
    "that a recovered anchor is by definition working from a position estimate briefly out of date, is " +
    "identified as an open design question and is discussed further as future work in Chapter 6." },

  { t: "h2", x: "4.4 Visualization Improvement" },
  { t: "p", x:
    "Replacing the point-marker output with a full segmentation-mask overlay, as described in Section " +
    "3.8, considerably improved the ability to visually verify what the system is detecting in the output " +
    "video, since a viewer can directly compare the rendered mask shape against the visible airway opening " +
    "rather than inferring correctness from a single point. This change also surfaced a minor, " +
    "pre-existing detection artifact that had not been visually obvious under the previous point-marker " +
    "rendering: an opening located near the edge of the video frame is occasionally detected with an " +
    "irregular, oversized mask, because the detector's mask boundary follows the edge of the visible frame " +
    "rather than terminating cleanly at the true physical edge of the opening. This is a property of the " +
    "raw detection output itself rather than a defect introduced by the tracking or association stages, " +
    "and is noted as a candidate for a dedicated filtering step in future work (Chapter 6) should it prove " +
    "distracting in practice." },

  { t: "h2", x: "4.5 Summary" },
  { t: "p", x:
    "Taken together, these results support adopting Candidate Solution B — the bounded, two-stage anchor " +
    "recovery mechanism — as the system's recommended, maintained design. It directly and substantially " +
    "addresses the anchor-loss failure mode characterized in Section 4.1, without the reliability " +
    "regression observed in the more ambitious Candidate Solution A redesign, and without disturbing the " +
    "baseline's existing, already-effective normal-condition behaviour. The one open question this " +
    "evaluation surfaces — whether the confirmed-detection verification threshold should differ for " +
    "recovered labels specifically — is a refinement to an already-working mechanism rather than evidence " +
    "against the mechanism itself, and is carried forward as future work." },
];

// ---------------------------------------------------------------------
// Chapter 5 -- Conclusions
// ---------------------------------------------------------------------
const CH5 = [
  { t: "p", x:
    "This thesis set out to identify and correct a specific limitation in a published vision-based " +
    "bronchoscopic localization method: the total and permanent loss of location reporting following any " +
    "interruption severe enough to lose every currently tracked anchor lumen simultaneously. This " +
    "limitation was first confirmed quantitatively rather than assumed, by evaluating a faithful baseline " +
    "reproduction of the published method on a real, 1,084-frame patient bronchoscopy video, where it left " +
    "the system unable to report a location for approximately 82% of the video following an anchor-loss " +
    "event roughly one-fifth of the way through the recording." },
  { t: "p", x:
    "Two candidate solutions were developed, implemented, and evaluated under identical conditions. The " +
    "more substantial redesign — replacing the anchor concept entirely with a continuous, multi-signal " +
    "per-frame re-evaluation — was found, on direct comparison, to perform worse overall than the existing " +
    "baseline, and was not adopted. The smaller, more conservative solution — retaining the original " +
    "anchor-based design and adding a bounded, two-stage recovery mechanism triggered only when every " +
    "anchor is simultaneously lost — increased the proportion of the evaluation video with a reported " +
    "location by a factor of approximately 2.6, while leaving the system's independent, stricter " +
    "geometry-verification metric essentially unaffected, consistent with the two mechanisms addressing " +
    "genuinely separate questions. This solution was adopted as the system's recommended, actively " +
    "maintained design." },
  { t: "p", x:
    "A secondary improvement, replacing the system's point-marker output visualization with a full " +
    "segmentation-mask overlay, measurably improved the visual verifiability of the system's detections " +
    "and, as a direct consequence of that improved verifiability, surfaced a minor pre-existing detection " +
    "artifact at the video frame's edge that had not previously been visually apparent." },
  { t: "p", x:
    "The central methodological conclusion of this work is that, for the specific failure mode " +
    "investigated here, a bounded and targeted correction to an already-sound design outperformed a more " +
    "ambitious architectural redesign intended to eliminate the same failure mode by construction. This is " +
    "a useful and generalizable finding for future work on this and related systems: it demonstrates that " +
    "the anchor-based association strategy underlying BronchoTrack-style methods is fundamentally robust, " +
    "and that isolated, well-characterized failure modes within such a design are often better addressed " +
    "with equally isolated corrections than with wholesale replacement." },
];

// ---------------------------------------------------------------------
// Chapter 6 -- Future Work
// ---------------------------------------------------------------------
const CH6 = [
  { t: "p", x:
    "Several concrete directions for future work follow directly from the results and open questions " +
    "identified in Chapter 4." },
  { t: "h2", x: "6.1 Confidence Threshold for Recovered Labels" },
  { t: "p", x:
    "As discussed in Section 4.3, the recovery mechanism restores a location label without relaxing the " +
    "subsequent diameter:distance verification check that governs whether a label is displayed as fully " +
    "confirmed. Since a recovered anchor is, by definition, working from a position estimate that may be " +
    "briefly out of date, future work should investigate whether a deliberately more lenient verification " +
    "threshold — applied specifically to labels restored through recovery, rather than globally — would " +
    "improve the confirmed-detection rate during and immediately after a recovery event without " +
    "compromising overall label reliability." },
  { t: "h2", x: "6.2 Frame-Edge Detection Artifact" },
  { t: "p", x:
    "The frame-edge mask distortion artifact surfaced by the visualization improvement in Section 4.4 " +
    "was not directly addressed in this project, since it is a property of the raw detector output rather " +
    "than of the tracking or association stages this thesis is centred on. A dedicated filtering or " +
    "post-processing step — for instance, suppressing or flagging masks whose boundary closely follows the " +
    "video frame's own edge — is a well-scoped and low-risk piece of future work that would further " +
    "improve output quality." },
  { t: "h2", x: "6.3 Automating the CT-to-Airway-Graph Pipeline" },
  { t: "p", x:
    "This project treats the patient-specific, CT-derived airway graph as a pre-existing input, consistent " +
    "with the scope defined in Chapter 1. Automating the full pipeline from a raw CT scan through airway " +
    "segmentation, three-dimensional surface mesh reconstruction, and semantic graph extraction into a " +
    "single, streamlined preprocessing step would make the overall system substantially more practical to " +
    "deploy against new patients, and is a natural, self-contained extension of this work." },
  { t: "h2", x: "6.4 Re-examining Candidate Solution A in a Noisier Setting" },
  { t: "p", x:
    "As discussed in Section 4.2, Candidate Solution A's continuous multi-signal re-evaluation design was " +
    "not adopted because it underperformed the simpler baseline on this project's evaluation video. This " +
    "does not rule out the design being genuinely useful in a different operating regime — for instance, " +
    "against a substantially noisier detector, a lower frame rate, or an airway anatomy with unusually " +
    "ambiguous branch geometry, where no single anchor could plausibly be trusted for an extended period. " +
    "Future work could evaluate Candidate Solution A specifically under such conditions, rather than " +
    "against the same evaluation video used throughout this thesis, before concluding more generally on " +
    "its usefulness." },
  { t: "h2", x: "6.5 Broader Clinical Evaluation" },
  { t: "p", x:
    "All results in this thesis are drawn from a single 1,084-frame patient bronchoscopy video. A broader " +
    "evaluation across multiple patients, airway anatomies, and bronchoscopy equipment would be needed " +
    "before any conclusion here could be considered clinically generalizable, and is the natural next step " +
    "beyond the scope of this thesis." },
];

const REFERENCES = [
  "[1]  E. M. Pickering, O. Kalchiem-Dekel, and A. Sachdeva, “Electromagnetic navigation bronchoscopy: a comprehensive review,” AME Medical Journal, vol. 3, 2018. doi: 10.21037/amj.2018.11.04.",
  "[2]  J. Lin, E. Ho, and L. Frye, “Guided Bronchoscopy for Peripheral Pulmonary Lesion Sampling: The Pros and Cons of Electromagnetic Navigation Bronchoscopy and Robotic-Assisted Bronchoscopy,” Current Pulmonology Reports, vol. 13, pp. 95–102, 2024. doi: 10.1007/s13665-023-00330-z.",
  "[3]  M. Shen, S. Giannarou, P. L. Shah, and G.-Z. Yang, “BRANCH: Bifurcation Recognition for Airway Navigation based on struCtural cHaracteristics,” in Medical Image Computing and Computer-Assisted Intervention – MICCAI 2017, Lecture Notes in Computer Science, vol. 10434, Springer, 2017. doi: 10.1007/978-3-319-66185-8_21.",
  "[4]  Q. Tian, H. Liao, X. Huang, B. Yang, J. Wu, J. Chen, L. Li, and H. Liu, “BronchoTrack: Airway Lumen Tracking for Branch-Level Bronchoscopic Localization,” arXiv:2402.12763, 2024.",
  "[5]  Q. Tian, Z. Chen, H. Liao, X. Huang, B. Yang, L. Li, and H. Liu, “PANS: Probabilistic Airway Navigation System for Real-Time Robust Bronchoscope Localization,” in Medical Image Computing and Computer Assisted Intervention – MICCAI 2024, Lecture Notes in Computer Science, vol. 15006, Springer, 2024, pp. 466–476. doi: 10.1007/978-3-031-72089-5_44.",
  "[6]  A. Bewley, Z. Ge, L. Ott, F. Ramos, and B. Upcroft, “Simple Online and Realtime Tracking,” arXiv:1602.00763, 2016.",
  "[7]  J. Redmon, S. Divvala, R. Girshick, and A. Farhadi, “You Only Look Once: Unified, Real-Time Object Detection,” in Proc. IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016. arXiv:1506.02640.",
  "[8]  R. Sapkota, R. H. Cheppally, A. Sharda, and M. Karkee, “YOLO26: Key Architectural Enhancements and Performance Benchmarking for Real-Time Object Detection,” arXiv:2509.25164, 2025.",
];

module.exports = {
  TITLE, AUTHOR, ABSTRACT, ACKNOWLEDGMENTS, NOMENCLATURE,
  CH1, CH2, CH3, CH4, CH5, CH6, REFERENCES,
};
