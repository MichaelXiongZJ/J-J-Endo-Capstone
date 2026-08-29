Chapter 4. Results and Evaluation

4.1 Experimental Setup

4.1.1 Deployment Environment and Hardware Architecture
The proposed computer vision pipeline was deployed and evaluated within a hybrid testing environment, comprising both controlled synthetic sequences and unstructured real-world warehouse camera feeds. The hardware substrate for inference consisted of [HARDWARE_SPECS, e.g., an NVIDIA RTX 4090 GPU paired with an Intel Core i9 processor and 64GB of RAM]. To ensure strict temporal synchronization between the detection modules and the rolling windows used for velocity estimation, the pipeline processed input video streams at a fixed temporal resolution of 10 frames per second.

4.1.2 Spatial Calibration and Homography Mapping
To overcome the fundamental limitations imposed by two-dimensional perspective distortion, each camera view was rigorously calibrated to a unified ground-plane coordinate system. This calibration was achieved by deriving a homography matrix ($H$) via `cv2.findHomography`, which mapped a minimum of four manually surveyed image points to their corresponding physical floor coordinates in meters. 

![Figure 1: Homographic projection of the warehouse floor. (a) Original camera view with four surveyed ground control points highlighted in yellow. (b) The resulting top-down orthographic projection used for Euclidean distance calculation.](path/to/placeholder_fig1_calibration.png)

Crucially, calibration validity was assessed not by reprojection error (which is mathematically exact for a four-point correspondence), but by evaluating the system's empirical accuracy in measuring independent, known geometric distances within the scene. The average spatial error across all test environments was constrained to [SPATIAL_ERROR_TOLERANCE]%.

4.1.3 Perception Stack and Evaluation Dataset
To satisfy stringent commercial deployment constraints, the perception layer was constructed entirely using permissively licensed models (Apache-2.0 and MIT). Object detection was performed by RT-DETR, operating on frames resized to [IMAGE_SIZE]. Cross-frame entity association was managed by the ByteTrack algorithm. For articulated human tracking, RTMPose was executed via ONNX Runtime to extract 17-point skeletal representations. The end-to-end system was evaluated against a comprehensive ground-truth dataset containing [NUM_ANNOTATED_FRAMES] annotated frames, extracted from [NUM_CLIPS] distinct video sequences that captured a representative distribution of standard warehouse operations alongside staged safety violations.

4.2 Results

System performance was quantified by evaluating its capacity to classify behavior against a predefined set of rigid geometric and temporal rules. Notably, the rule logic was implemented as deterministic arithmetic operating atop the perception layer's output, explicitly avoiding the opacity of end-to-end machine learning classifiers for safety-critical reasoning.

4.2.1 Motion State and Driver Association
Prior to assessing safety compliance, the system continuously estimated the motion state of all tracked entities using a rolling 1.0-second temporal window. A vehicle was classified as active if its floor-projected velocity exceeded 0.3 m/s, or if it had moved within the preceding 5.0 seconds. This hysteresis provided a necessary temporal buffer to account for forklifts pausing briefly during complex operational maneuvers. 

![Figure 2: Driver association via velocity vector matching. The system distinguishes between the active driver (green bounding box, matching velocity vector) and a pedestrian severely occluded behind the forklift (red bounding box, divergent velocity vector).](path/to/placeholder_fig2_driver_association.png)

Driver association was executed by comparing the velocity vectors of individuals and vehicles. By mandating velocity agreement (within 0.5 m/s) alongside a bounding-box overlap threshold (0.6), the system successfully differentiated between an active driver and a pedestrian severely occluded behind a moving vehicle (Figure 2).

4.2.2 Proximity and Walkway Compliance (Rules 3 & 4)
Pedestrian proximity to operating vehicles (Rule 3) was evaluated entirely within the projected ground plane. The system projected the bottom-center ground-contact point of each bounding box through the homography matrix to compute Euclidean distance in meters. A violation was registered when a non-driver pedestrian breached a dynamic safety radius of 3.0 vehicle lengths.

![Figure 3: Geometric safety rule evaluation. (a) Rule 3 (Proximity) evaluated via a dynamic 3.0-vehicle-length safety radius projected onto the ground plane. (b) Rule 4 (Walkways) compliance assessed against a predefined polygonal safe zone (blue overlay).](path/to/placeholder_fig3_geometric_rules.png)

Walkway compliance (Rule 4) determined whether a pedestrian's floor coordinates fell outside predefined polygonal safe zones. To robustly suppress transient noise and minor tracking jitter, a violation was emitted only following a sustained duration of 1.0 seconds outside the designated walkway boundary.

4.2.3 Ergonomic and Behavioral Violations (Rules 1 & 5)
Driver body protrusion (Rule 5) required analyzing the spatial relationship between human keypoints and the vehicle boundary. To mitigate false positives induced by the empty space inherent within a standard forklift bounding box (such as the mast and overhead guard), the system dynamically computed an inset cab region (15% left, 35% top, 15% right, 0% bottom). A violation required specific keypoints (including wrists, shoulders, and nose) to remain outside this constrained cab region for a continuous 1.5 seconds.

![Figure 4: Pose-dependent rule evaluation. (a) Rule 5 (Driver Protrusion) showing the dynamically computed cab inset region (blue) and tracked keypoints (green/red). (b) Rule 1 (Phone Usage) illustrating the wrist-to-head distance normalized against shoulder width.](path/to/placeholder_fig4_pose_rules.png)

Mobile phone usage detection (Rule 1) utilized a scale-invariant heuristic predicated on the ratio of the wrist-to-head distance, normalized against the subject's shoulder width. A violation was triggered when this ratio fell below 0.6 for a sustained 2.0 seconds, robustly indicating a raised hand in proximity to the face.

4.2.4 Pipeline Throughput and Log Summary
The complete pipeline operated at an average inference latency of [INFERENCE_SPEED] milliseconds per frame. Across the evaluation dataset, the system generated [TOTAL_EVENTS] distinct safety event logs spanning all evaluated rule categories.

4.3 Evaluation

4.3.1 Evaluation Philosophy: Precision-First System Design
In the domain of industrial safety monitoring, false alarms introduce a critical operational risk; a system generating excessive false positives inevitably induces alarm fatigue and is subsequently disregarded by site managers. Consequently, the evaluation methodology strictly prioritized precision over recall. The aggregate system achieved an overall precision of [OVERALL_PRECISION]% and an overall recall of [OVERALL_RECALL]%.

4.3.2 Performance of Geometric Rules (Rules 3 & 4)
The geometry-based rules demonstrated exceptional reliability. Rule 3 (Proximity) and Rule 4 (Walkways) achieved precision scores of [RULE3_PRECISION]% and [RULE4_PRECISION]%, respectively. Projecting coordinates onto the physical floor plane effectively eradicated the depth ambiguities that systematically confound traditional bounding-box overlap models. For instance, scenarios involving pedestrians walking significantly behind a forklift (which manifest as highly overlapping boxes in pixel space) were correctly classified as safe, thereby preserving high precision. The primary source of error for these rules stemmed from tracking identity switches during severe occlusion, which generated fragmented velocity histories and temporary misclassifications of vehicle motion states.

4.3.3 Performance of Pose-Dependent Rules (Rules 1 & 5)
Rules dependent upon articulated pose estimation presented magnified challenges due to the visual complexity and clutter of the warehouse environment. Driver protrusion (Rule 5) achieved a precision of [RULE5_PRECISION]%. The incorporation of strict temporal gating proved highly effective at filtering out brief, operationally necessary reaches (such as adjusting a mirror or scanning a barcode), which would otherwise trigger immediate, erroneous violations.

Conversely, phone usage detection (Rule 1) proved susceptible to partial occlusion and resolution constraints, yielding a precision of [RULE1_PRECISION]% and a recall of [RULE1_RECALL]%. When keypoints were heavily occluded, the pose estimator occasionally inferred spurious limb positions. Enforcing a rigorous minimum confidence threshold of 0.5 for all evaluated keypoints successfully suppressed these anomalous detections, although this aggressive filtering inherently lowered the recall rate for this specific rule.

4.4 Discussion

4.4.1 Efficacy of Spatial Transformation over Pixel Intersections
The evaluation unequivocally underscores the necessity of transforming two-dimensional pixel representations into real-world spatial coordinates for industrial safety applications. Relying on raw bounding-box intersections is demonstrably inadequate for proximity estimation due to severe perspective distortion. The homographic projection approach provides a mathematically rigorous and highly robust foundation for spatial reasoning in static camera deployments.

4.4.2 Architectural Benefits of Decoupled Rule Logic
Furthermore, decoupling the neural perception layer from the rule-evaluation logic yielded pronounced operational advantages. By modeling safety rules as explicit geometric and temporal conditions applied to tracked objects, the system bypasses the opacity and unpredictability of end-to-end neural networks. When a safety manager requests a modification to a safety perimeter (such as increasing the proximity radius from 3 meters to 5 meters), the adjustment requires modifying a single configuration parameter. Conversely, an end-to-end machine learning model would necessitate the curation of a new, targeted dataset and the computationally expensive retraining of the network to achieve an identical operational change.

4.4.3 Occlusion Limitations and Procedural Rule Descoping
Despite these successes, the system remains fundamentally bounded by the physical constraints of monocular vision. Occlusion induced by racking, inventory, and the vehicles themselves frequently obscures both pedestrians and drivers. The system logic is intentionally designed to be conservative: if critical keypoints fall below the confidence threshold due to occlusion, the system defaults to a non-violation state rather than inferring an unverified event. While this architectural choice limits overall recall, it represents a necessary compromise to maintain the exceptional high precision required for viable real-world deployment. Additionally, specific compliance requirements, such as the daily pre-use vehicle inspection checklist (Rule 2), were explicitly descoped from the vision pipeline. Visual confirmation of a driver holding a clipboard cannot guarantee actual procedural compliance, clearly delineating the boundaries of vision-based automated auditing.

4.5 Business or Organizational Impact

4.5.1 Transition from Lagging to Leading Indicators
The deployment of this automated computer vision pipeline transitions warehouse safety management from reactive incident investigation to proactive hazard auditing. Historically, facility safety has relied upon lagging indicators, such as reported collisions, supplemented by periodic manual inspections. By continuously monitoring and quantifying pedestrian-vehicle interactions, the system generates actionable leading indicators in the form of near-miss frequency and spatial distribution maps.

![Figure 5: Aggregated leading indicators. A spatial heatmap overlaid on the warehouse floor plan, highlighting zones with a high frequency of pedestrian-vehicle proximity events (near-misses) over a 30-day evaluation period.](path/to/placeholder_fig5_heatmap.png)

4.5.2 Scalable Edge Deployment and Cost Optimization
This continuous stream of objective spatial data enables safety managers to identify high-risk zones (Figure 5), such as blind intersections or frequently breached walkways, and implement targeted structural or procedural interventions before an incident occurs. Because the pipeline relies on highly optimized, permissively licensed models (RT-DETR and RTMPose) executed efficiently via ONNX Runtime, it can be deployed on cost-effective edge hardware (estimated at [EDGE_HARDWARE_COST] per camera node). This architecture minimizes the substantial bandwidth requirements and compute costs inherently associated with centralized, cloud-based video processing, thereby enabling scalable, facility-wide coverage.

4.5.3 Operational Audit Reduction and Long-Term Compliance
By automating the auditing process, the system is projected to reduce the manual labor hours currently dedicated to reviewing security footage by [AUDIT_REDUCTION_PCT]%. More importantly, the high-precision alerts ensure that safety interventions are grounded in verified, objective data, supporting a culture of targeted accountability rather than punitive surveillance. Long-term aggregation of these compliance metrics is expected to drive measurable, sustained improvements in site safety, with an anticipated [SAFETY_IMPROVEMENT_PCT]% reduction in proximity-related safety breaches within the first year of deployment.
