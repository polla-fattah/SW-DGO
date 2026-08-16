"""
Physical SI Units, Metric Calibration, and Timing Constants for the SW-DGO Framework.
Standardizes conversions between continuous metric coordinates and discrete pixel grid spaces.
"""

# Spatial calibration: 1 pixel = 0.03 meters (30 mm per pixel) -> 1 meter = 33.333 pixels
PX_TO_M: float = 0.03
M_TO_PX: float = 1.0 / PX_TO_M

# Time and clock frequencies:
# - Physics integration and algorithmic replanning execute at 20 Hz (dt = 0.05 s)
# - GUI rendering interpolates smoothly at 60 FPS (16.6 ms)
PHYSICS_DT: float = 0.05
PLANNING_FREQ_HZ: float = 1.0 / PHYSICS_DT  # 20.0 Hz
GUI_FPS: int = 60

# Physical Vehicle Dimensions (Int-Cart / Clinical Pushchair / Luggage Trolley)
# Chassis footprint: 0.48 m width x 0.72 m length (16 px x 24 px)
ROBOT_WIDTH_M: float = 0.48
ROBOT_LENGTH_M: float = 0.72
ROBOT_RADIUS_M: float = 0.40  # Inscribed safety radius (13.3 px)
ROBOT_RADIUS_PX: float = ROBOT_RADIUS_M * M_TO_PX

# Non-Holonomic Kinematic Limits
ROBOT_VMAX_MPS: float = 1.20   # Maximum linear velocity: 1.2 m/s (~40 px/s)
ROBOT_VMAX_PXPS: float = ROBOT_VMAX_MPS * M_TO_PX  # 40.0 px/s
ROBOT_AMAX_MPS2: float = 1.50  # Maximum linear acceleration: 1.5 m/s^2
ROBOT_AMAX_PXPS2: float = ROBOT_AMAX_MPS2 * M_TO_PX # 50.0 px/s^2
ROBOT_WMAX_RADPS: float = 2.50 # Maximum angular steering velocity: 2.5 rad/s

# Safety Clearance Margins
SHELF_CLEARANCE_MARGIN_M: float = 0.54  # 18 pixels margin from fixture corners
SHELF_CLEARANCE_MARGIN_PX: float = SHELF_CLEARANCE_MARGIN_M * M_TO_PX
FOLLOWING_DISTANCE_GAP_M: float = 1.08  # 36 pixels anti-tailgating gap
FOLLOWING_DISTANCE_GAP_PX: float = FOLLOWING_DISTANCE_GAP_M * M_TO_PX

# Docking / arrival tolerance
# A mission counts as complete when the chassis reaches the bay within this radius.
# It is ~one cart length (0.72 m), i.e. the cart body is effectively in the bay.
# This MUST be identical for every planner: the previous revision allowed D2RO
# 0.84 m while holding the baselines to 0.36-0.42 m, which inflated D2RO's success
# rate relative to every method it was compared against.
ARRIVAL_RADIUS_M: float = 0.84
ARRIVAL_RADIUS_PX: float = ARRIVAL_RADIUS_M * M_TO_PX  # 28.0 px

# Onboard Perception
# Line-of-sight sensing radius. An agent may only fold humans it can actually
# observe into its own cost field; anything beyond this radius is knowable only
# through peer telemetry. Without this bound the agent is effectively omniscient
# and the anticipatory horizon-extension claim is vacuous, because there is no
# horizon left to extend.
SENSING_RADIUS_M: float = 7.2
SENSING_RADIUS_PX: float = SENSING_RADIUS_M * M_TO_PX  # 240 px

# Wireless V2V Mesh Parameters
V2V_MESH_COMM_RANGE_M: float = 10.5  # 350 pixels ad-hoc transmission radius
V2V_MESH_COMM_RANGE_PX: float = V2V_MESH_COMM_RANGE_M * M_TO_PX
V2V_PACKET_SIZE_BYTES: int = 64      # Standard telemetry packet size (Timestamp, Edge, Cost, ID)
V2V_TTL_HOPS: int = 3                # Maximum multi-hop forwarding hops
V2V_DECAY_RATE_PER_SEC: float = 0.1386294  # Exponential decay rate lambda (s^-1), half-life = 5 s

# Distributed Corridor Reservation Protocol
# Lease duration after which an unrefreshed claim is presumed abandoned. This is what
# prevents an agent that fails while holding a corridor from blocking it permanently.
CORRIDOR_LOCK_LEASE_S: float = 10.0

# Asymmetric Anisotropic Human Proxemics Parameters (HA-VLN 2.0 / Hall's Proxemics)
HUMAN_RADIUS_M: float = 0.36         # Physical human body radius (12 px)
HUMAN_RADIUS_PX: float = HUMAN_RADIUS_M * M_TO_PX
PROXEMIC_SIGMA_FRONT_M: float = 1.35 # Front intimate personal space: 1.35 m (45 px)
PROXEMIC_SIGMA_FRONT_PX: float = PROXEMIC_SIGMA_FRONT_M * M_TO_PX
PROXEMIC_SIGMA_SIDE_M: float = 0.90  # Lateral intimate personal space: 0.90 m (30 px)
PROXEMIC_SIGMA_SIDE_PX: float = PROXEMIC_SIGMA_SIDE_M * M_TO_PX
PROXEMIC_SIGMA_REAR_M: float = 0.60  # Rear intimate personal space: 0.60 m (20 px)
PROXEMIC_SIGMA_REAR_PX: float = PROXEMIC_SIGMA_REAR_M * M_TO_PX
# ---------------------------------------------------------------------------- #
# COST-TERM NORMALISATION (equivalent-detour metres)
#
# Every penalty term is expressed in the same physical unit as the distance term:
# metres of equivalent detour. This is what makes the weight vector
# [w_D, w_M, w_H, w_S] genuinely dimensionless, and it is what the phrase
# "dimensionally normalised weights" in the manuscript must actually mean.
#
# Before normalisation the terms were incommensurable by two orders of magnitude
# (edge distance 3-16 m against a proxemic penalty of ~2028 cost units), so the
# distance term was effectively inert and the planner minimised discomfort alone.
#
# Calibration targets, chosen against the 36 m x 24 m floor plan:
#   a full-intensity human intrusion  ~ 27 m of detour  (avoid unless it is the only route)
#   a congestion alert on an edge     ~ 20 m of detour  (prefer a parallel aisle)
#   a maximally tight fixture passage ~  6 m of detour  (comparable to one aisle)
# ---------------------------------------------------------------------------- #
PROXEMIC_AMPLITUDE: float = 12.0     # Detour-metres per metre of full-intensity intrusion
MESH_ALERT_EQUIV_M: float = 20.0     # Congestion alert magnitude (equivalent detour metres)
MESH_FOLLOW_BLOCK_EQUIV_M: float = 12.0  # Milder penalty when merely queued behind a peer

# Intimate-space boundary used for SAFETY EVENT COUNTING (not for the cost field).
# Must be identical for every planner, otherwise the social-compliance comparison
# between D2RO and the baselines is measured against different thresholds.
INTIMATE_RADIUS_M: float = 0.80
INTIMATE_RADIUS_PX: float = INTIMATE_RADIUS_M * M_TO_PX  # 26.67 px

# Separation below which two opposed carts in a single-file aisle are deemed to be
# in a head-on conflict (used for EVENT counting, not for the cost field).
# NOTE: this must exceed the kinetic safety separation (2 x safety bubble = 1.56 m),
# otherwise the separation controller holds the carts apart at 1.56 m and a standoff
# can never be registered -- which silently reported zero conflicts in every trial.
HEAD_ON_CONFLICT_RADIUS_M: float = 1.80
HEAD_ON_CONFLICT_RADIUS_PX: float = HEAD_ON_CONFLICT_RADIUS_M * M_TO_PX  # 25.0 px

# Kinetic Trolley Safety Envelope Parameters (S_trolley)
# The safety term has two physically distinct contributions:
#   (a) a STATIC geometric component penalising edges whose lateral clearance to fixed
#       shelf fixtures is smaller than the chassis envelope (r_robot + clearance margin);
#   (b) a DYNAMIC component penalising edges whose endpoint lies inside the kinetic
#       clearance bubble of a peer trolley (isotropic Gaussian, cf. Eq. 8).
TROLLEY_CLEARANCE_AMPLITUDE: float = 1.5    # Detour-metres per metre of fully-tight passage
TROLLEY_PEER_AMPLITUDE: float = 3.0         # Detour-metres for a peer occupying the endpoint
TROLLEY_PEER_SIGMA_M: float = 1.00          # Kinetic clearance radius (1.0 m)
TROLLEY_PEER_SIGMA_PX: float = TROLLEY_PEER_SIGMA_M * M_TO_PX  # 33.3 px

# 5-Component SW-DGO Cost Function Weights (Dimensionally Normalized)
WEIGHT_DISTANCE_WD: float = 1.0   # Baseline physical kinematic distance weight
WEIGHT_MESH_WM: float = 1.5       # V2V collaborative mesh congestion weight
WEIGHT_PROXEMIC_WH: float = 2.0   # Human proxemic discomfort field weight
WEIGHT_MUTEX_LOCK_WR: float = 1.0 # Single-file corridor mutex reservation weight
WEIGHT_TROLLEY_WS: float = 1.2    # Kinetic chassis safety bubble weight
