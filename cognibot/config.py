"""All tunable constants in one place."""

# ---- World ----
GRID_W, GRID_H = 20, 15  # grid size in cells
BASE = (1, 1)  # robot start / recharge base
N_SURVIVORS = 5  # survivors hidden in each map
N_FIRE_SEEDS = 6  # number of fire sources
N_DEBRIS = 14  # number of debris (wall) segments

# ---- Sensors ----
SENSOR_RADIUS = 2  # robot "camera" sees a 5x5 square around itself
HEAT_DETECT_PROB = 0.70  # heat sensor detects an adjacent fire 70% of the time
HEAT_FALSE_ALARM = 0.05  # heat sensor false alarm on smoke without fire

# ---- Robot body ----
MAX_HEALTH = 100
FIRE_DAMAGE = 35  # damage taken when entering a fire cell
MAX_BATTERY = 300  # one step costs one unit of battery

# ---- Planning costs ----
DANGER_WEIGHT = 30  # extra path cost for a cell with danger = 1.0
UNKNOWN_COST = 1.3  # cost of stepping into an unexplored cell

# ---- Rewards (used by Q-learning and for scoring all agents) ----
R_STEP = -0.05
R_RESCUE = 10.0
R_FIRE = -8.0
R_HOME = 5.0
R_DESTROYED = -20.0
R_BATTERY_DEAD = -5.0

# ---- Learning ----
ALPHA = 0.2  # Q-learning rate
GAMMA = 0.95  # discount factor
Q_SCALE = 0.1  # weight of learned Q-value added to hand-made utility
PRIOR_FIRE_ALPHA = 1  # Beta prior for P(fire | smoke): a brand-new robot believes
PRIOR_FIRE_BETA = 49  # smoke is almost harmless (1/50 = 2%) and must learn otherwise
COMMIT_MARGIN = 1.25  # switch intention only if new option is 25% better (BDI-style commitment)
