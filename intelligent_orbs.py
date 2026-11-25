"""
Intelligent Orbs - an artificial-life ecosystem that lives on top of your desktop.

Every orb runs a sense -> think -> steer loop:

  * SENSE  - it looks around (vision range depends on curiosity and size), rating
             threats, prey, food, potential mates, allies and the mouse cursor.
  * THINK  - a utility AI scores every possible behaviour. Each score is biased by
             the orb's evolved neural network and by what it has learned during
             its own life (behaviours that paid off get preferred).
  * STEER  - Reynolds-style steering: pursuit with intercept prediction, stalking
             and pouncing, pack flanking, evasive juking, wall avoidance,
             flocking and wandering.

Evolution: offspring get a crossover of their parents' brains and traits plus
mutation. The best genomes ever seen are kept in a gene pool that is saved to
disk (orb_genepool.json), so every run starts from smarter ancestors.

Controls: use the tray icon (pause, toggle HUD / debug view, spawn, feed, quit).
"""

import sys
import os
import math
import json
import random
import signal
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np

try:
    from PyQt5 import QtCore, QtGui, QtWidgets
    HAVE_QT = True
except ImportError:  # The simulation itself runs headless (tests, benchmarks).
    HAVE_QT = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATS_FILE = os.path.join(BASE_DIR, "orb_stats.json")
GENEPOOL_FILE = os.path.join(BASE_DIR, "orb_genepool.json")

BEHAVIORS = ("exploring", "foraging", "hunting", "fleeing", "mating", "resting")
EXPLORE, FORAGE, HUNT, FLEE, MATE, REST = range(len(BEHAVIORS))

THINK_INTERVAL = 5          # frames between decisions (staggered per orb)
MATURITY_AGE = 900          # frames before an orb can reproduce
MIN_RADIUS = 7.0
MAX_RADIUS = 32.0
MAX_ENERGY = 100.0
HALL_OF_FAME_SIZE = 20

CURSOR = "cursor"           # sentinel target for cursor-hunting orbs
ALARM_RADIUS = 260          # how far an alarm call carries
ALARM_COOLDOWN = 150        # frames between alarm calls from the same orb
HISTORY_EVERY = 60          # frames between stats-history samples (1 simulated second)
HISTORY_LENGTH = 1800       # samples kept (30 minutes)

# --- Time: one in-sim day lasts 2 minutes at 60 fps, a season 3 days, a year 12 days.
DAY_FRAMES = 7200
DAYS_PER_SEASON = 3
SEASONS = ("spring", "summer", "autumn", "winter")
SEASON_PARAMS = {
    #           food growth, food amount, food value, cold (metabolism), breeding, food colour
    "spring": dict(food_rate=1.3, food_target=1.15, food_value=1.0, cold=1.0, breeding=1.4,
                   food_color=(120, 255, 150)),
    "summer": dict(food_rate=1.0, food_target=1.0, food_value=1.0, cold=0.95, breeding=1.0,
                   food_color=(210, 255, 110)),
    "autumn": dict(food_rate=0.8, food_target=0.85, food_value=1.35, cold=1.05, breeding=0.7,
                   food_color=(255, 165, 70)),
    "winter": dict(food_rate=0.35, food_target=0.45, food_value=0.9, cold=1.35, breeding=0.35,
                   food_color=(195, 230, 255)),
}
# Where the fertile patches drift each season (fractions of the screen) - herds migrate after them.
SEASON_ZONES = {"spring": (0.25, 0.5), "summer": (0.5, 0.3), "autumn": (0.75, 0.5), "winter": (0.5, 0.65)}


# ----------------------------------------------------------------------------
# Small math helpers (plain floats are much faster than numpy for 2D vectors)
# ----------------------------------------------------------------------------

def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


def normalize(x, y):
    d = math.hypot(x, y)
    if d < 1e-9:
        return 0.0, 0.0
    return x / d, y / d


def limit(x, y, m):
    d = math.hypot(x, y)
    if d > m > 0:
        s = m / d
        return x * s, y * s
    return x, y


def segment_hits_rect(ax, ay, bx, by, rect):
    """Liang-Barsky: does the segment a->b pass through the rectangle?"""
    x0, y0, x1, y1 = rect
    dx, dy = bx - ax, by - ay
    t0, t1 = 0.0, 1.0
    for p, q in ((-dx, ax - x0), (dx, x1 - ax), (-dy, ay - y0), (dy, y1 - ay)):
        if p == 0:
            if q < 0:
                return False
        else:
            t = q / p
            if p < 0:
                if t > t1:
                    return False
                if t > t0:
                    t0 = t
            else:
                if t < t0:
                    return False
                if t < t1:
                    t1 = t
    return t0 < t1


def closest_on_segment(px, py, ax, ay, bx, by):
    dx, dy = bx - ax, by - ay
    l2 = dx * dx + dy * dy
    if l2 < 1e-9:
        return ax, ay
    t = clamp(((px - ax) * dx + (py - ay) * dy) / l2, 0.0, 1.0)
    return ax + t * dx, ay + t * dy


def segments_cross(ax, ay, bx, by, cx, cy, dx, dy):
    """Do segments a-b and c-d intersect?"""
    def orient(px, py, qx, qy, rx, ry):
        return (qx - px) * (ry - py) - (qy - py) * (rx - px)
    o1 = orient(ax, ay, bx, by, cx, cy)
    o2 = orient(ax, ay, bx, by, dx, dy)
    o3 = orient(cx, cy, dx, dy, ax, ay)
    o4 = orient(cx, cy, dx, dy, bx, by)
    return (o1 > 0) != (o2 > 0) and (o3 > 0) != (o4 > 0)


class Wall:
    """A temporary straight wall drawn with the god tools (a thick line segment)."""
    __slots__ = ("x0", "y0", "x1", "y1", "half", "life", "max_life")

    def __init__(self, x0, y0, x1, y1, life=3600, thickness=12):
        self.x0, self.y0, self.x1, self.y1 = x0, y0, x1, y1
        self.half = thickness / 2
        self.life = self.max_life = life


def point_in_rect(x, y, rect, margin=0.0):
    return rect[0] - margin <= x <= rect[2] + margin and rect[1] - margin <= y <= rect[3] + margin


_NAME_START = ("b", "br", "c", "d", "dr", "f", "fl", "g", "gr", "h", "j", "k", "kr", "l", "m", "n", "p",
               "pr", "qu", "r", "s", "sk", "sl", "sn", "st", "t", "tr", "v", "w", "z", "th", "sh")
_NAME_VOWEL = ("a", "e", "i", "o", "u", "ai", "ea", "ou", "y", "ee", "oo")
_NAME_END = ("", "", "n", "x", "r", "k", "l", "s", "th", "sh", "m", "ck", "nd", "rk", "sp", "z")
_NAME_TAIL = ("a", "i", "o", "y", "ix", "el", "an", "in", "ie")
HOUSE_WORDS = ("Ember", "Thorn", "Mist", "Frost", "Gale", "Fang", "Bloom", "Shade", "Ash", "Briar", "Cinder",
               "Dusk", "Flint", "Glimmer", "Hollow", "Ivy", "Moss", "Nettle", "Onyx", "Pike", "Quartz", "Rook",
               "Sable", "Tide", "Umber", "Vale", "Wisp", "Yew", "Zephyr", "Storm", "Rime", "Coral", "Drift",
               "Echo", "Fern", "Garnet", "Haze", "Iris", "Jade", "Kestrel")


# Random syllables occasionally spell something rude; those names are rerolled.
_NAME_BLOCKLIST = ("sex", "fuck", "fuk", "shit", "cunt", "cock", "dick", "piss", "puss", "tit",
                   "twat", "fag", "nig", "rape", "slut", "whor", "wank", "jizz", "spunk", "porn",
                   "nazi", "kkk", "bitch", "bastar", "anal", "anus", "dildo", "pube", "boob")


def make_name():
    """A short pronounceable name like 'Brisk', 'Vou' or 'Sneath'."""
    while True:
        s = random.choice(_NAME_START) + random.choice(_NAME_VOWEL) + random.choice(_NAME_END)
        if random.random() < 0.35:
            s += random.choice(_NAME_TAIL)
        if not any(bad in s for bad in _NAME_BLOCKLIST):
            return s.capitalize()


def genetic_distance(a, b):
    """0 = identical twins, ~1 = completely different."""
    return (abs(a.aggression - b.aggression) +
            abs(a.curiosity - b.curiosity) +
            abs(a.social_tendency - b.social_tendency) +
            abs(a.cursor_affinity - b.cursor_affinity) * 0.5 +
            abs(a.size_gene - b.size_gene) / (MAX_RADIUS - MIN_RADIUS)) / 4.5


def _softmax_vec(x):
    """Stable softmax for numpy vectors (kept for backwards compatibility)."""
    x = np.asarray(x, dtype=np.float64)
    x = x - np.max(x)
    e = np.exp(x)
    s = e.sum()
    return (e / s) if s > 0 else np.ones_like(x) / len(x)


class SpatialHash:
    """Uniform grid so each orb only examines things near it."""

    def __init__(self, cell=150):
        self.cell = cell
        self.buckets = {}

    def rebuild(self, items):
        c = self.cell
        buckets = {}
        for it in items:
            buckets.setdefault((int(it.x // c), int(it.y // c)), []).append(it)
        self.buckets = buckets

    def query(self, x, y, r):
        c = self.cell
        r2 = r * r
        out = []
        get = self.buckets.get
        for gx in range(int((x - r) // c), int((x + r) // c) + 1):
            for gy in range(int((y - r) // c), int((y + r) // c) + 1):
                bucket = get((gx, gy))
                if bucket:
                    for it in bucket:
                        dx = it.x - x
                        dy = it.y - y
                        if dx * dx + dy * dy <= r2:
                            out.append(it)
        return out


# ----------------------------------------------------------------------------
# Visual effects
# ----------------------------------------------------------------------------

class ExplosionParticle:
    """A particle for explosion effects when orbs die."""

    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        angle = random.uniform(0, math.tau)
        speed = random.uniform(1.5, 8)
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed
        self.life = self.max_life = random.randint(40, 70)
        self.size = random.uniform(2, 6)
        self.color = color  # (r, g, b)
        self.gravity = 0.08

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vy += self.gravity
        self.vx *= 0.96
        self.vy *= 0.96
        self.life -= 1
        self.size *= 0.96
        return self.life > 0

    def get_alpha(self):
        return int(255 * max(0, self.life) / self.max_life)


class RingEffect:
    """Expanding ring used for kills, births and repelled attacks."""

    def __init__(self, x, y, color, max_radius=45, life=30):
        self.x, self.y = x, y
        self.color = color
        self.max_radius = max_radius
        self.life = self.max_life = life

    def update(self):
        self.life -= 1
        return self.life > 0

    @property
    def progress(self):
        return 1.0 - self.life / self.max_life


# ----------------------------------------------------------------------------
# Brain
# ----------------------------------------------------------------------------

class OrbBrain:
    """Small evolvable NumPy neural network.

    Inputs are a normalised summary of what the orb perceives; outputs are
    biases on each behaviour's utility plus two personality modulators
    (courage and sprint eagerness). Weights are inherited and mutated, never
    trained with gradients - evolution does the learning.
    """

    N_INPUTS = 22
    N_HIDDEN = 16
    N_OUTPUTS = len(BEHAVIORS) + 2

    def __init__(self, randomize=True):
        rng = np.random.default_rng()
        scale = 1.0 if randomize else 0.0
        self.W1 = (rng.normal(0, 0.45, (self.N_HIDDEN, self.N_INPUTS)) * scale).astype(np.float32)
        self.b1 = (rng.normal(0, 0.1, self.N_HIDDEN) * scale).astype(np.float32)
        self.W2 = (rng.normal(0, 0.45, (self.N_OUTPUTS, self.N_HIDDEN)) * scale).astype(np.float32)
        self.b2 = np.zeros(self.N_OUTPUTS, dtype=np.float32)

    PARAMS = ("W1", "b1", "W2", "b2")

    def forward(self, x):
        h = np.tanh(self.W1 @ x + self.b1)
        return np.tanh(self.W2 @ h + self.b2)

    forward_np = forward  # backwards-compatible name

    def copy(self):
        b = OrbBrain(randomize=False)
        for name in self.PARAMS:
            setattr(b, name, getattr(self, name).copy())
        return b

    @classmethod
    def crossover(cls, a, b, bias_a=0.5):
        """Uniform crossover: each weight comes from parent a with probability bias_a."""
        rng = np.random.default_rng()
        child = cls(randomize=False)
        for name in cls.PARAMS:
            pa, pb = getattr(a, name), getattr(b, name)
            mask = rng.random(pa.shape) < bias_a
            setattr(child, name, np.where(mask, pa, pb).astype(np.float32))
        return child

    def mutate(self, rate=0.08, sigma=0.3):
        rng = np.random.default_rng()
        for name in self.PARAMS:
            p = getattr(self, name)
            mask = rng.random(p.shape) < rate
            p += (mask * rng.normal(0, sigma, p.shape)).astype(np.float32)
            np.clip(p, -4.0, 4.0, out=p)
        return self

    def to_dict(self):
        return {name: getattr(self, name).round(5).tolist() for name in self.PARAMS}

    @classmethod
    def from_dict(cls, d):
        b = cls(randomize=False)
        for name in cls.PARAMS:
            arr = np.asarray(d[name], dtype=np.float32)
            want = getattr(b, name).shape
            if name == "W1" and arr.ndim == 2 and arr.shape[0] == want[0] and arr.shape[1] < want[1]:
                # Brain from an older version with fewer senses: new inputs start disconnected.
                arr = np.pad(arr, ((0, 0), (0, want[1] - arr.shape[1])))
            if arr.shape != want:
                raise ValueError(f"brain shape mismatch for {name}")
            setattr(b, name, arr.astype(np.float32))
        return b


# ----------------------------------------------------------------------------
# Food
# ----------------------------------------------------------------------------

class Food:
    __slots__ = ("x", "y", "value", "age", "phase", "eaten")

    def __init__(self, x, y, value):
        self.x, self.y = x, y
        self.value = value
        self.age = 0
        self.phase = random.uniform(0, math.tau)
        self.eaten = False


class Lure:
    """The glowing bait dangling in front of a lurking angler. To other orbs it
    looks exactly like food - until it's too late."""
    __slots__ = ("x", "y", "value", "age", "phase", "eaten", "angler")

    def __init__(self, angler):
        self.angler = angler
        self.x, self.y = angler.x, angler.y
        self.value = 13.0          # a juicy-looking mote
        self.age = 100
        self.phase = random.uniform(0, math.tau)
        self.eaten = False


class ScentField:
    """Smells that linger on the ground, on a coarse grid:

    * food   - laid by orbs that have just eaten, ant-style: following it uphill leads to food
    * prey   - body scent everyone leaves as they move; hungry hunters track it
    * danger - left at kills, by fleeing orbs and alarm calls; wary orbs steer clear

    Each layer fades with its own half-life and slowly spreads to neighbouring cells.
    """

    LAYERS = ("food", "prey", "danger")
    HALF_LIFE = {"food": 1500, "prey": 480, "danger": 1800}   # frames
    CAP = 40.0

    def __init__(self, width, height, cell=24):
        self.cell = cell
        self.nx = int(width // cell) + 1
        self.ny = int(height // cell) + 1
        self.grids = {k: np.zeros((self.ny, self.nx), dtype=np.float32) for k in self.LAYERS}
        self.decay4 = {k: np.float32(0.5 ** (4.0 / hl)) for k, hl in self.HALF_LIFE.items()}

    def _cell(self, x, y):
        ix, iy = int(x // self.cell), int(y // self.cell)
        if 0 <= ix < self.nx and 0 <= iy < self.ny:
            return iy, ix
        return None

    def deposit(self, layer, x, y, amount):
        c = self._cell(x, y)
        if c is not None:
            self.grids[layer][c] += amount

    def burst(self, layer, x, y, amount):
        """A strong mark spread over the 3x3 cells around a spot."""
        c = self._cell(x, y)
        if c is None:
            return
        iy, ix = c
        g = self.grids[layer]
        g[max(0, iy - 1):iy + 2, max(0, ix - 1):ix + 2] += amount * 0.5
        g[iy, ix] += amount * 0.5

    def value(self, layer, x, y):
        c = self._cell(x, y)
        return float(self.grids[layer][c]) if c is not None else 0.0

    def gradient(self, layer, x, y, reach=48):
        """Direction of the strongest nearby smell: (dx, dy, strength) or None."""
        here = self.value(layer, x, y)
        best, best_v = None, here
        for k in range(8):
            a = k * math.pi / 4
            v = self.value(layer, x + math.cos(a) * reach, y + math.sin(a) * reach)
            if v > best_v:
                best, best_v = a, v
        if best is None or best_v < 0.3:
            return None
        return math.cos(best), math.sin(best), best_v

    def update(self, frame):
        if frame % 4 == 0:
            for k, g in self.grids.items():
                g *= self.decay4[k]
                np.minimum(g, self.CAP, out=g)
        if frame % 12 == 0:
            for g in self.grids.values():
                # Gentle diffusion: each cell shares a little with its four neighbours.
                inner = g[1:-1, 1:-1]
                g[1:-1, 1:-1] = 0.6 * inner + 0.1 * (g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:])


class Carrion:
    """Remains left behind when an orb dies - a meal for scavengers only."""
    __slots__ = ("x", "y", "value", "age", "max_age", "name", "radius")

    def __init__(self, x, y, value, name, radius):
        self.x, self.y = x, y
        self.value = value
        self.age = 0
        self.max_age = 2400
        self.name = name
        self.radius = radius


class FoodField:
    """Nutrient motes that grow around slowly drifting fertile patches."""

    MAX_AGE = 5000

    def __init__(self, width, height):
        self.width, self.height = width, height
        self.obstacles = []
        self.target = int(clamp(width * height / 20000, 40, 300))
        self.items = []
        self.hash = SpatialHash(120)
        self.patches = []
        for _ in range(4):
            a = random.uniform(0, math.tau)
            self.patches.append([random.uniform(0.15, 0.85) * width,
                                 random.uniform(0.15, 0.85) * height,
                                 math.cos(a) * 0.25, math.sin(a) * 0.25])
        for _ in range(int(self.target * 0.7)):
            self._spawn_one()
        self.hash.rebuild(self.items)

    def _spawn_one(self, x=None, y=None, spread=80):
        fixed = x is not None
        for attempt in range(6):
            if not fixed:
                if random.random() < 0.75 and attempt < 3:
                    px, py = random.choice(self.patches)[:2]
                    x = random.gauss(px, spread)
                    y = random.gauss(py, spread)
                else:
                    x = random.uniform(0, self.width)
                    y = random.uniform(0, self.height)
            x = clamp(x, 10, self.width - 10)
            y = clamp(y, 10, self.height - 10)
            if not any(point_in_rect(x, y, r, 4) for r in self.obstacles):
                food = Food(x, y, random.uniform(6, 12))
                self.items.append(food)
                return food
            if fixed:
                return None
        return None

    PATCH_OFFSETS = ((-0.12, -0.12), (0.12, -0.1), (-0.1, 0.12), (0.12, 0.12))

    def update(self, env=None):
        env = env or SEASON_PARAMS["spring"]
        zx, zy = SEASON_ZONES.get(env.get("season", "spring"), (0.5, 0.5))
        for i, p in enumerate(self.patches):
            # Fertile ground drifts toward this season's zone; herds follow the food.
            ox, oy = self.PATCH_OFFSETS[i % len(self.PATCH_OFFSETS)]
            tx = clamp((zx + ox) * self.width, 80, self.width - 80)
            ty = clamp((zy + oy) * self.height, 80, self.height - 80)
            dx, dy = normalize(tx - p[0], ty - p[1])
            dist = math.hypot(tx - p[0], ty - p[1])
            pull = 0.006 if dist > 60 else 0.0
            p[2] = p[2] * 0.995 + dx * pull + random.uniform(-0.004, 0.004)
            p[3] = p[3] * 0.995 + dy * pull + random.uniform(-0.004, 0.004)
            p[2], p[3] = limit(p[2], p[3], 0.45)
            p[0] = clamp(p[0] + p[2], 80, self.width - 80)
            p[1] = clamp(p[1] + p[3], 80, self.height - 80)

        target = self.target * env["food_target"]
        deficit = target - len(self.items)
        if deficit > 0 and random.random() < (0.12 + 0.5 * deficit / target) * env["food_rate"]:
            food = self._spawn_one()
            if food is not None:
                food.value *= env["food_value"]
        elif deficit < -target * 0.15 and random.random() < 0.05:
            self.remove(self.items[0])   # frost / rot thins out the surplus
        if env.get("drought") and self.items and random.random() < 0.06:
            self.remove(random.choice(self.items))   # food withers in the heat

        for f in self.items:
            f.age += 1
        if self.items and self.items[0].age > self.MAX_AGE:
            self.items = [f for f in self.items if f.age <= self.MAX_AGE]
        self.hash.rebuild(self.items)

    def near(self, x, y, r):
        return [f for f in self.hash.query(x, y, r) if not f.eaten]

    def remove(self, food):
        if not food.eaten:
            food.eaten = True
            try:
                self.items.remove(food)
            except ValueError:
                pass

    def burst(self, x, y, n=15, spread=40, value_mult=1.0):
        made = []
        for _ in range(n):
            f = self._spawn_one(random.gauss(x, spread), random.gauss(y, spread))
            if f is not None:
                f.value *= value_mult
                made.append(f)
        return made


# ----------------------------------------------------------------------------
# Packs
# ----------------------------------------------------------------------------

class Pack:
    _next_id = 1

    def __init__(self, leader):
        self.id = Pack._next_id
        Pack._next_id += 1
        self.leader = leader
        self.members = [leader]
        self.target = None
        self.idle = 0
        self.territory = None
        self.formed = 0

    def add(self, orb):
        if orb not in self.members:
            self.members.append(orb)
            orb.pack = self

    def disband(self):
        for m in self.members:
            if m.pack is self:
                m.pack = None
        self.members = []

    def slot(self, orb):
        try:
            return self.members.index(orb)
        except ValueError:
            return 0


class Territory:
    """A pack's home range: members patrol it and chase rival hunters out of it."""

    def __init__(self, pack, x, y, radius, day):
        self.pack = pack
        self.house = pack.leader.house
        self.x, self.y = x, y
        self.radius = radius
        self.founded_day = day
        self.intrusions = 0
        self.last_report = -10 ** 9

    def contains(self, o, margin=0.0):
        return math.hypot(o.x - self.x, o.y - self.y) <= self.radius + margin

    def random_point(self, inset=30):
        a = random.uniform(0, math.tau)
        r = math.sqrt(random.random()) * max(10.0, self.radius - inset)
        return self.x + math.cos(a) * r, self.y + math.sin(a) * r


class Alarm:
    """A warning call: 'predator over there!' Heard by nearby kin and social orbs."""

    def __init__(self, caller, threat, tx, ty, level):
        self.caller = caller
        self.threat = threat
        self.x, self.y = caller.x, caller.y
        self.tx, self.ty = tx, ty
        self.level = level
        self.life = 24

    def update(self):
        self.life -= 1
        return self.life > 0


class Percept:
    """What an orb noticed during its last think step."""
    __slots__ = ("neighbors", "threats", "threat_level", "prey", "foods",
                 "mates", "allies", "cursor_dist", "cursor_threat", "wall_prox")

    def __init__(self):
        self.neighbors = []
        self.threats = []      # (orb or CURSOR, level, x, y)
        self.threat_level = 0.0
        self.prey = []         # (orb, score)
        self.foods = []        # (food, score)
        self.mates = []        # (orb, score)
        self.allies = []
        self.cursor_dist = 1e9
        self.cursor_threat = 0.0
        self.wall_prox = 0.0


# ----------------------------------------------------------------------------
# The orb
# ----------------------------------------------------------------------------

class IntelligentOrb:
    _next_id = 1

    def __init__(self, x, y, screen_width, screen_height, genome=None, species="orb"):
        self.id = IntelligentOrb._next_id
        IntelligentOrb._next_id += 1
        self.name = make_name()
        self.house = None          # family line, assigned by the World or inherited
        self.parents = None        # (name, name) for orbs born in the simulation
        self.children = []         # child orb objects (parents never referenced back)
        self.last_utilities = None
        self.last_escape_log = -10 ** 9

        self.x, self.y = float(x), float(y)
        angle = random.uniform(0, math.tau)
        self.vx, self.vy = math.cos(angle), math.sin(angle)
        self.heading = angle
        self.screen_width = screen_width
        self.screen_height = screen_height

        # --- Genome ----------------------------------------------------------
        self.aggression = random.uniform(0.05, 0.95)
        self.curiosity = random.uniform(0.1, 0.9)
        self.social_tendency = random.uniform(0.05, 0.95)
        self.cursor_affinity = random.uniform(-1.0, 1.0)  # -1 afraid .. +1 attracted
        self.size_gene = random.uniform(9, 22)
        self.wariness = random.uniform(0.05, 0.6)  # chance of seeing through camouflage and lures
        self.night_vision = random.uniform(0.0, 0.6)  # sees in the dark, but dazzled by daylight
        self.disguise = random.uniform(0.3, 0.7)      # how convincing a mimic's warning look is
        self.species = species
        if species == "angler":
            self.aggression = random.uniform(0.7, 1.0)
            self.curiosity = random.uniform(0.1, 0.35)
            self.social_tendency = random.uniform(0.0, 0.2)
            self.cursor_affinity = random.uniform(-0.2, 0.2)
            self.size_gene = random.uniform(14, 22)
            self.night_vision = random.uniform(0.4, 0.9)
        elif species == "mimic":
            # Harmless grazers that look like dangerous hunters; disguise = how convincing.
            self.aggression = random.uniform(0.0, 0.15)
            self.social_tendency = random.uniform(0.2, 0.7)
            self.size_gene = random.uniform(9, 13)
        elif species == "splitter":
            # Peaceful grazers; size_gene is the size at which they divide in two.
            self.aggression = random.uniform(0.0, 0.2)
            self.social_tendency = random.uniform(0.3, 0.8)
            self.size_gene = random.uniform(14, 20)
        self.generation = 0
        self.brain = OrbBrain()
        if genome:
            self.apply_genome(genome)

        self.radius = self.size_gene * (0.6 if self.species == "splitter" else 1.0)
        self.update_physique()

        # --- State -----------------------------------------------------------
        self.alive = True
        self.energy = random.uniform(70, 95)
        self.stamina = 1.0
        self.exhausted = False
        self.sprinting = False
        self.stalking = False
        self.age = 0
        self.max_age = random.randint(30000, 60000)
        self.reproductive_cooldown = random.randint(200, 900)
        self.behavior_state = "exploring"
        self.state_timer = 0
        self.target = None          # Orb, Food or CURSOR
        self.mate_target = None
        self.courted_by = None
        self.pack = None
        self.percept = None
        self.last_brain_output = np.zeros(OrbBrain.N_OUTPUTS, dtype=np.float32)
        self.think_offset = random.randrange(THINK_INTERVAL)
        self.cause_of_death = None

        # Memory & lifetime learning
        self.behavior_values = [0.0] * len(BEHAVIORS)
        self.failed_targets = {}     # orb id -> age at which it is forgiven
        self.food_scent_timer = 0    # > 0: recently ate, laying a food trail
        self.food_trail = None       # (dx, dy, strength) toward stronger food scent
        self.prey_trail = None       # ... toward stronger body scent (for tracking)
        self.danger_trail = None     # ... toward stronger danger scent
        self.tracking = False
        self.following_trail = False
        self.threat_memory = None    # [x, y, frames_left]
        self.food_memory = None      # (x, y)
        self.waypoint = None
        self.waypoint_timer = 0
        self.wander_angle = random.uniform(0, math.tau)
        self.chase_timer = 0
        self.chase_best = 1e9
        self.juke_sign = random.choice((-1, 1))
        self.juke_timer = 0
        self.known_anglers = set()   # anglers this orb has seen through
        self.known_mimics = {}       # mimic id -> age at which the memory fades
        self.mimic_wise = 0          # how many mimics it has eaten (makes it harder to fool)
        self.alarm_cooldown = 0
        self.alarm_flash = 0         # frames to show the "!" above a caller
        self.last_seen = None        # where the hunted target was last visible
        self.unseen = 0              # frames since the target was last visible
        self.forage_stuck = 0
        self.hiding = False
        # World events
        self.infected = 0            # frames of plague left (0 = healthy)
        self.immune = False
        self.next_bite = 0           # pack members biting a leviathan pace themselves
        self.departed = False
        # Angler-only state
        self.lure = None
        self.strike_timer = 0
        self.relocate_timer = random.randint(600, 1500)

        # Scores
        self.hunting_success_count = 0
        self.offspring_count = 0
        self.food_eaten = 0
        self.survival_score = 0.0
        self.fitness = 0

        # Visuals
        self.trail = []
        self.explosion_particles = []
        self.is_exploding = False
        self.explosion_timer = 0
        self.shape = self.determine_shape()
        self.base_color = self.generate_personality_color()
        self.current_color = self.base_color

    # ------------------------------------------------------------------ genome
    def genome(self):
        return {
            "aggression": float(self.aggression),
            "curiosity": float(self.curiosity),
            "social_tendency": float(self.social_tendency),
            "cursor_affinity": float(self.cursor_affinity),
            "size_gene": float(self.size_gene),
            "wariness": float(self.wariness),
            "night_vision": float(self.night_vision),
            "disguise": float(self.disguise),
            "species": self.species,
            "house": self.house,
            "generation": int(self.generation),
            "brain": self.brain.to_dict(),
        }

    def apply_genome(self, g):
        self.aggression = clamp(float(g.get("aggression", self.aggression)), 0, 1)
        self.curiosity = clamp(float(g.get("curiosity", self.curiosity)), 0, 1)
        self.social_tendency = clamp(float(g.get("social_tendency", self.social_tendency)), 0, 1)
        self.cursor_affinity = clamp(float(g.get("cursor_affinity", self.cursor_affinity)), -1, 1)
        self.size_gene = clamp(float(g.get("size_gene", self.size_gene)), MIN_RADIUS + 1, 24)
        self.wariness = clamp(float(g.get("wariness", self.wariness)), 0, 1)
        self.night_vision = clamp(float(g.get("night_vision", self.night_vision)), 0, 1)
        self.disguise = clamp(float(g.get("disguise", self.disguise)), 0, 1)
        if isinstance(g.get("house"), str):
            self.house = g["house"]
        self.species = g.get("species", self.species) if g.get("species") in ("orb", "angler") else self.species
        self.generation = int(g.get("generation", 0))
        if "brain" in g:
            self.brain = OrbBrain.from_dict(g["brain"])

    def mutate_traits(self, amount=0.06):
        self.aggression = clamp(self.aggression + random.gauss(0, amount), 0, 1)
        self.curiosity = clamp(self.curiosity + random.gauss(0, amount), 0, 1)
        self.social_tendency = clamp(self.social_tendency + random.gauss(0, amount), 0, 1)
        self.cursor_affinity = clamp(self.cursor_affinity + random.gauss(0, amount * 2), -1, 1)
        self.size_gene = clamp(self.size_gene + random.gauss(0, amount * 15), MIN_RADIUS + 1, 24)
        self.wariness = clamp(self.wariness + random.gauss(0, amount), 0, 1)
        self.night_vision = clamp(self.night_vision + random.gauss(0, amount), 0, 1)
        self.disguise = clamp(self.disguise + random.gauss(0, amount), 0, 1)

    @property
    def is_angler(self):
        return self.species == "angler"

    @property
    def full_name(self):
        return f"{self.name} of House {self.house}" if self.house else self.name

    def describe_activity(self, world=None):
        """What this orb is up to, in words (for the follow card)."""
        s = self.behavior_state
        t = self.target
        if not self.alive:
            return {"eaten": "was eaten", "starved": "starved", "old age": "died of old age"}.get(
                self.cause_of_death, "died")
        if s == "hunting":
            if t is CURSOR:
                return "chasing your cursor"
            if isinstance(t, IntelligentOrb):
                if self.is_angler:
                    return f"lunging at {t.name}"
                if self.unseen > 0:
                    return f"searching for {t.name}, who slipped out of sight"
                return f"{'stalking' if self.stalking else 'chasing'} {t.name}"
            if self.tracking:
                return "nose down, tracking a scent trail"
            return "on the hunt"
        if s == "defending" and isinstance(t, IntelligentOrb):
            return f"driving {t.name} out of House {self.house}'s territory"
        if s == "fleeing":
            if self.hiding:
                return "hiding behind a window"
            threats = self.percept.threats if self.percept else []
            if threats:
                top = max(threats, key=lambda th: th[1])[0]
                return "fleeing your cursor" if top is CURSOR else f"fleeing from {top.name}"
            return "running from danger"
        if s == "foraging":
            if isinstance(t, Lure):
                return f"heading for food... it's {t.angler.name}'s lure!"
            if t is None and self.following_trail:
                return "following a food trail"
            return "heading for food" if t is not None else "looking for food"
        if s == "mating":
            return f"courting {self.mate_target.name}" if self.mate_target is not None else "looking for a mate"
        if s == "resting":
            if self.exhausted:
                return "catching its breath"
            if world is not None:
                if world.obstacles and self.perch_spot(world) is not None and \
                        math.hypot(*(a - b for a, b in zip(self.perch_spot(world), (self.x, self.y)))) < 30:
                    return "napping on top of a window"
                if world.env["cold"] > 1.15:
                    return "hibernating"
                if world.is_night:
                    return "asleep"
            return "resting"
        if s == "lurking":
            return "lurking with its lure out" if self.lure is not None else "digesting a meal"
        if s == "exploring":
            if self.pack is not None:
                return "leading its pack" if self.pack_leader else f"roaming with {self.pack.leader.name}'s pack"
            if self.is_angler and self.waypoint is not None:
                return "moving to a new ambush spot"
            return "exploring"
        return s

    def effective_vision(self, world):
        """Darkness shrinks vision; night-adapted eyes see further at night but are
        dazzled by bright daylight (an evolutionary trade-off)."""
        light = world.light
        day_eyes = 1.0 - 0.25 * self.night_vision
        night_eyes = 0.3 + 0.6 * self.night_vision
        return self.vision * (light * day_eyes + (1.0 - light) * night_eyes) * world.env.get("visibility", 1.0)

    @property
    def nocturnal(self):
        return self.night_vision > 0.55

    def update_physique(self):
        size_frac = (self.radius - MIN_RADIUS) / (MAX_RADIUS - MIN_RADIUS)
        self.mass = self.radius * 0.5
        if self.species == "angler":
            # Sluggish swimmers with a devastating short lunge.
            self.max_speed = 2.0 - 0.5 * size_frac
            self.max_force = 0.55
            self.sprint_mult = 2.7
            self.vision = 230 + 2 * self.radius
            return
        self.max_speed = 4.6 - 1.7 * size_frac          # big = strong but slow
        self.max_force = 0.30 - 0.10 * size_frac        # big = less agile
        self.sprint_mult = 1.40 + 0.25 * self.aggression
        self.vision = 150 + 140 * self.curiosity + 3 * self.radius

    def lifetime_fitness(self):
        return (self.age / 60.0 + self.hunting_success_count * 25 +
                self.offspring_count * 20 + self.food_eaten * 1.5)

    # ---------------------------------------------------------------- compat
    @property
    def pack_leader(self):
        return self.pack is not None and self.pack.leader is self

    @property
    def pack_leader_id(self):
        return self.pack.leader.id if self.pack else None

    @property
    def pack_members(self):
        return [m for m in self.pack.members if m is not self] if self.pack else []

    @property
    def apparent_radius(self):
        """How big this orb looks to others (mimics puff themselves up)."""
        return self.radius * (1 + 0.7 * self.disguise) if self.species == "mimic" else self.radius

    # ------------------------------------------------------------- appearance
    def generate_personality_color(self):
        if self.species == "mimic":
            return (int(190 + 40 * self.curiosity), int(180 + 40 * self.social_tendency), 225)   # true, pale colours
        if self.species == "splitter":
            # Translucent sea-green blobs, tinted by personality.
            return (int(70 + 70 * self.curiosity), int(185 + 50 * self.social_tendency), int(150 + 70 * self.wariness))
        if self.species == "angler":
            # Murky deep-sea colours so they blend in.
            return (int(40 + 50 * self.aggression), int(45 + 40 * self.curiosity), int(80 + 50 * self.wariness))
        r = 255 * self.aggression
        g = 255 * self.social_tendency
        b = 255 * self.curiosity
        if self.cursor_affinity < -0.5:
            b, r = min(255, b + 80), max(0, r - 40)
        elif self.cursor_affinity > 0.5:
            r, g = min(255, r + 60), min(255, g + 30)
        # Keep colours vivid: lift very dark genomes.
        lift = max(0, 90 - max(r, g, b))
        return (int(min(255, r + lift)), int(min(255, g + lift)), int(min(255, b + lift)))

    def determine_shape(self):
        if self.species == "angler":
            return "angler"
        if self.species == "mimic":
            return "mimic"
        if self.species == "splitter":
            return "splitter"
        if self.aggression > 0.7 and self.cursor_affinity > 0.3:
            return "diamond"   # cursor hunters
        if self.social_tendency > 0.7:
            return "hexagon"   # social
        if self.cursor_affinity < -0.5:
            return "triangle"  # skittish
        if self.curiosity > 0.7:
            return "star"      # explorers
        return "circle"

    @property
    def display_shape(self):
        return "square" if self.pack_leader else self.shape

    def update_color(self):
        br, bg, bb = self.base_color
        e = 0.35 + 0.65 * clamp(self.energy / MAX_ENERGY, 0, 1)
        s = self.behavior_state
        if s == "hunting":
            c = (min(255, br + 90 * self.aggression + 40), bg * e * 0.7, bb * e * 0.7)
        elif s == "fleeing":
            c = (br * e * 0.7, bg * e, min(255, bb + 130))
        elif s == "mating":
            pulse = (math.sin(self.age * 0.12) + 1) / 2
            c = (200 + 55 * pulse, 90 + 60 * pulse, 200 + 55 * pulse)
        elif s == "foraging":
            c = (br * e * 0.8, min(255, bg * e + 50), bb * e * 0.8)
        elif s == "resting":
            c = (br * e * 0.6, bg * e * 0.6, bb * e * 0.6)
        elif s == "dying":
            c = (60, 60, 60)
        else:
            c = (br * e, bg * e, bb * e)
        self.current_color = (int(clamp(c[0], 0, 255)), int(clamp(c[1], 0, 255)), int(clamp(c[2], 0, 255)))

    def create_explosion(self):
        self.is_exploding = True
        self.explosion_timer = 70
        r, g, b = self.current_color if self.behavior_state != "dying" else self.base_color
        color = (min(255, r + 60), min(255, g + 60), min(255, b + 60))
        for _ in range(random.randint(14, 24)):
            self.explosion_particles.append(ExplosionParticle(self.x, self.y, color))

    def update_explosion(self):
        if not self.is_exploding:
            return False
        self.explosion_particles = [p for p in self.explosion_particles if p.update()]
        self.explosion_timer -= 1
        if self.explosion_timer <= 0 or not self.explosion_particles:
            self.is_exploding = False
            self.explosion_particles.clear()
            return False
        return True

    # ------------------------------------------------------------ relations
    def can_eat(self, other, assisted=False):
        if other is self or not other.alive:
            return False
        if self.pack is not None and other.pack is self.pack:
            return False
        if self.species == "leviathan":
            return other.species != "leviathan"          # it eats anything it touches
        if other.species == "leviathan":
            # Only a big, bold pack dares attack a leviathan (it gets worn down, not swallowed).
            return (assisted and self.species == "orb" and self.pack is not None and
                    len(self.pack.members) >= 3 and self.aggression > 0.55 and self.energy > 40)
        if self.species in ("splitter", "scavenger", "leech", "mimic"):
            return False    # grazers, carrion eaters and blood-suckers never kill
        if self.species == "angler":
            # Huge mouths: anglers can swallow prey almost their own size (but not each other).
            return other.species != "angler" and self.radius > other.radius * 0.95
        ratio = 0.72 if assisted else 1.08
        return self.radius > other.radius * ratio

    def can_hunt(self, target):  # compat
        return self.energy > 20 and self.can_eat(target)

    def is_kin(self, other):
        return self.species == other.species and genetic_distance(self, other) < 0.12

    def willing_to_mate(self):
        if self.species in ("splitter", "leech"):
            return False    # they reproduce on their own
        return (self.alive and self.age > MATURITY_AGE and self.reproductive_cooldown == 0
                and self.energy > 50 and self.behavior_state not in ("fleeing", "hunting", "dying"))

    def pack_assist(self, prey):
        """True if a packmate is close enough to the prey to help bring it down."""
        if self.pack is None:
            return False
        for m in self.pack.members:
            if m is not self and m.alive:
                if math.hypot(m.x - prey.x, m.y - prey.y) < m.radius + prey.radius + 60:
                    return True
        return False

    def reward(self, behavior, amount):
        """Lifetime learning: behaviours that pay off become preferred."""
        self.behavior_values[behavior] = clamp(self.behavior_values[behavior] + amount, -0.5, 0.5)

    # ---------------------------------------------------------------- update
    def update(self, cursor_pos, other_orbs, dt=1.0, world=None):
        if not self.alive:
            self.update_explosion()
            return
        if world is None:  # standalone use (e.g. smoke tests) without a World
            world = World.adhoc(other_orbs, cursor_pos, self.screen_width, self.screen_height)

        self.age += 1
        self.state_timer += 1
        if self.reproductive_cooldown > 0:
            self.reproductive_cooldown -= 1
        if self.alarm_cooldown > 0:
            self.alarm_cooldown -= 1
        if self.alarm_flash > 0:
            self.alarm_flash -= 1

        if self.percept is None or (self.age + self.think_offset) % THINK_INTERVAL == 0:
            if self.is_angler:
                self.think_angler(world)
            else:
                self.think(world)

        fx, fy = self.steer_angler(world) if self.is_angler else self.steer(world)
        self.integrate(fx, fy, dt, world)
        if self.is_angler:
            self.update_lure()
        self.interact(world)
        self.metabolize(dt, world)
        # Splitters divide in two once they've grown to their genetic size.
        if self.species == "splitter" and self.radius >= self.size_gene and self.energy > 50 and \
                self.age > 600 and world.room_for_births() and \
                world.species_count("splitter") < world.max_orbs * 0.3:   # crowding stops division
            world.split(self)
        if self.infected > 0:
            self.energy -= 0.012 * dt          # the plague saps strength
            self.infected -= 1
            if self.infected == 0:
                self.immune = True
                world.on_recover(self)
        self.update_color()

        if self.energy <= 0:
            self.die("plague" if self.infected > 0 else "starved", world)
        elif self.age > self.max_age:
            self.die("old age", world)

    def die(self, cause, world=None):
        if not self.alive:
            return
        self.alive = False
        self.cause_of_death = cause
        self.create_explosion()
        self.behavior_state = "dying"
        self.energy = 0
        self.target = self.mate_target = None
        self.lure = None
        self.trail.clear()
        if self.pack is not None:
            pack = self.pack
            self.pack = None
            if self in pack.members:
                pack.members.remove(self)
        if world is not None:
            world.on_death(self)

    # ----------------------------------------------------------------- sense
    def sense(self, world):
        p = Percept()
        vision = self.effective_vision(world)
        p.neighbors = world.neighbors(self, vision)

        # Forget grudges once they expire.
        if self.known_mimics and self.age % 300 == 0:
            self.known_mimics = {k: v for k, v in self.known_mimics.items() if v > self.age}
        if self.failed_targets:
            self.failed_targets = {k: v for k, v in self.failed_targets.items() if v > self.age}

        threats_total = 0.0
        threat_max = 0.0
        for o in p.neighbors:
            dx, dy = o.x - self.x, o.y - self.y
            d = math.hypot(dx, dy) or 0.01
            proximity = 1.0 - d / vision

            # Windows block line of sight - you can't react to what you can't see.
            if d > 45 and not world.line_of_sight(self.x, self.y, o.x, o.y):
                continue
            # Lurking anglers are camouflaged; only wary orbs (or ones that know) spot them.
            if o.is_angler and o.behavior_state == "lurking" and o.id not in self.known_anglers:
                if d > vision * (0.12 + 0.45 * self.wariness):
                    continue
                self.known_anglers.add(o.id)

            # --- Is it dangerous to me?
            # A big, bold pack sees a leviathan as the hunt of a lifetime, not a threat.
            if o.species == "leviathan" and self.can_eat(o, assisted=True):
                p.prey.append((o, 2.0))
                continue
            # Mimics: unless I've seen through this one, it looks like a dangerous hunter.
            if o.species == "mimic" and self.known_mimics.get(o.id, -1) < self.age and self.species != "mimic":
                # Close looks, wary eyes and past experience can reveal the bluff.
                # (rolled every think, so a decent disguise holds up for tens of seconds)
                see_through = 0.08 * (0.005 + 0.45 * (1 - o.disguise) ** 2 + 0.05 * self.wariness +
                                      0.015 * min(self.mimic_wise, 5)) * (1.6 if d < 80 else 1.0)
                if random.random() < see_through:
                    self.known_mimics[o.id] = self.age + 2400     # remembered for a while
                    if world.is_followed(self) or world.is_followed(o) or random.random() < 0.04:
                        world.log(f"{self.name} saw through {o.name}'s disguise - it's a harmless mimic!",
                                  "followed" if world.is_followed(self) or world.is_followed(o) else "info")
                else:
                    # Fooled: it looks as big and fierce as a real hunter.
                    if o.apparent_radius > self.radius * 1.08:
                        level = proximity * (0.35 + 0.5 * o.disguise)
                        p.threats.append((o, level, o.x, o.y))
                        threats_total += level
                        threat_max = max(threat_max, level)
                    continue      # and it certainly doesn't look like a meal
            # A territory owner charging at me: get out of their land, even if it couldn't eat me.
            if o.behavior_state == "defending" and o.target is self:
                level = proximity * 1.3
                p.threats.append((o, level, o.x + o.vx * 8, o.y + o.vy * 8))
                threats_total += level
                threat_max = max(threat_max, level)
                continue
            # A leech homing in on me: a mild threat that wary orbs try to dodge.
            if o.species == "leech" and o.host is None and o.target is self:
                level = proximity * 0.5 * (0.4 + self.wariness)
                p.threats.append((o, level, o.x, o.y))
                threats_total += level
                threat_max = max(threat_max, level)
            pack_hunter = o.pack is not None and len(o.pack.members) > 1
            kin_safe = o.social_tendency > 0.35 and o.is_kin(self)   # kin won't eat me
            if not kin_safe and o.can_eat(self, assisted=pack_hunter):
                ux, uy = -dx / d, -dy / d                     # direction from it to me
                closing = (o.vx * ux + o.vy * uy) / max(o.max_speed, 0.1)
                level = proximity * (0.5 + 0.5 * o.aggression) * (1.0 + max(0.0, closing))
                if o.behavior_state == "hunting":
                    level *= 1.5
                    if o.target is self:
                        level *= 1.6
                elif o.behavior_state == "resting":
                    level *= 0.4
                if o.stalking and o.target is self:
                    level *= 0.45   # stalkers are hard to notice
                if level > 0.05:
                    p.threats.append((o, level, o.x + o.vx * 8, o.y + o.vy * 8))
                    threats_total += level
                    threat_max = max(threat_max, level)
                continue

            # --- Could I eat it?
            assisted = self.pack is not None and len(self.pack.members) > 1
            # (You can't reach a leech stuck to your own back - but your friends can pick it off.)
            if self.can_eat(o, assisted=assisted) and not (self.social_tendency > 0.35 and self.is_kin(o)) \
                    and getattr(o, "host", None) is not self:
                value = o.energy * 0.5 + o.radius * 2.0
                catch = 1.0
                if o.stamina < 0.3 or o.exhausted:
                    catch += 0.6
                if o.behavior_state in ("resting", "mating", "foraging"):
                    catch += 0.4
                if o.sprinting:
                    catch -= 0.3
                # Cornered prey is easier to catch.
                margin = 70
                if (o.x < margin or o.x > self.screen_width - margin or
                        o.y < margin or o.y > self.screen_height - margin):
                    catch += 0.3
                score = value * catch / (d + 60)
                if o.id in self.failed_targets:
                    score *= 0.2
                if o.species == "leech" and o.host is None:
                    score *= 0.3    # a free-swimming leech is a bitter, wriggly snack
                elif o.species == "scavenger":
                    score *= 0.35   # scavengers are tough and stringy
                p.prey.append((o, score))

            # --- Friends & potential mates
            if self.is_kin(o) or (self.social_tendency > 0.5 and o.social_tendency > 0.5 and
                                  genetic_distance(self, o) < 0.3):
                p.allies.append(o)
            if o.species == self.species and o.willing_to_mate() and o.target is not self and \
                    max(o.radius, self.radius) < 1.6 * min(o.radius, self.radius):
                compat = 1.0 - genetic_distance(self, o)
                if 0.25 < compat:  # too different = incompatible species
                    quality = (o.lifetime_fitness() * 0.05 + o.energy * 0.02 +
                               o.hunting_success_count * 0.5)
                    p.mates.append((o, (quality + compat * 2) / (1 + d / 150)))

        # --- The cursor
        cx, cy = world.cursor
        p.cursor_dist = math.hypot(cx - self.x, cy - self.y)
        if self.cursor_affinity < -0.3 and p.cursor_dist < vision:
            cvx, cvy = world.cursor_vel
            cspeed = math.hypot(cvx, cvy)
            level = (1.0 - p.cursor_dist / vision) * abs(self.cursor_affinity) * (1.0 + min(cspeed, 30) / 15)
            p.cursor_threat = level
            if level > 0.05:
                p.threats.append((CURSOR, level, cx + cvx * 4, cy + cvy * 4))
                threats_total += level
                threat_max = max(threat_max, level)

        # --- Alarm calls from others: threats we can't see ourselves
        for alarm in world.alarms:
            if alarm.caller is self or not alarm.caller.alive:
                continue
            if math.hypot(alarm.x - self.x, alarm.y - self.y) > ALARM_RADIUS:
                continue
            # Kin always listen; very social orbs heed anyone's warning.
            if not (genetic_distance(self, alarm.caller) < 0.35 or self.social_tendency > 0.65):
                continue
            t = alarm.threat
            if t is CURSOR:
                if self.cursor_affinity >= -0.3:
                    continue
            elif not t.alive or not t.can_eat(self, assisted=t.pack is not None):
                continue
            if any(th[0] is t for th in p.threats):
                continue
            if t is not CURSOR and t.is_angler:
                self.known_anglers.add(t.id)   # the warning reveals the ambusher
            tx, ty = (alarm.tx, alarm.ty) if t is CURSOR else (t.x, t.y)
            level = alarm.level * 0.75
            p.threats.append((t, level, tx, ty))
            threats_total += level
            threat_max = max(threat_max, level)

        p.threat_level = threat_max + 0.3 * (threats_total - threat_max)

        # --- Food (lures look like food unless we've seen through them)
        if world.food is not None or world.lures:
            real_best = None
            # Lures glow, so they can be seen from afar even in the dark.
            lure_range = max(vision, self.vision * 0.9)
            for f in world.food_near(self.x, self.y, lure_range):
                d = math.hypot(f.x - self.x, f.y - self.y)
                if d > vision and not isinstance(f, Lure):
                    continue
                if isinstance(f, Lure):
                    if f.angler.id in self.known_anglers:
                        continue
                    if d < 90 and random.random() < self.wariness * 0.2:
                        self.known_anglers.add(f.angler.id)   # "wait... that food has teeth"
                        continue
                score = f.value / (d + 25)
                p.foods.append((f, score))
                if not isinstance(f, Lure) and (real_best is None or score > real_best[1]):
                    real_best = (f, score)
            if real_best:
                self.food_memory = (real_best[0].x, real_best[0].y)

        # --- Walls
        m = 90
        wx = min(self.x, self.screen_width - self.x)
        wy = min(self.y, self.screen_height - self.y)
        p.wall_prox = clamp(1.0 - min(wx, wy) / m, 0, 1)

        p.prey.sort(key=lambda t: -t[1])
        p.foods.sort(key=lambda t: -t[1])
        p.mates.sort(key=lambda t: -t[1])
        return p

    # ----------------------------------------------------------------- think
    def get_environmental_features(self, p, world):
        best_prey = p.prey[0][1] if p.prey else 0.0
        best_food = p.foods[0][1] if p.foods else 0.0
        cursor_prox = clamp(1.0 - p.cursor_dist / 600.0, 0, 1)
        return np.array([
            self.energy / MAX_ENERGY,
            self.stamina,
            self.age / self.max_age,
            clamp(p.threat_level / 1.5, 0, 1),
            min(len(p.threats), 5) / 5.0,
            clamp(best_prey * 2, 0, 1),
            clamp(best_food * 2, 0, 1),
            1.0 if p.mates else 0.0,
            min(len(p.allies), 6) / 6.0,
            cursor_prox,
            self.cursor_affinity,
            p.wall_prox,
            self.aggression,
            self.curiosity,
            self.social_tendency,
            self.radius / MAX_RADIUS,
            self.behavior_values[HUNT] * 2,
            self.behavior_values[FLEE] * 2,
            1.0 if self.pack is not None else 0.0,
            1.0,
            world.light,
            world.env["food_rate"] / 1.3,     # how bountiful the season is
        ], dtype=np.float32)

    def maybe_raise_alarm(self, p, world):
        """Warn the neighbourhood about the most dangerous threat we can see.
        Social orbs are much more likely to call - it mostly helps their kin."""
        if self.alarm_cooldown > 0 or not p.threats or p.threat_level < 0.4:
            return
        if random.random() > 0.25 + 0.75 * self.social_tendency:
            return
        threat = max(p.threats, key=lambda t: t[1])
        world.raise_alarm(self, threat[0], threat[2], threat[3], threat[1])
        self.alarm_cooldown = ALARM_COOLDOWN
        self.alarm_flash = 30
        self.energy -= 0.5

    def track_target(self, world):
        """Remember where a hunted target was last seen; windows can hide it."""
        t = self.target
        if not isinstance(t, IntelligentOrb):
            self.last_seen = None
            self.unseen = 0
            return
        d = math.hypot(t.x - self.x, t.y - self.y)
        if d < 45 or world.line_of_sight(self.x, self.y, t.x, t.y):
            self.last_seen = (t.x, t.y)
            self.unseen = 0
        else:
            self.unseen += THINK_INTERVAL
            if self.last_seen is None:
                self.last_seen = (t.x, t.y)

    def defend_territory(self, p, world):
        """Pack members chase rival hunters out of their territory. Returns True while defending."""
        pack = self.pack
        terr = pack.territory if pack is not None else None
        if terr is None or self.energy < 35 or p.threat_level > 0.6:
            if self.behavior_state == "defending":
                self.behavior_state, self.target, self.state_timer = "exploring", None, 0
            return False
        if self.behavior_state == "defending":
            t = self.target
            if isinstance(t, IntelligentOrb) and t.alive and terr.contains(t, 40) and self.state_timer < 420:
                return True
            if isinstance(t, IntelligentOrb) and t.alive and not terr.contains(t, 40):
                world.on_intruder_driven_off(terr, t, self)
            self.behavior_state, self.target, self.state_timer = "exploring", None, 0
            return False
        if self.behavior_state == "hunting" and isinstance(self.target, IntelligentOrb) and \
                math.hypot(self.target.x - self.x, self.target.y - self.y) < 100:
            return False   # finish the kill first
        intruders = [o for o in p.neighbors
                     if o.species == "orb" and o.pack is not pack and o.aggression > 0.45 and
                     o.radius > 9 and terr.contains(o) and not (o.house == self.house and o.is_kin(self))]
        if not intruders:
            return False
        t = min(intruders, key=lambda o: math.hypot(o.x - self.x, o.y - self.y))
        self.behavior_state, self.target, self.state_timer = "defending", t, 0
        world.on_intrusion(terr, t, self)
        return True

    def think(self, world):
        p = self.percept = self.sense(world)
        self.maybe_raise_alarm(p, world)
        if self.defend_territory(p, world):
            return
        out = self.brain.forward(self.get_environmental_features(p, world))
        self.last_brain_output = out
        courage = float(out[6])

        hunger = 1.0 - self.energy / MAX_ENERGY
        u = [0.0] * len(BEHAVIORS)

        # Smells: food trails, body scent to track, and places where something died.
        sc = world.scent
        if sc is not None:
            self.food_trail = sc.gradient("food", self.x, self.y) if not p.foods else None
            tracker = self.species == "orb" and self.aggression > 0.35
            self.prey_trail = sc.gradient("prey", self.x, self.y, 64) if tracker and not p.prey else None
            self.danger_trail = sc.gradient("danger", self.x, self.y)
        else:
            self.food_trail = self.prey_trail = self.danger_trail = None

        u[EXPLORE] = 0.20 + 0.25 * self.curiosity + 0.1 * (1 - hunger)

        food_signal = (1.0 if p.foods else 0.55 if self.food_memory else 0.65 if self.food_trail else 0.3)
        u[FORAGE] = (0.1 + hunger * 1.35) * food_signal * (1.0 - 0.35 * self.aggression)

        cursor_hunter = (self.cursor_affinity > 0.45 and self.aggression > 0.55 and
                         p.cursor_dist < self.vision * 1.3)
        if p.prey:
            prey_factor = clamp(p.prey[0][1] * 2.2, 0, 1)
        elif cursor_hunter:
            prey_factor = 0.35
        elif self.prey_trail is not None and hunger > 0.35:
            prey_factor = clamp(self.prey_trail[2] / 6.0, 0.15, 0.4)   # pick up a scent and track it
        else:
            prey_factor = 0.0
        # Satiation: well-fed predators lose interest in hunting.
        u[HUNT] = (0.05 + 0.25 * self.aggression + hunger * (0.5 + 1.2 * self.aggression)) * prey_factor
        if self.energy < 12:
            u[HUNT] *= 0.4     # too weak to chase
        if p.prey and p.prey[0][0].species == "leviathan":
            u[HUNT] += 0.5     # glory: the pack goes for the giant
        if self.pack is not None and self.pack.leader is not self and self.pack.leader.behavior_state == "hunting":
            u[HUNT] += 0.45    # follow the pack

        fear = (1.3 - 0.55 * self.aggression) * (1.0 - 0.3 * courage)
        u[FLEE] = p.threat_level * fear * 1.7
        if self.threat_memory and not p.threats:
            u[FLEE] = max(u[FLEE], 0.35 * self.threat_memory[2] / 90.0)

        if self.willing_to_mate() and world.room_for_births(self.species) and not world.species_full(self.species):
            u[MATE] = (0.4 + 0.35 * self.social_tendency + 0.4 * (self.energy - 50) / 50)
            u[MATE] *= 1.0 if p.mates else 0.3
            if self.courted_by is not None and self.courted_by.alive and self.courted_by.mate_target is self:
                u[MATE] += 0.4

        u[REST] = (1.0 - self.stamina) * 0.7
        if self.energy < 25 and not p.foods and not p.prey:
            u[REST] += 0.25   # conserve energy when there is nothing to do

        # --- Day/night and seasons
        darkness = 1.0 - world.light
        # Day creatures get sleepy at night; night creatures doze through bright days.
        u[REST] += darkness * (1.0 - self.night_vision) * 0.75 * (1.0 - 0.6 * hunger)
        u[REST] += world.light * max(0.0, self.night_vision - 0.5) * 0.6 * (1.0 - hunger)
        # Nocturnal hunters press their advantage in the dark.
        u[HUNT] *= 1.0 + darkness * self.night_vision * 0.8
        # Winter: hibernate when food is scarce and there are reserves to live on.
        if world.env["cold"] > 1.15 and not p.foods and self.energy > 30:
            u[REST] += 0.3
        if u[MATE] > 0:
            u[MATE] *= world.env["breeding"]

        if p.threat_level > 0.15:
            u[REST] = 0.0

        current = BEHAVIORS.index(self.behavior_state) if self.behavior_state in BEHAVIORS else EXPLORE
        for i in range(len(BEHAVIORS)):
            u[i] += 0.25 * float(out[i]) + self.behavior_values[i]
        u[current] += 0.12  # hysteresis: don't dither
        u[MATE] = u[MATE] if self.willing_to_mate() else -1
        self.last_utilities = list(u)
        choice = max(range(len(BEHAVIORS)), key=lambda i: u[i])

        new_state = BEHAVIORS[choice]
        if new_state != self.behavior_state:
            self.behavior_state = new_state
            self.state_timer = 0
            self.target = None
            self.chase_timer = 0
            self.chase_best = 1e9
            if new_state != "mating":
                self.mate_target = None

        # Choose / maintain targets for the active behaviour.
        if new_state == "hunting":
            self.select_prey(p, cursor_hunter)
            self.track_target(world)
        elif new_state == "foraging":
            self.target = p.foods[0][0] if p.foods else None
        elif new_state == "mating":
            self.select_mate(p)
        elif new_state == "fleeing":
            if p.threats:
                nearest = max(p.threats, key=lambda t: t[1])
                self.threat_memory = [nearest[2], nearest[3], 90]
                self.reward(FLEE, 0.005)

        for i in range(len(BEHAVIORS)):  # learned values slowly regress
            self.behavior_values[i] *= 0.997

    # ------------------------------------------------------------- angler AI
    def think_angler(self, world):
        """Ambush predator: sit still, dangle the lure, lunge at whatever bites."""
        p = self.percept = self.sense(world)
        s = self.behavior_state
        self.relocate_timer -= THINK_INTERVAL

        if p.threat_level > 0.45:
            new = "fleeing"
            if p.threats:
                top = max(p.threats, key=lambda t: t[1])
                self.threat_memory = [top[2], top[3], 90]
        elif s == "hunting" and self.strike_timer > 0 and isinstance(self.target, IntelligentOrb) \
                and self.target.alive:
            new = "hunting"
        else:
            strike_range = self.radius * 1.6 + 30
            hungry = self.energy < 80          # a full angler just digests
            in_reach = [o for o, _ in p.prey
                        if math.hypot(o.x - self.x, o.y - self.y) < strike_range] if hungry else []
            lure_reach = []
            if self.lure is not None:
                lx, ly = self.lure.x, self.lure.y
                lure_reach = [o for o, _ in p.prey if math.hypot(o.x - lx, o.y - ly) < o.radius + 25]
            if (in_reach or lure_reach) and self.energy > 3 and self.stamina > 0.25:
                new = "hunting"
                self.target = min(lure_reach or in_reach,
                                  key=lambda o: math.hypot(o.x - self.x, o.y - self.y))
                self.strike_timer = 30
            elif (self.willing_to_mate() and p.mates and world.room_for_births() and
                    world.species_count("angler") < world.max_orbs * 0.15):
                new = "mating"
                self.select_mate(p)
            elif s == "exploring" and self.waypoint is not None:
                new = "exploring"   # still relocating
            elif self.relocate_timer <= 0 or self.energy < 25 and self.relocate_timer < 600:
                new = "exploring"
                self.waypoint = world.ambush_spot(self)
                self.waypoint_timer = 900
                self.relocate_timer = random.randint(1800, 3200)
            else:
                new = "lurking"

        if new != s:
            self.behavior_state = new
            self.state_timer = 0
            if new not in ("hunting",):
                self.target = None
            if new != "mating":
                self.mate_target = None

    def update_lure(self):
        # Only dangle the lure when hungry; while digesting, it's tucked away.
        if self.behavior_state != "lurking" or self.energy > 78:
            self.lure = None
            return
        if self.lure is None:
            self.lure = Lure(self)
        bob = math.sin(self.age * 0.07) * 4
        reach = self.radius + 18
        h = self.heading
        self.lure.x = self.x + math.cos(h) * reach - math.sin(h) * bob
        self.lure.y = self.y + math.sin(h) * reach + math.cos(h) * bob

    def steer_angler(self, world):
        p = self.percept
        s = self.behavior_state
        fx = fy = 0.0
        want_sprint = False
        sep_ignore = None
        if s == "hunting":
            t = self.target
            self.strike_timer -= 1
            if not isinstance(t, IntelligentOrb) or not t.alive or self.strike_timer <= 0:
                self.behavior_state = "lurking"
                self.target = None
                fx, fy = -self.vx * 0.2, -self.vy * 0.2
            else:
                d = math.hypot(t.x - self.x, t.y - self.y)
                lead = clamp(d / (self.max_speed * self.sprint_mult), 0, 12)
                fx, fy = self.seek(t.x + t.vx * lead, t.y + t.vy * lead, self.max_speed * self.sprint_mult)
                want_sprint = True
                sep_ignore = t
        elif s == "fleeing":
            (fx, fy), want_sprint = self.steer_flee(world)
        elif s == "mating":
            fx, fy = self.steer_mate()
            sep_ignore = self.mate_target
        elif s == "exploring" and self.waypoint is not None:
            wx, wy = self.waypoint
            if math.hypot(wx - self.x, wy - self.y) < 25 or self.waypoint_timer <= 0:
                self.waypoint = None
                self.behavior_state = "lurking"
                # Face open water so the lure dangles where prey will find it.
                self.heading = math.atan2(world.height / 2 - self.y, world.width / 2 - self.x)
            else:
                self.waypoint_timer -= 1
                fx, fy = self.arrive(wx, wy, 60)
        else:
            # Lurking: hold perfectly still, just a slow "breathing" sway.
            fx, fy = -self.vx * 0.15, -self.vy * 0.15
            self.heading += math.sin(self.age * 0.01) * 0.003

        sx, sy = self.separation(p.neighbors, sep_ignore)
        wx2, wy2 = self.wall_avoidance(1.0)
        ox, oy = self.obstacle_avoidance(world)
        fx += sx + wx2 + ox
        fy += sy + wy2 + oy
        if self.exhausted and self.stamina > 0.35:
            self.exhausted = False
        if self.stamina <= 0.02:
            self.exhausted = True
        self.sprinting = want_sprint and not self.exhausted
        return fx, fy

    def select_prey(self, p, cursor_hunter):
        # Pack members coordinate on the leader's target.
        if self.pack is not None and self.pack.leader is not self:
            t = self.pack.target
            if isinstance(t, IntelligentOrb) and t.alive and self.can_eat(t, assisted=True):
                if math.hypot(t.x - self.x, t.y - self.y) < self.vision * 1.6:
                    self.target = t
                    return

        current = self.target
        best = p.prey[0] if p.prey else None
        if isinstance(current, IntelligentOrb) and current.alive and self.can_eat(current, True):
            cur_score = next((s for o, s in p.prey if o is current), 0.0)
            d = math.hypot(current.x - self.x, current.y - self.y)
            # Stay committed unless something far better shows up.
            if d < self.vision * 1.4 and (best is None or cur_score > best[1] * 0.7):
                return
        if best is not None:
            if best[0] is not current:
                self.chase_timer = 0
                self.chase_best = 1e9
            self.target = best[0]
        elif cursor_hunter:
            self.target = CURSOR
        else:
            self.target = None
        if self.pack_leader:
            self.pack.target = self.target if isinstance(self.target, IntelligentOrb) else None

    def select_mate(self, p):
        cur = self.mate_target
        if cur is not None and cur.alive and cur.willing_to_mate() and \
                math.hypot(cur.x - self.x, cur.y - self.y) < self.vision * 1.3:
            return
        self.mate_target = None
        if self.courted_by is not None and self.courted_by.alive and self.courted_by.willing_to_mate() \
                and self.courted_by.mate_target is self:
            self.mate_target = self.courted_by
        elif p.mates:
            self.mate_target = p.mates[0][0]
        if self.mate_target is not None:
            self.mate_target.courted_by = self

    # ----------------------------------------------------------------- steer
    def seek(self, tx, ty, speed=None):
        speed = self.max_speed if speed is None else speed
        dx, dy = normalize(tx - self.x, ty - self.y)
        return dx * speed - self.vx, dy * speed - self.vy

    def arrive(self, tx, ty, slow_radius=60, speed=None):
        speed = self.max_speed if speed is None else speed
        dx, dy = tx - self.x, ty - self.y
        d = math.hypot(dx, dy)
        if d < 1e-6:
            return -self.vx, -self.vy
        s = speed * min(1.0, d / slow_radius)
        return dx / d * s - self.vx, dy / d * s - self.vy

    def wall_avoidance(self, strength=1.0):
        fx = fy = 0.0
        margin = 70 + self.radius
        # Look ahead so orbs turn before hitting the edge.
        px = self.x + self.vx * 15
        py = self.y + self.vy * 15
        w, h = self.screen_width, self.screen_height
        if px < margin:
            fx += (margin - px) / margin
        elif px > w - margin:
            fx -= (px - (w - margin)) / margin
        if py < margin:
            fy += (margin - py) / margin
        elif py > h - margin:
            fy -= (py - (h - margin)) / margin
        k = self.max_force * 1.6 * strength
        return fx * k, fy * k

    def separation(self, neighbors, ignore=None):
        fx = fy = 0.0
        for o in neighbors:
            if o is ignore or not o.alive or getattr(o, "host", None) is self:
                continue
            dx, dy = self.x - o.x, self.y - o.y
            d = math.hypot(dx, dy)
            min_d = self.radius + o.radius + 10
            if 0 < d < min_d:
                push = (min_d - d) / min_d
                fx += dx / d * push
                fy += dy / d * push
        return fx * self.max_force * 1.5, fy * self.max_force * 1.5

    def flocking(self, allies):
        if not allies:
            return 0.0, 0.0
        n = len(allies)
        cx = sum(o.x for o in allies) / n
        cy = sum(o.y for o in allies) / n
        avx = sum(o.vx for o in allies) / n
        avy = sum(o.vy for o in allies) / n
        fx = fy = 0.0
        if math.hypot(cx - self.x, cy - self.y) > 70:
            sx, sy = self.seek(cx, cy, self.max_speed * 0.6)
            fx += sx * 0.5
            fy += sy * 0.5
        fx += (avx - self.vx) * 0.3
        fy += (avy - self.vy) * 0.3
        return fx * self.social_tendency, fy * self.social_tendency

    def wander(self):
        if self.waypoint is None or self.waypoint_timer <= 0 or \
                math.hypot(self.waypoint[0] - self.x, self.waypoint[1] - self.y) < 50:
            terr = self.pack.territory if self.pack is not None else None
            if terr is not None:
                # Pack members patrol their own territory.
                wx, wy = terr.random_point()
                self.waypoint = (clamp(wx, 60, self.screen_width - 60), clamp(wy, 60, self.screen_height - 60))
            else:
                # Curious orbs pick far-away destinations; others meander locally.
                reach = 120 + 600 * self.curiosity
                self.waypoint = (clamp(self.x + random.uniform(-reach, reach), 60, self.screen_width - 60),
                                 clamp(self.y + random.uniform(-reach, reach), 60, self.screen_height - 60))
            self.waypoint_timer = random.randint(240, 720)
        self.waypoint_timer -= 1
        self.wander_angle += random.uniform(-0.35, 0.35)
        cruise = self.max_speed * (0.45 + 0.3 * self.curiosity)
        sx, sy = self.seek(self.waypoint[0], self.waypoint[1], cruise)
        jx = math.cos(self.wander_angle) * 0.4
        jy = math.sin(self.wander_angle) * 0.4
        return sx * 0.6 + jx * self.max_force, sy * 0.6 + jy * self.max_force

    def steer(self, world):
        p = self.percept
        s = self.behavior_state
        want_sprint = False
        self.stalking = False
        self.hiding = False
        self.tracking = False
        self.following_trail = False
        perched = False
        fx = fy = 0.0
        sep_ignore = None

        if s == "hunting":
            (fx, fy), want_sprint = self.steer_hunt(world)
            sep_ignore = self.target if isinstance(self.target, IntelligentOrb) else None
        elif s == "defending" and isinstance(self.target, IntelligentOrb):
            t = self.target
            d = math.hypot(t.x - self.x, t.y - self.y)
            lead = clamp(d / (self.max_speed * 1.4), 0, 20)
            fx, fy = self.seek(t.x + t.vx * lead, t.y + t.vy * lead, self.max_speed * self.sprint_mult)
            want_sprint = d < 220
            sep_ignore = t
        elif s == "fleeing":
            (fx, fy), want_sprint = self.steer_flee(world)
        elif s == "foraging":
            fx, fy = self.steer_forage(world)
        elif s == "mating":
            fx, fy = self.steer_mate()
            sep_ignore = self.mate_target
        elif s == "resting":
            fx, fy = -self.vx * 0.08, -self.vy * 0.08
            perch = self.perch_spot(world)
            if perch is not None:
                # Settle down on top of a nearby window's title bar.
                fx, fy = self.arrive(perch[0], perch[1], 40, self.max_speed * 0.6)
                perched = math.hypot(perch[0] - self.x, perch[1] - self.y) < 30
            else:
                perched = False
                if p.allies and self.social_tendency > 0.5:
                    ax, ay = self.flocking(p.allies)
                    fx += ax * 0.3
                    fy += ay * 0.3
        else:  # exploring
            fx, fy = self.wander()
            if self.social_tendency > 0.45:
                ax, ay = self.flocking(p.allies)
                fx += ax
                fy += ay
            fx2, fy2 = self.cursor_curiosity(world)
            fx += fx2
            fy += fy2

        # Always-on reflexes.
        sx, sy = self.separation(p.neighbors, sep_ignore)
        wx, wy = self.wall_avoidance(2.0 if s == "fleeing" else 1.0)
        fx += sx + wx
        fy += sy + wy
        if not (s == "resting" and perched) and not self.hiding:
            ox, oy = self.obstacle_avoidance(world)
            fx += ox
            fy += oy
        # Steer clear of places that smell of death (wary orbs much more so). Hunters don't care.
        if self.danger_trail is not None and s not in ("hunting", "defending", "fleeing"):
            dx, dy, strength = self.danger_trail
            w = (0.15 + 0.85 * self.wariness) * min(1.0, strength / 6.0) * self.max_force * 1.5
            fx -= dx * w
            fy -= dy * w
        # Keep away from the sick during a plague (wary orbs are more careful).
        if world.plague is not None and not self.infected and not self.immune:
            caution = 0.3 + 0.7 * self.wariness
            for o in p.neighbors:
                if o.infected:
                    dx, dy = self.x - o.x, self.y - o.y
                    d = math.hypot(dx, dy)
                    keep = self.radius + o.radius + 60
                    if 0 < d < keep:
                        w = (keep - d) / keep * caution * self.max_force * 2
                        fx += dx / d * w
                        fy += dy / d * w

        # Sprint only with stamina; once exhausted wait for recovery.
        if self.exhausted and self.stamina > 0.35:
            self.exhausted = False
        if self.stamina <= 0.02:
            self.exhausted = True
        eager = 0.5 + 0.5 * float(self.last_brain_output[7])
        self.sprinting = want_sprint and not self.exhausted and (s == "fleeing" or self.stamina > 0.3 * eager)
        return fx, fy

    def obstacle_avoidance(self, world):
        """Steer around desktop windows and god-drawn walls, sliding along their edges."""
        if not world.obstacles and not world.walls:
            return 0.0, 0.0
        fx = fy = 0.0
        look = 22 + self.radius
        px, py = self.x + self.vx * 12, self.y + self.vy * 12
        for wall in world.walls:
            wl = look + wall.half
            for qx, qy in ((self.x, self.y), (px, py)):
                cx, cy = closest_on_segment(qx, qy, wall.x0, wall.y0, wall.x1, wall.y1)
                dx, dy = qx - cx, qy - cy
                d = math.hypot(dx, dy)
                if d < 1e-6 or d >= wl:
                    continue
                w = (wl - d) / wl
                nx, ny = dx / d, dy / d
                tx, ty = -ny, nx
                sign = 1.0 if self.vx * tx + self.vy * ty >= 0 else -1.0
                fx += (nx + tx * sign * 0.6) * w
                fy += (ny + ty * sign * 0.6) * w
        reach = look + abs(self.vx * 12) + abs(self.vy * 12)
        for rect in world.obstacles:
            if not point_in_rect(self.x, self.y, rect, reach):
                continue
            for qx, qy in ((self.x, self.y), (px, py)):
                cx, cy = clamp(qx, rect[0], rect[2]), clamp(qy, rect[1], rect[3])
                dx, dy = qx - cx, qy - cy
                d = math.hypot(dx, dy)
                if d < 1e-6 or d >= look:
                    continue
                w = (look - d) / look
                nx, ny = dx / d, dy / d
                fx += nx * w
                fy += ny * w
                # Slide around the window instead of stopping dead.
                tx, ty = -ny, nx
                sign = 1.0 if self.vx * tx + self.vy * ty >= 0 else -1.0
                fx += tx * sign * w * 0.6
                fy += ty * sign * w * 0.6
        k = self.max_force * 1.6
        return fx * k, fy * k

    def perch_spot(self, world):
        """Top edge of the nearest window - a nice place for a nap."""
        best = None
        best_d = 350.0
        for rect in world.obstacles:
            x0, y0, x1, y1 = rect
            if y0 < self.radius * 2 + 4 or x1 - x0 < self.radius * 3:
                continue
            spot = (clamp(self.x, x0 + self.radius, x1 - self.radius), y0 - self.radius - 2)
            d = math.hypot(spot[0] - self.x, spot[1] - self.y)
            if d < best_d:
                best, best_d = spot, d
        return best

    def cursor_curiosity(self, world):
        """Attracted orbs come over to investigate and orbit the cursor."""
        if self.cursor_affinity <= 0.2:
            return 0.0, 0.0
        cx, cy = world.cursor
        dx, dy = cx - self.x, cy - self.y
        d = math.hypot(dx, dy)
        if d > self.vision * 1.3 or d < 1:
            return 0.0, 0.0
        orbit_r = 55 + self.radius * 2
        ux, uy = dx / d, dy / d
        radial = (d - orbit_r) / orbit_r                     # >0: too far, <0: too close
        tx, ty = -uy * self.juke_sign, ux * self.juke_sign   # tangent
        desire_x = (ux * clamp(radial, -1, 1) + tx * 0.8) * self.max_speed * 0.8
        desire_y = (uy * clamp(radial, -1, 1) + ty * 0.8) * self.max_speed * 0.8
        w = self.cursor_affinity * (0.4 + 0.6 * self.curiosity)
        return (desire_x - self.vx) * w, (desire_y - self.vy) * w

    def steer_hunt(self, world):
        t = self.target
        if t is CURSOR:
            cx, cy = world.cursor
            cvx, cvy = world.cursor_vel
            d = math.hypot(cx - self.x, cy - self.y)
            lead = clamp(d / (self.max_speed * 1.5), 0, 25)
            return self.seek(cx + cvx * lead, cy + cvy * lead, self.max_speed * self.sprint_mult), d < 200

        if not isinstance(t, IntelligentOrb) or not t.alive:
            self.target = None
            if self.prey_trail is not None:
                # Nose down, following a scent trail toward prey it can't see yet.
                self.tracking = True
                dx, dy, _ = self.prey_trail
                return self.seek(self.x + dx * 60, self.y + dy * 60, self.max_speed * 0.75), False
            return self.wander(), False

        # Prey slipped behind a window: search where it was last seen.
        if self.unseen > 0 and self.last_seen is not None:
            lx, ly = self.last_seen
            if self.unseen > 110 or math.hypot(lx - self.x, ly - self.y) < self.radius + 10:
                self.failed_targets[t.id] = self.age + 500
                self.reward(HUNT, -0.03)
                self.target = None
                self.last_seen = None
                self.unseen = 0
                return self.wander(), False
            return self.seek(lx, ly, self.max_speed * 1.1), False

        dx, dy = t.x - self.x, t.y - self.y
        d = math.hypot(dx, dy)

        # Give up chases that are going nowhere (and remember the one that got away).
        if d < self.chase_best - 4:
            self.chase_best = d
            self.chase_timer = 0
        else:
            self.chase_timer += 1
        if self.chase_timer > 200 or (self.exhausted and d > 120):
            self.failed_targets[t.id] = self.age + 700
            self.reward(HUNT, -0.06)
            self.target = None
            self.chase_timer = 0
            self.chase_best = 1e9
            if self.pack_leader:
                self.pack.target = None
            return self.wander(), False

        # Predict where the prey will be (intercept course).
        closing_speed = self.max_speed * (self.sprint_mult if self.sprinting else 1.0)
        lead = clamp(d / max(closing_speed, 0.5), 0, 35)
        px, py = t.x + t.vx * lead, t.y + t.vy * lead

        # Pack flanking: members approach from different angles to cut off escape.
        pack = self.pack
        if pack is not None and len(pack.members) > 1 and d > 60:
            leader = pack.leader
            base = math.atan2(leader.y - t.y, leader.x - t.x)
            offsets = (0.0, 1.2, -1.2, 2.3, -2.3)
            ang = base + offsets[pack.slot(self) % len(offsets)]
            ring = clamp(d * 0.45, 0, 90)
            px += math.cos(ang) * ring
            py += math.sin(ang) * ring

        pounce_dist = 110 + self.radius * 2
        unaware = t.behavior_state not in ("fleeing",) and t.target is not self
        if d > pounce_dist and unaware and pack is None:
            # Stalk: creep in slowly so the prey doesn't notice, save stamina for the pounce.
            self.stalking = True
            return self.seek(px, py, self.max_speed * 0.55), False
        return self.seek(px, py, self.max_speed * self.sprint_mult), d < pounce_dist * 1.6

    def steer_flee(self, world):
        p = self.percept
        fx = fy = 0.0
        nearest = None
        nearest_d = 1e9
        for o, level, tx, ty in p.threats:
            dx, dy = self.x - tx, self.y - ty
            d = max(math.hypot(dx, dy), 1.0)
            w = level * 120.0 / d
            fx += dx / d * w
            fy += dy / d * w
            if d < nearest_d:
                nearest_d, nearest = d, o
        if not p.threats and self.threat_memory:
            tx, ty, frames = self.threat_memory
            self.threat_memory[2] = frames - 1
            if self.threat_memory[2] <= 0:
                self.threat_memory = None
                self.reward(FLEE, 0.02)  # got away
                if world.is_followed(self) and self.age - self.last_escape_log > 1200:
                    self.last_escape_log = self.age
                    world.log(f"{self.name} got away", "followed")
            if world.obstacles and not world.line_of_sight(self.x, self.y, tx, ty):
                # Safely behind cover: lie low instead of breaking cover.
                self.hiding = True
                return (-self.vx * 0.2, -self.vy * 0.2), False
            dx, dy = self.x - tx, self.y - ty
            fx, fy = normalize(dx, dy)

        ux, uy = normalize(fx, fy)
        if ux == 0 and uy == 0:
            return self.wander(), False

        # Duck behind the nearest window to break the predator's line of sight.
        if world.obstacles and p.threats:
            top = max(p.threats, key=lambda t: t[1])
            cover = world.cover_point(self, top[2], top[3])
            if cover is not None:
                cdx, cdy = normalize(cover[0] - self.x, cover[1] - self.y)
                if cdx * ux + cdy * uy > -0.3:   # don't run past the predator to get there
                    ux, uy = normalize(ux * 0.4 + cdx * 0.6, uy * 0.4 + cdy * 0.6)

        # Don't run into a corner: slide along walls toward open space.
        margin = 80 + self.radius
        w, h = self.screen_width, self.screen_height
        if (self.x < margin and ux < 0) or (self.x > w - margin and ux > 0):
            ux = 0.0
            uy = uy if abs(uy) > 0.1 else (1.0 if self.y < h / 2 else -1.0)
        if (self.y < margin and uy < 0) or (self.y > h - margin and uy > 0):
            uy = 0.0
            ux = ux if abs(ux) > 0.1 else (1.0 if self.x < w / 2 else -1.0)
        ux, uy = normalize(ux, uy)

        # Juke: zig-zag when a hunter is right on top of us.
        if nearest is not None and nearest is not CURSOR:
            danger_d = (self.radius + nearest.radius) * 4
            if nearest_d < danger_d:
                self.juke_timer -= 1
                if self.juke_timer <= 0:
                    self.juke_sign = -self.juke_sign
                    self.juke_timer = random.randint(14, 30)
                k = self.juke_sign * 0.9
                ux, uy = normalize(ux - uy * k, uy + ux * k)

        speed = self.max_speed * self.sprint_mult
        sx, sy = ux * speed - self.vx, uy * speed - self.vy

        # Safety in numbers: social orbs flee toward their group.
        if p.allies and self.social_tendency > 0.5:
            n = len(p.allies)
            cx = sum(o.x for o in p.allies) / n
            cy = sum(o.y for o in p.allies) / n
            hx, hy = self.seek(cx, cy, speed)
            sx += hx * 0.35 * self.social_tendency
            sy += hy * 0.35 * self.social_tendency
        return (sx, sy), True

    def steer_forage(self, world):
        f = self.target
        if not isinstance(f, Food) or f.eaten:
            p = self.percept
            live = [(food, s) for food, s in p.foods if not food.eaten]
            f = self.target = live[0][0] if live else None
        if f is not None:
            if math.hypot(f.x - self.x, f.y - self.y) < self.radius + 8:
                self.forage_stuck += 1
                if self.forage_stuck > 25:
                    # It won't come off... this isn't food at all!
                    if isinstance(f, Lure):
                        self.known_anglers.add(f.angler.id)
                        self.reward(FORAGE, -0.03)
                    self.target = None
                    self.forage_stuck = 0
            else:
                self.forage_stuck = 0
            return self.arrive(f.x, f.y, 40, self.max_speed * 0.9)
        if self.food_memory:
            mx, my = self.food_memory
            if math.hypot(mx - self.x, my - self.y) < 40:
                self.food_memory = None
            else:
                return self.seek(mx, my, self.max_speed * 0.8)
        if self.food_trail is not None:
            # Follow the food trail others left, ant-style.
            self.following_trail = True
            dx, dy, _ = self.food_trail
            return self.seek(self.x + dx * 60, self.y + dy * 60, self.max_speed * 0.8)
        return self.wander()

    def steer_mate(self):
        m = self.mate_target
        if m is None or not m.alive:
            self.mate_target = None
            return self.wander()
        return self.arrive(m.x, m.y, 50)

    # ------------------------------------------------------------- physics
    def resolve_obstacles(self, world):
        """Windows and walls are solid: push the orb out and kill velocity into them."""
        r = self.radius
        for wall in world.walls:
            cx, cy = closest_on_segment(self.x, self.y, wall.x0, wall.y0, wall.x1, wall.y1)
            dx, dy = self.x - cx, self.y - cy
            d = math.hypot(dx, dy)
            reach = r + wall.half
            if d < reach:
                if d < 1e-6:   # dead centre: push out perpendicular to the wall
                    dx, dy = -(wall.y1 - wall.y0), wall.x1 - wall.x0
                    d = math.hypot(dx, dy) or 1.0
                nx, ny = dx / d, dy / d
                self.x = cx + nx * reach
                self.y = cy + ny * reach
                vn = self.vx * nx + self.vy * ny
                if vn < 0:
                    self.vx -= 1.3 * vn * nx
                    self.vy -= 1.3 * vn * ny
        for rect in world.obstacles:
            x0, y0, x1, y1 = rect
            if self.x < x0 - r or self.x > x1 + r or self.y < y0 - r or self.y > y1 + r:
                continue
            cx, cy = clamp(self.x, x0, x1), clamp(self.y, y0, y1)
            dx, dy = self.x - cx, self.y - cy
            d = math.hypot(dx, dy)
            if d > 1e-6:
                if d < r:
                    nx, ny = dx / d, dy / d
                    self.x += nx * (r - d)
                    self.y += ny * (r - d)
                    vn = self.vx * nx + self.vy * ny
                    if vn < 0:
                        self.vx -= 1.3 * vn * nx
                        self.vy -= 1.3 * vn * ny
            else:
                # A window was moved on top of us: get shoved out the nearest open side.
                exits = []
                for dist, nx, ny in ((self.x - x0, -1, 0), (x1 - self.x, 1, 0),
                                     (self.y - y0, 0, -1), (y1 - self.y, 0, 1)):
                    ex, ey = self.x + nx * (dist + r), self.y + ny * (dist + r)
                    if r <= ex <= self.screen_width - r and r <= ey <= self.screen_height - r:
                        exits.append((dist, nx, ny))
                if exits:
                    dist, nx, ny = min(exits)
                    step = min(dist + r + 1, 7.0)
                    self.x += nx * step
                    self.y += ny * step
                    self.vx = self.vx * 0.5 + nx * 1.5
                    self.vy = self.vy * 0.5 + ny * 1.5

    def integrate(self, fx, fy, dt, world=None):
        max_f = self.max_force * (1.6 if self.sprinting else 1.0)
        fx, fy = limit(fx, fy, max_f)
        self.vx += fx * dt
        self.vy += fy * dt

        energy_factor = 0.65 + 0.35 * min(1.0, self.energy / 35.0)
        cap = self.max_speed * energy_factor * (self.sprint_mult if self.sprinting else 1.0)
        self.vx, self.vy = limit(self.vx, self.vy, cap)
        self.vx *= 0.995
        self.vy *= 0.995
        if world is not None:
            self.vx += world.env.get("wind", 0.0) * dt   # blizzard winds push everyone

        self.x += self.vx * dt
        self.y += self.vy * dt

        r = self.radius
        if self.x < r:
            self.x, self.vx = r, abs(self.vx) * 0.5
        elif self.x > self.screen_width - r:
            self.x, self.vx = self.screen_width - r, -abs(self.vx) * 0.5
        if self.y < r:
            self.y, self.vy = r, abs(self.vy) * 0.5
        elif self.y > self.screen_height - r:
            self.y, self.vy = self.screen_height - r, -abs(self.vy) * 0.5
        if world is not None and (world.obstacles or world.walls):
            self.resolve_obstacles(world)

        speed = math.hypot(self.vx, self.vy)
        if speed > 0.25:
            target_heading = math.atan2(self.vy, self.vx)
            diff = (target_heading - self.heading + math.pi) % math.tau - math.pi
            self.heading += diff * 0.25

        if self.sprinting:
            self.stamina = max(0.0, self.stamina - 1 / 120)
            self.trail.append((self.x, self.y))
            if len(self.trail) > 12:
                self.trail.pop(0)
        else:
            regen = 1 / 110 if self.behavior_state == "resting" else 1 / 320
            self.stamina = min(1.0, self.stamina + regen)
            if self.trail:
                self.trail.pop(0)

        # Growth from hunting slowly fades back toward the genetic size.
        if self.radius > self.size_gene + 0.05:
            self.radius -= (self.radius - self.size_gene) * 0.0004
            if self.age % 30 == 0:
                self.update_physique()

    def metabolize(self, dt, world=None):
        speed_frac = math.hypot(self.vx, self.vy) / max(self.max_speed, 0.1)
        cost = 0.008 + 0.0005 * self.radius + 0.018 * min(speed_frac, 2.0) ** 2
        if self.sprinting:
            cost += 0.015
        cold = world.env["cold"] if world is not None else 1.0
        if self.behavior_state == "resting":
            # Sleeping (and hibernating in the cold) slows metabolism a lot.
            cost *= 0.6 if cold < 1.15 else 0.4
        elif self.behavior_state == "lurking":
            cost *= 0.4
        if self.species == "mimic":
            cost += 0.003 * self.disguise    # keeping up the bluff isn't free
        # Cold costs more energy - bigger bodies hold heat better.
        cost *= 1.0 + (cold - 1.0) * (1.3 - self.radius / MAX_RADIUS)
        self.energy -= cost * dt
        self.survival_score += 0.01 * dt

    def interact(self, world):
        # Eat any food we touch while hungry (anglers are strict carnivores; scavengers
        # only graze when desperate).
        grazer = not self.is_angler and self.species != "leech" and \
            (self.species != "scavenger" or self.energy < 30)
        if world.food is not None and self.energy < MAX_ENERGY - 3 and grazer:
            for f in world.food.near(self.x, self.y, self.radius + 4):
                self.energy = min(MAX_ENERGY, self.energy + f.value)
                if self.species == "splitter":   # every meal makes a splitter bigger
                    self.radius = min(self.size_gene + 1.0, self.radius + 0.45)
                    self.update_physique()
                self.food_eaten += 1
                self.food_scent_timer = 600     # lay a food trail for a while
                self.survival_score += 1
                self.reward(FORAGE, 0.02)
                world.food.remove(f)
                world.counters["food_eaten"] += 1
                if self.target is f:
                    self.target = None
                break

        if self.behavior_state == "hunting" and isinstance(self.target, IntelligentOrb):
            t = self.target
            if t.alive and math.hypot(t.x - self.x, t.y - self.y) < (self.radius + t.radius) * 0.9:
                world.try_kill(self, t)

        if self.behavior_state == "defending" and isinstance(self.target, IntelligentOrb):
            t = self.target
            if t.alive and math.hypot(t.x - self.x, t.y - self.y) < self.radius + t.radius + 2:
                if self.can_eat(t, assisted=self.pack_assist(t)):
                    world.try_kill(self, t)
                elif self.age >= self.next_bite:
                    # A skirmish: a shove and a nip to drive the rival away.
                    self.next_bite = self.age + 45
                    dx, dy = normalize(t.x - self.x, t.y - self.y)
                    t.vx += dx * 3.5
                    t.vy += dy * 3.5
                    t.energy -= 4
                    t.threat_memory = [self.x, self.y, 120]
                    world.add_effect((t.x + self.x) / 2, (t.y + self.y) / 2, (255, 170, 60), 26, 16)

        if self.behavior_state == "mating" and self.mate_target is not None:
            m = self.mate_target
            if m.alive and math.hypot(m.x - self.x, m.y - self.y) < self.radius + m.radius + 6:
                world.reproduce(self, m)

    # ------------------------------------------------------------- compat
    def hunt(self, target):
        """Attempt to catch and eat target. Returns True on a kill."""
        return World.adhoc([self, target], (-1e4, -1e4), self.screen_width, self.screen_height).try_kill(self, target)

    def grow(self, amount):
        self.radius = min(MAX_RADIUS, self.radius + amount)
        self.update_physique()

    def mate_with(self, partner):
        """Create a child from two parents (crossover + mutation)."""
        if self.reproductive_cooldown > 0 or partner.reproductive_cooldown > 0:
            return None
        if not (self.alive and partner.alive):
            return None

        fa = self.lifetime_fitness() + 1
        fb = partner.lifetime_fitness() + 1
        wa = fa / (fa + fb)

        cx = (self.x + partner.x) / 2 + random.uniform(-20, 20)
        cy = (self.y + partner.y) / 2 + random.uniform(-20, 20)
        child = orb_class(self.species)(cx, cy, self.screen_width, self.screen_height, species=self.species)
        child.wariness = self.wariness * wa + partner.wariness * (1 - wa)
        child.night_vision = self.night_vision * wa + partner.night_vision * (1 - wa)
        child.disguise = self.disguise * wa + partner.disguise * (1 - wa)
        child.brain = OrbBrain.crossover(self.brain, partner.brain, bias_a=0.3 + 0.4 * wa)
        child.brain.mutate(rate=0.08, sigma=0.25)

        def blend(a, b):
            return a * wa + b * (1 - wa)

        child.aggression = blend(self.aggression, partner.aggression)
        child.curiosity = blend(self.curiosity, partner.curiosity)
        child.social_tendency = blend(self.social_tendency, partner.social_tendency)
        child.cursor_affinity = blend(self.cursor_affinity, partner.cursor_affinity)
        child.size_gene = blend(self.size_gene, partner.size_gene)
        child.mutate_traits(0.05)
        child.generation = max(self.generation, partner.generation) + 1
        # The family line follows the more successful parent.
        child.house = (self.house if wa >= 0.5 else partner.house) or self.house or partner.house
        child.parents = (self.name, partner.name)
        self.children.append(child)
        partner.children.append(child)
        child.radius = child.size_gene * 0.75   # babies start small and grow up
        child.update_physique()
        child.shape = child.determine_shape()
        child.base_color = child.current_color = child.generate_personality_color()

        # Cultural transmission: children inherit part of what their parents learned.
        child.behavior_values = [0.3 * blend(a, b) for a, b in zip(self.behavior_values, partner.behavior_values)]

        # Energy is transferred, not created.
        child.energy = 45.0
        self.energy -= 25
        partner.energy -= 25
        cooldown = random.randint(800, 1300)
        self.reproductive_cooldown = cooldown
        partner.reproductive_cooldown = cooldown
        self.offspring_count += 1
        partner.offspring_count += 1
        self.survival_score += 10
        partner.survival_score += 10
        self.reward(MATE, 0.1)
        partner.reward(MATE, 0.1)
        for o in (self, partner):
            o.mate_target = None
            o.courted_by = None
            o.behavior_state = "lurking" if o.is_angler else "exploring"
            o.state_timer = 0
        return child


# ----------------------------------------------------------------------------
# The leviathan: a rare giant that drifts across the world
# ----------------------------------------------------------------------------

LEVIATHAN_NAMES = ("Old Gorm", "Vastmaw", "the Drifting King", "Umbrel the Deep", "Grandmother Tide",
                   "the Hollow Whale", "Moorgloom", "the Pale Leviathan")
LEVIATHAN_HEALTH = 300.0


class Leviathan(IntelligentOrb):
    """Enters from one edge, cruises to the opposite one swallowing anything it touches,
    and sinks back into the deep. A big enough pack can wear it down and feast."""

    def __init__(self, world):
        w, h = world.width, world.height
        side = random.choice("lrtb")
        m = 80      # starts just off-screen...
        far = 260   # ...and swims to a point well past the far edge
        if side == "l":
            start, exit_ = (-m, random.uniform(0.2, 0.8) * h), (w + far, random.uniform(0.2, 0.8) * h)
        elif side == "r":
            start, exit_ = (w + m, random.uniform(0.2, 0.8) * h), (-far, random.uniform(0.2, 0.8) * h)
        elif side == "t":
            start, exit_ = (random.uniform(0.2, 0.8) * w, -m), (random.uniform(0.2, 0.8) * w, h + far)
        else:
            start, exit_ = (random.uniform(0.2, 0.8) * w, h + m), (random.uniform(0.2, 0.8) * w, -far)
        super().__init__(start[0], start[1], w, h, species="leviathan")
        self.name = random.choice(LEVIATHAN_NAMES)
        self.aggression, self.social_tendency, self.curiosity = 1.0, 0.0, 0.2
        self.cursor_affinity, self.wariness, self.night_vision = 0.0, 0.5, 0.8
        self.radius = self.size_gene = 54.0
        self.energy = LEVIATHAN_HEALTH       # its energy is its health
        self.reproductive_cooldown = 10 ** 9
        self.max_age = 10 ** 9
        self.exit = exit_
        self.behavior_state = "cruising"
        self.heading = math.atan2(exit_[1] - start[1], exit_[0] - start[0])
        self.vx, self.vy = math.cos(self.heading) * 1.2, math.sin(self.heading) * 1.2
        self.meals = 0
        self.digesting = 0
        self.base_color = self.current_color = (40, 70, 110)
        self.shape = "leviathan"
        self.update_physique()

    def update_physique(self):
        self.mass = self.radius * 0.5
        self.max_speed = 1.3
        self.max_force = 0.06
        self.sprint_mult = 2.0
        self.vision = 260

    def update(self, cursor_pos, other_orbs, dt=1.0, world=None):
        if not self.alive:
            self.update_explosion()
            return
        if world is None:
            return
        self.age += 1
        ex, ey = self.exit
        fx, fy = self.seek(ex, ey, self.max_speed)
        wobble = math.sin(self.age * 0.012) * 0.05     # a slow, serpentine sway
        fx += -math.sin(self.heading) * wobble
        fy += math.cos(self.heading) * wobble

        # Lunge at whatever strays too close (not while digesting the last meal).
        self.behavior_state, self.target, self.sprinting = "cruising", None, False
        if self.digesting > 0:
            self.digesting -= 1
        prey = [o for o in world.neighbors(self, 170) if o.species != "leviathan"] if not self.digesting else []
        if prey:
            t = min(prey, key=lambda o: math.hypot(o.x - self.x, o.y - self.y))
            self.behavior_state, self.target, self.sprinting = "hunting", t, True
            fx, fy = self.seek(t.x + t.vx * 10, t.y + t.vy * 10, self.max_speed * self.sprint_mult)

        fx, fy = limit(fx, fy, self.max_force * (2.0 if self.sprinting else 1.0))
        self.vx += fx * dt
        self.vy += fy * dt
        self.vx, self.vy = limit(self.vx, self.vy, self.max_speed * (self.sprint_mult if self.sprinting else 1.0))
        self.x += self.vx * dt
        self.y += self.vy * dt
        if math.hypot(self.vx, self.vy) > 0.2:
            target_heading = math.atan2(self.vy, self.vx)
            self.heading += ((target_heading - self.heading + math.pi) % math.tau - math.pi) * 0.05

        # Swallow whatever ends up right in its mouth, then pause to digest.
        if not self.digesting:
            for o in world.neighbors(self, self.radius + 40):
                if o.alive and o.species != "leviathan" and \
                        math.hypot(o.x - self.x, o.y - self.y) < self.radius * 0.85 + o.radius * 0.2:
                    world.leviathan_eat(self, o)
                    self.digesting = 90
                    break

        # Sink back into the deep once it has crossed the world.
        w, h = world.width, world.height
        gone = self.radius + 40
        if self.age > 300 and not (-gone < self.x < w + gone and -gone < self.y < h + gone):
            self.alive = False
            self.departed = True
            world.on_leviathan_leave(self)

    def describe_activity(self, world=None):
        if not self.alive:
            return "was slain" if self.cause_of_death == "slain" else "sank back into the deep"
        if self.behavior_state == "hunting" and self.target is not None:
            return f"lunging at {self.target.name}"
        return f"drifting across the world ({self.meals} swallowed)"


# ----------------------------------------------------------------------------
# Leeches: tiny parasites that latch onto big orbs and drain them
# ----------------------------------------------------------------------------

MAX_LEECHES_PER_HOST = 3


class Leech(IntelligentOrb):
    """Hunts for a big host, clings to its rim and sips its energy. Hosts shake leeches
    off by sprinting; relatives groom them off; anyone bigger can simply eat one.
    Well-fed leeches bud into two."""

    def __init__(self, x, y, screen_width, screen_height, genome=None, species="leech"):
        super().__init__(x, y, screen_width, screen_height, genome=genome, species="leech")
        self.aggression, self.social_tendency, self.cursor_affinity = 0.5, 0.0, 0.0
        self.radius = self.size_gene = random.uniform(4.5, 6.5)
        self.energy = 55.0
        self.host = None
        self.attach_angle = 0.0
        self.latch_cooldown = 0
        self.drained = 0.0
        self.hosts_bled = 0
        self.behavior_state = "seeking"
        self.base_color = self.current_color = (170, 35, 60)
        self.shape = "leech"
        self.update_physique()

    def update_physique(self):
        self.mass = self.radius * 0.5
        self.max_speed = 3.6
        self.max_force = 0.35
        self.sprint_mult = 1.3
        self.vision = 230

    def update_color(self):
        e = 0.5 + 0.5 * clamp(self.energy / MAX_ENERGY, 0, 1)
        self.current_color = (int(170 * e + 40), int(35 * e), int(60 * e))

    def determine_shape(self):
        return "leech"

    def detach(self, fling=True):
        h = self.host
        self.host = None
        self.latch_cooldown = 150
        self.behavior_state = "seeking"
        if h is not None and fling:
            a = self.attach_angle
            self.vx, self.vy = h.vx + math.cos(a) * 3.5, h.vy + math.sin(a) * 3.5

    def pick_host(self, world, neighbors):
        best, best_score = None, 0.0
        for o in neighbors:
            if o.species in ("leech", "leviathan") or o.radius < 12 or not o.alive:
                continue
            if world.leech_load(o) >= MAX_LEECHES_PER_HOST:
                continue
            d = math.hypot(o.x - self.x, o.y - self.y)
            # Big, slow and sleepy hosts are the best targets.
            score = o.radius * (1.8 if o.behavior_state in ("resting", "lurking") else 1.0) / (d + 40)
            if score > best_score:
                best, best_score = o, score
        return best

    def update(self, cursor_pos, other_orbs, dt=1.0, world=None):
        if not self.alive:
            self.update_explosion()
            return
        if world is None:
            return
        self.age += 1
        if self.latch_cooldown > 0:
            self.latch_cooldown -= 1
        h = self.host
        if h is not None and not h.alive:
            self.detach(fling=False)
            h = None

        if h is not None:
            # Clinging to the host's rim, slowly creeping around it.
            self.behavior_state = "attached"
            self.attach_angle += 0.008
            self.x = h.x + math.cos(self.attach_angle) * (h.radius + 1)
            self.y = h.y + math.sin(self.attach_angle) * (h.radius + 1)
            self.vx, self.vy = h.vx, h.vy
            drain = 0.025 * dt
            h.energy -= drain
            self.energy = min(MAX_ENERGY, self.energy + drain * 0.8 - 0.002 * dt)
            self.drained += drain
            # Shaking it off: sprinting hosts fling leeches; relatives groom them off.
            p = 0.0004 + (0.008 if h.sprinting else 0.0)
            if self.age % 10 == 0 and h.species == "orb":
                for n in world.neighbors(h, h.radius + 30):
                    if n is not self and n.species == h.species and n.is_kin(h) and n.alive:
                        p += 0.08     # (checked every 10 frames)
                        if random.random() < p:
                            self.detach()
                            if world.is_followed(h) or random.random() < 0.3:
                                world.log(f"{n.name} groomed a leech off {h.name}",
                                          "followed" if world.is_followed(h) else "life")
                            break
            if self.host is not None and random.random() < p:
                self.detach()
        else:
            self.behavior_state = "seeking"
            if self.percept is None or (self.age + self.think_offset) % THINK_INTERVAL == 0:
                self.percept = self.sense(world)
                self.target = self.pick_host(world, self.percept.neighbors)
            p = self.percept
            fx = fy = 0.0
            hunters = [t for t in p.threats if t[0] is not CURSOR and t[0].target is self]
            if hunters:
                self.behavior_state = "fleeing"
                (fx, fy), _ = self.steer_flee(world)
                self.sprinting = True
            elif isinstance(self.target, IntelligentOrb) and self.target.alive:
                t = self.target
                d = math.hypot(t.x - self.x, t.y - self.y)
                lead = clamp(d / self.max_speed, 0, 20)
                fx, fy = self.seek(t.x + t.vx * lead, t.y + t.vy * lead)
                self.sprinting = d < 120
                if d < t.radius + self.radius + 2 and self.latch_cooldown == 0 and \
                        world.leech_load(t) < MAX_LEECHES_PER_HOST:
                    self.host = t
                    self.attach_angle = math.atan2(self.y - t.y, self.x - t.x)
                    self.hosts_bled += 1
                    if world.is_followed(t):
                        world.log(f"A leech latched onto {t.name}!", "followed")
            else:
                fx, fy = self.wander()
                self.sprinting = False
            sx, sy = self.wall_avoidance()
            ox, oy = self.obstacle_avoidance(world)
            self.integrate(fx + sx + ox, fy + sy + oy, dt, world)
            self.energy -= (0.006 + (0.01 if self.sprinting else 0.0)) * dt

        # A well-fed leech buds into two.
        if self.energy > 85 and world.room_for_births() and world.species_count("leech") < world.leech_cap():
            world.bud_leech(self)
        self.update_color()
        if self.energy <= 0:
            if self.host is not None:
                self.detach(fling=False)
            self.die("starved", world)
        elif self.age > self.max_age:
            self.die("old age", world)

    def die(self, cause, world=None):
        self.host = None
        super().die(cause, world)

    def describe_activity(self, world=None):
        if not self.alive:
            return super().describe_activity(world)
        if self.host is not None:
            return f"feeding on {self.host.name}"
        if self.behavior_state == "fleeing":
            return "fleeing from a hungry orb"
        if isinstance(self.target, IntelligentOrb):
            return f"homing in on {self.target.name}"
        return "drifting in search of a host"


# ----------------------------------------------------------------------------
# Scavengers: carrion eaters that shadow predators
# ----------------------------------------------------------------------------

class Scavenger(IntelligentOrb):
    """Never kills. Smells remains from afar, shadows hunting predators at a safe distance
    waiting for leftovers, flees anything that might eat it, and pairs up to breed."""

    SMELL_RANGE = 700

    def __init__(self, x, y, screen_width, screen_height, genome=None, species="scavenger"):
        super().__init__(x, y, screen_width, screen_height, genome=genome, species="scavenger")
        if not genome:
            self.aggression = random.uniform(0.05, 0.25)
            self.social_tendency = random.uniform(0.3, 0.8)
            self.curiosity = random.uniform(0.4, 0.9)
            self.size_gene = random.uniform(9, 13)
        self.radius = self.size_gene
        self.carrion_eaten = 0
        self.shadowing = None
        self.base_color = self.current_color = self.generate_personality_color()
        self.shape = "scavenger"
        self.update_physique()

    def update_physique(self):
        super().update_physique()
        self.max_speed += 0.4           # light, fast flyers
        self.vision = 260 + 60 * self.curiosity

    def generate_personality_color(self):
        return (int(130 + 40 * self.curiosity), int(115 + 30 * self.social_tendency), 95)

    def determine_shape(self):
        return "scavenger"

    def update(self, cursor_pos, other_orbs, dt=1.0, world=None):
        if not self.alive:
            self.update_explosion()
            return
        if world is None:
            return
        self.age += 1
        self.state_timer += 1
        if self.reproductive_cooldown > 0:
            self.reproductive_cooldown -= 1

        if self.percept is None or (self.age + self.think_offset) % THINK_INTERVAL == 0:
            self.think_scavenger(world)

        p = self.percept
        s = self.behavior_state
        want_sprint = False
        sep_ignore = None
        if s == "fleeing":
            (fx, fy), want_sprint = self.steer_flee(world)
        elif s == "scavenging" and isinstance(self.target, Carrion):
            fx, fy = self.arrive(self.target.x, self.target.y, 40)
            want_sprint = math.hypot(self.target.x - self.x, self.target.y - self.y) > 200
        elif s == "shadowing" and self.shadowing is not None and self.shadowing.alive:
            pred = self.shadowing
            d = math.hypot(pred.x - self.x, pred.y - self.y)
            # Hang back behind the predator: close enough to see the kill, too far to be the kill.
            bx = pred.x - math.cos(pred.heading) * 130
            by = pred.y - math.sin(pred.heading) * 130
            fx, fy = self.arrive(bx, by, 60)
            if d < 90:
                ax, ay = normalize(self.x - pred.x, self.y - pred.y)
                fx += ax * self.max_force * 2
                fy += ay * self.max_force * 2
        elif s == "mating" and self.mate_target is not None:
            fx, fy = self.steer_mate()
            sep_ignore = self.mate_target
        elif s == "resting":
            fx, fy = -self.vx * 0.1, -self.vy * 0.1
        else:
            fx, fy = self.wander()
        sx, sy = self.separation(p.neighbors, sep_ignore)
        wx, wy = self.wall_avoidance(2.0 if s == "fleeing" else 1.0)
        ox, oy = self.obstacle_avoidance(world)
        if self.exhausted and self.stamina > 0.35:
            self.exhausted = False
        if self.stamina <= 0.02:
            self.exhausted = True
        self.sprinting = want_sprint and not self.exhausted
        self.integrate(fx + sx + wx + ox, fy + sy + wy + oy, dt, world)

        # Pick at any remains we reach.
        for c in world.carrion_near(self.x, self.y, self.radius + 8):
            bite = min(c.value, 18.0)
            self.energy = min(MAX_ENERGY, self.energy + bite)
            c.value -= bite
            self.carrion_eaten += 1
            self.survival_score += 2
            world.counters["carrion_eaten"] += 1
            if c.value <= 0.5:
                world.remove_carrion(c)
                if c.radius >= 22 and random.random() < 0.25:
                    world.log(f"{self.name} the scavenger picked clean the remains of {c.name}", "info")
            break
        self.interact(world)       # desperate grazing + mating
        # A well-fed scavenger can lay an egg on its own if no partner turns up.
        if (self.energy > 88 and self.age > MATURITY_AGE and self.reproductive_cooldown == 0 and
                world.room_for_births("scavenger") and not world.species_full("scavenger")):
            world.lay_egg(self)
        self.metabolize(dt, world)
        self.update_color()
        if self.energy <= 0:
            self.die("starved", world)
        elif self.age > self.max_age:
            self.die("old age", world)

    def think_scavenger(self, world):
        p = self.percept = self.sense(world)
        if p.threat_level > 0.3:
            new = "fleeing"
            top = max(p.threats, key=lambda t: t[1])
            self.threat_memory = [top[2], top[3], 60]
        else:
            new = None
            best, best_score = None, 0.0
            if self.energy < 92:
                for c in world.carrion_near(self.x, self.y, self.SMELL_RANGE):
                    score = c.value / (math.hypot(c.x - self.x, c.y - self.y) + 60)
                    if score > best_score:
                        best, best_score = c, score
            if best is not None:
                new, self.target = "scavenging", best
            elif self.willing_to_mate() and p.mates and world.room_for_births() and not world.species_full("scavenger"):
                new = "mating"
                self.select_mate(p)
            else:
                hunters = [o for o in world.neighbors(self, 500)
                           if o.behavior_state == "hunting" and o.species in ("orb", "angler", "leviathan")]
                if hunters and self.energy < 85:
                    new = "shadowing"
                    self.shadowing = min(hunters, key=lambda o: math.hypot(o.x - self.x, o.y - self.y))
                elif world.is_night and self.night_vision < 0.5 or self.stamina < 0.3:
                    new = "resting"
                else:
                    new = "exploring"
        if new != self.behavior_state:
            self.behavior_state = new
            self.state_timer = 0
            if new != "mating":
                self.mate_target = None
            if new != "shadowing":
                self.shadowing = None

    def describe_activity(self, world=None):
        if not self.alive:
            return super().describe_activity(world)
        s = self.behavior_state
        if s == "scavenging" and isinstance(self.target, Carrion):
            return f"heading for the remains of {self.target.name}"
        if s == "shadowing" and self.shadowing is not None:
            return f"shadowing {self.shadowing.name}, waiting for leftovers"
        return super().describe_activity(world)


def orb_class(species):
    """The Python class that implements a species."""
    return {"leech": Leech, "scavenger": Scavenger}.get(species, IntelligentOrb)


# ----------------------------------------------------------------------------
# World: the whole ecosystem, independent of any GUI
# ----------------------------------------------------------------------------

class World:
    def __init__(self, width, height, with_food=True, genepool_file=None):
        self.width, self.height = width, height
        self.orbs = []
        self.newborns = []
        self.food = FoodField(width, height) if with_food else None
        self.scent = ScentField(width, height) if with_food else None
        self.cursor = (-10000.0, -10000.0)
        self.cursor_vel = (0.0, 0.0)
        self.frame = 0
        self.max_orbs = int(clamp(width * height / 40000, 25, 70))
        self.min_orbs = int(clamp(width * height / 300000, 8, 20))
        self.orb_hash = SpatialHash(150)
        self._grid_frame = -1
        self.adhoc_mode = False
        self.packs = []
        self.effects = []
        self.obstacles = []      # desktop windows as (x0, y0, x1, y1)
        self.lures = []
        self.alarms = []
        # World events and god tools (must exist before update_environment runs).
        self.active_events = {}  # name -> {"start", "end", ...}
        self.random_events = True
        self.next_event_frame = int(DAY_FRAMES * random.uniform(1.0, 2.0))
        self.walls = []          # temporary god-drawn walls
        self.terrain_rects = []
        self.meteors = []
        self.bolts = []          # lightning strikes to draw: [points, life]
        self.plague = None
        self.leviathan = None
        self.carrion = []        # remains left by the dead (scavenger food)
        self.clock = DAY_FRAMES * 0.30   # start at about 7am on the first day of spring
        self.light = 1.0
        self.env = dict(SEASON_PARAMS["spring"])
        self.update_environment()
        # One sample per simulated second, 30 minutes of history, for the stats panel.
        self.history = deque(maxlen=HISTORY_LENGTH)
        self.counters = {"births": 0, "deaths": 0, "kills": 0, "starved": 0,
                         "old_age": 0, "food_eaten": 0, "repelled": 0,
                         "alarms": 0, "angler_kills": 0, "plague": 0, "struck": 0, "meteor": 0,
                         "leviathan_kills": 0, "events": 0, "splits": 0, "leech_buds": 0,
                         "carrion_eaten": 0,
                         "mimics_eaten": 0, "border_clashes": 0, "intruders_driven_off": 0}
        self.hall_of_fame = []   # list of {"score": float, "genome": dict}
        self.genepool_file = genepool_file
        # Stories: family lines, a log of notable events, and the orb the viewer follows.
        self.houses = {}          # house -> {"members", "founded_day", "kills", "returned"}
        self.events = deque(maxlen=200)
        self.event_seq = 0
        self.followed = None
        self.champion = None
        self.max_generation = 0
        self._last_season = self.season_index

    # ------------------------------------------------------------ stories
    def log(self, text, kind="info"):
        """Record a notable moment for the event ticker. Kinds: info, danger, life,
        season, champion, followed."""
        if self.adhoc_mode:
            return
        self.event_seq += 1
        self.events.append({"seq": self.event_seq, "frame": self.frame, "day": self.day,
                            "clock": self.clock_string(), "text": text, "kind": kind})

    def is_followed(self, orb):
        return orb is not None and orb is self.followed

    def new_house_name(self):
        unused = [w for w in HOUSE_WORDS if w not in self.houses]
        return random.choice(unused) if unused else make_name()

    def register(self, orb, founder=False):
        """Track membership of family lines (founders start or revive a house)."""
        if orb.species == "leech":
            return
        if orb.house is None:
            orb.house = self.new_house_name()
        info = self.houses.get(orb.house)
        if info is None:
            self.houses[orb.house] = {"members": 1, "founded_day": self.day, "kills": 0, "extinct": False}
        else:
            info["members"] += 1
            if info.get("extinct") and founder:
                info["extinct"] = False
                notable = info["members"] >= 6 or info["kills"] >= 5
                if self.frame > 0 and notable and self.day - info.get("returned_day", -99) >= 2:
                    info["returned_day"] = self.day
                    self.log(f"House {orb.house} returns: {orb.name} is descended from past champions", "life")

    def house_alive(self, house, excluding=None):
        return any(o.alive and o.house == house and o is not excluding for o in self.orbs) or \
            any(o.house == house for o in self.newborns)

    # ----------------------------------------------------------- creation
    @classmethod
    def adhoc(cls, orbs, cursor, width, height):
        """A throwaway world wrapping a plain list of orbs (no food, no grid)."""
        w = cls(width, height, with_food=False)
        w.orbs = orbs
        w.cursor = (float(cursor[0]), float(cursor[1]))
        w.adhoc_mode = True
        return w

    def populate(self, count=None):
        count = count or int(clamp(self.width * self.height / 120000, 14, 40))
        n_anglers = max(2, count // 9)
        for i in range(count - n_anglers):
            # Half of the founders descend from the best genomes of previous runs.
            use_pool = bool(self.hall_of_fame) and i % 2 == 0
            self.spawn_orb(from_pool=use_pool)
        for _ in range(n_anglers):
            self.spawn_orb(species="angler")
        for species, n in self.NEW_SPECIES_START(count):
            for _ in range(n):
                self.spawn_orb(species=species)

    @staticmethod
    def NEW_SPECIES_START(count):
        return (("splitter", max(3, count // 5)), ("scavenger", max(2, count // 8)), ("leech", max(2, count // 8)),
                ("mimic", max(3, count // 7)))

    def species_count(self, species):
        return (sum(1 for o in self.orbs if o.alive and o.species == species) +
                sum(1 for o in self.newborns if o.species == species))

    SPECIES_SHARE = {"mimic": 0.2, "splitter": 0.3, "angler": 0.15, "scavenger": 0.15}

    def species_full(self, species):
        """Some species stop breeding once they make up too much of the population."""
        share = self.SPECIES_SHARE.get(species)
        return share is not None and self.species_count(species) >= self.max_orbs * share

    def leech_cap(self):
        hosts = sum(1 for o in self.orbs if o.alive and o.radius >= 12 and o.species != "leech")
        return max(3, hosts // 3)

    def leech_load(self, host):
        return sum(1 for o in self.orbs if o.alive and o.species == "leech" and o.host is host)

    def random_position(self, margin=60):
        for _ in range(30):
            x = random.uniform(margin, self.width - margin)
            y = random.uniform(margin, self.height - margin)
            if math.hypot(x - self.cursor[0], y - self.cursor[1]) > 150 and \
                    not any(point_in_rect(x, y, r, 30) for r in self.obstacles):
                return x, y
        return x, y

    def angler_count(self):
        return sum(1 for o in self.orbs if o.alive and o.is_angler)

    def spawn_orb(self, from_pool=None, x=None, y=None, species="orb"):
        if x is None:
            x, y = self.random_position()
        if from_pool is None:
            from_pool = bool(self.hall_of_fame) and random.random() < 0.5
        pool = [e for e in self.hall_of_fame if e["genome"].get("species", "orb") == species]
        orb = None
        if from_pool and pool:
            entry = random.choice(pool)
            try:
                orb = orb_class(species)(x, y, self.width, self.height, genome=entry["genome"], species=species)
                orb.brain.mutate(rate=0.05, sigma=0.15)
                orb.mutate_traits(0.04)
                orb.radius = orb.size_gene * (0.6 if species == "splitter" else 1.0)
                orb.update_physique()
                orb.shape = orb.determine_shape()
                orb.base_color = orb.current_color = orb.generate_personality_color()
            except Exception:
                orb = None
        if orb is None:
            orb = orb_class(species)(x, y, self.width, self.height, species=species)
        if orb.is_angler:
            orb.behavior_state = "lurking"
        self.register(orb, founder=True)
        self.orbs.append(orb)
        return orb

    # ------------------------------------------------------------- queries
    def set_cursor(self, pos):
        x, y = float(pos[0]), float(pos[1])
        ox, oy = self.cursor
        if ox > -9000:
            vx = x - ox
            vy = y - oy
            self.cursor_vel = (self.cursor_vel[0] * 0.6 + vx * 0.4, self.cursor_vel[1] * 0.6 + vy * 0.4)
        self.cursor = (x, y)

    def neighbors(self, orb, r):
        if self._grid_frame == self.frame:
            cands = self.orb_hash.query(orb.x, orb.y, r)
            return [o for o in cands if o is not orb and o.alive]
        r2 = r * r
        return [o for o in self.orbs if o is not orb and o.alive and
                (o.x - orb.x) ** 2 + (o.y - orb.y) ** 2 <= r2]

    def alive_orbs(self):
        return [o for o in self.orbs if o.alive]

    # --------------------------------------------------------- time & season
    @property
    def time_of_day(self):
        """0.0 = midnight, 0.5 = noon."""
        return (self.clock % DAY_FRAMES) / DAY_FRAMES

    @property
    def day(self):
        return int(self.clock // DAY_FRAMES) + 1

    @property
    def season_index(self):
        return int(self.clock // (DAY_FRAMES * DAYS_PER_SEASON)) % len(SEASONS)

    @property
    def season(self):
        return SEASONS[self.season_index]

    @property
    def season_progress(self):
        span = DAY_FRAMES * DAYS_PER_SEASON
        return (self.clock % span) / span

    @property
    def is_night(self):
        return self.light < 0.4

    def clock_string(self):
        minutes = int(self.time_of_day * 24 * 60)
        return f"{minutes // 60:02d}:{minutes % 60:02d}"

    def update_environment(self):
        sun = -math.cos(self.time_of_day * math.tau)      # -1 midnight .. +1 noon
        self.light = clamp(0.55 + 0.75 * sun, 0.12, 1.0)
        # Blend smoothly into the next season over the last fifth of each season.
        cur = SEASON_PARAMS[self.season]
        nxt = SEASON_PARAMS[SEASONS[(self.season_index + 1) % len(SEASONS)]]
        b = clamp((self.season_progress - 0.8) / 0.2, 0, 1)
        env = {}
        for k, v in cur.items():
            if k == "food_color":
                env[k] = tuple(int(a + (c - a) * b) for a, c in zip(v, nxt[k]))
            else:
                env[k] = v + (nxt[k] - v) * b
        env["season"] = self.season

        # World events bend the rules for a while.
        ev = self.active_events
        if "eclipse" in ev:
            e = ev["eclipse"]
            p = clamp((self.frame - e["start"]) / max(1, e["end"] - e["start"]), 0, 1)
            cover = clamp(min(p / 0.2, (1 - p) / 0.2), 0, 1)     # moon slides over, then away
            self.light = self.light * (1 - cover) + 0.1 * cover
            env["eclipse"] = cover
        if "drought" in ev:
            env["food_rate"] *= 0.25
            env["food_target"] *= 0.4
            env["drought"] = True
        if "blizzard" in ev:
            env["cold"] *= 1.4
            env["food_rate"] *= 0.6
            env["visibility"] = 0.6
            env["wind"] = ev["blizzard"]["wind"]
            env["blizzard"] = True
        self.env = env

    def skip_time(self, frames):
        self.clock += frames
        self.update_environment()

    def next_season(self):
        span = DAY_FRAMES * DAYS_PER_SEASON
        self.skip_time(span - self.clock % span + self.time_of_day * DAY_FRAMES)

    def food_near(self, x, y, r):
        found = self.food.near(x, y, r) if self.food is not None else []
        if self.lures:
            r2 = r * r
            found.extend(l for l in self.lures if (l.x - x) ** 2 + (l.y - y) ** 2 <= r2)
        return found

    # ------------------------------------------------------------ terrain
    def set_obstacles(self, rects):
        """Install desktop windows as solid terrain (rects in overlay coordinates)."""
        self.obstacles = [tuple(float(v) for v in r) for r in rects]
        if self.food is not None:
            self.food.obstacles = self.obstacles
            for f in [f for f in self.food.items if any(point_in_rect(f.x, f.y, r) for r in self.obstacles)]:
                self.food.remove(f)

    def line_of_sight(self, ax, ay, bx, by):
        for rect in self.obstacles:
            if segment_hits_rect(ax, ay, bx, by, rect):
                return False
        for w in self.walls:
            if segments_cross(ax, ay, bx, by, w.x0, w.y0, w.x1, w.y1):
                return False
        return True

    # ------------------------------------------------------- world events
    EVENT_NAMES = {"meteor_shower": "Meteor shower", "plague": "Plague", "eclipse": "Eclipse",
                   "drought": "Drought", "blizzard": "Blizzard", "leviathan": "Leviathan"}

    def event_active(self, name):
        if name == "plague":
            return self.plague is not None
        if name == "leviathan":
            return self.leviathan is not None and self.leviathan.alive
        return name in self.active_events

    def trigger_event(self, name):
        """Start a world event now. Returns False if it can't happen at the moment."""
        if self.adhoc_mode or self.event_active(name):
            return False
        f = self.frame
        if name == "weather":   # seasonal pick: drought in the warm half, blizzard in the cold
            name = "drought" if self.season in ("spring", "summer") else "blizzard"
            if self.event_active(name):
                return False
        if name == "meteor_shower":
            self.active_events[name] = {"start": f, "end": f + 2400}
            self.log("Meteor shower! Rich food is falling from the sky - and so are rocks", "event")
        elif name == "plague":
            healthy = [o for o in self.orbs if o.alive and o.species == "orb" and not o.immune]
            if len(healthy) < 4:
                return False
            patients = random.sample(healthy, 2)
            for o in patients:
                o.infected = random.randint(5400, 8400)
            self.plague = {"start_day": self.day, "infected": 2, "died": 0, "recovered": 0}
            self.log(f"Plague! {patients[0].name} and {patients[1].name} have fallen sick. "
                     f"It spreads by contact", "event")
        elif name == "eclipse":
            self.active_events[name] = {"start": f, "end": f + 1800}
            self.log("Eclipse! Darkness at midday - the night hunters wake", "event")
        elif name == "drought":
            self.active_events[name] = {"start": f, "end": f + int(DAY_FRAMES * 1.5)}
            self.log("Drought: food is withering and barely growing back", "event")
        elif name == "blizzard":
            self.active_events[name] = {"start": f, "end": f + DAY_FRAMES,
                                        "wind": random.choice((-1, 1)) * 0.012}
            self.log("Blizzard! Biting cold, howling wind and almost no visibility", "event")
        elif name == "leviathan":
            lev = Leviathan(self)
            self.leviathan = lev
            self.orbs.append(lev)
            self.log(f"A leviathan approaches: {lev.name}. Everything scatters...", "event")
        else:
            return False
        self.counters["events"] += 1
        self.update_environment()
        return True

    def trigger_random_event(self):
        options = [("meteor_shower", 1.0), ("plague", 0.7), ("leviathan", 0.6),
                   ("drought" if self.season in ("spring", "summer") else "blizzard", 1.0)]
        if self.light > 0.85:
            options.append(("eclipse", 0.7))
        options = [(n, w) for n, w in options if not self.event_active(n)]
        if options:
            total = sum(w for _, w in options)
            r = random.uniform(0, total)
            for n, w in options:
                r -= w
                if r <= 0:
                    self.trigger_event(n)
                    break

    def update_events(self):
        if self.random_events and self.frame >= self.next_event_frame:
            self.next_event_frame = self.frame + int(DAY_FRAMES * random.uniform(1.5, 3.0))
            self.trigger_random_event()

        ends = {"meteor_shower": "The meteor shower is over",
                "eclipse": "The eclipse passes - daylight returns",
                "drought": "The drought breaks: rain at last",
                "blizzard": "The blizzard blows itself out"}
        for name, e in list(self.active_events.items()):
            if self.frame >= e["end"]:
                del self.active_events[name]
                self.log(ends.get(name, f"{name} is over"), "event")

        # Meteors
        if "meteor_shower" in self.active_events and random.random() < 1 / 35:
            tx, ty = self.random_position(40)
            dx, dy = normalize(random.choice((-1, 1)) * random.uniform(0.3, 0.8), 1.0)
            self.meteors.append({"x": tx - dx * 900, "y": ty - dy * 900, "tx": tx, "ty": ty,
                                 "vx": dx * 15, "vy": dy * 15, "trail": []})
        for m in list(self.meteors):
            m["trail"].append((m["x"], m["y"]))
            del m["trail"][:-14]
            m["x"] += m["vx"]
            m["y"] += m["vy"]
            if (m["x"] - m["tx"]) * m["vx"] + (m["y"] - m["ty"]) * m["vy"] >= 0:   # passed the target
                self.meteors.remove(m)
                self.meteor_impact(m["tx"], m["ty"])

        for b in self.bolts:
            b[1] -= 1
        self.bolts = [b for b in self.bolts if b[1] > 0]

        if self.walls:
            for w in self.walls:
                w.life -= 1
            self.walls = [w for w in self.walls if w.life > 0]

        # Plague: spreads between orbs that touch; ends when nobody is sick.
        if self.plague is not None and self.frame % 10 == 0:
            sick = [o for o in self.orbs if o.alive and o.infected]
            for o in sick:
                for n in self.neighbors(o, o.radius + 40):
                    if n.species == "orb" and not n.infected and not n.immune and \
                            math.hypot(n.x - o.x, n.y - o.y) < o.radius + n.radius + 15 and \
                            random.random() < 0.08 * (1 - 0.6 * n.wariness):
                        n.infected = random.randint(5400, 8400)
                        self.plague["infected"] += 1
                        if self.is_followed(n):
                            self.log(f"{n.name} caught the plague from {o.name}", "followed")
            if not sick:
                p = self.plague
                eaten = max(0, p["infected"] - p["died"] - p["recovered"])
                self.log(f"The plague is over: {p['infected']} fell sick - {p['died']} died of it, "
                         f"{p['recovered']} recovered (now immune)" +
                         (f", {eaten} were caught by predators while sick" if eaten else ""), "event")
                self.plague = None

    def meteor_impact(self, x, y):
        self.add_effect(x, y, (255, 150, 60), 90, 34)
        if self.scent is not None:
            self.scent.burst("danger", x, y, 6.0)
        self.add_effect(x, y, (255, 230, 160), 40, 18)
        if self.food is not None:
            self.food.burst(x, y, 10, spread=30, value_mult=1.8)
        for o in self.neighbors(SimpleNamespace(x=x, y=y), 130):
            d = math.hypot(o.x - x, o.y - y)
            if o.species == "leviathan":
                continue
            if d < 30 + o.radius:
                o.energy -= 30
                if o.energy <= 0:
                    o.die("meteor", self)
                    self.log(f"A meteor crushed {o.full_name}", "followed" if self.is_followed(o) else "danger")
                    continue
            dx, dy = normalize(o.x - x, o.y - y)
            push = 4.0 * (1 - d / 130)
            o.vx += dx * push
            o.vy += dy * push
            o.threat_memory = [x, y, 60]

    # ---------------------------------------------------------- god tools
    def lightning(self, x, y):
        """Strike a spot: kills what's right there, singes and scatters the rest."""
        pts = [(x + random.uniform(-40, 40), y - 900)]
        while pts[-1][1] < y - 30:
            px, py = pts[-1]
            pts.append((px + random.uniform(-35, 35) + (x - px) * 0.25, py + random.uniform(40, 90)))
        pts.append((x, y))
        self.bolts.append([pts, 18])
        if self.scent is not None:
            self.scent.burst("danger", x, y, 8.0)
        self.add_effect(x, y, (200, 220, 255), 80, 22)
        victims = []
        for o in self.neighbors(SimpleNamespace(x=x, y=y), 110):
            d = math.hypot(o.x - x, o.y - y)
            if o.species == "leviathan":
                o.energy -= 60
                if o.energy <= 0:
                    self.slay_leviathan(o, None)
                continue
            if d < o.radius + 22:
                o.die("struck", self)
                victims.append(o.name)
            else:
                o.energy -= 25 * (1 - d / 110)
                o.threat_memory = [x, y, 90]
                dx, dy = normalize(o.x - x, o.y - y)
                o.vx += dx * 3
                o.vy += dy * 3
        if victims:
            self.log(f"Lightning struck {', '.join(victims)}!", "danger")

    def add_wall(self, x0, y0, x1, y1):
        if math.hypot(x1 - x0, y1 - y0) < 15:
            return None
        wall = Wall(x0, y0, x1, y1)
        self.walls.append(wall)
        del self.walls[:-8]           # at most 8 walls at once
        if self.food is not None:
            for f in [f for f in self.food.items
                      if math.hypot(*(a - b for a, b in zip((f.x, f.y),
                                                            closest_on_segment(f.x, f.y, x0, y0, x1, y1)))) < 10]:
                self.food.remove(f)
        return wall

    def summon_predator(self, x, y):
        orb = IntelligentOrb(x, y, self.width, self.height)
        orb.aggression, orb.social_tendency, orb.wariness = 0.95, 0.2, 0.3
        orb.size_gene = orb.radius = 24.0
        orb.energy = 45.0                   # hungry
        orb.update_physique()
        orb.shape = orb.determine_shape()
        orb.base_color = orb.current_color = orb.generate_personality_color()
        self.register(orb, founder=True)
        self.orbs.append(orb)
        self.add_effect(x, y, (255, 60, 60), 60, 30)
        self.log(f"A hungry predator appears: {orb.full_name}", "danger")
        return orb

    def bless(self, x, y, r=100):
        """Heal, feed, rest and cure every orb nearby."""
        blessed = 0
        for o in self.neighbors(SimpleNamespace(x=x, y=y), r):
            if o.species == "leviathan":
                continue
            o.energy = min(MAX_ENERGY, o.energy + 40)
            o.stamina = 1.0
            o.exhausted = False
            if o.infected:
                o.infected = 0
                o.immune = True
                if self.plague is not None:
                    self.plague["recovered"] += 1
            blessed += 1
        self.add_effect(x, y, (255, 220, 120), r, 30)
        self.add_effect(x, y, (255, 245, 200), r * 0.5, 20)
        return blessed

    def drop_food(self, x, y, n=4):
        if self.food is not None:
            self.food.burst(x, y, n, spread=18)

    # ---------------------------------------------------- event callbacks
    def on_recover(self, orb):
        if self.plague is not None:
            self.plague["recovered"] += 1
        if self.is_followed(orb):
            self.log(f"{orb.name} recovered from the plague and is now immune", "followed")

    def leviathan_eat(self, lev, o):
        o.killer = lev
        o.die("eaten", self)
        lev.meals += 1
        self.counters["kills"] += 1
        self.counters["leviathan_kills"] += 1
        self.add_effect(o.x, o.y, (120, 170, 255), 50, 26)
        if lev.meals <= 2 or o.is_angler:
            self.log(f"{lev.name} swallowed {o.full_name} whole", "danger")

    def slay_leviathan(self, lev, hunter):
        lev.killer = hunter
        lev.die("slain", self)
        if self.food is not None:
            self.food.burst(lev.x, lev.y, 45, spread=70, value_mult=2.0)
        self.add_effect(lev.x, lev.y, (255, 220, 120), 160, 50)
        if hunter is not None and hunter.pack is not None:
            for m in hunter.pack.members:
                if m.alive:
                    m.survival_score += 60
                    m.hunting_success_count += 1
            self.log(f"LEGENDARY: {hunter.name}'s pack slew the leviathan {lev.name}! A feast for all",
                     "champion")
        else:
            self.log(f"The leviathan {lev.name} has fallen. A feast for all", "champion")

    def on_leviathan_leave(self, lev):
        self.log(f"{lev.name} sinks back into the deep after swallowing {lev.meals}", "event")

    def cover_point(self, orb, tx, ty, search=260):
        """A spot on the far side of a nearby window from the threat at (tx, ty)."""
        best = None
        best_d = search
        for rect in self.obstacles:
            x0, y0, x1, y1 = rect
            cx, cy = clamp(orb.x, x0, x1), clamp(orb.y, y0, y1)
            if math.hypot(cx - orb.x, cy - orb.y) > search:
                continue
            mx, my = (x0 + x1) / 2, (y0 + y1) / 2
            ux, uy = normalize(mx - tx, my - ty)
            # Project from the centre to just outside the rect, away from the threat.
            hx, hy = (x1 - x0) / 2, (y1 - y0) / 2
            scale = min(hx / abs(ux) if abs(ux) > 1e-6 else 1e9, hy / abs(uy) if abs(uy) > 1e-6 else 1e9)
            px = mx + ux * (scale + orb.radius + 18)
            py = my + uy * (scale + orb.radius + 18)
            if not (orb.radius < px < self.width - orb.radius and orb.radius < py < self.height - orb.radius):
                continue
            d = math.hypot(px - orb.x, py - orb.y)
            if d < best_d:
                best, best_d = (px, py), d
        return best

    def ambush_spot(self, angler):
        """Anglers set up near food patches or lurk just around window corners."""
        options = []
        off = angler.radius + 30
        for x0, y0, x1, y1 in self.obstacles:
            for cx, cy, sx, sy in ((x0, y0, -1, -1), (x1, y0, 1, -1), (x0, y1, -1, 1), (x1, y1, 1, 1)):
                px, py = cx + sx * off, cy + sy * off
                if off < px < self.width - off and off < py < self.height - off and \
                        not any(point_in_rect(px, py, r, angler.radius) for r in self.obstacles):
                    options.append((px, py))
        if self.food is not None:
            for p in self.food.patches:
                options.append((p[0] + random.uniform(-60, 60), p[1] + random.uniform(-60, 60)))
        if not options:
            return self.random_position()
        # Prefer spots that aren't on the other side of the screen.
        options.sort(key=lambda s: math.hypot(s[0] - angler.x, s[1] - angler.y))
        return random.choice(options[:4])

    def raise_alarm(self, caller, threat, tx, ty, level):
        if self.adhoc_mode:
            return
        self.alarms.append(Alarm(caller, threat, tx, ty, level))
        self.counters["alarms"] += 1
        if self.scent is not None:
            self.scent.burst("danger", caller.x, caller.y, 2.0)
        self.add_effect(caller.x, caller.y, (255, 205, 60), ALARM_RADIUS * 0.55, 26)

    def lay_scent(self, o):
        """Everyone leaves smells behind as they go."""
        sc = self.scent
        sp = o.species
        if sp == "leviathan":
            sc.deposit("danger", o.x, o.y, 1.0)
            return
        if sp in ("angler", "leech"):
            return     # ambushers and parasites stay scentless
        # Body scent: bigger orbs smell stronger; wary ones are more careful.
        sc.deposit("prey", o.x, o.y, 0.12 * (o.radius / 15.0) * (1 - 0.6 * o.wariness))
        if o.food_scent_timer > 0:
            o.food_scent_timer -= 1
            # Freshest near the food, so following it uphill leads back there. Social orbs share more.
            sc.deposit("food", o.x, o.y, 0.6 * (0.4 + 0.6 * o.social_tendency) * o.food_scent_timer / 600)
        if o.behavior_state == "fleeing":
            sc.deposit("danger", o.x, o.y, 0.25)

    def room_for_births(self, species=None):
        # The original orbs get extra headroom so the newer species can't crowd them out.
        limit = self.max_orbs * (1.3 if species == "orb" else 1.0)
        return len(self.orbs) + len(self.newborns) < limit

    # ------------------------------------------------------------- events
    def add_effect(self, x, y, color, max_radius=45, life=30):
        if not self.adhoc_mode and len(self.effects) < 80:
            self.effects.append(RingEffect(x, y, color, max_radius, life))

    def try_kill(self, hunter, prey):
        if not (hunter.alive and prey.alive):
            return False
        if math.hypot(prey.x - hunter.x, prey.y - hunter.y) > (hunter.radius + prey.radius) * 0.95:
            return False
        assisted = hunter.pack_assist(prey)
        if not hunter.can_eat(prey, assisted=assisted):
            return False

        if prey.species == "leviathan":
            # A pack wears a leviathan down bite by bite; each attacker paces its bites.
            if hunter.age < hunter.next_bite:
                return False
            hunter.next_bite = hunter.age + 40
            biters = sum(1 for m in hunter.pack.members
                         if m.alive and math.hypot(m.x - prey.x, m.y - prey.y) < prey.radius + m.radius + 40)
            prey.energy -= 10 + 4 * biters
            hunter.energy = min(MAX_ENERGY, hunter.energy + 6)
            hunter.reward(HUNT, 0.03)
            dx, dy = normalize(hunter.x - prey.x, hunter.y - prey.y)
            hunter.vx += dx * 3
            hunter.vy += dy * 3
            self.add_effect(hunter.x - dx * hunter.radius, hunter.y - dy * hunter.radius, (255, 90, 70), 24, 14)
            if prey.energy <= 0:
                self.slay_leviathan(prey, hunter)
                return True
            if not getattr(prey, "attack_logged", False):
                prey.attack_logged = True
                self.log(f"{hunter.name}'s pack is attacking the leviathan {prey.name}!", "danger")
            return False

        # Mobbing: a tight group of social prey can drive off a lone hunter.
        if not assisted and prey.social_tendency > 0.4:
            defenders = [n for n in self.neighbors(prey, 75)
                         if n is not hunter and n.social_tendency > 0.5 and
                         n.radius >= prey.radius * 0.8 and genetic_distance(n, prey) < 0.3 and
                         n.behavior_state != "fleeing"]
            if len(defenders) >= 2 and random.random() < 0.55:
                hunter.energy -= 6
                dx, dy = normalize(hunter.x - prey.x, hunter.y - prey.y)
                hunter.vx += dx * 4
                hunter.vy += dy * 4
                hunter.failed_targets[prey.id] = hunter.age + 900
                hunter.reward(HUNT, -0.08)
                hunter.target = None
                prey.stamina = min(1.0, prey.stamina + 0.3)
                self.counters["repelled"] += 1
                self.add_effect(prey.x, prey.y, (90, 170, 255), 40, 22)
                if self.is_followed(prey) or self.is_followed(hunter) or random.random() < 0.5:
                    self.log(f"House {prey.house} mobbed {hunter.name} and saved {prey.name}",
                             "followed" if self.is_followed(prey) or self.is_followed(hunter) else "life")
                return False

        gain = prey.energy * 0.55 + prey.radius * 2.0
        helpers = []
        if hunter.pack is not None:
            helpers = [m for m in hunter.pack.members if m is not hunter and m.alive and
                       math.hypot(m.x - prey.x, m.y - prey.y) < 130]
        if helpers:
            share = gain * 0.4 / len(helpers)
            for m in helpers:
                m.energy = min(MAX_ENERGY, m.energy + share)
                m.survival_score += 8
                m.reward(HUNT, 0.1)
            gain *= 0.6
        hunter.energy = min(MAX_ENERGY, hunter.energy + gain)
        hunter.grow(prey.radius * 0.06)
        hunter.hunting_success_count += 1
        hunter.fitness += 10
        hunter.survival_score += 20
        hunter.reward(HUNT, 0.22)
        hunter.target = None
        hunter.chase_timer = 0
        hunter.chase_best = 1e9
        if hunter.pack is not None and hunter.pack.target is prey:
            hunter.pack.target = None

        prey.killer = hunter
        prey.die("eaten", self)
        self.counters["kills"] += 1
        if prey.species == "mimic":
            # Experience teaches: after eating a mimic, disguises fool this hunter less often.
            hunter.mimic_wise += 1
            self.counters["mimics_eaten"] += 1
            for n in self.neighbors(hunter, 200):   # nearby relatives learn by watching
                if n.species == hunter.species and n.is_kin(hunter):
                    n.mimic_wise = max(n.mimic_wise, hunter.mimic_wise - 1)
        if hunter.house in self.houses:
            self.houses[hunter.house]["kills"] += 1
        if hunter.is_angler:
            self.counters["angler_kills"] += 1
            hunter.relocate_timer = max(hunter.relocate_timer, 2400)   # good spot - stay
            hunter.behavior_state = "lurking"
            hunter.strike_timer = 0
        self.add_effect(prey.x, prey.y, (255, 70, 60), 55, 28)
        if self.scent is not None:
            self.scent.burst("danger", prey.x, prey.y, 8.0)   # the smell of death lingers

        # Stories worth telling.
        k = hunter.hunting_success_count
        if self.is_followed(hunter):
            self.log(f"{hunter.name} caught {prey.name} (kill #{k})", "followed")
        elif hunter.is_angler:
            self.log(f"{prey.full_name} took the bait - eaten by {hunter.name} the angler",
                     "danger")
        elif len(helpers) >= 2:
            self.log(f"{hunter.name}'s pack of {len(helpers) + 1} brought down {prey.name}", "danger")
        elif assisted and prey.radius > hunter.radius:
            self.log(f"{hunter.name}'s pack took down {prey.name}, who was bigger than any of them", "danger")
        if k in (5, 10, 20, 30, 50, 75, 100) and not self.is_followed(hunter):
            self.log(f"{hunter.full_name} has made {k} kills", "champion")
        return True

    def reproduce(self, a, b):
        if not (a.willing_to_mate() and b.willing_to_mate()) or not self.room_for_births(a.species):
            return None
        child = a.mate_with(b)
        if child is None:
            return None
        if self.adhoc_mode:
            self.orbs.append(child)
        else:
            self.newborns.append(child)
            self.register(child)
        self.counters["births"] += 1
        self.add_effect(child.x, child.y, (255, 150, 220), 40, 30)
        self.announce_birth(child, a, b)
        return child

    def announce_birth(self, child, *parents):
        followed = next((p for p in parents if self.is_followed(p)), None)
        if followed is not None:
            verb = "split in two, making" if child.species == "splitter" else "had a child:"
            self.log(f"{followed.name} {verb} {child.name} ({len(followed.children)} so far)", "followed")
        if child.generation > self.max_generation:
            self.max_generation = child.generation
            if child.generation >= 3:
                self.log(f"First generation-{child.generation} orb born: {child.full_name}", "life")

    def add_newborn(self, child):
        if self.adhoc_mode:
            self.orbs.append(child)
        else:
            self.newborns.append(child)
            self.register(child)

    def split(self, o):
        """A splitter divides into two smaller copies of itself (with small mutations)."""
        child = IntelligentOrb(o.x, o.y, self.width, self.height, genome=o.genome(), species="splitter")
        child.brain = o.brain.copy().mutate(rate=0.06, sigma=0.2)
        child.mutate_traits(0.04)
        child.generation = o.generation + 1
        child.house = o.house
        child.parents = (o.name,)
        child.behavior_values = [0.5 * v for v in o.behavior_values]
        r = o.radius / math.sqrt(2)
        o.radius = child.radius = r
        o.energy = child.energy = o.energy / 2 - 2
        a = random.uniform(0, math.tau)
        c, s_ = math.cos(a), math.sin(a)
        o.x, o.y = o.x - c * r * 0.6, o.y - s_ * r * 0.6
        child.x, child.y = child.x + c * r * 0.6, child.y + s_ * r * 0.6
        o.vx, o.vy = o.vx - c * 1.5, o.vy - s_ * 1.5
        child.vx, child.vy = c * 1.5, s_ * 1.5
        for x in (o, child):
            x.update_physique()
        child.shape = child.determine_shape()
        child.base_color = child.current_color = child.generate_personality_color()
        child.reproductive_cooldown = o.reproductive_cooldown = 0
        o.children.append(child)
        o.offspring_count += 1
        o.survival_score += 10
        self.add_newborn(child)
        self.counters["births"] += 1
        self.counters["splits"] += 1
        self.add_effect(o.x + c * r * 0.6, o.y + s_ * r * 0.6, (120, 240, 200), 34, 24)
        self.announce_birth(child, o)
        return child

    def lay_egg(self, parent):
        child = orb_class(parent.species)(parent.x, parent.y, self.width, self.height,
                                          genome=parent.genome(), species=parent.species)
        child.brain = parent.brain.copy().mutate(rate=0.06, sigma=0.2)
        child.mutate_traits(0.04)
        child.radius = child.size_gene * 0.75
        child.update_physique()
        child.generation = parent.generation + 1
        child.house = parent.house
        child.parents = (parent.name,)
        child.energy = 40.0
        parent.energy -= 35
        parent.reproductive_cooldown = random.randint(1200, 1800)
        parent.children.append(child)
        parent.offspring_count += 1
        self.add_newborn(child)
        self.counters["births"] += 1
        self.add_effect(child.x, child.y, (255, 150, 220), 30, 24)
        self.announce_birth(child, parent)
        return child

    def bud_leech(self, leech):
        """A well-fed leech buds off a new one."""
        bud = Leech(leech.x, leech.y, self.width, self.height)
        bud.energy = leech.energy / 2
        leech.energy /= 2
        bud.generation = leech.generation + 1
        bud.parents = (leech.name,)
        a = random.uniform(0, math.tau)
        bud.vx, bud.vy = math.cos(a) * 3, math.sin(a) * 3
        bud.latch_cooldown = 120
        leech.children.append(bud)
        leech.offspring_count += 1
        self.add_newborn(bud)
        self.counters["births"] += 1
        self.counters["leech_buds"] += 1
        n = self.species_count("leech")
        if n >= 8 and self.day - getattr(self, "_leech_warned_day", -9) >= 2:
            self._leech_warned_day = self.day
            self.log(f"Leeches are multiplying - {n + 1} of them now", "danger")
        return bud

    # ------------------------------------------------------------ carrion
    def add_carrion(self, x, y, value, name, radius):
        if self.adhoc_mode or value <= 1:
            return
        self.carrion.append(Carrion(x, y, value, name, radius))
        del self.carrion[:-60]

    def carrion_near(self, x, y, r):
        r2 = r * r
        return [c for c in self.carrion if (c.x - x) ** 2 + (c.y - y) ** 2 <= r2]

    def remove_carrion(self, c):
        try:
            self.carrion.remove(c)
        except ValueError:
            pass

    def on_death(self, orb):
        self.counters["deaths"] += 1
        cause = orb.cause_of_death
        if cause == "starved":
            self.counters["starved"] += 1
        elif cause == "old age":
            self.counters["old_age"] += 1
        elif cause in ("plague", "struck", "meteor"):
            self.counters[cause] += 1
        if cause == "plague" and self.plague is not None:
            self.plague["died"] += 1
        # Bodies leave remains (predators leave scraps; a slain leviathan is a banquet).
        if orb.species == "leviathan":
            self.add_carrion(orb.x, orb.y, 160.0, orb.name, orb.radius)
        elif orb.species != "leech":
            self.add_carrion(orb.x, orb.y, orb.radius * (1.2 if cause == "eaten" else 2.5) + 4, orb.name, orb.radius)
        if self.adhoc_mode or orb.species == "leviathan":
            return
        self.tell_death(orb)
        score = orb.lifetime_fitness()
        hof = self.hall_of_fame
        if len(hof) < HALL_OF_FAME_SIZE or score > hof[-1]["score"]:
            hof.append({"score": round(float(score), 2), "genome": orb.genome()})
            hof.sort(key=lambda e: -e["score"])
            del hof[HALL_OF_FAME_SIZE:]

    def tell_death(self, orb):
        cause = orb.cause_of_death
        killer = getattr(orb, "killer", None)
        how = {"starved": "starved", "old age": "died of old age", "plague": "succumbed to the plague",
               "struck": "was struck by lightning", "meteor": "was crushed by a meteor"}.get(
            cause, f"was eaten by {killer.name}" if killer is not None else "died")
        kids = len(orb.children)
        legacy = f", leaving {kids} {'child' if kids == 1 else 'children'}" if kids else ""

        if self.is_followed(orb):
            days = orb.age / DAY_FRAMES
            self.log(f"{orb.name} {how} after {days:.1f} days{legacy}", "followed")
            heirs = [c for c in orb.children if c.alive]
            if heirs:
                heir = max(heirs, key=lambda c: c.lifetime_fitness())
                self.followed = heir
                self.log(f"Now following {heir.name}, {orb.name}'s child", "followed")
            else:
                self.followed = None
        elif orb is self.champion:
            self.log(f"Champion {orb.full_name} {how}{legacy}", "champion")
        elif cause == "old age" and (kids >= 3 or orb.hunting_success_count >= 5):
            self.log(f"{orb.full_name} {how} - a long life{legacy}", "life")
        if orb is self.champion:
            self.champion = None

        house = orb.house
        info = self.houses.get(house)
        if info and not info.get("extinct") and not self.house_alive(house, excluding=orb):
            info["extinct"] = True
            if info["members"] >= 5 or info["kills"] >= 5:
                lasted = max(0, self.day - info["founded_day"])
                span = ("less than a day" if lasted == 0 else
                        "1 day" if lasted == 1 else f"{lasted} days")
                self.log(f"House {house} has died out after {span} "
                         f"({info['members']} members, {info['kills']} kills)", "danger")

    def check_stories(self):
        """Periodic storytelling: new champions and changing seasons."""
        if self.season_index != self._last_season:
            self._last_season = self.season_index
            self.log({"spring": "Spring has come: food is plentiful and it's breeding season",
                      "summer": "Summer: long warm days, fireflies at night",
                      "autumn": "Autumn: food is thinning out but richer",
                      "winter": "Winter has arrived: food is scarce and the cold bites"}[self.season], "season")
        if self.frame % 600 == 0:
            alive = [o for o in self.orbs if o.alive and o.age > MATURITY_AGE]
            if alive:
                best = max(alive, key=lambda o: o.lifetime_fitness())
                cur = self.champion
                if best is not cur and (cur is None or not cur.alive or
                                        best.lifetime_fitness() > cur.lifetime_fitness() * 1.15):
                    self.champion = best
                    self.log(f"New champion: {best.full_name} ({best.hunting_success_count} kills, "
                             f"{len(best.children)} children)", "champion")

    # --------------------------------------------------------------- packs
    # ------------------------------------------------------------ territories
    @property
    def territories(self):
        return [p.territory for p in self.packs if p.territory is not None]

    def claim_territory(self, pack):
        lead = pack.leader
        radius = clamp(140 + 35 * len(pack.members), 140, 300)
        for other in self.territories:
            gap = math.hypot(other.x - lead.x, other.y - lead.y) - other.radius
            if gap < 110:
                return None                      # too close to someone else's land
            radius = min(radius, gap)            # don't overlap a neighbour
        terr = Territory(pack, lead.x, lead.y, radius, self.day)
        pack.territory = terr
        kind = "followed" if any(self.is_followed(m) for m in pack.members) else "info"
        self.log(f"House {lead.house} claims a territory - {lead.name}'s pack of {len(pack.members)} will defend it",
                 kind)
        return terr

    def release_territory(self, pack):
        t = pack.territory
        if t is not None:
            pack.territory = None
            if self.day - t.founded_day >= 1 or t.intrusions >= 3:
                self.log(f"House {t.house}'s territory lies abandoned (held {self.day - t.founded_day} days, "
                         f"{t.intrusions} intruders driven off)", "info")

    def on_intrusion(self, terr, intruder, defender):
        if self.is_followed(intruder) or self.is_followed(defender):
            self.log(f"{defender.name} charges at {intruder.name} for trespassing on House {terr.house}'s land",
                     "followed")
        # Two territorial packs at each other's throats: a border clash.
        if intruder.pack is not None and intruder.pack.territory is not None:
            key = tuple(sorted((terr.house or "", intruder.house or "")))
            last = getattr(self, "_clash_days", {})
            if self.day - last.get(key, -9) >= 1:
                last[key] = self.day
                self._clash_days = last
                self.counters["border_clashes"] += 1
                self.log(f"Border clash: House {terr.house} vs House {intruder.house}!", "danger")

    def on_intruder_driven_off(self, terr, intruder, defender):
        terr.intrusions += 1
        self.counters["intruders_driven_off"] += 1
        if self.is_followed(intruder) or self.is_followed(defender) or self.frame - terr.last_report > 3600:
            terr.last_report = self.frame
            self.log(f"House {terr.house} drove {intruder.name} out of their territory",
                     "followed" if self.is_followed(intruder) or self.is_followed(defender) else "info")

    def update_packs(self, alive):
        for pack in list(self.packs):
            pack.members = [m for m in pack.members if m.alive and m.pack is pack]
            if pack.leader not in pack.members:
                if len(pack.members) >= 2:
                    pack.leader = max(pack.members, key=lambda m: m.aggression * m.radius)
                    pack.members.remove(pack.leader)
                    pack.members.insert(0, pack.leader)
                else:
                    pack.disband()
            if pack.members:
                busy = pack.leader.behavior_state in ("hunting", "defending")
                pack.idle = 0 if busy else pack.idle + 20
                # A pack with a home stays together much longer between hunts.
                if pack.idle > (2400 if pack.territory else 300) or len(pack.members) < 2:
                    pack.disband()
            if not pack.members:
                self.release_territory(pack)
                self.packs.remove(pack)
            elif pack.territory is not None:
                # The territory slowly follows where the pack actually lives.
                t, lead = pack.territory, pack.leader
                if math.hypot(lead.x - t.x, lead.y - t.y) < t.radius * 2:
                    t.x += (lead.x - t.x) * 0.02
                    t.y += (lead.y - t.y) * 0.02
                t.radius = clamp(140 + 35 * len(pack.members), 140, 300)
                t.house = lead.house
            elif len(pack.members) >= 3 or (len(pack.members) >= 2 and self.frame - pack.formed > 900):
                self.claim_territory(pack)

        # Form new packs around ambitious, social hunters.
        for o in sorted(alive, key=lambda o: -(o.aggression * o.radius)):
            if o.pack is not None or o.behavior_state != "hunting" or o.social_tendency < 0.4 or o.species != "orb":
                continue
            recruits = [n for n in self.neighbors(o, 180)
                        if n.pack is None and n.species == "orb" and n.social_tendency > 0.35 and n.aggression > 0.3 and
                        n.energy > 30 and n.behavior_state in ("hunting", "exploring", "resting") and
                        genetic_distance(o, n) < 0.45 and not o.can_eat(n) and not n.can_eat(o)]
            if not recruits:
                continue
            recruits.sort(key=lambda n: math.hypot(n.x - o.x, n.y - o.y))
            pack = Pack(o)
            pack.formed = self.frame
            o.pack = pack
            for n in recruits[:3]:
                pack.add(n)
            self.packs.append(pack)
            if len(pack.members) >= 3 or any(self.is_followed(m) for m in pack.members):
                kind = "followed" if any(self.is_followed(m) for m in pack.members) else "info"
                self.log(f"{o.full_name} formed a hunting pack of {len(pack.members)}", kind)

        # Lone social hunters can join an existing pack nearby.
        for pack in self.packs:
            if len(pack.members) >= 5:
                continue
            for n in self.neighbors(pack.leader, 140):
                if (n.pack is None and n.species == "orb" and n.behavior_state == "hunting" and n.social_tendency > 0.45 and
                        genetic_distance(pack.leader, n) < 0.45 and
                        not pack.leader.can_eat(n) and not n.can_eat(pack.leader)):
                    pack.add(n)
                    if len(pack.members) >= 5:
                        break

    # ---------------------------------------------------------------- step
    def step(self):
        self.frame += 1
        self.clock += 1
        self.update_environment()
        if self.food is not None:
            self.food.update(self.env)
        if self.scent is not None:
            self.scent.update(self.frame)

        alive = self.alive_orbs()
        self.orb_hash.rebuild(alive)
        self._grid_frame = self.frame
        if self.frame % 20 == 0:
            self.update_packs(alive)
        self.lures = [o.lure for o in alive if o.is_angler and o.lure is not None]
        self.alarms = [a for a in self.alarms if a.update()]

        for orb in list(self.orbs):
            orb.update(self.cursor, self.orbs, 1.0, world=self)
            if orb.alive and self.scent is not None:
                self.lay_scent(orb)

        if self.newborns:
            self.orbs.extend(self.newborns)
            self.newborns.clear()
        self.orbs = [o for o in self.orbs if o.alive or o.is_exploding]

        # Keep the ecosystem going: minimum population plus occasional immigrants.
        n_alive = sum(1 for o in self.orbs if o.alive)
        if n_alive < self.min_orbs and self.frame % 20 == 0:
            self.spawn_orb()
        elif n_alive < self.max_orbs * 0.6 and random.random() < 0.002:
            self.spawn_orb()
        # Anglers are rare; if they die out, a new one eventually drifts in.
        if self.frame % 900 == 0 and self.angler_count() < max(2, n_alive // 12) and n_alive < self.max_orbs:
            self.spawn_orb(species="angler")
        # The newer species also trickle back in if they vanish.
        if self.frame % 1200 == 600:
            for species in ("splitter", "scavenger", "leech", "mimic"):
                n = self.species_count(species)
                if n == 0 or (n < 2 and n_alive < self.max_orbs):
                    self.spawn_orb(species=species)
        if self.carrion:
            for c in self.carrion:
                c.age += 1
            self.carrion = [c for c in self.carrion if c.age < c.max_age and c.value > 0.5]

        self.effects = [e for e in self.effects if e.update()]
        self.update_events()
        self.check_stories()

        if self.frame % HISTORY_EVERY == 0:
            self.record_history()

        if self.genepool_file and self.frame % 3600 == 0:
            self.save_genepool()

    def record_history(self):
        orbs = self.alive_orbs()
        regular = [o for o in orbs if o.species == "orb"]
        behaviors = {}
        for o in orbs:
            behaviors[o.behavior_state] = behaviors.get(o.behavior_state, 0) + 1

        def avg(attr):
            return sum(getattr(o, attr) for o in regular) / len(regular) if regular else 0.0

        self.history.append({
            "frame": self.frame,
            "light": self.light,
            "season": self.season_index,
            "orbs": len(regular),
            "anglers": sum(1 for o in orbs if o.is_angler),
            "splitters": sum(1 for o in orbs if o.species == "splitter"),
            "scavengers": sum(1 for o in orbs if o.species == "scavenger"),
            "leeches": sum(1 for o in orbs if o.species == "leech"),
            "mimics": sum(1 for o in orbs if o.species == "mimic"),
            "disguise": (sum(o.disguise for o in orbs if o.species == "mimic") /
                         max(1, sum(1 for o in orbs if o.species == "mimic"))),
            "territories": len(self.territories),
            "food": len(self.food.items) if self.food else 0,
            "behaviors": behaviors,
            "aggression": avg("aggression"),
            "curiosity": avg("curiosity"),
            "social": avg("social_tendency"),
            "wariness": avg("wariness"),
            "night_vision": avg("night_vision"),
            "energy": sum(o.energy for o in orbs) / len(orbs) if orbs else 0.0,
            "generation": sum(o.generation for o in orbs) / len(orbs) if orbs else 0.0,
            "counters": dict(self.counters),
        })

    # ---------------------------------------------------------- gene pool
    def load_genepool(self):
        if not self.genepool_file or not os.path.exists(self.genepool_file):
            return 0
        try:
            with open(self.genepool_file, "r") as f:
                data = json.load(f)
            entries = []
            for e in data.get("hall_of_fame", []):
                OrbBrain.from_dict(e["genome"]["brain"])  # validate
                entries.append({"score": float(e["score"]), "genome": e["genome"]})
            # Old champions slowly lose their lustre so new ones can replace them.
            for e in entries:
                e["score"] *= 0.8
            self.hall_of_fame = sorted(entries, key=lambda e: -e["score"])[:HALL_OF_FAME_SIZE]
            return len(self.hall_of_fame)
        except Exception as exc:
            print(f"Could not load gene pool ({exc}); starting fresh.")
            self.hall_of_fame = []
            return 0

    def save_genepool(self):
        if not self.genepool_file:
            return
        # Living orbs count too - the current champions should be preserved.
        pool = list(self.hall_of_fame)
        for o in self.orbs:
            if o.alive and o.age > MATURITY_AGE and o.species != "leviathan":
                pool.append({"score": round(float(o.lifetime_fitness()), 2), "genome": o.genome()})
        pool.sort(key=lambda e: -e["score"])
        data = {"version": 1, "hall_of_fame": pool[:HALL_OF_FAME_SIZE]}
        try:
            tmp = self.genepool_file + ".tmp"
            with open(tmp, "w") as f:
                json.dump(data, f)
            os.replace(tmp, self.genepool_file)
        except Exception:
            pass

    # --------------------------------------------------------- statistics
    def statistics(self):
        orbs = self.alive_orbs()
        count = len(orbs)
        if count == 0:
            return {"total_orbs": 0, "behaviors": {}, "timestamp": self.frame,
                    "simulation_time": round(self.frame / 60, 1)}

        def avg(vals):
            vals = list(vals)
            return float(sum(vals) / len(vals)) if vals else 0.0

        behaviors = {}
        for o in orbs:
            behaviors[o.behavior_state] = behaviors.get(o.behavior_state, 0) + 1
        energies = [float(o.energy) for o in orbs]
        affinities = [float(o.cursor_affinity) for o in orbs]
        generations = [int(o.generation) for o in orbs]
        sizes = [float(o.radius) for o in orbs]
        ages = [int(o.age) for o in orbs]
        cx, cy = self.cursor
        cursor_interactions = sum(1 for o in orbs if math.hypot(o.x - cx, o.y - cy) < 200)
        avg_aggr, avg_cur, avg_soc = (avg(o.aggression for o in orbs), avg(o.curiosity for o in orbs),
                                      avg(o.social_tendency for o in orbs))

        patterns = []
        if behaviors.get("hunting", 0) > count * 0.3:
            patterns.append("High aggression phase detected")
        if behaviors.get("mating", 0) > count * 0.2:
            patterns.append("Active reproduction period")
        if behaviors.get("fleeing", 0) > count * 0.4:
            patterns.append("Population under stress")
        if cursor_interactions > count * 0.3:
            patterns.append("Strong cursor interaction activity")
        if len(self.packs) >= 2:
            patterns.append(f"{len(self.packs)} hunting packs roaming")
        if behaviors.get("foraging", 0) > count * 0.5:
            patterns.append("Food scarcity - widespread foraging")

        dominant = []
        if avg_aggr > 0.6:
            dominant.append("High Aggression")
        if avg_cur > 0.7:
            dominant.append("High Curiosity")
        if avg_soc > 0.6:
            dominant.append("Social Behavior")

        top = sorted(orbs, key=lambda o: -o.lifetime_fitness())[:5]
        total_hunts = sum(o.hunting_success_count for o in orbs)
        return {
            "total_orbs": count,
            "behaviors": behaviors,
            "avg_energy": round(avg(energies), 1),
            "avg_age": int(avg(ages)),
            "avg_survival_score": round(avg(o.survival_score for o in orbs), 1),
            "avg_aggression": round(avg_aggr, 3),
            "avg_curiosity": round(avg_cur, 3),
            "avg_social": round(avg_soc, 3),
            "avg_cursor_affinity": round(avg(affinities), 3),
            "energy_distribution": {
                "low": sum(1 for e in energies if e < 30),
                "medium": sum(1 for e in energies if 30 <= e < 70),
                "high": sum(1 for e in energies if e >= 70),
            },
            "cursor_affinity_distribution": {
                "very_afraid": sum(1 for a in affinities if a < -0.5),
                "afraid": sum(1 for a in affinities if -0.5 <= a < -0.2),
                "neutral": sum(1 for a in affinities if -0.2 <= a <= 0.2),
                "attracted": sum(1 for a in affinities if 0.2 < a <= 0.5),
                "very_attracted": sum(1 for a in affinities if a > 0.5),
            },
            "generation_stats": {
                "min": min(generations), "max": max(generations),
                "avg": round(avg(generations), 2), "diversity": len(set(generations)),
            },
            "population_health": {
                "overall_score": round(avg(o.survival_score for o in orbs), 1),
                "energy_stability": round(1.0 - (max(energies) - min(energies)) / 100, 3),
                "age_diversity": len({a // 100 for a in ages}),
                "size_diversity": len({int(s) for s in sizes}),
            },
            "behavioral_patterns": patterns,
            "trait_analysis": {"dominant_traits": dominant, "emerging_traits": [], "declining_traits": []},
            "active_interactions": {
                "cursor_interactions": cursor_interactions,
                "interaction_percentage": round(cursor_interactions / count * 100, 1),
            },
            "total_hunting_success": int(total_hunts),
            "avg_hunting_success": round(total_hunts / count, 2),
            "reproduction_capable": sum(1 for o in orbs if o.reproductive_cooldown == 0),
            "size_stats": {
                "min_size": round(min(sizes), 1), "max_size": round(max(sizes), 1),
                "avg_size": round(avg(sizes), 1),
            },
            "food_count": len(self.food.items) if self.food else 0,
            "packs": len(self.packs),
            "anglers": sum(1 for o in orbs if o.is_angler),
            "avg_wariness": round(avg(o.wariness for o in orbs if not o.is_angler), 3),
            "windows_as_terrain": len(self.obstacles),
            "season": self.season,
            "day": self.day,
            "time_of_day": self.clock_string(),
            "light": round(self.light, 2),
            "avg_night_vision": round(avg(o.night_vision for o in orbs), 3),
            "nocturnal": sum(1 for o in orbs if o.nocturnal),
            "lifetime_events": dict(self.counters),
            "gene_pool_size": len(self.hall_of_fame),
            "best_ever_fitness": self.hall_of_fame[0]["score"] if self.hall_of_fame else 0,
            "top_orbs": [{"id": o.id, "generation": o.generation, "kills": o.hunting_success_count,
                          "offspring": o.offspring_count, "fitness": round(o.lifetime_fitness(), 1),
                          "behavior": o.behavior_state} for o in top],
            "timestamp": self.frame,
            "simulation_time": round(self.frame / 60, 1),
        }

    def export_statistics(self, path=STATS_FILE):
        """Write stats atomically so the monitor never reads a half-written file."""
        try:
            tmp = path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(self.statistics(), f, indent=2)
            os.replace(tmp, path)
        except Exception:
            pass


# ----------------------------------------------------------------------------
# Desktop windows as terrain (Windows only, via the Win32 API)
# ----------------------------------------------------------------------------

class DesktopTerrain:
    """Finds the visible top-level windows on the desktop so orbs can treat them as rocks."""

    IGNORED_CLASSES = {"Progman", "WorkerW", "Shell_TrayWnd", "Shell_SecondaryTrayWnd",
                       "NotifyIconOverflowWindow", "Windows.UI.Core.CoreWindow"}

    def __init__(self):
        self.available = False
        if sys.platform != "win32":
            return
        try:
            import ctypes
            from ctypes import wintypes
            self.ctypes = ctypes
            self.wintypes = wintypes
            self.user32 = ctypes.windll.user32
            self.dwm = ctypes.windll.dwmapi
            self.WNDENUMPROC = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
            self.user32.GetWindowLongW.restype = ctypes.c_long
            self.available = True
        except Exception:
            self.available = False

    def scan(self, origin_x, origin_y, width, height, scale=1.0, max_coverage=0.55):
        """Return window rects in overlay coordinates (largest windows dropped if they'd fill the screen)."""
        if not self.available:
            return []
        ctypes, wintypes, user32 = self.ctypes, self.wintypes, self.user32
        own_pid = os.getpid()
        GWL_EXSTYLE = -20
        WS_EX_TOOLWINDOW = 0x00000080
        WS_EX_NOACTIVATE = 0x08000000
        DWMWA_EXTENDED_FRAME_BOUNDS = 9
        DWMWA_CLOAKED = 14
        rects = []
        class_buf = ctypes.create_unicode_buffer(256)

        def callback(hwnd, _lparam):
            try:
                if not user32.IsWindowVisible(hwnd) or user32.IsIconic(hwnd) or user32.IsZoomed(hwnd):
                    return True
                ex = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
                if ex & WS_EX_TOOLWINDOW or ex & WS_EX_NOACTIVATE:
                    return True
                if user32.GetWindowTextLengthW(hwnd) == 0:
                    return True
                pid = wintypes.DWORD()
                user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
                if pid.value == own_pid:
                    return True
                user32.GetClassNameW(hwnd, class_buf, 256)
                if class_buf.value in self.IGNORED_CLASSES:
                    return True
                cloaked = ctypes.c_int(0)
                self.dwm.DwmGetWindowAttribute(hwnd, DWMWA_CLOAKED, ctypes.byref(cloaked), ctypes.sizeof(cloaked))
                if cloaked.value:
                    return True
                r = wintypes.RECT()
                if self.dwm.DwmGetWindowAttribute(hwnd, DWMWA_EXTENDED_FRAME_BOUNDS,
                                                  ctypes.byref(r), ctypes.sizeof(r)) != 0:
                    user32.GetWindowRect(hwnd, ctypes.byref(r))
                x0 = r.left * scale - origin_x
                y0 = r.top * scale - origin_y
                x1 = r.right * scale - origin_x
                y1 = r.bottom * scale - origin_y
                x0, x1 = max(0.0, x0), min(float(width), x1)
                y0, y1 = max(0.0, y0), min(float(height), y1)
                if x1 - x0 >= 80 and y1 - y0 >= 50:
                    rects.append((x0, y0, x1, y1))
            except Exception:
                pass
            return True

        try:
            user32.EnumWindows(self.WNDENUMPROC(callback), 0)
        except Exception:
            return []

        # Keep plenty of open space: drop the biggest windows if they'd cover too much.
        rects.sort(key=lambda r: (r[2] - r[0]) * (r[3] - r[1]))
        budget = width * height * max_coverage
        kept, used = [], 0.0
        for r in rects:
            area = (r[2] - r[0]) * (r[3] - r[1])
            if used + area > budget:
                continue
            kept.append(r)
            used += area
        return kept[:40]


# ----------------------------------------------------------------------------
# Desktop overlay (PyQt5)
# ----------------------------------------------------------------------------

if HAVE_QT:

    BEHAVIOR_RING = {
        "hunting": (255, 40, 40, 210),
        "fleeing": (60, 120, 255, 190),
        "mating": (255, 170, 210, 210),
        "foraging": (90, 230, 120, 160),
        "defending": (255, 160, 40, 230),
    }

    def qcolor(c, a=255):
        return QtGui.QColor(int(c[0]), int(c[1]), int(c[2]), int(clamp(a, 0, 255)))

    class WeatherParticle:
        __slots__ = ("kind", "x", "y", "vx", "vy", "size", "phase", "life", "color")

    class Weather:
        """Purely decorative seasonal ambience: snow, falling leaves, petals, fireflies."""

        KIND = {"winter": "snow", "autumn": "leaf", "spring": "petal", "summer": "firefly"}
        LEAF_COLORS = ((230, 120, 40), (200, 80, 30), (240, 170, 60), (160, 90, 40))

        def __init__(self, width, height):
            self.width, self.height = width, height
            self.cap = int(clamp(width * height / 25000, 40, 170))
            self.particles = []

        def target_count(self, kind, world):
            night = 1.0 - world.light
            share = {"snow": 1.0, "leaf": 0.3, "petal": 0.2,
                     "firefly": 0.45 * clamp((night - 0.4) / 0.4, 0, 1)}[kind]
            # Ease in during the first part of the season.
            ramp = clamp(world.season_progress / 0.1, 0, 1)
            return int(self.cap * share * ramp)

        def spawn(self, kind):
            p = WeatherParticle()
            p.kind = kind
            p.phase = random.uniform(0, math.tau)
            p.life = 1e9
            if kind == "firefly":
                p.x, p.y = random.uniform(0, self.width), random.uniform(0, self.height)
                p.vx, p.vy = random.uniform(-0.4, 0.4), random.uniform(-0.4, 0.4)
                p.size = random.uniform(1.5, 2.5)
                p.life = random.randint(600, 1800)
                p.color = (220, 255, 120)
            else:
                p.x, p.y = random.uniform(-20, self.width + 20), -10
                if kind == "snow":
                    p.vx, p.vy = random.uniform(-0.2, 0.2), random.uniform(0.5, 1.4)
                    p.size = random.uniform(1.2, 3.0)
                    p.color = (245, 250, 255)
                elif kind == "leaf":
                    p.vx, p.vy = random.uniform(-0.5, 0.5), random.uniform(0.6, 1.2)
                    p.size = random.uniform(3.0, 5.5)
                    p.color = random.choice(self.LEAF_COLORS)
                else:  # petal
                    p.vx, p.vy = random.uniform(-0.3, 0.3), random.uniform(0.3, 0.7)
                    p.size = random.uniform(2.0, 3.5)
                    p.color = (255, random.randint(170, 210), random.randint(200, 230))
            self.particles.append(p)

        def update(self, world):
            blizzard = world.env.get("blizzard", False)
            kind = "snow" if blizzard else self.KIND[world.season]
            want = int(self.cap * 2.2) if blizzard else self.target_count(kind, world)
            have = sum(1 for p in self.particles if p.kind == kind)
            for _ in range(4 if blizzard else 1):
                if have < want and random.random() < 0.35:
                    self.spawn(kind)
                    have += 1
            wind = world.env.get("wind", 0.0) * 300      # visible gusts
            t = world.frame * 0.03
            alive = []
            for p in self.particles:
                if p.kind == "firefly":
                    p.vx = clamp(p.vx + random.uniform(-0.05, 0.05), -0.6, 0.6)
                    p.vy = clamp(p.vy + random.uniform(-0.05, 0.05), -0.6, 0.6)
                    p.x += p.vx
                    p.y += p.vy
                    p.life -= 1
                    # Fireflies leave at dawn or once their season ends.
                    if (kind != "firefly" or want < have) and random.random() < 0.01:
                        p.life = min(p.life, 40)
                    if p.life > 0 and -20 < p.x < self.width + 20 and -20 < p.y < self.height + 20:
                        alive.append(p)
                    continue
                sway = 0.3 if p.kind == "snow" else 1.1
                p.x += p.vx + math.sin(t + p.phase) * sway + wind
                p.y += p.vy * (1.6 if wind else 1.0)
                if p.y < self.height + 10 and -40 < p.x < self.width + 40:
                    alive.append(p)
                elif wind and p.y < self.height + 10:      # blown off one side, re-enter the other
                    p.x = self.width + 30 if p.x < 0 else -30
                    alive.append(p)
            self.particles = alive

        def draw(self, painter, frame):
            painter.setPen(QtCore.Qt.NoPen)
            t = frame * 0.03
            for p in self.particles:
                if p.kind == "firefly":
                    blink = max(0.0, math.sin(t * 2.5 + p.phase))
                    fade = min(1.0, p.life / 40.0)
                    a = blink * fade
                    if a < 0.05:
                        continue
                    painter.setBrush(qcolor(p.color, 70 * a))
                    painter.drawEllipse(QtCore.QPointF(p.x, p.y), p.size * 4, p.size * 4)
                    painter.setBrush(qcolor(p.color, 240 * a))
                    painter.drawEllipse(QtCore.QPointF(p.x, p.y), p.size, p.size)
                elif p.kind == "snow":
                    painter.setBrush(qcolor(p.color, 190))
                    painter.drawEllipse(QtCore.QPointF(p.x, p.y), p.size, p.size)
                else:
                    # Tumbling leaf / petal: an ellipse that flips as it falls.
                    flip = abs(math.sin(t * 1.5 + p.phase))
                    painter.setBrush(qcolor(p.color, 200))
                    painter.drawEllipse(QtCore.QPointF(p.x, p.y), p.size, p.size * (0.3 + 0.7 * flip))

    class StatsPanel:
        """Dashboard of the ecosystem: stat tiles, time-series charts, histograms and a
        leaderboard. Rendered to an image a couple of times per second (not every frame)."""

        W = 680
        PAD = 16
        SURFACE = (21, 22, 26)
        TILE = (30, 32, 38)
        INK = (240, 239, 236)
        INK2 = (195, 194, 183)
        MUTED = (137, 135, 129)
        GRID = (44, 44, 42)
        AXIS = (56, 56, 53)
        # Categorical slots in fixed order (validated against SURFACE, dark mode).
        SERIES = [(0x39, 0x87, 0xe5), (0xd9, 0x59, 0x26), (0x19, 0x9e, 0x70), (0xc9, 0x85, 0x00),
                  (0xd5, 0x51, 0x81), (0x0c, 0xa3, 0x0c), (0x90, 0x85, 0xe9), (0xe6, 0x67, 0x67)]
        BEHAVIOR_ORDER = ("exploring", "hunting", "foraging", "resting", "mating", "fleeing", "lurking")

        def __init__(self):
            self.f_title = QtGui.QFont("Segoe UI", 12, QtGui.QFont.Bold)
            self.f_head = QtGui.QFont("Segoe UI", 9, QtGui.QFont.Bold)
            self.f_text = QtGui.QFont("Segoe UI", 9)
            self.f_small = QtGui.QFont("Segoe UI", 8)
            self.f_big = QtGui.QFont("Segoe UI", 16, QtGui.QFont.Bold)

        # ------------------------------------------------------------ helpers
        def text(self, p, x, y, s, color, font, align="left"):
            p.setFont(font)
            p.setPen(qcolor(color))
            fm = QtGui.QFontMetrics(font)
            w = fm.horizontalAdvance(s)
            if align == "right":
                x -= w
            elif align == "center":
                x -= w / 2
            p.drawText(QtCore.QPointF(x, y), s)
            return w

        @staticmethod
        def nice_max(v):
            if v <= 0:
                return 1.0
            mag = 10 ** math.floor(math.log10(v))
            for m in (1, 2, 2.5, 5, 10):
                if v <= m * mag:
                    return m * mag
            return 10 * mag

        @staticmethod
        def fmt(v):
            if abs(v) >= 1000:
                return f"{v / 1000:.1f}k"
            if abs(v) >= 10 or float(v).is_integer():
                return f"{v:.0f}"
            return f"{v:.2f}"

        @staticmethod
        def downsample(values, n):
            """Average values into at most n buckets so lines stay crisp."""
            if len(values) <= n:
                return list(values)
            out = []
            step = len(values) / n
            for i in range(n):
                a, b = int(i * step), max(int((i + 1) * step), int(i * step) + 1)
                chunk = values[a:b]
                out.append(sum(chunk) / len(chunk))
            return out

        def bar(self, p, x, y, w, h, color, horizontal=False):
            """Bar with 4px rounded data-end, square at the baseline."""
            if w <= 0 or h <= 0:
                return
            r = min(4.0, (h if horizontal else w) / 2, (w if horizontal else h))
            path = QtGui.QPainterPath()
            if horizontal:   # baseline on the left
                path.moveTo(x, y)
                path.lineTo(x + w - r, y)
                path.quadTo(x + w, y, x + w, y + r)
                path.lineTo(x + w, y + h - r)
                path.quadTo(x + w, y + h, x + w - r, y + h)
                path.lineTo(x, y + h)
            else:            # baseline at the bottom
                path.moveTo(x, y + h)
                path.lineTo(x, y + r)
                path.quadTo(x, y, x + r, y)
                path.lineTo(x + w - r, y)
                path.quadTo(x + w, y, x + w, y + r)
                path.lineTo(x + w, y + h)
            path.closeSubpath()
            p.setPen(QtCore.Qt.NoPen)
            p.setBrush(qcolor(color))
            p.drawPath(path)

        def chart_header(self, p, x, y, w, title, legend=None):
            """Title, then (for >= 2 series) a legend row. Returns the plot's top y."""
            self.text(p, x, y + 12, title, self.INK, self.f_head)
            y += 18
            if legend:
                fm = QtGui.QFontMetrics(self.f_small)
                lx = x
                for name, color in legend:
                    item_w = 11 + fm.horizontalAdvance(name) + 12
                    if lx > x and lx + item_w > x + w:   # wrap instead of running into neighbours
                        lx = x
                        y += 16
                    p.setPen(QtCore.Qt.NoPen)
                    p.setBrush(qcolor(color))
                    p.drawEllipse(QtCore.QPointF(lx + 4, y + 7), 4, 4)
                    self.text(p, lx + 11, y + 11, name, self.INK2, self.f_small)
                    lx += item_w
                y += 18
            return y

        def time_axis(self, p, rect, hist, season_labels=True):
            """Night shading, season boundaries and the time labels under a series plot."""
            x, y, w, h = rect
            n = len(hist)
            if n < 2:
                return
            # Shaded bands = night.
            p.setPen(QtCore.Qt.NoPen)
            p.setBrush(QtGui.QColor(120, 140, 255, 18))
            start = None
            for i, s in enumerate(hist):
                dark = s["light"] < 0.4
                if dark and start is None:
                    start = i
                if (not dark or i == n - 1) and start is not None:
                    end = i
                    x0 = x + w * start / (n - 1)
                    x1 = x + w * end / (n - 1)
                    p.drawRect(QtCore.QRectF(x0, y, max(1.0, x1 - x0), h))
                    start = None
            # Season boundaries.
            for i in range(1, n):
                if hist[i]["season"] != hist[i - 1]["season"]:
                    sx = x + w * i / (n - 1)
                    p.setPen(QtGui.QPen(qcolor(self.AXIS), 1, QtCore.Qt.DashLine))
                    p.drawLine(QtCore.QPointF(sx, y), QtCore.QPointF(sx, y + h))
                    if season_labels:
                        self.text(p, sx + 3, y + 10, SEASONS[hist[i]["season"]], self.MUTED, self.f_small)
            minutes = (hist[-1]["frame"] - hist[0]["frame"]) / 3600.0
            label = f"-{minutes:.0f} min" if minutes >= 1 else f"-{minutes * 60:.0f} s"
            self.text(p, x, y + h + 12, label, self.MUTED, self.f_small)
            self.text(p, x + w, y + h + 12, "now", self.MUTED, self.f_small, "right")

        def line_chart(self, p, x, y, w, plot_h, title, hist, series, ymax=None, direct_labels=True):
            legend = [(name, c) for name, _, c in series] if len(series) >= 2 else None
            top = self.chart_header(p, x, y, w, title, legend)
            left, right = 30, (64 if direct_labels else 6)
            px, pw = x + left, w - left - right
            rect = (px, top, pw, plot_h)
            vmax = ymax if ymax is not None else self.nice_max(
                max((max(v) for _, v, _ in series if v), default=1) * 1.1)
            # Recessive grid + labels.
            for frac in (0.0, 0.5, 1.0):
                gy = top + plot_h * (1 - frac)
                p.setPen(QtGui.QPen(qcolor(self.AXIS if frac == 0 else self.GRID), 1))
                p.drawLine(QtCore.QPointF(px, gy), QtCore.QPointF(px + pw, gy))
                self.text(p, px - 5, gy + 4, self.fmt(vmax * frac), self.MUTED, self.f_small, "right")
            self.time_axis(p, rect, hist)
            # Lines.
            label_slots = []
            for name, values, color in series:
                if len(values) < 2:
                    continue
                vals = self.downsample(values, int(pw / 2))
                m = len(vals)
                poly = QtGui.QPolygonF([QtCore.QPointF(px + pw * i / (m - 1),
                                                       top + plot_h * (1 - clamp(v / vmax, 0, 1)))
                                        for i, v in enumerate(vals)])
                pen = QtGui.QPen(qcolor(color), 2)
                pen.setJoinStyle(QtCore.Qt.RoundJoin)
                pen.setCapStyle(QtCore.Qt.RoundCap)
                p.setPen(pen)
                p.setBrush(QtCore.Qt.NoBrush)
                p.drawPolyline(poly)
                end = poly[poly.size() - 1]
                # End marker with a surface ring.
                p.setPen(QtGui.QPen(qcolor(self.SURFACE), 2))
                p.setBrush(qcolor(color))
                p.drawEllipse(end, 4, 4)
                label_slots.append([end.y(), name, values[-1]])
            if direct_labels:
                label_slots.sort()
                last_y = -1e9
                for slot in label_slots:   # nudge apart so labels never collide
                    slot[0] = max(slot[0], last_y + 12)
                    last_y = slot[0]
                for ly, name, v in label_slots:
                    self.text(p, px + pw + 8, ly + 4, f"{self.fmt(v)} {name}", self.INK2, self.f_small)
            return top + plot_h + 16

        def stacked_share(self, p, x, y, w, plot_h, title, hist):
            layers = [(b, self.SERIES[i]) for i, b in enumerate(self.BEHAVIOR_ORDER)]
            top = self.chart_header(p, x, y, w, title, layers)
            left = 30
            px, pw = x + left, w - left - 6
            n = len(hist)
            if n < 2:
                return top + plot_h + 16
            m = min(n, int(pw / 3))
            # Fractions per behaviour, downsampled.
            totals = [sum(s["behaviors"].get(k, 0) for k in self.BEHAVIOR_ORDER) or 1 for s in hist]
            fracs = [self.downsample([s["behaviors"].get(b, 0) / t for s, t in zip(hist, totals)], m)
                     for b, _ in layers]
            m = len(fracs[0])
            base = [0.0] * m
            for (b, color), vals in zip(layers, fracs):
                upper = [base[i] + vals[i] for i in range(m)]
                pts = [QtCore.QPointF(px + pw * i / (m - 1), top + plot_h * (1 - upper[i])) for i in range(m)]
                pts += [QtCore.QPointF(px + pw * i / (m - 1), top + plot_h * (1 - base[i]))
                        for i in range(m - 1, -1, -1)]
                p.setPen(QtCore.Qt.NoPen)
                p.setBrush(qcolor(color))
                p.drawPolygon(QtGui.QPolygonF(pts))
                # 2px surface spacer between stacked layers.
                p.setPen(QtGui.QPen(qcolor(self.SURFACE), 1.5))
                p.drawPolyline(QtGui.QPolygonF(pts[:m]))
                base = upper
            for frac in (0.0, 0.5, 1.0):
                gy = top + plot_h * (1 - frac)
                self.text(p, px - 5, gy + 4, f"{int(frac * 100)}%", self.MUTED, self.f_small, "right")
            self.time_axis(p, (px, top, pw, plot_h), hist, season_labels=False)
            return top + plot_h + 16

        def grouped_bars(self, p, x, y, w, plot_h, title, groups, series_names, xlabels):
            colors = self.SERIES[:len(series_names)]
            top = self.chart_header(p, x, y, w, title, list(zip(series_names, colors)))
            left = 30
            px, pw = x + left, w - left - 6
            vmax = self.nice_max(max((max(g) for g in groups if g), default=1) * 1.1)
            for frac in (0.0, 0.5, 1.0):
                gy = top + plot_h * (1 - frac)
                p.setPen(QtGui.QPen(qcolor(self.AXIS if frac == 0 else self.GRID), 1))
                p.drawLine(QtCore.QPointF(px, gy), QtCore.QPointF(px + pw, gy))
                self.text(p, px - 5, gy + 4, self.fmt(vmax * frac), self.MUTED, self.f_small, "right")
            if groups:
                gw = pw / len(groups)
                k = len(series_names)
                bw = max(2.0, (gw - 8 - 2 * (k - 1)) / k)
                for gi, g in enumerate(groups):
                    gx = px + gi * gw + 4
                    for si, v in enumerate(g):
                        bh = plot_h * clamp(v / vmax, 0, 1)
                        self.bar(p, gx + si * (bw + 2), top + plot_h - bh, bw, bh, colors[si])
                self.text(p, px, top + plot_h + 12, xlabels[0], self.MUTED, self.f_small)
                self.text(p, px + pw, top + plot_h + 12, xlabels[1], self.MUTED, self.f_small, "right")
            return top + plot_h + 16

        def hbars(self, p, x, y, w, title, items):
            top = self.chart_header(p, x, y, w, title)
            label_w = 112
            vmax = max((v for _, v in items), default=0) or 1
            row = 20
            for i, (name, v) in enumerate(items):
                ry = top + i * row
                self.text(p, x, ry + 12, name, self.INK2, self.f_small)
                bw = (w - label_w - 40) * v / vmax
                self.bar(p, x + label_w, ry + 3, bw, row - 6, self.SERIES[0], horizontal=True)
                self.text(p, x + label_w + bw + 6, ry + 12, self.fmt(v), self.INK, self.f_small)
            return top + len(items) * row + 4

        def histogram(self, p, x, y, w, plot_h, title, values, lo, hi, bins, fmt_lo, fmt_hi):
            top = self.chart_header(p, x, y, w, title)
            counts = [0] * bins
            span = (hi - lo) or 1
            for v in values:
                counts[int(clamp((v - lo) / span * bins, 0, bins - 1))] += 1
            vmax = max(counts) or 1
            bw = (w - 2 * (bins - 1)) / bins
            for i, c in enumerate(counts):
                bh = (plot_h - 4) * c / vmax
                self.bar(p, x + i * (bw + 2), top + plot_h - bh, bw, bh, self.SERIES[0])
            p.setPen(QtGui.QPen(qcolor(self.AXIS), 1))
            p.drawLine(QtCore.QPointF(x, top + plot_h), QtCore.QPointF(x + w, top + plot_h))
            self.text(p, x, top + plot_h + 12, fmt_lo, self.MUTED, self.f_small)
            self.text(p, x + w, top + plot_h + 12, fmt_hi, self.MUTED, self.f_small, "right")
            self.text(p, x + w, top - 6, f"peak {vmax}", self.MUTED, self.f_small, "right")
            return top + plot_h + 16

        def tile(self, p, x, y, w, h, label, value, delta, spark):
            path = QtGui.QPainterPath()
            path.addRoundedRect(QtCore.QRectF(x, y, w, h), 8, 8)
            p.fillPath(path, qcolor(self.TILE))
            self.text(p, x + 10, y + 16, label, self.MUTED, self.f_small)
            vw = self.text(p, x + 10, y + 40, value, self.INK, self.f_big)
            if delta:
                self.text(p, x + 10, y + h - 8, delta, self.INK2, self.f_small)
            if spark and len(spark) >= 2:
                sx, sw = x + max(vw + 22, w * 0.5), w - max(vw + 22, w * 0.5) - 10
                if sw > 20:
                    vals = self.downsample(spark, int(sw / 2))
                    lo, hi = min(vals), max(vals)
                    rng = (hi - lo) or 1
                    sy, sh = y + 20, h - 34
                    poly = QtGui.QPolygonF([QtCore.QPointF(sx + sw * i / (len(vals) - 1),
                                                           sy + sh * (1 - (v - lo) / rng))
                                            for i, v in enumerate(vals)])
                    p.setPen(QtGui.QPen(qcolor(self.SERIES[0]), 1.5))
                    p.setBrush(QtCore.Qt.NoBrush)
                    p.drawPolyline(poly)

        # -------------------------------------------------------------- render
        @staticmethod
        def snapshot(world):
            """Copy everything the panel needs, so it can be drawn on a worker thread
            while the simulation keeps changing the live objects."""
            orbs = [SimpleNamespace(
                id=o.id, name=o.name, house=o.house, species=o.species, generation=o.generation, energy=float(o.energy),
                radius=float(o.radius), hunting_success_count=o.hunting_success_count,
                offspring_count=o.offspring_count, food_eaten=o.food_eaten,
                behavior_state=o.behavior_state, fitness=o.lifetime_fitness())
                for o in world.orbs if o.alive]
            return SimpleNamespace(
                history=list(world.history), orbs=orbs, frame=world.frame, season=world.season,
                day=world.day, clock=world.clock_string(), counters=dict(world.counters),
                food_count=len(world.food.items) if world.food else 0)

        def render(self, snap, fps):
            if isinstance(snap, World):
                snap = self.snapshot(snap)
            W, PAD = self.W, self.PAD
            H = 1100   # cropped to the content height at the end
            img = QtGui.QImage(W, H, QtGui.QImage.Format_ARGB32_Premultiplied)
            img.fill(QtCore.Qt.transparent)
            p = QtGui.QPainter(img)
            p.setRenderHint(QtGui.QPainter.Antialiasing)
            p.setRenderHint(QtGui.QPainter.TextAntialiasing)

            bg = QtGui.QPainterPath()
            bg.addRoundedRect(QtCore.QRectF(0.5, 0.5, W - 1, H - 1), 12, 12)
            p.fillPath(bg, qcolor(self.SURFACE, 238))
            p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 26), 1))
            p.drawPath(bg)

            # Thin long histories to ~450 points: plenty for charts a few hundred px wide,
            # and keeps rendering cheap. `minute` = how many thinned samples span 60 s.
            full = snap.history
            stride = max(1, math.ceil(len(full) / 450))
            hist = full[::stride]
            if full and hist[-1] is not full[-1]:
                hist.append(full[-1])
            minute = max(1, round(60 / stride))
            orbs = snap.orbs
            cw = W - 2 * PAD
            x = PAD
            y = PAD

            # Header
            self.text(p, x, y + 16, "Ecosystem stats", self.INK, self.f_title)
            sim_min = snap.frame / 3600.0
            self.text(p, x + cw, y + 16,
                      f"{snap.season.title()}, day {snap.day}, {snap.clock}   "
                      f"running {sim_min:.0f} min   {fps:.0f} fps", self.INK2, self.f_small, "right")
            y += 34

            # Stat tiles
            def series(key):
                return [s.get(key, 0) for s in hist]

            def delta(values, unit=""):
                if len(values) < minute + 1:
                    return ""
                d = values[-1] - values[-1 - minute]
                sign = "+" if d >= 0 else "-"
                return f"{sign}{self.fmt(abs(d))}{unit} vs 1 min ago"

            pop = [s["orbs"] + s["anglers"] for s in hist]
            kills_rate = [0.0] * len(hist)
            births_rate = [0.0] * len(hist)
            for i in range(len(hist)):
                j = max(0, i - minute)
                mins = max((hist[i]["frame"] - hist[j]["frame"]) / 3600.0, 1 / 60)
                kills_rate[i] = (hist[i]["counters"]["kills"] - hist[j]["counters"]["kills"]) / mins
                births_rate[i] = (hist[i]["counters"]["births"] - hist[j]["counters"]["births"]) / mins
            tiles = [
                ("Population", str(len(orbs)), delta(pop), pop),
                ("Food on screen", str(snap.food_count), delta(series("food")),
                 series("food")),
                ("Kills / min", self.fmt(kills_rate[-1] if kills_rate else 0), "last 60 s", kills_rate),
                ("Births / min", self.fmt(births_rate[-1] if births_rate else 0), "last 60 s", births_rate),
                ("Average energy", f"{(sum(o.energy for o in orbs) / len(orbs)) if orbs else 0:.0f}",
                 delta(series("energy")), series("energy")),
                ("Average generation", f"{(sum(o.generation for o in orbs) / len(orbs)) if orbs else 0:.1f}",
                 f"max {max((o.generation for o in orbs), default=0)}", series("generation")),
            ]
            tw = (cw - 16) / 3
            th = 64
            for i, (label, value, d, spark) in enumerate(tiles):
                self.tile(p, x + (i % 3) * (tw + 8), y + (i // 3) * (th + 8), tw, th, label, value, d, spark)
            y += 2 * th + 8 + 16

            # Population over time
            y = self.line_chart(p, x, y, cw, 96, "Population   (shaded = night)", hist,
                                [("orbs", series("orbs"), self.SERIES[0]),
                                 ("anglers", series("anglers"), self.SERIES[1]),
                                 ("splitters", series("splitters"), self.SERIES[2]),
                                 ("scavengers", series("scavengers"), self.SERIES[3]),
                                 ("leeches", series("leeches"), self.SERIES[4]),
                                 ("mimics", series("mimics"), self.SERIES[5])])
            y += 8

            # Behaviour mix
            y = self.stacked_share(p, x, y, cw, 84, "What the orbs are doing (share of population)", hist)
            y += 8

            # Traits | food
            half = (cw - 20) / 2
            y_left = self.line_chart(p, x, y, half, 84, "Evolving traits (population average)", hist,
                                     [("aggression", series("aggression"), self.SERIES[0]),
                                      ("curiosity", series("curiosity"), self.SERIES[1]),
                                      ("social", series("social"), self.SERIES[2]),
                                      ("wariness", series("wariness"), self.SERIES[3]),
                                      ("night vision", series("night_vision"), self.SERIES[4]),
                                      ("mimic disguise", series("disguise"), self.SERIES[5])],
                                     ymax=1.0, direct_labels=False)
            # Align the food plot with the trait plot (whose legend may wrap onto two rows).
            y_right = self.line_chart(p, x + half + 20, y_left - 16 - 84 - 18, half, 84, "Food available", hist,
                                      [("food", series("food"), self.SERIES[0])], direct_labels=True)
            y = max(y_left, y_right) + 8

            # Events per minute | causes of death
            groups = []
            n_bins = 10
            if len(hist) >= 2:
                ends = list(range(len(hist) - 1, 0, -minute))[:n_bins][::-1]
                for e in ends:
                    s0 = hist[max(0, e - minute)]["counters"]
                    s1 = hist[e]["counters"]
                    groups.append([s1["births"] - s0["births"], s1["kills"] - s0["kills"],
                                   s1["starved"] - s0["starved"]])
            y_left = self.grouped_bars(p, x, y, half, 84, "Events per minute", groups,
                                       ["births", "kills", "starved"],
                                       (f"-{len(groups)} min" if groups else "", "now"))
            c = snap.counters
            causes = [("Eaten by hunters", c["kills"] - c["angler_kills"] - c.get("leviathan_kills", 0)),
                      ("Eaten by anglers", c["angler_kills"]),
                      ("Starved", c["starved"]),
                      ("Old age", c["old_age"])]
            # World-event causes only appear once they've happened.
            for label, key in (("Plague", "plague"), ("Lightning", "struck"), ("Meteors", "meteor"),
                               ("Leviathan", "leviathan_kills")):
                if c.get(key, 0):
                    causes.append((label, c[key]))
            causes = sorted(causes, key=lambda t: -t[1])[:6]
            y_right = self.hbars(p, x + half + 20, y, half, f"Causes of death (all time, {c['deaths']})", causes)
            y = max(y_left, y_right) + 8

            # Distributions right now
            third = (cw - 40) / 3
            gens = [o.generation for o in orbs]
            gmax = max(gens, default=0)
            self.histogram(p, x, y, third, 56, "Energy now", [o.energy for o in orbs], 0, 100, 10, "0", "100")
            self.histogram(p, x + third + 20, y, third, 56, "Size now", [o.radius for o in orbs],
                           MIN_RADIUS, MAX_RADIUS, 10, f"{MIN_RADIUS:.0f}px", f"{MAX_RADIUS:.0f}px")
            y = self.histogram(p, x + 2 * (third + 20), y, third, 56, "Generations now", gens,
                               0, max(gmax + 1, 1), min(10, gmax + 1), "0", str(gmax)) + 8

            # Leaderboard
            self.text(p, x, y + 12, "Top orbs (lifetime fitness)", self.INK, self.f_head)
            y += 20
            cols = [("orb", 0), ("species", 60), ("gen", 140), ("kills", 190), ("kids", 245), ("food", 300),
                    ("fitness", 355), ("doing", 430)]
            for name, cx in cols:
                self.text(p, x + cx, y + 12, name, self.MUTED, self.f_small)
            p.setPen(QtGui.QPen(qcolor(self.GRID), 1))
            p.drawLine(QtCore.QPointF(x, y + 17), QtCore.QPointF(x + cw, y + 17))
            y += 20
            for rank, o in enumerate(sorted(orbs, key=lambda o: -o.fitness)[:5]):
                ink = self.INK if rank == 0 else self.INK2
                vals = [o.name, o.species, str(o.generation), str(o.hunting_success_count),
                        str(o.offspring_count), str(o.food_eaten), f"{o.fitness:.0f}",
                        o.behavior_state]
                for (name, cx), v in zip(cols, vals):
                    self.text(p, x + cx, y + 12, v, ink, self.f_small)
                y += 17
            p.end()
            return img.copy(0, 0, W, min(H, int(y + PAD)))

    class IntelligentOrbOverlay(QtWidgets.QWidget):
        def __init__(self):
            super().__init__()
            self.setWindowFlags(QtCore.Qt.FramelessWindowHint |
                                QtCore.Qt.WindowStaysOnTopHint |
                                QtCore.Qt.Tool |
                                QtCore.Qt.WindowTransparentForInput)
            self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
            self.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)

            screens = QtGui.QGuiApplication.screens()
            rect = screens[0].geometry()
            for screen in screens[1:]:
                rect = rect.united(screen.geometry())
            self.setGeometry(rect)
            self._screen_rect = rect
            self.origin = (rect.x(), rect.y())
            prim = QtGui.QGuiApplication.primaryScreen().geometry()
            self._primary_local = QtCore.QRect(prim.x() - rect.x(), prim.y() - rect.y(),
                                               prim.width(), prim.height())
            self.screen_width = rect.width()
            self.screen_height = rect.height()

            self.world = World(self.screen_width, self.screen_height, genepool_file=GENEPOOL_FILE)
            loaded = self.world.load_genepool()
            self.world.set_cursor(self._cursor_local())

            # Desktop windows become terrain the orbs must swim around.
            self.terrain = DesktopTerrain()
            self.use_terrain = self.terrain.available
            self.refresh_terrain()
            self.world.populate()
            if loaded:
                print(f"Loaded {loaded} champion genomes from the gene pool.")

            self.paused = False
            self.show_hud = True
            self.show_debug = False
            self.show_intent = True
            self.show_weather = True
            self.tint_night = False    # off by default: don't darken the user's desktop
            self.weather = Weather(self.screen_width, self.screen_height)

            # Full screen "aquarium" mode: opaque dark sky instead of the desktop.
            self.fullscreen = False
            self._terrain_before_fullscreen = self.use_terrain
            self._last_cursor = None
            self._cursor_idle = QtCore.QElapsedTimer()
            self._cursor_idle.start()
            self._cursor_hidden = False
            self.stars = [(random.uniform(0, self.screen_width), random.uniform(0, self.screen_height),
                           random.uniform(0.5, 1.6), random.uniform(0, math.tau))
                          for _ in range(int(clamp(self.screen_width * self.screen_height / 12000, 80, 450)))]
            self.fullscreen_action = None

            self.show_stats = False
            self.stats_panel = StatsPanel()
            self._stats_image = None
            self._stats_dirty = True
            self._stats_future = None
            # The dashboard is drawn on a worker thread so the animation never stalls.
            self._stats_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="orb-stats")
            self.stats_action = None

            # Speed control: simulation steps per animation tick.
            self.speed_levels = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
            self.speed_index = 2
            self._step_debt = 0.0
            self.actual_speed = 1.0
            self.speed_actions = []

            # Event ticker and following an orb.
            self.show_ticker = True
            self._ticker = []          # [text, kind, shown_at_ms]
            self._ticker_seq = 0
            self._wall = QtCore.QElapsedTimer()
            self._wall.start()
            self.follow_cam = False
            anchor = self._primary_local.center()
            self.cam = [float(anchor.x()), float(anchor.y()), 1.0]
            self.cam_target_zoom = 2.2
            self._hud_bottom = 10

            # God tools (full screen) and world-event controls.
            self.show_scent = True
            self._scent_image = None
            self._scent_buf = None
            self.tool = "follow"
            self.show_toolbar = True
            self._toolbar_hits = []      # [(QRectF, ("tool"|"event", name))]
            self._drag = None            # {"tool", "start", "last"} while the mouse is held
            self._banner = None          # (text, shown_at_ms) for big world events
            self.banner_font = QtGui.QFont("Segoe UI", 22, QtGui.QFont.Bold)
            self.card_font = QtGui.QFont("Segoe UI", 9)
            self.card_title_font = QtGui.QFont("Segoe UI", 12, QtGui.QFont.Bold)
            self.ticker_font = QtGui.QFont("Segoe UI", 10)
            self.name_font = QtGui.QFont("Segoe UI", 8)

            self.hud_font = QtGui.QFont("Segoe UI", 10)
            self.hud_title_font = QtGui.QFont("Segoe UI", 11, QtGui.QFont.Bold)
            self.alarm_font = QtGui.QFont("Segoe UI", 13, QtGui.QFont.Bold)
            self.fps = 60.0
            self._clock = QtCore.QElapsedTimer()
            self._clock.start()

            self.timer = QtCore.QTimer(self)
            self.timer.setTimerType(QtCore.Qt.PreciseTimer)
            self.timer.timeout.connect(self.update_simulation)
            self.timer.start(16)

            self.terrain_timer = QtCore.QTimer(self)
            self.terrain_timer.timeout.connect(self.refresh_terrain)
            self.terrain_timer.start(400)

            self.tray = None
            self._build_tray()

        def refresh_terrain(self):
            if not self.use_terrain:
                if self.world.obstacles:
                    self.world.set_obstacles([])
                return
            try:
                scale = 1.0 / max(self.devicePixelRatioF(), 1e-6)
            except Exception:
                scale = 1.0
            rects = self.terrain.scan(self.origin[0], self.origin[1],
                                      self.screen_width, self.screen_height, scale)
            if rects != self.world.obstacles:
                self.world.set_obstacles(rects)

        def toggle_terrain(self, checked):
            self.use_terrain = bool(checked) and self.terrain.available
            self.refresh_terrain()

        # Backwards-compatible accessors
        @property
        def orbs(self):
            return self.world.orbs

        @property
        def frame_count(self):
            return self.world.frame

        def _cursor_local(self):
            p = self.mapFromGlobal(QtGui.QCursor.pos())
            return p.x(), p.y()

        # ------------------------------------------------------------ tray
        def _build_tray(self):
            if not QtWidgets.QSystemTrayIcon.isSystemTrayAvailable():
                return
            pix = QtGui.QPixmap(64, 64)
            pix.fill(QtCore.Qt.transparent)
            p = QtGui.QPainter(pix)
            p.setRenderHint(QtGui.QPainter.Antialiasing)
            g = QtGui.QRadialGradient(32, 32, 30)
            g.setColorAt(0, QtGui.QColor(255, 255, 255))
            g.setColorAt(0.4, QtGui.QColor(120, 200, 255))
            g.setColorAt(1, QtGui.QColor(60, 40, 200, 0))
            p.setBrush(QtGui.QBrush(g))
            p.setPen(QtCore.Qt.NoPen)
            p.drawEllipse(2, 2, 60, 60)
            p.end()

            self.tray = QtWidgets.QSystemTrayIcon(QtGui.QIcon(pix), self)
            self.tray.setToolTip("Intelligent Orbs")
            menu = QtWidgets.QMenu()

            def action(text, fn, checkable=False, checked=False):
                a = menu.addAction(text)
                a.setCheckable(checkable)
                if checkable:
                    a.setChecked(checked)
                a.triggered.connect(fn)
                return a

            self.fullscreen_action = action("Full screen (Esc to exit)", self.set_fullscreen, True, self.fullscreen)
            self.stats_action = action("Stats panel (S)", self.set_stats, True, self.show_stats)
            self.pause_action = action("Pause", self.toggle_pause, True, False)
            action("Show HUD", lambda c: setattr(self, "show_hud", c), True, True)
            action("Show intent lines", lambda c: setattr(self, "show_intent", c), True, True)
            action("Debug view (vision, terrain)", lambda c: setattr(self, "show_debug", c), True, False)
            terrain = action("Windows as terrain", self.toggle_terrain, True, self.use_terrain)
            terrain.setEnabled(self.terrain.available)
            action("Weather effects", lambda c: setattr(self, "show_weather", c), True, True)
            action("Tint screen at night", lambda c: setattr(self, "tint_night", c), True, False)
            action("Event ticker", lambda c: setattr(self, "show_ticker", c), True, True)
            self.scent_action = action("Scent trails (C)", self.set_scent, True, True)
            action("Random world events", lambda c: setattr(self.world, "random_events", c), True, True)
            events_menu = menu.addMenu("Trigger world event")
            for key, label, _hot in self.EVENT_BUTTONS:
                a = events_menu.addAction(label)
                a.triggered.connect(lambda _c, k=key: self.world.trigger_event(k))
            menu.addSeparator()
            speed_menu = menu.addMenu("Speed")
            group = QtWidgets.QActionGroup(speed_menu)
            for i, s in enumerate(self.speed_levels):
                a = speed_menu.addAction(self.speed_label(s))
                a.setCheckable(True)
                a.setChecked(i == self.speed_index)
                a.triggered.connect(lambda _c, i=i: self.set_speed(i))
                group.addAction(a)
                self.speed_actions.append(a)
            self._speed_group = group
            action("Skip ahead 6 hours", lambda *_: self.world.skip_time(DAY_FRAMES // 4))
            action("Next season", lambda *_: self.world.next_season())
            menu.addSeparator()
            action("Follow the champion (A)", self.follow_champion)
            action("Follow a random orb (R)", self.follow_random)
            action("Stop following (X)", self.stop_following)
            menu.addSeparator()
            action("Drop food at cursor", self.feed_at_cursor)
            action("Spawn 5 orbs", self.spawn_five)
            spawn_menu = menu.addMenu("Spawn species")
            for sp in ("orb", "angler", "splitter", "scavenger", "leech", "mimic"):
                a = spawn_menu.addAction(sp.title() + ("es" if sp == "leech" else "s") + " x3")
                a.triggered.connect(lambda _c, sp=sp: [self.world.spawn_orb(species=sp) for _ in range(3)])
            action("Save gene pool now", self.world.save_genepool)
            menu.addSeparator()
            action("Quit", QtWidgets.QApplication.quit)
            self._tray_menu = menu
            self.tray.setContextMenu(menu)
            self.tray.show()

        # ----------------------------------------------------------- speed
        @staticmethod
        def speed_label(s):
            return {0.25: "1/4x", 0.5: "1/2x"}.get(s, f"{s:g}x")

        def set_speed(self, index):
            self.speed_index = int(clamp(index, 0, len(self.speed_levels) - 1))
            self._step_debt = 0.0
            for i, a in enumerate(self.speed_actions):
                a.setChecked(i == self.speed_index)

        # ------------------------------------------------------- following
        def follow(self, orb, announce=True):
            if orb is None or not orb.alive:
                return
            self.world.followed = orb
            if announce:
                self.world.log(f"Now following {orb.full_name}", "followed")

        def follow_champion(self, *_):
            alive = self.world.alive_orbs()
            if alive:
                self.follow(max(alive, key=lambda o: o.lifetime_fitness()))

        def follow_random(self, *_):
            alive = self.world.alive_orbs()
            if alive:
                self.follow(random.choice(alive))

        def stop_following(self, *_):
            self.world.followed = None
            self.follow_cam = False

        def screen_to_world(self, x, y):
            ax, ay = self._primary_local.center().x(), self._primary_local.center().y()
            cx, cy, z = self.cam
            return (x - ax) / z + cx, (y - ay) / z + cy

        def update_camera(self):
            anchor = self._primary_local.center()
            ax, ay = float(anchor.x()), float(anchor.y())
            f = self.world.followed
            if self.fullscreen and self.follow_cam and f is not None and f.alive:
                tx, ty, tz = f.x, f.y, self.cam_target_zoom
            else:
                tx, ty, tz = ax, ay, 1.0
            cx, cy, z = self.cam
            z += (tz - z) * 0.08
            cx += (tx - cx) * 0.12
            cy += (ty - cy) * 0.12
            # Never show beyond the edge of the world.
            w, h = self.screen_width, self.screen_height
            cx = clamp(cx, ax / z, w - (w - ax) / z)
            cy = clamp(cy, ay / z, h - (h - ay) / z)
            if abs(z - 1.0) < 0.002 and tz == 1.0:
                z, cx, cy = 1.0, ax, ay
            self.cam = [cx, cy, z]

        # ------------------------------------------------------ god tools
        TOOLS = (("follow", "Follow", "1"), ("food", "Food", "2"), ("lightning", "Lightning", "3"),
                 ("wall", "Wall", "4"), ("predator", "Predator", "5"), ("bless", "Bless", "6"))
        EVENT_BUTTONS = (("meteor_shower", "Meteors", "F1"), ("plague", "Plague", "F2"),
                         ("eclipse", "Eclipse", "F3"), ("weather", "Drought / Blizzard", "F4"),
                         ("leviathan", "Leviathan", "F5"))

        def set_tool(self, tool):
            self.tool = tool
            if self.fullscreen and not self._cursor_hidden:
                self.setCursor(self.tool_cursor())

        def tool_cursor(self):
            return QtCore.Qt.ArrowCursor if self.tool == "follow" else QtCore.Qt.CrossCursor

        def orb_at(self, wx, wy):
            best, best_d = None, 1e9
            for o in self.world.alive_orbs():
                d = math.hypot(o.x - wx, o.y - wy)
                if d < o.radius + 30 / self.cam[2] and d < best_d:
                    best, best_d = o, d
            return best

        def mousePressEvent(self, event):
            # Only reachable in full screen (the overlay ignores the mouse).
            pos = event.pos()
            for rect, (kind, name) in self._toolbar_hits:
                if rect.contains(QtCore.QPointF(pos)):
                    if kind == "tool":
                        self.set_tool(name)
                    else:
                        self.world.trigger_event(name)
                    return
            wx, wy = self.screen_to_world(pos.x(), pos.y())
            if event.button() == QtCore.Qt.RightButton:   # right-click always follows
                orb = self.orb_at(wx, wy)
                if orb is not None:
                    self.follow(orb)
                return
            tool = self.tool
            if tool == "follow":
                orb = self.orb_at(wx, wy)
                if orb is not None:
                    self.follow(orb)
                elif self.world.followed is not None:
                    self.stop_following()
            elif tool == "food":
                self.world.drop_food(wx, wy, 6)
                self._drag = {"tool": "food", "start": (wx, wy), "last": (wx, wy)}
            elif tool == "lightning":
                self.world.lightning(wx, wy)
            elif tool == "wall":
                self._drag = {"tool": "wall", "start": (wx, wy), "last": (wx, wy)}
            elif tool == "predator":
                self.world.summon_predator(wx, wy)
            elif tool == "bless":
                self.world.bless(wx, wy)

        def mouseMoveEvent(self, event):
            if self._drag is None:
                return
            wx, wy = self.screen_to_world(event.pos().x(), event.pos().y())
            d = self._drag
            if d["tool"] == "food":
                lx, ly = d["last"]
                if math.hypot(wx - lx, wy - ly) > 18:      # sprinkle along the drag
                    self.world.drop_food(wx, wy, 3)
                    d["last"] = (wx, wy)
            else:
                d["last"] = (wx, wy)

        def mouseReleaseEvent(self, event):
            d = self._drag
            self._drag = None
            if d is not None and d["tool"] == "wall":
                (x0, y0), (x1, y1) = d["start"], d["last"]
                self.world.add_wall(x0, y0, x1, y1)

        def wheelEvent(self, event):
            if self.follow_cam:
                steps = event.angleDelta().y() / 120.0
                self.cam_target_zoom = clamp(self.cam_target_zoom * (1.15 ** steps), 1.2, 5.0)

        def toggle_pause(self, checked=None):
            self.paused = not self.paused if checked is None else bool(checked)
            if getattr(self, "pause_action", None) is not None:
                self.pause_action.setChecked(self.paused)

        def set_stats(self, on):
            self.show_stats = bool(on)
            self._stats_dirty = True
            if self.stats_action is not None:
                self.stats_action.setChecked(self.show_stats)

        def draw_stats(self, painter):
            img = self._stats_image
            if img is None:
                return
            area = self._primary_local
            avail_h = area.height() - 32
            scale = min(1.0, avail_h / img.height()) if img.height() > 0 else 1.0
            w, h = img.width() * scale, img.height() * scale
            x = area.x() + area.width() - w - 16
            y = area.y() + 16
            painter.setRenderHint(QtGui.QPainter.SmoothPixmapTransform)
            painter.drawImage(QtCore.QRectF(x, y, w, h), img)

        # ----------------------------------------------------- full screen
        def set_fullscreen(self, on):
            """Switch between the transparent desktop overlay and an opaque full screen sky."""
            on = bool(on)
            if on == self.fullscreen:
                return
            self.fullscreen = on
            self.hide()
            flags = QtCore.Qt.FramelessWindowHint | QtCore.Qt.WindowStaysOnTopHint | QtCore.Qt.Tool
            if not on:
                flags |= QtCore.Qt.WindowTransparentForInput
            # The attribute must change first: Qt re-adds the input-transparent flag
            # when window flags are set while WA_TransparentForMouseEvents is on.
            self.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, not on)
            self.setWindowFlags(flags)
            self.setGeometry(self._screen_rect)
            if on:
                # Your windows are hidden behind the sky, so they stop being terrain.
                self._terrain_before_fullscreen = self.use_terrain
                self.use_terrain = False
                self.setFocusPolicy(QtCore.Qt.StrongFocus)
            else:
                self.use_terrain = self._terrain_before_fullscreen and self.terrain.available
                self.setFocusPolicy(QtCore.Qt.NoFocus)
                self._show_cursor()
            self.refresh_terrain()
            self.show()
            if on:
                self.raise_()
                self.activateWindow()
                self.setFocus()
            if self.fullscreen_action is not None:
                self.fullscreen_action.setChecked(on)

        def _show_cursor(self):
            if self._cursor_hidden:
                if self.fullscreen:
                    self.setCursor(self.tool_cursor())
                else:
                    self.unsetCursor()
                self._cursor_hidden = False

        def keyPressEvent(self, event):
            key = event.key()
            if key in (QtCore.Qt.Key_Escape, QtCore.Qt.Key_F11):
                self.set_fullscreen(False)
            elif key == QtCore.Qt.Key_Space:
                self.toggle_pause()
            elif key == QtCore.Qt.Key_H:
                self.show_hud = not self.show_hud
            elif key == QtCore.Qt.Key_S:
                self.set_stats(not self.show_stats)
            elif key == QtCore.Qt.Key_W:
                self.show_weather = not self.show_weather
            elif key == QtCore.Qt.Key_N:
                self.world.next_season()
            elif key == QtCore.Qt.Key_T:
                self.world.skip_time(DAY_FRAMES // 4)
            elif key == QtCore.Qt.Key_F:
                self.feed_at_cursor()
            elif key in (QtCore.Qt.Key_BracketRight, QtCore.Qt.Key_Plus, QtCore.Qt.Key_Equal):
                self.set_speed(self.speed_index + 1)
            elif key in (QtCore.Qt.Key_BracketLeft, QtCore.Qt.Key_Minus):
                self.set_speed(self.speed_index - 1)
            elif key == QtCore.Qt.Key_0:
                self.set_speed(self.speed_levels.index(1.0))
            elif key == QtCore.Qt.Key_A:
                self.follow_champion()
            elif key == QtCore.Qt.Key_R:
                self.follow_random()
            elif key == QtCore.Qt.Key_X:
                self.stop_following()
            elif key == QtCore.Qt.Key_Z:
                self.follow_cam = not self.follow_cam
                if self.follow_cam and self.world.followed is None:
                    self.follow_champion()
            elif key == QtCore.Qt.Key_E:
                self.show_ticker = not self.show_ticker
            elif key == QtCore.Qt.Key_G:
                self.show_toolbar = not self.show_toolbar
            elif key == QtCore.Qt.Key_C:
                self.set_scent(not self.show_scent)
            elif QtCore.Qt.Key_1 <= key <= QtCore.Qt.Key_6:
                self.set_tool(self.TOOLS[key - QtCore.Qt.Key_1][0])
            elif QtCore.Qt.Key_F1 <= key <= QtCore.Qt.Key_F5:
                self.world.trigger_event(self.EVENT_BUTTONS[key - QtCore.Qt.Key_F1][0])
            else:
                super().keyPressEvent(event)

        def closeEvent(self, event):
            # Alt+F4 in full screen quits the whole app instead of leaving it invisible.
            QtWidgets.QApplication.quit()
            event.accept()

        def feed_at_cursor(self, *_):
            if self.world.food is not None:
                x, y = self.screen_to_world(*self._cursor_local())
                self.world.food.burst(x, y, 20)

        def spawn_five(self, *_):
            for _ in range(5):
                if len(self.world.orbs) < self.world.max_orbs + 10:
                    self.world.spawn_orb()

        def shutdown(self):
            self._stats_executor.shutdown(wait=False)
            self.world.save_genepool()
            self.world.export_statistics()
            if self.tray:
                self.tray.hide()

        # ------------------------------------------------------- simulation
        def update_simulation(self):
            elapsed = self._clock.restart()
            if elapsed > 0:
                self.fps = self.fps * 0.95 + (1000.0 / elapsed) * 0.05

            cursor = self._cursor_local()
            if self.fullscreen:
                # Hide an idle cursor so the view is clean; orbs ignore a hidden cursor.
                if cursor != self._last_cursor:
                    self._last_cursor = cursor
                    self._cursor_idle.restart()
                    self._show_cursor()
                elif not self._cursor_hidden and self._cursor_idle.elapsed() > 3000:
                    self.setCursor(QtCore.Qt.BlankCursor)
                    self._cursor_hidden = True
            if self._cursor_hidden:
                self.world.cursor = (-10000.0, -10000.0)
                self.world.cursor_vel = (0.0, 0.0)
            else:
                self.world.set_cursor(self.screen_to_world(*cursor))
            if not self.paused:
                # Run as many simulation steps as the speed asks for, within a frame budget.
                self._step_debt += self.speed_levels[self.speed_index]
                budget = QtCore.QElapsedTimer()
                budget.start()
                steps = 0
                while self._step_debt >= 1.0:
                    self.world.step()
                    steps += 1
                    self._step_debt -= 1.0
                    if self.show_weather:
                        self.weather.update(self.world)
                    if self.world.frame % 30 == 0:
                        self.export_statistics()
                    if self.world.frame % HISTORY_EVERY == 0:
                        self._stats_dirty = True
                    if budget.elapsed() > 11:     # can't keep up: run slower rather than stutter
                        self._step_debt = min(self._step_debt, 1.0)
                        break
                self.actual_speed = self.actual_speed * 0.95 + steps * 0.05
            if (self.show_scent and self.world.scent is not None and
                    (self._scent_image is None or self.world.frame % 10 == 0)):
                self.build_scent_image()
            self.pull_events()
            self.update_camera()
            if self.show_stats:
                self.update_stats_panel()
            self.update()

        def pull_events(self):
            now = self._wall.elapsed()
            for e in self.world.events:
                if e["seq"] > self._ticker_seq:
                    self._ticker.append([e["text"], e["kind"], now])
                    self._ticker_seq = e["seq"]
                    if e["kind"] == "event" or e["text"].startswith("LEGENDARY"):
                        self._banner = (e["text"], now)
            self._ticker = [t for t in self._ticker if now - t[2] < 15000][-7:]

        def update_stats_panel(self):
            """Collect a finished dashboard image; queue a new one when a history
            sample lands (once per simulated second)."""
            fut = self._stats_future
            if fut is not None and fut.done():
                try:
                    self._stats_image = fut.result()
                except Exception as exc:   # never let the dashboard take the overlay down
                    print(f"Stats panel render failed: {exc}")
                self._stats_future = None
            if self._stats_future is None and self._stats_dirty:
                snap = StatsPanel.snapshot(self.world)
                self._stats_future = self._stats_executor.submit(self.stats_panel.render, snap, self.fps)
                self._stats_dirty = False

        def export_statistics(self):
            self.world.export_statistics()

        # ---------------------------------------------------------- drawing
        def paintEvent(self, event):
            painter = QtGui.QPainter(self)
            painter.setRenderHint(QtGui.QPainter.Antialiasing)
            world = self.world

            if self.fullscreen:
                self.draw_sky(painter)
            elif self.tint_night:
                dusk = clamp((0.65 - world.light) / 0.5, 0, 1)
                if dusk > 0:
                    painter.fillRect(self.rect(), QtGui.QColor(6, 10, 40, int(85 * dusk)))
            # World layers go through the follow camera (identity unless zoomed in).
            painter.save()
            cx, cy, z = self.cam
            if z != 1.0 or (cx, cy) != (self._primary_local.center().x(), self._primary_local.center().y()):
                anchor = self._primary_local.center()
                painter.translate(anchor.x(), anchor.y())
                painter.scale(z, z)
                painter.translate(-cx, -cy)
            if self.show_debug:
                painter.setBrush(QtCore.Qt.NoBrush)
                painter.setPen(QtGui.QPen(QtGui.QColor(120, 200, 255, 90), 2, QtCore.Qt.DashLine))
                for x0, y0, x1, y1 in world.obstacles:
                    painter.drawRect(QtCore.QRectF(x0, y0, x1 - x0, y1 - y0))
            if self.show_scent and self._scent_image is not None:
                self.draw_scent(painter)
            self.draw_territories(painter)
            self.draw_walls(painter)
            self.draw_carrion(painter)
            self.draw_food(painter)
            self.draw_lures(painter)
            if self.show_intent:
                self.draw_intent(painter)
            for orb in world.orbs:
                if orb.alive:
                    self.draw_trail(painter, orb)
            for orb in world.orbs:
                if orb.alive:
                    sp = orb.species
                    if sp == "leech":
                        continue          # drawn on top of everyone, below
                    if sp == "angler":
                        self.draw_angler(painter, orb)
                    elif sp == "leviathan":
                        self.draw_leviathan(painter, orb)
                    elif sp == "splitter":
                        self.draw_splitter(painter, orb)
                    elif sp == "scavenger":
                        self.draw_scavenger(painter, orb)
                    elif sp == "mimic":
                        self.draw_mimic(painter, orb)
                    else:
                        self.draw_orb(painter, orb)
                    if orb.infected:
                        self.draw_sickness(painter, orb)
                    if orb.alarm_flash > 0:
                        self.draw_alarm_mark(painter, orb)
                if orb.is_exploding:
                    self.draw_explosion_particles(painter, orb)
            for orb in world.orbs:
                if orb.alive and orb.species == "leech":
                    self.draw_leech(painter, orb)
            self.draw_effects(painter)
            self.draw_meteors_and_bolts(painter)
            self.draw_followed_marker(painter)
            if z > 1.5:
                self.draw_name_tags(painter)
            painter.restore()

            if self.show_weather:
                self.weather.draw(painter, world.frame)
            if self.bolt_flash() > 0:
                painter.fillRect(self.rect(), QtGui.QColor(220, 230, 255, int(70 * self.bolt_flash())))
            if self.show_stats:
                self.draw_stats(painter)
            self._hud_bottom = 10
            if self.show_hud:
                self.draw_hud(painter)
            if world.followed is not None:
                self.draw_follow_card(painter)
            if self.show_ticker:
                self.draw_ticker(painter)
            self.draw_banner(painter)
            self._toolbar_hits = []
            if self.fullscreen and self.show_toolbar:
                self.draw_toolbar(painter)
            painter.end()

        # ------------------------------------------------------------ scent
        def set_scent(self, on):
            self.show_scent = bool(on)
            if getattr(self, "scent_action", None) is not None:
                self.scent_action.setChecked(self.show_scent)

        def build_scent_image(self):
            """Turn the scent grids into a small RGBA image (scaled up smoothly when drawn)."""
            g = self.world.scent.grids
            food = np.clip(np.sqrt(g["food"]) / 2.2, 0, 1)
            body = np.clip(np.sqrt(g["prey"]) / 1.8, 0, 1) * 0.65
            danger = np.clip(np.sqrt(g["danger"]) / 2.2, 0, 1)
            total = food + body + danger + 1e-6
            # Colour is the mix of whichever smells are present; strength sets the opacity.
            r = (90 * food + 110 * body + 255 * danger) / total
            gr = (235 * food + 150 * body + 80 * danger) / total
            b = (120 * food + 255 * body + 70 * danger) / total
            a = np.maximum(np.maximum(food * 0.8, body * 0.6), danger * 0.9) * 160
            buf = np.dstack([r, gr, b, a]).astype(np.uint8)
            self._scent_buf = np.ascontiguousarray(buf)     # keep the memory alive for QImage
            h, w = self._scent_buf.shape[:2]
            self._scent_image = QtGui.QImage(self._scent_buf.data, w, h, w * 4, QtGui.QImage.Format_RGBA8888)

        def draw_scent(self, painter):
            sc = self.world.scent
            painter.save()
            painter.setRenderHint(QtGui.QPainter.SmoothPixmapTransform)
            painter.drawImage(QtCore.QRectF(0, 0, sc.nx * sc.cell, sc.ny * sc.cell), self._scent_image)
            painter.restore()

        # ------------------------------------------------------ new species
        @staticmethod
        def house_color(house, s=150, v=235, a=255):
            hue = sum((i + 1) * ord(ch) for i, ch in enumerate(house or "?")) * 47 % 360
            return QtGui.QColor.fromHsv(hue, s, v, a)

        def draw_territories(self, painter):
            for t in self.world.territories:
                c = self.house_color(t.house)
                fill = QtGui.QColor(c)
                fill.setAlpha(14)
                painter.setBrush(fill)
                edge = QtGui.QColor(c)
                edge.setAlpha(110)
                painter.setPen(QtGui.QPen(edge, 2, QtCore.Qt.DashLine))
                painter.drawEllipse(QtCore.QPointF(t.x, t.y), t.radius, t.radius)
                painter.setFont(self.name_font)
                label = QtGui.QColor(c)
                label.setAlpha(190)
                painter.setPen(label)
                painter.drawText(QtCore.QRectF(t.x - 80, t.y - t.radius - 16, 160, 14), QtCore.Qt.AlignCenter,
                                 f"House {t.house}")

        def draw_mimic(self, painter, o):
            """Dressed up as a dangerous hunter: a red diamond with a hunting glow, puffed up
            to look bigger. Poor disguises show pale give-away spots."""
            d = o.disguise
            true_c = o.base_color
            warn = (235, 60, 50)
            c = tuple(int(a + (b - a) * (0.35 + 0.65 * d)) for a, b in zip(true_c, warn))
            r = o.apparent_radius
            center = QtCore.QPointF(o.x, o.y)
            glow = QtGui.QRadialGradient(center, r * 2.2)
            glow.setColorAt(0, qcolor(c, 150))
            glow.setColorAt(1, qcolor(c, 0))
            painter.setPen(QtCore.Qt.NoPen)
            painter.setBrush(QtGui.QBrush(glow))
            painter.drawEllipse(center, r * 2.2, r * 2.2)
            ch, sh = math.cos(o.heading), math.sin(o.heading)
            pts = [(r * 1.25, 0), (0, r * 0.8), (-r, 0), (0, -r * 0.8)]
            poly = QtGui.QPolygonF([QtCore.QPointF(o.x + px * ch - py * sh, o.y + px * sh + py * ch) for px, py in pts])
            body = QtGui.QRadialGradient(QtCore.QPointF(o.x - r * 0.3, o.y - r * 0.3), r * 1.3)
            body.setColorAt(0, qcolor((min(255, c[0] + 60), min(255, c[1] + 60), min(255, c[2] + 60))))
            body.setColorAt(1, qcolor((c[0] // 2, c[1] // 2, c[2] // 2)))
            painter.setBrush(QtGui.QBrush(body))
            painter.drawPolygon(poly)
            # The tells: pale spots that fade as the disguise improves.
            painter.setBrush(qcolor(true_c, 230 * (1 - d)))
            for k in (-1, 1):
                painter.drawEllipse(QtCore.QPointF(o.x - ch * r * 0.3 - sh * k * r * 0.3,
                                                   o.y - sh * r * 0.3 + ch * k * r * 0.3), r * 0.16, r * 0.16)
            # A fierce-looking eye (a fake hunting ring, too, while it bluffs).
            ex, ey = o.x + ch * r * 0.45, o.y + sh * r * 0.45
            painter.setBrush(QtGui.QColor(255, 255, 255, 230))
            painter.drawEllipse(QtCore.QPointF(ex, ey), r * 0.22, r * 0.22)
            painter.setBrush(QtGui.QColor(15, 15, 25))
            painter.drawEllipse(QtCore.QPointF(ex + ch * 1.5, ey + sh * 1.5), r * 0.11, r * 0.11)
            if o.behavior_state not in ("fleeing", "mating"):
                painter.setPen(QtGui.QPen(QtGui.QColor(255, 40, 40, int(200 * d)), 2))
                painter.setBrush(QtCore.Qt.NoBrush)
                painter.drawEllipse(center, r + 3, r + 3)
            else:
                ring = BEHAVIOR_RING.get(o.behavior_state)
                if ring:
                    painter.setPen(QtGui.QPen(QtGui.QColor(*ring), 2))
                    painter.setBrush(QtCore.Qt.NoBrush)
                    painter.drawEllipse(center, r + 3, r + 3)
            if self.show_debug:   # reveal its true size
                painter.setPen(QtGui.QPen(qcolor(true_c, 200), 1, QtCore.Qt.DashLine))
                painter.setBrush(QtCore.Qt.NoBrush)
                painter.drawEllipse(center, o.radius, o.radius)

        def draw_carrion(self, painter):
            painter.setPen(QtCore.Qt.NoPen)
            for c in self.world.carrion:
                fade = clamp(min(c.age / 20.0, (c.max_age - c.age) / 400.0), 0, 1)
                size = clamp(4 + c.value * 0.25, 4, 22)
                rng = random.Random(id(c))          # stable irregular shape per carcass
                painter.setBrush(QtGui.QColor(120, 50, 45, int(110 * fade)))
                painter.drawEllipse(QtCore.QPointF(c.x, c.y), size * 1.5, size * 1.1)
                painter.setBrush(QtGui.QColor(170, 80, 65, int(170 * fade)))
                for _ in range(4):
                    ox, oy = rng.uniform(-size, size) * 0.6, rng.uniform(-size, size) * 0.5
                    painter.drawEllipse(QtCore.QPointF(c.x + ox, c.y + oy), size * 0.35, size * 0.3)
                painter.setBrush(QtGui.QColor(230, 220, 200, int(150 * fade)))   # little bones
                for _ in range(2):
                    ox, oy = rng.uniform(-size, size) * 0.5, rng.uniform(-size, size) * 0.4
                    painter.drawEllipse(QtCore.QPointF(c.x + ox, c.y + oy), 1.8, 1.2)

        def draw_splitter(self, painter, o):
            """A wobbling amoeba that pinches into a peanut shape as it nears splitting."""
            t = self._wall.elapsed() / 1000.0
            color = o.current_color
            r = o.radius
            ready = clamp((r / o.size_gene - 0.8) / 0.2, 0, 1)   # 0 .. 1 as it nears its split size
            axis = o.id * 1.7
            ax, ay = math.cos(axis), math.sin(axis)
            centers = [(o.x - ax * r * 0.35 * ready, o.y - ay * r * 0.35 * ready),
                       (o.x + ax * r * 0.35 * ready, o.y + ay * r * 0.35 * ready)] if ready > 0 else [(o.x, o.y)]
            lobe_r = r * (1 - 0.25 * ready)
            glow = QtGui.QRadialGradient(QtCore.QPointF(o.x, o.y), r * 2)
            glow.setColorAt(0, qcolor(color, 90))
            glow.setColorAt(1, qcolor(color, 0))
            painter.setPen(QtCore.Qt.NoPen)
            painter.setBrush(QtGui.QBrush(glow))
            painter.drawEllipse(QtCore.QPointF(o.x, o.y), r * 2, r * 2)
            for i, (cx, cy) in enumerate(centers):
                poly = QtGui.QPolygonF()
                for k in range(18):
                    a = k * math.tau / 18
                    wob = 1 + 0.09 * math.sin(t * 3 + k * 1.3 + o.id + i) + 0.05 * math.sin(t * 5 + k * 2.1)
                    poly.append(QtCore.QPointF(cx + math.cos(a) * lobe_r * wob, cy + math.sin(a) * lobe_r * wob))
                body = QtGui.QRadialGradient(QtCore.QPointF(cx - lobe_r * 0.3, cy - lobe_r * 0.3), lobe_r * 1.4)
                body.setColorAt(0, qcolor((min(255, color[0] + 70), min(255, color[1] + 50), min(255, color[2] + 60)), 200))
                body.setColorAt(1, qcolor(color, 150))
                painter.setBrush(QtGui.QBrush(body))
                painter.setPen(QtGui.QPen(qcolor(color, 220), 1.2))
                painter.drawPolygon(poly)
                # Nucleus
                painter.setPen(QtCore.Qt.NoPen)
                painter.setBrush(qcolor((color[0] // 3, color[1] // 2, color[2] // 2), 200))
                painter.drawEllipse(QtCore.QPointF(cx + math.sin(t + i) * 1.5, cy + math.cos(t * 1.3 + i) * 1.5),
                                    lobe_r * 0.28, lobe_r * 0.28)
            ring = BEHAVIOR_RING.get(o.behavior_state)
            if ring:
                painter.setPen(QtGui.QPen(QtGui.QColor(*ring), 1.5))
                painter.setBrush(QtCore.Qt.NoBrush)
                painter.drawEllipse(QtCore.QPointF(o.x, o.y), r + 4, r + 4)

        def draw_scavenger(self, painter, o):
            """A vulture-ish flyer: body, beak and flapping wings."""
            t = self._wall.elapsed() / 1000.0
            c = o.current_color
            r = o.radius
            h = o.heading
            ch, sh = math.cos(h), math.sin(h)
            flap = math.sin(t * (9 if o.sprinting else 5) + o.id) * 0.5
            painter.setPen(QtGui.QPen(qcolor((c[0] // 2, c[1] // 2, c[2] // 2), 230), 1))
            painter.setBrush(qcolor((int(c[0] * 0.8), int(c[1] * 0.8), int(c[2] * 0.8)), 230))
            for side in (-1, 1):
                wing = QtGui.QPolygonF()
                base_a = h + side * (math.pi / 2 + flap)
                wing.append(QtCore.QPointF(o.x + ch * r * 0.3, o.y + sh * r * 0.3))
                wing.append(QtCore.QPointF(o.x + math.cos(base_a) * r * 2.0 - ch * r * 0.4,
                                           o.y + math.sin(base_a) * r * 2.0 - sh * r * 0.4))
                wing.append(QtCore.QPointF(o.x - ch * r * 0.6, o.y - sh * r * 0.6))
                painter.drawPolygon(wing)
            body = QtGui.QRadialGradient(QtCore.QPointF(o.x - ch * 2, o.y - sh * 2), r * 1.2)
            body.setColorAt(0, qcolor((min(255, c[0] + 50), min(255, c[1] + 45), min(255, c[2] + 40))))
            body.setColorAt(1, qcolor((c[0] // 2, c[1] // 2, c[2] // 2)))
            painter.setPen(QtCore.Qt.NoPen)
            painter.setBrush(QtGui.QBrush(body))
            painter.drawEllipse(QtCore.QPointF(o.x, o.y), r, r)
            # Pale bald head with a hooked beak.
            hx, hy = o.x + ch * r * 0.85, o.y + sh * r * 0.85
            painter.setBrush(QtGui.QColor(230, 200, 185))
            painter.drawEllipse(QtCore.QPointF(hx, hy), r * 0.42, r * 0.42)
            painter.setBrush(QtGui.QColor(220, 170, 60))
            painter.drawPolygon(QtGui.QPolygonF([
                QtCore.QPointF(hx + ch * r * 0.75, hy + sh * r * 0.75),
                QtCore.QPointF(hx - sh * r * 0.2, hy + ch * r * 0.2),
                QtCore.QPointF(hx + sh * r * 0.2, hy - ch * r * 0.2)]))
            painter.setBrush(QtGui.QColor(15, 15, 20))
            painter.drawEllipse(QtCore.QPointF(hx + ch * 1.5 - sh * 1.5, hy + sh * 1.5 + ch * 1.5), 1.6, 1.6)
            ring = BEHAVIOR_RING.get(o.behavior_state)
            if ring:
                painter.setPen(QtGui.QPen(QtGui.QColor(*ring), 1.5))
                painter.setBrush(QtCore.Qt.NoBrush)
                painter.drawEllipse(QtCore.QPointF(o.x, o.y), r + 4, r + 4)

        def draw_leech(self, painter, o):
            """A little segmented worm; attached ones pulse as they feed."""
            t = self._wall.elapsed() / 1000.0
            c = o.current_color
            r = o.radius
            if o.host is not None:
                a = o.attach_angle
                ux, uy = math.cos(a), math.sin(a)          # pointing away from the host
                pulse = 1 + 0.18 * math.sin(t * 8 + o.id)
                for i in range(3, -1, -1):
                    px = o.x + ux * i * r * 0.55
                    py = o.y + uy * i * r * 0.55
                    rr = r * (0.9 - i * 0.15) * (pulse if i == 0 else 1)
                    painter.setPen(QtCore.Qt.NoPen)
                    painter.setBrush(qcolor(c, 240 - i * 30))
                    painter.drawEllipse(QtCore.QPointF(px, py), rr, rr)
                # A little drop of drained energy at the bite.
                painter.setBrush(QtGui.QColor(255, 90, 110, int(150 + 100 * math.sin(t * 8 + o.id))))
                painter.drawEllipse(QtCore.QPointF(o.x - ux * r * 0.6, o.y - uy * r * 0.6), 1.8, 1.8)
                return
            h = o.heading
            ch, sh = math.cos(h), math.sin(h)
            for i in range(4, -1, -1):
                wig = math.sin(t * 12 - i * 1.1 + o.id) * r * 0.35
                px = o.x - ch * i * r * 0.7 - sh * wig
                py = o.y - sh * i * r * 0.7 + ch * wig
                painter.setPen(QtCore.Qt.NoPen)
                painter.setBrush(qcolor(c, 230 - i * 30))
                painter.drawEllipse(QtCore.QPointF(px, py), r * (1 - i * 0.12), r * (1 - i * 0.12))
            # Mouth
            painter.setBrush(QtGui.QColor(30, 5, 10))
            painter.drawEllipse(QtCore.QPointF(o.x + ch * r * 0.6, o.y + sh * r * 0.6), r * 0.35, r * 0.35)

        # --------------------------------------------- world events & god tools
        def bolt_flash(self):
            return max((b[1] / 18.0 for b in self.world.bolts if b[1] > 12), default=0.0)

        def draw_walls(self, painter):
            for w in self.world.walls:
                fade = min(1.0, w.life / 300.0)
                pen = QtGui.QPen(QtGui.QColor(150, 170, 200, int(60 * fade)), w.half * 2 + 8,
                                 QtCore.Qt.SolidLine, QtCore.Qt.RoundCap)
                painter.setPen(pen)
                painter.drawLine(QtCore.QPointF(w.x0, w.y0), QtCore.QPointF(w.x1, w.y1))
                pen = QtGui.QPen(QtGui.QColor(205, 215, 235, int(220 * fade)), w.half * 2,
                                 QtCore.Qt.SolidLine, QtCore.Qt.RoundCap)
                painter.setPen(pen)
                painter.drawLine(QtCore.QPointF(w.x0, w.y0), QtCore.QPointF(w.x1, w.y1))
            d = self._drag
            if d is not None and d["tool"] == "wall":
                painter.setPen(QtGui.QPen(QtGui.QColor(205, 215, 235, 150), 12, QtCore.Qt.DashLine,
                                          QtCore.Qt.RoundCap))
                painter.drawLine(QtCore.QPointF(*d["start"]), QtCore.QPointF(*d["last"]))

        def draw_meteors_and_bolts(self, painter):
            for m in self.world.meteors:
                pts = m["trail"] + [(m["x"], m["y"])]
                n = len(pts)
                for i in range(1, n):
                    a = i / n
                    painter.setPen(QtGui.QPen(QtGui.QColor(255, int(140 + 100 * a), 80, int(200 * a)),
                                              1 + 6 * a, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap))
                    painter.drawLine(QtCore.QPointF(*pts[i - 1]), QtCore.QPointF(*pts[i]))
                glow = QtGui.QRadialGradient(QtCore.QPointF(m["x"], m["y"]), 18)
                glow.setColorAt(0, QtGui.QColor(255, 250, 220, 255))
                glow.setColorAt(1, QtGui.QColor(255, 140, 60, 0))
                painter.setPen(QtCore.Qt.NoPen)
                painter.setBrush(QtGui.QBrush(glow))
                painter.drawEllipse(QtCore.QPointF(m["x"], m["y"]), 18, 18)
                # A faint target marker so you can see where it'll land.
                painter.setBrush(QtCore.Qt.NoBrush)
                painter.setPen(QtGui.QPen(QtGui.QColor(255, 150, 60, 90), 1.5, QtCore.Qt.DashLine))
                painter.drawEllipse(QtCore.QPointF(m["tx"], m["ty"]), 30, 30)
            for pts, life in self.world.bolts:
                a = life / 18.0
                poly = QtGui.QPolygonF([QtCore.QPointF(x, y) for x, y in pts])
                painter.setBrush(QtCore.Qt.NoBrush)
                painter.setPen(QtGui.QPen(QtGui.QColor(150, 180, 255, int(120 * a)), 9,
                                          QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin))
                painter.drawPolyline(poly)
                painter.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, int(255 * a)), 2.5,
                                          QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin))
                painter.drawPolyline(poly)

        def draw_sickness(self, painter, orb):
            t = self._wall.elapsed() / 400.0
            painter.setPen(QtCore.Qt.NoPen)
            for i in range(5):
                a = t + i * math.tau / 5
                r = orb.radius + 4 + 2 * math.sin(t * 2 + i)
                painter.setBrush(QtGui.QColor(170, 210, 60, 190))
                painter.drawEllipse(QtCore.QPointF(orb.x + math.cos(a) * r, orb.y + math.sin(a) * r), 2.6, 2.6)

        def draw_leviathan(self, painter, lev):
            t = self._wall.elapsed() / 1000.0
            r = lev.radius
            h = lev.heading
            c, s = math.cos(h), math.sin(h)
            # Wake / tail: fading discs trailing behind.
            painter.setPen(QtCore.Qt.NoPen)
            for i in range(1, 7):
                bx = lev.x - c * r * 0.55 * i + -s * math.sin(t * 2 + i * 0.7) * 8
                by = lev.y - s * r * 0.55 * i + c * math.sin(t * 2 + i * 0.7) * 8
                rr = r * (1 - i * 0.13)
                painter.setBrush(QtGui.QColor(30, 55, 90, int(150 - i * 18)))
                painter.drawEllipse(QtCore.QPointF(bx, by), rr, rr)
            # Body with a deep-sea gradient.
            body = QtGui.QRadialGradient(QtCore.QPointF(lev.x - c * r * 0.2, lev.y - s * r * 0.2), r * 1.3)
            hurt = 1 - clamp(lev.energy / LEVIATHAN_HEALTH, 0, 1)
            body.setColorAt(0, QtGui.QColor(int(60 + 120 * hurt), 95, 140, 240))
            body.setColorAt(1, QtGui.QColor(12, 22, 40, 240))
            painter.setBrush(QtGui.QBrush(body))
            painter.drawEllipse(QtCore.QPointF(lev.x, lev.y), r, r)
            # Glowing bioluminescent spots along its flanks.
            for i in range(8):
                a = h + math.pi * (0.35 + 0.3 * (i % 4)) * (1 if i < 4 else -1)
                pulse = 0.5 + 0.5 * math.sin(t * 3 + i)
                painter.setBrush(QtGui.QColor(120, 230, 255, int(120 + 120 * pulse)))
                painter.drawEllipse(QtCore.QPointF(lev.x + math.cos(a) * r * 0.75,
                                                   lev.y + math.sin(a) * r * 0.75), 3.5, 3.5)
            # Mouth - wide open when lunging.
            gape = 0.55 if lev.behavior_state == "hunting" else 0.2
            painter.setBrush(QtGui.QColor(5, 8, 15, 255))
            painter.drawPie(QtCore.QRectF(lev.x - r, lev.y - r, 2 * r, 2 * r),
                            int((-math.degrees(h) - math.degrees(gape)) * 16), int(math.degrees(2 * gape) * 16))
            # A pair of pale eyes.
            for side in (-1, 1):
                ex = lev.x + c * r * 0.45 - s * side * r * 0.42
                ey = lev.y + s * r * 0.45 + c * side * r * 0.42
                painter.setBrush(QtGui.QColor(230, 240, 255, 240))
                painter.drawEllipse(QtCore.QPointF(ex, ey), 5, 5)
                painter.setBrush(QtGui.QColor(10, 10, 20))
                painter.drawEllipse(QtCore.QPointF(ex + c * 1.5, ey + s * 1.5), 2.5, 2.5)
            # Health bar once it's been hurt.
            if hurt > 0:
                bw = r * 1.6
                painter.setBrush(QtGui.QColor(40, 40, 45, 200))
                painter.drawRoundedRect(QtCore.QRectF(lev.x - bw / 2, lev.y - r - 16, bw, 6), 3, 3)
                painter.setBrush(QtGui.QColor(230, 90, 90, 230))
                painter.drawRoundedRect(QtCore.QRectF(lev.x - bw / 2, lev.y - r - 16, bw * (1 - hurt), 6), 3, 3)

        def draw_banner(self, painter):
            if self._banner is None:
                return
            text, t0 = self._banner
            age = self._wall.elapsed() - t0
            if age > 5000:
                self._banner = None
                return
            alpha = clamp(min(age / 300.0, (5000 - age) / 800.0), 0, 1)
            area = self._primary_local
            painter.setFont(self.banner_font)
            fm = QtGui.QFontMetrics(self.banner_font)
            headline = text.split(":")[0].split("!")[0].split(" - ")[0]
            if len(headline) > 40:
                headline = headline[:40]
            tw = fm.horizontalAdvance(headline)
            box = QtCore.QRectF(area.center().x() - tw / 2 - 24, area.y() + 70, tw + 48, 48)
            painter.setPen(QtCore.Qt.NoPen)
            painter.setBrush(QtGui.QColor(21, 22, 26, int(200 * alpha)))
            painter.drawRoundedRect(box, 12, 12)
            painter.setPen(QtGui.QColor(0xd9, 0x59, 0x26, int(255 * alpha)))
            painter.drawRect(QtCore.QRectF(box.x() + 12, box.bottom() - 4, box.width() - 24, 2))
            painter.setPen(QtGui.QColor(240, 239, 236, int(255 * alpha)))
            painter.drawText(box, QtCore.Qt.AlignCenter, headline)

        def draw_tool_icon(self, painter, name, cx, cy, active):
            ink = QtGui.QColor(240, 239, 236) if active else QtGui.QColor(195, 194, 183)
            painter.setBrush(QtCore.Qt.NoBrush)
            painter.setPen(QtGui.QPen(ink, 2, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin))
            if name == "follow":
                painter.drawEllipse(QtCore.QPointF(cx, cy), 8, 8)
                painter.setPen(QtGui.QPen(ink, 1.5, QtCore.Qt.DashLine))
                painter.drawEllipse(QtCore.QPointF(cx, cy), 12, 12)
            elif name == "food":
                painter.setPen(QtCore.Qt.NoPen)
                painter.setBrush(QtGui.QColor(140, 240, 160))
                for dx, dy in ((-6, 3), (5, 4), (0, -5)):
                    painter.drawEllipse(QtCore.QPointF(cx + dx, cy + dy), 3.5, 3.5)
            elif name == "lightning":
                painter.setPen(QtGui.QPen(QtGui.QColor(200, 215, 255), 2.5, QtCore.Qt.SolidLine,
                                          QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin))
                painter.drawPolyline(QtGui.QPolygonF([QtCore.QPointF(cx + 3, cy - 12), QtCore.QPointF(cx - 4, cy),
                                                      QtCore.QPointF(cx + 4, cy), QtCore.QPointF(cx - 3, cy + 12)]))
            elif name == "wall":
                painter.setPen(QtGui.QPen(QtGui.QColor(205, 215, 235), 6, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap))
                painter.drawLine(QtCore.QPointF(cx - 10, cy + 6), QtCore.QPointF(cx + 10, cy - 6))
            elif name == "predator":
                painter.setPen(QtCore.Qt.NoPen)
                painter.setBrush(QtGui.QColor(230, 90, 80))
                painter.drawPolygon(QtGui.QPolygonF([QtCore.QPointF(cx + 12, cy), QtCore.QPointF(cx, cy + 8),
                                                     QtCore.QPointF(cx - 10, cy), QtCore.QPointF(cx, cy - 8)]))
                painter.setBrush(QtGui.QColor(255, 255, 255))
                painter.drawEllipse(QtCore.QPointF(cx + 3, cy - 1), 2.5, 2.5)
            elif name == "bless":
                painter.setPen(QtGui.QPen(QtGui.QColor(255, 215, 120), 2))
                for i in range(4):
                    a = i * math.pi / 4
                    painter.drawLine(QtCore.QPointF(cx - math.cos(a) * 11, cy - math.sin(a) * 11),
                                     QtCore.QPointF(cx + math.cos(a) * 11, cy + math.sin(a) * 11))
            elif name == "meteor_shower":
                painter.setPen(QtGui.QPen(QtGui.QColor(255, 160, 80), 2.5, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap))
                painter.drawLine(QtCore.QPointF(cx - 10, cy - 10), QtCore.QPointF(cx + 4, cy + 4))
                painter.setPen(QtCore.Qt.NoPen)
                painter.setBrush(QtGui.QColor(255, 235, 190))
                painter.drawEllipse(QtCore.QPointF(cx + 5, cy + 5), 4.5, 4.5)
            elif name == "plague":
                painter.setPen(QtCore.Qt.NoPen)
                painter.setBrush(QtGui.QColor(170, 210, 60))
                for i in range(5):
                    a = i * math.tau / 5
                    painter.drawEllipse(QtCore.QPointF(cx + math.cos(a) * 8, cy + math.sin(a) * 8), 3, 3)
            elif name == "eclipse":
                painter.setPen(QtCore.Qt.NoPen)
                painter.setBrush(QtGui.QColor(255, 215, 120))
                painter.drawEllipse(QtCore.QPointF(cx, cy), 11, 11)
                painter.setBrush(QtGui.QColor(15, 16, 22))
                painter.drawEllipse(QtCore.QPointF(cx + 3, cy - 1), 10, 10)
            elif name == "weather":
                painter.setPen(QtGui.QPen(QtGui.QColor(215, 235, 255), 2, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap))
                for i in range(3):
                    a = i * math.pi / 3
                    painter.drawLine(QtCore.QPointF(cx - math.cos(a) * 10, cy - math.sin(a) * 10),
                                     QtCore.QPointF(cx + math.cos(a) * 10, cy + math.sin(a) * 10))
            elif name == "leviathan":
                painter.setPen(QtCore.Qt.NoPen)
                painter.setBrush(QtGui.QColor(40, 75, 120))
                painter.drawEllipse(QtCore.QPointF(cx - 2, cy), 12, 9)
                painter.setBrush(QtGui.QColor(120, 230, 255))
                painter.drawEllipse(QtCore.QPointF(cx + 5, cy - 2), 2, 2)

        def draw_toolbar(self, painter):
            """Clickable god-tool and world-event buttons along the bottom of the main screen."""
            area = self._primary_local
            bw, bh, gap, sep = 86.0, 54.0, 6.0, 22.0
            items = [("tool", k, label, hot) for k, label, hot in self.TOOLS] + \
                    [("event", k, label.split(" /")[0] if k != "weather" else
                      ("Drought" if self.world.season in ("spring", "summer") else "Blizzard"), hot)
                     for k, label, hot in self.EVENT_BUTTONS]
            total = len(items) * bw + (len(items) - 1) * gap + sep
            x = area.center().x() - total / 2
            y = area.y() + area.height() - bh - 14
            panel = QtGui.QPainterPath()
            panel.addRoundedRect(QtCore.QRectF(x - 10, y - 8, total + 20, bh + 16), 12, 12)
            painter.fillPath(panel, QtGui.QColor(21, 22, 26, 200))
            painter.setFont(self.name_font)
            for i, (kind, key, label, hot) in enumerate(items):
                if kind == "event" and items[i - 1][0] == "tool":
                    x += sep
                rect = QtCore.QRectF(x, y, bw, bh)
                if kind == "tool":
                    active = self.tool == key
                else:
                    active = self.world.event_active(
                        key if key != "weather" else ("drought" if self.world.season in ("spring", "summer")
                                                      else "blizzard"))
                painter.setPen(QtCore.Qt.NoPen)
                painter.setBrush(QtGui.QColor(44, 46, 54) if active else QtGui.QColor(30, 32, 38))
                painter.drawRoundedRect(rect, 8, 8)
                if active:
                    color = QtGui.QColor(0x90, 0x85, 0xe9) if kind == "tool" else QtGui.QColor(0xd9, 0x59, 0x26)
                    painter.setPen(QtGui.QPen(color, 2))
                    painter.setBrush(QtCore.Qt.NoBrush)
                    painter.drawRoundedRect(rect.adjusted(1, 1, -1, -1), 8, 8)
                self.draw_tool_icon(painter, key, x + bw / 2, y + 20, active)
                painter.setPen(QtGui.QColor(240, 239, 236) if active else QtGui.QColor(195, 194, 183))
                painter.drawText(QtCore.QRectF(x, y + 34, bw, 16), QtCore.Qt.AlignCenter, label)
                painter.setPen(QtGui.QColor(137, 135, 129))
                painter.drawText(QtCore.QRectF(x + 5, y + 3, 20, 12), QtCore.Qt.AlignLeft, hot)
                self._toolbar_hits.append((rect, (kind, key)))
                x += bw + gap

        # ------------------------------------------------------- stories UI
        KIND_COLORS = {"info": (0x39, 0x87, 0xe5), "danger": (0xe6, 0x67, 0x67), "life": (0xd5, 0x51, 0x81),
                       "season": (0x19, 0x9e, 0x70), "champion": (0xc9, 0x85, 0x00),
                       "followed": (0x90, 0x85, 0xe9), "event": (0xd9, 0x59, 0x26)}

        def draw_followed_marker(self, painter):
            f = self.world.followed
            if f is None or not f.alive:
                return
            pulse = 0.5 + 0.5 * math.sin(self._wall.elapsed() / 250.0)
            r = f.radius + 10 + 3 * pulse
            pen = QtGui.QPen(QtGui.QColor(255, 255, 255, int(150 + 80 * pulse)), 2, QtCore.Qt.DashLine)
            painter.setPen(pen)
            painter.setBrush(QtCore.Qt.NoBrush)
            painter.drawEllipse(QtCore.QPointF(f.x, f.y), r, r)
            painter.setFont(self.card_font)
            fm = QtGui.QFontMetrics(self.card_font)
            label = f.name
            tw = fm.horizontalAdvance(label)
            box = QtCore.QRectF(f.x - tw / 2 - 6, f.y - r - 24, tw + 12, 18)
            painter.setPen(QtCore.Qt.NoPen)
            painter.setBrush(QtGui.QColor(21, 22, 26, 210))
            painter.drawRoundedRect(box, 6, 6)
            painter.setPen(QtGui.QColor(240, 239, 236))
            painter.drawText(box, QtCore.Qt.AlignCenter, label)

        def draw_name_tags(self, painter):
            """When zoomed in, everyone gets a little name tag."""
            painter.setFont(self.name_font)
            painter.setPen(QtGui.QColor(195, 194, 183, 200))
            f = self.world.followed
            for o in self.world.orbs:
                if o.alive and o is not f:
                    painter.drawText(QtCore.QRectF(o.x - 50, o.y + o.radius + 6, 100, 14),
                                     QtCore.Qt.AlignCenter, o.name)

        def draw_follow_card(self, painter):
            o = self.world.followed
            INK, INK2, MUTED = (240, 239, 236), (195, 194, 183), (137, 135, 129)
            x, w = 10.0, 330.0
            y = self._hud_bottom + 10
            thinker = o.species in ("orb", "splitter", "mimic")
            lines_h = 214 if thinker else 150
            path = QtGui.QPainterPath()
            path.addRoundedRect(QtCore.QRectF(x, y, w, lines_h), 10, 10)
            painter.fillPath(path, QtGui.QColor(21, 22, 26, 225))
            painter.setPen(QtGui.QPen(QtGui.QColor(0x90, 0x85, 0xe9, 160), 1.5))
            painter.setBrush(QtCore.Qt.NoBrush)   # outline only (don't reuse a leftover brush)
            painter.drawPath(path)

            def text(tx, ty, s, color, font=None, align=QtCore.Qt.AlignLeft):
                painter.setFont(font or self.card_font)
                painter.setPen(qcolor(color))
                painter.drawText(QtCore.QRectF(tx, ty - 13, w - (tx - x) - 12, 18),
                                 align | QtCore.Qt.AlignVCenter, s)

            px = x + 12
            text(px, y + 22, "FOLLOWING", (0x90, 0x85, 0xe9), self.name_font)
            text(px, y + 22, ("[Z] camera on" if self.follow_cam else "[Z] follow camera")
                 if self.fullscreen else "", MUTED, self.name_font, QtCore.Qt.AlignRight)
            text(px, y + 42, o.name, INK, self.card_title_font)
            if o.species == "leviathan":
                text(px, y + 60, f"a leviathan from the deep  ·  health {max(0, o.energy):.0f}/"
                                 f"{LEVIATHAN_HEALTH:.0f}", INK2)
            else:
                text(px, y + 60, f"House {o.house}  ·  {o.species}  ·  generation {o.generation}", INK2)
            age_days = o.age / DAY_FRAMES
            if not o.parents:
                parents = "founder"
            elif len(o.parents) == 1:
                verb = {"splitter": "split from", "leech": "budded from"}.get(o.species, "hatched from")
                parents = f"{verb} {o.parents[0]}"
            else:
                parents = f"child of {o.parents[0]} & {o.parents[1]}"
            text(px, y + 78, f"{age_days:.1f} days old  ·  {parents}", INK2)
            alive_kids = sum(1 for c in o.children if c.alive)
            if o.species == "mimic":
                tally = f"disguise {o.disguise:.0%} - looks {o.apparent_radius / o.radius:.1f}x its size"
            elif o.species == "leech":
                tally = f"{o.hosts_bled} hosts bled  ·  {o.drained:.0f} energy drained"
            elif o.species == "scavenger":
                tally = f"{o.carrion_eaten} bites of carrion  ·  {o.food_eaten} food"
            else:
                tally = f"{o.hunting_success_count} kills  ·  {o.food_eaten} food"
            load = self.world.leech_load(o) if o.species != "leech" else 0
            if load:
                tally += f"  ·  {load} leech{'es' if load > 1 else ''}!"
            text(px, y + 96, f"{len(o.children)} children ({alive_kids} alive)  ·  {tally}", INK2)
            status = o.describe_activity(self.world)
            if o.species == "splitter" and o.radius >= o.size_gene * 0.85:
                status += "  (about to split!)"
            if o.infected:
                status += "  (sick with the plague)"
            elif o.immune:
                status += "  (immune to the plague)"
            text(px, y + 118, status[:1].upper() + status[1:], INK, self.card_font)

            # Energy and stamina bars
            def meter(mx, my, label, value, color):
                text(mx, my, label, MUTED, self.name_font)
                bx, bw = mx + 52, 90
                painter.setPen(QtCore.Qt.NoPen)
                painter.setBrush(QtGui.QColor(44, 44, 42))
                painter.drawRoundedRect(QtCore.QRectF(bx, my - 8, bw, 7), 3, 3)
                painter.setBrush(qcolor(color))
                painter.drawRoundedRect(QtCore.QRectF(bx, my - 8, bw * clamp(value, 0, 1), 7), 3, 3)
            meter(px, y + 138, "energy", o.energy / MAX_ENERGY, (0x39, 0x87, 0xe5))
            meter(px + 160, y + 138, "stamina", o.stamina, (0x19, 0x9e, 0x70))

            if not thinker:
                return   # anglers and leviathans run on instinct, not the utility brain
            # What its mind wants right now (utility of each behaviour).
            u = o.last_utilities or [0.0] * len(BEHAVIORS)
            text(px, y + 160, "What it wants right now", MUTED, self.name_font)
            col_w = (w - 24) / 3
            best = max(range(len(u)), key=lambda i: u[i])
            for i, b in enumerate(BEHAVIORS):
                cx_ = px + (i % 3) * col_w
                cy_ = y + 178 + (i // 3) * 18
                text(cx_, cy_, b, INK if i == best else MUTED, self.name_font)
                bx, bw = cx_ + 54, col_w - 60
                painter.setPen(QtCore.Qt.NoPen)
                painter.setBrush(QtGui.QColor(44, 44, 42))
                painter.drawRoundedRect(QtCore.QRectF(bx, cy_ - 8, bw, 6), 3, 3)
                painter.setBrush(QtGui.QColor(0x90, 0x85, 0xe9) if i == best else QtGui.QColor(110, 110, 120))
                painter.drawRoundedRect(QtCore.QRectF(bx, cy_ - 8, bw * clamp(u[i] / 1.5, 0, 1), 6), 3, 3)

        def draw_ticker(self, painter):
            if not self._ticker:
                return
            now = self._wall.elapsed()
            area = self._primary_local
            fm = QtGui.QFontMetrics(self.ticker_font)
            max_w = 560
            row = 22
            n = len(self._ticker)
            h = n * row + 12
            x = area.x() + 10
            y0 = area.y() + area.height() - h - (110 if self.fullscreen and self.show_toolbar else 50)
            path = QtGui.QPainterPath()
            path.addRoundedRect(QtCore.QRectF(x, y0, max_w + 40, h), 10, 10)
            painter.fillPath(path, QtGui.QColor(21, 22, 26, 150))
            painter.setFont(self.ticker_font)
            for i, (txt, kind, t0) in enumerate(self._ticker):
                age = now - t0
                alpha = clamp(min(age / 300.0, (15000 - age) / 2500.0), 0, 1)   # fade in & out
                ry = y0 + 6 + i * row
                color = self.KIND_COLORS.get(kind, self.KIND_COLORS["info"])
                painter.setPen(QtCore.Qt.NoPen)
                painter.setBrush(qcolor(color, 255 * alpha))
                painter.drawEllipse(QtCore.QPointF(x + 16, ry + row / 2), 4, 4)
                ink = (240, 239, 236) if kind == "followed" else (215, 214, 208)
                painter.setPen(qcolor(ink, 255 * alpha))
                painter.drawText(QtCore.QRectF(x + 28, ry, max_w, row), QtCore.Qt.AlignVCenter,
                                 fm.elidedText(txt, QtCore.Qt.ElideRight, max_w))

        def draw_food(self, painter):
            food = self.world.food
            if food is None:
                return
            painter.setPen(QtCore.Qt.NoPen)
            t = self.world.frame * 0.05
            fc = self.world.env["food_color"]
            night = 1.0 - self.world.light
            glow = qcolor(fc, 40)
            core = qcolor((min(255, fc[0] + 50), min(255, fc[1] + 30), min(255, fc[2] + 40)), 200)
            glow_alpha = 45 + 40 * night        # food glows softly in the dark
            for f in food.items:
                fade = min(1.0, f.age / 30.0, (FoodField.MAX_AGE - f.age) / 300.0)
                if fade <= 0:
                    continue
                pulse = 1.0 + 0.25 * math.sin(t + f.phase)
                r = (1.5 + f.value * 0.18) * pulse
                glow.setAlpha(int(glow_alpha * fade))
                painter.setBrush(glow)
                painter.drawEllipse(QtCore.QPointF(f.x, f.y), r * 2.6, r * 2.6)
                core.setAlpha(int(210 * fade))
                painter.setBrush(core)
                painter.drawEllipse(QtCore.QPointF(f.x, f.y), r, r)

        def draw_lures(self, painter):
            """Lures are drawn exactly like food - that's the point."""
            t = self.world.frame * 0.05
            for orb in self.world.orbs:
                lure = orb.lure if orb.alive and orb.is_angler else None
                if lure is None:
                    continue
                # Faint stalk from the angler's head to the bait.
                painter.setPen(QtGui.QPen(QtGui.QColor(90, 110, 140, 70), 1.2))
                painter.setBrush(QtCore.Qt.NoBrush)
                hx = orb.x + math.cos(orb.heading) * orb.radius * 0.8
                hy = orb.y + math.sin(orb.heading) * orb.radius * 0.8
                path = QtGui.QPainterPath(QtCore.QPointF(hx, hy))
                mx = (hx + lure.x) / 2 - math.sin(orb.heading) * 6
                my = (hy + lure.y) / 2 - 8
                path.quadTo(QtCore.QPointF(mx, my), QtCore.QPointF(lure.x, lure.y))
                painter.drawPath(path)
                pulse = 1.0 + 0.3 * math.sin(t * 1.3 + lure.phase)
                r = (1.5 + lure.value * 0.18) * pulse
                fc = self.world.env["food_color"]   # mimics the season's food
                night = 1.0 - self.world.light
                painter.setPen(QtCore.Qt.NoPen)
                painter.setBrush(qcolor(fc, 55 + 60 * night))
                painter.drawEllipse(QtCore.QPointF(lure.x, lure.y), r * (2.8 + night), r * (2.8 + night))
                painter.setBrush(qcolor((min(255, fc[0] + 60), min(255, fc[1] + 40), min(255, fc[2] + 50)), 230))
                painter.drawEllipse(QtCore.QPointF(lure.x, lure.y), r, r)

        def draw_angler(self, painter, orb):
            center = QtCore.QPointF(orb.x, orb.y)
            color = orb.current_color
            lurking = orb.behavior_state == "lurking"
            alpha = 120 if lurking else 235          # camouflaged while waiting
            if orb.behavior_state == "hunting":
                glow = QtGui.QRadialGradient(center, orb.radius * 2.2)
                glow.setColorAt(0, QtGui.QColor(255, 60, 40, 140))
                glow.setColorAt(1, QtGui.QColor(255, 60, 40, 0))
                painter.setPen(QtCore.Qt.NoPen)
                painter.setBrush(QtGui.QBrush(glow))
                painter.drawEllipse(center, orb.radius * 2.2, orb.radius * 2.2)
            body = QtGui.QRadialGradient(QtCore.QPointF(orb.x - orb.radius * 0.3, orb.y - orb.radius * 0.3),
                                         orb.radius * 1.4)
            body.setColorAt(0, qcolor((min(255, color[0] + 40), min(255, color[1] + 40), min(255, color[2] + 40)), alpha))
            body.setColorAt(1, qcolor((color[0] // 3, color[1] // 3, color[2] // 3), alpha))
            painter.setBrush(QtGui.QBrush(body))
            painter.setPen(QtGui.QPen(qcolor((color[0] // 2, color[1] // 2, color[2] // 2), alpha), 1))
            # Body as a pie: the missing wedge is the mouth, wide open mid-lunge.
            mouth = 70 if orb.behavior_state == "hunting" else 24 + 6 * math.sin(orb.age * 0.05)
            facing = -math.degrees(orb.heading)
            r = orb.radius
            painter.drawPie(QtCore.QRectF(orb.x - r, orb.y - r, r * 2, r * 2),
                            int((facing + mouth / 2) * 16), int((360 - mouth) * 16))
            # Teeth
            painter.setPen(QtGui.QPen(QtGui.QColor(235, 235, 220, alpha), 1.2))
            for side in (-1, 1):
                a = orb.heading + side * math.radians(mouth / 2)
                for k in (0.55, 0.8):
                    bx, by = orb.x + math.cos(a) * r * k, orb.y + math.sin(a) * r * k
                    painter.drawLine(QtCore.QPointF(bx, by),
                                     QtCore.QPointF(bx - side * math.sin(a) * 3 + math.cos(a) * 1.5,
                                                    by + side * math.cos(a) * 3 + math.sin(a) * 1.5))
            # Small beady eye above the mouth.
            ea = orb.heading - 0.9
            ex, ey = orb.x + math.cos(ea) * r * 0.5, orb.y + math.sin(ea) * r * 0.5
            painter.setPen(QtCore.Qt.NoPen)
            painter.setBrush(QtGui.QColor(255, 230, 120, 200 if lurking else 255))
            painter.drawEllipse(QtCore.QPointF(ex, ey), max(1.8, r * 0.14), max(1.8, r * 0.14))

        def draw_alarm_mark(self, painter, orb):
            fade = orb.alarm_flash / 30.0
            painter.setFont(self.alarm_font)
            painter.setPen(QtGui.QColor(255, 210, 60, int(255 * fade)))
            painter.drawText(QtCore.QPointF(orb.x - 3, orb.y - orb.radius - 10 - (1 - fade) * 8), "!")

        def draw_intent(self, painter):
            world = self.world
            painter.setBrush(QtCore.Qt.NoBrush)
            # Pack bonds
            for pack in world.packs:
                leader = pack.leader
                if not leader.alive:
                    continue
                pen = QtGui.QPen(QtGui.QColor(255, 200, 80, 70), 1.2)
                painter.setPen(pen)
                for m in pack.members:
                    if m is not leader and m.alive:
                        painter.drawLine(QtCore.QPointF(leader.x, leader.y), QtCore.QPointF(m.x, m.y))
            # Hunt & courtship lines
            hunt_pen = QtGui.QPen(QtGui.QColor(255, 60, 60, 90), 1.2, QtCore.Qt.DashLine)
            stalk_pen = QtGui.QPen(QtGui.QColor(255, 120, 60, 55), 1.0, QtCore.Qt.DotLine)
            mate_pen = QtGui.QPen(QtGui.QColor(255, 170, 220, 90), 1.2, QtCore.Qt.DotLine)
            for o in world.orbs:
                if not o.alive:
                    continue
                if o.behavior_state == "defending" and isinstance(o.target, IntelligentOrb) and o.target.alive:
                    painter.setPen(QtGui.QPen(QtGui.QColor(255, 160, 40, 110), 1.5, QtCore.Qt.DashLine))
                    painter.drawLine(QtCore.QPointF(o.x, o.y), QtCore.QPointF(o.target.x, o.target.y))
                if o.behavior_state == "hunting":
                    t = o.target
                    if isinstance(t, IntelligentOrb) and t.alive:
                        painter.setPen(stalk_pen if o.stalking else hunt_pen)
                        painter.drawLine(QtCore.QPointF(o.x, o.y), QtCore.QPointF(t.x, t.y))
                elif o.behavior_state == "mating" and o.mate_target is not None and o.mate_target.alive:
                    painter.setPen(mate_pen)
                    m = o.mate_target
                    painter.drawLine(QtCore.QPointF(o.x, o.y), QtCore.QPointF(m.x, m.y))
                if self.show_debug:
                    painter.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 25), 1))
                    painter.drawEllipse(QtCore.QPointF(o.x, o.y), o.vision, o.vision)

        def draw_trail(self, painter, orb):
            if len(orb.trail) < 2:
                return
            r, g, b = orb.current_color
            n = len(orb.trail)
            for i in range(1, n):
                a = int(120 * i / n)
                painter.setPen(QtGui.QPen(QtGui.QColor(r, g, b, a), max(1.0, orb.radius * 0.6 * i / n),
                                          QtCore.Qt.SolidLine, QtCore.Qt.RoundCap))
                x0, y0 = orb.trail[i - 1]
                x1, y1 = orb.trail[i]
                painter.drawLine(QtCore.QPointF(x0, y0), QtCore.QPointF(x1, y1))

        def shape_polygon(self, orb):
            x, y, r = orb.x, orb.y, orb.radius
            shape = orb.display_shape
            rot = orb.heading
            if shape == "diamond":
                pts = [(r * 1.25, 0), (0, r * 0.8), (-r, 0), (0, -r * 0.8)]
            elif shape == "hexagon":
                pts = [(r * math.cos(i * math.pi / 3), r * math.sin(i * math.pi / 3)) for i in range(6)]
                rot = 0
            elif shape == "triangle":
                pts = [(r * 1.2, 0), (-r * 0.7, r * 0.85), (-r * 0.7, -r * 0.85)]
            elif shape == "star":
                pts = []
                spin = orb.age * 0.01
                for i in range(10):
                    rr = r if i % 2 == 0 else r * 0.5
                    a = i * math.pi / 5 + spin
                    pts.append((rr * math.cos(a), rr * math.sin(a)))
                rot = 0
            elif shape == "square":
                pts = [(r, r), (-r, r), (-r, -r), (r, -r)]
            else:
                return None
            c, s = math.cos(rot), math.sin(rot)
            return QtGui.QPolygonF([QtCore.QPointF(x + px * c - py * s, y + px * s + py * c) for px, py in pts])

        def draw_orb(self, painter, orb):
            center = QtCore.QPointF(orb.x, orb.y)
            color = orb.current_color
            intensity = 0.3 + 0.7 * clamp(orb.energy / MAX_ENERGY, 0, 1)
            night = 1.0 - self.world.light

            # Glow - bioluminescence blooms at night (sleepers dim theirs).
            awake = orb.behavior_state != "resting"
            glow_radius = orb.radius * ((2.4 + 1.2 * night) if awake else 1.8)
            glow_boost = 1.0 + (0.6 * night if awake else -0.3 * night)
            gradient = QtGui.QRadialGradient(center, glow_radius)
            gradient.setColorAt(0.0, qcolor(color, 170 * intensity * glow_boost))
            gradient.setColorAt(0.4, qcolor(color, 110 * intensity * glow_boost))
            gradient.setColorAt(1.0, qcolor(color, 0))
            painter.setBrush(QtGui.QBrush(gradient))
            painter.setPen(QtCore.Qt.NoPen)
            painter.drawEllipse(center, glow_radius, glow_radius)

            # Body (night-adapted orbs get a faint pale rim so you can spot the night shift)
            if orb.nocturnal:
                painter.setPen(QtGui.QPen(QtGui.QColor(200, 210, 255, int(60 + 80 * night)), 1))
                painter.setBrush(QtCore.Qt.NoBrush)
                painter.drawEllipse(center, orb.radius + 1.5, orb.radius + 1.5)
                painter.setPen(QtCore.Qt.NoPen)
            core = QtGui.QRadialGradient(QtCore.QPointF(orb.x - orb.radius * 0.3, orb.y - orb.radius * 0.3),
                                         orb.radius * 1.3)
            core.setColorAt(0.0, qcolor((min(255, color[0] + 80), min(255, color[1] + 80), min(255, color[2] + 80))))
            core.setColorAt(0.6, qcolor(color))
            core.setColorAt(1.0, qcolor((color[0] // 2, color[1] // 2, color[2] // 2)))
            painter.setBrush(QtGui.QBrush(core))
            poly = self.shape_polygon(orb)
            if poly is None:
                painter.drawEllipse(center, orb.radius, orb.radius)
            else:
                painter.drawPolygon(poly)

            # Eye: looks where the orb is paying attention.
            look = orb.heading
            t = orb.target if orb.behavior_state == "hunting" else None
            if isinstance(t, IntelligentOrb):
                look = math.atan2(t.y - orb.y, t.x - orb.x)
            elif orb.behavior_state == "fleeing" and orb.percept and orb.percept.threats:
                _, _, tx, ty = max(orb.percept.threats, key=lambda th: th[1])
                look = math.atan2(ty - orb.y, tx - orb.x)
            ex = orb.x + math.cos(orb.heading) * orb.radius * 0.38
            ey = orb.y + math.sin(orb.heading) * orb.radius * 0.38
            eye_r = max(2.2, orb.radius * 0.28)
            if orb.behavior_state == "resting":
                painter.setPen(QtGui.QPen(QtGui.QColor(20, 20, 30, 200), max(1.0, eye_r * 0.4)))
                painter.drawLine(QtCore.QPointF(ex - eye_r, ey), QtCore.QPointF(ex + eye_r, ey))
                painter.setPen(QtCore.Qt.NoPen)
            else:
                painter.setBrush(QtGui.QColor(255, 255, 255, 230))
                painter.drawEllipse(QtCore.QPointF(ex, ey), eye_r, eye_r)
                painter.setBrush(QtGui.QColor(15, 15, 25, 240))
                pr = eye_r * (0.62 if orb.behavior_state == "fleeing" else 0.5)
                painter.drawEllipse(QtCore.QPointF(ex + math.cos(look) * eye_r * 0.4,
                                                   ey + math.sin(look) * eye_r * 0.4), pr, pr)

            # Behaviour ring
            ring = BEHAVIOR_RING.get(orb.behavior_state)
            if ring:
                pen = QtGui.QPen(QtGui.QColor(*ring), 2)
                if orb.stalking:
                    pen.setStyle(QtCore.Qt.DashLine)
                painter.setPen(pen)
                painter.setBrush(QtCore.Qt.NoBrush)
                painter.drawEllipse(center, orb.radius + 3, orb.radius + 3)

            # Energy arc
            rr = orb.radius + 6
            painter.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 70), 1.5))
            painter.setBrush(QtCore.Qt.NoBrush)
            span = int(-360 * 16 * clamp(orb.energy / MAX_ENERGY, 0, 1))
            painter.drawArc(QtCore.QRectF(orb.x - rr, orb.y - rr, rr * 2, rr * 2), 90 * 16, span)

        def draw_explosion_particles(self, painter, orb):
            painter.setPen(QtCore.Qt.NoPen)
            for particle in orb.explosion_particles:
                painter.setBrush(qcolor(particle.color, particle.get_alpha()))
                painter.drawEllipse(QtCore.QPointF(particle.x, particle.y), particle.size, particle.size)

        def draw_effects(self, painter):
            painter.setBrush(QtCore.Qt.NoBrush)
            for e in self.world.effects:
                pr = e.progress
                painter.setPen(QtGui.QPen(qcolor(e.color, 220 * (1 - pr)), 2.5 * (1 - pr) + 0.5))
                r = 6 + e.max_radius * pr
                painter.drawEllipse(QtCore.QPointF(e.x, e.y), r, r)

        SKY_TINT = {"spring": (0, 8, 4), "summer": (0, 6, 10), "autumn": (10, 4, 0), "winter": (4, 8, 16)}

        def draw_sky(self, painter):
            """Always-dark sky for full screen mode: deep night blue, slightly lighter
            slate by day, a warm band at dawn/dusk, twinkling stars after dark."""
            world = self.world
            w, h = self.screen_width, self.screen_height
            night = 1.0 - world.light
            tint = self.SKY_TINT[world.season]

            def mix(day, nite):
                return tuple(clamp(d + (n - d) * night + t, 0, 255) for d, n, t in zip(day, nite, tint))

            top = mix((16, 30, 46), (3, 5, 16))
            bottom = mix((10, 20, 30), (8, 10, 26))
            grad = QtGui.QLinearGradient(0, 0, 0, h)
            grad.setColorAt(0.0, qcolor(top))
            grad.setColorAt(1.0, qcolor(bottom))
            painter.fillRect(QtCore.QRectF(0, 0, w, h), QtGui.QBrush(grad))

            # Dawn / dusk glow along the bottom of the screen.
            twilight = clamp(1.0 - abs(world.light - 0.5) / 0.3, 0, 1)
            if twilight > 0:
                morning = world.time_of_day < 0.5
                glow_c = (255, 150, 90) if morning else (255, 100, 120)
                band = QtGui.QLinearGradient(0, h * 0.45, 0, h)
                band.setColorAt(0.0, qcolor(glow_c, 0))
                band.setColorAt(1.0, qcolor(glow_c, 55 * twilight))
                painter.fillRect(QtCore.QRectF(0, h * 0.45, w, h * 0.55), QtGui.QBrush(band))

            # Stars come out as it gets dark.
            star_alpha = clamp((0.55 - world.light) / 0.35, 0, 1)
            if star_alpha > 0:
                painter.setPen(QtCore.Qt.NoPen)
                t = world.frame * 0.02
                for sx, sy, size, phase in self.stars:
                    twinkle = 0.55 + 0.45 * math.sin(t + phase)
                    painter.setBrush(QtGui.QColor(225, 232, 255, int(200 * star_alpha * twinkle)))
                    painter.drawEllipse(QtCore.QPointF(sx, sy), size, size)

            # Soft vignette to frame the scene.
            vig = QtGui.QRadialGradient(w / 2, h / 2, max(w, h) * 0.7)
            vig.setColorAt(0.6, QtGui.QColor(0, 0, 0, 0))
            vig.setColorAt(1.0, QtGui.QColor(0, 0, 0, 110))
            painter.fillRect(QtCore.QRectF(0, 0, w, h), QtGui.QBrush(vig))

        def draw_sky_icon(self, painter, x, y):
            """Sun by day, crescent moon by night, tinted by season."""
            painter.setPen(QtCore.Qt.NoPen)
            if self.world.is_night:
                painter.setBrush(QtGui.QColor(230, 235, 255, 230))
                painter.drawEllipse(QtCore.QPointF(x, y), 7, 7)
                painter.setBrush(QtGui.QColor(10, 12, 24, 255))
                painter.drawEllipse(QtCore.QPointF(x + 3.5, y - 2.5), 6.5, 6.5)
            else:
                sun = (255, 215, 80) if self.world.season != "winter" else (255, 240, 190)
                painter.setPen(QtGui.QPen(qcolor(sun, 200), 1.5))
                for i in range(8):
                    a = i * math.pi / 4 + self.world.frame * 0.01
                    painter.drawLine(QtCore.QPointF(x + math.cos(a) * 8, y + math.sin(a) * 8),
                                     QtCore.QPointF(x + math.cos(a) * 11, y + math.sin(a) * 11))
                painter.setPen(QtCore.Qt.NoPen)
                painter.setBrush(qcolor(sun, 240))
                painter.drawEllipse(QtCore.QPointF(x, y), 6, 6)

        def draw_hud(self, painter):
            world = self.world
            orbs = world.alive_orbs()
            if not orbs:
                return
            behaviors = {}
            for o in orbs:
                behaviors[o.behavior_state] = behaviors.get(o.behavior_state, 0) + 1
            gens = [o.generation for o in orbs]
            afraid = sum(1 for o in orbs if o.cursor_affinity < -0.3)
            attracted = sum(1 for o in orbs if o.cursor_affinity > 0.3)
            best = max(orbs, key=lambda o: o.lifetime_fitness())
            c = world.counters

            night_shift = sum(1 for o in orbs if o.nocturnal)
            lines = [
                f"{world.season.title()}, day {world.day}   {world.clock_string()} "
                f"({'night' if world.is_night else 'day'})   Night-adapted {night_shift}   "
                f"Food x{world.env['food_rate']:.1f}  Cold x{world.env['cold']:.2f}",
                f"Orbs {len(orbs)}   Food {len(world.food.items) if world.food else 0}   "
                f"Packs {len(world.packs)}   {self.fps:4.0f} fps   "
                f"Speed {self.speed_label(self.speed_levels[self.speed_index])}"
                + (f" (running at {self.actual_speed:.1f}x)"
                   if self.speed_levels[self.speed_index] > 1 and
                   self.actual_speed < self.speed_levels[self.speed_index] * 0.85 else "")
                + ("   PAUSED" if self.paused else ""),
                "   ".join(f"{b} {behaviors.get(b, 0)}" for b in BEHAVIORS + ("lurking",)),
                f"Species: orbs {sum(1 for o in orbs if o.species == 'orb')}  splitters {sum(1 for o in orbs if o.species == 'splitter')}  "
                f"scavengers {sum(1 for o in orbs if o.species == 'scavenger')}  "
                f"leeches {sum(1 for o in orbs if o.species == 'leech')} "
                f"({sum(1 for o in orbs if o.species == 'leech' and o.host is not None)} feeding)   "
                f"mimics {sum(1 for o in orbs if o.species == 'mimic')}   "
                f"Splits {c['splits']}  Carrion eaten {c['carrion_eaten']}",
                f"Scent{' (C to show)' if not self.show_scent else ''}: green food trails, blue body scent, red danger   "
                f"Tracking {sum(1 for o in orbs if o.tracking)}  Following trails "
                f"{sum(1 for o in orbs if o.following_trail)}",
                f"Territories {len(world.territories)}   Intruders driven off {c['intruders_driven_off']}   "
                f"Border clashes {c['border_clashes']}   Mimics eaten {c['mimics_eaten']}",
                f"Anglers {sum(1 for o in orbs if o.is_angler)} ({c['angler_kills']} ambushes)   "
                f"Alarm calls {c['alarms']}   Windows as terrain {len(world.obstacles)}",
                f"Generation avg {sum(gens) / len(gens):.1f}  max {max(gens)}   "
                f"Gene pool {len(world.hall_of_fame)}",
                f"Kills {c['kills']}   Births {c['births']}   Starved {c['starved']}   "
                f"Old age {c['old_age']}   Mobbed off {c['repelled']}",
                f"Cursor: {afraid} afraid / {attracted} attracted   "
                f"Champion {best.full_name}, gen {best.generation}: {best.hunting_success_count} kills, "
                f"{best.offspring_count} kids",
            ]
            active = []
            for name in ("meteor_shower", "eclipse", "drought", "blizzard", "plague", "leviathan"):
                if world.event_active(name):
                    label = World.EVENT_NAMES[name]
                    if name == "plague":
                        label += f" ({sum(1 for o in orbs if o.infected)} sick)"
                    elif name == "leviathan":
                        label += f" {world.leviathan.name} ({world.leviathan.meals} swallowed)"
                    active.append(label)
            lines.insert(1, "World events: " + (", ".join(active) if active else "none right now") +
                         ("" if world.random_events else "   (random events off)"))
            if self.fullscreen:
                lines.append("Esc exit full screen | Space pause | [ ] speed | H HUD | S stats | E events | "
                             "W weather")
                lines.append("Click an orb to follow it | A champion | R random | Z follow camera "
                             "(wheel zooms) | X stop | N season | T +6h | F food")
                lines.append("God tools 1-6 (right-click follows) | World events F1-F5 | G toolbar | C scent")
            fm = QtGui.QFontMetrics(self.hud_font)
            width = max(fm.horizontalAdvance(s) for s in lines) + 24
            height = 30 + len(lines) * 18 + 8
            painter.setPen(QtCore.Qt.NoPen)
            painter.setBrush(QtGui.QColor(10, 12, 24, 150))
            painter.drawRoundedRect(QtCore.QRectF(10, 10, width, height), 8, 8)
            painter.setFont(self.hud_title_font)
            painter.setPen(QtGui.QColor(150, 210, 255, 230))
            painter.drawText(22, 32, "Intelligent Orbs")
            self.draw_sky_icon(painter, 10 + width - 24, 26)
            painter.setFont(self.hud_font)
            painter.setPen(QtGui.QColor(235, 235, 245, 210))
            for i, s in enumerate(lines):
                painter.drawText(22, 52 + i * 18, s)
            self._hud_bottom = 10 + height


def main():
    if not HAVE_QT:
        print("PyQt5 is required to run the overlay: pip install PyQt5 numpy")
        sys.exit(1)
    signal.signal(signal.SIGINT, signal.SIG_DFL)  # Ctrl+C in the console quits
    app = QtWidgets.QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)
    overlay = IntelligentOrbOverlay()
    app.aboutToQuit.connect(overlay.shutdown)
    overlay.show()
    if "--fullscreen" in sys.argv[1:] or "-f" in sys.argv[1:]:
        overlay.set_fullscreen(True)
    if "--stats" in sys.argv[1:] or "-s" in sys.argv[1:]:
        overlay.set_stats(True)
    print("Intelligent Orbs started!")
    print("They sense, think, hunt in packs, flee, forage, mate and evolve.")
    print("Full screen mode: tray menu or --fullscreen (Esc to return to the desktop overlay).")
    if overlay.tray:
        print("Right-click the tray icon for controls (pause, HUD, feed, quit).")
    else:
        print("Press Ctrl+C here to quit.")
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
