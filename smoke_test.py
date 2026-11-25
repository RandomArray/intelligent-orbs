# Simple headless smoke test for Intelligent Orb logic
import math
import random
from statistics import mean

from intelligent_orbs import IntelligentOrb

WIDTH, HEIGHT = 800, 600

# Create a small population
orbs = [IntelligentOrb(random.uniform(50, WIDTH-50), random.uniform(50, HEIGHT-50), WIDTH, HEIGHT) for _ in range(12)]

# Simulate 200 ticks with a cursor moving in a circle
for t in range(200):
    ang = (t / 200.0) * 2 * math.pi
    cursor = (WIDTH/2 + 200 * math.cos(ang), HEIGHT/2 + 150 * math.sin(ang))
    # Update all orbs
    for o in orbs[:]:
        o.update(cursor, orbs, dt=1.0)
        if o.behavior_state == "dying" and not o.is_exploding and o.energy <= 0:
            # prune once it has finished explosion in the main app; here remove to keep simulation stable
            orbs.remove(o)

# Print basic stats
if orbs:
    print({
        'count': len(orbs),
        'avg_energy': round(mean(o.energy for o in orbs), 2),
        'avg_age': round(mean(o.age for o in orbs), 1),
        'behaviors': {b: sum(1 for o in orbs if o.behavior_state == b) for b in ['exploring','hunting','fleeing','mating','dying']},
        'avg_aggr': round(mean(o.aggression for o in orbs), 3),
        'avg_social': round(mean(o.social_tendency for o in orbs), 3),
        'avg_curiosity': round(mean(o.curiosity for o in orbs), 3),
    })
else:
    print({'count': 0})
