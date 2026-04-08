TIME_BUDGET = 300 # seconds constraints for training (5 minutes)
MAX_EPISODE_STEPS = 2000 # maximum steps per episode to prevent getting completely stuck
EVAL_SEEDS = [42] # Deterministic evaluation seeds

PRO_MOVEMENT = [
    ['right'],
    ['right', 'A'],
    ['right', 'B'],
    ['right', 'A', 'B'],
    ['A'],
    ['A', 'B']
]
