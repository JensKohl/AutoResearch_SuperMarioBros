TIME_BUDGET = 600 # seconds constraints for training (10 minutes)
MAX_EPISODE_STEPS = 3000 # maximum steps per episode to prevent getting completely stuck
EVAL_SEEDS = [42, 123, 456] # Deterministic evaluation seeds

PRO_MOVEMENT = [
    ['right'],
    ['right', 'A'],
    ['right', 'B'],
    ['right', 'A', 'B'],
    ['A'],
    ['A', 'B']
]
