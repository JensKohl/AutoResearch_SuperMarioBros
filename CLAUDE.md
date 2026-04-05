# AutoResearch for Super Mario

## Setup
To set up a new experiment, work with the user to:

1. Agree on a run tag: propose a tag based on today's date and time (e.g. April4_1600). The branch autoresearch/<tag> must not already exist — this is a fresh run.
2. Create the branch: git checkout -b autoresearch/<tag> from current master.
3. Read the in-scope files: The repo is small. Read these files for full context:
    - README.md — repository context.
    - train.py — the file you modify. Model architecture, optimizer, training loop.
    - evaluate.py - file to evaluate the training. Do not modify.
4. Verify data exists: Check that ROMS contains game files (*.nes, *.smc). If not, tell the human to put game files into the folder.
5. Initialize results.tsv: Create results.tsv with just the header row. The baseline will be recorded after the first run.
6. Confirm and go: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on a single GPU. The training script runs for a fixed time budget of 5 minutes (wall clock training time, excluding startup/compilation). You launch it simply as: 
```uv run --no-sync src/train.py```.

**What you CAN do:**

- Modify train.py — this is the only file you edit. Everything is fair game: model architecture, optimizer, reward functions, hyperparameters, training loop, batch size, model size, etc.

**What you CANNOT do:**

- Modify evaluate.py. It is read-only. It contains the test run.
- Modify constants.py. It contains training constants (TIME_BUDGET, MAX_EPISODE_STEPS, PRO_MOVEMENT, EVAL_SEEDS).
- Install new packages or add dependencies. You can only use what's already in pyproject.toml.
- Modify the evaluation harness. The defined reward in evaluate.py is the ground truth metric.
- Do not modify README.md.
- Do not modify AGENTS.md.
- Do not modify CLAUDE.md.

**The goal is simple**: maximize `total_reward` as computed by `evaluate.py`:
- If the agent finishes the level (`flag_get=True`): `10000 + (400 - seconds_to_finish) + score + max_x_dist`
- Otherwise: `score + max_x_dist`

Finishing the level is the dominant objective (10000 bonus). Among runs that finish, faster + higher score + further right is better. Among runs that don't finish, score and distance count equally. Everything is fair game: change the architecture, use other Reinforcement learning algorithms you can find in the Internet, use frame stacking for images, the optimizer, the hyperparameters, the batch size, the model size, the reward functions, the rewards. The only constraint is that the code runs without crashing and finishes within the time budget.

Document your changes in a file called CHANGES.MD, so people can later read and understand what you did. Be sure to explain your changes in that file so readers understand why you did the changes. Be aware that the reader might not have a lot of knowledge, so explain things clearly. But be concise: add references or links instead of very long explanations.

**VRAM** is a soft constraint. Some increase is acceptable for meaningful reward gains, but it should not blow up dramatically.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude. A 0.001 reward improvement that adds 20 lines of hacky code? Probably not worth it. A 0.001 reward improvement from deleting code? Definitely keep. An improvement of ~0 but much simpler code? Keep.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as is.

**Output format**
Once both scripts finish, the log contains lines like:
```
training_seconds: 300
total_seconds: 300
peak_vram_mb: 4500.0
train_best_reward: 850.0
train_seconds_to_finish: 999.0
train_best_score: 0
mean_episode_reward: 42.3
flag_get: False
max_x_dist: 314
score: 200
seconds_to_finish: 0.0
total_reward: 514.0
```

The key decision metric is `total_reward` (from evaluate.py). Note that the script is configured to always stop after 5 minutes, so numbers will vary by machine.

## Logging results

When an experiment is done, log it to results.tsv (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 8 columns. You write each row yourself after evaluating the results:

commit	seconds_to_finish	total_reward	flag_get	max_x_dist	score	status	description

1. git commit hash (short, 7 chars)
2. seconds_to_finish: seconds to finish the level (0 if not finished)
3. total_reward: the primary optimization target
4. flag_get: True/False — whether the agent finished the level
5. max_x_dist: furthest x position reached
6. score: in-game score achieved
7. status: `keep`, `discard`, or `crash` — update this manually after reviewing
8. description: short text of what this experiment tried — update this manually

Example:
```
commit	seconds_to_finish	total_reward	flag_get	max_x_dist	score	status	description
aff34	0.0	514.0	False	514	0	keep	baseline
a35gfd	44.0	11156.0	True	512	200	keep	optimized reward function
```

## The experiment loop
The experiment runs on a dedicated branch (e.g. ``autoresearch/mar5`` or ``autoresearch/mar5-gpu0``).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune train.py with an experimental idea by directly hacking the code.
<<<<<<< HEAD
3. Update CHANGES.MD: append a section describing what you changed and why.
4. git commit (include both train.py and CHANGES.MD)
5. Run training: ```uv run --no-sync src/train.py > run.log 2>&1``` (redirect everything — do NOT use tee or let output flood your context)
6. Run evaluation: ```uv run --no-sync src/evaluate.py >> run.log 2>&1``` (appends to same log)
7. Read out the results: grep "^total_reward:\|^flag_get:\|^max_x_dist:\|^score:\|^seconds_to_finish:\|^peak_vram_mb:" run.log
8. If the grep output is empty, the run crashed. Run tail -n 50 run.log to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
9. Append a row to results.tsv with the results and your decision. Use the git short hash, the metrics from the log, your status decision, and a short description of what you changed. (NOTE: do not commit results.tsv — leave it untracked by git)
10. If total_reward improved (higher value), you "advance" the branch, keeping the git commit.
<<<<<<< HEAD
11. If total_reward is equal or worse, run ```git checkout src/train.py CHANGES.MD``` to revert only the experiment files, then go back to step 1.
12. Sleep 60 seconds to let the GPU cool down: ```sleep 60```

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

**Timeout**: Each experiment should take ~5 minutes total (+ a few seconds for startup and eval overhead). If a run exceeds 10 minutes, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working indefinitely until you are manually stopped. You are autonomous. If you run out of ideas, think harder — read papers referenced in the code, re-read the in-scope files for new angles, search the web for new ideas, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. If each experiment takes you ~5 minutes then you can run approx 12/hour, for a total of about 100 over the duration of the average human sleep. The user then wakes up to experimental results, all completed by you while they slept!