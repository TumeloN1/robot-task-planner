# Robot Task Planner with Self-Correcting Execution — Project Report

_Status: MVP scaffold, full closed loop running on ground-truth perception._
_Planner LLM: `gemma-4-31b-it` (Google Gemini API). Simulator: custom MuJoCo + Franka Panda._

This document is a from-the-top account of the project: what we are building and
why, the design decisions and their motivations, every failure point we hit and
how it was addressed, and a concrete plan for building on the progress so far.

---

## 1. Executive summary

We built the scaffold and full closed loop for a simulated tabletop manipulation
agent that turns a natural-language instruction (e.g. _"Put the red mug on the
tray"_) into an executed, self-correcting plan. The pipeline is:

> **perceive → plan (LLM) → validate → execute + temporal postcondition check →
> on failure: diagnose → escalate (rule-based recovery → LLM replan → abort) →
> record episode.**

The research-interesting behavior is **not** generating the plan; it is what the
system does when a grasp slips, a target is blocked, or perception confuses two
objects. Everything is structured around making that recovery loop robust and,
later, learnable.

As of this report:

- The control stack (IK, arm, gripper, feasibility) works end-to-end.
- The agent completes a live episode with **Gemma 4 31B** as the planner
  (`run_episode.py` → SUCCESS, 0 replans).
- Batch evaluation with failure injection runs and produces metrics, isolating
  transient API errors per-episode.
- 13 unit/integration tests pass; `ruff` is clean.

---

## 2. Project goal and research thesis

The motivating interest is **using an LLM as an expert agent for online,
self-correcting execution** — closing the loop between high-level reasoning and
low-level control, and eventually _learning_ from the failures the loop
surfaces.

The MVP deliberately scopes to:

- **Closed loop with ground-truth perception** (no vision model yet), so we can
  exercise planning/replanning logic without fighting perception noise first.
- **Self-correction without learning** — diagnosis and replanning are
  rule/LLM-driven for now; episodes are logged in a learning-ready format so a
  behavior-cloning policy or learned failure model can come later.
- **Local development on macOS**, with a clean path to Colab/GPU for the
  vision/learning phase (sim stays headless via `MUJOCO_GL=egl`).

---

## 3. System architecture (closed loop)

```
instruction
   -> perceive    SceneState: only visible objects + last-seen memory
   -> plan        LLM -> JSON {rationale, tool_calls[]}
   -> validate    schema/grounding -> IK feasibility -> loop detection
   -> execute     per step: run primitive -> temporal postcondition check
        on failure -> escalate:
            (1) rule-based recovery (one immediate grasp retry)
            (2) LLM replan with a concise FailureContext (+ must_change directive)
            (3) abort
   -> record      HDF5 (robomimic-style) with explicit failure labels
```

The loop is implemented in `src/rtp/agent/loop.py`; failure classification and
the replan context in `src/rtp/agent/diagnosis.py`.

---

## 4. Repository layout

```
configs/      sim / planner / tasks / run YAML configs
assets/       MJCF scene + object snippets; vendored Panda model (fetched, not committed)
src/rtp/
  sim/        MjSpec scene assembly + MuJoCoScene wrapper
  control/    IK (mink + DLS fallback), arm controller, gripper, feasibility
  perception/ vision-ready API + ground-truth provider + noise wrapper
  primitives/ registry + motion / manipulation / queries / checks
  planner/    schema, prompt builder, LLM client, validator, scripted planner
  agent/      execution context, diagnosis, the closed loop, build factory, state
  tasks/      predicates, scene/task generator, failure injection
  data/        HDF5 recorder + record schema
  eval/       metrics
scripts/      runnable entrypoints + asset fetch
tests/        unit + integration tests
data/         generated scenes + episode logs (gitignored)
```

---

## 5. Key design decisions and their motivations

### 5.1 LLM backend: Gemini → Gemma 4 31B

- **Decision.** Start on the Gemini API with `gemini-2.5-flash`; later switch the
  planner model to `gemma-4-31b-it`.
- **Motivation.** Gemma 4 31B offers materially better rate limits, a usable
  context window, and — crucially for the research direction — **open weights we
  can later fine-tune** for our tool-calling/recovery distribution. Both models
  are reachable through the same `google-genai` client, so the swap is a config
  change rather than an integration.

### 5.2 Output mode: JSON-schema → JSON-mime + prompt-described shape + robust parse

- **Decision.** Do **not** rely on `response_json_schema`. Use
  `response_mime_type="application/json"`, describe the exact output shape in the
  prompt (`OUTPUT FORMAT` block), and parse robustly (strip code fences →
  `json.loads` → Pydantic validate, with one repair retry).
- **Motivation.** Gemma does not support server-side JSON-schema constraint — it
  returns `504 DEADLINE_EXCEEDED` (see §6). JSON-mime + prompt shaping + parsing
  is portable across Gemma and Gemini and removed the single worst hang in the
  system. The `Plan` Pydantic model remains the source of truth for parsing/
  validation client-side.

### 5.3 Simulator: custom lightweight MuJoCo scene via `MjSpec`

- **Decision.** Assemble scenes programmatically with `mujoco.MjSpec` (load the
  Panda MJCF, add `grip_site`, floor, table, camera, and object bodies with free
  joints) rather than authoring monolithic XML or using a heavyweight env.
- **Motivation.** Programmatic assembly makes procedural scene/task generation
  trivial and keeps the dependency surface tiny. It also sidestepped real
  limitations of static-XML `include` composition and let us drop the Panda's
  default `home` keyframe cleanly and set the home pose ourselves.

### 5.4 IK: `mink` primary, DLS fallback, top-down constraint

- **Decision.** Use `mink` (QP-based, respects joint limits via posture + frame
  tasks) as the primary solver; ship a dependency-light damped-least-squares
  Jacobian fallback for environments without a QP backend. Constrain MVP grasps
  to a fixed **top-down** orientation.
- **Motivation.** A hand-rolled DLS solver is a classic time sink — it fights
  singularities, joint limits, and local minima. Delegating to `mink` avoids
  weeks of damping-factor tuning; the top-down constraint makes IK well-behaved
  and is sufficient for tabletop pick-and-place. The fallback keeps the package
  runnable if `mink`/`daqp` are unavailable.

### 5.5 Partial observability baked in from day one

- **Decision.** `ObjectState` carries `visible`, `confidence`, and
  `last_seen_step`. The ground-truth provider uses `mujoco.mj_ray` to mark
  occluded objects `visible=False` and keeps a `remembered_objects` memory.
- **Motivation.** If the planner assumes global omniscient knowledge, swapping in
  a real vision model later (with occlusion and partial views) would break the
  planner, not just the perception module. Forcing the model to use
  `find_object` / `look_around` now means the eventual vision swap is a
  drop-in replacement behind a stable contract.

### 5.6 Temporal postcondition checkers

- **Decision.** A grasp/place is only "successful" if the predicate holds for a
  settle window (~0.5 s of sim time), via `stable_for` in `primitives/checks.py`.
- **Motivation.** A single-frame check can register success the instant before an
  object dynamically slips out of the gripper. Temporal filtering catches these
  transient/dynamic failures that are exactly the cases the recovery loop exists
  to handle.

### 5.7 Anti-loop replanning

- **Decision.** Three mutually reinforcing mechanisms:
  1. A **required `rationale`** field on every plan.
  2. A **concise `FailureContext`** (`attempted` / `result` / `prior_attempts` /
     a `must_change` directive) — never raw state dumps.
  3. **Validator-side loop detection** that rejects a replan whose tool-call
     signature is identical to one that already failed.
- **Motivation.** LLMs are prone to apologizing and re-emitting a near-identical
  failing plan, and stacked raw failure logs dilute attention. Compact, directive
  context plus a hard structural reject forces the model to actually change
  approach.

### 5.8 Layered, cheap-first validation

- **Decision.** `validate_plan` runs schema/grounding (tool exists, args match,
  object refs resolve to a visible/known/locatable object), then **IK
  feasibility** (grasp/approach poses reachable + collision-free on a cloned
  `mjData`), then **loop detection**.
- **Motivation.** A structurally valid plan can still be physically impossible
  (grasp inside the table, unreachable pose). Rejecting it _before_ moving the arm
  prevents corrupting the simulation state and ruining the episode log. Ordering
  checks cheap-first avoids paying for IK on a plan that fails grounding.

### 5.9 Escalation ladder for failures

- **Decision.** On a step failure: (1) one immediate rule-based grasp retry, (2)
  LLM replan with `FailureContext`, (3) abort. Total replans are capped
  (`max_replans=3`).
- **Motivation.** Cheap deterministic recovery handles the common transient
  (a marginal grasp) without an API round-trip; the LLM is reserved for genuine
  re-strategizing; the cap bounds cost and guarantees termination.

### 5.10 Procedural failure injection

- **Decision.** `tasks/failures.py` injects controlled failures (grasp slip,
  perception confusion, target blocked, actuation noise) at configurable rates.
- **Motivation.** We need to _exercise_ the recovery loop deterministically and
  generate labeled failure/recovery data, rather than waiting for organic
  failures. Injection rates and seeds make experiments reproducible.

### 5.11 Learning-ready data logging

- **Decision.** `data/recorder.py` writes one HDF5 group per episode
  (`data/demo_{i}`, robomimic-style) with per-step tool/args/success/
  postcondition + an explicit `failure_label`, and episode attrs (instruction,
  success, num_replans, injected_failures, seed).
- **Motivation.** The MVP has no learning, but every episode is captured in a
  format a behavior-cloning pipeline or learned failure model can consume
  directly later — the failure labels are the supervision signal for the
  online-learning thesis.

### 5.12 Offline `ScriptedPlanner`

- **Decision.** A `ScriptedPlanner` implements the same `propose` interface as the
  LLM planner but returns a fixed/callable plan.
- **Motivation.** Lets us test the entire agent loop deterministically and
  offline (no API key, no latency, no flakiness), which decouples loop
  development from LLM availability.

### 5.13 Environment & config hygiene

- **Decision.** `MUJOCO_GL` auto-set (`glfw` on macOS, `egl` on Linux/Colab);
  `.env` loaded with an explicit path; planner settings centralized in
  `configs/planner.yaml` and read by scripts via `planner_kwargs()`.
- **Motivation.** Headless rendering differs by platform; explicit paths avoid
  frame-inference bugs; centralized config means model/temperature/timeout
  changes are one-line edits, not code changes.

---

## 6. Failure points encountered and how they were addressed

Ordered roughly chronologically, from scaffolding through the live closed loop.

| # | Symptom | Root cause | Fix applied |
|---|---------|-----------|-------------|
| 1 | `AssertionError` in `dotenv.find_dotenv` | `.env` auto-discovery can't infer the calling frame when run from a heredoc/`-c` | Pass an explicit `dotenv_path` rooted at the repo in `scripts/_env.py` |
| 2 | `ValueError: no signature found for builtin` | `inspect.signature` on pybind11-wrapped `MjSpec` methods isn't introspectable | Stopped introspecting; used MuJoCo docs for signatures (no functional impact) |
| 3 | Headless render failures across platforms | MuJoCo needs a platform-specific GL backend | Auto-set `MUJOCO_GL` (glfw/macOS, egl/Linux) in `_env.py` |
| 4 | Static-XML `include` / Panda `home` keyframe friction | Monolithic-XML composition is brittle; menagerie ships a keyframe that fought our home pose | Switched to programmatic `MjSpec` assembly; removed the default keyframe and set home explicitly |
| 5 | `ruff` `E501` (long lines), `F401` (unused `numpy`) | Style/lint drift | `ruff check --fix` + manual reflows; removed unused import |
| 6 | `ModuleNotFoundError: _env_check` | Typo'd import in a sanity heredoc | Corrected to the real package path `rtp` |
| 7 | `gh auth login` blocked | Interactive browser auth can't run from the agent | Used the device-code flow; user completed auth in their browser |
| 8 | **194s hang** in `smoke_gemini` | **Gemma does not support `response_json_schema` → `504 DEADLINE_EXCEEDED`** | Switched planner to JSON-mime mode + prompt-described shape + robust parse (§5.2) |
| 9 | Valid plan rejected: "ungroundable object `red_mug`" | Model used the object **id** (`red_mug`); grounding only matched display names and didn't normalize underscores | Grounding/resolver now accept ids and treat `_`/space equivalently (`_norm`) |
| 10 | `BlockingIOError: errno 35` opening `episodes.hdf5` | A killed eval process still held the HDF5 file lock | Killed the stale interpreter; removed the stale file (regenerated each run) |
| 11 | Stale process survived the kill | The pipeline PID (the part before `tail`) differs from the Python interpreter PID | Killed the actual interpreter PID; avoid piping long runs through `tail` |
| 12 | Eval took >10 min for a few episodes | Intermittent Gemma `504`s **compounded** by the SDK's own retries; output hidden by `tail` buffering | Disabled SDK retries (`HttpRetryOptions(attempts=1)`), bounded app-level backoff on transient 5xx/429, fail-fast 30s timeout; run with `python -u` |
| 13 | One bad API call aborted the whole batch | No per-episode error isolation in `evaluate.py` | Wrapped each episode in try/except → record as failed and continue |
| 14 | **`src/rtp/data/` package untracked by git** | `.gitignore` rule `data/` (unanchored) also matched the nested package dir → a fresh clone can't import `rtp.data.recorder` | Anchored the rule to `/data/` so only the top-level artifacts dir is ignored; package is now trackable (found and fixed while writing this report) |

The standout root cause was **#8**: Gemma silently times out on schema-mode
requests. Probing the three request modes made it obvious — plain (1.9s) and
JSON-mime (3.8s) succeed; schema mode 504s at ~25s. That single finding drove the
output-mode redesign and the retry/timeout hardening.

---

## 7. Hardening recommendations (making the system more resistant)

Concrete, mostly small changes that would harden the codebase against the classes
of issues above.

### LLM / API resilience
- **Circuit-breaker + provider fallback.** If `gemma-4-31b-it` returns repeated
  5xx within an episode, fall back to a configured secondary model
  (`gemma-4-26b-a4b-it` or a Gemini model) instead of failing the episode.
- **Cache plans by (instruction, scene signature).** Avoids re-calling the API
  for identical states across a batch and during development.
- **Schema-mode capability probe at startup.** Detect per-model whether
  `response_json_schema` works and pick the request mode automatically, so a
  future model that _does_ support it benefits without manual config.
- **Token/latency budget per episode.** Hard wall-clock cap so a slow provider
  can't stall a batch indefinitely.

### Data / recorder robustness
- **Always close the recorder.** `evaluate.py` constructs `EpisodeRecorder`
  directly and never calls `close()`; use it as a context manager (or wrap the
  batch in `with EpisodeRecorder(...) as rec:`). This is the real fix for the
  HDF5-lock class of issue (#10).
- **Open HDF5 with `libver="latest"` + explicit locking control**, and write to a
  temp path then atomically rename, so a crashed run can't leave a half-locked
  file in place.

### Validation depth
- **Explicit AABB / workspace sanity layer.** Today, out-of-workspace and
  in-table poses are caught implicitly by IK feasibility. An explicit, cheaper
  AABB/workspace-bounds check (using `configs/sim.yaml` limits) before IK would
  reject obviously-bad coordinates faster and give clearer diagnostics.
- **Validate placement targets too** (not just grasp/approach) for reachability.

### Process / tooling hygiene
- **Never pipe long-running jobs through `tail`** (it buffers until exit); prefer
  `python -u` and read the streamed file. Capture the interpreter PID, not the
  pipeline PID, for clean shutdown.
- **CI smoke that imports from a clean checkout.** A `pip install -e .` + import
  test in CI would have caught the untracked `src/rtp/data/` package (#14)
  immediately. Add a fresh-clone import test.
- **Pin a `requirements.lock.txt`** (currently gitignored/regenerated) for
  reproducible environments across machines and Colab.

### Config / docs
- Update `README.md` setup to reference `GEMMA`/`gemma-4-31b-it` and
  `configs/planner.yaml` as the source of truth (it still says
  `GEMINI_API_KEY`/`gemini-2.5-flash` in places).

---

## 8. How to build further

Sequenced from "lock in the MVP" to the original research thesis.

### Phase A — solidify the MVP
1. **Commit the gitignore fix and the `src/rtp/data/` package** (currently
   untracked — see #14). Add the fresh-clone import CI check.
2. **Recorder-as-context-manager** in `evaluate.py`; add the provider fallback
   and per-episode wall-clock cap.
3. **Implement `scripts/replay_episode.py`** (currently stubbed) to re-render a
   logged episode for debugging recoveries.
4. **Larger eval sweep** once API resilience lands; track the metrics in
   `eval/metrics.py` (task success, recovery success rate, mean replans,
   recovery-by-failure-type) over seeds.

### Phase B — perception realism
5. **Swap ground truth for vision** behind the existing `PerceptionProvider`
   contract: start with Lab-1 geometry + Open3D, then SAM / GroundingDINO. Because
   partial observability (`visible`/`confidence`/`last_seen_step`) is already
   enforced, the planner should not need changes.
6. **Calibrate the noise model** (`perception/noise.py`) against the real vision
   provider's error modes (label swaps, pose jitter).

### Phase C — learning (the thesis)
7. **Train a learned failure model** on the logged HDF5: predict failure type
   from pre-failure state, to pre-empt failures and to inform replanning.
8. **Behavior-cloning primitives** from successful trajectories to replace
   scripted motion where it helps, keeping the LLM as the high-level planner.
9. **Fine-tune Gemma** on our tool-calling + recovery distribution (the original
   motivation for choosing open weights) — start with LoRA on the logged
   plan/replan pairs.
10. **GPU/Colab path**: keep the sim headless (`MUJOCO_GL=egl`) and move
    vision/learning to Colab; the SDK/agent code is already portable.

### Phase D — scale & rigor
11. **Expand the task/predicate space** (more relations, multi-object, ordering
    constraints) in `tasks/generator.py` and `tasks/predicates.py`.
12. **Ablations**: quantify the contribution of each anti-loop mechanism (§5.7),
    the temporal checker window, and the escalation ladder rungs.

---

## 9. Current status snapshot

- **Working end-to-end:** sim, control (IK/arm/gripper/feasibility), perception
  (ground truth + occlusion), primitives, planner (Gemma, JSON-mime + parse),
  validator, closed loop with escalation, task/failure generation, HDF5 logging,
  metrics.
- **Verified live:** single episode SUCCESS with Gemma; resilient batch eval with
  failure injection and metrics.
- **Quality gates:** 13 tests pass; `ruff` clean.
- **Known open items:** commit `src/rtp/data/` (post-gitignore-fix); recorder
  close in eval; `replay_episode.py` stub; README model references; Gemma `504`
  flakiness mitigated but provider-side.
