# Planner-Worker Overcooked Agent

This document explains the current planner-worker architecture used in this repository for Overcooked, how state is represented for the LLMs, what tools each agent can use, what prompt optimizations were added, which problems were encountered during development, and how those problems were mitigated. It is intended as a readable technical explanation rather than a slide deck.

## 1. Why This Architecture Exists

The core problem in Overcooked is not just picking an action every timestep. The harder part is coordinating two chefs under shared constraints:

- both chefs can block each other
- both chefs can redundantly chase the same object
- soup preparation has strict sequencing constraints
- layout topology changes what good coordination looks like

A single end-to-end LLM agent can sometimes act reasonably, but it does not naturally produce robust multi-agent coordination. Two independent worker LLMs can act in parallel, but without a centralized coordinator they tend to duplicate effort, interfere with each other, or leave critical tasks undone.

The planner-worker split addresses that. The system is structured as:

- one centralized planner LLM that reasons about the whole kitchen and assigns work
- two worker LLMs, one per chef, that execute those assignments in the environment

This creates a hierarchy:

- the planner handles strategy and coordination
- the workers handle local execution

That separation is the main design idea behind the system.

## 2. High-Level System Overview

The runtime system has four primary pieces:

- the Overcooked environment
- a shared `Planner`
- `WorkerAgent("worker_0")`
- `WorkerAgent("worker_1")`

The planner is not itself an `Agent` in the environment. It does not directly produce motion or interact actions. Instead, it acts as a coordination service shared by both workers.

Each worker is a normal agent from the environment's perspective. When the environment asks for actions, each worker:

1. checks whether the planner should replan
2. reads its current assigned task
3. receives a text serialization of the current game state
4. runs a ReAct-style LLM loop over worker tools
5. commits one environment action

The planner is invoked on demand rather than every single worker call. The first worker that acts on a timestep can trigger planning. The second worker on that same timestep usually reads the already-assigned plan from cache.

This matters because the environment's `AgentPair` API calls the workers sequentially, not simultaneously.

## 3. Main Components and Their Roles

The most important files are:

- [planner.py](c:/Users/vaibh/Downloads/overcooked_llm_agents/src/overcooked_ai_py/agents/llm/planner.py)
- [planner_tools.py](c:/Users/vaibh/Downloads/overcooked_llm_agents/src/overcooked_ai_py/agents/llm/planner_tools.py)
- [worker_agent.py](c:/Users/vaibh/Downloads/overcooked_llm_agents/src/overcooked_ai_py/agents/llm/worker_agent.py)
- [worker_tools.py](c:/Users/vaibh/Downloads/overcooked_llm_agents/src/overcooked_ai_py/agents/llm/worker_tools.py)
- [state_serializer.py](c:/Users/vaibh/Downloads/overcooked_llm_agents/src/overcooked_ai_py/agents/llm/state_serializer.py)
- [tool_state.py](c:/Users/vaibh/Downloads/overcooked_llm_agents/src/overcooked_ai_py/agents/llm/tool_state.py)
- [task.py](c:/Users/vaibh/Downloads/overcooked_llm_agents/src/overcooked_ai_py/agents/llm/task.py)
- [graph_builder.py](c:/Users/vaibh/Downloads/overcooked_llm_agents/src/overcooked_ai_py/agents/llm/graph_builder.py)
- [run_llm_agent.py](c:/Users/vaibh/Downloads/overcooked_llm_agents/scripts/run_llm_agent.py)

These files divide responsibilities cleanly:

- `planner.py` defines when planning happens and how tasks are assigned
- `worker_agent.py` defines how a worker turns a task into one action
- `planner_tools.py` defines the planner's powers
- `worker_tools.py` defines the worker's powers
- `state_serializer.py` defines what information the LLMs are given and how it is described
- `tool_state.py` stores per-agent runtime context and short history
- `task.py` defines the planner-to-worker task contract
- `graph_builder.py` builds the shared ReAct graph used by both planner and workers

This separation helps both development and explanation. If something fails, it is easier to ask whether the problem came from:

- bad state representation
- bad planner reasoning
- bad worker execution
- bad task contract
- bad validation

## 4. Planner Responsibilities

The planner is responsible for coordination, not control.

Its job is to reason about:

- what each worker should be doing right now
- how to split work so both workers stay useful
- whether the next priority is filling a pot, starting cooking, collecting a dish, or serving soup
- which worker is better positioned for a task
- whether the layout implies specialized roles or handoffs

The planner should not be deciding low-level motor control questions like:

- move left or move up this exact step
- turn first or interact now

Those are worker-level decisions.

### 4.1 Planner Lifecycle

The planner is initialized with:

- a model name
- a `replan_interval`
- debug flags
- optional API base and API key
- optional observability hooks

Internally, it stores:

- a planner `ToolState`
- a worker registry mapping worker IDs to worker `ToolState` instances
- a system prompt built once from the map and horizon
- a shared LangGraph ReAct graph
- `_last_plan_step` to prevent duplicate planning on the same timestep

### 4.2 When Replanning Happens

Current replanning logic in [planner.py](c:/Users/vaibh/Downloads/overcooked_llm_agents/src/overcooked_ai_py/agents/llm/planner.py) triggers when:

- this is the first planning step
- enough steps have passed since the previous plan
- any worker has no task
- any worker task has been marked completed

So replanning is interval-based, but not purely interval-based. It can happen early if worker task state indicates that the previous assignment is no longer valid.

### 4.3 Planner Input

When replanning happens, the planner prompt includes:

- current worker statuses
- a compact block of recent worker history
- the current serialized game state
- layout-specific strategy guidance

That means the planner does not reason from state alone. It also reasons from recent behavior, which helps identify when workers are drifting, blocked, or no longer aligned with the intended plan.

## 5. Worker Responsibilities

Each worker is a local executor.

The worker:

- reads its own current task from the planner
- reads the current game state
- optionally uses tools like `get_surroundings()` and `check_path()`
- chooses one action
- records that action in local history

Each worker only knows its own task. The worker does not get access to:

- the other worker's task
- the planner's internal worker registry
- any direct communication channel with the partner

This isolation is deliberate. Coordination should happen through planner assignments, not hidden side channels.

### 5.1 What the Worker Actually Does Each Timestep

At each step, the worker:

1. calls `planner.maybe_replan(state)`
2. reads `planner.get_task(self.worker_id)`
3. serializes the current game state from its own perspective
4. creates a prompt containing:
   - the current task
   - the current game state
   - the worker system prompt
5. runs the worker LLM graph
6. extracts the chosen action from its `ToolState`
7. falls back to `STAY` if no action was chosen
8. logs the committed action and updates task activity

This is important: the worker does not execute a precompiled multi-step controller. It re-runs the LLM each timestep. That is one of the main current reliability limitations.

## 6. ToolState and Runtime Memory

The `ToolState` object is a small but important part of the architecture.

Each planner or worker has its own `ToolState`. It stores:

- the MDP
- the current environment state
- the current agent index
- a motion planner
- the chosen action for this timestep
- the current assigned task
- recent step history

This state is instance-local, which preserves isolation between workers.

### 6.1 What History Is Used For

`ToolState.record_step(...)` stores:

- `timestep`
- `action`
- `task`

This is not long-term memory in the learning sense. It is short-horizon operational context. Its main use is to help the planner understand:

- whether a worker has been repeating the same action
- whether a worker has stayed on the same task for too long
- whether previous assignments may not be working

That history is especially useful in debug traces and in planner prompts.

## 7. Task Representation

Planner-to-worker communication is carried through the `Task` dataclass in [task.py](c:/Users/vaibh/Downloads/overcooked_llm_agents/src/overcooked_ai_py/agents/llm/task.py).

Each task stores:

- a natural-language description
- the target worker ID
- the timestep when it was created
- whether it is completed
- how many steps it has been active

This is intentionally lightweight. The system currently uses natural-language task descriptions rather than a strongly typed planner schema.

That has one clear advantage:

- it is easy to prototype and easy for the LLM to generate

But it also has one major drawback:

- the boundary between a good semantic task and a bad low-level action script is fragile

That drawback became one of the central issues during development.

## 8. ReAct Graph and Tool Loop

Both the planner and the workers use the same shared ReAct graph builder from [graph_builder.py](c:/Users/vaibh/Downloads/overcooked_llm_agents/src/overcooked_ai_py/agents/llm/graph_builder.py).

The graph structure is:

1. call the LLM
2. inspect whether the LLM requested any tools
3. if it asked for observation tools, run them and loop back
4. if it asked for an action tool, run it and terminate
5. if it asked for nothing, terminate

This is elegant because it gives both planner and workers a common reasoning pattern:

- observe
- reason
- act

It also comes with an obvious cost:

- the agent can make several LLM/tool turns inside a single environment timestep

That is one reason latency can become high even when the policy is otherwise reasonable.

## 9. Tools Available to the Planner

The planner gets access to four observation tools and one action tool.

Observation tools:

- `get_surroundings()`
- `get_pot_details()`
- `check_path(target)`
- `get_worker_status(worker_id)`

Action tool:

- `assign_tasks(...)`

The planner's action space is intentionally narrow. It does not move chefs directly. It only assigns tasks.

This is a good architectural boundary. It means:

- planner mistakes are easier to recognize
- planner outputs can be validated before reaching workers
- workers retain local flexibility

## 10. Tools Available to the Workers

Workers get observation tools:

- `get_surroundings()`
- `get_pot_details()`
- `check_path(target)`

And action tools:

- `move_up()`
- `move_down()`
- `move_left()`
- `move_right()`
- `wait()`
- `interact()`

These tools constrain the low-level action space to valid Overcooked actions. That is important because it prevents workers from generating arbitrary text as the final output.

The observation tools also reduce some common failure modes:

- guessing path lengths incorrectly
- misunderstanding pot status
- missing adjacency conditions

## 11. How State Is Presented to the LLMs

The state serializer in [state_serializer.py](c:/Users/vaibh/Downloads/overcooked_llm_agents/src/overcooked_ai_py/agents/llm/state_serializer.py) is one of the most important parts of the system.

It converts simulator state into text that includes:

- timestep and horizon
- an ASCII-style terrain grid
- both player positions
- both player orientations
- both held objects
- all pot states
- counter objects
- current orders

This text is what the planner and workers actually reason over.

### 11.1 Why Text Instead of Images

The system uses symbolic text rather than screenshots because:

- the Overcooked simulator already exposes structured symbolic state
- text can include both geometry and rules
- text makes it easy to inject layout-specific strategy notes
- text is easier to log, diff, and inspect during debugging

### 11.2 Pot Representation

Pot status is described carefully because it is one of the most important control points in the game. A pot may be:

- empty
- partially filled
- full but idle
- cooking
- ready

The serializer explicitly distinguishes these cases, especially under the newer non-auto-start cooking dynamics. This is crucial because `3/3 ingredients` does not mean `ready soup`.

## 12. System Prompts

There are separate system prompts for:

- the planner
- each worker

### 12.1 Planner Prompt

The planner prompt teaches:

- core game rules
- one-item-at-a-time constraint
- recipe rules for the current layout
- map geometry
- station locations
- worker IDs
- layout-specific coordination rules
- task assignment guidelines

The planner prompt is written to encourage:

- explicit coordinate-based tasks
- complementary worker assignments
- minimal idling
- valid cooking sequencing

### 12.2 Worker Prompt

The worker prompt teaches:

- adjacency rules
- facing rules
- interaction mechanics
- path checking
- item constraints
- pot lifecycle constraints
- how to interpret tasks from the current state instead of restarting the whole sentence each turn

The worker prompt is more operational than the planner prompt. Its purpose is to convert a semantic task into one good next action.

## 13. Layout-Specific Prompt Optimizations

One of the most important improvements on this branch was making the prompts layout-aware.

The generic prompting approach was not enough because the five layouts differ meaningfully:

- `cramped_room`
- `asymmetric_advantages`
- `coordination_ring`
- `forced_coordination`
- `counter_circuit`

These maps differ in:

- path bottlenecks
- reachable stations
- whether handoff counters are required
- whether early dish pickup is useful or harmful
- whether side specialization is beneficial
- whether both onion and tomato recipes are present

### 13.1 Cramped Room

For `cramped_room`, the prompt now emphasizes:

- avoid early dish pickup
- avoid blocking the central lane
- keep both workers contributing to the onion pipeline until the pot is full whenever possible
- do not assign vague waiting if the pot still needs ingredients

This was important because the planner often assigned a dish too early, causing one worker to stand around holding it while soup production stalled.

### 13.2 Asymmetric Advantages

For `asymmetric_advantages`, the prompt distinguishes left-side and right-side access patterns and encourages side-based specialization around the shared central pots.

### 13.3 Coordination Ring

For `coordination_ring`, the prompt emphasizes avoiding traffic on the same segment of the ring and using a pipeline that respects the layout's loop structure.

### 13.4 Forced Coordination

For `forced_coordination`, the prompt explains that the layout is split and that some stations are effectively inaccessible from one side. This requires explicit handoff behavior using shared counters.

Without this, generic coordination often failed because the planner assumed any worker could directly do any subtask.

### 13.5 Counter Circuit

For `counter_circuit`, the prompt highlights:

- long travel distance
- staging via counters
- mixed onion/tomato recipe support
- the importance of choosing who handles ingredients versus pot-side actions

## 14. The Planner-Worker Contract Problem

One of the main issues we observed was that the planner sometimes emitted raw action-style outputs such as:

- `right, down, interact`
- `left`
- `move up`

Those are poor planner outputs because the worker already has its own action-selection loop. The planner should assign semantic tasks, not brittle low-level scripts.

### 14.1 Good vs Bad Task Forms

Good:

- `Go to onion dispenser at (4,1), pick onion, deliver to pot at (2,0)`
- `Go to dish dispenser at (1,3), pick dish`
- `Go to pot at (2,0), interact to start cooking`

Bad:

- `right, down, interact`
- `up, interact`
- `move left then interact`

Why this matters:

- semantic tasks remain meaningful across several steps
- raw action scripts become invalid as soon as the state changes slightly

Since workers re-run the LLM every step, giving them a raw action script makes execution much less stable.

## 15. Planner-Side Normalization and Validation

To mitigate the planner contract issue, [planner_tools.py](c:/Users/vaibh/Downloads/overcooked_llm_agents/src/overcooked_ai_py/agents/llm/planner_tools.py) was extended with normalization and validation.

This layer now:

- rejects obvious raw action scripts
- canonicalizes some mixed-format assignments into semantic task descriptions
- validates `start cooking` tasks against actual pot state

For example, a planner string like:

`go to (0,2), interact to pick onion, go to (2,0), interact to place onion`

can be rewritten into:

`Go to onion dispenser at (0,2), pick onion, deliver to pot at (2,0)`

This improves the quality of the task contract without requiring a full typed output schema.

## 16. Hard Rule Enforcement

Prompts help, but they do not guarantee correctness. One concrete recurring issue was workers or planner outputs attempting to start cooking before a pot had the required ingredients.

To address that, the code now includes rule-level enforcement rather than relying only on prompt instructions.

### 16.1 Planner-Side Cooking Validation

Planner task assignment rejects `start cooking` tasks unless the target pot is actually:

- full
- idle
- not already cooking
- not already ready

### 16.2 Worker-Side Pot Interaction Guard

The worker includes a guard against invalid pot interactions such as:

- empty-handed `INTERACT` on a not-yet-full pot
- using a dish on a pot that is not ready
- adding ingredients to a pot that cannot accept more

This is one of the clearest lessons from the project:

- prompts should express soft preferences and strategy
- code should enforce hard rules

## 17. Observability and Debugging

The system includes observability support through:

- file logging
- step-level debug prints
- per-role event tracing
- optional LangFuse integration

This was essential because many failures look similar from reward alone.

For example, a zero-reward rollout could result from:

- bad planning
- bad worker execution
- endpoint failures
- one worker being idle while the other is overloaded
- repeated invalid interactions

The debug logs made it possible to distinguish these cases.

Useful signals included:

- planner assignments
- number of LLM calls per step
- worker action logs
- held objects and positions
- reward accumulation over time

## 18. Problems Encountered During Development

Several recurring issues were observed while iterating on the system.

### 18.1 Raw Planner Actions

The planner sometimes emitted action-level instructions instead of semantic tasks.

### 18.2 Worker Drift

Workers re-read their task every step and sometimes drifted away from the intended plan, especially on longer multi-clause task descriptions.

### 18.3 Premature Cooking

The system sometimes attempted to start cooking with fewer than three required ingredients in the pot.

### 18.4 Layout Mismatch

Generic prompts did not respect layout-specific coordination requirements, especially on:

- `forced_coordination`
- `cramped_room`
- `counter_circuit`

### 18.5 Early Dish Pickup

On some layouts, especially `cramped_room`, the planner assigned a dish too early. This often caused one worker to idle with the dish while the other tried to finish the ingredient pipeline alone.

### 18.6 Unicode / Arrow Logging Issues

Some debug output used non-ASCII or arrow-like rendering, which caused poor readability and occasional display issues in the Windows terminal.

### 18.7 Latency

Because the system can make:

- one planner call
- two worker calls
- multiple observation loops inside those calls

the average time per environment step can become very high.

## 19. What Was Solved and What Was Only Improved

Some issues were strongly addressed:

- state representation is much better than before
- layout-specific prompting significantly improved spatial and strategic understanding
- planner-side normalization reduced raw action-script assignments
- hard cooking guards improved rule correctness
- ASCII action labels improved debugging and log readability

Other issues were improved but not truly solved:

- worker execution still drifts on long natural-language tasks
- worker reasoning is still expensive because it happens every step
- the planner-worker contract is better, but still natural-language based
- replanning is still not strongly event-driven

So the system is more robust than it was initially, but it is not yet a fully reliable control architecture.

## 20. The Unicode / Arrow Issue

One small but practical issue involved how actions were represented in logs. Non-ASCII arrows or similar symbols were not ideal for Windows terminal output.

This was cleaned up in [actions.py](c:/Users/vaibh/Downloads/overcooked_llm_agents/src/overcooked_ai_py/mdp/actions.py), which now uses plain ASCII-friendly action labels:

- `up`
- `down`
- `left`
- `right`
- `stay`
- `interact`

This change did not improve policy quality directly, but it made debugging much easier and reduced logging noise.

## 21. Current Strengths of the Architecture

The planner-worker system is now strongest in the following ways:

- it has a clear and understandable separation of concerns
- it preserves worker isolation cleanly
- it has a strong symbolic state representation
- it can express map-specific strategy in prompts
- it has explicit planner-to-worker task passing
- it now includes some critical correctness checks beyond prompting
- it is instrumented well enough to diagnose many failure modes

From a research and engineering perspective, it is a good baseline because the boundaries between representation, planning, execution, and validation are explicit.

## 22. Remaining Weaknesses

The main remaining weakness is worker-side execution.

At the moment:

- the planner produces a task
- the worker reinterprets that task every step with the LLM

That means the system does not yet have a stable compiled execution layer for multi-step subtasks. As a result, the architecture remains strongest at strategic coordination and weakest at reliable local task completion.

This is why many future improvements likely belong on the worker side rather than the planner side.

## 23. Best Next Directions

The most sensible next steps are:

1. keep the planner as the strategic LLM
2. keep layout-specific prompt guidance
3. keep planner-side normalization/validation
4. keep hard rule enforcement for cooking and pot interactions
5. shorten planner assignments into more atomic semantic tasks
6. reduce repeated worker re-reasoning where possible
7. move gradually toward a stronger execution contract between planner and worker

In other words, the next wave of improvements should focus on making the worker more reliable without discarding the planner-worker split that already provides the system's main coordination benefits.

## 24. Summary

This project evolved from a more generic LLM-driven Overcooked agent into a structured planner-worker system.

The important architectural pieces are:

- centralized planning
- decentralized worker execution
- shared tool-based reasoning
- structured state serialization
- short-horizon task history
- layout-specific prompting
- planner-side task normalization
- code-level correctness guards

The main lesson is that prompt engineering alone is not enough in a rule-heavy, multi-agent simulator. The most robust progress came from combining:

- better prompts
- better state representation
- better planner-worker contracts
- hard execution-time validation

That combination is what made the system more understandable, more debuggable, and meaningfully more reliable.
