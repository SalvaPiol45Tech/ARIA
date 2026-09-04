# ARIA — Adaptive Reasoning & Imagination Agent

> A research project for building an integrated intelligent agent combining **Vision, Language, Multimodal Learning, Reasoning, Planning, World Models, Memory, and Reinforcement Learning**.

---

## 🧠 Overview

**ARIA (Adaptive Reasoning & Imagination Agent)** is an experimental AI agent architecture developed from scratch in Python and PyTorch.

The project explores how multiple intelligent capabilities can be integrated into a single agent:

```text
                    ARIA
                     │
        ┌────────────┴────────────┐
        │                         │
      Vision                   Language
        │                         │
        └────────────┬────────────┘
                     ▼
            Multimodal Fusion
                     │
                     ▼
             World State
                     │
          ┌──────────┼──────────┐
          │          │          │
          ▼          ▼          ▼
      Reasoning   Planning    Memory
          │          │          │
          └──────┬───┴──────────┘
                 ▼
             World Model
                 │
                 ▼
            Imagination
                 │
                 ▼
          RL Controller
                 │
                 ▼
               Action
                 │
                 ▼
            Environment
                 │
                 └──────────► Memory
```

ARIA is not intended to claim AGI. It is a research and engineering project exploring architectural components that are relevant to intelligent agents and general-purpose AI systems.

---

# 🎯 Project Goal

The goal of ARIA is to progressively move from a simple rule-based navigation agent toward an integrated learning-based agent capable of:

* Perceiving visual environments
* Understanding natural-language instructions
* Grounding language into visual space
* Building structured representations of the environment
* Reasoning about goals and obstacles
* Planning actions
* Predicting future states
* Imagining possible trajectories
* Learning from interaction
* Maintaining episodic memory
* Selecting actions using reinforcement learning

The project is developed incrementally so that every capability can be tested independently before being integrated into the final architecture.

---

# 🏗️ Project Evolution

ARIA was developed through several stages.

```text
aria_agent
    │
    ▼
ARIA v3
    │
    ▼
ARIA v3.1
    │
    ▼
ARIA v3.2
    │
    ▼
ARIA v3.2.1
    │
    ▼
ARIA v4.0
    │
    ▼
ARIA FINAL
```

Each version introduces a new capability while preserving the previous components.

---

# 1. ARIA Agent — Initial Foundation

### File

```text
aria_agent.py
```

The initial ARIA agent establishes the basic agent-environment interaction loop.

The early architecture focuses on:

* Environment interaction
* State representation
* Agent actions
* Navigation
* Basic planning
* Evaluation

The purpose of this stage is to establish a reliable foundation before introducing neural models.

---

# 2. ARIA v3 — Hard Navigation

### File

```text
aria_v3.py
```

ARIA v3 introduces a more difficult navigation environment.

### Environment

The environment contains:

* A 15×15 grid
* Multiple obstacles
* Multiple objects
* Distractors
* A target object
* Natural-language instructions

Example:

```text
Go to the red object while avoiding obstacles.
```

The agent must:

1. Understand the instruction
2. Identify the correct object
3. Locate the target
4. Avoid obstacles
5. Navigate toward the target

### Architecture

```text
Observation
     │
     ▼
Scene Understanding
     │
     ▼
Target Reasoning
     │
     ▼
A* Planner
     │
     ▼
Action
     │
     ▼
Environment
```

### Main Components

#### HardNavigationEnv

Creates the navigation environment.

#### SceneUnderstanding

Converts the raw environment into a structured scene representation.

#### GridPlanner

Implements an A* path planner.

#### ARIAv3

Combines perception, reasoning, planning and action.

### Purpose

ARIA v3 establishes a reliable symbolic baseline.

A* is intentionally used as an expert planner at this stage.

---

# 3. ARIA v3.1 — Neural Scene Understanding

### File

```text
aria_v3_1.py
```

ARIA v3.1 replaces simple symbolic target identification with a learned neural grounding model.

### Architecture

```text
Visual Grid
     │
     ▼
Vision Encoder
     │
     │
Instruction
     │
     ▼
Language Encoder
     │
     ▼
Multimodal Fusion
     │
     ▼
Target Grounding
     │
     ▼
A* Planning
     │
     ▼
Action
```

### Main Capability

The neural model learns to identify the target location from the visual scene and instruction.

Example:

```text
Instruction:
Go to the red object while avoiding obstacles.

Neural predicted target:
(3, 3)

Actual target:
(3, 3)

Confidence:
0.8765
```

### Training

The model was trained using generated navigation samples.

Reported experiment:

```text
Training samples: 2500
Validation samples: 500

Parameters:
498,049

Validation Accuracy:
100%
```

### Checkpoint

```text
checkpoints/
└── aria_v3_1_grounding.pt
```

This stage demonstrates that neural visual-language grounding can successfully replace the purely symbolic target-selection component.

---

# 4. ARIA v3.2 — Neural Reasoning

### File

```text
aria_v3_2.py
```

ARIA v3.2 introduces a dedicated neural reasoning module.

The reasoner attempts to predict several properties of the current world state.

### Reasoning Outputs

```text
Target
Blocked / Unblocked
Reachable / Unreachable
Nearest Object
Direction
```

The architecture becomes:

```text
Visual Input
     │
     ▼
Vision
     │
     ▼
Language
     │
     ▼
Multimodal Representation
     │
     ▼
Neural Reasoner
     │
     ├── Target
     ├── Blocked
     ├── Reachability
     ├── Nearest Object
     └── Direction
             │
             ▼
          Planner
             │
             ▼
           Action
```

### Important Result

The experiment exposed a critical weakness.

The reasoning model achieved:

```text
Target Reasoning:             0%
Blocked Reasoning:           ~51%
Reachability Reasoning:      100%
Nearest Object Reasoning:    ~23%
Navigation Success:            6%
```

This was an important result.

The experiment demonstrated that simply adding a neural reasoning head does not automatically produce reliable reasoning.

The model could learn some spatial properties while failing to correctly ground the requested target.

This motivated the next architectural improvement.

---

# 5. ARIA v3.2.1 — Spatial Neural Reasoning

### File

```text
aria_v3_2_1.py
```

ARIA v3.2.1 improves spatial representation and target grounding.

Instead of relying heavily on abstract reasoning outputs, the model learns a spatial target representation directly from the environment.

### Architecture

```text
Visual Grid
     │
     ▼
Spatial Encoder
     │
     ▼
Transformer Representation
     │
     ▼
Target Heatmap
     │
     ▼
Target Position
     │
     ▼
A* Planner
     │
     ▼
Action
```

### Results

Training:

```text
Training samples: 3000
Validation samples: 600

Parameters:
531,074
```

Reported validation accuracy:

```text
100%
```

Evaluation:

```text
Grounding Accuracy : 100%
Navigation Success : 100%
Collision Rate     : 0%
Final Distance     : 0
Average Steps      : 10.73
```

### Checkpoint

```text
checkpoints/
└── aria_v3_2_1_spatial_grounding.pt
```

This version demonstrated that better spatial representations can dramatically improve the reliability of neural grounding.

---

# 6. ARIA v4.0 — Learned Planning

### File

```text
aria_v4.py
```

ARIA v4 moves the learning boundary from perception into planning.

The neural planner is trained from expert demonstrations generated by A*.

### Architecture

```text
Visual Grid
     │
     ▼
Spatial Encoder
     │
     ▼
Neural Planner
     │
     ▼
Action
     │
     ▼
Environment
```

### Important Design Decision

A* is used **only to generate training data**.

During inference:

```text
A* → NOT USED
```

ARIA predicts actions directly using the learned planner.

### Training

```text
Training samples: 10,000
Validation samples: 2,000

Model parameters:
260,164
```

Reported best validation accuracy:

```text
69.70%
```

### Evaluation

```text
Average Reward:      -12.5307
Navigation Success:   40%
Collision Rate:       92.12%
Final Distance:        4.1208
```

### Lesson

This experiment demonstrated an important limitation of behavior cloning.

A model can achieve reasonable action classification accuracy while still failing badly when deployed sequentially.

Small action errors accumulate and cause the agent to enter states that were not represented correctly in its training distribution.

This motivates model-based planning and feedback.

---

# 7. ARIA FINAL — Integrated Agent

### File

```text
aria_final.py
```

ARIA FINAL integrates the major components developed throughout the project.

### Final Architecture

```text
                    ARIA FINAL

                     Vision
                       │
                       ▼
                Visual Encoder
                       │
                       │
                Language Encoder
                       │
                       ▼
             Multimodal Transformer
                       │
                       ▼
                  World State
                       │
             ┌─────────┴─────────┐
             │                   │
             ▼                   ▼
         Reasoning             Memory
             │                   │
             ▼                   │
          Planning ◄─────────────┘
             │
             ▼
         World Model
             │
             ▼
        Imagination
             │
             ▼
        RL Controller
             │
             ▼
           Action
             │
             ▼
        Environment
             │
             └────────────► Memory
```

---

# 🧩 ARIA FINAL Components

## Vision

The visual component converts the grid environment into a learned representation.

It provides information about:

* Agent position
* Objects
* Obstacles
* Target
* Spatial relationships

---

## Language

The language encoder processes the natural-language instruction.

Example:

```text
Go to the green object while avoiding obstacles.
```

The instruction is combined with visual information.

---

## Multimodal Fusion

Vision and language are projected into a shared representation.

```text
Vision ───────┐
              ├──► Multimodal Representation
Language ─────┘
```

This allows the agent to connect language instructions with visual entities.

---

# 🧠 Neural Reasoning

The reasoning component attempts to transform the multimodal world representation into structured decisions.

Conceptually:

```text
World State
     │
     ▼
Reasoning
     │
     ├── What is the target?
     ├── Where is it?
     ├── Is the route blocked?
     ├── Is the target reachable?
     └── What direction should be considered?
```

---

# 🗺️ Planning

The planning component determines how the agent should reach the objective.

ARIA explores both:

* Expert symbolic planning
* Learned planning

The final architecture allows planning to interact with:

* Reasoning
* Memory
* World-model predictions

---

# 🌎 World Model

The world model attempts to predict future states.

Conceptually:

```text
Current State + Action
          │
          ▼
      World Model
          │
          ▼
    Predicted Future
```

This allows ARIA to reason about possible consequences before executing actions.

---

# 🔮 Imagination

ARIA can use the world model to evaluate hypothetical trajectories.

```text
Current State
      │
      ├── Action A → Future A
      │
      ├── Action B → Future B
      │
      └── Action C → Future C
                       │
                       ▼
                  Evaluation
                       │
                       ▼
                Select Action
```

This creates the foundation for model-based decision making.

---

# 🧠 Episodic Memory

ARIA maintains a memory of previous interactions.

The memory can contain information such as:

```text
State
Action
Reward
Next State
```

The purpose is to allow future decisions to incorporate previous experience.

---

# 🎮 Reinforcement Learning

The RL controller learns to select actions through interaction with the environment.

Conceptually:

```text
State
  │
  ▼
RL Controller
  │
  ▼
Action
  │
  ▼
Environment
  │
  ├── Reward
  │
  └── Next State
       │
       ▼
     Memory
```

This provides a learning mechanism beyond supervised expert demonstrations.

---

# 🧪 ARIA FINAL Experiment

Reported final experiment:

```text
Device:
CPU/GPU

Grid:
15 × 15

Parameters:
3,212,403
```

### Sanity Tests

```text
✓ Vision encoder
✓ Language encoder
✓ Multimodal fusion
✓ Neural reasoning
✓ Neural planner
✓ World model
✓ Episodic memory
✓ RL controller
```

---

# 📊 Final Evaluation

Reported experiment:

```text
Average Reward:
6.1965

Success Rate:
100%

Collision Rate:
0%

Final Distance:
0.0000

Average Steps:
11.58
```

The final system successfully completed the tested navigation environment.

These results are **experimental results on the project's synthetic navigation environment**, not evidence of general intelligence.

---

# 🏋️ Training

ARIA FINAL contains separate training stages for the major learned components.

### Neural Reasoner

```text
Epoch 01/3
Loss: 5.57144

Epoch 02/3
Loss: 5.23737

Epoch 03/3
Loss: 5.19428
```

### World Model

The world model was trained through environment interaction.

Example training logs:

```text
Episode 0010 | World Model Loss: 13.00639
Episode 0020 | World Model Loss: 9.84539
Episode 0030 | World Model Loss: 7.07009
Episode 0040 | World Model Loss: 5.71940
Episode 0050 | World Model Loss: 5.90110
```

### RL Controller

Example training:

```text
Episode 0020 | Avg Reward: -4.146
Episode 0040 | Avg Reward: -3.795
Episode 0060 | Avg Reward: -3.884
Episode 0080 | Avg Reward: -3.997
Episode 0100 | Avg Reward: -3.018
```

---

# 💾 Checkpoints

The project produces checkpoints for the different stages.

```text
checkpoints/
│
├── aria_v3_1_grounding.pt
├── aria_v3_2_reasoning.pt
├── aria_v3_2_1_spatial_grounding.pt
├── aria_v4_learned_planner.pt
└── aria_final.pt
```

The final integrated checkpoint is:

```text
checkpoints/aria_final.pt
```

---

# 📁 Repository Structure

Recommended repository structure:

```text
ARIA/
│
├── README.md
│
├── aria_agent.py
├── aria_v3.py
├── aria_v3_1.py
├── aria_v3_2.py
├── aria_v3_2_1.py
├── aria_v4.py
├── aria_final.py
│
├── checkpoints/
│   ├── aria_v3_1_grounding.pt
│   ├── aria_v3_2_reasoning.pt
│   ├── aria_v3_2_1_spatial_grounding.pt
│   ├── aria_v4_learned_planner.pt
│   └── aria_final.pt
│
├── requirements.txt
│
└── LICENSE
```

---

# ⚙️ Requirements

The project uses Python and PyTorch.

Core dependencies:

```text
Python
PyTorch
NumPy
```

Install dependencies:

```bash
pip install torch numpy
```

---

# ▶️ Running ARIA

Run the initial agent:

```bash
python aria_agent.py
```

Run ARIA v3:

```bash
python aria_v3.py
```

Run neural scene understanding:

```bash
python aria_v3_1.py
```

Run neural reasoning:

```bash
python aria_v3_2.py
```

Run spatial reasoning:

```bash
python aria_v3_2_1.py
```

Run learned planning:

```bash
python aria_v4.py
```

Run the integrated system:

```bash
python aria_final.py
```

---

# 🔬 Research Philosophy

ARIA is intentionally developed incrementally.

Instead of immediately building a large end-to-end system, each capability is tested independently.

```text
Environment
    ↓
Perception
    ↓
Grounding
    ↓
Reasoning
    ↓
Planning
    ↓
World Model
    ↓
Imagination
    ↓
Memory
    ↓
Reinforcement Learning
    ↓
Integrated Agent
```

This makes it possible to identify where failures occur and improve individual components.

---

# 📈 What the Experiments Demonstrated

The development process revealed several important engineering lessons.

### 1. Neural grounding can work well

ARIA v3.1 and v3.2.1 achieved high target-grounding performance on the generated environment.

### 2. Reasoning is harder than classification

ARIA v3.2 showed that some reasoning outputs can remain poor even when other outputs perform well.

### 3. Better spatial representations matter

ARIA v3.2.1 improved grounding by explicitly modeling spatial information.

### 4. Action accuracy does not guarantee successful behavior

ARIA v4 achieved approximately 69.7% validation action accuracy but only 40% navigation success and a high collision rate.

This demonstrates the difference between:

```text
Prediction Accuracy
```

and

```text
Closed-Loop Agent Performance
```

### 5. World models provide a path toward model-based decision making

Instead of only predicting the next action, an agent can learn to predict possible future states and evaluate them before acting.

---

# 🚀 ARIA's Long-Term Research Direction

The long-term objective is to investigate an agent architecture where:

```text
Perception
    +
Language
    +
Multimodal Representation
    +
Reasoning
    +
Planning
    +
World Modeling
    +
Imagination
    +
Memory
    +
Reinforcement Learning
```

operate as a coordinated system.

The project can eventually be extended toward more complex environments, longer-horizon tasks, richer multimodal inputs, continual learning, and embodied interaction.

---

# ⚠️ Current Limitations

ARIA FINAL should be considered an **experimental research prototype**.

The current system operates in a simplified synthetic grid-world environment.

Important limitations include:

* Small environment
* Synthetic visual representation
* Limited language instructions
* Limited object vocabulary
* Small neural models
* Limited training data
* Simplified memory
* Simplified world model
* Simplified reinforcement learning
* No real robotic embodiment
* No large-scale multimodal pretraining
* No demonstrated general intelligence

Therefore:

> **ARIA is an experimental architecture for studying intelligent-agent components, not a claim of AGI.**

---

# 🧪 Reproducibility

Experiments use deterministic seeds where possible.

Example:

```python
seed = 42
```

The project records:

* Training loss
* Validation performance
* Navigation success
* Collision rate
* Final distance
* Average steps
* Model checkpoints

This makes it easier to compare future architecture versions.

---

# 📜 Version History

| Version         | Main Capability                                                                                 |
| --------------- | ----------------------------------------------------------------------------------------------- |
| `aria_agent.py` | Initial agent foundation                                                                        |
| `ARIA v3`       | Hard navigation + A* planning                                                                   |
| `ARIA v3.1`     | Neural scene understanding                                                                      |
| `ARIA v3.2`     | Neural reasoning                                                                                |
| `ARIA v3.2.1`   | Spatial neural reasoning                                                                        |
| `ARIA v4.0`     | Learned neural planning                                                                         |
| `ARIA FINAL`    | Vision + Language + Multimodal + Reasoning + Planning + World Model + Imagination + Memory + RL |

---

# 👨‍💻 Project

**ARIA — Adaptive Reasoning & Imagination Agent**

Built as a from-scratch AI research project using:

```text
Python
PyTorch
Deep Learning
Transformers
Multimodal Learning
Neural Reasoning
Planning
World Models
Imagination
Memory
Reinforcement Learning
```

---

# ⭐ Why ARIA?

The central idea behind ARIA is simple:

> An intelligent agent should not only predict what to do. It should perceive the world, understand instructions, reason about its current state, plan possible actions, imagine future outcomes, learn from experience, and use memory to improve future decisions.

ARIA is an ongoing exploration of how these components can be combined into one coherent agent architecture.

---

## Project Status

```text
ARIA FINAL
────────────────────────────

✓ Vision
✓ Language
✓ Multimodal Fusion
✓ Neural Scene Understanding
✓ Neural Reasoning
✓ Planning
✓ World Model
✓ Imagination
✓ Episodic Memory
✓ Reinforcement Learning

Status:
Experimental Research Prototype
```

---

## License

Add your preferred open-source license before publishing the repository.

For example:

```text
MIT License
```

---

## Disclaimer

This project is for research, education, and experimentation in artificial intelligence.

The name **ARIA** refers to this research project and does not imply that the system possesses artificial general intelligence, consciousness, or human-level reasoning.
