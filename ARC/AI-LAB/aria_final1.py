"""
======================================================================
ARIA FINAL
Adaptive Reasoning & Imagination Agent
======================================================================

Research prototype:
    Vision + Language + Multimodal Fusion
    + Memory + Reasoning + Planning
    + World Model + Imagination
    + RL Policy / Value

Important:
    This is an AGI-inspired research architecture, not a claim of AGI.

Inference pipeline:

    Observation
        |
        v
    Visual Encoder
        |
        +------ Language Encoder
        |             |
        +------ Multimodal Fusion
                      |
                      v
                   Memory
                      |
                      v
                  Reasoning
                      |
                      v
                   Planner
                      |
              +-------+-------+
              |               |
              v               v
        World Model      RL Policy
              |
              v
        Imagination
              |
              v
        Trajectory Scoring
              |
              v
            Action
              |
              v
         Environment
              |
              +------> Memory
"""

import math
import random
from dataclasses import dataclass
from collections import deque
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ======================================================================
# CONFIG
# ======================================================================

@dataclass
class Config:

    grid_size: int = 15

    num_obstacles: int = 22
    num_objects: int = 4

    max_steps: int = 100

    hidden_dim: int = 128

    memory_size: int = 32

    imagination_horizon: int = 8
    imagination_samples: int = 4

    gamma: float = 0.99

    seed: int = 42

    device: str = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


# ======================================================================
# SEED
# ======================================================================

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ======================================================================
# ENVIRONMENT
# ======================================================================

Position = Tuple[int, int]


class ARIAEnvironment:

    COLORS = [
        "red",
        "green",
        "blue",
        "yellow"
    ]

    SYMBOLS = {
        "red": "R",
        "green": "G",
        "blue": "B",
        "yellow": "Y"
    }

    ACTIONS = [
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1)
    ]

    def __init__(self, cfg):

        self.cfg = cfg
        self.size = cfg.grid_size

        self.agent = None
        self.objects = {}
        self.obstacles = set()

        self.target_color = None
        self.goal = None

        self.steps = 0
        self.previous_distance = None

    # ------------------------------------------------------------------
    # RESET
    # ------------------------------------------------------------------

    def reset(self):

        self.steps = 0

        self.obstacles = set()
        self.objects = {}

        # Agent
        self.agent = self.random_position()

        # Obstacles
        attempts = 0

        while len(self.obstacles) < self.cfg.num_obstacles:

            pos = self.random_position()

            if pos != self.agent:
                self.obstacles.add(pos)

            attempts += 1

            if attempts > 10000:
                break

        # Objects
        for color in self.COLORS[:self.cfg.num_objects]:

            for _ in range(1000):

                pos = self.random_position()

                if self.valid_object_position(pos):

                    self.objects[color] = pos
                    break

        self.target_color = random.choice(
            list(self.objects.keys())
        )

        self.goal = self.objects[
            self.target_color
        ]

        self.previous_distance = self.distance(
            self.agent,
            self.goal
        )

        return self.observation()

    # ------------------------------------------------------------------
    # POSITION
    # ------------------------------------------------------------------

    def random_position(self):

        return (
            random.randint(0, self.size - 1),
            random.randint(0, self.size - 1)
        )

    def valid_object_position(self, pos):

        if pos in self.obstacles:
            return False

        if pos == self.agent:
            return False

        if pos in self.objects.values():
            return False

        return True

    # ------------------------------------------------------------------
    # DISTANCE
    # ------------------------------------------------------------------

    @staticmethod
    def distance(a, b):

        return math.sqrt(
            (a[0] - b[0]) ** 2 +
            (a[1] - b[1]) ** 2
        )

    # ------------------------------------------------------------------
    # INSTRUCTION
    # ------------------------------------------------------------------

    def instruction(self):

        return (
            f"Go to the {self.target_color} object "
            f"while avoiding obstacles."
        )

    # ------------------------------------------------------------------
    # OBSERVATION
    # ------------------------------------------------------------------

    def observation(self):

        return {
            "agent": self.agent,
            "objects": dict(self.objects),
            "obstacles": set(self.obstacles),
            "goal": self.goal,
            "target": self.target_color,
            "instruction": self.instruction()
        }

    # ------------------------------------------------------------------
    # STEP
    # ------------------------------------------------------------------

    def step(self, action):

        self.steps += 1

        dx, dy = action

        candidate = (
            self.agent[0] + int(dx),
            self.agent[1] + int(dy)
        )

        collision = False

        reward = -0.01

        # Bounds
        if (
            candidate[0] < 0
            or candidate[0] >= self.size
            or candidate[1] < 0
            or candidate[1] >= self.size
        ):

            collision = True
            reward -= 0.10

            candidate = self.agent

        # Obstacle
        elif candidate in self.obstacles:

            collision = True
            reward -= 0.25

            candidate = self.agent

        self.agent = candidate

        new_distance = self.distance(
            self.agent,
            self.goal
        )

        reward += (
            self.previous_distance -
            new_distance
        ) * 0.15

        self.previous_distance = new_distance

        success = (
            self.agent == self.goal
        )

        if success:
            reward += 5.0

        done = (
            success
            or self.steps >= self.cfg.max_steps
        )

        info = {
            "success": success,
            "collision": collision,
            "distance": new_distance,
            "steps": self.steps
        }

        return (
            self.observation(),
            reward,
            done,
            info
        )

    # ------------------------------------------------------------------
    # RENDER
    # ------------------------------------------------------------------

    def render(self):

        grid = [
            ["." for _ in range(self.size)]
            for _ in range(self.size)
        ]

        for x, y in self.obstacles:
            grid[y][x] = "#"

        for color, pos in self.objects.items():

            x, y = pos
            grid[y][x] = self.SYMBOLS[color]

        gx, gy = self.goal
        grid[gy][gx] = "T"

        ax, ay = self.agent
        grid[ay][ax] = "A"

        print()

        for row in grid:
            print(" ".join(row))

        print()

        print(
            "Instruction:",
            self.instruction()
        )


# ======================================================================
# ACTION ENCODING
# ======================================================================

ACTION_LIST = [
    (1, 0),      # right
    (-1, 0),     # left
    (0, 1),      # down
    (0, -1),     # up
    (0, 0)       # stay
]


def action_to_index(action):

    return ACTION_LIST.index(tuple(action))


def index_to_action(index):

    return ACTION_LIST[int(index)]


# ======================================================================
# VISUAL ENCODER
# ======================================================================

class VisualEncoder(nn.Module):

    """
    Converts the grid into a learned spatial representation.
    """

    def __init__(self, hidden_dim):

        super().__init__()

        self.encoder = nn.Sequential(

            nn.Conv2d(
                7,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.GELU(),

            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1
            ),

            nn.GELU(),

            nn.Conv2d(
                64,
                hidden_dim,
                kernel_size=3,
                padding=1
            ),

            nn.GELU()
        )

    def forward(self, x):

        features = self.encoder(x)

        return features.mean(
            dim=(2, 3)
        )


# ======================================================================
# LANGUAGE ENCODER
# ======================================================================

class LanguageEncoder(nn.Module):

    """
    Small instruction encoder.

    For this prototype the instruction is converted into
    a compact task representation.
    """

    def __init__(self, hidden_dim):

        super().__init__()

        self.embedding = nn.Embedding(
            5,
            hidden_dim
        )

        self.projection = nn.Sequential(

            nn.Linear(
                hidden_dim,
                hidden_dim
            ),

            nn.GELU(),

            nn.Linear(
                hidden_dim,
                hidden_dim
            )
        )

    def forward(self, target_index):

        x = self.embedding(
            target_index
        )

        return self.projection(x)


# ======================================================================
# MULTIMODAL FUSION
# ======================================================================

class MultimodalFusion(nn.Module):

    def __init__(self, hidden_dim):

        super().__init__()

        self.fusion = nn.Sequential(

            nn.Linear(
                hidden_dim * 2,
                hidden_dim
            ),

            nn.GELU(),

            nn.Linear(
                hidden_dim,
                hidden_dim
            )
        )

    def forward(
        self,
        vision,
        language
    ):

        x = torch.cat(
            [vision, language],
            dim=-1
        )

        return self.fusion(x)


# ======================================================================
# MEMORY
# ======================================================================

class EpisodicMemory:

    """
    Simple episodic memory.

    Stores previous state/action/reward experiences.

    Later this can become:
        - learned memory
        - vector memory
        - recurrent memory
        - external memory
    """

    def __init__(self, capacity):

        self.memory = deque(
            maxlen=capacity
        )

    def add(
        self,
        state,
        action,
        reward,
        next_state
    ):

        self.memory.append({

            "state": state,

            "action": action,

            "reward": reward,

            "next_state": next_state
        })

    def retrieve(self, query_state, k=4):

        if len(self.memory) == 0:
            return []

        # Simple recency retrieval for v1.
        return list(
            self.memory
        )[-k:]


# ======================================================================
# REASONING MODULE
# ======================================================================

class NeuralReasoner(nn.Module):

    """
    Produces a latent reasoning state.

    Outputs:
        target representation
        direction logits
        obstacle-risk
        reachability
    """

    def __init__(self, hidden_dim):

        super().__init__()

        self.core = nn.Sequential(

            nn.Linear(
                hidden_dim,
                hidden_dim * 2
            ),

            nn.GELU(),

            nn.Linear(
                hidden_dim * 2,
                hidden_dim
            ),

            nn.GELU()
        )

        self.direction = nn.Linear(
            hidden_dim,
            4
        )

        self.risk = nn.Linear(
            hidden_dim,
            1
        )

        self.reachability = nn.Linear(
            hidden_dim,
            1
        )

    def forward(self, x):

        reasoning = self.core(x)

        return {

            "state": reasoning,

            "direction_logits":
                self.direction(reasoning),

            "risk":
                torch.sigmoid(
                    self.risk(reasoning)
                ),

            "reachable":
                torch.sigmoid(
                    self.reachability(reasoning)
                )
        }


# ======================================================================
# PLANNER
# ======================================================================

class NeuralPlanner(nn.Module):

    """
    Learned local planner.

    Predicts the next action.

    IMPORTANT:
        This does NOT call A*.
    """

    def __init__(self, hidden_dim):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                hidden_dim,
                hidden_dim
            ),

            nn.GELU(),

            nn.Linear(
                hidden_dim,
                hidden_dim
            ),

            nn.GELU(),

            nn.Linear(
                hidden_dim,
                len(ACTION_LIST)
            )
        )

    def forward(self, reasoning_state):

        return self.network(
            reasoning_state
        )


# ======================================================================
# WORLD MODEL
# ======================================================================

class WorldModel(nn.Module):

    """
    Predicts the next latent state.

    Given:

        current latent state
        action

    predicts:

        next latent state
        reward
        termination probability
    """

    def __init__(self, hidden_dim):

        super().__init__()

        self.action_embedding = nn.Embedding(
            len(ACTION_LIST),
            hidden_dim
        )

        self.transition = nn.Sequential(

            nn.Linear(
                hidden_dim * 2,
                hidden_dim * 2
            ),

            nn.GELU(),

            nn.Linear(
                hidden_dim * 2,
                hidden_dim
            ),

            nn.GELU()
        )

        self.reward_head = nn.Linear(
            hidden_dim,
            1
        )

        self.done_head = nn.Linear(
            hidden_dim,
            1
        )

    def forward(
        self,
        state,
        action
    ):

        action_embedding = (
            self.action_embedding(action)
        )

        x = torch.cat(
            [state, action_embedding],
            dim=-1
        )

        next_state = self.transition(x)

        reward = self.reward_head(
            next_state
        )

        done = torch.sigmoid(
            self.done_head(
                next_state
            )
        )

        return (
            next_state,
            reward,
            done
        )


# ======================================================================
# RL POLICY
# ======================================================================

class RLPolicy(nn.Module):

    """
    Policy + value network.

    Policy:
        chooses actions.

    Value:
        estimates future return.
    """

    def __init__(self, hidden_dim):

        super().__init__()

        self.policy = nn.Sequential(

            nn.Linear(
                hidden_dim,
                hidden_dim
            ),

            nn.GELU(),

            nn.Linear(
                hidden_dim,
                len(ACTION_LIST)
            )
        )

        self.value = nn.Sequential(

            nn.Linear(
                hidden_dim,
                hidden_dim
            ),

            nn.GELU(),

            nn.Linear(
                hidden_dim,
                1
            )
        )

    def forward(self, state):

        logits = self.policy(
            state
        )

        value = self.value(
            state
        )

        return logits, value


# ======================================================================
# ARIA FINAL
# ======================================================================

class ARIAFinal(nn.Module):

    """
    Complete ARIA architecture.

    Vision
       ↓
    Language
       ↓
    Fusion
       ↓
    Reasoning
       ↓
    Planning
       ↓
    World Model
       ↓
    Imagination
       ↓
    RL
       ↓
    Action
    """

    def __init__(self, cfg):

        super().__init__()

        self.cfg = cfg

        h = cfg.hidden_dim

        self.visual_encoder = VisualEncoder(h)

        self.language_encoder = LanguageEncoder(h)

        self.fusion = MultimodalFusion(h)

        self.reasoner = NeuralReasoner(h)

        self.planner = NeuralPlanner(h)

        self.world_model = WorldModel(h)

        self.rl_policy = RLPolicy(h)

        self.memory = EpisodicMemory(
            cfg.memory_size
        )

    # ------------------------------------------------------------------
    # GRID ENCODING
    # ------------------------------------------------------------------

    def encode_grid(
        self,
        observation
    ):

        size = self.cfg.grid_size

        grid = np.zeros(
            (7, size, size),
            dtype=np.float32
        )

        # Channels:
        #
        # 0 = agent
        # 1 = obstacles
        # 2 = red
        # 3 = green
        # 4 = blue
        # 5 = yellow
        # 6 = target

        ax, ay = observation["agent"]

        grid[
            0,
            ay,
            ax
        ] = 1.0

        for x, y in observation["obstacles"]:

            grid[
                1,
                y,
                x
            ] = 1.0

        color_channel = {

            "red": 2,

            "green": 3,

            "blue": 4,

            "yellow": 5
        }

        for color, pos in observation["objects"].items():

            x, y = pos

            grid[
                color_channel[color],
                y,
                x
            ] = 1.0

        gx, gy = observation["goal"]

        grid[
            6,
            gy,
            gx
        ] = 1.0

        return torch.tensor(
            grid,
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(0)

    # ------------------------------------------------------------------
    # TARGET INDEX
    # ------------------------------------------------------------------

    def target_index(self, target):

        colors = [
            "red",
            "green",
            "blue",
            "yellow"
        ]

        return colors.index(
            target
        )

    # ------------------------------------------------------------------
    # DEVICE
    # ------------------------------------------------------------------

    @property
    def device(self):

        return next(
            self.parameters()
        ).device

    # ------------------------------------------------------------------
    # PERCEPTION
    # ------------------------------------------------------------------

    def perceive(
        self,
        observation
    ):

        visual = self.visual_encoder(
            self.encode_grid(
                observation
            )
        )

        target = torch.tensor(
            [self.target_index(
                observation["target"]
            )],
            device=self.device
        )

        language = self.language_encoder(
            target
        )

        fused = self.fusion(
            visual,
            language
        )

        return fused

    # ------------------------------------------------------------------
    # REASON
    # ------------------------------------------------------------------

    def reason(
        self,
        observation
    ):

        latent = self.perceive(
            observation
        )

        reasoning = self.reasoner(
            latent
        )

        return reasoning

    # ------------------------------------------------------------------
    # PLAN
    # ------------------------------------------------------------------

    def plan(
        self,
        observation
    ):

        reasoning = self.reason(
            observation
        )

        planner_logits = self.planner(
            reasoning["state"]
        )

        return (
            planner_logits,
            reasoning
        )

    # ------------------------------------------------------------------
    # WORLD MODEL IMAGINATION
    # ------------------------------------------------------------------

    @torch.no_grad()
    def imagine(
        self,
        latent,
        action_sequence
    ):

        state = latent

        total_reward = 0.0

        discount = 1.0

        trajectory = []

        for action_idx in action_sequence:

            action = torch.tensor(
                [action_idx],
                dtype=torch.long,
                device=self.device
            )

            state, reward, done = (
                self.world_model(
                    state,
                    action
                )
            )

            reward_value = (
                reward.item()
            )

            total_reward += (
                discount *
                reward_value
            )

            discount *= self.cfg.gamma

            trajectory.append({

                "action": action_idx,

                "reward": reward_value,

                "done": done.item(),

                "state": state.clone()
            })

            if done.item() > 0.5:
                break

        return (
            total_reward,
            trajectory
        )

    # ------------------------------------------------------------------
    # MODEL-BASED ACTION SELECTION
    # ------------------------------------------------------------------

    @torch.no_grad()
    def select_action(
        self,
        observation
    ):

        latent = self.perceive(
            observation
        )

        reasoning = self.reasoner(
            latent
        )

        planner_logits = self.planner(
            reasoning["state"]
        )

        rl_logits, value = (
            self.rl_policy(
                reasoning["state"]
            )
        )

        # --------------------------------------------------------------
        # Candidate actions
        # --------------------------------------------------------------

        planner_probs = F.softmax(
            planner_logits,
            dim=-1
        )

        rl_probs = F.softmax(
            rl_logits,
            dim=-1
        )

        combined = (
            0.5 * planner_probs
            +
            0.5 * rl_probs
        )

        # --------------------------------------------------------------
        # Imagination
        # --------------------------------------------------------------

        candidates = torch.topk(
            combined,
            k=min(
                3,
                len(ACTION_LIST)
            ),
            dim=-1
        ).indices[0]

        best_action = None
        best_score = -float("inf")

        for action_idx in candidates:

            sequence = [
                int(action_idx)
            ]

            # Short imagined continuation
            for _ in range(
                self.cfg.imagination_horizon - 1
            ):

                next_action = int(
                    torch.argmax(
                        combined
                    ).item()
                )

                sequence.append(
                    next_action
                )

            score, trajectory = (
                self.imagine(
                    latent,
                    sequence
                )
            )

            # Combine imagination + policy value
            score += (
                0.1 *
                value.item()
            )

            if score > best_score:

                best_score = score

                best_action = int(
                    action_idx
                )

        return (
            index_to_action(
                best_action
            ),
            {
                "action_index": best_action,

                "planner_confidence":
                    planner_probs.max().item(),

                "policy_confidence":
                    rl_probs.max().item(),

                "value":
                    value.item(),

                "imagination_score":
                    best_score,

                "risk":
                    reasoning["risk"].item(),

                "reachable":
                    reasoning[
                        "reachable"
                    ].item()
            }
        )

    # ------------------------------------------------------------------
    # FULL STEP
    # ------------------------------------------------------------------

    def act(
        self,
        observation
    ):

        action, info = (
            self.select_action(
                observation
            )
        )

        return action, info


# ======================================================================
# PARAMETER REPORT
# ======================================================================

def count_parameters(model):

    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


# ======================================================================
# SANITY TESTS
# ======================================================================

def sanity_tests(
    model,
    env
):

    print()
    print("=" * 70)
    print("ARIA FINAL SANITY TESTS")
    print("=" * 70)

    observation = env.reset()

    # Environment
    assert observation["agent"] is not None
    assert observation["goal"] is not None

    print("✓ Environment")

    # Vision
    latent = model.perceive(
        observation
    )

    assert latent.shape == (
        1,
        model.cfg.hidden_dim
    )

    print("✓ Vision + language + fusion")

    # Reasoning
    reasoning = model.reason(
        observation
    )

    assert "state" in reasoning
    assert "risk" in reasoning
    assert "reachable" in reasoning

    print("✓ Neural reasoning")

    # Planner
    planner_logits, _ = model.plan(
        observation
    )

    assert planner_logits.shape[-1] == (
        len(ACTION_LIST)
    )

    print("✓ Neural planner")

    # World model
    action = torch.tensor(
        [0],
        dtype=torch.long,
        device=model.device
    )

    next_state, reward, done = (
        model.world_model(
            latent,
            action
        )
    )

    assert next_state.shape == latent.shape

    print("✓ World model")

    # RL
    logits, value = model.rl_policy(
        latent
    )

    assert logits.shape[-1] == len(
        ACTION_LIST
    )

    assert value.shape == (1, 1)

    print("✓ RL policy + value")

    # Action
    action, info = model.act(
        observation
    )

    assert action in ACTION_LIST

    print("✓ End-to-end action")

    print()
    print("All sanity tests passed.")


# ======================================================================
# DEMONSTRATION
# ======================================================================

def demonstration(
    model,
    env
):

    print()
    print("=" * 70)
    print("ARIA FINAL DEMONSTRATION")
    print("=" * 70)

    observation = env.reset()

    print()
    print(
        "Instruction:",
        observation["instruction"]
    )

    env.render()

    total_reward = 0.0

    for step in range(
        env.cfg.max_steps
    ):

        action, info = (
            model.act(
                observation
            )
        )

        (
            observation,
            reward,
            done,
            env_info
        ) = env.step(
            action
        )

        total_reward += reward

        print(
            f"Step {step + 1:03d} | "
            f"Action={action} | "
            f"Distance={env_info['distance']:.2f} | "
            f"Risk={info['risk']:.3f} | "
            f"Value={info['value']:.3f} | "
            f"Imagine={info['imagination_score']:.3f}"
        )

        if done:
            break

    env.render()

    print()
    print(
        "Success:",
        env_info["success"]
    )

    print(
        "Steps:",
        env_info["steps"]
    )

    print(
        "Final Distance:",
        f"{env_info['distance']:.3f}"
    )

    print(
        "Total Reward:",
        f"{total_reward:.3f}"
    )


# ======================================================================
# RANDOM BASELINE EVALUATION
# ======================================================================

def evaluate(
    model,
    env,
    episodes=50
):

    successes = 0

    collisions = 0

    total_steps = 0

    rewards = []

    distances = []

    for episode in range(
        episodes
    ):

        observation = env.reset()

        episode_reward = 0.0

        episode_collisions = 0

        for _ in range(
            env.cfg.max_steps
        ):

            action, _ = model.act(
                observation
            )

            (
                observation,
                reward,
                done,
                info
            ) = env.step(
                action
            )

            episode_reward += reward

            if info["collision"]:
                episode_collisions += 1

            if done:
                break

        if info["success"]:
            successes += 1

        collisions += episode_collisions

        total_steps += info["steps"]

        rewards.append(
            episode_reward
        )

        distances.append(
            info["distance"]
        )

    print()
    print("=" * 70)
    print("ARIA FINAL EVALUATION")
    print("=" * 70)

    print(
        f"Episodes             : {episodes}"
    )

    print(
        f"Average Reward       : "
        f"{np.mean(rewards):.4f}"
    )

    print(
        f"Navigation Success   : "
        f"{successes / episodes * 100:.2f}%"
    )

    print(
        f"Collision Rate       : "
        f"{collisions / max(total_steps, 1) * 100:.2f}%"
    )

    print(
        f"Final Distance       : "
        f"{np.mean(distances):.4f}"
    )

    print(
        f"Average Steps        : "
        f"{total_steps / episodes:.2f}"
    )

    print("=" * 70)

    return {

        "reward":
            float(np.mean(rewards)),

        "success_rate":
            successes / episodes,

        "collision_rate":
            collisions / max(
                total_steps,
                1
            ),

        "final_distance":
            float(np.mean(distances)),

        "average_steps":
            total_steps / episodes
    }


# ======================================================================
# MAIN
# ======================================================================

def main():

    print()
    print("=" * 70)
    print(
        "ARIA FINAL"
    )
    print(
        "Adaptive Reasoning & Imagination Agent"
    )
    print("=" * 70)

    cfg = Config()

    set_seed(
        cfg.seed
    )

    print(
        f"Device: {cfg.device}"
    )

    print(
        f"Grid: "
        f"{cfg.grid_size}x{cfg.grid_size}"
    )

    print(
        f"Obstacles: "
        f"{cfg.num_obstacles}"
    )

    print(
        f"Hidden dimension: "
        f"{cfg.hidden_dim}"
    )

    # --------------------------------------------------------------
    # Environment
    # --------------------------------------------------------------

    env = ARIAEnvironment(
        cfg
    )

    # --------------------------------------------------------------
    # Model
    # --------------------------------------------------------------

    model = ARIAFinal(
        cfg
    ).to(
        cfg.device
    )

    print(
        f"Trainable parameters: "
        f"{count_parameters(model):,}"
    )

    # --------------------------------------------------------------
    # Tests
    # --------------------------------------------------------------

    sanity_tests(
        model,
        env
    )

    # --------------------------------------------------------------
    # Demonstration
    # --------------------------------------------------------------

    demonstration(
        model,
        env
    )

    # --------------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------------

    evaluate(
        model,
        env,
        episodes=50
    )

    # --------------------------------------------------------------
    # Architecture report
    # --------------------------------------------------------------

    print()
    print("=" * 70)
    print("ARIA FINAL ARCHITECTURE")
    print("=" * 70)

    print(
        """
Vision
  ↓
Language
  ↓
Multimodal Fusion
  ↓
Episodic Memory
  ↓
Neural Reasoning
  ↓
Neural Planning
  ↓
World Model
  ↓
Imagination
  ↓
RL Policy + Value
  ↓
Action
  ↓
Environment
  ↓
Experience → Memory
        """
    )

    print("=" * 70)
    print(
        "ARIA FINAL PROTOTYPE COMPLETE"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()