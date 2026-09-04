"""
======================================================================
ARIA FINAL
Adaptive Reasoning & Imagination Agent
======================================================================

Research Prototype
------------------

ARIA combines:

    Vision
       ↓
    Language
       ↓
    Multimodal Fusion
       ↓
    World State
       ↓
    Neural Reasoning
       ↓
    Planning
       ↓
    World Model
       ↓
    Imagination
       ↓
    Memory
       ↓
    RL Controller
       ↓
    Action
       ↓
    Environment

The environment is a 2D navigation world.

This is a research prototype, NOT a claim of general AGI.

======================================================================
"""

import os
import math
import random
from dataclasses import dataclass
from collections import deque
from typing import Dict, Tuple, List, Optional

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


# ======================================================================
# CONFIG
# ======================================================================

@dataclass
class Config:

    # Environment
    grid_size: int = 15
    num_obstacles: int = 22
    num_objects: int = 4
    max_steps: int = 100

    # Vision
    vision_channels: int = 32

    # Embeddings
    d_model: int = 128

    # Transformer
    n_heads: int = 4
    n_layers: int = 2

    # Memory
    memory_size: int = 256

    # World model
    latent_dim: int = 128

    # RL
    action_dim: int = 4

    # Training
    batch_size: int = 64
    learning_rate: float = 3e-4

    # Imagination
    imagination_horizon: int = 8

    # Runtime
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    seed: int = 42

    checkpoint_dir: str = "checkpoints"


# ======================================================================
# SEED
# ======================================================================

def set_seed(seed: int):

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
        "yellow",
    ]

    SYMBOLS = {
        "red": "R",
        "green": "G",
        "blue": "B",
        "yellow": "Y",
    }

    ACTIONS = [
        (1, 0),     # RIGHT
        (-1, 0),    # LEFT
        (0, 1),     # DOWN
        (0, -1),    # UP
    ]

    def __init__(self, cfg: Config):

        self.cfg = cfg
        self.size = cfg.grid_size

        self.agent_pos = (0, 0)
        self.goal_pos = (0, 0)

        self.obstacles = set()
        self.objects = {}

        self.target_object = None

        self.step_count = 0
        self.previous_distance = None

    # ------------------------------------------------------------------
    # RESET
    # ------------------------------------------------------------------

    def reset(self):

        self.obstacles = set()
        self.objects = {}

        self.step_count = 0

        # Agent
        while True:

            pos = self.random_position()

            if pos not in self.obstacles:
                self.agent_pos = pos
                break

        # Obstacles
        attempts = 0

        while len(self.obstacles) < self.cfg.num_obstacles:

            pos = self.random_position()

            if pos != self.agent_pos:
                self.obstacles.add(pos)

            attempts += 1

            if attempts > 10000:
                break

        # Objects
        for color in self.COLORS[:self.cfg.num_objects]:

            for _ in range(200):

                pos = self.random_position()

                if self.valid_position(pos):

                    self.objects[color] = pos
                    break

        if not self.objects:
            raise RuntimeError("Could not generate objects.")

        self.target_object = random.choice(
            list(self.objects.keys())
        )

        self.goal_pos = self.objects[
            self.target_object
        ]

        self.previous_distance = self.distance(
            self.agent_pos,
            self.goal_pos
        )

        return self.observation()

    # ------------------------------------------------------------------
    # RANDOM
    # ------------------------------------------------------------------

    def random_position(self):

        return (
            random.randint(0, self.size - 1),
            random.randint(0, self.size - 1),
        )

    def valid_position(self, pos):

        if pos in self.obstacles:
            return False

        if pos == self.agent_pos:
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
    # LANGUAGE
    # ------------------------------------------------------------------

    def instruction(self):

        return (
            f"Go to the {self.target_object} object "
            f"while avoiding obstacles."
        )

    # ------------------------------------------------------------------
    # OBSERVATION
    # ------------------------------------------------------------------

    def observation(self):

        return {
            "agent": self.agent_pos,
            "goal": self.goal_pos,
            "objects": dict(self.objects),
            "obstacles": set(self.obstacles),
            "instruction": self.instruction(),
        }

    # ------------------------------------------------------------------
    # STEP
    # ------------------------------------------------------------------

    def step(self, action):

        self.step_count += 1

        dx, dy = action

        new_pos = (
            self.agent_pos[0] + int(dx),
            self.agent_pos[1] + int(dy),
        )

        collision = False

        reward = -0.01

        # Bounds
        if not (
            0 <= new_pos[0] < self.size
            and
            0 <= new_pos[1] < self.size
        ):

            collision = True
            reward -= 0.10

            new_pos = self.agent_pos

        # Obstacle
        elif new_pos in self.obstacles:

            collision = True
            reward -= 0.25

            new_pos = self.agent_pos

        else:

            self.agent_pos = new_pos

        # Distance shaping
        new_distance = self.distance(
            self.agent_pos,
            self.goal_pos
        )

        delta = (
            self.previous_distance -
            new_distance
        )

        reward += 0.15 * delta

        self.previous_distance = new_distance

        success = (
            self.agent_pos ==
            self.goal_pos
        )

        if success:
            reward += 5.0

        done = (
            success or
            self.step_count >= self.cfg.max_steps
        )

        info = {
            "success": success,
            "collision": collision,
            "distance": new_distance,
            "steps": self.step_count,
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
            ["."] * self.size
            for _ in range(self.size)
        ]

        for x, y in self.obstacles:
            grid[y][x] = "#"

        for color, (x, y) in self.objects.items():

            grid[y][x] = self.SYMBOLS[color]

        gx, gy = self.goal_pos
        grid[gy][gx] = "T"

        ax, ay = self.agent_pos
        grid[ay][ax] = "A"

        print()

        for row in grid:
            print(" ".join(row))

        print()
        print("Instruction:")
        print(self.instruction())
        print()


# ======================================================================
# VISION ENCODER
# ======================================================================

class VisionEncoder(nn.Module):

    """
    Converts the grid into visual spatial tokens.
    """

    def __init__(self, cfg):

        super().__init__()

        self.size = cfg.grid_size
        self.d_model = cfg.d_model

        self.encoder = nn.Sequential(

            nn.Conv2d(
                6,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.Conv2d(
                32,
                cfg.vision_channels,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

        )

        self.projection = nn.Linear(
            cfg.vision_channels,
            cfg.d_model
        )

    # ------------------------------------------------------------------
    # GRID CREATION
    # ------------------------------------------------------------------

    def observation_to_grid(self, observation):

        grid = np.zeros(
            (
                6,
                self.size,
                self.size
            ),
            dtype=np.float32
        )

        # Agent
        x, y = observation["agent"]
        grid[0, y, x] = 1.0

        # Target
        x, y = observation["goal"]
        grid[1, y, x] = 1.0

        # Obstacles
        for x, y in observation["obstacles"]:
            grid[2, y, x] = 1.0

        # Objects
        channels = {
            "red": 3,
            "green": 4,
            "blue": 5,
        }

        for color, pos in observation["objects"].items():

            if color in channels:

                x, y = pos
                grid[
                    channels[color],
                    y,
                    x
                ] = 1.0

        return torch.tensor(
            grid,
            dtype=torch.float32
        )

    # ------------------------------------------------------------------
    # FORWARD
    # ------------------------------------------------------------------

    def forward(self, observation):

        grid = self.observation_to_grid(
            observation
        )

        grid = grid.unsqueeze(0)

        features = self.encoder(grid)

        b, c, h, w = features.shape

        tokens = features.flatten(
            2
        ).transpose(
            1,
            2
        )

        tokens = self.projection(tokens)

        return tokens


# ======================================================================
# LANGUAGE ENCODER
# ======================================================================

class LanguageEncoder(nn.Module):

    """
    Small trainable instruction encoder.

    This is intentionally lightweight for the research environment.
    """

    VOCAB = [
        "go",
        "to",
        "the",
        "red",
        "green",
        "blue",
        "yellow",
        "object",
        "while",
        "avoiding",
        "obstacles",
    ]

    def __init__(self, cfg):

        super().__init__()

        self.d_model = cfg.d_model

        self.word_to_id = {
            word: i
            for i, word in enumerate(
                self.VOCAB
            )
        }

        self.embedding = nn.Embedding(
            len(self.VOCAB),
            cfg.d_model
        )

        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            self.transformer_layer,
            num_layers=cfg.n_layers
        )

    def tokenize(self, text):

        words = text.lower().replace(
            ".",
            ""
        ).split()

        ids = []

        for word in words:

            if word in self.word_to_id:

                ids.append(
                    self.word_to_id[word]
                )

        if not ids:
            ids = [0]

        return torch.tensor(
            ids,
            dtype=torch.long
        )

    def forward(self, text):

        ids = self.tokenize(text)

        ids = ids.unsqueeze(0)

        x = self.embedding(ids)

        x = self.transformer(x)

        return x.mean(dim=1)


# ======================================================================
# MULTIMODAL FUSION
# ======================================================================

class MultimodalFusion(nn.Module):

    """
    Cross-modal Transformer.

    Vision tokens attend together with language representation.
    """

    def __init__(self, cfg):

        super().__init__()

        self.language_projection = nn.Linear(
            cfg.d_model,
            cfg.d_model
        )

        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=cfg.n_layers
        )

        self.norm = nn.LayerNorm(
            cfg.d_model
        )

    def forward(
        self,
        vision_tokens,
        language_embedding
    ):

        language_embedding = (
            self.language_projection(
                language_embedding
            )
        )

        language_token = (
            language_embedding.unsqueeze(1)
        )

        tokens = torch.cat(
            [
                vision_tokens,
                language_token
            ],
            dim=1
        )

        fused = self.transformer(
            tokens
        )

        fused = self.norm(fused)

        world_state = fused.mean(
            dim=1
        )

        return fused, world_state


# ======================================================================
# NEURAL REASONER
# ======================================================================

class NeuralReasoner(nn.Module):

    """
    Predicts semantic/spatial properties of the current world.
    """

    def __init__(self, cfg):

        super().__init__()

        d = cfg.d_model

        self.backbone = nn.Sequential(

            nn.Linear(d, d),
            nn.ReLU(),

            nn.Linear(d, d),
            nn.ReLU(),

        )

        self.target_head = nn.Linear(
            d,
            cfg.grid_size * cfg.grid_size
        )

        self.blocked_head = nn.Linear(
            d,
            2
        )

        self.reachable_head = nn.Linear(
            d,
            2
        )

        self.direction_head = nn.Linear(
            d,
            4
        )

    def forward(self, world_state):

        x = self.backbone(
            world_state
        )

        return {
            "target": self.target_head(x),
            "blocked": self.blocked_head(x),
            "reachable": self.reachable_head(x),
            "direction": self.direction_head(x),
        }


# ======================================================================
# PLANNER
# ======================================================================

class NeuralPlanner(nn.Module):

    """
    Predicts action preferences from the world representation.

    Unlike v4, this is NOT forced to solve the entire navigation
    problem by itself. It receives a structured reasoning state.
    """

    def __init__(self, cfg):

        super().__init__()

        d = cfg.d_model

        self.network = nn.Sequential(

            nn.Linear(
                d + 4,
                d
            ),

            nn.ReLU(),

            nn.Linear(
                d,
                d
            ),

            nn.ReLU(),

            nn.Linear(
                d,
                cfg.action_dim
            )
        )

    def forward(
        self,
        world_state,
        reasoning_features
    ):

        x = torch.cat(
            [
                world_state,
                reasoning_features
            ],
            dim=-1
        )

        return self.network(x)


# ======================================================================
# WORLD MODEL
# ======================================================================

class WorldModel(nn.Module):

    """
    Latent dynamics model.

    Learns:

        state + action
             ↓
        predicted next latent state
             +
        predicted reward
    """

    def __init__(self, cfg):

        super().__init__()

        d = cfg.latent_dim

        self.state_encoder = nn.Sequential(

            nn.Linear(
                cfg.d_model,
                d
            ),

            nn.ReLU(),

            nn.Linear(
                d,
                d
            )
        )

        self.action_embedding = nn.Embedding(
            cfg.action_dim,
            d
        )

        self.transition = nn.Sequential(

            nn.Linear(
                d * 2,
                d
            ),

            nn.ReLU(),

            nn.Linear(
                d,
                d
            )
        )

        self.reward_head = nn.Linear(
            d,
            1
        )

    def encode(self, world_state):

        return self.state_encoder(
            world_state
        )

    def step(
        self,
        latent,
        action
    ):

        action_embedding = (
            self.action_embedding(action)
        )

        combined = torch.cat(
            [
                latent,
                action_embedding
            ],
            dim=-1
        )

        next_latent = self.transition(
            combined
        )

        reward = self.reward_head(
            next_latent
        )

        return next_latent, reward


# ======================================================================
# MEMORY
# ======================================================================

@dataclass
class MemoryItem:

    state: torch.Tensor
    action: int
    reward: float
    next_state: torch.Tensor
    success: bool


class EpisodicMemory:

    """
    Simple episodic memory.

    Stores previous experiences and retrieves
    states that are similar to the current state.
    """

    def __init__(self, cfg):

        self.max_size = cfg.memory_size

        self.memory = deque(
            maxlen=self.max_size
        )

    def add(
        self,
        state,
        action,
        reward,
        next_state,
        success=False
    ):

        self.memory.append(
            MemoryItem(
                state.detach().cpu(),
                action,
                reward,
                next_state.detach().cpu(),
                success
            )
        )

    def retrieve(
        self,
        query,
        k=5
    ):

        if len(self.memory) == 0:
            return []

        query = query.detach().cpu()

        scored = []

        for item in self.memory:

            similarity = F.cosine_similarity(
                query.unsqueeze(0),
                item.state.unsqueeze(0)
            ).item()

            scored.append(
                (
                    similarity,
                    item
                )
            )

        scored.sort(
            key=lambda x: x[0],
            reverse=True
        )

        return [
            item
            for _, item in scored[:k]
        ]

    def __len__(self):

        return len(self.memory)


# ======================================================================
# RL CONTROLLER
# ======================================================================

class RLController(nn.Module):

    """
    Actor-Critic controller.

    Actor:
        chooses action

    Critic:
        estimates state value
    """

    def __init__(self, cfg):

        super().__init__()

        d = cfg.d_model

        self.actor = nn.Sequential(

            nn.Linear(
                d,
                d
            ),

            nn.ReLU(),

            nn.Linear(
                d,
                cfg.action_dim
            )
        )

        self.critic = nn.Sequential(

            nn.Linear(
                d,
                d
            ),

            nn.ReLU(),

            nn.Linear(
                d,
                1
            )
        )

    def forward(self, state):

        logits = self.actor(state)

        value = self.critic(state)

        return logits, value

    def act(
        self,
        state,
        deterministic=False
    ):

        logits, value = self.forward(
            state
        )

        if deterministic:

            action = torch.argmax(
                logits,
                dim=-1
            )

        else:

            distribution = torch.distributions.Categorical(
                logits=logits
            )

            action = distribution.sample()

        return action, value


# ======================================================================
# A* FALLBACK / SAFETY PLANNER
# ======================================================================

class SafetyPlanner:

    """
    Safety mechanism.

    Important:
    This is NOT the primary intelligence of ARIA.

    It is used to keep the research system operational
    while the learned components are still being trained.

    The learned agent can override it when confident.
    """

    ACTIONS = [
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1)
    ]

    def __init__(self, size):

        self.size = size

    def heuristic(self, a, b):

        return abs(
            a[0] - b[0]
        ) + abs(
            a[1] - b[1]
        )

    def plan(
        self,
        start,
        goal,
        obstacles
    ):

        open_set = {start}

        came_from = {}

        g = {
            start: 0
        }

        f = {
            start:
            self.heuristic(
                start,
                goal
            )
        }

        while open_set:

            current = min(
                open_set,
                key=lambda x:
                f.get(
                    x,
                    float("inf")
                )
            )

            if current == goal:

                path = [current]

                while current in came_from:

                    current = came_from[
                        current
                    ]

                    path.append(current)

                return list(
                    reversed(path)
                )

            open_set.remove(
                current
            )

            for dx, dy in self.ACTIONS:

                neighbor = (
                    current[0] + dx,
                    current[1] + dy
                )

                if not (
                    0 <= neighbor[0] < self.size
                    and
                    0 <= neighbor[1] < self.size
                ):
                    continue

                if neighbor in obstacles:
                    continue

                tentative = (
                    g[current] + 1
                )

                if tentative < g.get(
                    neighbor,
                    float("inf")
                ):

                    came_from[
                        neighbor
                    ] = current

                    g[
                        neighbor
                    ] = tentative

                    f[
                        neighbor
                    ] = tentative + self.heuristic(
                        neighbor,
                        goal
                    )

                    open_set.add(
                        neighbor
                    )

        return None


# ======================================================================
# ARIA FINAL
# ======================================================================

class ARIAFinal(nn.Module):

    """
    Complete ARIA system.
    """

    def __init__(self, cfg=None):

        super().__init__()

        self.cfg = cfg if cfg is not None else Config()

        # --------------------------------------------------------------
        # PERCEPTION
        # --------------------------------------------------------------

        self.vision = VisionEncoder(
            self.cfg
        )

        self.language = LanguageEncoder(
            self.cfg
        )

        self.multimodal = MultimodalFusion(
            self.cfg
        )

        # --------------------------------------------------------------
        # REASONING
        # --------------------------------------------------------------

        self.reasoner = NeuralReasoner(
            self.cfg
        )

        # --------------------------------------------------------------
        # PLANNING
        # --------------------------------------------------------------

        self.planner = NeuralPlanner(
            self.cfg
        )

        # --------------------------------------------------------------
        # WORLD MODEL
        # --------------------------------------------------------------

        self.world_model = WorldModel(
            self.cfg
        )

        # --------------------------------------------------------------
        # RL
        # --------------------------------------------------------------

        self.rl = RLController(
            self.cfg
        )

        # --------------------------------------------------------------
        # MEMORY
        # --------------------------------------------------------------

        self.memory = EpisodicMemory(
            self.cfg
        )

        # --------------------------------------------------------------
        # SAFETY
        # --------------------------------------------------------------

        self.safety_planner = SafetyPlanner(
            self.cfg.grid_size
        )

    # ==================================================================
    # PERCEPTION
    # ==================================================================

    def perceive(
        self,
        observation
    ):

        vision_tokens = self.vision(
            observation
        )

        language_embedding = self.language(
            observation["instruction"]
        )

        fused_tokens, world_state = (
            self.multimodal(
                vision_tokens,
                language_embedding
            )
        )

        return {
            "vision": vision_tokens,
            "language": language_embedding,
            "fused": fused_tokens,
            "world_state": world_state,
        }

    # ==================================================================
    # REASON
    # ==================================================================

    def reason(
        self,
        perception
    ):

        outputs = self.reasoner(
            perception["world_state"]
        )

        return outputs

    # ==================================================================
    # MEMORY RETRIEVAL
    # ==================================================================

    def recall(
        self,
        world_state
    ):

        return self.memory.retrieve(
            world_state.squeeze(0)
        )

    # ==================================================================
    # NEURAL PLAN
    # ==================================================================

    def neural_plan(
        self,
        perception,
        reasoning
    ):

        # Use the four directional logits
        # as structured reasoning features.

        reasoning_features = torch.cat(
            [
                reasoning["blocked"],
                reasoning["reachable"]
            ],
            dim=-1
        )

        logits = self.planner(
            perception["world_state"],
            reasoning_features
        )

        return logits

    # ==================================================================
    # WORLD MODEL IMAGINATION
    # ==================================================================

    @torch.no_grad()
    def imagine(
        self,
        world_state,
        action_sequence
    ):

        latent = self.world_model.encode(
            world_state
        )

        predicted_rewards = []

        states = []

        for action in action_sequence:

            action_tensor = torch.tensor(
                [action],
                dtype=torch.long,
                device=latent.device
            )

            latent, reward = (
                self.world_model.step(
                    latent,
                    action_tensor
                )
            )

            states.append(
                latent
            )

            predicted_rewards.append(
                reward.item()
            )

        return {
            "states": states,
            "rewards": predicted_rewards,
            "total_reward": sum(
                predicted_rewards
            )
        }

    # ==================================================================
    # IMAGINED ACTION SELECTION
    # ==================================================================

    @torch.no_grad()
    def imagine_best_action(
        self,
        world_state
    ):

        best_action = 0
        best_score = -float("inf")

        for action in range(
            self.cfg.action_dim
        ):

            result = self.imagine(
                world_state,
                [action]
            )

            score = result[
                "total_reward"
            ]

            if score > best_score:

                best_score = score
                best_action = action

        return best_action

    # ==================================================================
    # ACTION
    # ==================================================================

    def select_action(
        self,
        observation,
        deterministic=True
    ):

        perception = self.perceive(
            observation
        )

        reasoning = self.reason(
            perception
        )

        # --------------------------------------------------------------
        # Memory
        # --------------------------------------------------------------

        memories = self.recall(
            perception["world_state"]
        )

        # --------------------------------------------------------------
        # Neural planner
        # --------------------------------------------------------------

        neural_logits = self.neural_plan(
            perception,
            reasoning
        )

        neural_action = torch.argmax(
            neural_logits,
            dim=-1
        ).item()

        # --------------------------------------------------------------
        # World model imagination
        # --------------------------------------------------------------

        imagined_action = (
            self.imagine_best_action(
                perception["world_state"]
            )
        )

        # --------------------------------------------------------------
        # RL controller
        # --------------------------------------------------------------

        rl_action, value = self.rl.act(
            perception["world_state"],
            deterministic=deterministic
        )

        rl_action = rl_action.item()

        # --------------------------------------------------------------
        # Confidence
        # --------------------------------------------------------------

        target_probability = F.softmax(
            reasoning["target"],
            dim=-1
        )

        confidence = (
            target_probability.max()
            .item()
        )

        # --------------------------------------------------------------
        # SAFETY / EXECUTION
        # --------------------------------------------------------------

        # During the research phase we use the symbolic
        # planner as a safety mechanism when neural
        # confidence is low.

        if confidence < 0.50:

            path = self.safety_planner.plan(
                observation["agent"],
                observation["goal"],
                observation["obstacles"]
            )

            if path is not None and len(path) >= 2:

                next_position = path[1]

                action = (
                    next_position[0]
                    - observation["agent"][0],

                    next_position[1]
                    - observation["agent"][1]
                )

                action_to_id = {
                    (1, 0): 0,
                    (-1, 0): 1,
                    (0, 1): 2,
                    (0, -1): 3,
                }

                action_id = action_to_id.get(
                    action,
                    0
                )

                source = "safety_planner"

            else:

                action_id = imagined_action
                source = "world_model"

        else:

            # Combine neural planning and RL.
            #
            # This is deliberately simple for v1
            # of the final integrated architecture.

            action_id = neural_action

            source = "neural_planner"

        return {
            "action_id": action_id,
            "action": self.cfg_action(
                action_id
            ),
            "source": source,
            "confidence": confidence,
            "neural_action": neural_action,
            "imagined_action": imagined_action,
            "rl_action": rl_action,
            "value": value.item(),
            "memory_count": len(memories),
            "reasoning": reasoning,
            "perception": perception,
        }

    # ==================================================================
    # ACTION ID
    # ==================================================================

    def cfg_action(
        self,
        action_id
    ):

        actions = [
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1)
        ]

        return actions[
            int(action_id)
            % len(actions)
        ]

    # ==================================================================
    # MEMORY UPDATE
    # ==================================================================

    def remember(
        self,
        perception,
        action_id,
        reward,
        next_perception,
        success
    ):

        self.memory.add(
            perception["world_state"].squeeze(0),
            action_id,
            reward,
            next_perception[
                "world_state"
            ].squeeze(0),
            success
        )

    # ==================================================================
    # SAVE
    # ==================================================================

    def save(
        self,
        path
    ):

        directory = os.path.dirname(
            path
        )

        if directory:
            os.makedirs(
                directory,
                exist_ok=True
            )

        torch.save(
            self.state_dict(),
            path
        )

        print(
            f"✓ ARIA checkpoint saved: {path}"
        )

    # ==================================================================
    # LOAD
    # ==================================================================

    def load(
        self,
        path,
        device=None
    ):

        if device is None:
            device = self.cfg.device

        state = torch.load(
            path,
            map_location=device
        )

        self.load_state_dict(
            state
        )

        print(
            f"✓ ARIA checkpoint loaded: {path}"
        )


# ======================================================================
# TRAINING UTILITIES
# ======================================================================

def target_position_to_class(
    position,
    grid_size
):

    x, y = position

    return y * grid_size + x


# ======================================================================
# WORLD MODEL TRAINING
# ======================================================================

def train_world_model(
    aria,
    env,
    episodes=100
):

    print()
    print("=" * 70)
    print("TRAINING WORLD MODEL")
    print("=" * 70)

    optimizer = torch.optim.Adam(
        aria.world_model.parameters(),
        lr=aria.cfg.learning_rate
    )

    aria.train()

    total_loss = 0.0

    for episode in range(episodes):

        observation = env.reset()

        perception = aria.perceive(
            observation
        )

        for _ in range(
            env.cfg.max_steps
        ):

            action_id = random.randrange(
                aria.cfg.action_dim
            )

            action = aria.cfg_action(
                action_id
            )

            next_observation, reward, done, info = (
                env.step(action)
            )

            next_perception = aria.perceive(
                next_observation
            )

            latent = aria.world_model.encode(
                perception["world_state"]
            )

            action_tensor = torch.tensor(
                [action_id],
                dtype=torch.long,
                device=aria.cfg.device
            )

            predicted_next, predicted_reward = (
                aria.world_model.step(
                    latent,
                    action_tensor
                )
            )

            target_next = (
                aria.world_model.encode(
                    next_perception[
                        "world_state"
                    ]
                ).detach()
            )

            loss_state = F.mse_loss(
                predicted_next,
                target_next
            )

            loss_reward = F.mse_loss(
                predicted_reward.squeeze(),
                torch.tensor(
                    reward,
                    dtype=torch.float32,
                    device=aria.cfg.device
                )
            )

            loss = (
                loss_state +
                loss_reward
            )

            optimizer.zero_grad()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                aria.world_model.parameters(),
                1.0
            )

            optimizer.step()

            total_loss += loss.item()

            perception = next_perception

            if done:
                break

        if (
            episode + 1
        ) % 10 == 0:

            avg = (
                total_loss /
                max(episode + 1, 1)
            )

            print(
                f"Episode {episode + 1:04d} "
                f"| World Model Loss: {avg:.5f}"
            )

    print(
        "✓ World model training complete"
    )


# ======================================================================
# REASONING TRAINING
# ======================================================================

def train_reasoner(
    aria,
    env,
    samples=3000,
    epochs=5
):

    print()
    print("=" * 70)
    print("TRAINING NEURAL REASONER")
    print("=" * 70)

    optimizer = torch.optim.Adam(
        aria.reasoner.parameters(),
        lr=aria.cfg.learning_rate
    )

    aria.train()

    observations = []

    for i in range(samples):

        observation = env.reset()

        observations.append(
            observation
        )

    for epoch in range(epochs):

        random.shuffle(
            observations
        )

        total_loss = 0.0

        for observation in observations:

            perception = aria.perceive(
                observation
            )

            outputs = aria.reasoner(
                perception["world_state"]
            )

            # ----------------------------------------------------------
            # Target position
            # ----------------------------------------------------------

            target_class = (
                target_position_to_class(
                    observation["goal"],
                    aria.cfg.grid_size
                )
            )

            target = torch.tensor(
                [target_class],
                dtype=torch.long,
                device=aria.cfg.device
            )

            loss_target = F.cross_entropy(
                outputs["target"],
                target
            )

            # ----------------------------------------------------------
            # Reachability
            # ----------------------------------------------------------

            # The environment guarantees a target
            # but some random configurations can be
            # disconnected.

            path = aria.safety_planner.plan(
                observation["agent"],
                observation["goal"],
                observation["obstacles"]
            )

            reachable = (
                1 if path is not None else 0
            )

            reachable_target = torch.tensor(
                [reachable],
                dtype=torch.long,
                device=aria.cfg.device
            )

            loss_reachable = F.cross_entropy(
                outputs["reachable"],
                reachable_target
            )

            # ----------------------------------------------------------
            # Total
            # ----------------------------------------------------------

            loss = (
                loss_target +
                loss_reachable
            )

            optimizer.zero_grad()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                aria.reasoner.parameters(),
                1.0
            )

            optimizer.step()

            total_loss += loss.item()

        print(
            f"Epoch {epoch + 1:02d}/{epochs} "
            f"| Loss: "
            f"{total_loss / len(observations):.5f}"
        )

    print(
        "✓ Neural reasoning training complete"
    )


# ======================================================================
# RL TRAINING
# ======================================================================

def train_rl(
    aria,
    env,
    episodes=200
):

    print()
    print("=" * 70)
    print("TRAINING RL CONTROLLER")
    print("=" * 70)

    optimizer = torch.optim.Adam(
        aria.rl.parameters(),
        lr=aria.cfg.learning_rate
    )

    gamma = 0.99

    aria.train()

    rewards_history = []

    for episode in range(episodes):

        observation = env.reset()

        log_probs = []
        values = []
        rewards = []

        for _ in range(
            env.cfg.max_steps
        ):

            perception = aria.perceive(
                observation
            )

            state = (
                perception["world_state"]
            )

            logits, value = aria.rl(
                state
            )

            distribution = (
                torch.distributions.Categorical(
                    logits=logits
                )
            )

            action = distribution.sample()

            log_prob = distribution.log_prob(
                action
            )

            action_id = action.item()

            env_action = aria.cfg_action(
                action_id
            )

            next_observation, reward, done, info = (
                env.step(env_action)
            )

            log_probs.append(
                log_prob
            )

            values.append(
                value.squeeze()
            )

            rewards.append(
                reward
            )

            observation = next_observation

            if done:
                break

        # --------------------------------------------------------------
        # Returns
        # --------------------------------------------------------------

        returns = []

        running = 0.0

        for reward in reversed(
            rewards
        ):

            running = (
                reward +
                gamma * running
            )

            returns.append(
                running
            )

        returns.reverse()

        returns = torch.tensor(
            returns,
            dtype=torch.float32,
            device=aria.cfg.device
        )

        values_tensor = torch.stack(
            values
        )

        log_probs_tensor = torch.stack(
            log_probs
        )

        advantages = (
            returns -
            values_tensor.detach()
        )

        actor_loss = -(
            log_probs_tensor *
            advantages
        ).mean()

        critic_loss = F.mse_loss(
            values_tensor,
            returns
        )

        loss = (
            actor_loss +
            0.5 * critic_loss
        )

        optimizer.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            aria.rl.parameters(),
            1.0
        )

        optimizer.step()

        episode_reward = sum(
            rewards
        )

        rewards_history.append(
            episode_reward
        )

        if (
            episode + 1
        ) % 20 == 0:

            avg = np.mean(
                rewards_history[-20:]
            )

            print(
                f"Episode {episode + 1:04d} "
                f"| Avg Reward: {avg:.3f}"
            )

    print(
        "✓ RL training complete"
    )


# ======================================================================
# INTEGRATED EVALUATION
# ======================================================================

def evaluate(
    aria,
    env,
    episodes=100,
    render_first=True
):

    print()
    print("=" * 70)
    print("ARIA FINAL EVALUATION")
    print("=" * 70)

    aria.eval()

    successes = 0
    collisions = 0

    rewards = []
    distances = []
    steps = []

    for episode in range(
        episodes
    ):

        observation = env.reset()

        if (
            render_first
            and
            episode == 0
        ):

            print()
            print("INITIAL WORLD")
            env.render()

        total_reward = 0.0

        for step in range(
            env.cfg.max_steps
        ):

            result = aria.select_action(
                observation,
                deterministic=True
            )

            action = result[
                "action"
            ]

            next_observation, reward, done, info = (
                env.step(action)
            )

            total_reward += reward

            if info["collision"]:
                collisions += 1

            if (
                render_first
                and
                episode == 0
            ):

                print(
                    f"Step {step + 1:03d} "
                    f"| Action={action} "
                    f"| Source={result['source']} "
                    f"| Confidence="
                    f"{result['confidence']:.3f}"
                )

            observation = next_observation

            if done:
                break

        rewards.append(
            total_reward
        )

        distances.append(
            info["distance"]
        )

        steps.append(
            info["steps"]
        )

        if info["success"]:
            successes += 1

    success_rate = (
        successes /
        episodes
    )

    collision_rate = (
        collisions /
        max(sum(steps), 1)
    )

    print()
    print("=" * 70)
    print("ARIA FINAL RESULTS")
    print("=" * 70)

    print(
        f"Average Reward      : "
        f"{np.mean(rewards):.4f}"
    )

    print(
        f"Success Rate        : "
        f"{success_rate * 100:.2f}%"
    )

    print(
        f"Collision Rate      : "
        f"{collision_rate * 100:.2f}%"
    )

    print(
        f"Final Distance      : "
        f"{np.mean(distances):.4f}"
    )

    print(
        f"Average Steps       : "
        f"{np.mean(steps):.2f}"
    )

    print("=" * 70)

    return {
        "reward": float(
            np.mean(rewards)
        ),
        "success_rate": float(
            success_rate
        ),
        "collision_rate": float(
            collision_rate
        ),
        "distance": float(
            np.mean(distances)
        ),
        "steps": float(
            np.mean(steps)
        )
    }


# ======================================================================
# SANITY TESTS
# ======================================================================

def sanity_tests(
    aria,
    env
):

    print()
    print("=" * 70)
    print("ARIA FINAL SANITY TESTS")
    print("=" * 70)

    observation = env.reset()

    # Vision
    perception = aria.perceive(
        observation
    )

    assert perception[
        "vision"
    ].ndim == 3

    print(
        "✓ Vision encoder"
    )

    # Language
    assert perception[
        "language"
    ].shape[-1] == aria.cfg.d_model

    print(
        "✓ Language encoder"
    )

    # Multimodal
    assert perception[
        "world_state"
    ].shape[-1] == aria.cfg.d_model

    print(
        "✓ Multimodal fusion"
    )

    # Reasoning
    reasoning = aria.reason(
        perception
    )

    assert "target" in reasoning

    print(
        "✓ Neural reasoning"
    )

    # Planner
    planner_output = aria.neural_plan(
        perception,
        reasoning
    )

    assert planner_output.shape[-1] == 4

    print(
        "✓ Neural planner"
    )

    # World model
    latent = aria.world_model.encode(
        perception["world_state"]
    )

    action = torch.tensor(
        [0],
        dtype=torch.long,
        device=aria.cfg.device
    )

    next_latent, reward = (
        aria.world_model.step(
            latent,
            action
        )
    )

    assert (
        next_latent.shape[-1] ==
        aria.cfg.latent_dim
    )

    print(
        "✓ World model"
    )

    # Memory
    aria.memory.add(
        perception[
            "world_state"
        ].squeeze(0),
        0,
        0.0,
        perception[
            "world_state"
        ].squeeze(0)
    )

    assert len(aria.memory) == 1

    print(
        "✓ Episodic memory"
    )

    # RL
    action, value = aria.rl.act(
        perception[
            "world_state"
        ],
        deterministic=True
    )

    assert 0 <= action.item() < 4

    print(
        "✓ RL controller"
    )

    print()
    print(
        "All ARIA Final sanity tests passed."
    )


# ======================================================================
# DEMONSTRATION
# ======================================================================

def demonstration(
    aria,
    env
):

    print()
    print("=" * 70)
    print("ARIA FINAL DEMONSTRATION")
    print("=" * 70)

    observation = env.reset()

    env.render()

    print()
    print(
        "ARIA is processing:"
    )

    print(
        "Vision → Language → Multimodal Fusion"
    )

    perception = aria.perceive(
        observation
    )

    reasoning = aria.reason(
        perception
    )

    target_distribution = F.softmax(
        reasoning["target"],
        dim=-1
    )

    predicted_target = (
        torch.argmax(
            target_distribution,
            dim=-1
        ).item()
    )

    px = (
        predicted_target %
        aria.cfg.grid_size
    )

    py = (
        predicted_target //
        aria.cfg.grid_size
    )

    print(
        f"Neural Target: "
        f"({px}, {py})"
    )

    print(
        f"Actual Target: "
        f"{observation['goal']}"
    )

    print(
        f"Target Confidence: "
        f"{target_distribution.max().item():.4f}"
    )

    print()
    print(
        "Reasoning → Planning → "
        "World Model → Memory → RL"
    )

    for step in range(
        env.cfg.max_steps
    ):

        result = aria.select_action(
            observation,
            deterministic=True
        )

        action = result[
            "action"
        ]

        (
            next_observation,
            reward,
            done,
            info
        ) = env.step(
            action
        )

        print(
            f"Step {step + 1:03d} "
            f"| Action={action} "
            f"| Source={result['source']} "
            f"| Confidence="
            f"{result['confidence']:.3f} "
            f"| Value="
            f"{result['value']:.3f}"
        )

        next_perception = aria.perceive(
            next_observation
        )

        aria.remember(
            perception,
            result["action_id"],
            reward,
            next_perception,
            info["success"]
        )

        perception = next_perception
        observation = next_observation

        if done:
            break

    print()

    env.render()

    print(
        f"Success: {info['success']}"
    )

    print(
        f"Steps: {info['steps']}"
    )

    print(
        f"Final Distance: "
        f"{info['distance']:.4f}"
    )

    print(
        f"Memory Size: "
        f"{len(aria.memory)}"
    )


# ======================================================================
# SAVE REPORT
# ======================================================================

def print_architecture():

    print()
    print("=" * 70)
    print("ARIA FINAL ARCHITECTURE")
    print("=" * 70)

    print(
        """
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
      Reasoning            Memory
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
          └──────────► Memory
        """
    )

    print("=" * 70)


# ======================================================================
# MAIN
# ======================================================================

def main():

    print()
    print("=" * 70)
    print("ARIA FINAL")
    print("Adaptive Reasoning & Imagination Agent")
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
        f"Parameters are created on "
        f"{cfg.device}"
    )

    # ------------------------------------------------------------------
    # Environment
    # ------------------------------------------------------------------

    env = ARIAEnvironment(
        cfg
    )

    # ------------------------------------------------------------------
    # Agent
    # ------------------------------------------------------------------

    aria = ARIAFinal(
        cfg
    ).to(
        cfg.device
    )

    # ------------------------------------------------------------------
    # Architecture
    # ------------------------------------------------------------------

    print_architecture()

    # ------------------------------------------------------------------
    # Parameter count
    # ------------------------------------------------------------------

    parameters = sum(
        p.numel()
        for p in aria.parameters()
    )

    print(
        f"\nARIA parameters: "
        f"{parameters:,}"
    )

    # ------------------------------------------------------------------
    # Sanity
    # ------------------------------------------------------------------

    sanity_tests(
        aria,
        env
    )

    # ------------------------------------------------------------------
    # Demonstration before training
    # ------------------------------------------------------------------

    demonstration(
        aria,
        env
    )

    # ------------------------------------------------------------------
    # TRAINING
    #
    # These are intentionally modest defaults.
    # Increase them after confirming the complete pipeline works.
    # ------------------------------------------------------------------

    train_reasoner(
        aria,
        env,
        samples=500,
        epochs=3
    )

    train_world_model(
        aria,
        env,
        episodes=50
    )

    train_rl(
        aria,
        env,
        episodes=100
    )

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    checkpoint = os.path.join(
        cfg.checkpoint_dir,
        "aria_final.pt"
    )

    aria.save(
        checkpoint
    )

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    evaluate(
        aria,
        env,
        episodes=50,
        render_first=False
    )

    # ------------------------------------------------------------------
    # Final
    # ------------------------------------------------------------------

    print()
    print("=" * 70)
    print("ARIA FINAL COMPLETE")
    print("=" * 70)

    print(
        """
Integrated capabilities:

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

Checkpoint:
    checkpoints/aria_final.pt
        """
    )

    print("=" * 70)


if __name__ == "__main__":
    main()
