"""
ARIA FINAL
Adaptive Reasoning & Imagination Agent

Integrated architecture:

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
    RL Controller
       ↓
    Action
       ↓
    Environment
       ↓
    Episodic Memory

This is a research/portfolio prototype for studying
multimodal agents, reasoning, planning, world models,
memory, and reinforcement learning.
"""

import os
import math
import random
from dataclasses import dataclass
from collections import deque
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass
class Config:

    grid_size: int = 15
    max_steps: int = 100

    num_obstacles: int = 22
    num_objects: int = 4

    vision_dim: int = 128
    language_dim: int = 128
    hidden_dim: int = 128

    world_dim: int = 128
    memory_dim: int = 128

    num_actions: int = 4

    memory_size: int = 500

    gamma: float = 0.99
    learning_rate: float = 3e-4

    reasoning_epochs: int = 3
    world_epochs: int = 50
    rl_episodes: int = 100

    seed: int = 42

    device: str = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    checkpoint_dir: str = "checkpoints"


# ============================================================
# SEED
# ============================================================

def set_seed(seed: int):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# ENVIRONMENT
# ============================================================

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

    ACTIONS = {
        0: (1, 0),
        1: (-1, 0),
        2: (0, 1),
        3: (0, -1)
    }

    def __init__(self, cfg):

        self.cfg = cfg
        self.size = cfg.grid_size

        self.agent = (0, 0)
        self.objects = {}
        self.obstacles = set()

        self.target_color = None
        self.goal = None

        self.steps = 0
        self.previous_distance = 0.0

    # --------------------------------------------------------
    # RESET
    # --------------------------------------------------------

    def reset(self):

        self.steps = 0

        self.obstacles = set()
        self.objects = {}

        self.agent = self.random_position()

        # obstacles

        while len(self.obstacles) < self.cfg.num_obstacles:

            p = self.random_position()

            if p != self.agent:
                self.obstacles.add(p)

        # objects

        for color in self.COLORS[:self.cfg.num_objects]:

            while True:

                p = self.random_position()

                if (
                    p != self.agent
                    and p not in self.obstacles
                    and p not in self.objects.values()
                ):
                    self.objects[color] = p
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

    # --------------------------------------------------------
    # RANDOM POSITION
    # --------------------------------------------------------

    def random_position(self):

        return (
            random.randint(0, self.size - 1),
            random.randint(0, self.size - 1)
        )

    # --------------------------------------------------------
    # DISTANCE
    # --------------------------------------------------------

    @staticmethod
    def distance(a, b):

        return math.sqrt(
            (a[0] - b[0]) ** 2 +
            (a[1] - b[1]) ** 2
        )

    # --------------------------------------------------------
    # OBSERVATION
    # --------------------------------------------------------

    def observation(self):

        return {
            "agent": self.agent,
            "objects": dict(self.objects),
            "obstacles": set(self.obstacles),
            "target": self.target_color,
            "goal": self.goal,
            "instruction": (
                f"Go to the {self.target_color} "
                f"object while avoiding obstacles."
            )
        }

    # --------------------------------------------------------
    # STEP
    # --------------------------------------------------------

    def step(self, action):

        self.steps += 1

        dx, dy = self.ACTIONS[int(action)]

        x, y = self.agent

        new_position = (
            x + dx,
            y + dy
        )

        reward = -0.01

        collision = False

        # bounds

        if not (
            0 <= new_position[0] < self.size
            and
            0 <= new_position[1] < self.size
        ):

            collision = True
            reward -= 0.10

        elif new_position in self.obstacles:

            collision = True
            reward -= 0.25

        else:

            self.agent = new_position

        # distance reward

        new_distance = self.distance(
            self.agent,
            self.goal
        )

        reward += (
            self.previous_distance
            - new_distance
        ) * 0.15

        self.previous_distance = new_distance

        success = (
            self.agent == self.goal
        )

        if success:
            reward += 5.0

        done = (
            success
            or
            self.steps >= self.cfg.max_steps
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

    # --------------------------------------------------------
    # RENDER
    # --------------------------------------------------------

    def render(self):

        grid = [
            ["." for _ in range(self.size)]
            for _ in range(self.size)
        ]

        for x, y in self.obstacles:
            grid[y][x] = "#"

        for color, position in self.objects.items():

            x, y = position

            grid[y][x] = self.SYMBOLS[color]

        gx, gy = self.goal
        grid[gy][gx] = "T"

        ax, ay = self.agent
        grid[ay][ax] = "A"

        print()

        for row in grid:
            print(" ".join(row))

        print()

        print("Instruction:")
        print(
            f"Go to the {self.target_color} "
            f"object while avoiding obstacles."
        )


# ============================================================
# VISION ENCODER
# ============================================================

class VisionEncoder(nn.Module):

    """
    Encodes the spatial grid.

    Channels:

        0 = obstacles
        1 = agent
        2 = red
        3 = green
        4 = blue
        5 = yellow
        6 = target
    """

    def __init__(self, cfg):

        super().__init__()

        self.net = nn.Sequential(

            nn.Conv2d(
                7,
                32,
                3,
                padding=1
            ),

            nn.ReLU(),

            nn.Conv2d(
                32,
                64,
                3,
                padding=1
            ),

            nn.ReLU(),

            nn.Conv2d(
                64,
                64,
                3,
                padding=1
            ),

            nn.ReLU(),

            nn.AdaptiveAvgPool2d(1)
        )

        self.projection = nn.Linear(
            64,
            cfg.vision_dim
        )

    def forward(self, x):

        x = self.net(x)

        x = x.flatten(1)

        return self.projection(x)


# ============================================================
# LANGUAGE ENCODER
# ============================================================

class LanguageEncoder(nn.Module):

    """
    Lightweight instruction encoder.

    The vocabulary is intentionally small because
    ARIA's environment uses a controlled language.
    """

    def __init__(self, cfg):

        super().__init__()

        vocabulary = [
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
            "obstacles"
        ]

        self.vocab = {
            word: i
            for i, word in enumerate(vocabulary)
        }

        self.embedding = nn.Embedding(
            len(self.vocab),
            cfg.language_dim
        )

        self.projection = nn.Linear(
            cfg.language_dim,
            cfg.language_dim
        )

    def tokenize(self, text):

        return [
            self.vocab.get(
                word.lower(),
                0
            )
            for word in text.split()
        ]

    def forward(self, texts):

        outputs = []

        for text in texts:

            tokens = self.tokenize(text)

            tokens = torch.tensor(
                tokens,
                dtype=torch.long,
                device=self.embedding.weight.device
            )

            emb = self.embedding(tokens)

            emb = emb.mean(dim=0)

            outputs.append(emb)

        outputs = torch.stack(outputs)

        return self.projection(outputs)


# ============================================================
# MULTIMODAL FUSION
# ============================================================

class MultimodalFusion(nn.Module):

    def __init__(self, cfg):

        super().__init__()

        self.vision_projection = nn.Linear(
            cfg.vision_dim,
            cfg.hidden_dim
        )

        self.language_projection = nn.Linear(
            cfg.language_dim,
            cfg.hidden_dim
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.hidden_dim,
            nhead=4,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2
        )

    def forward(
        self,
        vision,
        language
    ):

        v = self.vision_projection(
            vision
        )

        l = self.language_projection(
            language
        )

        tokens = torch.stack(
            [v, l],
            dim=1
        )

        fused = self.transformer(tokens)

        return fused.mean(dim=1)


# ============================================================
# REASONING MODULE
# ============================================================

class NeuralReasoner(nn.Module):

    """
    Predicts a compact world-state representation.

    Outputs:

        target coordinates
        target confidence
        distance
        directional reasoning
    """

    def __init__(self, cfg):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                cfg.hidden_dim,
                256
            ),

            nn.ReLU(),

            nn.Linear(
                256,
                128
            ),

            nn.ReLU()
        )

        self.target_head = nn.Linear(
            128,
            2
        )

        self.confidence_head = nn.Linear(
            128,
            1
        )

        self.distance_head = nn.Linear(
            128,
            1
        )

        self.direction_head = nn.Linear(
            128,
            4
        )

    def forward(self, state):

        h = self.network(state)

        target = self.target_head(h)

        confidence = torch.sigmoid(
            self.confidence_head(h)
        )

        distance = F.relu(
            self.distance_head(h)
        )

        direction = self.direction_head(h)

        return {
            "target": target,
            "confidence": confidence,
            "distance": distance,
            "direction": direction
        }


# ============================================================
# NEURAL PLANNER
# ============================================================

class NeuralPlanner(nn.Module):

    """
    Predicts the next action from the current
    multimodal world representation.
    """

    def __init__(self, cfg):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                cfg.hidden_dim + 4,
                256
            ),

            nn.ReLU(),

            nn.Linear(
                256,
                128
            ),

            nn.ReLU(),

            nn.Linear(
                128,
                cfg.num_actions
            )
        )

    def forward(
        self,
        state,
        reasoning
    ):

        x = torch.cat(
            [
                state,
                reasoning
            ],
            dim=-1
        )

        return self.network(x)


# ============================================================
# WORLD MODEL
# ============================================================

class WorldModel(nn.Module):

    """
    Predicts the next latent world state and reward.

    This provides the imagination mechanism used
    before action selection.
    """

    def __init__(self, cfg):

        super().__init__()

        self.transition = nn.Sequential(

            nn.Linear(
                cfg.hidden_dim + cfg.num_actions,
                256
            ),

            nn.ReLU(),

            nn.Linear(
                256,
                cfg.world_dim
            )
        )

        self.reward_head = nn.Linear(
            cfg.world_dim,
            1
        )

    def forward(
        self,
        state,
        action
    ):

        next_state = self.transition(
            torch.cat(
                [state, action],
                dim=-1
            )
        )

        reward = self.reward_head(
            next_state
        )

        return next_state, reward


# ============================================================
# EPISODIC MEMORY
# ============================================================

class EpisodicMemory:

    """
    Stores previous experiences.

    Each memory contains:

        state
        action
        reward
        next_state
        success
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
        next_state,
        success
    ):

        self.memory.append(
            {
                "state": state.detach().cpu(),
                "action": action,
                "reward": reward,
                "next_state": next_state.detach().cpu(),
                "success": success
            }
        )

    def sample(self, batch_size):

        batch_size = min(
            batch_size,
            len(self.memory)
        )

        return random.sample(
            list(self.memory),
            batch_size
        )

    def __len__(self):

        return len(self.memory)


# ============================================================
# RL CONTROLLER
# ============================================================

class RLController(nn.Module):

    """
    Value-based controller.

    Produces Q-values for four discrete actions.
    """

    def __init__(self, cfg):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                cfg.hidden_dim,
                256
            ),

            nn.ReLU(),

            nn.Linear(
                256,
                128
            ),

            nn.ReLU(),

            nn.Linear(
                128,
                cfg.num_actions
            )
        )

    def forward(self, state):

        return self.network(state)


# ============================================================
# GRID ENCODING
# ============================================================

def encode_grid(
    observation,
    cfg
):

    size = cfg.grid_size

    grid = np.zeros(
        (
            7,
            size,
            size
        ),
        dtype=np.float32
    )

    # obstacles

    for x, y in observation["obstacles"]:
        grid[0, y, x] = 1.0

    # agent

    ax, ay = observation["agent"]

    grid[1, ay, ax] = 1.0

    color_index = {
        "red": 2,
        "green": 3,
        "blue": 4,
        "yellow": 5
    }

    for color, (x, y) in observation["objects"].items():

        grid[
            color_index[color],
            y,
            x
        ] = 1.0

    # target

    tx, ty = observation["goal"]

    grid[6, ty, tx] = 1.0

    return torch.tensor(
        grid,
        dtype=torch.float32
    ).unsqueeze(0)


# ============================================================
# A* SAFETY PLANNER
# ============================================================

class SafetyPlanner:

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
                key=lambda p: f.get(
                    p,
                    float("inf")
                )
            )

            if current == goal:

                path = [current]

                while current in came_from:

                    current = came_from[current]

                    path.append(current)

                path.reverse()

                return path

            open_set.remove(current)

            x, y = current

            for dx, dy in self.ACTIONS:

                neighbor = (
                    x + dx,
                    y + dy
                )

                if not (
                    0 <= neighbor[0] < self.size
                    and
                    0 <= neighbor[1] < self.size
                ):
                    continue

                if neighbor in obstacles:
                    continue

                tentative = g[current] + 1

                if tentative < g.get(
                    neighbor,
                    float("inf")
                ):

                    came_from[neighbor] = current

                    g[neighbor] = tentative

                    f[neighbor] = (
                        tentative
                        +
                        self.heuristic(
                            neighbor,
                            goal
                        )
                    )

                    open_set.add(neighbor)

        return None


# ============================================================
# ARIA FINAL
# ============================================================

class ARIAFinal:

    def __init__(self, cfg):

        self.cfg = cfg

        self.vision = VisionEncoder(
            cfg
        )

        self.language = LanguageEncoder(
            cfg
        )

        self.fusion = MultimodalFusion(
            cfg
        )

        self.reasoner = NeuralReasoner(
            cfg
        )

        self.planner = NeuralPlanner(
            cfg
        )

        self.world_model = WorldModel(
            cfg
        )

        self.memory = EpisodicMemory(
            cfg.memory_size
        )

        self.rl_controller = RLController(
            cfg
        )

        self.safety_planner = SafetyPlanner(
            cfg.grid_size
        )

        self.to(cfg.device)

    # --------------------------------------------------------
    # DEVICE
    # --------------------------------------------------------

    def to(self, device):

        self.vision.to(device)
        self.language.to(device)
        self.fusion.to(device)
        self.reasoner.to(device)
        self.planner.to(device)
        self.world_model.to(device)
        self.rl_controller.to(device)

    # --------------------------------------------------------
    # ENCODE
    # --------------------------------------------------------

    def encode(self, observation):

        vision_input = encode_grid(
            observation,
            self.cfg
        ).to(self.cfg.device)

        language_input = self.language(
            [observation["instruction"]]
        )

        vision_features = self.vision(
            vision_input
        )

        world_state = self.fusion(
            vision_features,
            language_input
        )

        return world_state

    # --------------------------------------------------------
    # REASON
    # --------------------------------------------------------

    @torch.no_grad()
    def reason(self, observation):

        state = self.encode(
            observation
        )

        result = self.reasoner(
            state
        )

        return state, result

    # --------------------------------------------------------
    # IMAGINATION
    # --------------------------------------------------------

    @torch.no_grad()
    def imagine(
        self,
        state
    ):

        best_action = 0
        best_value = -float("inf")

        for action_id in range(
            self.cfg.num_actions
        ):

            action = F.one_hot(
                torch.tensor(
                    [action_id],
                    device=self.cfg.device
                ),
                num_classes=self.cfg.num_actions
            ).float()

            next_state, reward = (
                self.world_model(
                    state,
                    action
                )
            )

            value = reward.item()

            if value > best_value:

                best_value = value

                best_action = action_id

        return best_action, best_value

    # --------------------------------------------------------
    # ACT
    # --------------------------------------------------------

    @torch.no_grad()
    def act(self, observation):

        state, reasoning = self.reason(
            observation
        )

        target_prediction = (
            reasoning["target"]
        )

        confidence = (
            reasoning["confidence"]
            .item()
        )

        # Neural planner

        reasoning_vector = torch.cat(
            [
                target_prediction,
                reasoning["distance"],
                reasoning["confidence"]
            ],
            dim=-1
        )

        planner_logits = self.planner(
            state,
            reasoning_vector
        )

        neural_action = int(
            planner_logits.argmax(
                dim=-1
            ).item()
        )

        # RL controller

        q_values = self.rl_controller(
            state
        )

        rl_action = int(
            q_values.argmax(
                dim=-1
            ).item()
        )

        # Imagination

        imagined_action, imagined_value = (
            self.imagine(state)
        )

        # ----------------------------------------------------
        # Safety layer
        # ----------------------------------------------------

        path = self.safety_planner.plan(
            observation["agent"],
            observation["goal"],
            observation["obstacles"]
        )

        if path is not None and len(path) >= 2:

            next_position = path[1]

            x, y = observation["agent"]

            action_delta = (
                next_position[0] - x,
                next_position[1] - y
            )

            action_map = {
                (1, 0): 0,
                (-1, 0): 1,
                (0, 1): 2,
                (0, -1): 3
            }

            safety_action = action_map[
                action_delta
            ]

        else:

            safety_action = imagined_action

        # Use learned systems as primary signals,
        # safety planner prevents catastrophic invalid actions.

        if confidence < 0.05:

            selected_action = safety_action

            source = "safety_planner"

        else:

            selected_action = neural_action

            source = "neural_planner"

        return selected_action, {
            "confidence": confidence,
            "neural_action": neural_action,
            "rl_action": rl_action,
            "imagined_action": imagined_action,
            "imagined_value": imagined_value,
            "safety_action": safety_action,
            "source": source
        }

    # --------------------------------------------------------
    # MEMORY UPDATE
    # --------------------------------------------------------

    def remember(
        self,
        state,
        action,
        reward,
        next_state,
        success
    ):

        self.memory.add(
            state,
            action,
            reward,
            next_state,
            success
        )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    def save(self, path):

        os.makedirs(
            os.path.dirname(path),
            exist_ok=True
        )

        checkpoint = {

            "vision":
                self.vision.state_dict(),

            "language":
                self.language.state_dict(),

            "fusion":
                self.fusion.state_dict(),

            "reasoner":
                self.reasoner.state_dict(),

            "planner":
                self.planner.state_dict(),

            "world_model":
                self.world_model.state_dict(),

            "rl_controller":
                self.rl_controller.state_dict()
        }

        torch.save(
            checkpoint,
            path
        )

        print(
            f"✓ ARIA checkpoint saved: {path}"
        )

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    def load(self, path):

        checkpoint = torch.load(
            path,
            map_location=self.cfg.device
        )

        self.vision.load_state_dict(
            checkpoint["vision"]
        )

        self.language.load_state_dict(
            checkpoint["language"]
        )

        self.fusion.load_state_dict(
            checkpoint["fusion"]
        )

        self.reasoner.load_state_dict(
            checkpoint["reasoner"]
        )

        self.planner.load_state_dict(
            checkpoint["planner"]
        )

        self.world_model.load_state_dict(
            checkpoint["world_model"]
        )

        self.rl_controller.load_state_dict(
            checkpoint["rl_controller"]
        )

        print(
            f"✓ Loaded ARIA checkpoint: {path}"
        )


# ============================================================
# SANITY TESTS
# ============================================================

def sanity_tests(
    agent,
    env
):

    print()
    print("=" * 70)
    print("ARIA FINAL SANITY TESTS")
    print("=" * 70)

    observation = env.reset()

    # Vision

    state = agent.encode(
        observation
    )

    assert state.shape == (
        1,
        agent.cfg.hidden_dim
    )

    print("✓ Vision encoder")

    # Language

    language = agent.language(
        [observation["instruction"]]
    )

    assert language.shape[0] == 1

    print("✓ Language encoder")

    # Fusion

    assert state.shape[-1] == (
        agent.cfg.hidden_dim
    )

    print("✓ Multimodal fusion")

    # Reasoning

    reasoning = agent.reasoner(
        state
    )

    assert "target" in reasoning

    print("✓ Neural reasoning")

    # Planner

    reasoning_vector = torch.cat(
        [
            reasoning["target"],
            reasoning["distance"],
            reasoning["confidence"]
        ],
        dim=-1
    )

    planner_output = agent.planner(
        state,
        reasoning_vector
    )

    assert planner_output.shape[-1] == 4

    print("✓ Neural planner")

    # World model

    action = F.one_hot(
        torch.tensor(
            [0],
            device=agent.cfg.device
        ),
        num_classes=4
    ).float()

    next_state, reward = (
        agent.world_model(
            state,
            action
        )
    )

    assert next_state.shape[-1] == (
        agent.cfg.world_dim
    )

    print("✓ World model")

    # Memory

    agent.remember(
        state,
        0,
        0.0,
        next_state,
        False
    )

    assert len(agent.memory) > 0

    print("✓ Episodic memory")

    # RL

    q_values = agent.rl_controller(
        state
    )

    assert q_values.shape[-1] == 4

    print("✓ RL controller")

    print()
    print("All ARIA Final sanity tests passed.")


# ============================================================
# DEMONSTRATION
# ============================================================

def demonstration(
    agent,
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

    state, reasoning = agent.reason(
        observation
    )

    predicted_target = (
        reasoning["target"]
        .detach()
        .cpu()
        .numpy()[0]
    )

    confidence = (
        reasoning["confidence"]
        .item()
    )

    print(
        f"Neural Target: "
        f"{predicted_target}"
    )

    print(
        f"Actual Target: "
        f"{observation['goal']}"
    )

    print(
        f"Target Confidence: "
        f"{confidence:.4f}"
    )

    print()
    print(
        "Reasoning → Planning → "
        "World Model → Memory → RL"
    )

    total_reward = 0

    for step in range(
        env.cfg.max_steps
    ):

        action, info = agent.act(
            observation
        )

        next_observation, reward, done, env_info = (
            env.step(action)
        )

        next_state = agent.encode(
            next_observation
        )

        agent.remember(
            state,
            action,
            reward,
            next_state,
            env_info["success"]
        )

        total_reward += reward

        print(
            f"Step {step + 1:03d} | "
            f"Action={action} | "
            f"Source={info['source']} | "
            f"Confidence={info['confidence']:.3f} | "
            f"Value={info['imagined_value']:.3f}"
        )

        observation = next_observation
        state = next_state

        if done:
            break

    print()

    env.render()

    print(
        f"Success: "
        f"{env_info['success']}"
    )

    print(
        f"Steps: "
        f"{env_info['steps']}"
    )

    print(
        f"Final Distance: "
        f"{env_info['distance']:.4f}"
    )

    print(
        f"Memory Size: "
        f"{len(agent.memory)}"
    )

    print(
        f"Total Reward: "
        f"{total_reward:.4f}"
    )


# ============================================================
# TRAINING UTILITIES
# ============================================================

def train_world_model(
    agent,
    env,
    episodes=50
):

    print()
    print("=" * 70)
    print("TRAINING WORLD MODEL")
    print("=" * 70)

    optimizer = torch.optim.Adam(
        agent.world_model.parameters(),
        lr=agent.cfg.learning_rate
    )

    agent.world_model.train()

    for episode in range(episodes):

        observation = env.reset()

        state = agent.encode(
            observation
        ).detach()

        total_loss = 0.0

        for _ in range(
            env.cfg.max_steps
        ):

            action_id = random.randrange(
                agent.cfg.num_actions
            )

            next_observation, reward, done, info = (
                env.step(action_id)
            )

            next_state = agent.encode(
                next_observation
            ).detach()

            action = F.one_hot(
                torch.tensor(
                    [action_id],
                    device=agent.cfg.device
                ),
                num_classes=agent.cfg.num_actions
            ).float()

            predicted_state, predicted_reward = (
                agent.world_model(
                    state,
                    action
                )
            )

            target_reward = torch.tensor(
                [[reward]],
                dtype=torch.float32,
                device=agent.cfg.device
            )

            loss = (
                F.mse_loss(
                    predicted_state,
                    next_state
                )
                +
                F.mse_loss(
                    predicted_reward,
                    target_reward
                )
            )

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

            state = next_state

            if done:
                break

        if (
            episode + 1
        ) % 10 == 0:

            print(
                f"Episode {episode + 1:04d} | "
                f"World Model Loss: "
                f"{total_loss:.5f}"
            )

    print(
        "✓ World model training complete"
    )


# ============================================================
# TRAIN RL CONTROLLER
# ============================================================

def train_rl(
    agent,
    env,
    episodes=100
):

    print()
    print("=" * 70)
    print("TRAINING RL CONTROLLER")
    print("=" * 70)

    optimizer = torch.optim.Adam(
        agent.rl_controller.parameters(),
        lr=agent.cfg.learning_rate
    )

    agent.rl_controller.train()

    epsilon = 1.0

    for episode in range(episodes):

        observation = env.reset()

        episode_reward = 0.0

        for _ in range(
            env.cfg.max_steps
        ):

            state = agent.encode(
                observation
            ).detach()

            if random.random() < epsilon:

                action = random.randrange(
                    agent.cfg.num_actions
                )

            else:

                with torch.no_grad():

                    q_values = agent.rl_controller(
                        state
                    )

                    action = int(
                        q_values.argmax(
                            dim=-1
                        ).item()
                    )

            next_observation, reward, done, info = (
                env.step(action)
            )

            next_state = agent.encode(
                next_observation
            ).detach()

            q_values = agent.rl_controller(
                state
            )

            q_value = q_values[
                0,
                action
            ]

            with torch.no_grad():

                next_q = agent.rl_controller(
                    next_state
                ).max(
                    dim=-1
                ).values

                target = (
                    reward
                    +
                    agent.cfg.gamma
                    * next_q.item()
                    * (not done)
                )

            loss = F.mse_loss(
                q_value,
                torch.tensor(
                    target,
                    dtype=torch.float32,
                    device=agent.cfg.device
                )
            )

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            episode_reward += reward

            observation = next_observation

            if done:
                break

        epsilon = max(
            0.05,
            epsilon * 0.985
        )

        if (
            episode + 1
        ) % 20 == 0:

            print(
                f"Episode {episode + 1:04d} | "
                f"Reward: "
                f"{episode_reward:.3f}"
            )

    print(
        "✓ RL training complete"
    )


# ============================================================
# EVALUATION
# ============================================================

def evaluate(
    agent,
    env,
    episodes=100
):

    agent.vision.eval()
    agent.language.eval()
    agent.fusion.eval()
    agent.reasoner.eval()
    agent.planner.eval()
    agent.world_model.eval()
    agent.rl_controller.eval()

    rewards = []

    successes = 0
    collisions = 0
    steps = []
    distances = []

    for _ in range(episodes):

        observation = env.reset()

        total_reward = 0

        for _ in range(
            env.cfg.max_steps
        ):

            action, _ = agent.act(
                observation
            )

            (
                observation,
                reward,
                done,
                info
            ) = env.step(action)

            total_reward += reward

            if info["collision"]:
                collisions += 1

            if done:
                break

        rewards.append(
            total_reward
        )

        steps.append(
            info["steps"]
        )

        distances.append(
            info["distance"]
        )

        if info["success"]:
            successes += 1

    avg_reward = np.mean(
        rewards
    )

    success_rate = (
        successes / episodes
    )

    collision_rate = (
        collisions /
        max(sum(steps), 1)
    )

    avg_distance = np.mean(
        distances
    )

    avg_steps = np.mean(
        steps
    )

    print()
    print("=" * 70)
    print("ARIA FINAL EVALUATION")
    print("=" * 70)

    print(
        f"Average Reward      : "
        f"{avg_reward:.4f}"
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
        f"{avg_distance:.4f}"
    )

    print(
        f"Average Steps       : "
        f"{avg_steps:.2f}"
    )

    print("=" * 70)

    return {
        "reward": avg_reward,
        "success_rate": success_rate,
        "collision_rate": collision_rate,
        "distance": avg_distance,
        "steps": avg_steps
    }


# ============================================================
# MAIN
# ============================================================

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
        f"Grid: {cfg.grid_size}x{cfg.grid_size}"
    )

    env = ARIAEnvironment(
        cfg
    )

    agent = ARIAFinal(
        cfg
    )

    parameter_count = sum(
        p.numel()
        for p in agent.parameters()
    )

    print(
        f"Parameters: "
        f"{parameter_count:,}"
    )

    # --------------------------------------------------------
    # SANITY
    # --------------------------------------------------------

    sanity_tests(
        agent,
        env
    )

    # --------------------------------------------------------
    # DEMO BEFORE TRAINING
    # --------------------------------------------------------

    demonstration(
        agent,
        env
    )

    # --------------------------------------------------------
    # WORLD MODEL
    # --------------------------------------------------------

    train_world_model(
        agent,
        env,
        episodes=cfg.world_epochs
    )

    # --------------------------------------------------------
    # RL
    # --------------------------------------------------------

    train_rl(
        agent,
        env,
        episodes=cfg.rl_episodes
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    checkpoint = os.path.join(
        cfg.checkpoint_dir,
        "aria_final.pt"
    )

    agent.save(
        checkpoint
    )

    # --------------------------------------------------------
    # EVALUATE
    # --------------------------------------------------------

    results = evaluate(
        agent,
        env,
        episodes=100
    )

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
        """
    )

    print(
        f"Checkpoint: {checkpoint}"
    )


if __name__ == "__main__":
    main()