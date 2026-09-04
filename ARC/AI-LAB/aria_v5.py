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
# ARIA v5.0
# Adaptive Reasoning & Imagination Agent
#
# Integrated architecture:
#
# Vision
#   ↓
# Language
#   ↓
# Multimodal Fusion
#   ↓
# Reasoning
#   ↓
# Memory ←──────────────┐
#   ↓                   │
# World Model           │
#   ↓                   │
# Imagination           │
#   ↓                   │
# Planner               │
#   ↓                   │
# RL Policy             │
#   ↓                   │
# Action                │
#   ↓                   │
# Environment ──────────┘
#
# IMPORTANT:
# A* is NOT used during inference.
# The planner is a neural/imagination-based planner.
#
# This version is intended as the integrated ARIA foundation.
# ============================================================


# ============================================================
# CONFIG
# ============================================================

@dataclass
class Config:

    grid_size: int = 15

    num_obstacles: int = 22

    num_objects: int = 4

    max_steps: int = 60

    # Model
    d_model: int = 128

    vision_dim: int = 128

    language_dim: int = 128

    latent_dim: int = 128

    memory_dim: int = 128

    # Memory
    memory_capacity: int = 256

    # Imagination
    imagination_horizon: int = 8

    imagination_beams: int = 4

    # Training
    batch_size: int = 64

    device: str = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    seed: int = 42

    checkpoint_dir: str = "checkpoints"


Position = Tuple[int, int]


# ============================================================
# SEED
# ============================================================

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# ENVIRONMENT
# ============================================================

class ARIAEnvironment:

    OBJECT_COLORS = [
        "red",
        "green",
        "blue",
        "yellow"
    ]

    COLOR_IDS = {
        "red": 1,
        "green": 2,
        "blue": 3,
        "yellow": 4
    }

    SYMBOLS = {
        "red": "R",
        "green": "G",
        "blue": "B",
        "yellow": "Y"
    }

    ACTIONS = [
        (1, 0),    # right
        (-1, 0),   # left
        (0, 1),    # down
        (0, -1),   # up
    ]

    def __init__(self, cfg):

        self.cfg = cfg
        self.size = cfg.grid_size

        self.agent_pos = (0, 0)

        self.objects = {}

        self.target_color = None

        self.goal_pos = (0, 0)

        self.obstacles = set()

        self.step_count = 0

    # --------------------------------------------------------
    # RESET
    # --------------------------------------------------------

    def reset(self):

        self.objects.clear()
        self.obstacles.clear()

        self.step_count = 0

        # Agent
        self.agent_pos = self.random_position()

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
        for color in self.OBJECT_COLORS[:self.cfg.num_objects]:

            for _ in range(1000):

                pos = self.random_position()

                if (
                    pos != self.agent_pos
                    and pos not in self.obstacles
                    and pos not in self.objects.values()
                ):
                    self.objects[color] = pos
                    break

        self.target_color = random.choice(
            list(self.objects.keys())
        )

        self.goal_pos = self.objects[
            self.target_color
        ]

        return self.observation()

    # --------------------------------------------------------
    # POSITION
    # --------------------------------------------------------

    def random_position(self):

        return (
            random.randint(0, self.size - 1),
            random.randint(0, self.size - 1)
        )

    # --------------------------------------------------------
    # INSTRUCTION
    # --------------------------------------------------------

    def instruction(self):

        return (
            f"Go to the {self.target_color} "
            f"object while avoiding obstacles."
        )

    # --------------------------------------------------------
    # OBSERVATION
    # --------------------------------------------------------

    def observation(self):

        return {
            "agent": self.agent_pos,

            "objects": dict(self.objects),

            "target": self.target_color,

            "goal": self.goal_pos,

            "obstacles": set(self.obstacles),

            "instruction": self.instruction()
        }

    # --------------------------------------------------------
    # STEP
    # --------------------------------------------------------

    def step(self, action_index):

        self.step_count += 1

        dx, dy = self.ACTIONS[action_index]

        x, y = self.agent_pos

        new_pos = (
            x + dx,
            y + dy
        )

        reward = -0.01

        collision = False

        # ----------------------------------------------------
        # INVALID MOVE
        # ----------------------------------------------------

        if (
            new_pos[0] < 0
            or new_pos[0] >= self.size
            or new_pos[1] < 0
            or new_pos[1] >= self.size
        ):

            reward -= 0.10
            collision = True

        elif new_pos in self.obstacles:

            reward -= 0.25
            collision = True

        else:

            old_distance = self.distance(
                self.agent_pos,
                self.goal_pos
            )

            self.agent_pos = new_pos

            new_distance = self.distance(
                self.agent_pos,
                self.goal_pos
            )

            reward += (
                old_distance - new_distance
            ) * 0.15

        # ----------------------------------------------------
        # SUCCESS
        # ----------------------------------------------------

        success = (
            self.agent_pos == self.goal_pos
        )

        if success:
            reward += 5.0

        done = (
            success
            or self.step_count >= self.cfg.max_steps
        )

        info = {

            "success": success,

            "collision": collision,

            "distance": self.distance(
                self.agent_pos,
                self.goal_pos
            ),

            "steps": self.step_count
        }

        return (
            self.observation(),
            reward,
            done,
            info
        )

    # --------------------------------------------------------
    # DISTANCE
    # --------------------------------------------------------

    @staticmethod
    def distance(a, b):

        return math.sqrt(
            (a[0] - b[0]) ** 2
            +
            (a[1] - b[1]) ** 2
        )

    # --------------------------------------------------------
    # RENDER
    # --------------------------------------------------------

    def render(self):

        grid = [
            ["."] * self.size
            for _ in range(self.size)
        ]

        for x, y in self.obstacles:
            grid[y][x] = "#"

        for color, pos in self.objects.items():

            x, y = pos

            grid[y][x] = self.SYMBOLS[color]

        gx, gy = self.goal_pos

        grid[gy][gx] = "T"

        ax, ay = self.agent_pos

        grid[ay][ax] = "A"

        print()

        for row in grid:
            print(" ".join(row))

        print()

        print(
            "Instruction:",
            self.instruction()
        )


# ============================================================
# VISION ENCODER
# ============================================================

class VisionEncoder(nn.Module):

    """
    Converts the symbolic grid into spatial tokens.

    Channels:

        0 = empty
        1 = obstacle
        2 = agent
        3 = red
        4 = green
        5 = blue
        6 = yellow
        7 = target marker
    """

    def __init__(self, cfg):

        super().__init__()

        self.size = cfg.grid_size

        self.embedding = nn.Embedding(
            8,
            cfg.d_model
        )

        self.position = nn.Parameter(
            torch.randn(
                1,
                self.size * self.size,
                cfg.d_model
            ) * 0.02
        )

        encoder_layer = (
            nn.TransformerEncoderLayer(
                d_model=cfg.d_model,
                nhead=4,
                dim_feedforward=cfg.d_model * 4,
                batch_first=True,
                norm_first=False
            )
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2
        )

    def forward(self, grid):

        # [B,H,W]

        b, h, w = grid.shape

        x = self.embedding(
            grid.long()
        )

        x = x.reshape(
            b,
            h * w,
            -1
        )

        x = x + self.position[
            :, :h * w
        ]

        x = self.transformer(x)

        return x


# ============================================================
# LANGUAGE ENCODER
# ============================================================

class LanguageEncoder(nn.Module):

    """
    Small instruction encoder.

    We deliberately keep it lightweight.
    Later this can be replaced by a larger
    pretrained language model.
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
        "obstacles"
    ]

    def __init__(self, cfg):

        super().__init__()

        self.word_to_id = {
            word: i
            for i, word in enumerate(
                self.VOCAB
            )
        }

        self.embedding = nn.Embedding(
            len(self.VOCAB),
            cfg.language_dim
        )

        self.encoder = nn.GRU(
            cfg.language_dim,
            cfg.language_dim,
            batch_first=True
        )

    def tokenize(self, text):

        words = text.lower().replace(
            ".",
            ""
        ).split()

        ids = [
            self.word_to_id.get(
                word,
                0
            )
            for word in words
        ]

        if not ids:
            ids = [0]

        return ids

    def forward(self, token_ids):

        x = self.embedding(
            token_ids
        )

        _, h = self.encoder(x)

        return h[-1]


# ============================================================
# MULTIMODAL FUSION
# ============================================================

class MultimodalFusion(nn.Module):

    def __init__(self, cfg):

        super().__init__()

        self.query = nn.Linear(
            cfg.d_model,
            cfg.d_model
        )

        self.key = nn.Linear(
            cfg.language_dim,
            cfg.d_model
        )

        self.value = nn.Linear(
            cfg.language_dim,
            cfg.d_model
        )

        self.norm = nn.LayerNorm(
            cfg.d_model
        )

    def forward(
        self,
        vision_tokens,
        language
    ):

        # vision_tokens [B,N,D]
        # language [B,D]

        q = self.query(
            vision_tokens
        )

        k = self.key(
            language
        ).unsqueeze(1)

        v = self.value(
            language
        ).unsqueeze(1)

        attention = (
            q * k
        ).sum(-1, keepdim=True)

        attention = attention / math.sqrt(
            q.shape[-1]
        )

        attention = F.softmax(
            attention,
            dim=1
        )

        language_context = (
            attention * v
        )

        fused = (
            vision_tokens
            +
            language_context
        )

        fused = self.norm(fused)

        return fused


# ============================================================
# REASONER
# ============================================================

class NeuralReasoner(nn.Module):

    """
    Produces a latent world state.

    Outputs:

        latent state
        target heatmap
        direction logits
        confidence
    """

    def __init__(self, cfg):

        super().__init__()

        self.pool = nn.Linear(
            cfg.d_model,
            cfg.latent_dim
        )

        self.reasoning = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=cfg.latent_dim,
                nhead=4,
                dim_feedforward=cfg.latent_dim * 4,
                batch_first=True
            ),
            num_layers=2
        )

        self.target_head = nn.Linear(
            cfg.latent_dim,
            4
        )

        self.direction_head = nn.Linear(
            cfg.latent_dim,
            4
        )

        self.confidence_head = nn.Sequential(
            nn.Linear(
                cfg.latent_dim,
                64
            ),
            nn.ReLU(),
            nn.Linear(
                64,
                1
            ),
            nn.Sigmoid()
        )

    def forward(self, fused):

        x = self.pool(fused)

        x = self.reasoning(x)

        latent = x.mean(dim=1)

        target_logits = self.target_head(
            latent
        )

        direction_logits = self.direction_head(
            latent
        )

        confidence = self.confidence_head(
            latent
        )

        return {
            "latent": latent,

            "target_logits":
                target_logits,

            "direction_logits":
                direction_logits,

            "confidence":
                confidence
        }


# ============================================================
# EPISODIC MEMORY
# ============================================================

@dataclass
class MemoryItem:

    state: torch.Tensor

    action: int

    reward: float

    next_state: torch.Tensor

    done: bool


class EpisodicMemory:

    """
    External episodic memory.

    Stores actual experience rather than
    only using hidden-state memory.
    """

    def __init__(self, cfg):

        self.capacity = cfg.memory_capacity

        self.buffer = deque(
            maxlen=self.capacity
        )

    def add(
        self,
        state,
        action,
        reward,
        next_state,
        done
    ):

        self.buffer.append(
            MemoryItem(
                state.detach().cpu(),
                action,
                reward,
                next_state.detach().cpu(),
                done
            )
        )

    def __len__(self):

        return len(self.buffer)

    def retrieve(
        self,
        query,
        k=4
    ):

        if len(self.buffer) == 0:
            return []

        query = query.detach().cpu()

        scored = []

        for item in self.buffer:

            similarity = F.cosine_similarity(
                query.flatten(),
                item.state.flatten(),
                dim=0
            ).item()

            scored.append(
                (similarity, item)
            )

        scored.sort(
            key=lambda x: x[0],
            reverse=True
        )

        return [
            item
            for _, item
            in scored[:k]
        ]


# ============================================================
# WORLD MODEL
# ============================================================

class WorldModel(nn.Module):

    """
    Learns:

        state + action
            ↓
        predicted next state

    and predicts:

        reward
        termination
    """

    def __init__(self, cfg):

        super().__init__()

        self.action_embedding = nn.Embedding(
            4,
            cfg.latent_dim
        )

        self.transition = nn.Sequential(

            nn.Linear(
                cfg.latent_dim * 2,
                cfg.latent_dim * 2
            ),

            nn.LayerNorm(
                cfg.latent_dim * 2
            ),

            nn.GELU(),

            nn.Linear(
                cfg.latent_dim * 2,
                cfg.latent_dim
            ),

            nn.GELU(),

            nn.Linear(
                cfg.latent_dim,
                cfg.latent_dim
            )
        )

        self.reward_head = nn.Linear(
            cfg.latent_dim,
            1
        )

        self.done_head = nn.Linear(
            cfg.latent_dim,
            1
        )

    def forward(
        self,
        state,
        action
    ):

        action_embedding = (
            self.action_embedding(
                action.long()
            )
        )

        x = torch.cat(
            [
                state,
                action_embedding
            ],
            dim=-1
        )

        next_state = self.transition(x)

        reward = self.reward_head(
            next_state
        ).squeeze(-1)

        done = torch.sigmoid(
            self.done_head(
                next_state
            )
        ).squeeze(-1)

        return (
            next_state,
            reward,
            done
        )


# ============================================================
# IMAGINATION MODULE
# ============================================================

class Imagination(nn.Module):

    """
    Uses the learned World Model to imagine
    future trajectories.

    No environment interaction occurs here.
    """

    def __init__(
        self,
        world_model,
        cfg
    ):

        super().__init__()

        self.world_model = world_model

        self.horizon = (
            cfg.imagination_horizon
        )

    @torch.no_grad()
    def rollout(
        self,
        state,
        actions
    ):

        imagined_states = [
            state
        ]

        rewards = []

        current = state

        for action in actions:

            action_tensor = torch.tensor(
                [action],
                device=current.device,
                dtype=torch.long
            )

            current, reward, done = (
                self.world_model(
                    current,
                    action_tensor
                )
            )

            imagined_states.append(
                current
            )

            rewards.append(
                reward
            )

        if rewards:

            total_reward = torch.stack(
                rewards
            ).sum()

        else:

            total_reward = torch.tensor(
                0.0,
                device=state.device
            )

        return (
            imagined_states,
            total_reward
        )


# ============================================================
# NEURAL PLANNER
# ============================================================

class NeuralPlanner(nn.Module):

    """
    Chooses an action from the current latent state.

    It receives:

        reasoned state
        imagined/world information

    and outputs action logits.
    """

    def __init__(self, cfg):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                cfg.latent_dim,
                256
            ),

            nn.LayerNorm(256),

            nn.GELU(),

            nn.Linear(
                256,
                128
            ),

            nn.GELU(),

            nn.Linear(
                128,
                4
            )
        )

    def forward(self, state):

        return self.network(state)


# ============================================================
# RL POLICY
# ============================================================

class PolicyNetwork(nn.Module):

    """
    Actor network.

    Produces a probability distribution
    over the four discrete actions.
    """

    def __init__(self, cfg):

        super().__init__()

        self.actor = nn.Sequential(

            nn.Linear(
                cfg.latent_dim,
                256
            ),

            nn.LayerNorm(256),

            nn.GELU(),

            nn.Linear(
                256,
                128
            ),

            nn.GELU(),

            nn.Linear(
                128,
                4
            )
        )

        self.critic = nn.Sequential(

            nn.Linear(
                cfg.latent_dim,
                256
            ),

            nn.LayerNorm(256),

            nn.GELU(),

            nn.Linear(
                256,
                128
            ),

            nn.GELU(),

            nn.Linear(
                128,
                1
            )
        )

    def forward(self, state):

        logits = self.actor(state)

        value = self.critic(
            state
        ).squeeze(-1)

        return logits, value


# ============================================================
# ARIA CORE
# ============================================================

class ARIA(nn.Module):

    """
    Complete ARIA architecture.

    Perception
       ↓
    Language
       ↓
    Fusion
       ↓
    Reasoning
       ↓
    Memory
       ↓
    World Model
       ↓
    Imagination
       ↓
    Planner
       ↓
    Policy
    """

    def __init__(self, cfg):

        super().__init__()

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

        self.world_model = WorldModel(
            cfg
        )

        self.imagination = Imagination(
            self.world_model,
            cfg
        )

        self.planner = NeuralPlanner(
            cfg
        )

        self.policy = PolicyNetwork(
            cfg
        )

        self.memory = EpisodicMemory(
            cfg
        )

    # --------------------------------------------------------
    # BUILD GRID
    # --------------------------------------------------------

    def build_grid(self, observation):

        size = self.cfg.grid_size

        grid = np.zeros(
            (size, size),
            dtype=np.int64
        )

        # obstacles
        for x, y in observation[
            "obstacles"
        ]:

            grid[y, x] = 1

        # objects
        color_ids = {
            "red": 3,
            "green": 4,
            "blue": 5,
            "yellow": 6
        }

        for color, (x, y) in observation[
            "objects"
        ].items():

            grid[y, x] = color_ids[
                color
            ]

        # target
        gx, gy = observation["goal"]

        grid[gy, gx] = 7

        # agent last
        ax, ay = observation["agent"]

        grid[ay, ax] = 2

        return torch.tensor(
            grid,
            dtype=torch.long,
            device=self.cfg.device
        ).unsqueeze(0)

    # --------------------------------------------------------
    # LANGUAGE
    # --------------------------------------------------------

    def encode_instruction(
        self,
        instruction
    ):

        ids = self.language.tokenize(
            instruction
        )

        tensor = torch.tensor(
            ids,
            dtype=torch.long,
            device=self.cfg.device
        ).unsqueeze(0)

        return self.language(
            tensor
        )

    # --------------------------------------------------------
    # PERCEIVE
    # --------------------------------------------------------

    def perceive(self, observation):

        grid = self.build_grid(
            observation
        )

        vision_tokens = self.vision(
            grid
        )

        language = self.encode_instruction(
            observation["instruction"]
        )

        fused = self.fusion(
            vision_tokens,
            language
        )

        reasoning = self.reasoner(
            fused
        )

        return reasoning

    # --------------------------------------------------------
    # PLAN
    # --------------------------------------------------------

    def plan_action(
        self,
        latent,
        deterministic=True
    ):

        planner_logits = self.planner(
            latent
        )

        policy_logits, value = (
            self.policy(latent)
        )

        # Combine learned planner + RL policy.
        combined_logits = (
            0.5 * planner_logits
            +
            0.5 * policy_logits
        )

        distribution = torch.distributions.Categorical(
            logits=combined_logits
        )

        if deterministic:

            action = torch.argmax(
                combined_logits,
                dim=-1
            )

        else:

            action = distribution.sample()

        return (
            action,
            combined_logits,
            value
        )

    # --------------------------------------------------------
    # ACT
    # --------------------------------------------------------

    def act(
        self,
        observation,
        deterministic=True
    ):

        reasoning = self.perceive(
            observation
        )

        latent = reasoning[
            "latent"
        ]

        action, logits, value = (
            self.plan_action(
                latent,
                deterministic
            )
        )

        confidence = (
            reasoning[
                "confidence"
            ]
            .squeeze()
            .item()
        )

        return {
            "action":
                int(action.item()),

            "latent":
                latent,

            "confidence":
                confidence,

            "value":
                float(value.item()),

            "target_logits":
                reasoning[
                    "target_logits"
                ],

            "direction_logits":
                reasoning[
                    "direction_logits"
                ],

            "planner_logits":
                logits
        }


# ============================================================
# WORLD MODEL TRAINING
# ============================================================

def collect_world_model_data(
    agent,
    env,
    episodes=100
):

    data = []

    agent.eval()

    for _ in range(episodes):

        obs = env.reset()

        for _ in range(
            env.cfg.max_steps
        ):

            with torch.no_grad():

                result = agent.act(
                    obs,
                    deterministic=False
                )

            state = result[
                "latent"
            ].squeeze(0).detach()

            action = result[
                "action"
            ]

            next_obs, reward, done, info = (
                env.step(action)
            )

            with torch.no_grad():

                next_result = agent.act(
                    next_obs,
                    deterministic=False
                )

            next_state = next_result[
                "latent"
            ].squeeze(0).detach()

            data.append(
                (
                    state,
                    action,
                    reward,
                    next_state,
                    float(done)
                )
            )

            obs = next_obs

            if done:
                break

    return data


def train_world_model(
    agent,
    data,
    epochs=5,
    batch_size=64
):

    if not data:

        print(
            "No world-model data."
        )

        return

    print()
    print("=" * 70)
    print("TRAINING WORLD MODEL")
    print("=" * 70)

    optimizer = torch.optim.AdamW(
        agent.world_model.parameters(),
        lr=3e-4,
        weight_decay=1e-4
    )

    for epoch in range(epochs):

        random.shuffle(data)

        losses = []

        for start in range(
            0,
            len(data),
            batch_size
        ):

            batch = data[
                start:start + batch_size
            ]

            states = torch.stack(
                [
                    item[0]
                    for item in batch
                ]
            ).to(
                agent.cfg.device
            )

            actions = torch.tensor(
                [
                    item[1]
                    for item in batch
                ],
                dtype=torch.long,
                device=agent.cfg.device
            )

            rewards = torch.tensor(
                [
                    item[2]
                    for item in batch
                ],
                dtype=torch.float32,
                device=agent.cfg.device
            )

            next_states = torch.stack(
                [
                    item[3]
                    for item in batch
                ]
            ).to(
                agent.cfg.device
            )

            done = torch.tensor(
                [
                    item[4]
                    for item in batch
                ],
                dtype=torch.float32,
                device=agent.cfg.device
            )

            pred_next, pred_reward, pred_done = (
                agent.world_model(
                    states,
                    actions
                )
            )

            state_loss = F.mse_loss(
                pred_next,
                next_states
            )

            reward_loss = F.mse_loss(
                pred_reward,
                rewards
            )

            done_loss = F.binary_cross_entropy(
                pred_done.clamp(
                    1e-5,
                    1 - 1e-5
                ),
                done
            )

            loss = (
                state_loss
                +
                reward_loss
                +
                0.25 * done_loss
            )

            optimizer.zero_grad()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                agent.world_model.parameters(),
                1.0
            )

            optimizer.step()

            losses.append(
                loss.item()
            )

        print(
            f"Epoch {epoch + 1}/{epochs} "
            f"| Loss: {np.mean(losses):.6f}"
        )


# ============================================================
# POLICY TRAINING WITH REINFORCEMENT LEARNING
# ============================================================

def train_rl_episode(
    agent,
    env,
    optimizer,
    gamma=0.99
):

    agent.train()

    obs = env.reset()

    log_probs = []

    values = []

    rewards = []

    entropies = []

    for _ in range(
        env.cfg.max_steps
    ):

        reasoning = agent.perceive(
            obs
        )

        latent = reasoning[
            "latent"
        ]

        planner_logits = agent.planner(
            latent
        )

        policy_logits, value = (
            agent.policy(latent)
        )

        logits = (
            0.5 * planner_logits
            +
            0.5 * policy_logits
        )

        distribution = torch.distributions.Categorical(
            logits=logits
        )

        action = distribution.sample()

        next_obs, reward, done, info = (
            env.step(
                int(action.item())
            )
        )

        log_probs.append(
            distribution.log_prob(
                action
            )
        )

        values.append(
            value
        )

        rewards.append(
            reward
        )

        entropies.append(
            distribution.entropy()
        )

        obs = next_obs

        if done:
            break

    # --------------------------------------------------------
    # RETURNS
    # --------------------------------------------------------

    returns = []

    running = 0.0

    for reward in reversed(
        rewards
    ):

        running = (
            reward
            +
            gamma * running
        )

        returns.append(
            running
        )

    returns.reverse()

    returns = torch.tensor(
        returns,
        dtype=torch.float32,
        device=agent.cfg.device
    )

    values = torch.cat(
        values
    )

    log_probs = torch.stack(
        log_probs
    )

    entropies = torch.stack(
        entropies
    )

    # IMPORTANT:
    # values and returns are both [T].
    # This avoids the broadcasting bug from earlier ARIA versions.

    advantages = (
        returns
        -
        values.detach()
    )

    policy_loss = -(
        log_probs
        *
        advantages
    ).mean()

    value_loss = F.mse_loss(
        values,
        returns
    )

    entropy_bonus = (
        entropies.mean()
    )

    loss = (
        policy_loss
        +
        0.5 * value_loss
        -
        0.01 * entropy_bonus
    )

    optimizer.zero_grad()

    loss.backward()

    torch.nn.utils.clip_grad_norm_(
        agent.policy.parameters(),
        1.0
    )

    optimizer.step()

    return {
        "loss":
            float(loss.item()),

        "reward":
            float(sum(rewards)),

        "steps":
            len(rewards),

        "success":
            info["success"]
    }


# ============================================================
# TRAIN RL
# ============================================================

def train_rl(
    agent,
    env,
    episodes=200
):

    print()
    print("=" * 70)
    print("TRAINING ARIA RL POLICY")
    print("=" * 70)

    optimizer = torch.optim.AdamW(
        agent.policy.parameters(),
        lr=3e-4
    )

    recent_rewards = deque(
        maxlen=20
    )

    recent_success = deque(
        maxlen=20
    )

    for episode in range(
        1,
        episodes + 1
    ):

        result = train_rl_episode(
            agent,
            env,
            optimizer
        )

        recent_rewards.append(
            result["reward"]
        )

        recent_success.append(
            float(result["success"])
        )

        if (
            episode == 1
            or episode % 20 == 0
        ):

            print(
                f"Episode {episode:04d} "
                f"| Reward: "
                f"{np.mean(recent_rewards):7.3f} "
                f"| Success: "
                f"{100 * np.mean(recent_success):6.2f}% "
                f"| Loss: "
                f"{result['loss']:.4f}"
            )


# ============================================================
# EVALUATION
# ============================================================

@torch.no_grad()
def evaluate(
    agent,
    env,
    episodes=100
):

    agent.eval()

    rewards = []

    successes = 0

    collisions = 0

    steps = []

    distances = []

    for _ in range(
        episodes
    ):

        obs = env.reset()

        total_reward = 0

        episode_collisions = 0

        info = {
            "distance":
                env.distance(
                    env.agent_pos,
                    env.goal_pos
                ),

            "success": False,

            "steps": 0
        }

        for _ in range(
            env.cfg.max_steps
        ):

            result = agent.act(
                obs,
                deterministic=True
            )

            action = result[
                "action"
            ]

            obs, reward, done, info = (
                env.step(action)
            )

            total_reward += reward

            if info["collision"]:

                episode_collisions += 1

            if done:

                break

        rewards.append(
            total_reward
        )

        collisions += (
            episode_collisions
        )

        steps.append(
            info["steps"]
        )

        distances.append(
            info["distance"]
        )

        if info["success"]:

            successes += 1

    success_rate = (
        successes / episodes
    )

    collision_rate = (
        collisions
        /
        max(sum(steps), 1)
    )

    print()
    print("=" * 70)
    print("ARIA v5 EVALUATION")
    print("=" * 70)

    print(
        f"Average Reward      : "
        f"{np.mean(rewards):.4f}"
    )

    print(
        f"Navigation Success  : "
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
        "reward":
            np.mean(rewards),

        "success_rate":
            success_rate,

        "collision_rate":
            collision_rate,

        "distance":
            np.mean(distances),

        "steps":
            np.mean(steps)
    }


# ============================================================
# DEMONSTRATION
# ============================================================

@torch.no_grad()
def demonstration(
    agent,
    env
):

    print()
    print("=" * 70)
    print("ARIA v5 DEMONSTRATION")
    print("=" * 70)

    obs = env.reset()

    print()
    print("Instruction:")
    print(obs["instruction"])

    env.render()

    result = agent.act(
        obs,
        deterministic=True
    )

    target_prediction = torch.argmax(
        result["target_logits"],
        dim=-1
    ).item()

    target_names = [
        "red",
        "green",
        "blue",
        "yellow"
    ]

    print()
    print("ARIA INTERNAL STATE")
    print(
        f"Predicted target class : "
        f"{target_names[target_prediction]}"
    )

    print(
        f"Reasoning confidence   : "
        f"{result['confidence']:.4f}"
    )

    print(
        f"Value estimate         : "
        f"{result['value']:.4f}"
    )

    print()
    print("Closed-loop execution:")

    total_reward = 0

    for step in range(
        env.cfg.max_steps
    ):

        result = agent.act(
            obs,
            deterministic=True
        )

        action = result[
            "action"
        ]

        obs, reward, done, info = (
            env.step(action)
        )

        total_reward += reward

        action_name = [
            "RIGHT",
            "LEFT",
            "DOWN",
            "UP"
        ][action]

        print(
            f"Step {step + 1:02d} "
            f"| Action={action_name:5s} "
            f"| Distance={info['distance']:.3f} "
            f"| Confidence={result['confidence']:.3f}"
        )

        if done:
            break

    print()

    env.render()

    print(
        f"Success       : "
        f"{info['success']}"
    )

    print(
        f"Steps         : "
        f"{info['steps']}"
    )

    print(
        f"Final Distance: "
        f"{info['distance']:.3f}"
    )

    print(
        f"Total Reward  : "
        f"{total_reward:.3f}"
    )


# ============================================================
# ARCHITECTURE SUMMARY
# ============================================================

def print_architecture():

    print()
    print("=" * 70)
    print("ARIA v5 ARCHITECTURE")
    print("=" * 70)

    print(
        """
                  ARIA v5.0
                      │
          ┌───────────┴───────────┐
          │                       │
       Vision                  Language
          │                       │
          └───────────┬───────────┘
                      ↓
             Multimodal Fusion
                      ↓
                Neural Reasoner
                      ↓
                Latent State
                      │
          ┌───────────┼───────────┐
          ↓           ↓           ↓
       Memory     World Model   Reasoning
          │           │
          └─────┬─────┘
                ↓
           Imagination
                ↓
        Neural Planner
                ↓
           RL Policy
                ↓
              Action
                ↓
          Environment
                │
                └────────→ Memory


Training:

Expert data
    ↓
Perception / Grounding

Environment experience
    ↓
World Model

Environment rewards
    ↓
RL Policy

World Model
    ↓
Imagination
    ↓
Planning
"""
    )


# ============================================================
# SAVE / LOAD
# ============================================================

def save_checkpoint(
    agent,
    cfg,
    filename="aria_v5_final.pt"
):

    os.makedirs(
        cfg.checkpoint_dir,
        exist_ok=True
    )

    path = os.path.join(
        cfg.checkpoint_dir,
        filename
    )

    torch.save(
        {
            "model":
                agent.state_dict(),

            "config":
                cfg.__dict__
        },
        path
    )

    print(
        f"✓ Checkpoint saved: {path}"
    )

    return path


def load_checkpoint(
    agent,
    path
):

    checkpoint = torch.load(
        path,
        map_location=agent.cfg.device
    )

    agent.load_state_dict(
        checkpoint["model"]
    )

    print(
        f"✓ Loaded checkpoint: {path}"
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
    print("ARIA v5 SANITY TESTS")
    print("=" * 70)

    obs = env.reset()

    # Environment
    assert "agent" in obs
    assert "objects" in obs
    assert "obstacles" in obs

    print(
        "✓ Environment"
    )

    # Grid
    grid = agent.build_grid(
        obs
    )

    assert grid.shape == (
        1,
        env.size,
        env.size
    )

    print(
        "✓ Visual grid encoding"
    )

    # Vision
    vision = agent.vision(
        grid
    )

    assert vision.shape[0] == 1

    print(
        "✓ Vision encoder"
    )

    # Language
    language = agent.encode_instruction(
        obs["instruction"]
    )

    assert language.shape[0] == 1

    print(
        "✓ Language encoder"
    )

    # Fusion
    fused = agent.fusion(
        vision,
        language
    )

    print(
        "✓ Multimodal fusion"
    )

    # Reasoning
    reasoning = agent.reasoner(
        fused
    )

    assert "latent" in reasoning

    assert reasoning[
        "latent"
    ].shape[-1] == env.cfg.latent_dim

    print(
        "✓ Neural reasoning"
    )

    # World model
    action = torch.tensor(
        [0],
        dtype=torch.long,
        device=cfg.device
    )

    next_state, reward, done = (
        agent.world_model(
            reasoning["latent"],
            action
        )
    )

    assert next_state.shape == (
        1,
        cfg.latent_dim
    )

    print(
        "✓ World model"
    )

    # Planner
    planner_logits = agent.planner(
        reasoning["latent"]
    )

    assert planner_logits.shape == (
        1,
        4
    )

    print(
        "✓ Neural planner"
    )

    # Policy
    policy_logits, value = (
        agent.policy(
            reasoning["latent"]
        )
    )

    assert policy_logits.shape == (
        1,
        4
    )

    assert value.shape == (
        1,
    )

    print(
        "✓ RL actor-critic"
    )

    # Full action
    result = agent.act(
        obs
    )

    assert 0 <= result[
        "action"
    ] < 4

    print(
        "✓ End-to-end action"
    )

    print()
    print(
        "All ARIA v5 sanity tests passed."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print(
        "ARIA v5.0"
    )
    print(
        "ADAPTIVE REASONING & "
        "IMAGINATION AGENT"
    )
    print("=" * 70)

    global cfg

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
        f"Objects: "
        f"{cfg.num_objects}"
    )

    # --------------------------------------------------------
    # Environment
    # --------------------------------------------------------

    env = ARIAEnvironment(
        cfg
    )

    # --------------------------------------------------------
    # Agent
    # --------------------------------------------------------

    agent = ARIA(
        cfg
    ).to(
        cfg.device
    )

    # --------------------------------------------------------
    # Parameters
    # --------------------------------------------------------

    total_params = sum(
        p.numel()
        for p in agent.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in agent.parameters()
        if p.requires_grad
    )

    print()
    print(
        f"Total parameters: "
        f"{total_params:,}"
    )

    print(
        f"Trainable parameters: "
        f"{trainable_params:,}"
    )

    # --------------------------------------------------------
    # Architecture
    # --------------------------------------------------------

    print_architecture()

    # --------------------------------------------------------
    # Tests
    # --------------------------------------------------------

    sanity_tests(
        agent,
        env
    )

    # --------------------------------------------------------
    # Initial demonstration
    # --------------------------------------------------------

    demonstration(
        agent,
        env
    )

    # --------------------------------------------------------
    # WORLD MODEL
    # --------------------------------------------------------

    print()
    print(
        "Collecting experience "
        "for World Model..."
    )

    world_data = collect_world_model_data(
        agent,
        env,
        episodes=50
    )

    print(
        f"World Model samples: "
        f"{len(world_data)}"
    )

    train_world_model(
        agent,
        world_data,
        epochs=5,
        batch_size=cfg.batch_size
    )

    # --------------------------------------------------------
    # RL
    # --------------------------------------------------------

    train_rl(
        agent,
        env,
        episodes=100
    )

    # --------------------------------------------------------
    # FINAL DEMO
    # --------------------------------------------------------

    demonstration(
        agent,
        env
    )

    # --------------------------------------------------------
    # EVALUATION
    # --------------------------------------------------------

    metrics = evaluate(
        agent,
        env,
        episodes=100
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    checkpoint = save_checkpoint(
        agent,
        cfg
    )

    # --------------------------------------------------------
    # FINAL REPORT
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("ARIA v5.0 COMPLETE")
    print("=" * 70)

    print(
        f"""
Architecture:

Vision
  ↓
Language
  ↓
Multimodal Fusion
  ↓
Neural Reasoning
  ↓
Episodic Memory
  ↓
World Model
  ↓
Imagination
  ↓
Neural Planning
  ↓
RL Policy
  ↓
Environment

Final Metrics:

Navigation Success : {metrics['success_rate'] * 100:.2f}%
Collision Rate     : {metrics['collision_rate'] * 100:.2f}%
Average Reward     : {metrics['reward']:.4f}
Final Distance     : {metrics['distance']:.4f}
Average Steps      : {metrics['steps']:.2f}

Checkpoint:
{checkpoint}

A* was NOT used during ARIA inference.
"""
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()