import math
import random
from dataclasses import dataclass
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# ARIA v5.0
# Integrated Cognitive Agent
#
# Vision
#   ↓
# Language
#   ↓
# Multimodal Fusion
#   ↓
# Neural Reasoning
#   ↓
# Episodic Memory
#   ↓
# World Model
#   ↓
# Imagination Planning
#   ↓
# RL Controller
#   ↓
# Action
#
# This is a research prototype.
# ============================================================


# ============================================================
# CONFIG
# ============================================================

@dataclass
class Config:

    grid_size: int = 15
    max_steps: int = 80

    num_obstacles: int = 22
    num_objects: int = 4

    vision_dim: int = 128
    language_dim: int = 128
    hidden_dim: int = 256

    memory_dim: int = 256

    world_hidden: int = 256

    num_actions: int = 5

    imagination_horizon: int = 8
    imagination_candidates: int = 12

    gamma: float = 0.99

    seed: int = 42

    device: str = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


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
# ACTION SPACE
# ============================================================

ACTIONS = [
    (0, 0),      # stay
    (1, 0),      # right
    (-1, 0),     # left
    (0, 1),      # down
    (0, -1),     # up
]


# ============================================================
# ENVIRONMENT
# ============================================================

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

    def __init__(self, cfg):

        self.cfg = cfg
        self.size = cfg.grid_size

        self.agent = None
        self.objects = {}
        self.obstacles = set()

        self.target_color = None
        self.goal = None

        self.steps = 0
        self.previous_distance = 0.0

    # --------------------------------------------------------
    # RANDOM POSITION
    # --------------------------------------------------------

    def random_position(self):

        return (
            random.randint(0, self.size - 1),
            random.randint(0, self.size - 1),
        )

    # --------------------------------------------------------
    # RESET
    # --------------------------------------------------------

    def reset(self):

        self.steps = 0

        self.obstacles = set()
        self.objects = {}

        # Agent

        self.agent = self.random_position()

        # Obstacles

        while len(self.obstacles) < self.cfg.num_obstacles:

            p = self.random_position()

            if p != self.agent:

                self.obstacles.add(p)

        # Objects

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

        # Target

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

            "agent": self.agent,

            "objects": dict(
                self.objects
            ),

            "obstacles": set(
                self.obstacles
            ),

            "goal": self.goal,

            "target_color":
                self.target_color,

            "instruction":
                self.instruction(),
        }

    # --------------------------------------------------------
    # STEP
    # --------------------------------------------------------

    def step(self, action):

        self.steps += 1

        dx, dy = ACTIONS[action]

        new_position = (
            self.agent[0] + dx,
            self.agent[1] + dy,
        )

        collision = False

        if (
            new_position[0] < 0
            or new_position[0] >= self.size
            or new_position[1] < 0
            or new_position[1] >= self.size
        ):

            collision = True

        elif new_position in self.obstacles:

            collision = True

        if not collision:

            self.agent = new_position

        new_distance = self.distance(
            self.agent,
            self.goal
        )

        reward = -0.01

        reward += (
            self.previous_distance
            - new_distance
        ) * 0.15

        if collision:

            reward -= 0.25

        success = (
            self.agent == self.goal
        )

        if success:

            reward += 5.0

        self.previous_distance = new_distance

        done = (
            success
            or self.steps >= self.cfg.max_steps
        )

        info = {

            "success": success,

            "collision": collision,

            "distance":
                new_distance,

            "steps":
                self.steps,
        }

        return (
            self.observation(),
            reward,
            done,
            info,
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


# ============================================================
# VISION ENCODER
# ============================================================

class VisionEncoder(nn.Module):

    """
    Converts the symbolic grid into visual features.

    Later this can be replaced by a real image encoder.
    """

    def __init__(self, cfg):

        super().__init__()

        self.grid_size = cfg.grid_size

        self.conv = nn.Sequential(

            nn.Conv2d(
                7,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.Conv2d(
                64,
                64,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

        )

        self.fc = nn.Linear(
            64 * cfg.grid_size * cfg.grid_size,
            cfg.vision_dim
        )

    def encode_grid(self, obs):

        grid = torch.zeros(
            7,
            self.grid_size,
            self.grid_size
        )

        # Channel 0 = agent

        x, y = obs["agent"]

        grid[0, y, x] = 1.0

        # Channels 1-4 = objects

        color_to_channel = {
            "red": 1,
            "green": 2,
            "blue": 3,
            "yellow": 4,
        }

        for color, pos in obs["objects"].items():

            x, y = pos

            grid[
                color_to_channel[color],
                y,
                x
            ] = 1.0

        # Channel 5 = obstacles

        for x, y in obs["obstacles"]:

            grid[5, y, x] = 1.0

        # Channel 6 = target

        x, y = obs["goal"]

        grid[6, y, x] = 1.0

        return grid

    def forward(self, obs):

        grid = self.encode_grid(obs)

        grid = grid.unsqueeze(0)

        features = self.conv(grid)

        features = features.flatten(1)

        return self.fc(features)


# ============================================================
# LANGUAGE ENCODER
# ============================================================

class LanguageEncoder(nn.Module):

    """
    Tiny instruction encoder.

    Current language space is intentionally simple.
    """

    def __init__(self, cfg):

        super().__init__()

        self.embedding = nn.Embedding(
            32,
            cfg.language_dim
        )

        self.gru = nn.GRU(
            cfg.language_dim,
            cfg.language_dim,
            batch_first=True
        )

        self.vocab = {

            "go": 1,
            "to": 2,
            "the": 3,
            "red": 4,
            "green": 5,
            "blue": 6,
            "yellow": 7,
            "object": 8,
            "while": 9,
            "avoiding": 10,
            "obstacles": 11,
        }

    def tokenize(self, text):

        tokens = []

        for word in text.lower().split():

            word = word.strip(
                ".,!?;"
            )

            tokens.append(
                self.vocab.get(
                    word,
                    0
                )
            )

        return tokens

    def forward(self, text):

        tokens = self.tokenize(text)

        x = torch.tensor(
            tokens,
            dtype=torch.long
        ).unsqueeze(0)

        x = self.embedding(x)

        _, hidden = self.gru(x)

        return hidden[-1]


# ============================================================
# MULTIMODAL FUSION
# ============================================================

class MultimodalFusion(nn.Module):

    def __init__(self, cfg):

        super().__init__()

        self.fc = nn.Sequential(

            nn.Linear(
                cfg.vision_dim
                +
                cfg.language_dim,
                cfg.hidden_dim
            ),

            nn.LayerNorm(
                cfg.hidden_dim
            ),

            nn.GELU(),

            nn.Linear(
                cfg.hidden_dim,
                cfg.hidden_dim
            ),

            nn.GELU(),
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

        return self.fc(x)


# ============================================================
# NEURAL REASONER
# ============================================================

class NeuralReasoner(nn.Module):

    """
    Produces a structured latent world state.

    Outputs:

        target representation
        spatial representation
        obstacle representation
        confidence
    """

    def __init__(self, cfg):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                cfg.hidden_dim,
                cfg.hidden_dim
            ),

            nn.LayerNorm(
                cfg.hidden_dim
            ),

            nn.GELU(),

            nn.Linear(
                cfg.hidden_dim,
                cfg.hidden_dim
            ),

            nn.GELU(),
        )

        self.target_head = nn.Linear(
            cfg.hidden_dim,
            cfg.grid_size * cfg.grid_size
        )

        self.spatial_head = nn.Linear(
            cfg.hidden_dim,
            4
        )

        self.confidence_head = nn.Linear(
            cfg.hidden_dim,
            1
        )

    def forward(self, x):

        h = self.network(x)

        target = self.target_head(h)

        spatial = self.spatial_head(h)

        confidence = torch.sigmoid(
            self.confidence_head(h)
        )

        return {

            "latent": h,

            "target": target,

            "spatial": spatial,

            "confidence":
                confidence,
        }


# ============================================================
# EPISODIC MEMORY
# ============================================================

class EpisodicMemory:

    """
    Stores previous experiences.

    Memory item:

        state
        action
        reward
        next_state
        success
    """

    def __init__(
        self,
        capacity=5000
    ):

        self.memory = deque(
            maxlen=capacity
        )

    def store(
        self,
        state,
        action,
        reward,
        next_state,
        success
    ):

        self.memory.append({

            "state": state,

            "action": action,

            "reward": reward,

            "next_state":
                next_state,

            "success":
                success,
        })

    def retrieve(
        self,
        state,
        k=5
    ):

        if len(self.memory) == 0:

            return []

        # Simple similarity using agent position.

        current = state["agent"]

        scored = []

        for item in self.memory:

            old = item["state"]["agent"]

            distance = (
                abs(
                    current[0]
                    -
                    old[0]
                )
                +
                abs(
                    current[1]
                    -
                    old[1]
                )
            )

            scored.append(
                (
                    distance,
                    item
                )
            )

        scored.sort(
            key=lambda x: x[0]
        )

        return [
            item
            for _, item
            in scored[:k]
        ]

    def __len__(self):

        return len(self.memory)


# ============================================================
# WORLD MODEL
# ============================================================

class WorldModel(nn.Module):

    """
    Learns:

        current latent
        +
        action
        ↓
        predicted next latent
        +
        reward
        +
        termination
    """

    def __init__(self, cfg):

        super().__init__()

        self.action_embedding = nn.Embedding(
            cfg.num_actions,
            64
        )

        self.transition = nn.Sequential(

            nn.Linear(
                cfg.hidden_dim + 64,
                cfg.world_hidden
            ),

            nn.LayerNorm(
                cfg.world_hidden
            ),

            nn.GELU(),

            nn.Linear(
                cfg.world_hidden,
                cfg.hidden_dim
            ),

            nn.GELU(),
        )

        self.reward_head = nn.Linear(
            cfg.hidden_dim,
            1
        )

        self.done_head = nn.Linear(
            cfg.hidden_dim,
            1
        )

    def forward(
        self,
        state_latent,
        action
    ):

        action_embedding = (
            self.action_embedding(
                action
            )
        )

        x = torch.cat(
            [
                state_latent,
                action_embedding
            ],
            dim=-1
        )

        next_latent = self.transition(x)

        reward = self.reward_head(
            next_latent
        )

        done = torch.sigmoid(
            self.done_head(
                next_latent
            )
        )

        return (
            next_latent,
            reward,
            done
        )


# ============================================================
# IMAGINATION PLANNER
# ============================================================

class ImaginationPlanner:

    """
    Model-based planner.

    It does NOT use A*.

    It rolls actions through the learned
    world model and scores imagined futures.
    """

    def __init__(self, cfg):

        self.cfg = cfg

    @torch.no_grad()
    def plan(
        self,
        latent,
        world_model
    ):

        best_action = 0
        best_score = -float("inf")

        for _ in range(
            self.cfg.imagination_candidates
        ):

            current = latent.clone()

            total_reward = 0.0

            discount = 1.0

            first_action = None

            for step in range(
                self.cfg.imagination_horizon
            ):

                action = torch.randint(
                    0,
                    self.cfg.num_actions,
                    (1,)
                )

                if first_action is None:

                    first_action = (
                        action.item()
                    )

                (
                    current,
                    reward,
                    done
                ) = world_model(
                    current,
                    action
                )

                total_reward += (
                    discount
                    *
                    reward.item()
                )

                discount *= self.cfg.gamma

                if done.item() > 0.8:

                    break

            if total_reward > best_score:

                best_score = total_reward

                best_action = (
                    first_action
                )

        return (
            best_action,
            best_score
        )


# ============================================================
# RL CONTROLLER
# ============================================================

class RLController(nn.Module):

    """
    Policy + value network.

    This is the bridge between planning
    and reinforcement learning.
    """

    def __init__(self, cfg):

        super().__init__()

        self.body = nn.Sequential(

            nn.Linear(
                cfg.hidden_dim,
                cfg.hidden_dim
            ),

            nn.GELU(),

            nn.Linear(
                cfg.hidden_dim,
                128
            ),

            nn.GELU(),
        )

        self.policy = nn.Linear(
            128,
            cfg.num_actions
        )

        self.value = nn.Linear(
            128,
            1
        )

    def forward(self, latent):

        h = self.body(latent)

        logits = self.policy(h)

        value = self.value(h)

        return logits, value

    def act(
        self,
        latent,
        deterministic=True
    ):

        logits, value = self(
            latent
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

        return (
            action.item(),
            value.item()
        )


# ============================================================
# ARIA v5
# ============================================================

class ARIAv5:

    """
    Integrated cognitive architecture.
    """

    def __init__(self, cfg):

        self.cfg = cfg

        self.device = torch.device(
            cfg.device
        )

        # ----------------------------------------------------
        # Modules
        # ----------------------------------------------------

        self.vision = VisionEncoder(
            cfg
        ).to(self.device)

        self.language = LanguageEncoder(
            cfg
        ).to(self.device)

        self.fusion = MultimodalFusion(
            cfg
        ).to(self.device)

        self.reasoner = NeuralReasoner(
            cfg
        ).to(self.device)

        self.world_model = WorldModel(
            cfg
        ).to(self.device)

        self.controller = RLController(
            cfg
        ).to(self.device)

        self.memory = EpisodicMemory()

        self.planner = ImaginationPlanner(
            cfg
        )

        self.last_latent = None

    # --------------------------------------------------------
    # PERCEIVE
    # --------------------------------------------------------

    def perceive(self, observation):

        vision = self.vision(
            observation
        )

        language = self.language(
            observation["instruction"]
        )

        fused = self.fusion(
            vision,
            language
        )

        reasoning = self.reasoner(
            fused
        )

        return reasoning

    # --------------------------------------------------------
    # ACT
    # --------------------------------------------------------

    def act(self, observation):

        reasoning = self.perceive(
            observation
        )

        latent = reasoning["latent"]

        self.last_latent = latent

        # ----------------------------------------------------
        # Retrieve memory
        # ----------------------------------------------------

        memories = self.memory.retrieve(
            observation
        )

        # ----------------------------------------------------
        # Imagination
        # ----------------------------------------------------

        imagined_action, imagined_score = (
            self.planner.plan(
                latent,
                self.world_model
            )
        )

        # ----------------------------------------------------
        # RL policy
        # ----------------------------------------------------

        rl_action, value = (
            self.controller.act(
                latent
            )
        )

        # ----------------------------------------------------
        # Hybrid decision
        # ----------------------------------------------------

        # During the initial research phase,
        # prefer the learned policy while keeping
        # imagination available.

        action = rl_action

        info = {

            "reasoning_confidence":
                reasoning[
                    "confidence"
                ].item(),

            "memory_size":
                len(self.memory),

            "retrieved_memories":
                len(memories),

            "imagined_action":
                imagined_action,

            "imagined_score":
                imagined_score,

            "rl_action":
                rl_action,

            "value":
                value,
        }

        return action, info

    # --------------------------------------------------------
    # LEARN WORLD MODEL
    # --------------------------------------------------------

    def world_model_loss(
        self,
        latent,
        action,
        target_latent,
        target_reward,
        target_done
    ):

        (
            predicted_latent,
            predicted_reward,
            predicted_done
        ) = self.world_model(
            latent,
            action
        )

        latent_loss = F.mse_loss(
            predicted_latent,
            target_latent
        )

        reward_loss = F.mse_loss(
            predicted_reward.squeeze(-1),
            target_reward
        )

        done_loss = F.binary_cross_entropy(
            predicted_done.squeeze(-1),
            target_done
        )

        return (
            latent_loss
            +
            reward_loss
            +
            done_loss
        )


# ============================================================
# PARAMETER REPORT
# ============================================================

def count_parameters(model):

    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


def print_model_report(agent):

    print()
    print("=" * 70)
    print("ARIA v5 MODEL REPORT")
    print("=" * 70)

    modules = {

        "Vision Encoder":
            agent.vision,

        "Language Encoder":
            agent.language,

        "Multimodal Fusion":
            agent.fusion,

        "Neural Reasoner":
            agent.reasoner,

        "World Model":
            agent.world_model,

        "RL Controller":
            agent.controller,
    }

    total = 0

    for name, module in modules.items():

        params = count_parameters(
            module
        )

        total += params

        print(
            f"{name:<25} : "
            f"{params:,}"
        )

    print("-" * 70)

    print(
        f"{'Total trainable':<25} : "
        f"{total:,}"
    )

    print("=" * 70)


# ============================================================
# SANITY TESTS
# ============================================================

def sanity_tests(agent, env):

    print()
    print("=" * 70)
    print("ARIA v5 SANITY TESTS")
    print("=" * 70)

    observation = env.reset()

    # Vision

    vision = agent.vision(
        observation
    )

    assert vision.shape == (
        1,
        agent.cfg.vision_dim
    )

    print(
        "✓ Vision encoder"
    )

    # Language

    language = agent.language(
        observation["instruction"]
    )

    assert language.shape == (
        1,
        agent.cfg.language_dim
    )

    print(
        "✓ Language encoder"
    )

    # Fusion

    fused = agent.fusion(
        vision,
        language
    )

    assert fused.shape == (
        1,
        agent.cfg.hidden_dim
    )

    print(
        "✓ Multimodal fusion"
    )

    # Reasoning

    reasoning = agent.reasoner(
        fused
    )

    assert (
        reasoning["target"].shape
        ==
        (
            1,
            agent.cfg.grid_size
            *
            agent.cfg.grid_size
        )
    )

    print(
        "✓ Neural reasoning"
    )

    # World model

    action = torch.tensor(
        [1],
        device=agent.device
    )

    next_latent, reward, done = (
        agent.world_model(
            reasoning["latent"],
            action
        )
    )

    assert next_latent.shape == (
        1,
        agent.cfg.hidden_dim
    )

    print(
        "✓ World model"
    )

    # Controller

    logits, value = (
        agent.controller(
            reasoning["latent"]
        )
    )

    assert logits.shape == (
        1,
        agent.cfg.num_actions
    )

    print(
        "✓ RL controller"
    )

    # Full action

    action, info = agent.act(
        observation
    )

    assert 0 <= action < 5

    print(
        "✓ End-to-end action"
    )

    print()
    print(
        "All ARIA v5 sanity tests passed."
    )


# ============================================================
# DEMONSTRATION
# ============================================================

def demonstration(agent, env):

    print()
    print("=" * 70)
    print("ARIA v5 DEMONSTRATION")
    print("=" * 70)

    observation = env.reset()

    env.render()

    print()

    print(
        "ARIA cognitive cycle:"
    )

    print(
        "Observation"
        " → Vision"
        " → Language"
        " → Fusion"
        " → Reasoning"
        " → Memory"
        " → World Model"
        " → Imagination"
        " → RL"
        " → Action"
    )

    total_reward = 0.0

    for step in range(
        env.cfg.max_steps
    ):

        action, info = agent.act(
            observation
        )

        (
            next_observation,
            reward,
            done,
            env_info
        ) = env.step(
            action
        )

        total_reward += reward

        agent.memory.store(

            observation,

            action,

            reward,

            next_observation,

            env_info["success"]
        )

        print(
            f"Step {step + 1:02d} | "
            f"Action={ACTIONS[action]} | "
            f"Reward={reward:+.3f} | "
            f"Reasoning={info['reasoning_confidence']:.3f} | "
            f"Memory={info['memory_size']} | "
            f"Imagined={info['imagined_action']}"
        )

        observation = next_observation

        if done:

            break

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
        f"{env_info['distance']:.3f}"
    )

    print(
        f"Total Reward: "
        f"{total_reward:.3f}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("ARIA v5.0")
    print("INTEGRATED COGNITIVE AGENT")
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

    agent = ARIAv5(
        cfg
    )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    print_model_report(
        agent
    )

    # --------------------------------------------------------
    # Tests
    # --------------------------------------------------------

    sanity_tests(
        agent,
        env
    )

    # --------------------------------------------------------
    # Demonstration
    # --------------------------------------------------------

    demonstration(
        agent,
        env
    )

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("ARIA v5.0 ARCHITECTURE")
    print("=" * 70)

    print(
        """
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
Planning
   ↓
RL Controller
   ↓
Action
   ↓
Environment
   ↓
New Experience
   ↓
Memory / World Model
        """
    )

    print(
        "ARIA v5 prototype complete."
    )


if __name__ == "__main__":
    main()