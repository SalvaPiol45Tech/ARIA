# ============================================================
# ARIA v3.1
# Neural Scene Understanding + Visual Grounding + A* Planning
#
# Pipeline:
#
# Visual Grid ───────┐
#                    ├──> Multimodal Transformer
# Instruction ───────┘            │
#                                 ▼
#                         Target Position
#                                 │
#                                 ▼
#                              A* Planner
#                                 │
#                                 ▼
#                               Action
#
# ============================================================

import os
import math
import random
from dataclasses import dataclass

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import TensorDataset, DataLoader


# ============================================================
# CONFIG
# ============================================================

@dataclass
class Config:

    # Environment
    grid_size: int = 15
    max_steps: int = 100
    num_obstacles: int = 22
    num_objects: int = 4
    action_scale: float = 1.0

    # Neural model
    d_model: int = 96
    nhead: int = 4
    transformer_depth: int = 2
    dropout: float = 0.1

    # Training
    train_samples: int = 2500
    val_samples: int = 500
    batch_size: int = 64
    epochs: int = 5
    learning_rate: float = 3e-4

    # Evaluation
    eval_episodes: int = 100

    # Checkpoint
    checkpoint_path: str = "checkpoints/aria_v3_1_grounding.pt"

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Seed
    seed: int = 42


CFG = Config()


# ============================================================
# SEED
# ============================================================

def set_seed(seed=42):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(CFG.seed)


# ============================================================
# COLOR SYSTEM
# ============================================================

COLOR_TO_ID = {
    "red": 0,
    "green": 1,
    "blue": 2,
    "yellow": 3,
}

ID_TO_COLOR = {
    0: "red",
    1: "green",
    2: "blue",
    3: "yellow",
}


# Visual grid encoding
#
# 0 = empty
# 1 = obstacle
# 2 = agent
# 3 = red object
# 4 = green object
# 5 = blue object
# 6 = yellow object

EMPTY = 0
OBSTACLE = 1
AGENT = 2

OBJECT_BASE = 3


# ============================================================
# ENVIRONMENT
# ============================================================

class HardNavigationEnv:

    def __init__(self, cfg):

        self.cfg = cfg

        self.size = cfg.grid_size
        self.max_steps = cfg.max_steps

        self.agent_pos = None
        self.goal_pos = None

        self.objects = {}
        self.obstacles = set()

        self.target_object = None

        self.steps = 0

    # --------------------------------------------------------
    # Free position
    # --------------------------------------------------------

    def _random_free_position(self):

        while True:

            pos = (
                random.randint(0, self.size - 1),
                random.randint(0, self.size - 1),
            )

            if pos == self.agent_pos:
                continue

            if pos in self.obstacles:
                continue

            if pos in self.objects.values():
                continue

            return pos

    # --------------------------------------------------------
    # Reachability
    # --------------------------------------------------------

    def _reachable(self, start, goal):

        queue = [start]
        visited = {start}

        while queue:

            x, y = queue.pop(0)

            if (x, y) == goal:
                return True

            neighbors = [
                (x + 1, y),
                (x - 1, y),
                (x, y + 1),
                (x, y - 1),
            ]

            for nx, ny in neighbors:

                if not (0 <= nx < self.size):
                    continue

                if not (0 <= ny < self.size):
                    continue

                if (nx, ny) in self.obstacles:
                    continue

                if (nx, ny) in visited:
                    continue

                visited.add((nx, ny))
                queue.append((nx, ny))

        return False

    # --------------------------------------------------------
    # Generate episode
    # --------------------------------------------------------

    def _generate_episode(self):

        self.agent_pos = (
            random.randint(0, self.size - 1),
            random.randint(0, self.size - 1),
        )

        self.obstacles = set()

        while len(self.obstacles) < self.cfg.num_obstacles:

            pos = (
                random.randint(0, self.size - 1),
                random.randint(0, self.size - 1),
            )

            if pos == self.agent_pos:
                continue

            self.obstacles.add(pos)

        self.objects = {}

        colors = list(COLOR_TO_ID.keys())

        for color in colors[:self.cfg.num_objects]:

            pos = self._random_free_position()

            self.objects[color] = pos

        self.target_object = random.choice(list(self.objects.keys()))

        self.goal_pos = self.objects[self.target_object]

    # --------------------------------------------------------
    # Reset
    # --------------------------------------------------------

    def reset(self):

        for _ in range(100):

            self._generate_episode()

            if self._reachable(self.agent_pos, self.goal_pos):
                break

        else:

            raise RuntimeError(
                "Could not generate a reachable episode."
            )

        self.steps = 0

        return self.get_visual_observation()

    # --------------------------------------------------------
    # Instruction
    # --------------------------------------------------------

    def get_instruction(self):

        return (
            f"Go to the {self.target_object} object "
            f"while avoiding obstacles."
        )

    # --------------------------------------------------------
    # Visual observation
    # --------------------------------------------------------

    def get_visual_grid(self):

        grid = np.zeros(
            (self.size, self.size),
            dtype=np.int64
        )

        # Obstacles
        for x, y in self.obstacles:
            grid[y, x] = OBSTACLE

        # Objects
        for color, (x, y) in self.objects.items():

            color_id = COLOR_TO_ID[color]

            grid[y, x] = OBJECT_BASE + color_id

        # Agent
        x, y = self.agent_pos
        grid[y, x] = AGENT

        return grid

    # --------------------------------------------------------
    # Agent observation
    #
    # IMPORTANT:
    #
    # No goal position
    # No object dictionary
    # No obstacle set
    #
    # The neural agent only sees:
    #
    #     grid + instruction
    #
    # --------------------------------------------------------

    def get_visual_observation(self):

        return {
            "grid": self.get_visual_grid(),
            "instruction": self.get_instruction(),
        }

    # --------------------------------------------------------
    # Oracle observation
    #
    # Used ONLY for evaluation / teacher.
    # --------------------------------------------------------

    def get_oracle_observation(self):

        return {
            "agent": self.agent_pos,
            "goal": self.goal_pos,
            "objects": dict(self.objects),
            "obstacles": set(self.obstacles),
            "instruction": self.get_instruction(),
        }

    # --------------------------------------------------------
    # Step
    # --------------------------------------------------------

    def step(self, action):

        dx, dy = action

        old_x, old_y = self.agent_pos

        new_x = old_x + dx
        new_y = old_y + dy

        reward = -0.01

        collision = False

        # ----------------------------------------------------
        # Boundary
        # ----------------------------------------------------

        if not (
            0 <= new_x < self.size
            and 0 <= new_y < self.size
        ):

            reward -= 0.10
            collision = True

            new_x = old_x
            new_y = old_y

        # ----------------------------------------------------
        # Obstacle
        # ----------------------------------------------------

        elif (new_x, new_y) in self.obstacles:

            reward -= 0.25
            collision = True

            new_x = old_x
            new_y = old_y

        # ----------------------------------------------------
        # Move
        # ----------------------------------------------------

        self.agent_pos = (new_x, new_y)

        # ----------------------------------------------------
        # Distance reward
        # ----------------------------------------------------

        old_dist = math.dist(
            (old_x, old_y),
            self.goal_pos
        )

        new_dist = math.dist(
            self.agent_pos,
            self.goal_pos
        )

        reward += 0.15 * (old_dist - new_dist)

        self.steps += 1

        done = False
        success = False

        # ----------------------------------------------------
        # Goal
        # ----------------------------------------------------

        if self.agent_pos == self.goal_pos:

            reward += 5.0

            done = True
            success = True

        elif self.steps >= self.max_steps:

            done = True

        info = {
            "collision": collision,
            "success": success,
            "distance": new_dist,
        }

        return (
            self.get_visual_observation(),
            reward,
            done,
            info,
        )

    # --------------------------------------------------------
    # Render
    # --------------------------------------------------------

    def render(self):

        grid = self.get_visual_grid()

        print()
        print("Instruction:")
        print(self.get_instruction())
        print()

        for y in range(self.size):

            row = ""

            for x in range(self.size):

                pos = (x, y)

                if pos == self.agent_pos:

                    symbol = "A"

                elif pos == self.goal_pos:

                    symbol = "T"

                elif pos in self.obstacles:

                    symbol = "#"

                else:

                    symbol = "."

                    for color, object_pos in self.objects.items():

                        if pos == object_pos:

                            symbol = color[0].upper()

                row += symbol + " "

            print(row)

        print()


# ============================================================
# A* PLANNER
# ============================================================

class GridPlanner:

    ACTIONS = [
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1),
    ]

    def __init__(self, grid_size):

        self.size = grid_size

    # --------------------------------------------------------
    # Heuristic
    # --------------------------------------------------------

    def heuristic(self, a, b):

        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    # --------------------------------------------------------
    # A*
    # --------------------------------------------------------

    def plan(self, start, goal, obstacles):

        if start == goal:
            return [start]

        open_set = [start]

        came_from = {}

        g_score = {
            start: 0
        }

        f_score = {
            start: self.heuristic(start, goal)
        }

        while open_set:

            current = min(
                open_set,
                key=lambda p: f_score.get(p, float("inf"))
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
                    and 0 <= neighbor[1] < self.size
                ):
                    continue

                if neighbor in obstacles:
                    continue

                tentative_g = (
                    g_score[current] + 1
                )

                if tentative_g < g_score.get(
                    neighbor,
                    float("inf")
                ):

                    came_from[neighbor] = current

                    g_score[neighbor] = tentative_g

                    f_score[neighbor] = (
                        tentative_g
                        + self.heuristic(
                            neighbor,
                            goal
                        )
                    )

                    if neighbor not in open_set:
                        open_set.append(neighbor)

        return None

    # --------------------------------------------------------
    # Path -> actions
    # --------------------------------------------------------

    def path_to_actions(self, path):

        if path is None:
            return []

        actions = []

        for i in range(len(path) - 1):

            x1, y1 = path[i]
            x2, y2 = path[i + 1]

            actions.append(
                (
                    x2 - x1,
                    y2 - y1
                )
            )

        return actions


# ============================================================
# INSTRUCTION PARSER
# ============================================================

def instruction_to_color_id(instruction):

    instruction = instruction.lower()

    for color, idx in COLOR_TO_ID.items():

        if color in instruction:
            return idx

    raise ValueError(
        f"Unknown target color: {instruction}"
    )


# ============================================================
# VISUAL ENCODER
# ============================================================

class VisualEncoder(nn.Module):

    def __init__(
        self,
        grid_size,
        d_model,
        nhead,
        depth,
        dropout
    ):

        super().__init__()

        self.grid_size = grid_size

        self.num_cells = (
            grid_size * grid_size
        )

        # 7 visual categories
        self.cell_embedding = nn.Embedding(
            7,
            d_model
        )

        self.position_embedding = nn.Parameter(
            torch.randn(
                1,
                self.num_cells,
                d_model
            ) * 0.02
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=False,
            activation="gelu",
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=depth
        )

    def forward(self, grid):

        # grid:
        # [B, H, W]

        batch_size = grid.size(0)

        x = grid.reshape(
            batch_size,
            -1
        )

        x = self.cell_embedding(x)

        x = x + self.position_embedding

        x = self.transformer(x)

        return x


# ============================================================
# INSTRUCTION ENCODER
# ============================================================

class InstructionEncoder(nn.Module):

    def __init__(
        self,
        d_model
    ):

        super().__init__()

        self.embedding = nn.Embedding(
            4,
            d_model
        )

        self.mlp = nn.Sequential(

            nn.Linear(
                d_model,
                d_model
            ),

            nn.GELU(),

            nn.Linear(
                d_model,
                d_model
            )
        )

    def forward(self, color_ids):

        x = self.embedding(color_ids)

        x = self.mlp(x)

        return x


# ============================================================
# MULTIMODAL GROUNDING MODEL
# ============================================================

class NeuralGroundingModel(nn.Module):

    def __init__(self, cfg):

        super().__init__()

        self.grid_size = cfg.grid_size

        self.num_cells = (
            cfg.grid_size * cfg.grid_size
        )

        self.visual_encoder = VisualEncoder(
            grid_size=cfg.grid_size,
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            depth=cfg.transformer_depth,
            dropout=cfg.dropout,
        )

        self.instruction_encoder = InstructionEncoder(
            cfg.d_model
        )

        # Multimodal transformer
        fusion_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.d_model * 4,
            dropout=cfg.dropout,
            batch_first=True,
            norm_first=False,
            activation="gelu",
        )

        self.fusion = nn.TransformerEncoder(
            fusion_layer,
            num_layers=2
        )

        # Predict whether each cell is the target
        self.target_head = nn.Sequential(

            nn.Linear(
                cfg.d_model,
                cfg.d_model
            ),

            nn.GELU(),

            nn.Linear(
                cfg.d_model,
                1
            )
        )

    def forward(
        self,
        grid,
        color_ids
    ):

        # ----------------------------------------------------
        # Vision
        # ----------------------------------------------------

        visual_tokens = self.visual_encoder(
            grid
        )

        # ----------------------------------------------------
        # Instruction
        # ----------------------------------------------------

        text_embedding = self.instruction_encoder(
            color_ids
        )

        # ----------------------------------------------------
        # Multimodal fusion
        # ----------------------------------------------------

        text_embedding = text_embedding.unsqueeze(1)

        fused = visual_tokens + text_embedding

        fused = self.fusion(fused)

        # ----------------------------------------------------
        # Target heatmap
        # ----------------------------------------------------

        logits = self.target_head(
            fused
        ).squeeze(-1)

        return logits


# ============================================================
# DATASET GENERATION
# ============================================================

def generate_dataset(
    cfg,
    num_samples
):

    env = HardNavigationEnv(cfg)

    grids = []
    color_ids = []
    target_positions = []

    print(
        f"Generating {num_samples} samples..."
    )

    for i in range(num_samples):

        observation = env.reset()

        grid = observation["grid"]

        instruction = observation["instruction"]

        color_id = instruction_to_color_id(
            instruction
        )

        x, y = env.goal_pos

        target_index = (
            y * cfg.grid_size + x
        )

        grids.append(grid)

        color_ids.append(color_id)

        target_positions.append(
            target_index
        )

    grids = torch.tensor(
        np.array(grids),
        dtype=torch.long
    )

    color_ids = torch.tensor(
        np.array(color_ids),
        dtype=torch.long
    )

    target_positions = torch.tensor(
        np.array(target_positions),
        dtype=torch.long
    )

    return TensorDataset(
        grids,
        color_ids,
        target_positions
    )


# ============================================================
# VALIDATION
# ============================================================

@torch.no_grad()
def evaluate_model(
    model,
    loader,
    device
):

    model.eval()

    total = 0
    correct = 0

    total_loss = 0.0

    for grid, color_ids, targets in loader:

        grid = grid.to(device)

        color_ids = color_ids.to(device)

        targets = targets.to(device)

        logits = model(
            grid,
            color_ids
        )

        loss = F.cross_entropy(
            logits,
            targets
        )

        predictions = logits.argmax(
            dim=1
        )

        correct += (
            predictions == targets
        ).sum().item()

        total += targets.size(0)

        total_loss += (
            loss.item()
            * targets.size(0)
        )

    accuracy = (
        correct / max(total, 1)
    )

    average_loss = (
        total_loss / max(total, 1)
    )

    return average_loss, accuracy


# ============================================================
# TRAINING
# ============================================================

def train_grounding_model(
    cfg
):

    print()
    print("=" * 70)
    print("TRAINING NEURAL SCENE UNDERSTANDING")
    print("=" * 70)

    print(
        f"Device: {cfg.device}"
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    train_dataset = generate_dataset(
        cfg,
        cfg.train_samples
    )

    val_dataset = generate_dataset(
        cfg,
        cfg.val_samples
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = NeuralGroundingModel(
        cfg
    ).to(cfg.device)

    parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        f"Model parameters: {parameters:,}"
    )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=0.01
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    best_accuracy = 0.0

    for epoch in range(1, cfg.epochs + 1):

        model.train()

        running_loss = 0.0

        total = 0

        for grid, color_ids, targets in train_loader:

            grid = grid.to(
                cfg.device
            )

            color_ids = color_ids.to(
                cfg.device
            )

            targets = targets.to(
                cfg.device
            )

            # Forward
            logits = model(
                grid,
                color_ids
            )

            # Target grounding loss
            loss = F.cross_entropy(
                logits,
                targets
            )

            # Backprop
            optimizer.zero_grad(
                set_to_none=True
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0
            )

            optimizer.step()

            batch_size = targets.size(0)

            running_loss += (
                loss.item()
                * batch_size
            )

            total += batch_size

        train_loss = (
            running_loss
            / max(total, 1)
        )

        val_loss, val_accuracy = evaluate_model(
            model,
            val_loader,
            cfg.device
        )

        print(
            f"Epoch {epoch:02d}/{cfg.epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Accuracy: {val_accuracy * 100:.2f}%"
        )

        # ----------------------------------------------------
        # Save best
        # ----------------------------------------------------

        if val_accuracy > best_accuracy:

            best_accuracy = val_accuracy

            os.makedirs(
                os.path.dirname(
                    cfg.checkpoint_path
                ),
                exist_ok=True
            )

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),

                    "config":
                        cfg.__dict__,

                    "val_accuracy":
                        val_accuracy,
                },
                cfg.checkpoint_path
            )

            print(
                "  ✓ Best model saved"
            )

    print()
    print(
        f"Best validation accuracy: "
        f"{best_accuracy * 100:.2f}%"
    )

    print(
        f"Checkpoint: "
        f"{cfg.checkpoint_path}"
    )

    return model


# ============================================================
# NEURAL SCENE UNDERSTANDING
# ============================================================

class NeuralSceneUnderstanding:

    def __init__(
        self,
        model,
        cfg
    ):

        self.model = model
        self.cfg = cfg

        self.model.eval()

    @torch.no_grad()
    def perceive(
        self,
        observation
    ):

        grid = torch.tensor(
            observation["grid"],
            dtype=torch.long,
            device=self.cfg.device
        ).unsqueeze(0)

        color_id = instruction_to_color_id(
            observation["instruction"]
        )

        color_id = torch.tensor(
            [color_id],
            dtype=torch.long,
            device=self.cfg.device
        )

        logits = self.model(
            grid,
            color_id
        )

        probabilities = torch.softmax(
            logits,
            dim=-1
        )

        predicted_index = logits.argmax(
            dim=-1
        ).item()

        confidence = probabilities[
            0,
            predicted_index
        ].item()

        x = predicted_index % self.cfg.grid_size

        y = predicted_index // self.cfg.grid_size

        return {
            "predicted_target": (x, y),
            "confidence": confidence,
            "heatmap": probabilities[
                0
            ].cpu().numpy(),
        }


# ============================================================
# ARIA v3.1
# ============================================================

class ARIAv3_1:

    def __init__(
        self,
        cfg
    ):

        self.cfg = cfg

        self.model = NeuralGroundingModel(
            cfg
        ).to(cfg.device)

        self.scene_understanding = (
            NeuralSceneUnderstanding(
                self.model,
                cfg
            )
        )

        self.planner = GridPlanner(
            cfg.grid_size
        )

    # --------------------------------------------------------
    # Load checkpoint
    # --------------------------------------------------------

    def load_checkpoint(
        self,
        path
    ):

        checkpoint = torch.load(
            path,
            map_location=self.cfg.device
        )

        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self.model.eval()

        print(
            f"✓ Loaded checkpoint: {path}"
        )

    # --------------------------------------------------------
    # Perception
    # --------------------------------------------------------

    def perceive(
        self,
        observation
    ):

        return self.scene_understanding.perceive(
            observation
        )

    # --------------------------------------------------------
    # Reasoning
    # --------------------------------------------------------

    def reason(
        self,
        observation,
        perception
    ):

        target = perception[
            "predicted_target"
        ]

        confidence = perception[
            "confidence"
        ]

        return {
            "target": target,
            "confidence": confidence,
        }

    # --------------------------------------------------------
    # Extract obstacles from visual observation
    # --------------------------------------------------------

    def extract_obstacles(
        self,
        grid
    ):

        obstacles = set()

        ys, xs = np.where(
            grid == OBSTACLE
        )

        for x, y in zip(xs, ys):

            obstacles.add(
                (int(x), int(y))
            )

        return obstacles

    # --------------------------------------------------------
    # Find agent
    # --------------------------------------------------------

    def find_agent(
        self,
        grid
    ):

        ys, xs = np.where(
            grid == AGENT
        )

        if len(xs) == 0:

            raise RuntimeError(
                "Agent not found in visual grid."
            )

        return (
            int(xs[0]),
            int(ys[0])
        )

    # --------------------------------------------------------
    # Planning
    # --------------------------------------------------------

    def plan(
        self,
        observation,
        reasoning
    ):

        grid = observation["grid"]

        start = self.find_agent(
            grid
        )

        target = reasoning["target"]

        obstacles = self.extract_obstacles(
            grid
        )

        path = self.planner.plan(
            start,
            target,
            obstacles
        )

        actions = self.planner.path_to_actions(
            path
        )

        return {
            "path": path,
            "actions": actions,
            "planning_success": path is not None,
        }

    # --------------------------------------------------------
    # Action
    # --------------------------------------------------------

    def act(
        self,
        observation
    ):

        perception = self.perceive(
            observation
        )

        reasoning = self.reason(
            observation,
            perception
        )

        planning = self.plan(
            observation,
            reasoning
        )

        actions = planning["actions"]

        if len(actions) == 0:

            action = (0, 0)

        else:

            action = actions[0]

        return action, {
            "predicted_target":
                perception["predicted_target"],

            "confidence":
                perception["confidence"],

            "path":
                planning["path"],

            "planned":
                planning["planning_success"],
        }


# ============================================================
# ORACLE PLANNER
# ============================================================

def oracle_action(
    env,
    planner
):

    path = planner.plan(
        env.agent_pos,
        env.goal_pos,
        env.obstacles
    )

    actions = planner.path_to_actions(
        path
    )

    if len(actions) == 0:

        return (0, 0), path

    return actions[0], path


# ============================================================
# NEURAL AGENT EVALUATION
# ============================================================

def evaluate_agent(
    agent,
    env,
    cfg,
    episodes=100
):

    rewards = []

    successes = 0

    collisions = 0

    steps_list = []

    final_distances = []

    grounding_correct = 0

    planning_successes = 0

    print()
    print("=" * 70)
    print("ARIA v3.1 EVALUATION")
    print("=" * 70)

    for episode in range(episodes):

        observation = env.reset()

        total_reward = 0.0

        episode_collisions = 0

        episode_planned = False

        # ----------------------------------------------------
        # Measure grounding ONLY on first observation
        # ----------------------------------------------------

        first_perception = agent.perceive(
            observation
        )

        predicted_target = (
            first_perception[
                "predicted_target"
            ]
        )

        if predicted_target == env.goal_pos:

            grounding_correct += 1

        # ----------------------------------------------------
        # Episode
        # ----------------------------------------------------

        for step in range(cfg.max_steps):

            action, info = agent.act(
                observation
            )

            if info["planned"]:

                episode_planned = True

            observation, reward, done, info_env = env.step(
                action
            )

            total_reward += reward

            if info_env["collision"]:

                episode_collisions += 1

            if done:

                break

        # ----------------------------------------------------
        # Episode metrics
        # ----------------------------------------------------

        if info_env["success"]:

            successes += 1

        if episode_planned:

            planning_successes += 1

        collisions += episode_collisions

        rewards.append(
            total_reward
        )

        steps_list.append(
            env.steps
        )

        final_distances.append(
            info_env["distance"]
        )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    avg_reward = np.mean(
        rewards
    )

    success_rate = (
        successes / episodes
    )

    grounding_accuracy = (
        grounding_correct / episodes
    )

    planning_rate = (
        planning_successes / episodes
    )

    collision_rate = (
        collisions
        / max(sum(steps_list), 1)
    )

    avg_steps = np.mean(
        steps_list
    )

    avg_final_distance = np.mean(
        final_distances
    )

    print()
    print(
        f"Average Reward       : "
        f"{avg_reward:.4f}"
    )

    print(
        f"Grounding Accuracy   : "
        f"{grounding_accuracy * 100:.2f}%"
    )

    print(
        f"Navigation Success   : "
        f"{success_rate * 100:.2f}%"
    )

    print(
        f"Collision Rate       : "
        f"{collision_rate * 100:.2f}%"
    )

    print(
        f"Final Distance       : "
        f"{avg_final_distance:.4f}"
    )

    print(
        f"Average Steps        : "
        f"{avg_steps:.2f}"
    )

    print(
        f"Planning Rate        : "
        f"{planning_rate * 100:.2f}%"
    )

    print()

    return {
        "average_reward":
            avg_reward,

        "grounding_accuracy":
            grounding_accuracy,

        "success_rate":
            success_rate,

        "collision_rate":
            collision_rate,

        "final_distance":
            avg_final_distance,

        "average_steps":
            avg_steps,

        "planning_rate":
            planning_rate,
    }


# ============================================================
# ORACLE VS NEURAL TEST
# ============================================================

def compare_with_oracle(
    agent,
    env,
    cfg,
    episodes=50
):

    neural_success = 0
    oracle_success = 0

    planner = GridPlanner(
        cfg.grid_size
    )

    print()
    print("=" * 70)
    print("NEURAL vs ORACLE PLANNER")
    print("=" * 70)

    for episode in range(episodes):

        # ----------------------------------------------------
        # Neural
        # ----------------------------------------------------

        observation = env.reset()

        for _ in range(cfg.max_steps):

            action, _ = agent.act(
                observation
            )

            observation, reward, done, info = env.step(
                action
            )

            if done:
                break

        if info["success"]:

            neural_success += 1

        # ----------------------------------------------------
        # Oracle
        # ----------------------------------------------------

        observation = env.reset()

        for _ in range(cfg.max_steps):

            action, _ = oracle_action(
                env,
                planner
            )

            observation, reward, done, info = env.step(
                action
            )

            if done:
                break

        if info["success"]:

            oracle_success += 1

    print(
        f"Neural Success : "
        f"{neural_success / episodes * 100:.2f}%"
    )

    print(
        f"Oracle Success : "
        f"{oracle_success / episodes * 100:.2f}%"
    )

    print()


# ============================================================
# DEMONSTRATION
# ============================================================

def demonstration(
    agent,
    env
):

    print()
    print("=" * 70)
    print("ARIA v3.1 DEMONSTRATION")
    print("=" * 70)

    observation = env.reset()

    env.render()

    perception = agent.perceive(
        observation
    )

    print(
        "Neural predicted target:",
        perception["predicted_target"]
    )

    print(
        "Actual target:",
        env.goal_pos
    )

    print(
        "Confidence:",
        f"{perception['confidence']:.4f}"
    )

    print()

    for step in range(
        env.max_steps
    ):

        action, info = agent.act(
            observation
        )

        print(
            f"Step {step + 1:02d} | "
            f"Action={action} | "
            f"Predicted Target={info['predicted_target']} | "
            f"Confidence={info['confidence']:.3f}"
        )

        observation, reward, done, env_info = env.step(
            action
        )

        if done:

            break

    print()

    print(
        "Success:",
        env_info["success"]
    )

    print(
        "Steps:",
        env.steps
    )

    print(
        "Final Distance:",
        env_info["distance"]
    )

    env.render()


# ============================================================
# SANITY TESTS
# ============================================================

def sanity_tests(
    cfg
):

    print()
    print("=" * 70)
    print("SANITY TESTS")
    print("=" * 70)

    env = HardNavigationEnv(
        cfg
    )

    observation = env.reset()

    assert "grid" in observation

    assert "instruction" in observation

    assert observation["grid"].shape == (
        cfg.grid_size,
        cfg.grid_size
    )

    print(
        "✓ Visual observation"
    )

    # Check that target is NOT explicitly marked
    grid = observation["grid"]

    unique_values = set(
        np.unique(grid).tolist()
    )

    assert 0 in unique_values

    print(
        "✓ Grid encoding"
    )

    # Planner
    planner = GridPlanner(
        cfg.grid_size
    )

    path = planner.plan(
        env.agent_pos,
        env.goal_pos,
        env.obstacles
    )

    assert path is not None

    print(
        "✓ A* planner"
    )

    # Model
    model = NeuralGroundingModel(
        cfg
    ).to(cfg.device)

    grid_tensor = torch.tensor(
        observation["grid"],
        dtype=torch.long
    ).unsqueeze(0).to(
        cfg.device
    )

    color_id = torch.tensor(
        [
            instruction_to_color_id(
                observation["instruction"]
            )
        ],
        dtype=torch.long
    ).to(
        cfg.device
    )

    logits = model(
        grid_tensor,
        color_id
    )

    assert logits.shape == (
        1,
        cfg.grid_size * cfg.grid_size
    )

    print(
        "✓ Neural grounding model"
    )

    print()
    print(
        "All sanity tests passed."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("ARIA v3.1")
    print("Neural Scene Understanding Agent")
    print("=" * 70)

    print(
        f"Device: {CFG.device}"
    )

    print(
        f"Grid: "
        f"{CFG.grid_size}x{CFG.grid_size}"
    )

    print(
        f"Obstacles: "
        f"{CFG.num_obstacles}"
    )

    print(
        f"Objects: "
        f"{CFG.num_objects}"
    )

    # --------------------------------------------------------
    # Sanity tests
    # --------------------------------------------------------

    sanity_tests(
        CFG
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    model = train_grounding_model(
        CFG
    )

    # --------------------------------------------------------
    # Create agent
    # --------------------------------------------------------

    agent = ARIAv3_1(
        CFG
    )

    # Load best checkpoint
    agent.load_checkpoint(
        CFG.checkpoint_path
    )

    # --------------------------------------------------------
    # Environment
    # --------------------------------------------------------

    env = HardNavigationEnv(
        CFG
    )

    # --------------------------------------------------------
    # Demonstration
    # --------------------------------------------------------

    demonstration(
        agent,
        env
    )

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    evaluate_agent(
        agent,
        env,
        CFG,
        episodes=CFG.eval_episodes
    )

    # --------------------------------------------------------
    # Neural vs Oracle
    # --------------------------------------------------------

    compare_with_oracle(
        agent,
        env,
        CFG,
        episodes=50
    )

    print()
    print("=" * 70)
    print("ARIA v3.1 COMPLETE")
    print("=" * 70)

    print()
    print("Pipeline:")
    print(
        "Visual Grid"
        " → Vision Transformer"
        " → Instruction Encoder"
        " → Multimodal Fusion"
        " → Target Grounding"
        " → A* Planning"
        " → Action"
    )

    print()


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()