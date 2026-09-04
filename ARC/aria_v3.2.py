# ============================================================
# ARIA v3.2
# Neural Reasoning Agent
#
# Visual Scene Understanding
# + Multimodal Transformer
# + Neural Reasoning
# + Spatial Relations
# + Reachability Prediction
# + A* Planning
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

    # Model
    d_model: int = 96
    nhead: int = 4
    vision_depth: int = 2
    reasoning_depth: int = 2
    dropout: float = 0.1

    # Training
    train_samples: int = 3000
    val_samples: int = 600

    batch_size: int = 64
    epochs: int = 6
    learning_rate: float = 3e-4

    # Loss weights
    target_loss_weight: float = 1.0
    distance_loss_weight: float = 0.5
    blocked_loss_weight: float = 0.5
    reachable_loss_weight: float = 0.5
    relation_loss_weight: float = 0.5

    # Evaluation
    eval_episodes: int = 100

    # Checkpoint
    checkpoint_path: str = (
        "checkpoints/aria_v3_2_reasoning.pt"
    )

    # Device
    device: str = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

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
# GRID ENCODING
# ============================================================

EMPTY = 0
OBSTACLE = 1
AGENT = 2

OBJECT_BASE = 3

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
                random.randint(
                    0,
                    self.size - 1
                ),
                random.randint(
                    0,
                    self.size - 1
                ),
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

    def is_reachable(
        self,
        start,
        goal
    ):

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

                if not (
                    0 <= nx < self.size
                    and 0 <= ny < self.size
                ):
                    continue

                if (nx, ny) in self.obstacles:
                    continue

                if (nx, ny) in visited:
                    continue

                visited.add(
                    (nx, ny)
                )

                queue.append(
                    (nx, ny)
                )

        return False

    # --------------------------------------------------------
    # Generate episode
    # --------------------------------------------------------

    def _generate_episode(self):

        self.agent_pos = (
            random.randint(
                0,
                self.size - 1
            ),
            random.randint(
                0,
                self.size - 1
            ),
        )

        self.obstacles = set()

        while (
            len(self.obstacles)
            < self.cfg.num_obstacles
        ):

            pos = (
                random.randint(
                    0,
                    self.size - 1
                ),
                random.randint(
                    0,
                    self.size - 1
                ),
            )

            if pos == self.agent_pos:
                continue

            self.obstacles.add(pos)

        self.objects = {}

        for color in COLOR_TO_ID.keys():

            pos = self._random_free_position()

            self.objects[color] = pos

        self.target_object = random.choice(
            list(self.objects.keys())
        )

        self.goal_pos = self.objects[
            self.target_object
        ]

    # --------------------------------------------------------
    # Reset
    # --------------------------------------------------------

    def reset(self):

        for _ in range(100):

            self._generate_episode()

            if self.is_reachable(
                self.agent_pos,
                self.goal_pos
            ):

                break

        else:

            raise RuntimeError(
                "Could not generate valid scene."
            )

        self.steps = 0

        return self.get_visual_observation()

    # --------------------------------------------------------
    # Instruction
    # --------------------------------------------------------

    def get_instruction(self):

        return (
            f"Go to the {self.target_object} "
            f"object while avoiding obstacles."
        )

    # --------------------------------------------------------
    # Visual grid
    # --------------------------------------------------------

    def get_visual_grid(self):

        grid = np.zeros(
            (
                self.size,
                self.size
            ),
            dtype=np.int64
        )

        # Obstacles
        for x, y in self.obstacles:

            grid[y, x] = OBSTACLE

        # Objects
        for color, pos in self.objects.items():

            x, y = pos

            grid[y, x] = (
                OBJECT_BASE
                + COLOR_TO_ID[color]
            )

        # Agent
        x, y = self.agent_pos

        grid[y, x] = AGENT

        return grid

    # --------------------------------------------------------
    # Visual observation
    # --------------------------------------------------------

    def get_visual_observation(self):

        return {
            "grid":
                self.get_visual_grid(),

            "instruction":
                self.get_instruction(),
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

        # Boundary
        if not (
            0 <= new_x < self.size
            and 0 <= new_y < self.size
        ):

            reward -= 0.10

            collision = True

            new_x = old_x
            new_y = old_y

        # Obstacle
        elif (
            new_x,
            new_y
        ) in self.obstacles:

            reward -= 0.25

            collision = True

            new_x = old_x
            new_y = old_y

        self.agent_pos = (
            new_x,
            new_y
        )

        old_distance = math.dist(
            (old_x, old_y),
            self.goal_pos
        )

        new_distance = math.dist(
            self.agent_pos,
            self.goal_pos
        )

        reward += (
            0.15
            * (
                old_distance
                - new_distance
            )
        )

        self.steps += 1

        done = False
        success = False

        if self.agent_pos == self.goal_pos:

            reward += 5.0

            done = True
            success = True

        elif self.steps >= self.max_steps:

            done = True

        info = {

            "collision":
                collision,

            "success":
                success,

            "distance":
                new_distance,
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

        print()

        print("Instruction:")
        print(
            self.get_instruction()
        )

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

                    for color, object_pos in (
                        self.objects.items()
                    ):

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

    def heuristic(self, a, b):

        return (
            abs(a[0] - b[0])
            + abs(a[1] - b[1])
        )

    def plan(
        self,
        start,
        goal,
        obstacles
    ):

        if start == goal:

            return [start]

        open_set = [start]

        came_from = {}

        g_score = {
            start: 0
        }

        f_score = {
            start:
                self.heuristic(
                    start,
                    goal
                )
        }

        while open_set:

            current = min(
                open_set,
                key=lambda p:
                    f_score.get(
                        p,
                        float("inf")
                    )
            )

            if current == goal:

                path = [current]

                while current in came_from:

                    current = came_from[
                        current
                    ]

                    path.append(
                        current
                    )

                path.reverse()

                return path

            open_set.remove(
                current
            )

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

                tentative_g = (
                    g_score[current]
                    + 1
                )

                if tentative_g < g_score.get(
                    neighbor,
                    float("inf")
                ):

                    came_from[
                        neighbor
                    ] = current

                    g_score[
                        neighbor
                    ] = tentative_g

                    f_score[
                        neighbor
                    ] = (
                        tentative_g
                        + self.heuristic(
                            neighbor,
                            goal
                        )
                    )

                    if neighbor not in open_set:

                        open_set.append(
                            neighbor
                        )

        return None

    def path_to_actions(
        self,
        path
    ):

        if path is None:

            return []

        actions = []

        for i in range(
            len(path) - 1
        ):

            x1, y1 = path[i]

            x2, y2 = path[
                i + 1
            ]

            actions.append(
                (
                    x2 - x1,
                    y2 - y1
                )
            )

        return actions


# ============================================================
# INSTRUCTION
# ============================================================

def instruction_to_color_id(
    instruction
):

    instruction = instruction.lower()

    for color, idx in COLOR_TO_ID.items():

        if color in instruction:

            return idx

    raise ValueError(
        f"Unknown instruction: {instruction}"
    )


# ============================================================
# SCENE ANALYSIS
# ============================================================

def analyze_scene(
    env
):

    agent = env.agent_pos

    target = env.goal_pos

    distance = math.dist(
        agent,
        target
    )

    # --------------------------------------------------------
    # Direct line test
    #
    # Check whether the straight Manhattan route
    # encounters an obstacle.
    # --------------------------------------------------------

    blocked = False

    x, y = agent

    target_x, target_y = target

    # Horizontal first
    step = (
        1
        if target_x > x
        else -1
    )

    while x != target_x:

        x += step

        if (x, y) in env.obstacles:

            blocked = True
            break

    # Vertical
    if not blocked:

        step = (
            1
            if target_y > y
            else -1
        )

        while y != target_y:

            y += step

            if (x, y) in env.obstacles:

                blocked = True
                break

    reachable = env.is_reachable(
        agent,
        target
    )

    # --------------------------------------------------------
    # Object distances
    # --------------------------------------------------------

    object_distances = {}

    for color, pos in env.objects.items():

        object_distances[color] = math.dist(
            agent,
            pos
        )

    nearest_object = min(
        object_distances,
        key=object_distances.get
    )

    # --------------------------------------------------------
    # Target relation
    # --------------------------------------------------------

    tx, ty = target
    ax, ay = agent

    if abs(tx - ax) > abs(ty - ay):

        if tx > ax:
            direction = 0
        else:
            direction = 1

    else:

        if ty > ay:
            direction = 2
        else:
            direction = 3

    return {

        "distance":
            distance,

        "blocked":
            int(blocked),

        "reachable":
            int(reachable),

        "nearest_object":
            COLOR_TO_ID[
                nearest_object
            ],

        "direction":
            direction,
    }


# ============================================================
# VISUAL ENCODER
# ============================================================

class VisualEncoder(nn.Module):

    def __init__(
        self,
        cfg
    ):

        super().__init__()

        num_cells = (
            cfg.grid_size
            * cfg.grid_size
        )

        self.cell_embedding = nn.Embedding(
            7,
            cfg.d_model
        )

        self.position_embedding = nn.Parameter(
            torch.randn(
                1,
                num_cells,
                cfg.d_model
            )
            * 0.02
        )

        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=
                cfg.d_model * 4,
            dropout=cfg.dropout,
            batch_first=True,
            norm_first=False,
            activation="gelu",
        )

        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=cfg.vision_depth
        )

    def forward(
        self,
        grid
    ):

        batch_size = grid.size(0)

        x = grid.reshape(
            batch_size,
            -1
        )

        x = self.cell_embedding(x)

        x = (
            x
            + self.position_embedding
        )

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

    def forward(
        self,
        color_ids
    ):

        x = self.embedding(
            color_ids
        )

        return self.mlp(x)


# ============================================================
# NEURAL REASONER
# ============================================================

class NeuralReasoner(nn.Module):

    def __init__(
        self,
        cfg
    ):

        super().__init__()

        self.cfg = cfg

        self.visual_encoder = VisualEncoder(
            cfg
        )

        self.instruction_encoder = (
            InstructionEncoder(
                cfg.d_model
            )
        )

        # ----------------------------------------------------
        # Multimodal reasoning transformer
        # ----------------------------------------------------

        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=
                cfg.d_model * 4,
            dropout=cfg.dropout,
            batch_first=True,
            norm_first=False,
            activation="gelu",
        )

        self.reasoning_transformer = (
            nn.TransformerEncoder(
                layer,
                num_layers=
                    cfg.reasoning_depth
            )
        )

        # ----------------------------------------------------
        # Scene pooling
        # ----------------------------------------------------

        self.scene_query = nn.Parameter(
            torch.randn(
                1,
                1,
                cfg.d_model
            )
            * 0.02
        )

        self.scene_attention = nn.MultiheadAttention(
            embed_dim=cfg.d_model,
            num_heads=cfg.nhead,
            batch_first=True
        )

        # ----------------------------------------------------
        # Shared reasoning representation
        # ----------------------------------------------------

        self.reasoning_mlp = nn.Sequential(

            nn.Linear(
                cfg.d_model,
                cfg.d_model
            ),

            nn.GELU(),

            nn.LayerNorm(
                cfg.d_model
            )
        )

        # ----------------------------------------------------
        # Target position
        # ----------------------------------------------------

        self.target_head = nn.Sequential(

            nn.Linear(
                cfg.d_model,
                cfg.d_model
            ),

            nn.GELU(),

            nn.Linear(
                cfg.d_model,
                cfg.grid_size
                * cfg.grid_size
            )
        )

        # ----------------------------------------------------
        # Distance
        # ----------------------------------------------------

        self.distance_head = nn.Sequential(

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

        # ----------------------------------------------------
        # Direct path blocked?
        # ----------------------------------------------------

        self.blocked_head = nn.Linear(
            cfg.d_model,
            2
        )

        # ----------------------------------------------------
        # Reachable?
        # ----------------------------------------------------

        self.reachable_head = nn.Linear(
            cfg.d_model,
            2
        )

        # ----------------------------------------------------
        # Nearest object
        # ----------------------------------------------------

        self.nearest_object_head = nn.Linear(
            cfg.d_model,
            4
        )

        # ----------------------------------------------------
        # Direction
        #
        # 0 = right
        # 1 = left
        # 2 = down
        # 3 = up
        # ----------------------------------------------------

        self.direction_head = nn.Linear(
            cfg.d_model,
            4
        )

    def forward(
        self,
        grid,
        color_ids
    ):

        # ----------------------------------------------------
        # Visual representation
        # ----------------------------------------------------

        visual_tokens = (
            self.visual_encoder(
                grid
            )
        )

        # ----------------------------------------------------
        # Language
        # ----------------------------------------------------

        instruction_embedding = (
            self.instruction_encoder(
                color_ids
            )
        )

        instruction_embedding = (
            instruction_embedding
            .unsqueeze(1)
        )

        # ----------------------------------------------------
        # Multimodal fusion
        # ----------------------------------------------------

        x = (
            visual_tokens
            + instruction_embedding
        )

        x = self.reasoning_transformer(
            x
        )

        # ----------------------------------------------------
        # Scene attention pooling
        # ----------------------------------------------------

        query = (
            self.scene_query
            .expand(
                x.size(0),
                -1,
                -1
            )
        )

        scene, _ = self.scene_attention(
            query,
            x,
            x
        )

        scene = scene.squeeze(1)

        # ----------------------------------------------------
        # Reasoning representation
        # ----------------------------------------------------

        reasoning_state = (
            self.reasoning_mlp(
                scene
            )
        )

        # ----------------------------------------------------
        # Heads
        # ----------------------------------------------------

        target_logits = self.target_head(
            reasoning_state
        )

        distance = self.distance_head(
            reasoning_state
        ).squeeze(-1)

        blocked_logits = self.blocked_head(
            reasoning_state
        )

        reachable_logits = self.reachable_head(
            reasoning_state
        )

        nearest_logits = (
            self.nearest_object_head(
                reasoning_state
            )
        )

        direction_logits = (
            self.direction_head(
                reasoning_state
            )
        )

        return {

            "target":
                target_logits,

            "distance":
                distance,

            "blocked":
                blocked_logits,

            "reachable":
                reachable_logits,

            "nearest_object":
                nearest_logits,

            "direction":
                direction_logits,

            "reasoning_state":
                reasoning_state,
        }


# ============================================================
# DATASET
# ============================================================

def generate_reasoning_dataset(
    cfg,
    num_samples
):

    env = HardNavigationEnv(
        cfg
    )

    grids = []

    color_ids = []

    target_positions = []

    distances = []

    blocked = []

    reachable = []

    nearest_objects = []

    directions = []

    print(
        f"Generating {num_samples} reasoning samples..."
    )

    for _ in range(num_samples):

        observation = env.reset()

        analysis = analyze_scene(
            env
        )

        grid = observation["grid"]

        color_id = (
            instruction_to_color_id(
                observation[
                    "instruction"
                ]
            )
        )

        tx, ty = env.goal_pos

        target_index = (
            ty * cfg.grid_size
            + tx
        )

        grids.append(grid)

        color_ids.append(
            color_id
        )

        target_positions.append(
            target_index
        )

        distances.append(
            analysis["distance"]
        )

        blocked.append(
            analysis["blocked"]
        )

        reachable.append(
            analysis["reachable"]
        )

        nearest_objects.append(
            analysis["nearest_object"]
        )

        directions.append(
            analysis["direction"]
        )

    return TensorDataset(

        torch.tensor(
            np.array(grids),
            dtype=torch.long
        ),

        torch.tensor(
            np.array(color_ids),
            dtype=torch.long
        ),

        torch.tensor(
            np.array(target_positions),
            dtype=torch.long
        ),

        torch.tensor(
            np.array(distances),
            dtype=torch.float32
        ),

        torch.tensor(
            np.array(blocked),
            dtype=torch.long
        ),

        torch.tensor(
            np.array(reachable),
            dtype=torch.long
        ),

        torch.tensor(
            np.array(nearest_objects),
            dtype=torch.long
        ),

        torch.tensor(
            np.array(directions),
            dtype=torch.long
        ),
    )


# ============================================================
# TRAINING
# ============================================================

def train_reasoner(
    cfg
):

    print()
    print("=" * 70)
    print("TRAINING ARIA v3.2 NEURAL REASONER")
    print("=" * 70)

    train_dataset = (
        generate_reasoning_dataset(
            cfg,
            cfg.train_samples
        )
    )

    val_dataset = (
        generate_reasoning_dataset(
            cfg,
            cfg.val_samples
        )
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

    model = NeuralReasoner(
        cfg
    ).to(cfg.device)

    num_parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        f"Model parameters: "
        f"{num_parameters:,}"
    )

    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=0.01
    )

    best_score = 0.0

    for epoch in range(
        1,
        cfg.epochs + 1
    ):

        model.train()

        running_loss = 0.0

        total = 0

        for batch in train_loader:

            (
                grids,
                color_ids,
                target_positions,
                distances,
                blocked_labels,
                reachable_labels,
                nearest_labels,
                direction_labels,
            ) = batch

            grids = grids.to(
                cfg.device
            )

            color_ids = color_ids.to(
                cfg.device
            )

            target_positions = (
                target_positions.to(
                    cfg.device
                )
            )

            distances = distances.to(
                cfg.device
            )

            blocked_labels = (
                blocked_labels.to(
                    cfg.device
                )
            )

            reachable_labels = (
                reachable_labels.to(
                    cfg.device
                )
            )

            nearest_labels = (
                nearest_labels.to(
                    cfg.device
                )
            )

            direction_labels = (
                direction_labels.to(
                    cfg.device
                )
            )

            # ------------------------------------------------
            # Forward
            # ------------------------------------------------

            outputs = model(
                grids,
                color_ids
            )

            # ------------------------------------------------
            # Losses
            # ------------------------------------------------

            target_loss = F.cross_entropy(
                outputs["target"],
                target_positions
            )

            distance_loss = F.mse_loss(
                outputs["distance"],
                distances
            )

            blocked_loss = F.cross_entropy(
                outputs["blocked"],
                blocked_labels
            )

            reachable_loss = F.cross_entropy(
                outputs["reachable"],
                reachable_labels
            )

            nearest_loss = F.cross_entropy(
                outputs["nearest_object"],
                nearest_labels
            )

            direction_loss = F.cross_entropy(
                outputs["direction"],
                direction_labels
            )

            loss = (

                cfg.target_loss_weight
                * target_loss

                + cfg.distance_loss_weight
                * distance_loss

                + cfg.blocked_loss_weight
                * blocked_loss

                + cfg.reachable_loss_weight
                * reachable_loss

                + cfg.relation_loss_weight
                * nearest_loss

                + cfg.relation_loss_weight
                * direction_loss
            )

            # ------------------------------------------------
            # Backprop
            # ------------------------------------------------

            optimizer.zero_grad(
                set_to_none=True
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0
            )

            optimizer.step()

            batch_size = (
                target_positions.size(0)
            )

            running_loss += (
                loss.item()
                * batch_size
            )

            total += batch_size

        train_loss = (
            running_loss
            / max(total, 1)
        )

        metrics = evaluate_reasoner(
            model,
            val_loader,
            cfg.device
        )

        print(
            f"Epoch {epoch:02d}/{cfg.epochs} | "
            f"Loss: {train_loss:.4f} | "
            f"Target: {metrics['target_accuracy'] * 100:.2f}% | "
            f"Blocked: {metrics['blocked_accuracy'] * 100:.2f}% | "
            f"Reachable: {metrics['reachable_accuracy'] * 100:.2f}% | "
            f"Nearest: {metrics['nearest_accuracy'] * 100:.2f}% | "
            f"Direction: {metrics['direction_accuracy'] * 100:.2f}%"
        )

        # ----------------------------------------------------
        # Combined score
        # ----------------------------------------------------

        score = (
            metrics["target_accuracy"]
            + metrics["blocked_accuracy"]
            + metrics["reachable_accuracy"]
            + metrics["nearest_accuracy"]
            + metrics["direction_accuracy"]
        ) / 5.0

        if score > best_score:

            best_score = score

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

                    "score":
                        score,
                },
                cfg.checkpoint_path
            )

            print(
                "  ✓ Best reasoning model saved"
            )

    print()

    print(
        f"Best reasoning score: "
        f"{best_score * 100:.2f}%"
    )

    print(
        f"Checkpoint: "
        f"{cfg.checkpoint_path}"
    )

    return model


# ============================================================
# REASONING EVALUATION
# ============================================================

@torch.no_grad()
def evaluate_reasoner(
    model,
    loader,
    device
):

    model.eval()

    total = 0

    target_correct = 0

    blocked_correct = 0

    reachable_correct = 0

    nearest_correct = 0

    direction_correct = 0

    distance_error = 0.0

    for batch in loader:

        (
            grids,
            color_ids,
            target_positions,
            distances,
            blocked_labels,
            reachable_labels,
            nearest_labels,
            direction_labels,
        ) = batch

        grids = grids.to(device)

        color_ids = color_ids.to(device)

        target_positions = (
            target_positions.to(device)
        )

        distances = distances.to(device)

        blocked_labels = (
            blocked_labels.to(device)
        )

        reachable_labels = (
            reachable_labels.to(device)
        )

        nearest_labels = (
            nearest_labels.to(device)
        )

        direction_labels = (
            direction_labels.to(device)
        )

        outputs = model(
            grids,
            color_ids
        )

        target_pred = (
            outputs["target"]
            .argmax(dim=1)
        )

        blocked_pred = (
            outputs["blocked"]
            .argmax(dim=1)
        )

        reachable_pred = (
            outputs["reachable"]
            .argmax(dim=1)
        )

        nearest_pred = (
            outputs["nearest_object"]
            .argmax(dim=1)
        )

        direction_pred = (
            outputs["direction"]
            .argmax(dim=1)
        )

        target_correct += (
            target_pred
            == target_positions
        ).sum().item()

        blocked_correct += (
            blocked_pred
            == blocked_labels
        ).sum().item()

        reachable_correct += (
            reachable_pred
            == reachable_labels
        ).sum().item()

        nearest_correct += (
            nearest_pred
            == nearest_labels
        ).sum().item()

        direction_correct += (
            direction_pred
            == direction_labels
        ).sum().item()

        distance_error += (
            torch.abs(
                outputs["distance"]
                - distances
            )
        ).sum().item()

        total += (
            target_positions.size(0)
        )

    return {

        "target_accuracy":
            target_correct
            / max(total, 1),

        "blocked_accuracy":
            blocked_correct
            / max(total, 1),

        "reachable_accuracy":
            reachable_correct
            / max(total, 1),

        "nearest_accuracy":
            nearest_correct
            / max(total, 1),

        "direction_accuracy":
            direction_correct
            / max(total, 1),

        "distance_error":
            distance_error
            / max(total, 1),
    }


# ============================================================
# ARIA v3.2 AGENT
# ============================================================

class ARIAv3_2:

    def __init__(
        self,
        cfg
    ):

        self.cfg = cfg

        self.model = NeuralReasoner(
            cfg
        ).to(cfg.device)

        self.model.eval()

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
            checkpoint[
                "model_state_dict"
            ]
        )

        self.model.eval()

        print(
            f"✓ Loaded checkpoint: {path}"
        )

    # --------------------------------------------------------
    # Neural reasoning
    # --------------------------------------------------------

    @torch.no_grad()
    def reason(
        self,
        observation
    ):

        grid = torch.tensor(
            observation["grid"],
            dtype=torch.long,
            device=self.cfg.device
        ).unsqueeze(0)

        color_id = (
            instruction_to_color_id(
                observation[
                    "instruction"
                ]
            )
        )

        color_id = torch.tensor(
            [color_id],
            dtype=torch.long,
            device=self.cfg.device
        )

        outputs = self.model(
            grid,
            color_id
        )

        # ----------------------------------------------------
        # Target
        # ----------------------------------------------------

        target_index = (
            outputs["target"]
            .argmax(dim=1)
            .item()
        )

        x = (
            target_index
            % self.cfg.grid_size
        )

        y = (
            target_index
            // self.cfg.grid_size
        )

        target = (
            int(x),
            int(y)
        )

        # ----------------------------------------------------
        # Other reasoning predictions
        # ----------------------------------------------------

        distance = (
            outputs["distance"]
            .item()
        )

        blocked = (
            outputs["blocked"]
            .argmax(dim=1)
            .item()
        )

        reachable = (
            outputs["reachable"]
            .argmax(dim=1)
            .item()
        )

        nearest = (
            outputs["nearest_object"]
            .argmax(dim=1)
            .item()
        )

        direction = (
            outputs["direction"]
            .argmax(dim=1)
            .item()
        )

        target_confidence = (
            torch.softmax(
                outputs["target"],
                dim=1
            )[0, target_index]
            .item()
        )

        return {

            "target":
                target,

            "target_confidence":
                target_confidence,

            "distance":
                distance,

            "blocked":
                blocked,

            "reachable":
                reachable,

            "nearest_object":
                nearest,

            "direction":
                direction,
        }

    # --------------------------------------------------------
    # Planning
    # --------------------------------------------------------

    def plan(
        self,
        observation,
        reasoning
    ):

        grid = observation["grid"]

        # Find agent
        ys, xs = np.where(
            grid == AGENT
        )

        if len(xs) == 0:

            raise RuntimeError(
                "Agent not found."
            )

        start = (
            int(xs[0]),
            int(ys[0])
        )

        # Extract obstacles
        ys, xs = np.where(
            grid == OBSTACLE
        )

        obstacles = set()

        for x, y in zip(xs, ys):

            obstacles.add(
                (
                    int(x),
                    int(y)
                )
            )

        target = reasoning[
            "target"
        ]

        path = self.planner.plan(
            start,
            target,
            obstacles
        )

        actions = (
            self.planner.path_to_actions(
                path
            )
        )

        return {
            "path": path,
            "actions": actions,
            "planning_success":
                path is not None,
        }

    # --------------------------------------------------------
    # Act
    # --------------------------------------------------------

    def act(
        self,
        observation
    ):

        reasoning = self.reason(
            observation
        )

        planning = self.plan(
            observation,
            reasoning
        )

        actions = planning[
            "actions"
        ]

        if actions:

            action = actions[0]

        else:

            action = (0, 0)

        return action, {

            "reasoning":
                reasoning,

            "path":
                planning["path"],

            "planned":
                planning[
                    "planning_success"
                ],
        }


# ============================================================
# DEMONSTRATION
# ============================================================

def demonstration(
    agent,
    env
):

    print()
    print("=" * 70)
    print("ARIA v3.2 DEMONSTRATION")
    print("=" * 70)

    observation = env.reset()

    env.render()

    reasoning = agent.reason(
        observation
    )

    print(
        "Neural Reasoning:"
    )

    print(
        f"Target: "
        f"{reasoning['target']}"
    )

    print(
        f"Actual Target: "
        f"{env.goal_pos}"
    )

    print(
        f"Target Confidence: "
        f"{reasoning['target_confidence']:.4f}"
    )

    print(
        f"Predicted Distance: "
        f"{reasoning['distance']:.3f}"
    )

    print(
        f"Direct Path Blocked: "
        f"{bool(reasoning['blocked'])}"
    )

    print(
        f"Target Reachable: "
        f"{bool(reasoning['reachable'])}"
    )

    print(
        f"Nearest Object: "
        f"{ID_TO_COLOR[reasoning['nearest_object']]}"
    )

    direction_names = [
        "RIGHT",
        "LEFT",
        "DOWN",
        "UP",
    ]

    print(
        f"Direction: "
        f"{direction_names[reasoning['direction']]}"
    )

    print()

    for step in range(
        env.max_steps
    ):

        action, info = agent.act(
            observation
        )

        r = info["reasoning"]

        print(
            f"Step {step + 1:02d} | "
            f"Action={action} | "
            f"Target={r['target']} | "
            f"Conf={r['target_confidence']:.3f}"
        )

        (
            observation,
            reward,
            done,
            env_info
        ) = env.step(
            action
        )

        if done:

            break

    print()

    print(
        f"Success: "
        f"{env_info['success']}"
    )

    print(
        f"Steps: "
        f"{env.steps}"
    )

    print(
        f"Final Distance: "
        f"{env_info['distance']:.3f}"
    )

    env.render()


# ============================================================
# FULL AGENT EVALUATION
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

    target_correct = 0

    blocked_correct = 0

    reachable_correct = 0

    nearest_correct = 0

    planning_successes = 0

    for episode in range(
        episodes
    ):

        observation = env.reset()

        # ----------------------------------------------------
        # Ground-truth reasoning
        # ----------------------------------------------------

        analysis = analyze_scene(
            env
        )

        reasoning = agent.reason(
            observation
        )

        if (
            reasoning["target"]
            == env.goal_pos
        ):

            target_correct += 1

        if (
            reasoning["blocked"]
            == analysis["blocked"]
        ):

            blocked_correct += 1

        if (
            reasoning["reachable"]
            == analysis["reachable"]
        ):

            reachable_correct += 1

        if (
            reasoning["nearest_object"]
            == analysis["nearest_object"]
        ):

            nearest_correct += 1

        episode_planned = False

        total_reward = 0.0

        episode_collisions = 0

        for _ in range(
            cfg.max_steps
        ):

            action, info = agent.act(
                observation
            )

            if info["planned"]:

                episode_planned = True

            (
                observation,
                reward,
                done,
                env_info
            ) = env.step(
                action
            )

            total_reward += reward

            if env_info["collision"]:

                episode_collisions += 1

            if done:

                break

        if env_info["success"]:

            successes += 1

        if episode_planned:

            planning_successes += 1

        collisions += (
            episode_collisions
        )

        rewards.append(
            total_reward
        )

        steps_list.append(
            env.steps
        )

        final_distances.append(
            env_info["distance"]
        )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    total_steps = max(
        sum(steps_list),
        1
    )

    print()
    print("=" * 70)
    print("ARIA v3.2 EVALUATION")
    print("=" * 70)

    print(
        f"Average Reward          : "
        f"{np.mean(rewards):.4f}"
    )

    print(
        f"Target Reasoning        : "
        f"{target_correct / episodes * 100:.2f}%"
    )

    print(
        f"Blocked Reasoning       : "
        f"{blocked_correct / episodes * 100:.2f}%"
    )

    print(
        f"Reachability Reasoning  : "
        f"{reachable_correct / episodes * 100:.2f}%"
    )

    print(
        f"Nearest Object Reasoning: "
        f"{nearest_correct / episodes * 100:.2f}%"
    )

    print(
        f"Navigation Success      : "
        f"{successes / episodes * 100:.2f}%"
    )

    print(
        f"Collision Rate          : "
        f"{collisions / total_steps * 100:.2f}%"
    )

    print(
        f"Final Distance          : "
        f"{np.mean(final_distances):.4f}"
    )

    print(
        f"Average Steps           : "
        f"{np.mean(steps_list):.2f}"
    )

    print(
        f"Planning Rate           : "
        f"{planning_successes / episodes * 100:.2f}%"
    )

    print()


# ============================================================
# SANITY TESTS
# ============================================================

def sanity_tests(
    cfg
):

    print()
    print("=" * 70)
    print("ARIA v3.2 SANITY TESTS")
    print("=" * 70)

    env = HardNavigationEnv(
        cfg
    )

    observation = env.reset()

    assert "grid" in observation

    assert "instruction" in observation

    print(
        "✓ Visual observation"
    )

    model = NeuralReasoner(
        cfg
    ).to(cfg.device)

    grid = torch.tensor(
        observation["grid"],
        dtype=torch.long,
        device=cfg.device
    ).unsqueeze(0)

    color_id = torch.tensor(
        [
            instruction_to_color_id(
                observation[
                    "instruction"
                ]
            )
        ],
        dtype=torch.long,
        device=cfg.device
    )

    outputs = model(
        grid,
        color_id
    )

    assert outputs["target"].shape == (
        1,
        cfg.grid_size * cfg.grid_size
    )

    assert outputs["blocked"].shape == (
        1,
        2
    )

    assert outputs["reachable"].shape == (
        1,
        2
    )

    assert outputs["nearest_object"].shape == (
        1,
        4
    )

    assert outputs["direction"].shape == (
        1,
        4
    )

    print(
        "✓ Neural reasoning outputs"
    )

    analysis = analyze_scene(
        env
    )

    assert "distance" in analysis

    assert "blocked" in analysis

    assert "reachable" in analysis

    assert "nearest_object" in analysis

    print(
        "✓ Ground-truth reasoning labels"
    )

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
    print("ARIA v3.2")
    print("NEURAL REASONING AGENT")
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
    # Sanity
    # --------------------------------------------------------

    sanity_tests(
        CFG
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    train_reasoner(
        CFG
    )

    # --------------------------------------------------------
    # Agent
    # --------------------------------------------------------

    agent = ARIAv3_2(
        CFG
    )

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
    # Demo
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
    # Complete
    # --------------------------------------------------------

    print("=" * 70)
    print("ARIA v3.2 COMPLETE")
    print("=" * 70)

    print()

    print(
        "Pipeline:"
    )

    print(
        "Visual Grid"
        " → Vision Transformer"
        " → Instruction Encoder"
        " → Multimodal Transformer"
        " → Neural Reasoner"
        " → World State"
        " → A* Planner"
        " → Action"
    )

    print()


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()