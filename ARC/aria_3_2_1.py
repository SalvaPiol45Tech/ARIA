import os
import random
from collections import deque
from heapq import heappush, heappop

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader


# ============================================================
# ARIA v3.2.1
# Spatial Neural Reasoning
#
# Visual Grid -> Spatial Transformer -> Target Heatmap -> A*
#
# Important:
# The model keeps one representation per spatial cell.
# We DO NOT globally pool 225 cells before predicting target.
# ============================================================


# ============================================================
# CONFIG
# ============================================================

SEED = 42

GRID_SIZE = 15
NUM_OBSTACLES = 22
NUM_OBJECTS = 4

TRAIN_SAMPLES = 3000
VAL_SAMPLES = 600

EPOCHS = 5
BATCH_SIZE = 64
LR = 3e-4

D_MODEL = 96
N_HEADS = 4
DEPTH = 2
DROPOUT = 0.1

CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_PATH = os.path.join(
    CHECKPOINT_DIR,
    "aria_v3_2_1_spatial_grounding.pt"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# COLORS
# ============================================================

COLOR_NAMES = [
    "none",
    "red",
    "green",
    "blue",
    "yellow",
]

COLOR_TO_ID = {
    "red": 1,
    "green": 2,
    "blue": 3,
    "yellow": 4,
}

ID_TO_COLOR = {
    1: "red",
    2: "green",
    3: "blue",
    4: "yellow",
}


# ============================================================
# ACTIONS
# ============================================================

ACTION_NAMES = {
    0: "stay",
    1: "up",
    2: "down",
    3: "left",
    4: "right",
}

ACTION_DELTAS = {
    0: (0, 0),
    1: (-1, 0),
    2: (1, 0),
    3: (0, -1),
    4: (0, 1),
}


# ============================================================
# UTILS
# ============================================================

def inside(y, x):
    return (
        0 <= y < GRID_SIZE
        and 0 <= x < GRID_SIZE
    )


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def bfs_reachable(start, goal, obstacles):
    """
    Check whether goal is reachable from start.
    """

    if start in obstacles or goal in obstacles:
        return False

    queue = deque([start])
    visited = {start}

    while queue:
        current = queue.popleft()

        if current == goal:
            return True

        y, x = current

        for dy, dx in ACTION_DELTAS.values():
            if dy == 0 and dx == 0:
                continue

            ny = y + dy
            nx = x + dx

            nxt = (ny, nx)

            if not inside(ny, nx):
                continue

            if nxt in obstacles:
                continue

            if nxt in visited:
                continue

            visited.add(nxt)
            queue.append(nxt)

    return False


# ============================================================
# ENVIRONMENT
# ============================================================

class HardNavigationEnv:

    def __init__(
        self,
        grid_size=GRID_SIZE,
        num_obstacles=NUM_OBSTACLES,
        num_objects=NUM_OBJECTS,
    ):

        self.grid_size = grid_size
        self.num_obstacles = num_obstacles
        self.num_objects = num_objects

        self.reset()

    # --------------------------------------------------------

    def reset(self):

        while True:

            self.agent_pos = (
                random.randrange(self.grid_size),
                random.randrange(self.grid_size),
            )

            cells = [
                (y, x)
                for y in range(self.grid_size)
                for x in range(self.grid_size)
            ]

            random.shuffle(cells)

            forbidden = {self.agent_pos}

            self.obstacles = set()

            for cell in cells:

                if len(self.obstacles) >= self.num_obstacles:
                    break

                if cell in forbidden:
                    continue

                self.obstacles.add(cell)

            free_cells = [
                c for c in cells
                if c not in self.obstacles
                and c != self.agent_pos
            ]

            if len(free_cells) < self.num_objects:
                continue

            object_cells = free_cells[:self.num_objects]

            colors = [
                "red",
                "green",
                "blue",
                "yellow",
            ]

            self.objects = {}

            for color, pos in zip(colors, object_cells):
                self.objects[color] = pos

            self.target_color = random.choice(colors)
            self.goal_pos = self.objects[self.target_color]

            if bfs_reachable(
                self.agent_pos,
                self.goal_pos,
                self.obstacles
            ):
                break

        self.steps = 0
        self.done = False

        return self.get_visual_observation()

    # --------------------------------------------------------

    def get_visual_observation(self):

        """
        Visual observation only.

        0 = empty
        1 = obstacle
        2 = agent
        3 = red
        4 = green
        5 = blue
        6 = yellow

        IMPORTANT:
        There is NO target marker.
        """

        grid = torch.zeros(
            self.grid_size,
            self.grid_size,
            dtype=torch.long
        )

        for y, x in self.obstacles:
            grid[y, x] = 1

        ay, ax = self.agent_pos
        grid[ay, ax] = 2

        object_ids = {
            "red": 3,
            "green": 4,
            "blue": 5,
            "yellow": 6,
        }

        for color, pos in self.objects.items():

            y, x = pos

            # Do not overwrite the agent.
            if (y, x) != self.agent_pos:
                grid[y, x] = object_ids[color]

        return grid

    # --------------------------------------------------------

    def step(self, action):

        if self.done:
            return (
                self.get_visual_observation(),
                0.0,
                True,
                {}
            )

        dy, dx = ACTION_DELTAS[action]

        y, x = self.agent_pos

        ny = y + dy
        nx = x + dx

        reward = -0.01

        if not inside(ny, nx):

            reward -= 0.05

        elif (ny, nx) in self.obstacles:

            reward -= 0.20

        else:

            self.agent_pos = (ny, nx)

        self.steps += 1

        if self.agent_pos == self.goal_pos:

            reward += 1.0
            self.done = True

        if self.steps >= 100:

            self.done = True

        return (
            self.get_visual_observation(),
            reward,
            self.done,
            {}
        )


# ============================================================
# VISUAL ENCODER
# ============================================================

class VisualEncoder(nn.Module):

    def __init__(
        self,
        grid_size=GRID_SIZE,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        depth=DEPTH,
        dropout=DROPOUT,
    ):

        super().__init__()

        self.grid_size = grid_size
        self.num_cells = grid_size * grid_size

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
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=depth
        )

        self.norm = nn.LayerNorm(d_model)

    # --------------------------------------------------------

    def forward(self, grid):

        B = grid.size(0)

        x = grid.reshape(
            B,
            self.num_cells
        )

        x = self.cell_embedding(x)

        x = x + self.position_embedding

        x = self.transformer(x)

        x = self.norm(x)

        # IMPORTANT:
        # Shape = [B, 225, D]
        #
        # We preserve every spatial token.
        return x


# ============================================================
# INSTRUCTION ENCODER
# ============================================================

class InstructionEncoder(nn.Module):

    def __init__(
        self,
        d_model=D_MODEL
    ):

        super().__init__()

        self.embedding = nn.Embedding(
            5,
            d_model
        )

        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    # --------------------------------------------------------

    def forward(self, color_id):

        x = self.embedding(color_id)

        x = self.mlp(x)

        return x


# ============================================================
# SPATIAL FUSION
# ============================================================

class SpatialFusion(nn.Module):

    def __init__(
        self,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        depth=2,
        dropout=DROPOUT,
    ):

        super().__init__()

        self.instruction_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.Sigmoid(),
        )

        layers = []

        for _ in range(depth):

            layers.append(
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=n_heads,
                    dim_feedforward=d_model * 4,
                    dropout=dropout,
                    batch_first=True,
                    activation="gelu",
                    norm_first=True,
                )
            )

        self.transformer = nn.Sequential(*layers)

        self.norm = nn.LayerNorm(d_model)

    # --------------------------------------------------------

    def forward(
        self,
        visual_tokens,
        instruction
    ):

        B, N, D = visual_tokens.shape

        instruction = instruction.unsqueeze(1)

        instruction = instruction.expand(
            B,
            N,
            D
        )

        combined = torch.cat(
            [
                visual_tokens,
                instruction
            ],
            dim=-1
        )

        gate = self.instruction_gate(combined)

        fused = visual_tokens + gate * instruction

        fused = self.transformer(fused)

        fused = self.norm(fused)

        # [B, 225, D]
        return fused


# ============================================================
# ARIA v3.2.1 MODEL
# ============================================================

class ARIAv321(nn.Module):

    def __init__(self):

        super().__init__()

        self.visual = VisualEncoder()

        self.instruction = InstructionEncoder()

        self.fusion = SpatialFusion()

        # One score for every spatial cell.
        self.target_head = nn.Sequential(
            nn.Linear(D_MODEL, D_MODEL),
            nn.GELU(),
            nn.Linear(D_MODEL, 1),
        )

        # Auxiliary confidence head.
        self.confidence_head = nn.Sequential(
            nn.Linear(D_MODEL, D_MODEL // 2),
            nn.GELU(),
            nn.Linear(D_MODEL // 2, 1),
        )

    # --------------------------------------------------------

    def forward(
        self,
        grid,
        color_id
    ):

        visual_tokens = self.visual(grid)

        instruction = self.instruction(color_id)

        fused = self.fusion(
            visual_tokens,
            instruction
        )

        # ----------------------------------------------------
        # TARGET HEATMAP
        # ----------------------------------------------------
        #
        # [B, 225, D]
        #       |
        #       v
        # [B, 225, 1]
        #       |
        #       v
        # [B, 225]
        #
        # No global pooling.
        #

        target_logits = self.target_head(
            fused
        ).squeeze(-1)

        # ----------------------------------------------------
        # Confidence
        # ----------------------------------------------------

        confidence = torch.sigmoid(
            self.confidence_head(fused)
        ).squeeze(-1)

        return {
            "target_logits": target_logits,
            "confidence": confidence,
            "spatial_tokens": fused,
        }


# ============================================================
# DATASET GENERATION
# ============================================================

def generate_dataset(num_samples):

    grids = []
    colors = []
    targets = []

    print(
        f"Generating {num_samples} samples..."
    )

    for i in range(num_samples):

        env = HardNavigationEnv()

        grid = env.get_visual_observation()

        color_id = COLOR_TO_ID[
            env.target_color
        ]

        target_y, target_x = env.goal_pos

        target_index = (
            target_y * GRID_SIZE
            + target_x
        )

        grids.append(grid)

        colors.append(color_id)

        targets.append(target_index)

        if (i + 1) % 500 == 0:
            print(
                f"  {i + 1}/{num_samples}"
            )

    grids = torch.stack(grids)

    colors = torch.tensor(
        colors,
        dtype=torch.long
    )

    targets = torch.tensor(
        targets,
        dtype=torch.long
    )

    return grids, colors, targets


# ============================================================
# ACCURACY
# ============================================================

@torch.no_grad()
def target_accuracy(
    model,
    loader
):

    model.eval()

    correct = 0
    total = 0

    for grid, color, target in loader:

        grid = grid.to(DEVICE)
        color = color.to(DEVICE)
        target = target.to(DEVICE)

        output = model(
            grid,
            color
        )

        pred = output[
            "target_logits"
        ].argmax(dim=-1)

        correct += (
            pred == target
        ).sum().item()

        total += target.size(0)

    return correct / total


# ============================================================
# TRAIN
# ============================================================

def train_model(
    model,
    train_loader,
    val_loader
):

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=1e-4
    )

    best_accuracy = 0.0

    print()
    print("=" * 70)
    print("TRAINING ARIA v3.2.1")
    print("=" * 70)

    for epoch in range(1, EPOCHS + 1):

        model.train()

        total_loss = 0.0

        for (
            grid,
            color,
            target
        ) in train_loader:

            grid = grid.to(DEVICE)
            color = color.to(DEVICE)
            target = target.to(DEVICE)

            output = model(
                grid,
                color
            )

            logits = output[
                "target_logits"
            ]

            loss = F.cross_entropy(
                logits,
                target
            )

            optimizer.zero_grad()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0
            )

            optimizer.step()

            total_loss += loss.item()

        train_loss = (
            total_loss
            / len(train_loader)
        )

        val_acc = target_accuracy(
            model,
            val_loader
        )

        print(
            f"Epoch {epoch}/{EPOCHS} | "
            f"Loss: {train_loss:.4f} | "
            f"Val Accuracy: {val_acc * 100:.2f}%"
        )

        if val_acc > best_accuracy:

            best_accuracy = val_acc

            os.makedirs(
                CHECKPOINT_DIR,
                exist_ok=True
            )

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),

                    "accuracy":
                        best_accuracy,

                    "config": {
                        "grid_size":
                            GRID_SIZE,
                        "d_model":
                            D_MODEL,
                        "n_heads":
                            N_HEADS,
                        "depth":
                            DEPTH,
                    }
                },
                CHECKPOINT_PATH
            )

            print(
                f"  ✓ Best model saved: "
                f"{CHECKPOINT_PATH}"
            )

    return best_accuracy


# ============================================================
# A*
# ============================================================

class GridPlanner:

    def __init__(self, grid_size=GRID_SIZE):

        self.grid_size = grid_size

    # --------------------------------------------------------

    def plan(
        self,
        start,
        goal,
        obstacles
    ):

        if start in obstacles:
            return []

        if goal in obstacles:
            return []

        open_set = []

        heappush(
            open_set,
            (
                manhattan(start, goal),
                0,
                start
            )
        )

        came_from = {}

        g_score = {
            start: 0
        }

        counter = 0

        while open_set:

            _, _, current = heappop(
                open_set
            )

            if current == goal:

                return self.reconstruct(
                    came_from,
                    current
                )

            y, x = current

            for dy, dx in [
                (-1, 0),
                (1, 0),
                (0, -1),
                (0, 1),
            ]:

                ny = y + dy
                nx = x + dx

                neighbor = (ny, nx)

                if not inside(ny, nx):
                    continue

                if neighbor in obstacles:
                    continue

                tentative = (
                    g_score[current] + 1
                )

                if (
                    neighbor not in g_score
                    or tentative < g_score[neighbor]
                ):

                    came_from[neighbor] = current

                    g_score[neighbor] = tentative

                    f = (
                        tentative
                        + manhattan(
                            neighbor,
                            goal
                        )
                    )

                    counter += 1

                    heappush(
                        open_set,
                        (
                            f,
                            counter,
                            neighbor
                        )
                    )

        return []

    # --------------------------------------------------------

    def reconstruct(
        self,
        came_from,
        current
    ):

        path = [current]

        while current in came_from:

            current = came_from[current]

            path.append(current)

        path.reverse()

        return path


# ============================================================
# NEURAL GROUNDING
# ============================================================

@torch.no_grad()
def predict_target(
    model,
    grid,
    color_id
):

    model.eval()

    grid = grid.unsqueeze(0).to(DEVICE)

    color_id = torch.tensor(
        [color_id],
        dtype=torch.long,
        device=DEVICE
    )

    output = model(
        grid,
        color_id
    )

    logits = output[
        "target_logits"
    ]

    probabilities = torch.softmax(
        logits,
        dim=-1
    )

    pred_index = probabilities.argmax(
        dim=-1
    ).item()

    confidence = probabilities[
        0,
        pred_index
    ].item()

    y = pred_index // GRID_SIZE
    x = pred_index % GRID_SIZE

    return (
        (y, x),
        confidence
    )


# ============================================================
# VISUAL GRID
# ============================================================

def print_grid(
    grid,
    predicted=None,
    actual=None
):

    symbols = {
        0: ".",
        1: "#",
        2: "A",
        3: "R",
        4: "G",
        5: "B",
        6: "Y",
    }

    print()

    for y in range(GRID_SIZE):

        row = ""

        for x in range(GRID_SIZE):

            pos = (y, x)

            if (
                predicted is not None
                and pos == predicted
                and pos != actual
            ):
                row += "P"

            elif (
                actual is not None
                and pos == actual
            ):
                row += "T"

            else:
                row += symbols[
                    grid[y, x].item()
                ]

            row += " "

        print(row)

    print()


# ============================================================
# DEMONSTRATION
# ============================================================

def demonstration(model):

    print()
    print("=" * 70)
    print("ARIA v3.2.1 DEMONSTRATION")
    print("=" * 70)

    env = HardNavigationEnv()

    grid = env.get_visual_observation()

    color_id = COLOR_TO_ID[
        env.target_color
    ]

    predicted_target, confidence = (
        predict_target(
            model,
            grid,
            color_id
        )
    )

    actual_target = env.goal_pos

    print(
        f"Instruction: "
        f"Go to the {env.target_color} object"
    )

    print(
        f"Actual target:     {actual_target}"
    )

    print(
        f"Neural target:     {predicted_target}"
    )

    print(
        f"Confidence:        {confidence:.4f}"
    )

    print(
        f"Grounding correct: "
        f"{predicted_target == actual_target}"
    )

    print_grid(
        grid,
        predicted_target,
        actual_target
    )

    planner = GridPlanner()

    path = planner.plan(
        env.agent_pos,
        predicted_target,
        env.obstacles
    )

    if not path:

        print(
            "No path found."
        )

        return

    print(
        f"Planned path length: "
        f"{len(path) - 1}"
    )

    print(
        f"Agent start: "
        f"{env.agent_pos}"
    )

    print(
        f"Goal: "
        f"{predicted_target}"
    )

    # Follow the path.
    for next_pos in path[1:]:

        y, x = env.agent_pos

        ny, nx = next_pos

        dy = ny - y
        dx = nx - x

        if dy == -1:
            action = 1

        elif dy == 1:
            action = 2

        elif dx == -1:
            action = 3

        elif dx == 1:
            action = 4

        else:
            action = 0

        _, _, done, _ = env.step(
            action
        )

        if done:
            break

    print(
        f"Final position: "
        f"{env.agent_pos}"
    )

    print(
        f"Success: "
        f"{env.agent_pos == actual_target}"
    )


# ============================================================
# EVALUATION
# ============================================================

@torch.no_grad()
def evaluate_agent(
    model,
    episodes=100
):

    print()
    print("=" * 70)
    print("EVALUATION")
    print("=" * 70)

    grounding_correct = 0
    navigation_success = 0
    collisions = 0

    total_steps = 0
    total_distance = 0.0

    planner = GridPlanner()

    for episode in range(episodes):

        env = HardNavigationEnv()

        grid = env.get_visual_observation()

        color_id = COLOR_TO_ID[
            env.target_color
        ]

        predicted_target, confidence = (
            predict_target(
                model,
                grid,
                color_id
            )
        )

        if predicted_target == env.goal_pos:
            grounding_correct += 1

        path = planner.plan(
            env.agent_pos,
            predicted_target,
            env.obstacles
        )

        if not path:

            total_distance += manhattan(
                env.agent_pos,
                env.goal_pos
            )

            continue

        for next_pos in path[1:]:

            y, x = env.agent_pos

            ny, nx = next_pos

            dy = ny - y
            dx = nx - x

            if dy == -1:
                action = 1

            elif dy == 1:
                action = 2

            elif dx == -1:
                action = 3

            elif dx == 1:
                action = 4

            else:
                action = 0

            old_pos = env.agent_pos

            _, reward, done, _ = env.step(
                action
            )

            if (
                env.agent_pos == old_pos
                and action != 0
            ):
                collisions += 1

            if done:
                break

        final_distance = manhattan(
            env.agent_pos,
            env.goal_pos
        )

        total_distance += final_distance

        total_steps += env.steps

        if env.agent_pos == env.goal_pos:
            navigation_success += 1

    grounding_rate = (
        grounding_correct / episodes
    )

    success_rate = (
        navigation_success / episodes
    )

    collision_rate = (
        collisions
        / max(total_steps, 1)
    )

    avg_steps = (
        total_steps
        / episodes
    )

    avg_distance = (
        total_distance
        / episodes
    )

    print(
        f"Grounding Accuracy : "
        f"{grounding_rate * 100:.2f}%"
    )

    print(
        f"Navigation Success : "
        f"{success_rate * 100:.2f}%"
    )

    print(
        f"Collision Rate     : "
        f"{collision_rate * 100:.2f}%"
    )

    print(
        f"Final Distance     : "
        f"{avg_distance:.4f}"
    )

    print(
        f"Average Steps      : "
        f"{avg_steps:.2f}"
    )

    return {
        "grounding": grounding_rate,
        "success": success_rate,
        "collision": collision_rate,
        "distance": avg_distance,
        "steps": avg_steps,
    }


# ============================================================
# ORACLE COMPARISON
# ============================================================

def oracle_evaluation(
    episodes=100
):

    print()
    print("=" * 70)
    print("ORACLE A* COMPARISON")
    print("=" * 70)

    success = 0

    planner = GridPlanner()

    for _ in range(episodes):

        env = HardNavigationEnv()

        path = planner.plan(
            env.agent_pos,
            env.goal_pos,
            env.obstacles
        )

        if not path:
            continue

        for next_pos in path[1:]:

            y, x = env.agent_pos

            ny, nx = next_pos

            dy = ny - y
            dx = nx - x

            if dy == -1:
                action = 1

            elif dy == 1:
                action = 2

            elif dx == -1:
                action = 3

            elif dx == 1:
                action = 4

            else:
                action = 0

            _, _, done, _ = env.step(
                action
            )

            if done:
                break

        if env.agent_pos == env.goal_pos:
            success += 1

    rate = success / episodes

    print(
        f"Oracle Navigation Success: "
        f"{rate * 100:.2f}%"
    )

    return rate


# ============================================================
# SANITY TESTS
# ============================================================

def sanity_tests():

    print()
    print("=" * 70)
    print("SANITY TESTS")
    print("=" * 70)

    env = HardNavigationEnv()

    grid = env.get_visual_observation()

    assert grid.shape == (
        GRID_SIZE,
        GRID_SIZE
    )

    assert grid.min() >= 0
    assert grid.max() <= 6

    model = ARIAv321().to(DEVICE)

    test_grid = grid.unsqueeze(0).to(DEVICE)

    color = torch.tensor(
        [
            COLOR_TO_ID[
                env.target_color
            ]
        ],
        device=DEVICE
    )

    output = model(
        test_grid,
        color
    )

    assert output[
        "target_logits"
    ].shape == (
        1,
        GRID_SIZE * GRID_SIZE
    )

    assert output[
        "spatial_tokens"
    ].shape == (
        1,
        GRID_SIZE * GRID_SIZE,
        D_MODEL
    )

    print("✓ Environment test passed")
    print("✓ Model forward test passed")
    print("✓ Spatial representation test passed")
    print("✓ Target heatmap shape test passed")

    print(
        "\nAll sanity tests passed."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("ARIA v3.2.1")
    print("Spatial Neural Reasoning")
    print("=" * 70)

    print(
        f"Device: {DEVICE}"
    )

    print(
        f"Grid: {GRID_SIZE}x{GRID_SIZE}"
    )

    print(
        f"Obstacles: {NUM_OBSTACLES}"
    )

    print(
        f"Objects: {NUM_OBJECTS}"
    )

    sanity_tests()

    # --------------------------------------------------------
    # DATA
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("CREATING TRAINING DATA")
    print("=" * 70)

    train_grids, train_colors, train_targets = (
        generate_dataset(
            TRAIN_SAMPLES
        )
    )

    val_grids, val_colors, val_targets = (
        generate_dataset(
            VAL_SAMPLES
        )
    )

    train_dataset = TensorDataset(
        train_grids,
        train_colors,
        train_targets
    )

    val_dataset = TensorDataset(
        val_grids,
        val_colors,
        val_targets
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    print(
        f"\nTrain samples: "
        f"{len(train_dataset)}"
    )

    print(
        f"Validation samples: "
        f"{len(val_dataset)}"
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model = ARIAv321().to(DEVICE)

    parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print()
    print("=" * 70)
    print("MODEL")
    print("=" * 70)

    print(
        f"Total parameters: "
        f"{parameters:,}"
    )

    print(
        f"Trainable parameters: "
        f"{trainable:,}"
    )

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    best_accuracy = train_model(
        model,
        train_loader,
        val_loader
    )

    # --------------------------------------------------------
    # LOAD BEST
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("LOADING BEST CHECKPOINT")
    print("=" * 70)

    if os.path.exists(CHECKPOINT_PATH):

        checkpoint = torch.load(
            CHECKPOINT_PATH,
            map_location=DEVICE
        )

        model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )

        print(
            f"✓ Loaded checkpoint"
        )

        print(
            f"Best accuracy: "
            f"{checkpoint['accuracy'] * 100:.2f}%"
        )

    # --------------------------------------------------------
    # DEMO
    # --------------------------------------------------------

    demonstration(model)

    # --------------------------------------------------------
    # EVALUATION
    # --------------------------------------------------------

    results = evaluate_agent(
        model,
        episodes=100
    )

    # --------------------------------------------------------
    # ORACLE
    # --------------------------------------------------------

    oracle_success = oracle_evaluation(
        episodes=100
    )

    # --------------------------------------------------------
    # FINAL REPORT
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("ARIA v3.2.1 FINAL REPORT")
    print("=" * 70)

    print(
        f"Neural Grounding : "
        f"{results['grounding'] * 100:.2f}%"
    )

    print(
        f"Navigation        : "
        f"{results['success'] * 100:.2f}%"
    )

    print(
        f"Oracle A*         : "
        f"{oracle_success * 100:.2f}%"
    )

    print(
        f"Collision Rate    : "
        f"{results['collision'] * 100:.2f}%"
    )

    print(
        f"Final Distance    : "
        f"{results['distance']:.4f}"
    )

    print(
        f"Average Steps     : "
        f"{results['steps']:.2f}"
    )

    print()
    print(
        "Checkpoint:"
    )

    print(
        CHECKPOINT_PATH
    )

    print()
    print(
        "ARIA v3.2.1 completed."
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()