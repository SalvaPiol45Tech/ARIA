import math
import random
from dataclasses import dataclass
from typing import Tuple, Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# ARIA v4.0
# LEARNED PLANNING AGENT
#
# v3.2.1:
#   Neural Spatial Grounding -> A* -> Action
#
# v4.0:
#   Visual Scene
#        ↓
#   Neural Planner
#        ↓
#   Action
#
# A* is used ONLY as an oracle during training.
# A* is NOT used during neural inference.
# ============================================================


# ============================================================
# CONFIG
# ============================================================

@dataclass
class Config:

    grid_size: int = 15

    num_obstacles: int = 22
    num_objects: int = 4

    max_steps: int = 100

    train_samples: int = 10000
    val_samples: int = 2000

    epochs: int = 10
    batch_size: int = 128

    learning_rate: float = 1e-3

    hidden_dim: int = 256

    device: str = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    seed: int = 42

    checkpoint: str = (
        "checkpoints/aria_v4_learned_planner.pt"
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
# TYPES
# ============================================================

Position = Tuple[int, int]


# ============================================================
# ENVIRONMENT
# ============================================================

class HardNavigationEnv:

    OBJECT_COLORS = [
        "red",
        "green",
        "blue",
        "yellow"
    ]

    COLOR_IDS = {
        "red": 0,
        "green": 1,
        "blue": 2,
        "yellow": 3
    }

    SYMBOLS = {
        "red": "R",
        "green": "G",
        "blue": "B",
        "yellow": "Y"
    }

    def __init__(self, cfg):

        self.cfg = cfg

        self.size = cfg.grid_size

        self.agent_pos = (0, 0)

        self.goal_pos = (0, 0)

        self.objects = {}

        self.obstacles = set()

        self.target_object = None

        self.step_count = 0

    # --------------------------------------------------------
    # RESET
    # --------------------------------------------------------

    def reset(self):

        self.objects.clear()
        self.obstacles.clear()

        self.step_count = 0

        # Agent
        self.agent_pos = self._random_position()

        # Obstacles
        while len(self.obstacles) < self.cfg.num_obstacles:

            pos = self._random_position()

            if pos != self.agent_pos:

                self.obstacles.add(pos)

        # Objects
        for color in self.OBJECT_COLORS[:self.cfg.num_objects]:

            for _ in range(1000):

                pos = self._random_position()

                if self._valid_position(pos):

                    self.objects[color] = pos
                    break

        # Target
        self.target_object = random.choice(
            list(self.objects.keys())
        )

        self.goal_pos = self.objects[
            self.target_object
        ]

        return self.get_observation()

    # --------------------------------------------------------
    # POSITION
    # --------------------------------------------------------

    def _random_position(self):

        return (
            random.randint(0, self.size - 1),
            random.randint(0, self.size - 1)
        )

    def _valid_position(self, pos):

        if pos in self.obstacles:
            return False

        if pos == self.agent_pos:
            return False

        if pos in self.objects.values():
            return False

        return True

    # --------------------------------------------------------
    # OBSERVATION
    # --------------------------------------------------------

    def get_observation(self):

        return {

            "agent": self.agent_pos,

            "goal": self.goal_pos,

            "target": self.target_object,

            "objects": dict(self.objects),

            "obstacles": set(self.obstacles),

            "instruction":
                f"Go to the {self.target_object} object "
                f"while avoiding obstacles."
        }

    # --------------------------------------------------------
    # STEP
    # --------------------------------------------------------

    def step(self, action):

        self.step_count += 1

        dx, dy = action

        x, y = self.agent_pos

        new_pos = (
            x + dx,
            y + dy
        )

        collision = False

        success = False

        reward = -0.01

        # Invalid
        if (
            new_pos[0] < 0
            or new_pos[0] >= self.size
            or new_pos[1] < 0
            or new_pos[1] >= self.size
            or new_pos in self.obstacles
        ):

            collision = True

            reward -= 0.25

            new_pos = self.agent_pos

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

        # Success
        if self.agent_pos == self.goal_pos:

            reward += 5.0

            success = True

        done = (
            success
            or self.step_count >= self.cfg.max_steps
        )

        info = {

            "success": success,

            "collision": collision,

            "steps": self.step_count,

            "distance":
                self.distance(
                    self.agent_pos,
                    self.goal_pos
                )
        }

        return (
            self.get_observation(),
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
            ["." for _ in range(self.size)]
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

        print("Instruction:")

        print(
            f"Go to the {self.target_object} "
            f"object while avoiding obstacles."
        )

        print()


# ============================================================
# A* ORACLE
# ============================================================

class AStarOracle:

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

    def neighbors(self, pos, obstacles):

        x, y = pos

        for dx, dy in self.ACTIONS:

            nx = x + dx
            ny = y + dy

            if (
                nx < 0
                or nx >= self.size
                or ny < 0
                or ny >= self.size
            ):

                continue

            new_pos = (nx, ny)

            if new_pos in obstacles:

                continue

            yield new_pos, (dx, dy)

    def plan(self, start, goal, obstacles):

        open_set = {start}

        came_from = {}

        action_from = {}

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
                key=lambda x:
                    f_score.get(
                        x,
                        float("inf")
                    )
            )

            if current == goal:

                return self._reconstruct(
                    came_from,
                    action_from,
                    current
                )

            open_set.remove(current)

            for neighbor, action in self.neighbors(
                current,
                obstacles
            ):

                tentative = (
                    g_score[current] + 1
                )

                if tentative < g_score.get(
                    neighbor,
                    float("inf")
                ):

                    came_from[neighbor] = current

                    action_from[neighbor] = action

                    g_score[neighbor] = tentative

                    f_score[neighbor] = (
                        tentative
                        +
                        self.heuristic(
                            neighbor,
                            goal
                        )
                    )

                    open_set.add(neighbor)

        return None

    def _reconstruct(
        self,
        came_from,
        action_from,
        current
    ):

        actions = []

        while current in came_from:

            actions.append(
                action_from[current]
            )

            current = came_from[current]

        actions.reverse()

        return actions


# ============================================================
# GRID ENCODING
# ============================================================

def encode_observation(obs, cfg):

    size = cfg.grid_size

    # Channels:
    #
    # 0 = empty
    # 1 = obstacle
    # 2 = agent
    # 3 = red
    # 4 = green
    # 5 = blue
    # 6 = yellow
    # 7 = target

    grid = np.zeros(
        (8, size, size),
        dtype=np.float32
    )

    # Empty
    grid[0] = 1.0

    # Obstacles
    for x, y in obs["obstacles"]:

        grid[0, y, x] = 0.0

        grid[1, y, x] = 1.0

    # Objects
    for color, (x, y) in obs["objects"].items():

        channel = (
            3
            +
            HardNavigationEnv.COLOR_IDS[color]
        )

        grid[0, y, x] = 0.0

        grid[channel, y, x] = 1.0

    # Agent
    ax, ay = obs["agent"]

    grid[0, ay, ax] = 0.0

    grid[2, ay, ax] = 1.0

    # Target
    gx, gy = obs["goal"]

    grid[7, gy, gx] = 1.0

    return grid


# ============================================================
# NEURAL LEARNED PLANNER
# ============================================================

class LearnedPlanner(nn.Module):

    """
    Neural planner.

    Input:
        spatial grid

    Output:
        action logits

    Actions:

        0 = RIGHT
        1 = LEFT
        2 = DOWN
        3 = UP
    """

    ACTIONS = [
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1)
    ]

    def __init__(self, cfg):

        super().__init__()

        size = cfg.grid_size

        self.encoder = nn.Sequential(

            nn.Conv2d(
                8,
                64,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.Conv2d(
                128,
                128,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.AdaptiveAvgPool2d(
                (1, 1)
            )
        )

        self.policy = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                128,
                cfg.hidden_dim
            ),

            nn.ReLU(),

            nn.Linear(
                cfg.hidden_dim,
                4
            )
        )

    def forward(self, x):

        features = self.encoder(x)

        logits = self.policy(features)

        return logits

    def action(self, x):

        with torch.no_grad():

            logits = self.forward(x)

            index = torch.argmax(
                logits,
                dim=-1
            ).item()

        return self.ACTIONS[index]


# ============================================================
# DATASET GENERATION
# ============================================================

def generate_dataset(
    env,
    oracle,
    cfg,
    samples
):

    inputs = []

    labels = []

    print(
        f"Generating {samples} planning samples..."
    )

    generated = 0

    while generated < samples:

        obs = env.reset()

        actions = oracle.plan(
            obs["agent"],
            obs["goal"],
            obs["obstacles"]
        )

        if actions is None:

            continue

        # Generate one sample from
        # the first action of the optimal path.

        for action in actions:

            if generated >= samples:

                break

            grid = encode_observation(
                obs,
                cfg
            )

            action_index = (
                oracle.ACTIONS.index(
                    action
                )
            )

            inputs.append(grid)

            labels.append(action_index)

            # Simulate expert action
            (
                obs,
                _,
                done,
                _
            ) = env.step(action)

            generated += 1

            if generated % 1000 == 0:

                print(
                    f"  {generated}/{samples}"
                )

            if done:

                break

    return (
        torch.tensor(
            np.array(inputs),
            dtype=torch.float32
        ),
        torch.tensor(
            np.array(labels),
            dtype=torch.long
        )
    )


# ============================================================
# TRAINING
# ============================================================

def train_model(
    model,
    train_x,
    train_y,
    val_x,
    val_y,
    cfg
):

    print()
    print("=" * 70)
    print("TRAINING ARIA v4 LEARNED PLANNER")
    print("=" * 70)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate
    )

    best_accuracy = 0.0

    model.to(cfg.device)

    for epoch in range(1, cfg.epochs + 1):

        model.train()

        permutation = torch.randperm(
            len(train_x)
        )

        total_loss = 0.0

        for start in range(
            0,
            len(train_x),
            cfg.batch_size
        ):

            indices = permutation[
                start:
                start + cfg.batch_size
            ]

            x = train_x[
                indices
            ].to(cfg.device)

            y = train_y[
                indices
            ].to(cfg.device)

            optimizer.zero_grad()

            logits = model(x)

            loss = F.cross_entropy(
                logits,
                y
            )

            loss.backward()

            optimizer.step()

            total_loss += (
                loss.item()
                *
                len(indices)
            )

        train_loss = (
            total_loss
            /
            len(train_x)
        )

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        model.eval()

        with torch.no_grad():

            logits = model(
                val_x.to(cfg.device)
            )

            predictions = torch.argmax(
                logits,
                dim=1
            )

            accuracy = (
                (
                    predictions
                    ==
                    val_y.to(cfg.device)
                )
                .float()
                .mean()
                .item()
            )

        print(
            f"Epoch {epoch:02d}/{cfg.epochs} "
            f"| Train Loss: {train_loss:.4f} "
            f"| Val Accuracy: "
            f"{accuracy * 100:.2f}%"
        )

        if accuracy > best_accuracy:

            best_accuracy = accuracy

            torch.save(
                model.state_dict(),
                cfg.checkpoint
            )

            print(
                "  ✓ Best planner saved"
            )

    print()

    print(
        f"Best validation accuracy: "
        f"{best_accuracy * 100:.2f}%"
    )

    print(
        f"Checkpoint: {cfg.checkpoint}"
    )

    return best_accuracy


# ============================================================
# NEURAL AGENT
# ============================================================

class ARIAv4:

    """
    ARIA v4.

    IMPORTANT:

    During inference there is NO A*.

        Observation
             ↓
        Neural Planner
             ↓
           Action
    """

    def __init__(self, model, cfg):

        self.model = model

        self.cfg = cfg

        self.model.eval()

    def act(self, observation):

        grid = encode_observation(
            observation,
            self.cfg
        )

        x = torch.tensor(
            grid,
            dtype=torch.float32
        ).unsqueeze(0)

        x = x.to(self.cfg.device)

        action = self.model.action(x)

        return action


# ============================================================
# EVALUATION
# ============================================================

def evaluate(
    agent,
    env,
    cfg,
    episodes=100
):

    successes = 0

    collisions = 0

    total_steps = 0

    rewards = []

    distances = []

    for episode in range(episodes):

        obs = env.reset()

        total_reward = 0

        episode_collisions = 0

        final_info = None

        for _ in range(
            cfg.max_steps
        ):

            action = agent.act(
                obs
            )

            (
                obs,
                reward,
                done,
                info
            ) = env.step(
                action
            )

            total_reward += reward

            if info["collision"]:

                episode_collisions += 1

            final_info = info

            if done:

                break

        rewards.append(
            total_reward
        )

        collisions += (
            episode_collisions
        )

        total_steps += (
            final_info["steps"]
        )

        distances.append(
            final_info["distance"]
        )

        if final_info["success"]:

            successes += 1

    success_rate = (
        successes
        /
        episodes
    )

    collision_rate = (
        collisions
        /
        max(total_steps, 1)
    )

    print()
    print("=" * 70)
    print("ARIA v4 NEURAL PLANNER EVALUATION")
    print("=" * 70)

    print(
        f"Average Reward       : "
        f"{np.mean(rewards):.4f}"
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
        f"{np.mean(distances):.4f}"
    )

    print(
        f"Average Steps        : "
        f"{total_steps / episodes:.2f}"
    )

    print("=" * 70)

    return {
        "reward":
            np.mean(rewards),

        "success_rate":
            success_rate,

        "collision_rate":
            collision_rate,

        "final_distance":
            np.mean(distances),

        "steps":
            total_steps / episodes
    }


# ============================================================
# DEMONSTRATION
# ============================================================

def demonstration(
    agent,
    env,
    cfg
):

    print()
    print("=" * 70)
    print("ARIA v4 NEURAL PLANNING DEMONSTRATION")
    print("=" * 70)

    obs = env.reset()

    print()

    print(
        "Instruction:"
    )

    print(
        obs["instruction"]
    )

    env.render()

    print(
        "ARIA is now planning WITHOUT A*..."
    )

    total_reward = 0

    for step in range(
        1,
        cfg.max_steps + 1
    ):

        action = agent.act(
            obs
        )

        (
            obs,
            reward,
            done,
            info
        ) = env.step(
            action
        )

        total_reward += reward

        print(
            f"Step {step:02d} | "
            f"Action={action} | "
            f"Distance={info['distance']:.2f}"
        )

        if done:

            break

    env.render()

    print(
        f"Success: {info['success']}"
    )

    print(
        f"Steps: {info['steps']}"
    )

    print(
        f"Final Distance: "
        f"{info['distance']:.3f}"
    )

    print(
        f"Total Reward: "
        f"{total_reward:.3f}"
    )


# ============================================================
# SANITY TESTS
# ============================================================

def sanity_tests(
    model,
    env,
    cfg
):

    print()
    print("=" * 70)
    print("ARIA v4 SANITY TESTS")
    print("=" * 70)

    obs = env.reset()

    assert "agent" in obs
    assert "goal" in obs
    assert "objects" in obs
    assert "obstacles" in obs

    print(
        "✓ Environment test"
    )

    grid = encode_observation(
        obs,
        cfg
    )

    assert grid.shape == (
        8,
        cfg.grid_size,
        cfg.grid_size
    )

    print(
        "✓ Spatial encoding test"
    )

    x = torch.tensor(
        grid,
        dtype=torch.float32
    ).unsqueeze(0)

    model.eval()

    with torch.no_grad():

        logits = model(
            x
        )

    assert logits.shape == (
        1,
        4
    )

    print(
        "✓ Neural planner forward test"
    )

    action = model.action(
        x
    )

    assert action in [
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1)
    ]

    print(
        "✓ Action prediction test"
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
    print("ARIA v4.0")
    print("LEARNED PLANNING AGENT")
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
        f"Training samples: "
        f"{cfg.train_samples}"
    )

    # --------------------------------------------------------
    # Environment
    # --------------------------------------------------------

    env = HardNavigationEnv(
        cfg
    )

    oracle = AStarOracle(
        cfg.grid_size
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = LearnedPlanner(
        cfg
    )

    parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        f"Model parameters: "
        f"{parameters:,}"
    )

    # --------------------------------------------------------
    # Sanity tests
    # --------------------------------------------------------

    sanity_tests(
        model,
        env,
        cfg
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("CREATING EXPERT PLANNING DATA")
    print("=" * 70)

    train_x, train_y = generate_dataset(
        env,
        oracle,
        cfg,
        cfg.train_samples
    )

    val_x, val_y = generate_dataset(
        env,
        oracle,
        cfg,
        cfg.val_samples
    )

    print()

    print(
        f"Train samples: {len(train_x)}"
    )

    print(
        f"Validation samples: {len(val_x)}"
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    train_model(
        model,
        train_x,
        train_y,
        val_x,
        val_y,
        cfg
    )

    # --------------------------------------------------------
    # Load best
    # --------------------------------------------------------

    print()
    print(
        "=" * 70
    )

    print(
        "LOADING BEST CHECKPOINT"
    )

    print(
        "=" * 70
    )

    model.load_state_dict(
        torch.load(
            cfg.checkpoint,
            map_location=cfg.device
        )
    )

    model.to(
        cfg.device
    )

    print(
        "✓ Loaded checkpoint"
    )

    # --------------------------------------------------------
    # Neural agent
    # --------------------------------------------------------

    agent = ARIAv4(
        model,
        cfg
    )

    # --------------------------------------------------------
    # Demonstration
    # --------------------------------------------------------

    demonstration(
        agent,
        env,
        cfg
    )

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    results = evaluate(
        agent,
        env,
        cfg,
        episodes=100
    )

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("ARIA v4.0 COMPLETE")
    print("=" * 70)

    print()

    print(
        "Architecture:"
    )

    print(
        """
Visual Grid
     ↓
Spatial Encoder
     ↓
Neural Planner
     ↓
Action
     ↓
Environment
     """
    )

    print(
        "Important:"
    )

    print(
        "A* was used only to generate expert training data."
    )

    print(
        "A* is NOT used by ARIA during inference."
    )

    print()

    print(
        f"Checkpoint:"
    )

    print(
        cfg.checkpoint
    )

    print()

    print(
        "Next:"
    )

    print(
        """
ARIA v4.1
    ↓
World Model
    ↓
Imagine future states
    ↓
Evaluate imagined trajectories
    ↓
Model-Based Planning
    """
    )


if __name__ == "__main__":

    main()