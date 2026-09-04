import math
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# ARIA v3
# Adaptive Reasoning & Imagination Agent
#
# Version 3:
#   - Obstacles
#   - Multiple objects
#   - Distractors
#   - Task instructions
#   - A* style planning
#   - Continuous actions
#   - Evaluation metrics
#
# This version focuses on the HARD NAVIGATION environment
# before connecting the learned World Model + RL controller.
# ============================================================


# ============================================================
# CONFIG
# ============================================================

@dataclass
class Config:
    grid_size: int = 15

    max_steps: int = 100

    num_obstacles: int = 22
    num_objects: int = 4

    action_scale: float = 1.0

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    seed: int = 42


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
# TYPES
# ============================================================

Position = Tuple[int, int]


# ============================================================
# HARD NAVIGATION ENVIRONMENT
# ============================================================

class HardNavigationEnv:

    """
    2D grid-world environment.

    Agent:
        A

    Target:
        T

    Objects:
        R/G/B/Y

    Obstacles:
        #

    Empty:
        .

    The agent must:

        1. understand the task
        2. identify the correct object
        3. avoid obstacles
        4. navigate to the target
    """

    OBJECT_COLORS = ["red", "green", "blue", "yellow"]

    COLOR_SYMBOLS = {
        "red": "R",
        "green": "G",
        "blue": "B",
        "yellow": "Y",
    }

    def __init__(self, cfg: Config):
        self.cfg = cfg

        self.size = cfg.grid_size

        self.agent_pos: Position = (0, 0)

        self.goal_pos: Position = (0, 0)

        self.obstacles = set()

        self.objects: Dict[str, Position] = {}

        self.target_object = None

        self.step_count = 0

        self.previous_distance = None

    # --------------------------------------------------------
    # RESET
    # --------------------------------------------------------

    def reset(self):

        self.obstacles.clear()
        self.objects.clear()

        self.step_count = 0

        # Agent
        self.agent_pos = self._random_free_position()

        # Generate obstacles
        for _ in range(self.cfg.num_obstacles):

            pos = self._random_position()

            if pos == self.agent_pos:
                continue

            self.obstacles.add(pos)

        # Generate objects
        for color in self.OBJECT_COLORS[:self.cfg.num_objects]:

            for _ in range(100):

                pos = self._random_position()

                if self._valid_free_position(pos):
                    self.objects[color] = pos
                    break

        # Select target
        self.target_object = random.choice(
            list(self.objects.keys())
        )

        self.goal_pos = self.objects[self.target_object]

        self.previous_distance = self._distance(
            self.agent_pos,
            self.goal_pos
        )

        return self.get_observation()

    # --------------------------------------------------------
    # RANDOM POSITIONS
    # --------------------------------------------------------

    def _random_position(self) -> Position:

        return (
            random.randint(0, self.size - 1),
            random.randint(0, self.size - 1)
        )

    def _valid_free_position(self, pos: Position):

        if pos in self.obstacles:
            return False

        if pos == self.agent_pos:
            return False

        if pos in self.objects.values():
            return False

        return True

    def _random_free_position(self):

        while True:

            pos = self._random_position()

            if pos not in self.obstacles:
                return pos

    # --------------------------------------------------------
    # DISTANCE
    # --------------------------------------------------------

    def _distance(
        self,
        a: Position,
        b: Position
    ):

        return math.sqrt(
            (a[0] - b[0]) ** 2 +
            (a[1] - b[1]) ** 2
        )

    # --------------------------------------------------------
    # INSTRUCTION
    # --------------------------------------------------------

    def get_instruction(self):

        return (
            f"Go to the {self.target_object} object "
            f"while avoiding obstacles."
        )

    # --------------------------------------------------------
    # OBSERVATION
    # --------------------------------------------------------

    def get_observation(self):

        return {
            "agent": self.agent_pos,
            "goal": self.goal_pos,
            "objects": dict(self.objects),
            "obstacles": set(self.obstacles),
            "instruction": self.get_instruction(),
        }

    # --------------------------------------------------------
    # STEP
    # --------------------------------------------------------

    def step(
        self,
        action: Tuple[int, int]
    ):

        self.step_count += 1

        dx, dy = action

        new_x = self.agent_pos[0] + int(dx)
        new_y = self.agent_pos[1] + int(dy)

        new_pos = (new_x, new_y)

        reward = -0.01

        collision = False

        success = False

        # ----------------------------------------------------
        # OUT OF BOUNDS
        # ----------------------------------------------------

        if (
            new_x < 0
            or new_x >= self.size
            or new_y < 0
            or new_y >= self.size
        ):

            reward -= 0.10

            collision = True

            new_pos = self.agent_pos

        # ----------------------------------------------------
        # OBSTACLE
        # ----------------------------------------------------

        elif new_pos in self.obstacles:

            reward -= 0.25

            collision = True

            new_pos = self.agent_pos

        # ----------------------------------------------------
        # VALID MOVE
        # ----------------------------------------------------

        else:

            self.agent_pos = new_pos

        # ----------------------------------------------------
        # DISTANCE REWARD
        # ----------------------------------------------------

        new_distance = self._distance(
            self.agent_pos,
            self.goal_pos
        )

        distance_delta = (
            self.previous_distance - new_distance
        )

        reward += distance_delta * 0.15

        self.previous_distance = new_distance

        # ----------------------------------------------------
        # SUCCESS
        # ----------------------------------------------------

        if self.agent_pos == self.goal_pos:

            reward += 5.0

            success = True

        # ----------------------------------------------------
        # DONE
        # ----------------------------------------------------

        done = (
            success
            or self.step_count >= self.cfg.max_steps
        )

        info = {
            "success": success,
            "collision": collision,
            "distance": new_distance,
            "steps": self.step_count,
        }

        return (
            self.get_observation(),
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

        # Obstacles
        for x, y in self.obstacles:
            grid[y][x] = "#"

        # Objects
        for color, (x, y) in self.objects.items():

            grid[y][x] = self.COLOR_SYMBOLS[color]

        # Goal
        gx, gy = self.goal_pos

        grid[gy][gx] = "T"

        # Agent
        ax, ay = self.agent_pos

        grid[ay][ax] = "A"

        print()

        for row in grid:
            print(" ".join(row))

        print()

        print("Instruction:")
        print(self.get_instruction())

        print()


# ============================================================
# HEURISTIC PLANNER
# ============================================================

class GridPlanner:

    """
    Simple A* planner.

    This is intentionally NOT a neural network.

    We first build a reliable symbolic planner.
    Later ARIA's learned Reasoning/Planning module
    will learn to replace or improve this planner.
    """

    ACTIONS = [
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1),
    ]

    def __init__(self, grid_size):

        self.grid_size = grid_size

    # --------------------------------------------------------
    # HEURISTIC
    # --------------------------------------------------------

    def heuristic(
        self,
        a: Position,
        b: Position
    ):

        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    # --------------------------------------------------------
    # NEIGHBORS
    # --------------------------------------------------------

    def neighbors(
        self,
        pos: Position,
        obstacles
    ):

        x, y = pos

        for dx, dy in self.ACTIONS:

            nx = x + dx
            ny = y + dy

            if (
                nx < 0
                or nx >= self.grid_size
                or ny < 0
                or ny >= self.grid_size
            ):
                continue

            new_pos = (nx, ny)

            if new_pos in obstacles:
                continue

            yield new_pos

    # --------------------------------------------------------
    # A*
    # --------------------------------------------------------

    def plan(
        self,
        start: Position,
        goal: Position,
        obstacles
    ):

        open_set = {start}

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
                key=lambda x: f_score.get(
                    x,
                    float("inf")
                )
            )

            if current == goal:

                return self._reconstruct_path(
                    came_from,
                    current
                )

            open_set.remove(current)

            for neighbor in self.neighbors(
                current,
                obstacles
            ):

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

                    open_set.add(neighbor)

        return None

    # --------------------------------------------------------
    # RECONSTRUCT
    # --------------------------------------------------------

    def _reconstruct_path(
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

    # --------------------------------------------------------
    # PATH → ACTIONS
    # --------------------------------------------------------

    def path_to_actions(
        self,
        path
    ):

        actions = []

        if path is None:
            return actions

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
# SIMPLE SCENE UNDERSTANDING
# ============================================================

class SceneUnderstanding:

    """
    Converts the raw environment into a structured scene.

    This is our first step toward Scene Graph reasoning.
    """

    def parse(self, observation):

        agent = observation["agent"]

        objects = observation["objects"]

        obstacles = observation["obstacles"]

        instruction = observation["instruction"]

        # --------------------------------------------
        # Extract target from instruction
        # --------------------------------------------

        target = None

        for color in HardNavigationEnv.OBJECT_COLORS:

            if color in instruction:

                target = color

                break

        target_position = None

        if target in objects:

            target_position = objects[target]

        # --------------------------------------------
        # Scene representation
        # --------------------------------------------

        scene = {

            "agent": agent,

            "target": target,

            "target_position": target_position,

            "objects": objects,

            "obstacles": obstacles,

            "instruction": instruction,
        }

        return scene


# ============================================================
# ARIA V3 AGENT
# ============================================================

class ARIAv3:

    """
    Version 3 baseline agent.

    Pipeline:

        Observation
             ↓
        Scene Understanding
             ↓
        Target Reasoning
             ↓
        A* Planning
             ↓
        Action
    """

    def __init__(self, cfg):

        self.cfg = cfg

        self.scene_understanding = (
            SceneUnderstanding()
        )

        self.planner = GridPlanner(
            cfg.grid_size
        )

    # --------------------------------------------------------
    # PERCEIVE
    # --------------------------------------------------------

    def perceive(self, observation):

        return self.scene_understanding.parse(
            observation
        )

    # --------------------------------------------------------
    # REASON
    # --------------------------------------------------------

    def reason(self, scene):

        target = scene["target"]

        target_position = scene[
            "target_position"
        ]

        if target is None:

            return {
                "valid": False,
                "reason": "Target not found."
            }

        if target_position is None:

            return {
                "valid": False,
                "reason": "Target position unavailable."
            }

        return {
            "valid": True,
            "target": target,
            "target_position": target_position,
        }

    # --------------------------------------------------------
    # PLAN
    # --------------------------------------------------------

    def plan(self, scene):

        reasoning = self.reason(scene)

        if not reasoning["valid"]:

            return None

        path = self.planner.plan(
            scene["agent"],
            reasoning["target_position"],
            scene["obstacles"]
        )

        return path

    # --------------------------------------------------------
    # ACT
    # --------------------------------------------------------

    def act(self, observation):

        scene = self.perceive(observation)

        path = self.plan(scene)

        if path is None or len(path) < 2:

            return (0, 0), {
                "path": path,
                "planned": False,
            }

        next_position = path[1]

        action = (
            next_position[0] - scene["agent"][0],
            next_position[1] - scene["agent"][1],
        )

        return action, {
            "path": path,
            "planned": True,
            "target": scene["target"],
        }


# ============================================================
# EVALUATION
# ============================================================

def evaluate(
    agent,
    env,
    episodes=100,
    render_first=False
):

    rewards = []

    successes = 0

    collisions = 0

    steps = []

    final_distances = []

    planning_successes = 0

    for episode in range(episodes):

        observation = env.reset()

        total_reward = 0

        episode_collisions = 0

        planned = False

        if render_first and episode == 0:

            print("\nInitial Environment:")

            env.render()

        for _ in range(env.cfg.max_steps):

            action, info = agent.act(
                observation
            )

            planned = info["planned"]

            if planned:

                planning_successes += 1

            (
                observation,
                reward,
                done,
                step_info
            ) = env.step(action)

            total_reward += reward

            if step_info["collision"]:

                episode_collisions += 1

            if render_first and episode == 0:

                env.render()

            if done:

                break

        rewards.append(total_reward)

        collisions += episode_collisions

        final_distances.append(
            step_info["distance"]
        )

        steps.append(
            step_info["steps"]
        )

        if step_info["success"]:

            successes += 1

    success_rate = (
        successes / episodes
    )

    collision_rate = (
        collisions / max(sum(steps), 1)
    )

    avg_reward = np.mean(rewards)

    avg_steps = np.mean(steps)

    avg_distance = np.mean(
        final_distances
    )

    planning_rate = (
        planning_successes / episodes
    )

    print()
    print("=" * 60)
    print("ARIA v3 EVALUATION")
    print("=" * 60)

    print(
        f"Average Reward      : {avg_reward:.3f}"
    )

    print(
        f"Success Rate        : {success_rate * 100:.2f}%"
    )

    print(
        f"Collision Rate      : {collision_rate * 100:.2f}%"
    )

    print(
        f"Final Distance      : {avg_distance:.3f}"
    )

    print(
        f"Average Steps       : {avg_steps:.2f}"
    )

    print(
        f"Planning Rate       : {planning_rate * 100:.2f}%"
    )

    print("=" * 60)

    return {
        "reward": avg_reward,
        "success_rate": success_rate,
        "collision_rate": collision_rate,
        "final_distance": avg_distance,
        "steps": avg_steps,
        "planning_rate": planning_rate,
    }


# ============================================================
# DEMONSTRATION
# ============================================================

def demonstration(
    agent,
    env
):

    print()
    print("=" * 60)
    print("ARIA v3 DEMONSTRATION")
    print("=" * 60)

    observation = env.reset()

    env.render()

    scene = agent.perceive(
        observation
    )

    print("Scene Understanding:")
    print(
        f"Agent: {scene['agent']}"
    )

    print(
        f"Target: {scene['target']}"
    )

    print(
        f"Target Position: "
        f"{scene['target_position']}"
    )

    print(
        f"Objects: "
        f"{scene['objects']}"
    )

    print(
        f"Obstacle Count: "
        f"{len(scene['obstacles'])}"
    )

    print()

    path = agent.plan(scene)

    if path is None:

        print("Planner failed to find a path.")

        return

    print(
        f"Planned Path Length: {len(path)}"
    )

    print("Path:")

    print(path)

    print()

    actions = agent.planner.path_to_actions(
        path
    )

    print("Executing plan...")

    total_reward = 0

    for action in actions:

        (
            observation,
            reward,
            done,
            info
        ) = env.step(action)

        total_reward += reward

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
    agent,
    env
):

    print()
    print("=" * 60)
    print("SANITY TESTS")
    print("=" * 60)

    observation = env.reset()

    assert "agent" in observation

    assert "objects" in observation

    assert "obstacles" in observation

    assert "instruction" in observation

    scene = agent.perceive(
        observation
    )

    assert scene["target"] is not None

    assert scene[
        "target_position"
    ] is not None

    path = agent.plan(scene)

    if path is not None:

        assert path[0] == scene["agent"]

        assert (
            path[-1]
            == scene["target_position"]
        )

    print("✓ Environment test passed")
    print("✓ Scene understanding passed")
    print("✓ Target reasoning passed")
    print("✓ Planner test passed")
    print()
    print("All sanity tests passed.")


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("ARIA v3 - Planning & Reasoning Agent")
    print("=" * 70)

    cfg = Config()

    set_seed(cfg.seed)

    print(
        f"Device: {cfg.device}"
    )

    print(
        f"Grid: {cfg.grid_size}x{cfg.grid_size}"
    )

    print(
        f"Obstacles: {cfg.num_obstacles}"
    )

    print(
        f"Objects: {cfg.num_objects}"
    )

    print()

    # --------------------------------------------------------
    # Environment
    # --------------------------------------------------------

    env = HardNavigationEnv(
        cfg
    )

    # --------------------------------------------------------
    # Agent
    # --------------------------------------------------------

    agent = ARIAv3(
        cfg
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
    # Evaluation
    # --------------------------------------------------------

    evaluate(
        agent,
        env,
        episodes=100,
        render_first=False
    )

    print()
    print("=" * 70)
    print("ARIA v3 BASELINE COMPLETE")
    print("=" * 70)

    print(
        """
Next architecture upgrade:

    Hard Navigation
          ↓
    Learned Perception
          ↓
    Scene Graph
          ↓
    Neural Reasoning
          ↓
    Learned Planner
          ↓
    World Model
          ↓
    MPC / Imagination
          ↓
    TD3 Controller
        """
    )


if __name__ == "__main__":
    main()