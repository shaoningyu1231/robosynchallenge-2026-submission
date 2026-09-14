"""scripts/eval_policy.py seeds the global RNGs before it builds the environment.

Randomizers draw state at construction (EmbodiChain's visual-material palette),
so seeding only per episode left each evaluation process with its own colours.
Read from the source, so the check needs neither the simulator nor torch.

    python -m unittest discover -s tests -p test_eval_policy_seeding.py -v
"""

import ast
import unittest
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1] / "scripts" / "eval_policy.py"


def calls_in_order(function: ast.FunctionDef) -> list[tuple[int, str]]:
    found = []
    for node in ast.walk(function):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            found.append((node.lineno, node.func.id))
    return sorted(found)


class EnvironmentCreationSeed(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tree = ast.parse(SOURCE.read_text())
        cls.functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}

    def test_main_seeds_before_building_the_environment(self):
        names = [name for _, name in calls_in_order(self.functions["main"])]
        self.assertIn("seed_global_rngs", names)
        self.assertLess(names.index("seed_global_rngs"), names.index("make_env_from_configs"))

    def test_episode_reset_still_seeds_every_episode(self):
        names = [name for _, name in calls_in_order(self.functions["prepare_episode_reset"])]
        self.assertIn("seed_global_rngs", names)

    def test_metrics_record_it(self):
        self.assertIn('"env_creation_seed": seed', SOURCE.read_text())


if __name__ == "__main__":
    unittest.main()
