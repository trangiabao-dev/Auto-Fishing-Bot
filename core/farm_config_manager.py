"""
Farm Configuration Manager.
Handles persistent storage of fruit filtering rules and harvest settings (config/farm_config.json).
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional


@dataclass
class FruitRule:
    name: str
    max_kg: float
    enabled: bool = True
    template_path: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> FruitRule:
        return cls(
            name=d.get("name", ""),
            max_kg=float(d.get("max_kg", 0.0)),
            enabled=bool(d.get("enabled", True)),
            template_path=d.get("template_path", "")
        )


class FarmConfigManager:
    def __init__(self, config_path: Optional[str] = None):
        if config_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            config_path = os.path.join(base_dir, "config", "farm_config.json")

        self.config_path = os.path.abspath(config_path)
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        self.rules: Dict[str, FruitRule] = {}
        self.auto_sell: bool = True
        self.shift_delay_s: float = 0.28
        self.load()

    def load(self) -> None:
        """Loads configuration from JSON file or initializes defaults."""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.auto_sell = bool(data.get("auto_sell", True))
                self.shift_delay_s = float(data.get("shift_delay_s", 0.28))
                self.rules = {}
                for r in data.get("fruits", []):
                    rule = FruitRule.from_dict(r)
                    self.rules[rule.name] = rule
                if self.rules:
                    return
            except Exception as e:
                print(f"[FarmConfigManager] Error reading config: {e}. Reverting to defaults.")

        # Default fallback
        self._set_defaults()
        self.save()

    def _set_defaults(self) -> None:
        self.auto_sell = True
        self.shift_delay_s = 0.28
        self.rules = {
            "Hạt dẻ": FruitRule(name="Hạt dẻ", max_kg=3.0, enabled=True, template_path="assets/farm/thu_hoach/B6.jpg"),
            "Sầu riêng": FruitRule(name="Sầu riêng", max_kg=18.0, enabled=True, template_path="assets/farm/thu_hoach/item_sau_rieng.png"),
            "Đu đủ": FruitRule(name="Đu đủ", max_kg=3.0, enabled=True, template_path="assets/farm/thu_hoach/item_du_du.png"),
            "Xoài": FruitRule(name="Xoài", max_kg=0.0, enabled=True, template_path="assets/farm/thu_hoach/item_xoai.png"),
            "Dừa": FruitRule(name="Dừa", max_kg=0.0, enabled=True, template_path="assets/farm/thu_hoach/item_dua.png"),
            "Cây đậu": FruitRule(name="Cây đậu", max_kg=0.0, enabled=True, template_path="assets/farm/thu_hoach/item_cay_dau.png"),
        }

    def save(self) -> None:
        """Saves current configuration to JSON file."""
        data = {
            "auto_sell": self.auto_sell,
            "shift_delay_s": self.shift_delay_s,
            "fruits": [r.to_dict() for r in self.rules.values()]
        }
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def get_rule(self, name: str) -> Optional[FruitRule]:
        return self.rules.get(name)

    def set_rule(self, name: str, max_kg: float, enabled: bool = True, template_path: str = "") -> None:
        self.rules[name] = FruitRule(name=name, max_kg=float(max_kg), enabled=enabled, template_path=template_path)
        self.save()

    def remove_rule(self, name: str) -> bool:
        if name in self.rules:
            del self.rules[name]
            self.save()
            return True
        return False

    def list_rules(self) -> List[FruitRule]:
        return list(self.rules.values())

    def get_enabled_rules(self) -> List[FruitRule]:
        """Returns all fruit rules enabled/ticked by the user."""
        return [r for r in self.rules.values() if r.enabled]

    def get_category_a_rules(self) -> List[FruitRule]:
        """Returns enabled rules with a strict kg threshold (max_kg > 0)."""
        return [r for r in self.rules.values() if r.enabled and r.max_kg > 0.0]

    def get_category_b_rules(self) -> List[FruitRule]:
        """Returns enabled rules without kg limits (max_kg <= 0), intended for harvest-all."""
        return [r for r in self.rules.values() if r.enabled and r.max_kg <= 0.0]

    def get_active_rule(self) -> Optional[FruitRule]:
        """Returns the first enabled fruit rule to harvest, if any."""
        for r in self.rules.values():
            if r.enabled:
                return r
        return None
