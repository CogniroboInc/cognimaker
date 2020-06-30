import json

from cognimaker.util import DefaultEncoder


def test_encoder_supports_dataclass():
    from dataclasses import dataclass
    from typing import Optional

    @dataclass
    class Person:
        name: str
        age: int
        gender: Optional[str] = None

    bob = Person("Bob", 24, "M")
    alice = Person("Alice", 42)

    d = {"people": [bob, alice]}

    assert json.loads(json.dumps(d, cls=DefaultEncoder)) == {
        "people": [
            {"name": "Bob", "age": 24, "gender": "M"},
            {"name": "Alice", "age": 42, "gender": None},
        ]
    }
