from dataclasses import dataclass, field


@dataclass
class Box:
    hi: str
    ho: dict[int] = field(default_factory=dict)


inner = (
    Box("moin"),
    Box("hallo")
)

arr: list[tuple[Box, Box]] = [
    inner,
    inner
]

arr[0][0].hi = "hi"

print(arr[1][0].hi)

