def standalone_function(x: int) -> int:
    return x + 1


async def async_function(x: int) -> int:
    return x + 1


class MyClass:
    def method_one(self) -> None:
        pass

    def method_two(self) -> str:
        return "hello"


def duplicate_name() -> None:
    pass


def duplicate_name() -> int:  # noqa: F811
    return 42
