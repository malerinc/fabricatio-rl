from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fabricatio_rl.core_state import State
    from fabricatio_rl.interface import FabricatioRL


class Control:
    def __init__(self, name):
        self.name = name

    def play_game(self, env: 'FabricatioRL', initial_state=None) -> 'State':
        raise NotImplementedError
