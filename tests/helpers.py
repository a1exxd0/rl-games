from tienlen.engine import GameState, Play, Rules, parse_cards


def play(text: str) -> Play:
    return Play.of(parse_cards(text))


def table(*hands: str, to_act: int = 0, rules: Rules | None = None, **fields) -> GameState:
    rules = rules or Rules(n_players=len(hands), first_lead_includes_lowest=False)
    return GameState(
        rules=rules,
        seed=None,
        hands=tuple(parse_cards(hand) for hand in hands),
        undealt=(),
        first_leader=to_act,
        to_act=to_act,
        **fields,
    )
