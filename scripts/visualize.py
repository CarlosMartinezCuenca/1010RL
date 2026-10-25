"""Watches an agent play in a window (or saves a GIF).

uv run python scripts/visualize.py --agent ntuple --search
uv run python scripts/visualize.py --agent dqn --model models/5x5_small_L/dqn_seed0_best.pt --size 5 --pieces small_L --hand-size 1 --max-steps 200
uv run python scripts/visualize.py --agent heuristic --gif game.gif
Keys: space pause, right arrow step, +/- speed, R new game, Q quit.
"""

import argparse
import os

# Must be set before importing pygame
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import numpy as np  # noqa: E402
import pygame  # noqa: E402
from PIL import Image  # noqa: E402

from tenten.agents.registry import NAMES, load_agent  # noqa: E402
from tenten.game.env import TenTenEnv  # noqa: E402
from tenten.game.pieces import PIECES  # noqa: E402

# Colours as (R, G, B)
BACKGROUND = (246, 246, 243)
EMPTY = (226, 226, 220)
FILLED = (42, 120, 214)
GHOST = (235, 104, 52)
FLASH = (237, 161, 0)
TEXT = (31, 35, 40)
MUTED = (107, 114, 128)

CELL, GAP, MARGIN = 44, 3, 28
MINI = 16
PANEL_W = 320
PANEL_H = 380  # minimum height so the side panel fits next to small boards


def area_height(env):
    """Height of the drawing area: the board or the side panel, whichever is taller."""
    return max(env.size * (CELL + GAP), PANEL_H)


def preview(env, action):
    """Cells of the move about to be played, lines it would clear and its hand slot."""
    slot, anchor = divmod(action, env.n)
    cells = env.engine.mask_at(env.hand[slot], anchor)
    placed = (env.board | cells).reshape(env.size, env.size)

    lines = np.zeros_like(placed)
    lines[placed.all(axis=1), :] = True
    lines[:, placed.all(axis=0)] = True

    return cells, lines.ravel(), slot


def draw(screen, fonts, env, info, name, action, game_over, paused, speed):
    """Draws the board (next move in orange, lines to clear in yellow) and the side panel."""
    big, normal, small = fonts
    screen.fill(BACKGROUND)

    board_px = env.size * (CELL + GAP)
    top = MARGIN + 40

    ghost = flash = None
    if action is not None:
        ghost, flash, _ = preview(env, action)

    for i in range(env.n):
        row, col = divmod(i, env.size)
        color = FILLED if env.board[i] else EMPTY

        if ghost is not None and ghost[i]:
            color = GHOST

        if flash is not None and flash[i] and (env.board[i] or ghost[i]):
            color = FLASH

        rect = (MARGIN + col * (CELL + GAP), top + row * (CELL + GAP), CELL, CELL)
        pygame.draw.rect(screen, color, rect, border_radius=6)

    screen.blit(big.render("1010!", True, TEXT), (MARGIN, MARGIN))
    if game_over:
        text = big.render("GAME OVER", True, TEXT)
        screen.blit(
            text, text.get_rect(center=(MARGIN + board_px // 2, top + board_px // 2))
        )

    x = MARGIN * 2 + board_px
    y = MARGIN

    screen.blit(big.render(name, True, TEXT), (x, y))
    y += 50

    for label, value in [
        ("Score", info["score"]),
        ("Lines", info["lines"]),
        ("Pieces placed", info["steps"]),
    ]:
        screen.blit(normal.render(label, True, MUTED), (x, y))
        screen.blit(normal.render(str(value), True, TEXT), (x + 200, y))
        y += 28

    y += 20
    screen.blit(normal.render("Hand", True, MUTED), (x, y))
    y += 30
    chosen_slot = preview(env, action)[2] if action is not None else None
    for slot, piece in enumerate(env.hand):
        if piece is not None:
            shape = env.engine.pieces[piece]
            color = GHOST if slot == chosen_slot else FILLED
            for r in range(shape.shape[0]):
                for c in range(shape.shape[1]):
                    if shape[r, c]:
                        rect = (x + c * (MINI + 2), y + r * (MINI + 2), MINI, MINI)
                        pygame.draw.rect(screen, color, rect, border_radius=3)
        x += 5 * (MINI + 2) + 12

    x = MARGIN * 2 + board_px
    state = "PAUSED" if paused else f"speed x{speed:g}"
    bottom = top + area_height(env)
    screen.blit(small.render(state, True, TEXT), (x, bottom - 60))
    screen.blit(
        small.render("space pause · → step · +/- speed", True, MUTED),
        (x, bottom - 40),
    )
    screen.blit(small.render("R new game · Q quit", True, MUTED), (x, bottom - 20))


def main():
    """Each tick alternates: choose a move (shown as a preview), then play it."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", default="heuristic", help=" | ".join(NAMES))
    parser.add_argument(
        "--model",
        default="models/10x10_all/ntuple_seed0_best.npy",
        help="agent weights",
    )
    parser.add_argument(
        "--search", action="store_true", help="n-tuples with one-step search"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--size", type=int, default=10, help="board side")
    parser.add_argument(
        "--pieces", nargs="+", default=list(PIECES), help="pieces that can be dealt"
    )
    parser.add_argument("--hand-size", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument(
        "--gif", default=None, help="save a GIF instead of opening a window"
    )
    parser.add_argument(
        "--gif-moves", type=int, default=40, help="moves shown in the GIF"
    )
    args = parser.parse_args()

    env = TenTenEnv(
        size=args.size,
        piece_names=args.pieces,
        hand_size=args.hand_size,
        max_steps=args.max_steps,
    )
    agent_name = (
        "ntuple_search" if args.agent == "ntuple" and args.search else args.agent
    )
    agent = load_agent(agent_name, env, args.model, seed=args.seed)
    name = NAMES[agent_name]
    seed = args.seed
    obs, info = env.reset(seed=seed)

    width = MARGIN * 3 + env.size * (CELL + GAP) + PANEL_W
    height = MARGIN * 2 + 40 + area_height(env)

    if args.gif:
        os.environ["SDL_VIDEODRIVER"] = "dummy"  # render off-screen

    pygame.display.init()
    pygame.font.init()

    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption(f"1010! · {name}")

    font = "dejavusans,arial"
    fonts = (
        pygame.font.SysFont(font, 24, bold=True),
        pygame.font.SysFont(font, 17),
        pygame.font.SysFont(font, 13),
    )
    clock = pygame.time.Clock()

    action = None
    game_over = False
    paused = False
    speed = args.speed
    timer = 0
    frames = []

    running = True
    while running:
        advance = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_RIGHT:
                    advance = True
                elif event.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                    speed = min(speed * 2, 64)
                elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    speed = max(speed / 2, 0.125)
                elif event.key == pygame.K_r:
                    seed += 1
                    obs, info = env.reset(seed=seed)
                    action, game_over = None, False

        timer += clock.tick(60)
        wait = 2500 if game_over else 400
        if args.gif or (not paused and timer >= wait / speed):
            advance = True

        if advance:
            timer = 0
            if game_over:
                seed += 1
                obs, info = env.reset(seed=seed)
                action, game_over = None, False

            elif action is None:
                action = agent.act(env, info)

            else:
                obs, reward, terminated, truncated, info = env.step(action)
                action = None
                game_over = terminated or truncated

        draw(screen, fonts, env, info, name, action, game_over, paused, speed)
        pygame.display.flip()

        if args.gif:
            frames.append(pygame.image.tobytes(screen, "RGB"))
            if info["steps"] >= args.gif_moves or game_over:
                running = False

    pygame.quit()

    if args.gif:
        images = [Image.frombytes("RGB", (width, height), f) for f in frames]
        durations = [400] * len(images)
        durations[-1] = 2000

        images[0].save(
            args.gif,
            save_all=True,
            append_images=images[1:],
            duration=durations,
            loop=0,
        )
        print(f"GIF saved to {args.gif} ({len(images)} frames)")


if __name__ == "__main__":
    main()
