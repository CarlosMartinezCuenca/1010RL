import os
import argparse

import numpy as np  # noqa: E402
import pygame  # noqa: E402

from PIL import Image

from agents.loading import NAMES, load_agent
from tenten.env import TenTenEnv
from tenten.pieces import PIECES

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

# (R, G, B)
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
PANEL_H = 380  # alto mínimo para que el panel de la derecha quepa con tableros pequeños


def area_height(env):
    return max(env.size * (CELL + GAP), PANEL_H)


def preview(env, action):
    slot, anchor = divmod(action, env.n)
    cells = env.engine.mask_at(env.hand[slot], anchor)
    placed = (env.board | cells).reshape(env.size, env.size)

    lines = np.zeros_like(placed)
    lines[placed.all(axis=1), :] = True
    lines[:, placed.all(axis=0)] = True

    return cells, lines.ravel(), slot


def draw(screen, fonts, env, info, name, action, game_over, paused, speed):
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
        ("Puntos", info["score"]),
        ("Líneas", info["lines"]),
        ("Piezas colocadas", info["steps"]),
    ]:
        screen.blit(normal.render(label, True, MUTED), (x, y))
        screen.blit(normal.render(str(value), True, TEXT), (x + 200, y))
        y += 28

    y += 20
    screen.blit(normal.render("Mano", True, MUTED), (x, y))
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
    state = "PAUSA" if paused else f"velocidad x{speed:g}"
    bottom = top + area_height(env)
    screen.blit(small.render(state, True, TEXT), (x, bottom - 60))
    screen.blit(
        small.render("espacio pausa · → paso · +/- velocidad", True, MUTED),
        (x, bottom - 40),
    )
    screen.blit(
        small.render("R nueva partida · Q salir", True, MUTED), (x, bottom - 20)
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--agent", default="heuristic", help=" | ".join(NAMES)
    )
    parser.add_argument(
        "--model", default="models/ntuple/ntuple_seed0_last.npy", help="pesos del agente"
    )
    parser.add_argument("--search", action="store_true", help="n-tuplas con búsqueda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--size", type=int, default=10, help="lado del tablero")
    parser.add_argument("--pieces", nargs="+", default=list(PIECES), help="piezas posibles")
    parser.add_argument("--hand-size", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument(
        "--gif", default=None, help="en vez de abrir ventana, guarda un GIF"
    )
    parser.add_argument(
        "--gif-moves", type=int, default=40, help="jugadas que salen en el GIF"
    )
    args = parser.parse_args()

    env = TenTenEnv(
        size=args.size,
        piece_names=args.pieces,
        hand_size=args.hand_size,
        max_steps=args.max_steps,
    )
    agent_name = "ntuple_search" if args.agent == "ntuple" and args.search else args.agent
    agent = load_agent(agent_name, env, args.model, seed=args.seed)
    name = NAMES[agent_name]
    seed = args.seed
    obs, info = env.reset(seed=seed)

    width = MARGIN * 3 + env.size * (CELL + GAP) + PANEL_W
    height = MARGIN * 2 + 40 + area_height(env)

    if args.gif:
        os.environ["SDL_VIDEODRIVER"] = "dummy"

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
        print(f"GIF guardado en {args.gif} ({len(images)} fotogramas)")


if __name__ == "__main__":
    main()
