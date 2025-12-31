import math
import random
import sys
from dataclasses import dataclass, field

import pygame

# ============================================================
# ISEKAI-STYLE PARTY MICRO-RPG (NO EXTERNAL ASSETS)
# Fixes:
# - No isolated rooms: reachability flood-fill & placement rules
# - Wider corridors (2 tiles) to prevent "stuck" feeling
# - Companion AI only attacks enemies in reachable region + line-of-sight
# - Companion unstuck failsafe (warps near fighter if stuck)
# ============================================================

W, H = 1000, 600
FPS = 60
UI_H = 110

ARENA = pygame.Rect(18, 18, W - 36, H - 36 - UI_H)
UI = pygame.Rect(18, H - 18 - UI_H, W - 36, UI_H)

TILE = 20
GW = ARENA.w // TILE
GH = ARENA.h // TILE

# Colors
BG = (15, 15, 20)
PANEL = (30, 30, 40)
PANEL2 = (24, 24, 32)
BORDER = (95, 95, 120)

WHITE = (240, 240, 245)
GRAY = (170, 170, 185)

RED = (220, 70, 70)
GREEN = (70, 220, 120)
BLUE = (80, 150, 240)
YELLOW = (235, 220, 120)
PURPLE = (170, 120, 230)
ORANGE = (240, 160, 80)
CYAN = (110, 235, 235)

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def length(x, y):
    return math.hypot(x, y)

def normalize(x, y):
    d = length(x, y)
    if d == 0:
        return 0.0, 0.0
    return x / d, y / d

def draw_text(surf, font, text, x, y, color=WHITE):
    img = font.render(text, True, color)
    surf.blit(img, (x, y))
    return img.get_width(), img.get_height()

def draw_center_text(surf, font, text, y, color=WHITE):
    img = font.render(text, True, color)
    surf.blit(img, ((surf.get_width() - img.get_width()) // 2, y))
    return img.get_width(), img.get_height()

def bar(surf, x, y, w, h, ratio, fg, bg=(60, 60, 70), border=(18, 18, 22)):
    ratio = clamp(ratio, 0, 1)
    r = 7
    pygame.draw.rect(surf, bg, pygame.Rect(x, y, w, h), border_radius=r)
    pygame.draw.rect(surf, fg, pygame.Rect(x, y, int(w * ratio), h), border_radius=r)
    pygame.draw.rect(surf, border, pygame.Rect(x, y, w, h), 2, border_radius=r)

def grid_to_world(gx, gy):
    return ARENA.x + gx * TILE + TILE / 2, ARENA.y + gy * TILE + TILE / 2

def world_to_grid(x, y):
    gx = int((x - ARENA.x) // TILE)
    gy = int((y - ARENA.y) // TILE)
    return gx, gy

# ---------- Particles / Floaters ----------
@dataclass
class Floater:
    text: str
    x: float
    y: float
    vy: float
    t: float
    color: tuple

@dataclass
class Particle:
    x: float
    y: float
    vx: float
    vy: float
    r: float
    t: float
    color: tuple

# ---------- Items ----------
@dataclass
class Item:
    name: str
    slot: str  # Weapon / Armor / Trinket
    str_mod: int = 0
    int_mod: int = 0
    vit_mod: int = 0
    value: int = 10

def item_line(it: Item):
    mods = []
    if it.str_mod: mods.append(f"STR {it.str_mod:+d}")
    if it.int_mod: mods.append(f"INT {it.int_mod:+d}")
    if it.vit_mod: mods.append(f"VIT {it.vit_mod:+d}")
    m = ", ".join(mods) if mods else "no mods"
    return f"{it.name} [{it.slot}] ({m})"

def roll_item(floor: int) -> Item:
    slot = random.choice(["Weapon", "Armor", "Trinket"])
    tier = 1 + floor // 2
    budget = 1 + tier + random.randint(0, 2)

    s = i = v = 0
    for _ in range(budget):
        pick = random.choice(["STR", "INT", "VIT"])
        if pick == "STR": s += 1
        elif pick == "INT": i += 1
        else: v += 1

    if slot == "Weapon":
        base = random.choice(["Iron Sword", "Steel Blade", "Rune Edge", "Knight Saber", "Oath Cutter"])
    elif slot == "Armor":
        base = random.choice(["Leather Vest", "Chainmail", "Ward Plate", "Traveler Coat", "Guard Cuirass"])
    else:
        base = random.choice(["Charm of Sparks", "Lucky Coin", "Glass Sigil", "Old Locket", "Moon Token"])

    value = 18 + floor * 6 + budget * 4
    return Item(name=f"{base} +{tier}", slot=slot, str_mod=s, int_mod=i, vit_mod=v, value=value)

# ---------- Entities ----------
@dataclass
class Entity:
    name: str
    x: float
    y: float
    r: int
    hp: int
    max_hp: int
    speed: float
    team: str
    color: tuple
    iframes: float = 0.0
    alive: bool = True

    # for “unstuck” detection
    _stuck_t: float = 0.0
    _last_x: float = 0.0
    _last_y: float = 0.0

@dataclass
class Party(Entity):
    role: str = "Fighter"
    level: int = 1
    xp: int = 0
    xp_next: int = 40
    gold: int = 0
    potions: int = 2

    base_str: int = 3
    base_int: int = 2
    base_vit: int = 3

    weapon: Item | None = None
    armor: Item | None = None
    trinket: Item | None = None

    slash_cd: float = 0.0
    dash_cd: float = 0.0
    dash_t: float = 0.0
    bolt_cd: float = 0.0
    heal_cd: float = 0.0

    face_x: float = 1.0
    face_y: float = 0.0

    def total_str(self):
        m = self.base_str
        for it in (self.weapon, self.armor, self.trinket):
            if it: m += it.str_mod
        return m

    def total_int(self):
        m = self.base_int
        for it in (self.weapon, self.armor, self.trinket):
            if it: m += it.int_mod
        return m

    def total_vit(self):
        m = self.base_vit
        for it in (self.weapon, self.armor, self.trinket):
            if it: m += it.vit_mod
        return m

@dataclass
class Enemy(Entity):
    kind: str = "Goblin"
    dps: float = 16.0
    bounty_xp: int = 10
    bounty_gold: int = 6
    _rewarded: bool = False

@dataclass
class Projectile:
    x: float
    y: float
    vx: float
    vy: float
    r: int
    dmg: int
    t: float
    color: tuple
    owner_team: str

# ---------- Dialog ----------
@dataclass
class Dialog:
    lines: list[str] = field(default_factory=list)
    idx: int = 0
    open: bool = False

    def start(self, lines):
        self.lines = lines
        self.idx = 0
        self.open = True

    def current(self):
        return self.lines[self.idx] if self.open else None

    def next(self):
        if not self.open:
            return
        self.idx += 1
        if self.idx >= len(self.lines):
            self.open = False

# ---------- Combat tuning ----------
IFRAMES_PARTY = 0.55
IFRAMES_ENEMY = 0.12

SLASH_RANGE = 72
SLASH_ARC = math.radians(120)
SLASH_CD = 0.33

DASH_SPEED = 660
DASH_TIME = 0.12
DASH_CD = 1.1

BOLT_SPEED = 560
BOLT_LIFE = 1.2
BOLT_CD = 0.6

HEAL_RANGE = 170
HEAL_CD = 2.3

FLOORS_TO_CLEAR = 5

# ---------- Dungeon ----------
@dataclass
class Dungeon:
    floor: int = 1
    tiles: list[list[int]] = field(default_factory=list)  # 1 wall, 0 floor
    rooms: list[pygame.Rect] = field(default_factory=list)  # grid coords
    exit_cell: tuple[int, int] = (1, 1)
    chest_cell: tuple[int, int] | None = None
    cleared: bool = False
    reachable: list[list[bool]] = field(default_factory=list)  # flood fill map

def in_bounds(gx, gy):
    return 0 <= gx < GW and 0 <= gy < GH

def tile_wall(dun: Dungeon, gx, gy):
    if not in_bounds(gx, gy):
        return True
    return dun.tiles[gy][gx] == 1

def carve_room(tiles, rx, ry, rw, rh):
    for y in range(ry, ry + rh):
        for x in range(rx, rx + rw):
            if 0 <= x < GW and 0 <= y < GH:
                tiles[y][x] = 0

def carve_hall_wide(tiles, x1, y1, x2, y2, width=2):
    # L-shaped corridor but "wide": carve 2-tile thickness
    def carve_cell(cx, cy):
        for oy in range(-(width//2), (width - width//2)):
            for ox in range(-(width//2), (width - width//2)):
                xx, yy = cx + ox, cy + oy
                if 0 <= xx < GW and 0 <= yy < GH:
                    tiles[yy][xx] = 0

    x, y = x1, y1
    while x != x2:
        carve_cell(x, y)
        x += 1 if x2 > x else -1
    while y != y2:
        carve_cell(x, y)
        y += 1 if y2 > y else -1
    carve_cell(x, y)

def flood_reachable(tiles, start):
    reachable = [[False for _ in range(GW)] for _ in range(GH)]
    sx, sy = start
    if not in_bounds(sx, sy) or tiles[sy][sx] == 1:
        return reachable

    q = [(sx, sy)]
    reachable[sy][sx] = True
    head = 0
    while head < len(q):
        x, y = q[head]
        head += 1
        for ox, oy in ((1,0),(-1,0),(0,1),(0,-1)):
            nx, ny = x + ox, y + oy
            if in_bounds(nx, ny) and not reachable[ny][nx] and tiles[ny][nx] == 0:
                reachable[ny][nx] = True
                q.append((nx, ny))
    return reachable

def random_reachable_cell(dun: Dungeon, avoid=None, min_dist=0):
    # pick a random reachable floor cell
    tries = 2000
    for _ in range(tries):
        gx = random.randint(1, GW - 2)
        gy = random.randint(1, GH - 2)
        if dun.tiles[gy][gx] != 0:
            continue
        if not dun.reachable[gy][gx]:
            continue
        if avoid:
            ax, ay = avoid
            if abs(gx - ax) + abs(gy - ay) < min_dist:
                continue
        return (gx, gy)
    # fallback: scan
    for gy in range(GH):
        for gx in range(GW):
            if dun.tiles[gy][gx] == 0 and dun.reachable[gy][gx]:
                return (gx, gy)
    return (1, 1)

def generate_dungeon(floor: int) -> Dungeon:
    tiles = [[1 for _ in range(GW)] for _ in range(GH)]
    rooms: list[pygame.Rect] = []

    n_rooms = 7 + floor
    for _ in range(n_rooms):
        rw = random.randint(6, 12)
        rh = random.randint(5, 10)
        rx = random.randint(1, GW - rw - 2)
        ry = random.randint(1, GH - rh - 2)
        r = pygame.Rect(rx, ry, rw, rh)
        if any(r.colliderect(o.inflate(2, 2)) for o in rooms):
            continue
        rooms.append(r)
        carve_room(tiles, r.x, r.y, r.w, r.h)

    if len(rooms) < 3:
        r = pygame.Rect(GW//2 - 10, GH//2 - 7, 20, 14)
        rooms = [r]
        carve_room(tiles, r.x, r.y, r.w, r.h)

    centers = [(r.centerx, r.centery) for r in rooms]
    # connect in a chain
    for i in range(1, len(centers)):
        x1, y1 = centers[i-1]
        x2, y2 = centers[i]
        carve_hall_wide(tiles, x1, y1, x2, y2, width=2)

    # spawn is center of first room
    start = (rooms[0].centerx, rooms[0].centery)
    reachable = flood_reachable(tiles, start)

    dun = Dungeon(floor=floor, tiles=tiles, rooms=rooms, reachable=reachable)

    # place exit & chest ONLY on reachable tiles
    exit_cell = random_reachable_cell(dun, avoid=start, min_dist=16)
    dun.exit_cell = exit_cell
    tiles[exit_cell[1]][exit_cell[0]] = 0

    if random.random() < 0.85:
        chest_cell = random_reachable_cell(dun, avoid=start, min_dist=10)
        dun.chest_cell = chest_cell
        tiles[chest_cell[1]][chest_cell[0]] = 0
    else:
        dun.chest_cell = None

    return dun

def collides_walls(dun: Dungeon, x, y, r):
    left = x - r
    right = x + r
    top = y - r
    bot = y + r
    for px, py in ((left, top), (right, top), (left, bot), (right, bot)):
        gx, gy = world_to_grid(px, py)
        if tile_wall(dun, gx, gy):
            return True
    return False

def move_with_collision(dun: Dungeon, ent: Entity, dx, dy):
    # step-based collision with sliding; smaller steps reduce snagging
    steps = max(1, int(max(abs(dx), abs(dy)) // 3) + 1)
    sx = dx / steps
    sy = dy / steps
    for _ in range(steps):
        nx = ent.x + sx
        if not collides_walls(dun, nx, ent.y, ent.r):
            ent.x = nx
        ny = ent.y + sy
        if not collides_walls(dun, ent.x, ny, ent.r):
            ent.y = ny

# ---------- LOS (line of sight) ----------
def bresenham_line(x0, y0, x1, y1):
    # yields grid points from (x0,y0) to (x1,y1)
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    x, y = x0, y0
    while True:
        yield x, y
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x += sx
        if e2 <= dx:
            err += dx
            y += sy

def has_line_of_sight(dun: Dungeon, ax, ay, bx, by):
    # check wall blocking between two world points
    x0, y0 = world_to_grid(ax, ay)
    x1, y1 = world_to_grid(bx, by)
    for gx, gy in bresenham_line(x0, y0, x1, y1):
        if not in_bounds(gx, gy):
            return False
        if dun.tiles[gy][gx] == 1:
            return False
    return True

# ---------- Spawning enemies only on reachable tiles ----------
def spawn_enemies(dun: Dungeon, floor: int) -> list[Enemy]:
    enemies: list[Enemy] = []
    count = 6 + floor * 2

    start = (dun.rooms[0].centerx, dun.rooms[0].centery)

    def roll_kind():
        r = random.random()
        if floor >= 3 and r > 0.78:
            return "Bat"
        if floor >= 2 and r > 0.55:
            return "Slime"
        return "Goblin"

    for _ in range(count):
        kind = roll_kind()
        if kind == "Goblin":
            rr, hp, spd, dps, xp, gold, col = 14, 60 + floor*8, 125 + floor*6, 18, 10, 7, PURPLE
        elif kind == "Slime":
            rr, hp, spd, dps, xp, gold, col = 16, 95 + floor*12, 98 + floor*4, 22, 14, 9, (120, 210, 120)
        else:
            rr, hp, spd, dps, xp, gold, col = 12, 55 + floor*7, 170 + floor*7, 16, 12, 8, (190, 190, 240)

        cell = random_reachable_cell(dun, avoid=start, min_dist=14)
        x, y = grid_to_world(cell[0], cell[1])
        enemies.append(Enemy(
            name=kind, kind=kind, x=x, y=y, r=rr,
            hp=hp, max_hp=hp, speed=spd,
            team="enemy", color=col, dps=dps,
            bounty_xp=xp, bounty_gold=gold
        ))

    if floor == FLOORS_TO_CLEAR:
        cell = random_reachable_cell(dun, avoid=start, min_dist=18)
        x, y = grid_to_world(cell[0], cell[1])
        enemies.append(Enemy(
            name="Ogre Captain", kind="Boss",
            x=x, y=y, r=26,
            hp=320 + floor*60, max_hp=320 + floor*60,
            speed=112, team="enemy", color=(210, 140, 160),
            dps=30, bounty_xp=70, bounty_gold=60
        ))
    return enemies

def nearest_enemy_visible(dun: Dungeon, enemies, x, y, max_range=420):
    best = None
    best_d = 10**18
    for e in enemies:
        if not e.alive:
            continue
        dx = e.x - x
        dy = e.y - y
        d2 = dx*dx + dy*dy
        if d2 > max_range*max_range:
            continue
        # must be reachable region + LOS
        gx, gy = world_to_grid(e.x, e.y)
        if not in_bounds(gx, gy) or not dun.reachable[gy][gx]:
            continue
        if not has_line_of_sight(dun, x, y, e.x, e.y):
            continue
        if d2 < best_d:
            best_d = d2
            best = e
    return best

def nearest_ally(party, x, y, pred):
    best = None
    best_d = 10**18
    for a in party:
        if not a.alive:
            continue
        if not pred(a):
            continue
        d2 = (a.x - x)**2 + (a.y - y)**2
        if d2 < best_d:
            best_d = d2
            best = a
    return best

def cone_hit(att: Party, e: Enemy):
    dx = e.x - att.x
    dy = e.y - att.y
    dist = length(dx, dy)
    if dist > SLASH_RANGE + e.r:
        return False
    nx, ny = normalize(dx, dy)
    fx, fy = normalize(att.face_x, att.face_y)
    dot = clamp(nx*fx + ny*fy, -1, 1)
    ang = math.acos(dot)
    return ang <= (SLASH_ARC / 2)

def xp_next_for(level: int):
    return int(40 * (1.35 ** (level - 1))) + 10 * (level - 1)

def apply_vit_maxhp(base: int, vit: int):
    return base + vit * 18

def recompute_maxhp(party: list[Party], fighter: Party):
    for p in party:
        base = 120 if p.role == "Fighter" else 85 if p.role == "Mage" else 92
        vit = fighter.total_vit() if p.role == "Fighter" else p.base_vit
        ratio = p.hp / p.max_hp if p.max_hp else 1
        p.max_hp = apply_vit_maxhp(base, vit)
        p.hp = max(1, int(clamp(ratio, 0, 1) * p.max_hp))

def level_up(m: Party, floaters):
    m.level += 1
    m.xp_next = xp_next_for(m.level)
    m.hp = min(m.max_hp, m.hp + 25)
    floaters.append(Floater(f"{m.role} LEVEL {m.level}!", m.x, m.y - 28, -40, 1.0, YELLOW))

def grant_xp_gold(party, xp, gold, fighter, floaters, x, y):
    alive_count = sum(1 for p in party if p.alive)
    per = max(1, xp // max(1, alive_count))
    for p in party:
        if not p.alive:
            continue
        p.xp += per
        while p.xp >= p.xp_next:
            p.xp -= p.xp_next
            level_up(p, floaters)
    fighter.gold += gold
    floaters.append(Floater(f"+{xp}xp  +{gold}g", x, y, -30, 1.0, GRAY))

def deal_damage(target: Entity, dmg: int, floaters, particles, shake_ref, color=WHITE):
    if not target.alive:
        return False
    if target.iframes > 0:
        return False
    target.hp -= dmg
    target.iframes = IFRAMES_ENEMY if target.team == "enemy" else IFRAMES_PARTY
    floaters.append(Floater(f"-{dmg}", target.x, target.y - 18, -55, 0.8, RED))
    shake_ref[0] = max(shake_ref[0], 7)
    for _ in range(10):
        a = random.uniform(0, math.tau)
        sp = random.uniform(70, 200)
        particles.append(Particle(target.x, target.y, math.cos(a)*sp, math.sin(a)*sp,
                                  random.uniform(1.4, 3.0), random.uniform(0.2, 0.5), color))
    if target.hp <= 0:
        target.alive = False
    return True

def update_stuck(ent: Entity, dt: float):
    # track movement; if barely moved, accumulate stuck time
    if ent._last_x == 0 and ent._last_y == 0:
        ent._last_x, ent._last_y = ent.x, ent.y
        return
    moved = length(ent.x - ent._last_x, ent.y - ent._last_y)
    ent._last_x, ent._last_y = ent.x, ent.y
    if moved < 0.5:
        ent._stuck_t += dt
    else:
        ent._stuck_t = max(0.0, ent._stuck_t - dt * 2)

def warp_near(ent: Entity, dun: Dungeon, near_x: float, near_y: float):
    # pick a reachable cell near fighter
    fx, fy = world_to_grid(near_x, near_y)
    best = None
    for radius in range(1, 8):
        for _ in range(60):
            gx = clamp(fx + random.randint(-radius, radius), 1, GW-2)
            gy = clamp(fy + random.randint(-radius, radius), 1, GH-2)
            if dun.tiles[gy][gx] == 0 and dun.reachable[gy][gx]:
                wx, wy = grid_to_world(gx, gy)
                if not collides_walls(dun, wx, wy, ent.r):
                    best = (wx, wy)
                    break
        if best:
            break
    if best:
        ent.x, ent.y = best

# ============================================================
# MAIN
# ============================================================
def main():
    pygame.init()
    pygame.display.set_caption("Isekai-Style Micro RPG (No External Assets)")
    screen = pygame.display.set_mode((W, H))
    clock = pygame.time.Clock()

    font = pygame.font.SysFont(None, 22)
    mid = pygame.font.SysFont(None, 30)
    big = pygame.font.SysFont(None, 52)

    dialog = Dialog()
    floaters: list[Floater] = []
    particles: list[Particle] = []
    projectiles: list[Projectile] = []
    shake = [0.0]

    fighter = Party(
        name="Fighter", role="Fighter",
        x=ARENA.centerx, y=ARENA.centery,
        r=16, hp=120, max_hp=120,
        speed=260, team="party", color=BLUE,
        base_str=4, base_int=2, base_vit=4,
        gold=0, potions=2
    )
    mage = Party(
        name="Mage", role="Mage",
        x=ARENA.centerx + 26, y=ARENA.centery + 18,
        r=14, hp=85, max_hp=85,
        speed=250, team="party", color=ORANGE,
        base_str=2, base_int=5, base_vit=3
    )
    healer = Party(
        name="Healer", role="Healer",
        x=ARENA.centerx + 42, y=ARENA.centery - 10,
        r=14, hp=92, max_hp=92,
        speed=245, team="party", color=CYAN,
        base_str=2, base_int=3, base_vit=4
    )
    party = [fighter, mage, healer]

    inventory: list[Item] = []

    state = "title"  # title, town, shop, character, dungeon, win, lose
    dun = generate_dungeon(1)
    enemies: list[Enemy] = []
    chest_opened = False

    def compute_fighter_slash_damage():
        return 18 + fighter.total_str() * 3 + fighter.total_int()

    def compute_mage_bolt_damage():
        return 14 + mage.base_int * 3 + mage.level

    def compute_heal_amount():
        return 22 + healer.base_int * 2 + healer.level

    def reset_to_town(msg=True):
        nonlocal state
        state = "town"
        dialog.open = False
        projectiles.clear()
        particles.clear()
        floaters.clear()
        shake[0] = 0.0
        for p in party:
            p.alive = True
            p.iframes = 0.0
            p.hp = min(p.max_hp, p.hp + int(p.max_hp * 0.35))
        if msg:
            floaters.append(Floater("Back in town…", W/2, ARENA.y + 28, -10, 1.1, GRAY))

    def start_dungeon(floor: int):
        nonlocal dun, enemies, chest_opened, state
        dun = generate_dungeon(floor)
        enemies = spawn_enemies(dun, floor)
        chest_opened = False

        # spawn in first room center
        sx, sy = grid_to_world(dun.rooms[0].centerx, dun.rooms[0].centery)
        fighter.x, fighter.y = sx, sy
        mage.x, mage.y = sx + 24, sy + 18
        healer.x, healer.y = sx + 40, sy - 10

        for p in party:
            p.alive = True
            p.iframes = 0.0
            p.slash_cd = p.dash_cd = p.dash_t = p.bolt_cd = p.heal_cd = 0.0
            p._stuck_t = 0.0
            p._last_x = p._last_y = 0.0

        projectiles.clear()
        particles.clear()
        floaters.clear()
        shake[0] = 0.0

        state = "dungeon"
        floaters.append(Floater(f"Floor {floor}", W/2, ARENA.y + 28, -10, 1.0, YELLOW))

    def open_chest():
        nonlocal chest_opened
        if chest_opened:
            return
        chest_opened = True
        it = roll_item(dun.floor)
        inventory.append(it)
        gain = 10 + dun.floor * 6
        fighter.gold += gain
        floaters.append(Floater("CHEST!", fighter.x, fighter.y - 30, -35, 1.0, YELLOW))
        floaters.append(Floater(f"+{gain}g", fighter.x, fighter.y - 12, -35, 1.0, GRAY))
        dialog.start([
            "You open the chest.",
            f"You found: {item_line(it)}",
            "Tip: Equip items in Character (C) back in town."
        ])

    def try_exit_floor():
        if any(e.alive for e in enemies):
            dialog.start(["A seal blocks the way…", "Defeat all enemies on this floor."])
            return False
        return True

    def sell_item(idx: int):
        if 0 <= idx < len(inventory):
            it = inventory.pop(idx)
            fighter.gold += max(1, it.value // 2)
            floaters.append(Floater(f"Sold +{max(1, it.value//2)}g", W/2, UI.y + 16, -10, 0.8, GRAY))

    def equip_item(idx: int):
        if not (0 <= idx < len(inventory)):
            return
        it = inventory[idx]

        def swap(slot_attr: str):
            cur = getattr(fighter, slot_attr)
            setattr(fighter, slot_attr, it)
            if cur is None:
                inventory.pop(idx)
            else:
                inventory[idx] = cur

        if it.slot == "Weapon":
            swap("weapon")
        elif it.slot == "Armor":
            swap("armor")
        else:
            swap("trinket")

        recompute_maxhp(party, fighter)
        floaters.append(Floater("Equipped!", W/2, UI.y + 16, -10, 0.8, YELLOW))

    def buy_potion():
        cost = 20
        if fighter.gold >= cost:
            fighter.gold -= cost
            fighter.potions += 1
            floaters.append(Floater("Bought Potion", W/2, UI.y + 16, -10, 0.8, CYAN))
        else:
            floaters.append(Floater("Not enough gold", W/2, UI.y + 16, -10, 0.8, RED))

    def upgrade_stat(which: str):
        base_cost = 35
        current = fighter.base_str if which == "STR" else fighter.base_int if which == "INT" else fighter.base_vit
        cost = base_cost + (current - 2) * 18
        if fighter.gold < cost:
            floaters.append(Floater("Not enough gold", W/2, UI.y + 16, -10, 0.8, RED))
            return
        fighter.gold -= cost
        if which == "STR":
            fighter.base_str += 1
        elif which == "INT":
            fighter.base_int += 1
        else:
            fighter.base_vit += 1
            recompute_maxhp(party, fighter)
        floaters.append(Floater(f"{which} +1", W/2, UI.y + 16, -10, 0.8, GREEN))

    recompute_maxhp(party, fighter)
    reset_to_town(msg=False)
    state = "title"

    while True:
        dt = clock.tick(FPS) / 1000.0

        # events
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if ev.type == pygame.KEYDOWN:
                if dialog.open and ev.key in (pygame.K_RETURN, pygame.K_SPACE):
                    dialog.next()
                    continue

                if state == "title":
                    if ev.key in (pygame.K_RETURN, pygame.K_SPACE):
                        reset_to_town()
                    continue

                if state in ("win", "lose"):
                    if ev.key == pygame.K_r:
                        fighter.gold = 0
                        fighter.potions = 2
                        fighter.weapon = fighter.armor = fighter.trinket = None
                        inventory.clear()
                        fighter.base_str, fighter.base_int, fighter.base_vit = 4, 2, 4
                        for p in party:
                            p.level = 1
                            p.xp = 0
                            p.xp_next = xp_next_for(1)
                        recompute_maxhp(party, fighter)
                        reset_to_town()
                    if ev.key == pygame.K_ESCAPE:
                        state = "title"
                    continue

                if state == "town":
                    if ev.key == pygame.K_d:
                        start_dungeon(1)
                    elif ev.key == pygame.K_s:
                        state = "shop"
                    elif ev.key == pygame.K_c:
                        state = "character"
                    elif ev.key == pygame.K_q:
                        dialog.start([
                            "Quest: The Glass Gate",
                            f"Clear {FLOORS_TO_CLEAR} dungeon floors and return.",
                            "Rumor: deeper floors drop better gear…",
                            "Enter dungeon with (D)."
                        ])
                    elif ev.key == pygame.K_ESCAPE:
                        state = "title"

                elif state == "shop":
                    if ev.key == pygame.K_1:
                        buy_potion()
                    elif ev.key == pygame.K_2:
                        upgrade_stat("STR")
                    elif ev.key == pygame.K_3:
                        upgrade_stat("INT")
                    elif ev.key == pygame.K_4:
                        upgrade_stat("VIT")
                    elif ev.key == pygame.K_ESCAPE:
                        state = "town"

                elif state == "character":
                    if ev.key == pygame.K_ESCAPE:
                        state = "town"
                    elif pygame.K_1 <= ev.key <= pygame.K_9:
                        idx = ev.key - pygame.K_1
                        if pygame.key.get_pressed()[pygame.K_LSHIFT] or pygame.key.get_pressed()[pygame.K_RSHIFT]:
                            sell_item(idx)
                        else:
                            equip_item(idx)

                elif state == "dungeon":
                    if ev.key == pygame.K_ESCAPE:
                        reset_to_town()
                    if ev.key == pygame.K_RETURN:
                        gx, gy = world_to_grid(fighter.x, fighter.y)
                        if dun.chest_cell and not chest_opened:
                            cx, cy = dun.chest_cell
                            if abs(gx - cx) <= 1 and abs(gy - cy) <= 1:
                                open_chest()
                        ex, ey = dun.exit_cell
                        if abs(gx - ex) <= 1 and abs(gy - ey) <= 1:
                            if try_exit_floor():
                                if dun.floor >= FLOORS_TO_CLEAR:
                                    state = "win"
                                else:
                                    start_dungeon(dun.floor + 1)

        keys = pygame.key.get_pressed()

        # generic decay
        shake[0] = max(0.0, shake[0] - dt * 18)
        for p in list(particles):
            p.x += p.vx * dt
            p.y += p.vy * dt
            p.vx *= (1 - dt * 6)
            p.vy *= (1 - dt * 6)
            p.t -= dt
            if p.t <= 0:
                particles.remove(p)
        for f in list(floaters):
            f.y += f.vy * dt
            f.t -= dt
            if f.t <= 0:
                floaters.remove(f)

        # dungeon update
        if state == "dungeon" and not dialog.open:
            # cooldown/iframes
            for m in party:
                if not m.alive:
                    continue
                m.iframes = max(0.0, m.iframes - dt)
                m.slash_cd = max(0.0, m.slash_cd - dt)
                m.dash_cd = max(0.0, m.dash_cd - dt)
                m.dash_t = max(0.0, m.dash_t - dt)
                m.bolt_cd = max(0.0, m.bolt_cd - dt)
                m.heal_cd = max(0.0, m.heal_cd - dt)

            # PLAYER (fighter)
            if fighter.alive:
                mx = (1 if keys[pygame.K_d] or keys[pygame.K_RIGHT] else 0) - (1 if keys[pygame.K_a] or keys[pygame.K_LEFT] else 0)
                my = (1 if keys[pygame.K_s] or keys[pygame.K_DOWN] else 0) - (1 if keys[pygame.K_w] or keys[pygame.K_UP] else 0)

                if mx or my:
                    nx, ny = normalize(mx, my)
                    fighter.face_x, fighter.face_y = nx, ny

                if (keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]) and fighter.dash_cd <= 0 and fighter.dash_t <= 0:
                    fighter.dash_cd = DASH_CD
                    fighter.dash_t = DASH_TIME
                    floaters.append(Floater("DASH", fighter.x, fighter.y - 28, -30, 0.6, YELLOW))

                spd = DASH_SPEED if fighter.dash_t > 0 else fighter.speed
                if mx or my:
                    nx, ny = normalize(mx, my)
                    move_with_collision(dun, fighter, nx * spd * dt, ny * spd * dt)

                # potion
                if keys[pygame.K_p] and fighter.potions > 0 and fighter.hp < fighter.max_hp:
                    fighter.potions -= 1
                    fighter.hp = min(fighter.max_hp, fighter.hp + 48)
                    floaters.append(Floater("+48", fighter.x, fighter.y - 24, -40, 0.8, GREEN))

                # slash
                if keys[pygame.K_SPACE] and fighter.slash_cd <= 0:
                    fighter.slash_cd = SLASH_CD
                    dmg = compute_fighter_slash_damage()
                    shake[0] = max(shake[0], 6)
                    for e in enemies:
                        if e.alive and cone_hit(fighter, e):
                            deal_damage(e, dmg, floaters, particles, shake, color=WHITE)

            # companion “unstuck” tracking
            for comp in (mage, healer):
                if not comp.alive:
                    continue
                update_stuck(comp, dt)
                if comp._stuck_t > 1.25:
                    # warp near fighter if stuck > 1.25s
                    warp_near(comp, dun, fighter.x, fighter.y)
                    comp._stuck_t = 0.0

            # MAGE AI: follow + only shoot visible enemies
            if mage.alive:
                dx, dy = fighter.x - mage.x, fighter.y - mage.y
                d = length(dx, dy)
                if d > 86:
                    nx, ny = normalize(dx, dy)
                    move_with_collision(dun, mage, nx * mage.speed * dt, ny * mage.speed * dt)

                t = nearest_enemy_visible(dun, enemies, mage.x, mage.y, max_range=420)
                if t and mage.bolt_cd <= 0:
                    tx, ty = t.x - mage.x, t.y - mage.y
                    nx, ny = normalize(tx, ty)
                    mage.bolt_cd = BOLT_CD
                    projectiles.append(Projectile(
                        x=mage.x + nx * (mage.r + 8),
                        y=mage.y + ny * (mage.r + 8),
                        vx=nx * BOLT_SPEED,
                        vy=ny * BOLT_SPEED,
                        r=6,
                        dmg=compute_mage_bolt_damage(),
                        t=BOLT_LIFE,
                        color=ORANGE,
                        owner_team="party"
                    ))

            # HEALER AI: follow + heal lowest ally in range
            if healer.alive:
                dx, dy = fighter.x - healer.x, fighter.y - healer.y
                d = length(dx, dy)
                if d > 98:
                    nx, ny = normalize(dx, dy)
                    move_with_collision(dun, healer, nx * healer.speed * dt, ny * healer.speed * dt)

                def needs_heal(a: Party):
                    return a.alive and (a.hp / a.max_hp) < 0.55

                ally = nearest_ally(party, healer.x, healer.y, needs_heal)
                if ally and healer.heal_cd <= 0:
                    if length(ally.x - healer.x, ally.y - healer.y) <= HEAL_RANGE:
                        healer.heal_cd = HEAL_CD
                        amt = compute_heal_amount()
                        ally.hp = min(ally.max_hp, ally.hp + amt)
                        floaters.append(Floater(f"+{amt}", ally.x, ally.y - 24, -40, 0.8, GREEN))

            # ENEMIES AI
            for e in enemies:
                if not e.alive:
                    continue
                e.iframes = max(0.0, e.iframes - dt)

                # target nearest party member in reachable region
                target = None
                best = 10**18
                for m in party:
                    if not m.alive:
                        continue
                    d2 = (m.x - e.x)**2 + (m.y - e.y)**2
                    if d2 < best:
                        best = d2
                        target = m
                if not target:
                    continue

                dx, dy = target.x - e.x, target.y - e.y
                nx, ny = normalize(dx, dy)
                spd = e.speed
                if e.kind == "Slime":
                    spd *= 0.75 + 0.25 * (0.5 + 0.5 * math.sin(pygame.time.get_ticks() * 0.007 + e.y * 0.02))
                if e.kind == "Bat":
                    wob = math.sin(pygame.time.get_ticks() * 0.01 + e.x * 0.01) * 0.9
                    rx, ry = -ny * wob, nx * wob
                    nx, ny = normalize(nx + rx, ny + ry)

                move_with_collision(dun, e, nx * spd * dt, ny * spd * dt)

                # contact damage
                if target.alive and target.iframes <= 0 and length(e.x - target.x, e.y - target.y) <= (e.r + target.r):
                    dmg = max(1, int(e.dps * dt * 60))
                    deal_damage(target, dmg, floaters, particles, shake, color=RED)

            # projectiles
            for pr in list(projectiles):
                pr.x += pr.vx * dt
                pr.y += pr.vy * dt
                pr.t -= dt
                particles.append(Particle(pr.x, pr.y, -pr.vx*0.02, -pr.vy*0.02, random.uniform(1.0, 2.0), 0.16, pr.color))
                if pr.t <= 0 or collides_walls(dun, pr.x, pr.y, pr.r):
                    projectiles.remove(pr)
                    continue
                for e in enemies:
                    if e.alive and length(pr.x - e.x, pr.y - e.y) <= (pr.r + e.r):
                        deal_damage(e, pr.dmg, floaters, particles, shake, color=pr.color)
                        if pr in projectiles:
                            projectiles.remove(pr)
                        break

            # death / rewards
            for m in party:
                if m.alive and m.hp <= 0:
                    m.alive = False
                    floaters.append(Floater(f"{m.role} falls!", m.x, m.y - 20, -35, 1.1, RED))

            for e in enemies:
                if (not e.alive) and not e._rewarded:
                    e._rewarded = True
                    grant_xp_gold(party, e.bounty_xp, e.bounty_gold, fighter, floaters, e.x, e.y)
                    if random.random() < 0.12:
                        it = roll_item(dun.floor)
                        inventory.append(it)
                        floaters.append(Floater("LOOT!", e.x, e.y - 22, -30, 0.9, YELLOW))
                        floaters.append(Floater(it.name, e.x, e.y - 4, -30, 0.9, GRAY))
                    if random.random() < 0.08:
                        fighter.potions += 1
                        floaters.append(Floater("+Potion", e.x, e.y + 10, -25, 0.9, CYAN))

            # lose
            if all(not m.alive for m in party):
                state = "lose"

            # floor cleared flag
            dun.cleared = not any(e.alive for e in enemies)

        # ---------- draw ----------
        sx = int(random.uniform(-shake[0], shake[0])) if shake[0] > 0 else 0
        sy = int(random.uniform(-shake[0], shake[0])) if shake[0] > 0 else 0

        screen.fill(BG)
        pygame.draw.rect(screen, PANEL2, ARENA.move(sx, sy), border_radius=18)
        pygame.draw.rect(screen, BORDER, ARENA.move(sx, sy), 3, border_radius=18)

        pygame.draw.rect(screen, PANEL, UI, border_radius=16)
        pygame.draw.rect(screen, BORDER, UI, 2, border_radius=16)

        if state == "title":
            draw_center_text(screen, big, "ISEKAI-STYLE MICRO RPG", 150, WHITE)
            draw_center_text(screen, mid, "Town → Dungeon → Loot → Upgrades (no external assets)", 215, GRAY)
            draw_center_text(screen, font, "Press Enter / Space", 325, YELLOW)

        elif state in ("town", "shop", "character"):
            draw_text(screen, mid, "TOWN HUB", ARENA.x + 18, ARENA.y + 12, WHITE)
            draw_text(screen, font, "D: Enter Dungeon   S: Shop   C: Character   Q: Quest", ARENA.x + 18, ARENA.y + 40, GRAY)

            draw_text(screen, font, f"Gold: {fighter.gold}g   Potions: {fighter.potions}", UI.x + 14, UI.y + 10, YELLOW)
            draw_text(screen, font, f"STR {fighter.total_str()}   INT {fighter.total_int()}   VIT {fighter.total_vit()}", UI.x + 14, UI.y + 34, WHITE)

            wpn = item_line(fighter.weapon) if fighter.weapon else "(none)"
            arm = item_line(fighter.armor) if fighter.armor else "(none)"
            trk = item_line(fighter.trinket) if fighter.trinket else "(none)"
            draw_text(screen, font, f"Weapon: {wpn}", UI.x + 360, UI.y + 10, GRAY)
            draw_text(screen, font, f"Armor:  {arm}", UI.x + 360, UI.y + 34, GRAY)
            draw_text(screen, font, f"Trinket:{trk}", UI.x + 360, UI.y + 58, GRAY)

            if state == "shop":
                draw_text(screen, mid, "SHOP", ARENA.x + 18, ARENA.y + 90, WHITE)
                draw_text(screen, font, "1) Buy Potion (20g)", ARENA.x + 18, ARENA.y + 125, CYAN)
                draw_text(screen, font, "2) Upgrade STR (cost scales)", ARENA.x + 18, ARENA.y + 150, GREEN)
                draw_text(screen, font, "3) Upgrade INT (cost scales)", ARENA.x + 18, ARENA.y + 175, GREEN)
                draw_text(screen, font, "4) Upgrade VIT (cost scales)", ARENA.x + 18, ARENA.y + 200, GREEN)
                draw_text(screen, font, "Esc) Back", ARENA.x + 18, ARENA.y + 235, GRAY)

            if state == "character":
                draw_text(screen, mid, "CHARACTER", ARENA.x + 18, ARENA.y + 90, WHITE)
                draw_text(screen, font, "Inventory (1-9 = Equip, Shift+1-9 = Sell)", ARENA.x + 18, ARENA.y + 125, GRAY)
                if not inventory:
                    draw_text(screen, font, "(No items yet — clear floors and open chests.)", ARENA.x + 18, ARENA.y + 155, GRAY)
                else:
                    y0 = ARENA.y + 155
                    for i, it in enumerate(inventory[:9]):
                        draw_text(screen, font, f"{i+1}) {item_line(it)}  (sell {max(1, it.value//2)}g)", ARENA.x + 18, y0 + i*22, WHITE)

        elif state == "dungeon":
            # tiles
            for gy in range(GH):
                for gx in range(GW):
                    t = dun.tiles[gy][gx]
                    wx = ARENA.x + gx * TILE + sx
                    wy = ARENA.y + gy * TILE + sy
                    if t == 1:
                        pygame.draw.rect(screen, (22, 22, 30), pygame.Rect(wx, wy, TILE, TILE))
                    else:
                        pygame.draw.rect(screen, (28, 28, 38), pygame.Rect(wx, wy, TILE, TILE))
                        if (gx + gy) % 7 == 0:
                            pygame.draw.rect(screen, (31, 31, 44), pygame.Rect(wx, wy, TILE, TILE), 1)

            # exit
            ex, ey = dun.exit_cell
            xx, yy = grid_to_world(ex, ey)
            r = pygame.Rect(int(xx - 10) + sx, int(yy - 10) + sy, 20, 20)
            pygame.draw.rect(screen, (70, 70, 95), r, border_radius=6)
            pygame.draw.rect(screen, YELLOW if dun.cleared else GRAY, r, 2, border_radius=6)

            # chest
            if dun.chest_cell and not chest_opened:
                cx, cy = dun.chest_cell
                cxw, cyw = grid_to_world(cx, cy)
                cr = pygame.Rect(int(cxw - 9) + sx, int(cyw - 9) + sy, 18, 18)
                pygame.draw.rect(screen, (90, 70, 40), cr, border_radius=4)
                pygame.draw.rect(screen, YELLOW, cr, 2, border_radius=4)

            # particles
            for p in particles:
                pygame.draw.circle(screen, p.color, (int(p.x)+sx, int(p.y)+sy), int(p.r))

            # projectiles
            for pr in projectiles:
                pygame.draw.circle(screen, pr.color, (int(pr.x)+sx, int(pr.y)+sy), pr.r)
                pygame.draw.circle(screen, WHITE, (int(pr.x)+sx, int(pr.y)+sy), pr.r, 2)

            # enemies
            for e in enemies:
                if not e.alive:
                    continue
                exw, eyw = int(e.x)+sx, int(e.y)+sy
                pygame.draw.circle(screen, e.color, (exw, eyw), e.r)
                pygame.draw.circle(screen, WHITE, (exw, eyw), e.r, 2)
                if e.iframes > 0:
                    pygame.draw.circle(screen, YELLOW, (exw, eyw), e.r + 6, 2)
                bw = 70 if e.kind == "Boss" else 34
                bar(screen, exw - bw//2, eyw - e.r - 14, bw, 6, e.hp / max(1, e.max_hp),
                    RED if e.kind == "Boss" else GREEN, bg=(55, 55, 65))

            # party
            for m in party:
                if not m.alive:
                    continue
                px, py = int(m.x)+sx, int(m.y)+sy
                pygame.draw.circle(screen, m.color, (px, py), m.r)
                pygame.draw.circle(screen, WHITE, (px, py), m.r, 2)
                if m.iframes > 0:
                    pygame.draw.circle(screen, YELLOW, (px, py), m.r + 6, 2)
                role_dot = BLUE if m.role == "Fighter" else ORANGE if m.role == "Mage" else CYAN
                pygame.draw.circle(screen, role_dot, (px + m.r - 2, py - m.r + 2), 4)

            # floaters
            for f in floaters:
                img = font.render(f.text, True, f.color)
                screen.blit(img, (int(f.x - img.get_width()/2), int(f.y)))

            # UI
            draw_text(screen, font, f"Floor {dun.floor}/{FLOORS_TO_CLEAR}", UI.x + 14, UI.y + 10, YELLOW)
            draw_text(screen, font, f"Gold {fighter.gold}g   Potions {fighter.potions} (P)", UI.x + 14, UI.y + 34, GRAY)

            x0 = UI.x + 250
            for i, m in enumerate(party):
                y = UI.y + 10 + i*28
                col = WHITE if m.alive else GRAY
                draw_text(screen, font, f"{m.role} Lv{m.level}", x0, y, col)
                bar(screen, x0 + 120, y + 4, 160, 12, (m.hp / m.max_hp) if m.alive else 0, GREEN if m.alive else GRAY)
                if m.alive:
                    bar(screen, x0 + 290, y + 4, 120, 12, m.xp / max(1, m.xp_next), PURPLE, bg=(55, 55, 68))

            right = UI.x + UI.w - 330
            atk = "READY" if fighter.slash_cd <= 0 else f"{fighter.slash_cd:.1f}s"
            dash = "READY" if fighter.dash_cd <= 0 else f"{fighter.dash_cd:.1f}s"
            draw_text(screen, font, f"Slash: {atk} (Space)", right, UI.y + 10, WHITE)
            draw_text(screen, font, f"Dash:  {dash} (Shift)", right, UI.y + 34, WHITE)
            draw_text(screen, font, "Mage/Healer act only with LOS (no attacking through rooms).", right, UI.y + 58, GRAY)

            if dun.cleared:
                draw_text(screen, font, "Floor cleared! Go to the exit and press Enter.", UI.x + 14, UI.y + 74, YELLOW)
            else:
                draw_text(screen, font, f"Enemies remaining: {sum(1 for e in enemies if e.alive)}", UI.x + 14, UI.y + 74, GRAY)

        if state == "win":
            overlay = pygame.Surface((W, H), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 160))
            screen.blit(overlay, (0, 0))
            draw_center_text(screen, big, "YOU ESCAPE THE GLASS GATE", 170, GREEN)
            draw_center_text(screen, mid, "The town celebrates… for now.", 235, WHITE)
            draw_center_text(screen, font, "Press R to reset demo, or Esc for title.", 300, GRAY)

        if state == "lose":
            overlay = pygame.Surface((W, H), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 175))
            screen.blit(overlay, (0, 0))
            draw_center_text(screen, big, "RUN ENDS", 180, RED)
            draw_center_text(screen, mid, "The dungeon claims another party.", 240, WHITE)
            draw_center_text(screen, font, "Press R to reset demo, or Esc for title.", 300, GRAY)

        # dialog overlay
        if dialog.open:
            box = pygame.Rect(60, H - UI_H - 120, W - 120, 90)
            pygame.draw.rect(screen, (10, 10, 14), box, border_radius=12)
            pygame.draw.rect(screen, BORDER, box, 2, border_radius=12)
            line = dialog.current() or ""
            draw_text(screen, mid, line, box.x + 16, box.y + 14, WHITE)
            draw_text(screen, font, "Enter/Space to continue", box.right - 190, box.bottom - 26, GRAY)

        pygame.display.flip()

if __name__ == "__main__":
    main()
