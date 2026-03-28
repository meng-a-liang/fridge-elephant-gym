
import pygame


def blit_sprite(screen, surface, pos):
    """无阴影平铺，像矢量/扁平物体叠在背景上，避免「卡片贴图」感。"""
    screen.blit(surface, pos)


def draw_with_shadow(
    surface,
    pos,
    screen,
    shadow_offset=(3, 3),
    shadow_color=(100, 100, 100, 120),
    soft=False,
    shadow=True,
):
    """
    shadow=False：仅 blit，用于抠图精灵 + 纯色背景。
    shadow=True：带阴影（演示用）。
    """
    if not shadow:
        screen.blit(surface, pos)
        return
    if soft:
        shadow_offset = (2, 2)
        shadow_color = (90, 95, 105, 55)
    shadow_surface = surface.copy()
    shadow_surface.fill(shadow_color, special_flags=pygame.BLEND_RGBA_MULT)
    shadow_pos = (pos[0] + shadow_offset[0], pos[1] + shadow_offset[1])
    screen.blit(shadow_surface, shadow_pos)
    screen.blit(surface, pos)