
import pygame

def draw_with_shadow(surface, pos, screen, shadow_offset=(3, 3), shadow_color=(100, 100, 100, 120)):
    """绘制带阴影的元素（突出冰箱/大象，单调背景下更醒目）"""
    # 绘制阴影
    shadow_surface = surface.copy()
    shadow_surface.fill(shadow_color, special_flags=pygame.BLEND_RGBA_MULT)
    shadow_pos = (pos[0] + shadow_offset[0], pos[1] + shadow_offset[1])
    screen.blit(shadow_surface, shadow_pos)
    # 绘制元素本体
    screen.blit(surface, pos)