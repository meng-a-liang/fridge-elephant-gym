import pygame

def scale_bg(bg, screen_width, screen_height):
    """按比例缩放背景图片，使其居中适配屏幕"""
    bg_w, bg_h = bg.get_size()
    scale_x = screen_width / bg_w
    scale_y = screen_height / bg_h
    scale = min(scale_x, scale_y, 1)  # 避免放大超过原图
    scaled_bg = pygame.transform.scale(bg, (int(bg_w*scale), int(bg_h*scale)))
    bg_rect = scaled_bg.get_rect(center=(screen_width//2, screen_height//2))
    return scaled_bg, bg_rect

def draw_with_shadow(surface, pos, screen):
    """绘制带阴影的元素，减少突兀感"""
    shadow = surface.copy()
    shadow.set_alpha(80)
    screen.blit(shadow, (pos[0]+5, pos[1]+5))
    screen.blit(surface, pos)


def scale_bg():
    return None


def draw_with_shadow():
    return None