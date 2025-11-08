#!/usr/bin/env python3
"""
Quick test script to verify controller connectivity and see button/axis mappings.
"""

import pygame

pygame.init()
pygame.joystick.init()

joystick_count = pygame.joystick.get_count()

if joystick_count == 0:
    print("No controllers detected!")
    print("Please connect your DualSense controller and run this script again.")
else:
    print(f"\nFound {joystick_count} controller(s):\n")
    
    for i in range(joystick_count):
        joystick = pygame.joystick.Joystick(i)
        joystick.init()
        
        print(f"Controller {i}: {joystick.get_name()}")
        print(f"  Axes: {joystick.get_numaxes()}")
        print(f"  Buttons: {joystick.get_numbuttons()}")
        print(f"  Hats: {joystick.get_numhats()}")
        print()
        
        # Test reading values
        print("  Press buttons and move sticks to test...")
        print("  Press ESC or close window to exit\n")
        
        screen = pygame.display.set_mode((400, 300))
        pygame.display.set_caption(f"Controller {i} Test")
        clock = pygame.time.Clock()
        
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                elif event.type == pygame.JOYBUTTONDOWN:
                    print(f"  Button {event.button} pressed")
                elif event.type == pygame.JOYAXISMOTION:
                    if abs(event.value) > 0.1:  # Only print significant movements
                        print(f"  Axis {event.axis}: {event.value:.3f}")
            
            # Display current state
            screen.fill((0, 0, 0))
            font = pygame.font.Font(None, 24)
            
            y = 10
            screen.blit(font.render(f"Controller: {joystick.get_name()}", True, (255, 255, 255)), (10, y))
            y += 30
            
            # Display axes
            for axis in range(joystick.get_numaxes()):
                value = joystick.get_axis(axis)
                text = f"Axis {axis}: {value:.3f}"
                screen.blit(font.render(text, True, (255, 255, 255)), (10, y))
                y += 25
            
            # Display buttons
            for button in range(joystick.get_numbuttons()):
                pressed = joystick.get_button(button)
                color = (0, 255, 0) if pressed else (128, 128, 128)
                text = f"Button {button}: {'PRESSED' if pressed else 'released'}"
                screen.blit(font.render(text, True, color), (10, y))
                y += 25
            
            pygame.display.flip()
            clock.tick(60)
        
        pygame.quit()
        break  # Only test first controller for now

pygame.quit()
print("\nTest complete!")

