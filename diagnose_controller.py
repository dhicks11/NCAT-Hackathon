#!/usr/bin/env python3
"""
Diagnostic script to identify controller issues.
Run this to see detailed information about what's wrong.
"""

import sys

print("="*60)
print("Controller Diagnostic Tool")
print("="*60)

# Test 1: Python version
print("\n1. Python Version:")
print(f"   {sys.version}")

# Test 2: Import pygame
print("\n2. Testing pygame import...")
try:
    import pygame
    print(f"   ✓ Pygame imported successfully")
    print(f"   Pygame version: {pygame.version.ver}")
except ImportError as e:
    print(f"   ✗ ERROR: Failed to import pygame: {e}")
    print("   Solution: pip install pygame")
    sys.exit(1)
except Exception as e:
    print(f"   ✗ ERROR: Unexpected error importing pygame: {e}")
    sys.exit(1)

# Test 3: Initialize pygame
print("\n3. Testing pygame initialization...")
try:
    pygame.init()
    if pygame.get_init():
        print("   ✓ Pygame initialized successfully")
    else:
        print("   ✗ ERROR: Pygame initialization returned False")
        sys.exit(1)
except Exception as e:
    print(f"   ✗ ERROR: Failed to initialize pygame: {e}")
    sys.exit(1)

# Test 4: Initialize joystick subsystem
print("\n4. Testing joystick subsystem...")
try:
    pygame.joystick.init()
    if pygame.joystick.get_init():
        print("   ✓ Joystick subsystem initialized")
    else:
        print("   ✗ ERROR: Joystick subsystem initialization returned False")
        pygame.quit()
        sys.exit(1)
except Exception as e:
    print(f"   ✗ ERROR: Failed to initialize joystick subsystem: {e}")
    pygame.quit()
    sys.exit(1)

# Test 5: Check for controllers
print("\n5. Checking for connected controllers...")
try:
    joystick_count = pygame.joystick.get_count()
    print(f"   Found {joystick_count} controller(s)")
    
    if joystick_count == 0:
        print("   ✗ WARNING: No controllers detected!")
        print("\n   Troubleshooting steps:")
        print("   1. Connect your DualSense controller via USB")
        print("   2. Or pair it via Bluetooth")
        print("   3. On Windows: Check Device Manager")
        print("   4. Try unplugging and replugging the controller")
        pygame.quit()
        sys.exit(1)
    else:
        print("   ✓ Controller(s) detected!")
        
        # Test 6: Get controller info
        print("\n6. Controller Information:")
        for i in range(joystick_count):
            try:
                joystick = pygame.joystick.Joystick(i)
                joystick.init()
                
                name = joystick.get_name()
                axes = joystick.get_numaxes()
                buttons = joystick.get_numbuttons()
                hats = joystick.get_numhats()
                
                print(f"\n   Controller {i}: {name}")
                print(f"   - Axes: {axes}")
                print(f"   - Buttons: {buttons}")
                print(f"   - Hats: {hats}")
                
                # Test reading axes
                print(f"\n   Testing axis readings:")
                for axis_idx in range(min(axes, 6)):  # Test first 6 axes
                    try:
                        value = joystick.get_axis(axis_idx)
                        print(f"     Axis {axis_idx}: {value:.3f}")
                    except Exception as e:
                        print(f"     Axis {axis_idx}: ERROR - {e}")
                
                # Test reading buttons
                print(f"\n   Testing button readings:")
                for btn_idx in range(min(buttons, 10)):  # Test first 10 buttons
                    try:
                        pressed = joystick.get_button(btn_idx)
                        status = "PRESSED" if pressed else "released"
                        print(f"     Button {btn_idx}: {status}")
                    except Exception as e:
                        print(f"     Button {btn_idx}: ERROR - {e}")
                
            except Exception as e:
                print(f"   ✗ ERROR accessing controller {i}: {e}")
                
except Exception as e:
    print(f"   ✗ ERROR: Failed to get joystick count: {e}")
    pygame.quit()
    sys.exit(1)

# Test 7: Test event handling
print("\n7. Testing event handling...")
try:
    pygame.event.pump()
    events = pygame.event.get()
    print(f"   ✓ Event system working ({len(events)} events in queue)")
except Exception as e:
    print(f"   ✗ ERROR: Event handling failed: {e}")

# Cleanup
pygame.quit()

print("\n" + "="*60)
print("Diagnostic complete!")
print("="*60)
print("\nIf all tests passed, your controller should work with the simulation.")
print("If you see errors, follow the troubleshooting steps above.")

