#!/usr/bin/env python3
import numpy as np
import pygame
import time
from typing import Tuple, List

class HapticFeedback:
    def __init__(self):
        pygame.init()
        pygame.joystick.init()
        self.joystick = None
        if pygame.joystick.get_count() > 0:
            self.joystick = pygame.joystick.Joystick(0)
            self.joystick.init()
        
        self.force_feedback = True
        self.tissue_stiffness = 0.5  # 0-1 scale
        self.collision_force = 0.0
        
    def calculate_tissue_resistance(self, position: List[float], velocity: List[float]) -> float:
        """Simulate tissue resistance based on position and movement speed"""
        # Simulate different tissue densities
        depth = abs(position[2]) if len(position) > 2 else 0
        resistance = self.tissue_stiffness * (1 + depth * 0.1)
        
        # Add velocity-dependent damping
        speed = np.linalg.norm(velocity) if velocity else 0
        damping = min(speed * 0.2, 1.0)
        
        return resistance + damping
    
    def detect_collision(self, position: List[float]) -> bool:
        """Detect collision with virtual organs/tissues"""
        # Simulate organ boundaries
        organ_zones = [
            {"center": [5, 0, 3], "radius": 2.0, "stiffness": 0.8},  # Heart
            {"center": [-3, 2, 1], "radius": 1.5, "stiffness": 0.6},  # Liver
        ]
        
        for organ in organ_zones:
            distance = np.linalg.norm(np.array(position[:3]) - np.array(organ["center"]))
            if distance < organ["radius"]:
                self.collision_force = organ["stiffness"] * (1 - distance/organ["radius"])
                return True
        
        self.collision_force = 0.0
        return False
    
    def apply_force_feedback(self, position: List[float], velocity: List[float] = None):
        """Apply haptic force feedback through controller vibration"""
        if not self.joystick or not self.force_feedback:
            return
        
        # Calculate total force
        tissue_force = self.calculate_tissue_resistance(position, velocity or [0,0,0])
        collision_detected = self.detect_collision(position)
        
        total_force = tissue_force + self.collision_force
        
        # Map force to vibration intensity (0-1)
        vibration_intensity = min(total_force, 1.0)
        
        # Apply different vibration patterns
        if collision_detected:
            # Sharp, pulsing vibration for collision
            self.pulse_vibration(0.8, 0.1)
        elif tissue_force > 0.3:
            # Continuous vibration for tissue resistance
            self.continuous_vibration(vibration_intensity)
    
    def pulse_vibration(self, intensity: float, duration: float):
        """Create pulsing vibration pattern"""
        if self.joystick and hasattr(self.joystick, 'rumble'):
            self.joystick.rumble(intensity, intensity, int(duration * 1000))
    
    def continuous_vibration(self, intensity: float):
        """Create continuous vibration"""
        if self.joystick and hasattr(self.joystick, 'rumble'):
            self.joystick.rumble(intensity * 0.3, intensity * 0.7, 50)
    
    def simulate_heartbeat(self):
        """Simulate heartbeat when near heart"""
        # 72 BPM = 1.2 Hz
        heartbeat_pattern = [0.6, 0.0, 0.4, 0.0]
        for intensity in heartbeat_pattern:
            if self.joystick and hasattr(self.joystick, 'rumble'):
                self.joystick.rumble(intensity, intensity, 100)
            time.sleep(0.2)