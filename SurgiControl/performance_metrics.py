#!/usr/bin/env python3
import time
import numpy as np
from typing import List, Dict

class PerformanceMetrics:
    def __init__(self):
        self.start_time = time.time()
        self.movements = []
        self.precision_score = 100.0
        self.efficiency_score = 100.0
        
    def track_movement(self, joint_angles: List[float], target_pos: List[float] = None):
        """Track robot movements for performance analysis"""
        current_time = time.time()
        
        # Calculate movement smoothness (jerk)
        if len(self.movements) > 1:
            prev_angles = self.movements[-1]['angles']
            jerk = sum(abs(a - b) for a, b in zip(joint_angles, prev_angles))
            
            # Penalize jerky movements
            if jerk > 0.5:
                self.precision_score = max(0, self.precision_score - 1)
        
        self.movements.append({
            'time': current_time,
            'angles': joint_angles.copy(),
            'target': target_pos.copy() if target_pos else None
        })
    
    def get_metrics(self) -> Dict:
        """Get current performance metrics"""
        elapsed = time.time() - self.start_time
        
        # Calculate efficiency based on movement count and time
        movement_efficiency = max(0, 100 - len(self.movements) * 0.1)
        
        # Calculate tremor (movement stability)
        tremor_score = 100.0
        if len(self.movements) > 10:
            recent_moves = self.movements[-10:]
            variations = []
            for i in range(1, len(recent_moves)):
                var = sum(abs(a - b) for a, b in zip(recent_moves[i]['angles'], recent_moves[i-1]['angles']))
                variations.append(var)
            avg_variation = np.mean(variations)
            tremor_score = max(0, 100 - avg_variation * 50)
        
        return {
            'precision': round(self.precision_score, 1),
            'efficiency': round(movement_efficiency, 1),
            'tremor_control': round(tremor_score, 1),
            'elapsed_time': round(elapsed, 1),
            'total_movements': len(self.movements),
            'overall_score': round((self.precision_score + movement_efficiency + tremor_score) / 3, 1)
        }