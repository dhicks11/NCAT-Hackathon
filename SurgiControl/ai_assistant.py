import speech_recognition as sr
import pyttsx3
import threading
import numpy as np
from typing import Dict, List, Tuple
import re

class SurgicalAIAssistant:
    def __init__(self, robot_sim):
        self.robot_sim = robot_sim
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.tts = pyttsx3.init()
        self.is_listening = False
        
        # Calibrate microphone
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)
    
    def speak(self, text: str):
        """Convert text to speech"""
        self.tts.say(text)
        self.tts.runAndWait()
    
    def listen_for_command(self) -> str:
        """Listen for voice command"""
        try:
            with self.microphone as source:
                audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=5)
            command = self.recognizer.recognize_google(audio).lower()
            return command
        except:
            return ""
    
    def parse_command(self, command: str) -> Dict:
        """Parse voice command into robot actions"""
        # Movement commands
        if "move to" in command:
            coords = self.extract_coordinates(command)
            if coords:
                return {"action": "move_to", "coordinates": coords}
        
        # Joint commands
        if "rotate joint" in command:
            joint_info = self.extract_joint_command(command)
            if joint_info:
                return {"action": "rotate_joint", **joint_info}
        
        # Preset positions
        if "home position" in command or "reset" in command:
            return {"action": "home"}
        
        if "demo" in command:
            return {"action": "demo"}
        
        return {"action": "unknown"}
    
    def extract_coordinates(self, command: str) -> Tuple[float, float, float]:
        """Extract x, y, z coordinates from command"""
        numbers = re.findall(r'-?\d+\.?\d*', command)
        if len(numbers) >= 3:
            return (float(numbers[0]), float(numbers[1]), float(numbers[2]))
        return None
    
    def extract_joint_command(self, command: str) -> Dict:
        """Extract joint number and angle from command"""
        numbers = re.findall(r'-?\d+\.?\d*', command)
        if len(numbers) >= 2:
            return {"joint": int(numbers[0]), "angle": float(numbers[1])}
        return None
    
    def execute_command(self, parsed_command: Dict):
        """Execute the parsed command on the robot"""
        action = parsed_command["action"]
        
        if action == "move_to":
            x, y, z = parsed_command["coordinates"]
            try:
                q = self.robot_sim.demo_inverse_kinematics(x, y, z)
                self.robot_sim.draw_pose(q)
                self.speak(f"Moving to position {x}, {y}, {z}")
            except:
                self.speak("Cannot reach that position")
        
        elif action == "rotate_joint":
            joint = parsed_command["joint"] - 1  # Convert to 0-indexed
            angle = np.radians(parsed_command["angle"])
            if 0 <= joint < self.robot_sim.dof:
                q = [0.0] * self.robot_sim.dof
                q[joint] = angle
                self.robot_sim.draw_pose(q)
                self.speak(f"Rotating joint {joint + 1} to {parsed_command['angle']} degrees")
            else:
                self.speak("Invalid joint number")
        
        elif action == "home":
            q = [0.0] * self.robot_sim.dof
            self.robot_sim.draw_pose(q)
            self.speak("Moving to home position")
        
        elif action == "demo":
            self.speak("Starting demonstration")
            # Run a simple demo trajectory
            from simulation_environment import simple_ik_trajectory_demo
            simple_ik_trajectory_demo()
        
        else:
            self.speak("Command not recognized")
    
    def start_listening(self):
        """Start the voice command loop"""
        self.is_listening = True
        self.speak("AI Assistant ready. Say commands like 'move to 5 3 2' or 'rotate joint 1 45 degrees'")
        
        while self.is_listening:
            command = self.listen_for_command()
            if command:
                print(f"Heard: {command}")
                parsed = self.parse_command(command)
                self.execute_command(parsed)
                
                if "stop" in command or "exit" in command:
                    self.is_listening = False
                    self.speak("Assistant stopped")
    
    def stop_listening(self):
        """Stop the voice command loop"""
        self.is_listening = False

def demo_ai_assistant():
    """Demo the AI assistant with the robot simulator"""
    from simulation_environment import SerialRobotSim
    
    # Create robot and assistant
    robot = SerialRobotSim()
    robot.setup_axes_3d(title="AI-Controlled Surgical Robot")
    
    assistant = SurgicalAIAssistant(robot)
    
    # Start in a separate thread to keep GUI responsive
    thread = threading.Thread(target=assistant.start_listening)
    thread.daemon = True
    thread.start()
    
    # Show the robot visualization
    import matplotlib.pyplot as plt
    plt.show()
    
    return assistant, robot

if __name__ == "__main__":
    demo_ai_assistant()