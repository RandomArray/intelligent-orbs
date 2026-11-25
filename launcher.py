#!/usr/bin/env python3
"""
Intelligent Orbs Launcher
Run this script to start the neural network-based orb simulation with monitoring
"""

import sys
import os
import subprocess
import time

def main():
    print("🧠 Intelligent Orbs - Neural Network Life Simulation")
    print("=" * 55)
    print()
    print("This advanced AI ecosystem features:")
    print("• 🧠 Neural network brains for each orb")
    print("• 🧬 Evolutionary behaviors and learning")
    print("• 🎯 Hunting, mating, and survival dynamics")
    print("• 📊 Real-time monitoring and analysis")
    print("• 🎨 Beautiful dark-themed interface")
    print()
    
    while True:
        print("Choose an option:")
        print("1. 🚀 Start Orbs with Monitor (Recommended)")
        print("2. 🎮 Start Orbs Only")
        print("3. 📊 Start Monitor Only")
        print("4. ❓ Help & Documentation")
        print("5. 🚪 Exit")
        print()
        
        choice = input("Enter your choice (1-5): ").strip()
        
        if choice == "1":
            print("\n🖥️ Starting Monitor Interface...")
            try:
                monitor_process = subprocess.Popen([sys.executable, "orb_monitor.py"])
                print("✅ Monitor started successfully!")
                print("Use the monitor interface to start and control the simulation.")
                print("Press Ctrl+C to return to menu.")
                try:
                    monitor_process.wait()
                except KeyboardInterrupt:
                    print("\n🔄 Returning to menu...")
                    monitor_process.terminate()
            except Exception as e:
                print(f"❌ Error starting monitor: {e}")
        
        elif choice == "2":
            print("\n🚀 Starting Intelligent Orbs simulation...")
            print("🖱️ Move your cursor around to interact with the AI orbs!")
            print("Press Ctrl+C to stop and return to menu.")
            print()
            
            try:
                orb_process = subprocess.Popen([sys.executable, "intelligent_orbs.py"])
                orb_process.wait()
            except KeyboardInterrupt:
                print("\n🛑 Simulation stopped.")
                if 'orb_process' in locals():
                    orb_process.terminate()
            except Exception as e:
                print(f"❌ Error: {e}")
                print("Make sure PyQt5 and numpy are installed.")
        
        elif choice == "3":
            print("\n� Starting Monitor Only...")
            try:
                subprocess.run([sys.executable, "orb_monitor.py"])
            except Exception as e:
                print(f"❌ Error: {e}")
        
        elif choice == "4":
            show_help()
        
        elif choice == "5":
            print("\n👋 Thanks for exploring the world of Intelligent Orbs!")
            print("🧠 Keep evolving! 🧬")
            break
        
        else:
            print("❌ Invalid choice. Please enter 1-5.")
        
        print()

def show_help():
    """Display help and documentation"""
    print("\n" + "="*60)
    print("🆘 HELP & DOCUMENTATION")
    print("="*60)
    print()
    print("🧠 WHAT ARE INTELLIGENT ORBS?")
    print("Each orb is an AI agent with its own neural network brain.")
    print("They exhibit complex behaviors like hunting, mating, and evolution.")
    print()
    print("🎮 HOW TO INTERACT:")
    print("• Move your mouse cursor around the screen")
    print("• Orbs will react based on their AI personalities")
    print("• Some follow, some flee, some hunt - all unpredictable!")
    print()
    print("🎨 VISUAL INDICATORS:")
    print("• 🔴 Red glow: Hunting mode (aggressive)")
    print("• 💕 Pink glow: Mating mode (reproductive)")
    print("• 🔵 Blue glow: Fleeing mode (defensive)")
    print("• ✨ Brightness: Energy level")
    print("• 📏 Size: Grows with successful hunting")
    print()
    print("🧬 EVOLUTION MECHANICS:")
    print("• Orbs mate and create offspring with mixed traits")
    print("• Neural networks evolve through genetic algorithms")
    print("• Successful hunters live longer and reproduce more")
    print("• Traits like aggression, curiosity adapt over time")
    print()
    print("📊 MONITOR FEATURES:")
    print("• Real-time population statistics")
    print("• Behavior pattern analysis")
    print("• Evolution tracking and genetics")
    print("• Activity logging")
    print("• Dark theme for comfortable viewing")
    print()
    print("🔧 SYSTEM REQUIREMENTS:")
    print("• Python 3.7+")
    print("• PyQt5 (GUI framework)")
    print("• NumPy (math and lightweight neural nets)")
    print()
    print("🐛 TROUBLESHOOTING:")
    print("• If orbs don't appear: Check that PyQt5 is installed")
    print("• If simulation crashes: Ensure all dependencies are installed")
    print("• If monitor shows no data: Start orbs first, then monitor")
    print()
    print("Press Enter to return to menu...")
    input()

if __name__ == "__main__":
    main()
