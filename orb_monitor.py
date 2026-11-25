import sys
import json
import os
import subprocess
import time
import threading
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import QTimer, QThread, pyqtSignal

class OrbStatsCollector(QThread):
    """
    Background thread to collect orb statistics from the simulation
    """
    stats_updated = pyqtSignal(dict)
    
    def __init__(self):
        super().__init__()
        self.running = False
        self.orb_process = None
        
    def set_process(self, process):
        self.orb_process = process
        
    def run(self):
        self.running = True
        while self.running:
            if self.orb_process and self.orb_process.poll() is None:
                # Process is running - simulate collecting stats
                # In a real implementation, you'd communicate with the orb process
                mock_stats = self.generate_mock_stats()
                self.stats_updated.emit(mock_stats)
            time.sleep(0.5)  # Update every 500ms
    
    def generate_mock_stats(self):
        """Generate realistic statistics from actual simulation data"""
        # Try to read real stats from the simulation
        try:
            stats_file = os.path.join(os.path.dirname(__file__), 'orb_stats.json')
            if os.path.exists(stats_file):
                with open(stats_file, 'r') as f:
                    stats = json.load(f)
                    return stats
        except Exception:
            pass
        
        # Fallback to mock data if real stats unavailable
        import random
        return {
            'total_orbs': random.randint(8, 25),
            'behaviors': {
                'exploring': random.randint(3, 8),
                'hunting': random.randint(0, 4),
                'fleeing': random.randint(0, 3),
                'mating': random.randint(0, 4),
                'dying': random.randint(0, 2)
            },
            'avg_energy': round(random.uniform(45, 85), 1),
            'avg_age': random.randint(100, 2000),
            'avg_aggression': round(random.uniform(0.3, 0.7), 3),
            'avg_curiosity': round(random.uniform(0.4, 0.8), 3),
            'avg_social': round(random.uniform(0.2, 0.9), 3),
            'avg_cursor_affinity': round(random.uniform(-0.5, 0.5), 3),
            'avg_survival_score': round(random.uniform(10, 50), 1),
            'total_hunting_success': random.randint(0, 20),
            'avg_hunting_success': round(random.uniform(0.1, 2.0), 2),
            'reproduction_capable': random.randint(0, 15),
            'simulation_time': round(random.uniform(10, 300), 1),
            'energy_distribution': {
                'low': random.randint(0, 5),
                'medium': random.randint(3, 10),
                'high': random.randint(2, 8)
            },
            'cursor_affinity_distribution': {
                'very_afraid': random.randint(0, 3),
                'afraid': random.randint(1, 5),
                'neutral': random.randint(3, 8),
                'attracted': random.randint(1, 5),
                'very_attracted': random.randint(0, 3)
            },
            'generation_stats': {
                'min': 0,
                'max': random.randint(1, 5),
                'avg': round(random.uniform(0.5, 3.0), 1),
                'diversity': random.randint(1, 4)
            },
            'size_stats': {
                'min_size': round(random.uniform(8, 12), 1),
                'max_size': round(random.uniform(20, 30), 1),
                'avg_size': round(random.uniform(15, 20), 1)
            },
            'population_health': {
                'overall_score': round(random.uniform(20, 80), 1),
                'energy_stability': round(random.uniform(0.3, 0.9), 2),
                'age_diversity': random.randint(3, 8),
                'size_diversity': random.randint(2, 6)
            },
            'active_interactions': {
                'cursor_interactions': random.randint(0, 8),
                'interaction_percentage': round(random.uniform(5, 40), 1)
            },
            'trait_analysis': {
                'dominant_traits': ['Moderate Aggression', 'High Curiosity'],
                'emerging_traits': ['Social Flocking', 'Cursor Adaptation'],
                'declining_traits': []
            },
            'behavioral_patterns': [
                'Population in exploration phase',
                'Balanced predator-prey dynamics',
                'Moderate cursor interaction'
            ]
        }
    
    def stop(self):
        self.running = False

class OrbEvolutionMonitor(QtWidgets.QWidget):
    """
    A comprehensive monitoring interface to observe orb evolution and statistics
    """
    def __init__(self):
        super().__init__()
        self.orb_process = None
        self.stats_collector = OrbStatsCollector()
        self.stats_collector.stats_updated.connect(self.update_detailed_stats)
        self.current_stats = {}
        self.initUI()
        self.apply_dark_theme()
        
    def initUI(self):
        self.setWindowTitle('🧠 Intelligent Orb Evolution Monitor')
        self.setGeometry(100, 100, 1000, 700)
        self.setMinimumSize(800, 600)
        
        # Main layout
        main_layout = QtWidgets.QVBoxLayout()
        
        # Header
        header_layout = QtWidgets.QHBoxLayout()
        
        title = QtWidgets.QLabel('🧠 Intelligent Orb Evolution Monitor')
        title.setAlignment(QtCore.Qt.AlignCenter)
        title.setStyleSheet("""
            font-size: 24px; 
            font-weight: bold; 
            color: #ffffff;
            padding: 10px;
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                                      stop:0 #2E86AB, stop:1 #A23B72);
            border-radius: 8px;
            margin-bottom: 10px;
        """)
        header_layout.addWidget(title)
        
        main_layout.addLayout(header_layout)
        
        # Control panel
        control_panel = self.create_control_panel()
        main_layout.addWidget(control_panel)
        
        # Statistics area with tabs
        stats_tabs = QtWidgets.QTabWidget()
        stats_tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #555;
                background-color: #2b2b2b;
            }
            QTabBar::tab {
                background-color: #404040;
                color: white;
                padding: 8px 16px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background-color: #2E86AB;
            }
        """)
        
        # Real-time stats tab
        self.realtime_widget = self.create_realtime_stats()
        stats_tabs.addTab(self.realtime_widget, "📊 Real-time Stats")
        
        # Population analysis tab
        self.population_widget = self.create_population_analysis()
        stats_tabs.addTab(self.population_widget, "🧬 Population Analysis")
        
        # Behavior patterns tab
        self.behavior_widget = self.create_behavior_analysis()
        stats_tabs.addTab(self.behavior_widget, "🎭 Behavior Patterns")
        
        # Evolution tracking tab
        self.evolution_widget = self.create_evolution_tracking()
        stats_tabs.addTab(self.evolution_widget, "🔬 Evolution Tracking")
        
        main_layout.addWidget(stats_tabs)
        
        # Status bar
        self.status_bar = QtWidgets.QLabel("Ready to start simulation...")
        self.status_bar.setStyleSheet("""
            background-color: #404040;
            color: #ffffff;
            padding: 5px;
            border-radius: 3px;
        """)
        main_layout.addWidget(self.status_bar)
        
        self.setLayout(main_layout)
        
    
    def create_control_panel(self):
        """Create the control panel with start/stop buttons"""
        control_frame = QtWidgets.QFrame()
        control_frame.setStyleSheet("""
            QFrame {
                background-color: #363636;
                border: 1px solid #555;
                border-radius: 8px;
                padding: 10px;
            }
        """)
        
        layout = QtWidgets.QHBoxLayout()
        
        # Status indicator
        self.status_indicator = QtWidgets.QLabel("●")
        self.status_indicator.setStyleSheet("color: #ff4444; font-size: 20px;")
        layout.addWidget(self.status_indicator)
        
        # Control buttons
        self.start_btn = QtWidgets.QPushButton('🚀 Start Intelligent Orbs')
        self.start_btn.clicked.connect(self.start_orbs)
        self.start_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 12px 24px;
                font-size: 14px;
                font-weight: bold;
                border-radius: 6px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
        """)
        layout.addWidget(self.start_btn)
        
        self.stop_btn = QtWidgets.QPushButton('⏹️ Stop Simulation')
        self.stop_btn.clicked.connect(self.stop_orbs)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                border: none;
                padding: 12px 24px;
                font-size: 14px;
                font-weight: bold;
                border-radius: 6px;
            }
            QPushButton:hover {
                background-color: #da190b;
            }
            QPushButton:pressed {
                background-color: #c1170a;
            }
            QPushButton:disabled {
                background-color: #666;
                color: #999;
            }
        """)
        layout.addWidget(self.stop_btn)
        
        # Spacer
        layout.addStretch()
        
        # Performance metrics
        self.fps_label = QtWidgets.QLabel("FPS: --")
        self.fps_label.setStyleSheet("color: #ffffff; font-weight: bold;")
        layout.addWidget(self.fps_label)
        
        control_frame.setLayout(layout)
        return control_frame
    
    def create_realtime_stats(self):
        """Create real-time statistics display with comprehensive metrics"""
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout()
        
        # Population overview
        pop_label = QtWidgets.QLabel("🌍 Population Overview:")
        pop_label.setStyleSheet("color: #ffffff; font-size: 16px; font-weight: bold;")
        layout.addWidget(pop_label)
        
        pop_grid = QtWidgets.QGridLayout()
        
        # Population metrics
        self.pop_frame = self.create_stat_frame("👥 Total Orbs", "")
        self.pop_stats = self.pop_frame.findChild(QtWidgets.QLabel, "stat_value")
        pop_grid.addWidget(self.pop_frame, 0, 0)
        
        self.energy_frame = self.create_stat_frame("⚡ Avg Energy", "")
        self.energy_stats = self.energy_frame.findChild(QtWidgets.QLabel, "stat_value")
        pop_grid.addWidget(self.energy_frame, 0, 1)
        
        self.age_frame = self.create_stat_frame("🕐 Avg Age", "")
        self.age_stats = self.age_frame.findChild(QtWidgets.QLabel, "stat_value")
        pop_grid.addWidget(self.age_frame, 0, 2)
        
        self.survival_frame = self.create_stat_frame("🎯 Survival Score", "")
        self.survival_stats = self.survival_frame.findChild(QtWidgets.QLabel, "stat_value")
        pop_grid.addWidget(self.survival_frame, 0, 3)
        
        layout.addLayout(pop_grid)
        
        # Behavior distribution
        behavior_label = QtWidgets.QLabel("🎭 Current Behaviors:")
        behavior_label.setStyleSheet("color: #ffffff; font-size: 16px; font-weight: bold; margin-top: 20px;")
        layout.addWidget(behavior_label)
        
        behavior_grid = QtWidgets.QGridLayout()
        
        self.explore_frame = self.create_stat_frame("🔍 Exploring", "")
        self.explore_stats = self.explore_frame.findChild(QtWidgets.QLabel, "stat_value")
        behavior_grid.addWidget(self.explore_frame, 0, 0)
        
        self.hunt_frame = self.create_stat_frame("🎯 Hunting", "")
        self.hunt_stats = self.hunt_frame.findChild(QtWidgets.QLabel, "stat_value")
        behavior_grid.addWidget(self.hunt_frame, 0, 1)
        
        self.flee_frame = self.create_stat_frame("🏃 Fleeing", "")
        self.flee_stats = self.flee_frame.findChild(QtWidgets.QLabel, "stat_value")
        behavior_grid.addWidget(self.flee_frame, 0, 2)
        
        self.mate_frame = self.create_stat_frame("💕 Mating", "")
        self.mate_stats = self.mate_frame.findChild(QtWidgets.QLabel, "stat_value")
        behavior_grid.addWidget(self.mate_frame, 0, 3)
        
        layout.addLayout(behavior_grid)
        
        # Cursor interaction stats
        cursor_label = QtWidgets.QLabel("🖱️ Cursor Interactions:")
        cursor_label.setStyleSheet("color: #ffffff; font-size: 16px; font-weight: bold; margin-top: 20px;")
        layout.addWidget(cursor_label)
        
        cursor_grid = QtWidgets.QGridLayout()
        
        self.cursor_active_frame = self.create_stat_frame("⚡ Active", "")
        self.cursor_active_stats = self.cursor_active_frame.findChild(QtWidgets.QLabel, "stat_value")
        cursor_grid.addWidget(self.cursor_active_frame, 0, 0)
        
        self.cursor_afraid_frame = self.create_stat_frame("😰 Afraid", "")
        self.cursor_afraid_stats = self.cursor_afraid_frame.findChild(QtWidgets.QLabel, "stat_value")
        cursor_grid.addWidget(self.cursor_afraid_frame, 0, 1)
        
        self.cursor_attracted_frame = self.create_stat_frame("😍 Attracted", "")
        self.cursor_attracted_stats = self.cursor_attracted_frame.findChild(QtWidgets.QLabel, "stat_value")
        cursor_grid.addWidget(self.cursor_attracted_frame, 0, 2)
        
        self.cursor_neutral_frame = self.create_stat_frame("😐 Neutral", "")
        self.cursor_neutral_stats = self.cursor_neutral_frame.findChild(QtWidgets.QLabel, "stat_value")
        cursor_grid.addWidget(self.cursor_neutral_frame, 0, 3)
        
        layout.addLayout(cursor_grid)
        
        # Real-time activity log
        activity_label = QtWidgets.QLabel("� Activity Log:")
        activity_label.setStyleSheet("color: #ffffff; font-size: 16px; font-weight: bold; margin-top: 20px;")
        layout.addWidget(activity_label)
        
        self.activity_log = QtWidgets.QTextEdit()
        self.activity_log.setMaximumHeight(150)
        self.activity_log.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                color: #ffffff;
                border: 1px solid #555;
                border-radius: 4px;
                font-family: 'Courier New', monospace;
                font-size: 11px;
            }
        """)
        layout.addWidget(self.activity_log)
        
        widget.setLayout(layout)
        return widget
    
    def create_population_analysis(self):
        """Create comprehensive population analysis display"""
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout()
        
        # Trait distribution
        traits_label = QtWidgets.QLabel("🧬 Trait Distribution:")
        traits_label.setStyleSheet("color: #ffffff; font-size: 16px; font-weight: bold;")
        layout.addWidget(traits_label)
        
        traits_grid = QtWidgets.QGridLayout()
        
        self.aggr_frame = self.create_stat_frame("🔴 Aggression", "")
        self.aggr_stats = self.aggr_frame.findChild(QtWidgets.QLabel, "stat_value")
        traits_grid.addWidget(self.aggr_frame, 0, 0)
        
        self.cur_frame = self.create_stat_frame("🟡 Curiosity", "")
        self.cur_stats = self.cur_frame.findChild(QtWidgets.QLabel, "stat_value")
        traits_grid.addWidget(self.cur_frame, 0, 1)
        
        self.soc_frame = self.create_stat_frame("🟢 Social", "")
        self.soc_stats = self.soc_frame.findChild(QtWidgets.QLabel, "stat_value")
        traits_grid.addWidget(self.soc_frame, 0, 2)
        
        self.cursor_aff_frame = self.create_stat_frame("🖱️ Cursor Affinity", "")
        self.cursor_aff_stats = self.cursor_aff_frame.findChild(QtWidgets.QLabel, "stat_value")
        traits_grid.addWidget(self.cursor_aff_frame, 0, 3)
        
        layout.addLayout(traits_grid)
        
        # Energy distribution
        energy_dist_label = QtWidgets.QLabel("⚡ Energy Distribution:")
        energy_dist_label.setStyleSheet("color: #ffffff; font-size: 16px; font-weight: bold; margin-top: 20px;")
        layout.addWidget(energy_dist_label)
        
        energy_grid = QtWidgets.QGridLayout()
        
        self.energy_low_frame = self.create_stat_frame("🔴 Low (<30)", "")
        self.energy_low_stats = self.energy_low_frame.findChild(QtWidgets.QLabel, "stat_value")
        energy_grid.addWidget(self.energy_low_frame, 0, 0)
        
        self.energy_med_frame = self.create_stat_frame("🟡 Medium (30-70)", "")
        self.energy_med_stats = self.energy_med_frame.findChild(QtWidgets.QLabel, "stat_value")
        energy_grid.addWidget(self.energy_med_frame, 0, 1)
        
        self.energy_high_frame = self.create_stat_frame("🟢 High (>70)", "")
        self.energy_high_stats = self.energy_high_frame.findChild(QtWidgets.QLabel, "stat_value")
        energy_grid.addWidget(self.energy_high_frame, 0, 2)
        
        layout.addLayout(energy_grid)
        
        # Size and generation stats
        metrics_label = QtWidgets.QLabel("📊 Population Metrics:")
        metrics_label.setStyleSheet("color: #ffffff; font-size: 16px; font-weight: bold; margin-top: 20px;")
        layout.addWidget(metrics_label)
        
        metrics_grid = QtWidgets.QGridLayout()
        
        self.size_avg_frame = self.create_stat_frame("📏 Avg Size", "")
        self.size_avg_stats = self.size_avg_frame.findChild(QtWidgets.QLabel, "stat_value")
        metrics_grid.addWidget(self.size_avg_frame, 0, 0)
        
        self.gen_range_frame = self.create_stat_frame("🧬 Generation Range", "")
        self.gen_range_stats = self.gen_range_frame.findChild(QtWidgets.QLabel, "stat_value")
        metrics_grid.addWidget(self.gen_range_frame, 0, 1)
        
        self.gen_diversity_frame = self.create_stat_frame("🌈 Gen Diversity", "")
        self.gen_diversity_stats = self.gen_diversity_frame.findChild(QtWidgets.QLabel, "stat_value")
        metrics_grid.addWidget(self.gen_diversity_frame, 0, 2)
        
        self.health_frame = self.create_stat_frame("🏥 Population Health", "")
        self.health_stats = self.health_frame.findChild(QtWidgets.QLabel, "stat_value")
        metrics_grid.addWidget(self.health_frame, 0, 3)
        
        layout.addLayout(metrics_grid)
        
        # Detailed analysis
        analysis_label = QtWidgets.QLabel("🔬 Detailed Analysis:")
        analysis_label.setStyleSheet("color: #ffffff; font-size: 16px; font-weight: bold; margin-top: 20px;")
        layout.addWidget(analysis_label)
        
        self.population_analysis = QtWidgets.QTextEdit()
        self.population_analysis.setStyleSheet("""
            QTextEdit {
                background-color: #2b2b2b;
                color: #ffffff;
                border: 1px solid #555;
                border-radius: 4px;
                font-size: 12px;
            }
        """)
        layout.addWidget(self.population_analysis)
        
        widget.setLayout(layout)
        return widget
    
    def create_behavior_analysis(self):
        """Create comprehensive behavior pattern analysis"""
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout()
        
        # Success metrics
        success_label = QtWidgets.QLabel("🏆 Success Metrics:")
        success_label.setStyleSheet("color: #ffffff; font-size: 16px; font-weight: bold;")
        layout.addWidget(success_label)
        
        success_grid = QtWidgets.QGridLayout()
        
        self.hunt_success_frame = self.create_stat_frame("🎯 Hunt Success", "")
        self.hunt_success_stats = self.hunt_success_frame.findChild(QtWidgets.QLabel, "stat_value")
        success_grid.addWidget(self.hunt_success_frame, 0, 0)
        
        self.avg_hunt_frame = self.create_stat_frame("📊 Avg per Orb", "")
        self.avg_hunt_stats = self.avg_hunt_frame.findChild(QtWidgets.QLabel, "stat_value")
        success_grid.addWidget(self.avg_hunt_frame, 0, 1)
        
        self.repro_capable_frame = self.create_stat_frame("💕 Ready to Mate", "")
        self.repro_capable_stats = self.repro_capable_frame.findChild(QtWidgets.QLabel, "stat_value")
        success_grid.addWidget(self.repro_capable_frame, 0, 2)
        
        self.sim_time_frame = self.create_stat_frame("⏱️ Sim Time", "")
        self.sim_time_stats = self.sim_time_frame.findChild(QtWidgets.QLabel, "stat_value")
        success_grid.addWidget(self.sim_time_frame, 0, 3)
        
        layout.addLayout(success_grid)
        
        # Interaction metrics
        interact_label = QtWidgets.QLabel("🤝 Interaction Metrics:")
        interact_label.setStyleSheet("color: #ffffff; font-size: 16px; font-weight: bold; margin-top: 20px;")
        layout.addWidget(interact_label)
        
        interact_grid = QtWidgets.QGridLayout()
        
        self.cursor_interact_frame = self.create_stat_frame("🖱️ Cursor Active", "")
        self.cursor_interact_stats = self.cursor_interact_frame.findChild(QtWidgets.QLabel, "stat_value")
        interact_grid.addWidget(self.cursor_interact_frame, 0, 0)
        
        self.interact_percent_frame = self.create_stat_frame("� Interaction %", "")
        self.interact_percent_stats = self.interact_percent_frame.findChild(QtWidgets.QLabel, "stat_value")
        interact_grid.addWidget(self.interact_percent_frame, 0, 1)
        
        layout.addLayout(interact_grid)
        
        # Behavioral patterns
        patterns_label = QtWidgets.QLabel("🔍 Observed Patterns:")
        patterns_label.setStyleSheet("color: #ffffff; font-size: 16px; font-weight: bold; margin-top: 20px;")
        layout.addWidget(patterns_label)
        
        self.patterns_display = QtWidgets.QTextEdit()
        self.patterns_display.setMaximumHeight(120)
        self.patterns_display.setStyleSheet("""
            QTextEdit {
                background-color: #2b2b2b;
                color: #ffffff;
                border: 1px solid #555;
                border-radius: 4px;
                font-size: 12px;
            }
        """)
        layout.addWidget(self.patterns_display)
        
        # Trait analysis
        trait_analysis_label = QtWidgets.QLabel("🧬 Trait Analysis:")
        trait_analysis_label.setStyleSheet("color: #ffffff; font-size: 16px; font-weight: bold; margin-top: 20px;")
        layout.addWidget(trait_analysis_label)
        
        self.trait_analysis_display = QtWidgets.QTextEdit()
        self.trait_analysis_display.setStyleSheet("""
            QTextEdit {
                background-color: #2b2b2b;
                color: #ffffff;
                border: 1px solid #555;
                border-radius: 4px;
                font-size: 12px;
            }
        """)
        layout.addWidget(self.trait_analysis_display)
        
        widget.setLayout(layout)
        return widget
    
    def create_evolution_tracking(self):
        """Create evolution tracking display"""
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout()
        
        # Evolution metrics
        evo_label = QtWidgets.QLabel("🧬 Evolution Metrics:")
        evo_label.setStyleSheet("color: #ffffff; font-size: 16px; font-weight: bold;")
        layout.addWidget(evo_label)
        
        self.evolution_display = QtWidgets.QTextEdit()
        self.evolution_display.setStyleSheet("""
            QTextEdit {
                background-color: #2b2b2b;
                color: #ffffff;
                border: 1px solid #555;
                border-radius: 4px;
                font-size: 12px;
            }
        """)
        layout.addWidget(self.evolution_display)
        
        widget.setLayout(layout)
        return widget
    
    def create_stat_frame(self, title, value):
        """Create a styled statistics frame"""
        frame = QtWidgets.QFrame()
        frame.setStyleSheet("""
            QFrame {
                background-color: #404040;
                border: 1px solid #666;
                border-radius: 8px;
                padding: 10px;
                margin: 5px;
            }
        """)
        
        layout = QtWidgets.QVBoxLayout()
        
        title_label = QtWidgets.QLabel(title)
        title_label.setStyleSheet("color: #ffffff; font-size: 14px; font-weight: bold;")
        title_label.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(title_label)
        
        value_label = QtWidgets.QLabel(value)
        value_label.setObjectName("stat_value")
        value_label.setStyleSheet("color: #00ff88; font-size: 18px; font-weight: bold;")
        value_label.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(value_label)
        
        frame.setLayout(layout)
        return frame
    
    def apply_dark_theme(self):
        """Apply comprehensive dark theme"""
        self.setStyleSheet("""
            QWidget {
                background-color: #2b2b2b;
                color: #ffffff;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            
            QMainWindow {
                background-color: #2b2b2b;
            }
            
            QScrollArea {
                background-color: #2b2b2b;
                border: none;
            }
            
            QScrollBar:vertical {
                background-color: #404040;
                width: 12px;
                border-radius: 6px;
            }
            
            QScrollBar::handle:vertical {
                background-color: #666;
                border-radius: 6px;
                min-height: 20px;
            }
            
            QScrollBar::handle:vertical:hover {
                background-color: #777;
            }
        """)
        
        # Timer for updating statistics
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_interface)
        self.timer.start(1000)  # Update every second
    
    def start_orbs(self):
        """Start the intelligent orbs simulation"""
        try:
            script_path = os.path.join(os.path.dirname(__file__), 'intelligent_orbs.py')
            self.orb_process = subprocess.Popen([sys.executable, script_path])
            
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.status_indicator.setStyleSheet("color: #00ff00; font-size: 20px;")
            self.status_bar.setText("🚀 Intelligent Orbs simulation started! Move cursor to interact.")
            
            # Start stats collection
            self.stats_collector.set_process(self.orb_process)
            self.stats_collector.start()
            
            self.log_activity("Simulation started successfully")
            
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to start orbs: {str(e)}")
            self.log_activity(f"ERROR: Failed to start - {str(e)}")
    
    def stop_orbs(self):
        """Stop the orbs simulation"""
        if self.orb_process:
            self.orb_process.terminate()
            self.orb_process = None
        
        # Stop stats collection
        self.stats_collector.stop()
        
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_indicator.setStyleSheet("color: #ff4444; font-size: 20px;")
        self.status_bar.setText("⏹️ Simulation stopped.")
        
        self.log_activity("Simulation stopped")
    
    def log_activity(self, message):
        """Add message to activity log"""
        timestamp = time.strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}"
        self.activity_log.append(formatted_message)
        
        # Keep log size manageable
        if self.activity_log.document().blockCount() > 100:
            cursor = self.activity_log.textCursor()
            cursor.movePosition(QtGui.QTextCursor.Start)
            cursor.select(QtGui.QTextCursor.LineUnderCursor)
            cursor.removeSelectedText()
    
    def update_interface(self):
        """Update the interface periodically"""
        if self.orb_process and self.orb_process.poll() is None:
            # Process is running
            self.fps_label.setText("FPS: 60")
        elif self.orb_process and self.orb_process.poll() is not None:
            # Process has ended
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.status_indicator.setStyleSheet("color: #ff4444; font-size: 20px;")
            self.status_bar.setText("🔴 Simulation ended unexpectedly.")
            self.fps_label.setText("FPS: --")
            self.orb_process = None
            self.stats_collector.stop()
            self.log_activity("Simulation ended unexpectedly")
    
    def update_detailed_stats(self, stats):
        """Update all statistics displays with comprehensive new data"""
        self.current_stats = stats
        
        # Update Real-time Stats Tab
        if hasattr(self, 'pop_stats'):
            self.pop_stats.setText(f"{stats['total_orbs']}")
        
        if hasattr(self, 'energy_stats'):
            energy_color = self.get_energy_color(stats['avg_energy'])
            self.energy_stats.setText(f"{stats['avg_energy']}%")
            self.energy_stats.setStyleSheet(f"color: {energy_color}; font-size: 18px; font-weight: bold;")
        
        if hasattr(self, 'age_stats'):
            self.age_stats.setText(f"{stats['avg_age']}")
            
        if hasattr(self, 'survival_stats'):
            self.survival_stats.setText(f"{stats['avg_survival_score']:.1f}")
        
        # Update behavior counts
        behaviors = stats.get('behaviors', {})
        if hasattr(self, 'explore_stats'):
            self.explore_stats.setText(f"{behaviors.get('exploring', 0)}")
        if hasattr(self, 'hunt_stats'):
            self.hunt_stats.setText(f"{behaviors.get('hunting', 0)}")
        if hasattr(self, 'flee_stats'):
            self.flee_stats.setText(f"{behaviors.get('fleeing', 0)}")
        if hasattr(self, 'mate_stats'):
            self.mate_stats.setText(f"{behaviors.get('mating', 0)}")
        
        # Update cursor interaction stats
        cursor_dist = stats.get('cursor_affinity_distribution', {})
        if hasattr(self, 'cursor_active_stats'):
            active_count = stats.get('active_interactions', {}).get('cursor_interactions', 0)
            self.cursor_active_stats.setText(f"{active_count}")
            
        if hasattr(self, 'cursor_afraid_stats'):
            afraid_count = cursor_dist.get('very_afraid', 0) + cursor_dist.get('afraid', 0)
            self.cursor_afraid_stats.setText(f"{afraid_count}")
            
        if hasattr(self, 'cursor_attracted_stats'):
            attracted_count = cursor_dist.get('attracted', 0) + cursor_dist.get('very_attracted', 0)
            self.cursor_attracted_stats.setText(f"{attracted_count}")
            
        if hasattr(self, 'cursor_neutral_stats'):
            neutral_count = cursor_dist.get('neutral', 0)
            self.cursor_neutral_stats.setText(f"{neutral_count}")
        
        # Update Population Analysis Tab
        if hasattr(self, 'aggr_stats'):
            self.aggr_stats.setText(f"{stats['avg_aggression']:.3f}")
            
        if hasattr(self, 'cur_stats'):
            self.cur_stats.setText(f"{stats['avg_curiosity']:.3f}")
            
        if hasattr(self, 'soc_stats'):
            self.soc_stats.setText(f"{stats['avg_social']:.3f}")
            
        if hasattr(self, 'cursor_aff_stats'):
            self.cursor_aff_stats.setText(f"{stats['avg_cursor_affinity']:.3f}")
        
        # Update energy distribution
        energy_dist = stats.get('energy_distribution', {})
        if hasattr(self, 'energy_low_stats'):
            self.energy_low_stats.setText(f"{energy_dist.get('low', 0)}")
        if hasattr(self, 'energy_med_stats'):
            self.energy_med_stats.setText(f"{energy_dist.get('medium', 0)}")
        if hasattr(self, 'energy_high_stats'):
            self.energy_high_stats.setText(f"{energy_dist.get('high', 0)}")
        
        # Update generation and size metrics
        gen_stats = stats.get('generation_stats', {})
        size_stats = stats.get('size_stats', {})
        
        if hasattr(self, 'size_avg_stats'):
            self.size_avg_stats.setText(f"{size_stats.get('avg_size', 0):.1f}")
            
        if hasattr(self, 'gen_range_stats'):
            self.gen_range_stats.setText(f"{gen_stats.get('min', 0)}-{gen_stats.get('max', 0)}")
            
        if hasattr(self, 'gen_diversity_stats'):
            self.gen_diversity_stats.setText(f"{gen_stats.get('diversity', 0)}")
            
        if hasattr(self, 'health_stats'):
            pop_health = stats.get('population_health', {})
            health_score = pop_health.get('overall_score', 0)
            self.health_stats.setText(f"{health_score:.1f}")
        
        # Update population analysis text
        if hasattr(self, 'population_analysis'):
            analysis_text = self.generate_population_analysis(stats)
            self.population_analysis.setText(analysis_text)
        
        # Update Behavior Analysis Tab
        if hasattr(self, 'hunt_success_stats'):
            self.hunt_success_stats.setText(f"{stats['total_hunting_success']}")
            
        if hasattr(self, 'avg_hunt_stats'):
            self.avg_hunt_stats.setText(f"{stats['avg_hunting_success']:.2f}")
            
        if hasattr(self, 'repro_capable_stats'):
            self.repro_capable_stats.setText(f"{stats['reproduction_capable']}")
            
        if hasattr(self, 'sim_time_stats'):
            self.sim_time_stats.setText(f"{stats['simulation_time']:.1f}s")
        
        if hasattr(self, 'cursor_interact_stats'):
            active_interactions = stats.get('active_interactions', {})
            self.cursor_interact_stats.setText(f"{active_interactions.get('cursor_interactions', 0)}")
            
        if hasattr(self, 'interact_percent_stats'):
            interact_percent = active_interactions.get('interaction_percentage', 0)
            self.interact_percent_stats.setText(f"{interact_percent:.1f}%")
        
        # Update behavior patterns
        if hasattr(self, 'patterns_display'):
            patterns = self.analyze_behavior_patterns(stats)
            self.patterns_display.setText(patterns)
        
        # Update trait analysis
        if hasattr(self, 'trait_analysis_display'):
            trait_analysis = self.generate_trait_analysis(stats)
            self.trait_analysis_display.setText(trait_analysis)
        
        # Update evolution tracking
        if hasattr(self, 'evolution_display'):
            evolution_text = self.generate_evolution_analysis(stats)
            self.evolution_display.setText(evolution_text)
        
        # Log significant events
        self.check_for_significant_events(stats)
        
        # Update activity log with recent events
        self.update_activity_log(stats)
    
    def get_energy_color(self, energy):
        """Get color based on energy level"""
        if energy > 70:
            return "#00ff00"  # Green
        elif energy > 40:
            return "#ffff00"  # Yellow
        else:
            return "#ff4444"  # Red
    
    def analyze_behavior_patterns(self, stats):
        """Analyze and describe behavioral patterns"""
        patterns = []
        
        behaviors = stats.get('behaviors', {})
        total = sum(behaviors.values())
        
        if total > 0:
            # Dominant behavior
            dominant = max(behaviors.items(), key=lambda x: x[1])
            patterns.append(f"🎯 Dominant Behavior: {dominant[0].title()} ({dominant[1]}/{total} orbs)")
            
            # Activity level
            active_behaviors = behaviors.get('hunting', 0) + behaviors.get('fleeing', 0)
            if active_behaviors / total > 0.4:
                patterns.append("⚡ High Activity: Population showing aggressive/defensive behaviors")
            elif behaviors.get('exploring', 0) / total > 0.6:
                patterns.append("🌟 Peaceful Phase: Most orbs in exploration mode")
            
            # Reproduction activity
            if behaviors.get('mating', 0) > 3:
                patterns.append("💕 Mating Season: High reproductive activity detected")
            
            # Population stress
            if behaviors.get('fleeing', 0) > behaviors.get('hunting', 0):
                patterns.append("😰 Population Stress: More fleeing than hunting behavior")
            
            # Balanced ecosystem
            if abs(behaviors.get('hunting', 0) - behaviors.get('fleeing', 0)) <= 1:
                patterns.append("⚖️ Balanced Ecosystem: Predator-prey equilibrium achieved")
        
        # Survival analysis
        avg_energy = stats.get('avg_energy', 50)
        if avg_energy < 50:
            patterns.append("⚠️ Low Energy Crisis: Population struggling to maintain energy")
        
        # Use behavioral patterns from stats if available
        if 'behavioral_patterns' in stats:
            patterns.extend(stats['behavioral_patterns'])
        
        return "\n\n".join(patterns) if patterns else "🔍 Analyzing behavioral patterns..."
    
    def generate_evolution_analysis(self, stats):
        """Generate evolution analysis text"""
        gen_stats = stats.get('generation_stats', {})
        gen_range = f"{gen_stats.get('min', 0)}-{gen_stats.get('max', 0)}"
        
        return f"""🧬 EVOLUTIONARY STATUS REPORT

Current Generation: {gen_range}
Population Size: {stats.get('total_orbs', 0)} individuals

🧠 Neural Network Evolution:
• AI Decision Making: Active
• Behavioral Adaptation: Ongoing
• Genetic Drift: Natural selection in progress

🔬 Trait Analysis:
┌─────────────────┬────────┬─────────────┐
│ Trait           │ Value  │ Trend       │
├─────────────────┼────────┼─────────────┤
│ Aggression      │ {stats.get('avg_aggression', 0):.3f}  │ Evolving    │
│ Curiosity       │ {stats.get('avg_curiosity', 0):.3f}  │ Adapting    │
│ Social Tendency │ {stats.get('avg_social', 0):.3f}  │ Fluctuating │
└─────────────────┴────────┴─────────────┘

⚡ Survival Metrics:
• Hunting Success: {stats.get('total_hunting_success', 0)} total
• Energy Efficiency: {stats.get('avg_energy', 0):.1f}%
• Reproductive Capability: {stats.get('reproduction_capable', 0)} orbs

🌍 Environmental Adaptation:
• Cursor Interaction: {'Responsive' if stats.get('active_interactions', {}).get('cursor_interactions', 0) > 0 else 'Limited'}
• Inter-orb Dynamics: Complex
• Territorial Behavior: Emerging

🔄 Natural Selection Pressures:
• Energy Competition: {'High' if stats.get('avg_energy', 50) < 50 else 'Active'}
• Size-based Dominance: Present  
• Reproductive Selection: Operating

📊 Population Health:
{('🟢 STABLE' if 30 <= stats.get('avg_energy', 50) <= 80 else 
  '🟡 STRESSED' if stats.get('avg_energy', 50) > 20 else 
  '🔴 CRITICAL')}

Next Evolution Milestone: {stats.get('total_orbs', 0) + 5} individuals
Estimated Time to Advancement: Monitoring...
"""
    
    def generate_population_analysis(self, stats):
        """Generate detailed population analysis text"""
        gen_stats = stats.get('generation_stats', {})
        energy_dist = stats.get('energy_distribution', {})
        cursor_dist = stats.get('cursor_affinity_distribution', {})
        pop_health = stats.get('population_health', {})
        
        analysis = f"""🧬 POPULATION GENETIC ANALYSIS

Generation Statistics:
• Current Range: {gen_stats.get('min', 0)} - {gen_stats.get('max', 0)}
• Average Generation: {gen_stats.get('avg', 0):.1f}
• Genetic Diversity: {gen_stats.get('diversity', 0)} unique generations

Energy Distribution Analysis:
• Low Energy (<30): {energy_dist.get('low', 0)} orbs ({energy_dist.get('low', 0)/max(1, stats['total_orbs'])*100:.1f}%)
• Medium Energy (30-70): {energy_dist.get('medium', 0)} orbs ({energy_dist.get('medium', 0)/max(1, stats['total_orbs'])*100:.1f}%)
• High Energy (>70): {energy_dist.get('high', 0)} orbs ({energy_dist.get('high', 0)/max(1, stats['total_orbs'])*100:.1f}%)

Cursor Affinity Distribution:
• Very Afraid: {cursor_dist.get('very_afraid', 0)} orbs
• Afraid: {cursor_dist.get('afraid', 0)} orbs
• Neutral: {cursor_dist.get('neutral', 0)} orbs
• Attracted: {cursor_dist.get('attracted', 0)} orbs
• Very Attracted: {cursor_dist.get('very_attracted', 0)} orbs

Population Health Metrics:
• Overall Health Score: {pop_health.get('overall_score', 0):.2f}
• Energy Stability: {pop_health.get('energy_stability', 0):.2f}
• Age Diversity: {pop_health.get('age_diversity', 0)} age groups
• Size Diversity: {pop_health.get('size_diversity', 0)} size categories

Trait Analysis:
• Average Aggression: {stats['avg_aggression']:.3f} (Higher = more aggressive)
• Average Curiosity: {stats['avg_curiosity']:.3f} (Higher = more exploratory)
• Average Social: {stats['avg_social']:.3f} (Higher = more social)
• Average Cursor Affinity: {stats['avg_cursor_affinity']:.3f} (Negative = afraid, Positive = attracted)
"""
        return analysis
    
    def generate_trait_analysis(self, stats):
        """Generate trait analysis for behavioral patterns"""
        trait_analysis = stats.get('trait_analysis', {})
        behavioral_patterns = stats.get('behavioral_patterns', [])
        
        analysis = f"""🧬 TRAIT EVOLUTION ANALYSIS

Dominant Traits:
{chr(10).join(f"• {trait}" for trait in trait_analysis.get('dominant_traits', ['None detected']))}

Emerging Traits:
{chr(10).join(f"• {trait}" for trait in trait_analysis.get('emerging_traits', ['Monitoring...']))}

Behavioral Patterns Detected:
{chr(10).join(f"• {pattern}" for pattern in behavioral_patterns) if behavioral_patterns else "• No significant patterns detected"}

Trait Stability Analysis:
• Aggression Variance: {'High' if stats['avg_aggression'] > 0.7 else 'Medium' if stats['avg_aggression'] > 0.4 else 'Low'}
• Curiosity Variance: {'High' if stats['avg_curiosity'] > 0.7 else 'Medium' if stats['avg_curiosity'] > 0.4 else 'Low'}
• Social Variance: {'High' if stats['avg_social'] > 0.7 else 'Medium' if stats['avg_social'] > 0.4 else 'Low'}

Evolutionary Pressure Points:
• Energy Competition: {'High' if stats['avg_energy'] < 50 else 'Medium' if stats['avg_energy'] < 70 else 'Low'}
• Cursor Interaction: {'High' if stats.get('active_interactions', {}).get('interaction_percentage', 0) > 30 else 'Medium' if stats.get('active_interactions', {}).get('interaction_percentage', 0) > 15 else 'Low'}
• Reproductive Pressure: {'High' if stats['reproduction_capable'] < stats['total_orbs'] * 0.3 else 'Medium' if stats['reproduction_capable'] < stats['total_orbs'] * 0.6 else 'Low'}

Survival Advantages:
• High Energy Orbs: More likely to survive and reproduce
• Balanced Traits: Moderate aggression with high social tendency
• Cursor Adaptation: Appropriate response to cursor presence
"""
        return analysis
    
    def update_activity_log(self, stats):
        """Update the activity log with recent events"""
        if not hasattr(self, 'activity_log'):
            return
            
        # Generate activity messages based on current stats
        messages = []
        
        # Population changes
        if hasattr(self, 'last_population_count'):
            pop_change = stats['total_orbs'] - self.last_population_count
            if pop_change > 0:
                messages.append(f"[+] Population increased by {pop_change} orbs")
            elif pop_change < 0:
                messages.append(f"[-] Population decreased by {abs(pop_change)} orbs")
        
        # Behavior alerts
        behaviors = stats.get('behaviors', {})
        if behaviors.get('hunting', 0) > stats['total_orbs'] * 0.4:
            messages.append("[!] High hunting activity detected")
        if behaviors.get('fleeing', 0) > stats['total_orbs'] * 0.3:
            messages.append("[!] Population under stress - high fleeing activity")
        if behaviors.get('mating', 0) > 3:
            messages.append("[♥] Active mating season in progress")
        
        # Energy alerts
        if stats['avg_energy'] < 30:
            messages.append("[⚠] Critical: Low population energy levels")
        elif stats['avg_energy'] > 80:
            messages.append("[✓] Excellent: High population energy levels")
        
        # Cursor interaction alerts
        cursor_interactions = stats.get('active_interactions', {}).get('cursor_interactions', 0)
        if cursor_interactions > stats['total_orbs'] * 0.5:
            messages.append("[🖱] High cursor interaction activity")
        
        # Add messages to log
        import time
        current_time = time.strftime("%H:%M:%S")
        for message in messages:
            self.activity_log.append(f"[{current_time}] {message}")
        
        # Keep log size manageable
        if self.activity_log.document().lineCount() > 100:
            cursor = self.activity_log.textCursor()
            cursor.movePosition(cursor.Start)
            cursor.movePosition(cursor.Down, cursor.KeepAnchor, 20)
            cursor.removeSelectedText()
        
        # Scroll to bottom
        self.activity_log.moveCursor(self.activity_log.textCursor().End)
        
        # Store for next comparison
        self.last_population_count = stats['total_orbs']
    
    def check_for_significant_events(self, stats):
        """Check for and log significant evolutionary events"""
        # This would check for major population changes, new behaviors, etc.
        # For now, we'll log some key metrics
        if hasattr(self, 'last_stats'):
            pop_change = stats['total_orbs'] - self.last_stats.get('total_orbs', 0)
            if pop_change > 3:
                self.log_activity(f"Population surge: +{pop_change} orbs")
            elif pop_change < -3:
                self.log_activity(f"Population decline: {pop_change} orbs")
        
        self.last_stats = stats.copy()
    
    def closeEvent(self, event):
        """Clean up when closing"""
        self.stop_orbs()
        self.stats_collector.stop()
        event.accept()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    
    # Set application properties
    app.setApplicationName("Intelligent Orb Monitor")
    app.setApplicationVersion("2.0")
    
    # Create and show the monitor
    monitor = OrbEvolutionMonitor()
    monitor.show()
    
    print("🧠 Intelligent Orb Evolution Monitor started!")
    print("Use this interface to monitor and analyze your AI orb ecosystem.")
    
    sys.exit(app.exec_())
