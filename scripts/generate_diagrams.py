#!/usr/bin/env python3
"""
Generate architecture diagrams for the Italian Teacher multi-agent system.
Creates professional diagrams and exports them as PNG files.
"""

from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import ConnectionPatch, FancyBboxPatch

# Create output directory
output_dir = Path(__file__).parent.parent / "docs" / "diagrams"
output_dir.mkdir(exist_ok=True)

# Set up matplotlib for better-looking diagrams
plt.style.use("default")
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 10


def create_system_architecture_diagram():
    """Create the main system architecture diagram."""
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 12)
    ax.axis("off")

    # Define colors
    colors = {
        "user": "#FF6B6B",
        "coordinator": "#4ECDC4",
        "agents": "#45B7D1",
        "infrastructure": "#96CEB4",
        "storage": "#FECA57",
        "external": "#FF9FF3",
    }

    # User layer
    user_box = FancyBboxPatch(
        (1, 10),
        2,
        1.5,
        boxstyle="round,pad=0.1",
        facecolor=colors["user"],
        alpha=0.8,
        edgecolor="black",
        linewidth=2,
    )
    ax.add_patch(user_box)
    ax.text(2, 10.75, "User\nInterface", ha="center", va="center", fontweight="bold", fontsize=12)

    # Coordinator layer
    coord_box = FancyBboxPatch(
        (6, 9.5),
        4,
        2,
        boxstyle="round,pad=0.1",
        facecolor=colors["coordinator"],
        alpha=0.8,
        edgecolor="black",
        linewidth=2,
    )
    ax.add_patch(coord_box)
    ax.text(8, 10.5, "Coordinator Agent", ha="center", va="center", fontweight="bold", fontsize=14)
    ax.text(
        8,
        10,
        "• Session Management\n• Context Switching\n• Progress Tracking",
        ha="center",
        va="center",
        fontsize=10,
    )

    # Discovery Service
    discovery_box = FancyBboxPatch(
        (12, 10),
        3,
        1.5,
        boxstyle="round,pad=0.1",
        facecolor=colors["infrastructure"],
        alpha=0.8,
        edgecolor="black",
        linewidth=2,
    )
    ax.add_patch(discovery_box)
    ax.text(
        13.5,
        10.75,
        "Agent Discovery\nService",
        ha="center",
        va="center",
        fontweight="bold",
        fontsize=11,
    )

    # Agent Registry
    registry_box = FancyBboxPatch(
        (12, 7.5),
        3,
        1.5,
        boxstyle="round,pad=0.1",
        facecolor=colors["infrastructure"],
        alpha=0.8,
        edgecolor="black",
        linewidth=2,
    )
    ax.add_patch(registry_box)
    ax.text(13.5, 8.25, "Agent Registry", ha="center", va="center", fontweight="bold", fontsize=11)

    # Event Bus
    eventbus_box = FancyBboxPatch(
        (1, 7.5),
        3,
        1.5,
        boxstyle="round,pad=0.1",
        facecolor=colors["infrastructure"],
        alpha=0.8,
        edgecolor="black",
        linewidth=2,
    )
    ax.add_patch(eventbus_box)
    ax.text(2.5, 8.25, "Agent Event Bus", ha="center", va="center", fontweight="bold", fontsize=11)

    # Specialized Agents
    agents = [
        ("Marco\n(Conversation)", 2, 5.5),
        ("Professoressa\n(Grammar)", 5.5, 5.5),
        ("Nonna\n(Culture)", 9, 5.5),
        ("Lorenzo\n(General)", 12.5, 5.5),
    ]

    agent_boxes = []
    for name, x, y in agents:
        box = FancyBboxPatch(
            (x - 0.75, y - 0.75),
            1.5,
            1.5,
            boxstyle="round,pad=0.1",
            facecolor=colors["agents"],
            alpha=0.8,
            edgecolor="black",
            linewidth=2,
        )
        ax.add_patch(box)
        agent_boxes.append(box)
        ax.text(x, y, name, ha="center", va="center", fontweight="bold", fontsize=10)

    # Storage layer
    conv_state_box = FancyBboxPatch(
        (1, 3),
        3.5,
        1.5,
        boxstyle="round,pad=0.1",
        facecolor=colors["storage"],
        alpha=0.8,
        edgecolor="black",
        linewidth=2,
    )
    ax.add_patch(conv_state_box)
    ax.text(
        2.75,
        3.75,
        "Conversation State\nManager",
        ha="center",
        va="center",
        fontweight="bold",
        fontsize=11,
    )

    retention_box = FancyBboxPatch(
        (5.5, 3),
        3.5,
        1.5,
        boxstyle="round,pad=0.1",
        facecolor=colors["storage"],
        alpha=0.8,
        edgecolor="black",
        linewidth=2,
    )
    ax.add_patch(retention_box)
    ax.text(
        7.25,
        3.75,
        "Retention Policy\nManager",
        ha="center",
        va="center",
        fontweight="bold",
        fontsize=11,
    )

    # External Services
    llm_box = FancyBboxPatch(
        (11, 3),
        3,
        1.5,
        boxstyle="round,pad=0.1",
        facecolor=colors["external"],
        alpha=0.8,
        edgecolor="black",
        linewidth=2,
    )
    ax.add_patch(llm_box)
    ax.text(
        12.5,
        3.75,
        "LLM API\n(OpenAI/Claude)",
        ha="center",
        va="center",
        fontweight="bold",
        fontsize=11,
    )

    # Add arrows/connections
    connections = [
        # User to Coordinator
        ((3, 10.75), (6, 10.5)),
        # Coordinator to Discovery
        ((10, 10.5), (12, 10.75)),
        # Discovery to Registry
        ((13.5, 10), (13.5, 9)),
        # Coordinator to Event Bus
        ((6, 9.5), (4, 8.25)),
        # Event Bus to Agents
        ((2.5, 7.5), (2, 6.25)),
        ((3.5, 8), (5.5, 6.25)),
        ((4, 8.5), (9, 6.25)),
        ((4, 9), (12.5, 6.25)),
        # Coordinator to Storage
        ((7, 9.5), (2.75, 4.5)),
        ((9, 9.5), (7.25, 4.5)),
        # Agents to LLM
        ((9, 4.75), (11, 3.75)),
        ((12.5, 4.75), (12.5, 4.5)),
    ]

    for start, end in connections:
        arrow = ConnectionPatch(
            start,
            end,
            "data",
            "data",
            arrowstyle="->",
            shrinkA=5,
            shrinkB=5,
            mutation_scale=20,
            fc="black",
            alpha=0.7,
            linewidth=1.5,
        )
        ax.add_patch(arrow)

    # Title
    ax.text(
        8,
        11.5,
        "Italian Teacher Multi-Agent System Architecture",
        ha="center",
        va="center",
        fontweight="bold",
        fontsize=18,
    )

    # Add legend
    legend_elements = [
        patches.Patch(color=colors["user"], label="User Interface"),
        patches.Patch(color=colors["coordinator"], label="Coordination Layer"),
        patches.Patch(color=colors["agents"], label="Specialized Agents"),
        patches.Patch(color=colors["infrastructure"], label="Infrastructure"),
        patches.Patch(color=colors["storage"], label="Storage Layer"),
        patches.Patch(color=colors["external"], label="External Services"),
    ]
    ax.legend(handles=legend_elements, loc="lower left", bbox_to_anchor=(0, 0))

    plt.tight_layout()
    plt.savefig(output_dir / "system_architecture.png", dpi=300, bbox_inches="tight")
    plt.close()


def create_agent_selection_flow():
    """Create agent selection workflow diagram."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis("off")

    # Define process steps with positions
    steps = [
        ("User Message\nReceived", 2, 8.5, "#FF6B6B"),
        ("Analyze Message\nIntent", 6, 8.5, "#4ECDC4"),
        ("Context Switch\nNeeded?", 10, 8.5, "#FECA57"),
        ("Query Discovery\nService", 6, 6.5, "#96CEB4"),
        ("Find Available\nAgents", 10, 6.5, "#96CEB4"),
        ("Score Agents\nby Capability", 6, 4.5, "#45B7D1"),
        ("Select Best\nMatch", 10, 4.5, "#45B7D1"),
        ("Handoff to\nSelected Agent", 2, 2.5, "#FF9FF3"),
        ("Continue with\nCurrent Agent", 12, 2.5, "#FF9FF3"),
    ]

    # Draw process boxes
    for step, x, y, color in steps:
        box = FancyBboxPatch(
            (x - 1, y - 0.75),
            2,
            1.5,
            boxstyle="round,pad=0.1",
            facecolor=color,
            alpha=0.8,
            edgecolor="black",
            linewidth=1.5,
        )
        ax.add_patch(box)
        ax.text(x, y, step, ha="center", va="center", fontweight="bold", fontsize=10)

    # Add decision diamond - remove the overlapping text since it's already in the step
    diamond = patches.RegularPolygon(
        (10, 8.5),
        4,
        radius=1.2,
        orientation=np.pi / 4,
        facecolor="#FECA57",
        alpha=0.8,
        edgecolor="black",
        linewidth=1.5,
    )
    ax.add_patch(diamond)

    # Add arrows
    arrows = [
        ((3, 8.5), (5, 8.5)),  # User to Analyze
        ((7, 8.5), (8.8, 8.5)),  # Analyze to Decision
        ((10, 7.3), (6, 7.25)),  # Decision to Discovery (Yes)
        ((11.2, 8.5), (12, 3.25)),  # Decision to Continue (No)
        ((6, 5.75), (10, 5.75)),  # Discovery to Find
        ((10, 5.75), (6, 5.25)),  # Find to Score
        ((7, 4.5), (9, 4.5)),  # Score to Select
        ((10, 3.75), (2, 3.25)),  # Select to Handoff
    ]

    for start, end in arrows:
        arrow = ConnectionPatch(
            start,
            end,
            "data",
            "data",
            arrowstyle="->",
            shrinkA=5,
            shrinkB=5,
            mutation_scale=20,
            fc="black",
            alpha=0.8,
            linewidth=2,
        )
        ax.add_patch(arrow)

    # Add Yes/No labels
    ax.text(8.5, 7.8, "Yes", ha="center", va="center", fontweight="bold", fontsize=11, color="red")
    ax.text(
        11.5, 7.8, "No", ha="center", va="center", fontweight="bold", fontsize=11, color="green"
    )

    # Title
    ax.text(
        7,
        9.5,
        "Agent Selection and Context Switching Flow",
        ha="center",
        va="center",
        fontweight="bold",
        fontsize=16,
    )

    plt.tight_layout()
    plt.savefig(output_dir / "agent_selection_flow.png", dpi=300, bbox_inches="tight")
    plt.close()


def create_session_lifecycle():
    """Create session lifecycle state diagram."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis("off")

    # Define states
    states = [
        ("Session\nCreated", 2, 8, "#FF6B6B"),
        ("Agent\nSelected", 6, 8, "#4ECDC4"),
        ("Processing\nMessage", 10, 8, "#45B7D1"),
        ("Context\nAnalysis", 10, 5.5, "#FECA57"),
        ("Agent\nHandoff", 6, 5.5, "#96CEB4"),
        ("Progress\nUpdate", 10, 3, "#FF9FF3"),
        ("Session\nEnded", 2, 3, "#DDA0DD"),
    ]

    # Draw states
    for state, x, y, color in states:
        if "Analysis" in state:
            # Diamond for decision
            diamond = patches.RegularPolygon(
                (x, y),
                4,
                radius=1.2,
                orientation=np.pi / 4,
                facecolor=color,
                alpha=0.8,
                edgecolor="black",
                linewidth=1.5,
            )
            ax.add_patch(diamond)
        else:
            # Rectangle for states
            box = FancyBboxPatch(
                (x - 1, y - 0.75),
                2,
                1.5,
                boxstyle="round,pad=0.1",
                facecolor=color,
                alpha=0.8,
                edgecolor="black",
                linewidth=1.5,
            )
            ax.add_patch(box)

        ax.text(x, y, state, ha="center", va="center", fontweight="bold", fontsize=10)

    # Add start/end states
    start_circle = patches.Circle((2, 9.5), 0.3, facecolor="black")
    ax.add_patch(start_circle)
    ax.text(2, 9, "START", ha="center", va="center", fontweight="bold", fontsize=8)

    end_circle = patches.Circle((2, 1.5), 0.3, facecolor="black")
    inner_circle = patches.Circle((2, 1.5), 0.2, facecolor="white")
    ax.add_patch(end_circle)
    ax.add_patch(inner_circle)
    ax.text(2, 1, "END", ha="center", va="center", fontweight="bold", fontsize=8)

    # Add transitions
    transitions = [
        ((2, 9.2), (2, 8.75)),  # Start to Created
        ((3, 8), (5, 8)),  # Created to Selected
        ((7, 8), (9, 8)),  # Selected to Processing
        ((10, 7.25), (10, 6.7)),  # Processing to Analysis
        ((8.8, 5.5), (7, 5.5)),  # Analysis to Handoff (switch needed)
        ((6, 6.25), (6, 7.25)),  # Handoff back to Selected
        ((10, 4.3), (10, 3.75)),  # Analysis to Progress (no switch)
        ((9, 3), (9, 7.25)),  # Progress back to Processing (continue)
        ((10, 7.25), (2, 3.75)),  # Processing to End (end session)
        ((2, 2.25), (2, 1.8)),  # Ended to End
    ]

    for start, end in transitions:
        arrow = ConnectionPatch(
            start,
            end,
            "data",
            "data",
            arrowstyle="->",
            shrinkA=5,
            shrinkB=5,
            mutation_scale=20,
            fc="black",
            alpha=0.8,
            linewidth=2,
        )
        ax.add_patch(arrow)

    # Add labels on transitions
    ax.text(4, 8.3, "start_session()", ha="center", va="bottom", fontsize=9, style="italic")
    ax.text(8, 8.3, "select_agent()", ha="center", va="bottom", fontsize=9, style="italic")
    ax.text(10.5, 6.8, "handle_message()", ha="left", va="center", fontsize=9, style="italic")
    ax.text(
        7.5, 5.8, "switch needed", ha="center", va="bottom", fontsize=9, style="italic", color="red"
    )
    ax.text(10.5, 4, "no switch", ha="left", va="center", fontsize=9, style="italic", color="green")
    ax.text(8.5, 5.2, "continue", ha="center", va="bottom", fontsize=9, style="italic")
    ax.text(6, 5.2, "end_session()", ha="center", va="bottom", fontsize=9, style="italic")

    # Title
    ax.text(
        7,
        9.5,
        "Session Lifecycle State Machine",
        ha="center",
        va="center",
        fontweight="bold",
        fontsize=16,
    )

    plt.tight_layout()
    plt.savefig(output_dir / "session_lifecycle.png", dpi=300, bbox_inches="tight")
    plt.close()


def create_communication_patterns():
    """Create agent communication patterns diagram with corrected architecture."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # Pattern 1: Direct Agent-to-User Communication
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 8)
    ax1.set_title("Direct Agent-to-User Communication", fontweight="bold", fontsize=14)
    ax1.axis("off")

    # Actors
    user_box1 = FancyBboxPatch(
        (2, 6), 2, 1, boxstyle="round,pad=0.1", facecolor="#FF6B6B", alpha=0.8, edgecolor="black"
    )
    ax1.add_patch(user_box1)
    ax1.text(3, 6.5, "User", ha="center", va="center", fontweight="bold")

    agent_box1 = FancyBboxPatch(
        (6, 6), 2, 1, boxstyle="round,pad=0.1", facecolor="#45B7D1", alpha=0.8, edgecolor="black"
    )
    ax1.add_patch(agent_box1)
    ax1.text(7, 6.5, "Marco", ha="center", va="center", fontweight="bold")

    # Coordinator working behind scenes
    coord_box1 = FancyBboxPatch(
        (4, 3),
        2,
        1,
        boxstyle="round,pad=0.1",
        facecolor="#DDD",
        alpha=0.6,
        edgecolor="gray",
        linestyle="--",
    )
    ax1.add_patch(coord_box1)
    ax1.text(
        5, 3.5, "Coordinator\n(background)", ha="center", va="center", fontsize=9, style="italic"
    )

    # Direct conversation arrows
    arrow1 = ConnectionPatch(
        (4, 6),
        (6, 6),
        "data",
        "data",
        arrowstyle="->",
        shrinkA=5,
        shrinkB=5,
        mutation_scale=20,
        fc="blue",
        alpha=0.8,
        linewidth=2,
    )
    ax1.add_patch(arrow1)
    ax1.text(5, 6.3, '"Ciao! Come stai?"', ha="center", va="bottom", fontsize=9, style="italic")

    arrow2 = ConnectionPatch(
        (6, 5.5),
        (4, 5.5),
        "data",
        "data",
        arrowstyle="->",
        shrinkA=5,
        shrinkB=5,
        mutation_scale=20,
        fc="green",
        alpha=0.8,
        linewidth=2,
    )
    ax1.add_patch(arrow2)
    ax1.text(5, 5.2, '"Bene, grazie!"', ha="center", va="top", fontsize=9, style="italic")

    ax1.text(
        5,
        1.5,
        "Users talk directly to agents.\nCoordinator monitors in background.",
        ha="center",
        va="center",
        fontsize=10,
        style="italic",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.5),
    )

    # Pattern 2: Agent-to-Agent Collaboration via Events
    ax2.set_xlim(0, 12)
    ax2.set_ylim(0, 8)
    ax2.set_title("Agent Collaboration via Event Bus", fontweight="bold", fontsize=14)
    ax2.axis("off")

    # Create vertical layout for better clarity
    marco_box2 = FancyBboxPatch(
        (2, 6.5),
        2,
        0.8,
        boxstyle="round,pad=0.1",
        facecolor="#45B7D1",
        alpha=0.8,
        edgecolor="black",
    )
    ax2.add_patch(marco_box2)
    ax2.text(3, 6.9, "Marco", ha="center", va="center", fontweight="bold")

    coord_box2 = FancyBboxPatch(
        (5, 4.5),
        2,
        0.8,
        boxstyle="round,pad=0.1",
        facecolor="#4ECDC4",
        alpha=0.8,
        edgecolor="black",
    )
    ax2.add_patch(coord_box2)
    ax2.text(6, 4.9, "Coordinator", ha="center", va="center", fontweight="bold")

    prof_box2 = FancyBboxPatch(
        (8, 6.5),
        2,
        0.8,
        boxstyle="round,pad=0.1",
        facecolor="#45B7D1",
        alpha=0.8,
        edgecolor="black",
    )
    ax2.add_patch(prof_box2)
    ax2.text(9, 6.9, "Professoressa", ha="center", va="center", fontweight="bold")

    # Event flow arrows
    arrow2_1 = ConnectionPatch(
        (4, 6.7),
        (5, 5.3),
        "data",
        "data",
        arrowstyle="->",
        shrinkA=5,
        shrinkB=5,
        mutation_scale=15,
        fc="red",
        alpha=0.8,
        linewidth=2,
    )
    ax2.add_patch(arrow2_1)
    ax2.text(
        4.2,
        6,
        "REQUEST_HELP\n(grammar question)",
        ha="left",
        va="center",
        fontsize=8,
        style="italic",
    )

    arrow2_2 = ConnectionPatch(
        (7, 5.1),
        (8, 6.3),
        "data",
        "data",
        arrowstyle="->",
        shrinkA=5,
        shrinkB=5,
        mutation_scale=15,
        fc="green",
        alpha=0.8,
        linewidth=2,
    )
    ax2.add_patch(arrow2_2)
    ax2.text(7.5, 5.7, "routes to\nexpert", ha="center", va="center", fontsize=8, style="italic")

    arrow2_3 = ConnectionPatch(
        (8, 6.1),
        (4, 6.1),
        "data",
        "data",
        arrowstyle="->",
        shrinkA=5,
        shrinkB=5,
        mutation_scale=15,
        fc="blue",
        alpha=0.8,
        linewidth=2,
    )
    ax2.add_patch(arrow2_3)
    ax2.text(6, 5.8, "expert advice", ha="center", va="bottom", fontsize=8, style="italic")

    ax2.text(
        6,
        2,
        "Agents collaborate through coordinator\nwhile user sees seamless conversation.",
        ha="center",
        va="center",
        fontsize=10,
        style="italic",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.5),
    )

    # Pattern 3: Context Switching (Handoff)
    ax3.set_xlim(0, 14)
    ax3.set_ylim(0, 8)
    ax3.set_title("Context Switching via Handoff", fontweight="bold", fontsize=14)
    ax3.axis("off")

    user_box3 = FancyBboxPatch(
        (1, 6), 2, 1, boxstyle="round,pad=0.1", facecolor="#FF6B6B", alpha=0.8, edgecolor="black"
    )
    ax3.add_patch(user_box3)
    ax3.text(2, 6.5, "User", ha="center", va="center", fontweight="bold")

    marco_box3 = FancyBboxPatch(
        (5, 6), 2, 1, boxstyle="round,pad=0.1", facecolor="#45B7D1", alpha=0.8, edgecolor="black"
    )
    ax3.add_patch(marco_box3)
    ax3.text(6, 6.5, "Marco", ha="center", va="center", fontweight="bold")

    coord_box3 = FancyBboxPatch(
        (5, 3.5), 2, 1, boxstyle="round,pad=0.1", facecolor="#4ECDC4", alpha=0.8, edgecolor="black"
    )
    ax3.add_patch(coord_box3)
    ax3.text(6, 4, "Coordinator", ha="center", va="center", fontweight="bold")

    prof_box3 = FancyBboxPatch(
        (9, 6), 3, 1, boxstyle="round,pad=0.1", facecolor="#45B7D1", alpha=0.8, edgecolor="black"
    )
    ax3.add_patch(prof_box3)
    ax3.text(10.5, 6.5, "Professoressa", ha="center", va="center", fontweight="bold")

    # Handoff sequence
    # 1. User to Marco
    arrow3_1 = ConnectionPatch(
        (3, 6.2),
        (5, 6.2),
        "data",
        "data",
        arrowstyle="->",
        shrinkA=5,
        shrinkB=5,
        mutation_scale=15,
        fc="blue",
        alpha=0.8,
        linewidth=2,
    )
    ax3.add_patch(arrow3_1)
    ax3.text(4, 6.4, '"Grammar question"', ha="center", va="bottom", fontsize=8, style="italic")

    # 2. Marco requests handoff
    arrow3_2 = ConnectionPatch(
        (6, 5.5),
        (6, 4.5),
        "data",
        "data",
        arrowstyle="->",
        shrinkA=5,
        shrinkB=5,
        mutation_scale=15,
        fc="red",
        alpha=0.8,
        linewidth=2,
    )
    ax3.add_patch(arrow3_2)
    ax3.text(
        6.5, 5, "REQUEST_HANDOFF", ha="left", va="center", fontsize=8, style="italic", rotation=270
    )

    # 3. Marco introduces Professoressa
    arrow3_3 = ConnectionPatch(
        (5, 5.8),
        (3, 5.8),
        "data",
        "data",
        arrowstyle="->",
        shrinkA=5,
        shrinkB=5,
        mutation_scale=15,
        fc="orange",
        alpha=0.8,
        linewidth=2,
    )
    ax3.add_patch(arrow3_3)
    ax3.text(
        4,
        5.5,
        '"My nonna knows\neverything about grammar!"',
        ha="center",
        va="top",
        fontsize=8,
        style="italic",
    )

    # 4. Professoressa takes over
    arrow3_4 = ConnectionPatch(
        (9, 6.2),
        (3, 6.8),
        "data",
        "data",
        arrowstyle="->",
        shrinkA=5,
        shrinkB=5,
        mutation_scale=15,
        fc="green",
        alpha=0.8,
        linewidth=2,
    )
    ax3.add_patch(arrow3_4)
    ax3.text(
        6,
        7,
        '"Ciao! Let me help with grammar..."',
        ha="center",
        va="bottom",
        fontsize=8,
        style="italic",
    )

    ax3.text(
        7,
        1.5,
        "Natural handoffs orchestrated by coordinator.\nUser experiences smooth transitions.",
        ha="center",
        va="center",
        fontsize=10,
        style="italic",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.5),
    )

    # Pattern 4: Background Session Monitoring
    ax4.set_xlim(0, 12)
    ax4.set_ylim(0, 8)
    ax4.set_title("Background Session Monitoring", fontweight="bold", fontsize=14)
    ax4.axis("off")

    # Session state visualization
    session_box = FancyBboxPatch(
        (4, 6), 4, 1.5, boxstyle="round,pad=0.1", facecolor="#FECA57", alpha=0.8, edgecolor="black"
    )
    ax4.add_patch(session_box)
    ax4.text(6, 6.75, "Session State", ha="center", va="center", fontweight="bold", fontsize=12)

    coord_box4 = FancyBboxPatch(
        (1, 3.5),
        2.5,
        1,
        boxstyle="round,pad=0.1",
        facecolor="#4ECDC4",
        alpha=0.8,
        edgecolor="black",
    )
    ax4.add_patch(coord_box4)
    ax4.text(
        2.25, 4, "Coordinator\nService", ha="center", va="center", fontweight="bold", fontsize=10
    )

    discovery_box4 = FancyBboxPatch(
        (8.5, 3.5),
        2.5,
        1,
        boxstyle="round,pad=0.1",
        facecolor="#96CEB4",
        alpha=0.8,
        edgecolor="black",
    )
    ax4.add_patch(discovery_box4)
    ax4.text(
        9.75, 4, "Discovery\nService", ha="center", va="center", fontweight="bold", fontsize=10
    )

    # Monitoring arrows
    arrow4_1 = ConnectionPatch(
        (3.5, 4),
        (4, 6),
        "data",
        "data",
        arrowstyle="->",
        shrinkA=5,
        shrinkB=5,
        mutation_scale=15,
        fc="purple",
        alpha=0.8,
        linewidth=2,
    )
    ax4.add_patch(arrow4_1)
    ax4.text(3.5, 5.2, "monitors\nprogress", ha="center", va="center", fontsize=8, style="italic")

    arrow4_2 = ConnectionPatch(
        (8.5, 4),
        (8, 6),
        "data",
        "data",
        arrowstyle="->",
        shrinkA=5,
        shrinkB=5,
        mutation_scale=15,
        fc="purple",
        alpha=0.8,
        linewidth=2,
    )
    ax4.add_patch(arrow4_2)
    ax4.text(8.5, 5.2, "finds best\nagents", ha="center", va="center", fontsize=8, style="italic")

    # Session details
    ax4.text(
        6,
        6.2,
        "• Current agent: Marco\n• Messages: 5\n• Topics: food, travel\n• Engagement: 0.8",
        ha="center",
        va="center",
        fontsize=9,
    )

    ax4.text(
        6,
        1.5,
        "Coordinator tracks all session data\nand manages agent assignments.",
        ha="center",
        va="center",
        fontsize=10,
        style="italic",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lavender", alpha=0.5),
    )

    plt.tight_layout()
    plt.savefig(output_dir / "communication_patterns.png", dpi=300, bbox_inches="tight")
    plt.close()


def create_data_flow_diagram():
    """Create data flow and retention diagram."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis("off")

    # Data lifecycle stages
    stages = [
        ("User Input", 2, 8.5, "#FF6B6B"),
        ("Active Session\nData", 6, 8.5, "#4ECDC4"),
        ("Conversation\nHistory", 10, 8.5, "#45B7D1"),
        ("Learning\nProgress", 2, 6, "#96CEB4"),
        ("Archived\nData", 6, 6, "#FECA57"),
        ("Anonymized\nData", 10, 6, "#FF9FF3"),
        ("Analytics\nInsights", 2, 3.5, "#DDA0DD"),
        ("Deleted\nData", 6, 3.5, "#D3D3D3"),
        ("Retention\nPolicies", 10, 3.5, "#FFB6C1"),
    ]

    # Draw stages
    for stage, x, y, color in stages:
        box = FancyBboxPatch(
            (x - 1, y - 0.75),
            2,
            1.5,
            boxstyle="round,pad=0.1",
            facecolor=color,
            alpha=0.8,
            edgecolor="black",
            linewidth=1.5,
        )
        ax.add_patch(box)
        ax.text(x, y, stage, ha="center", va="center", fontweight="bold", fontsize=10)

    # Data flow arrows
    flows = [
        ((3, 8.5), (5, 8.5), "real-time"),
        ((7, 8.5), (9, 8.5), "persist"),
        ((10, 7.75), (2, 6.75), "extract"),
        ((3, 6), (5, 6), "archive"),
        ((7, 6), (9, 6), "anonymize"),
        ((2, 5.25), (2, 4.25), "analyze"),
        ((6, 5.25), (6, 4.25), "delete"),
        ((10, 5.25), (10, 4.25), "enforce"),
    ]

    for start, end, label in flows:
        arrow = ConnectionPatch(
            start,
            end,
            "data",
            "data",
            arrowstyle="->",
            shrinkA=5,
            shrinkB=5,
            mutation_scale=20,
            fc="black",
            alpha=0.7,
            linewidth=2,
        )
        ax.add_patch(arrow)
        mid_x, mid_y = (start[0] + end[0]) / 2, (start[1] + end[1]) / 2
        ax.text(
            mid_x,
            mid_y + 0.2,
            label,
            ha="center",
            va="bottom",
            fontsize=9,
            style="italic",
            color="blue",
        )

    # Add retention timeline
    timeline_y = 1.5
    timeline_stages = [
        (1, "Active\n(Session)"),
        (4, "Recent\n(7 days)"),
        (7, "Archived\n(30 days)"),
        (10, "Anonymized\n(1 year)"),
        (13, "Deleted"),
    ]

    # Draw timeline
    ax.plot([0.5, 13.5], [timeline_y, timeline_y], "k-", linewidth=3)

    for x, stage in timeline_stages:
        ax.plot([x, x], [timeline_y - 0.1, timeline_y + 0.1], "k-", linewidth=3)
        ax.text(x, timeline_y - 0.4, stage, ha="center", va="top", fontweight="bold", fontsize=9)

    ax.text(
        7, 0.5, "Data Retention Timeline", ha="center", va="center", fontweight="bold", fontsize=12
    )

    # Title
    ax.text(
        7,
        9.5,
        "Data Flow and Retention Management",
        ha="center",
        va="center",
        fontweight="bold",
        fontsize=16,
    )

    plt.tight_layout()
    plt.savefig(output_dir / "data_flow_diagram.png", dpi=300, bbox_inches="tight")
    plt.close()


def main():
    """Generate all diagrams."""
    print("Generating architecture diagrams...")

    print("1. Creating system architecture diagram...")
    create_system_architecture_diagram()

    print("2. Creating agent selection flow...")
    create_agent_selection_flow()

    print("3. Creating session lifecycle diagram...")
    create_session_lifecycle()

    print("4. Creating communication patterns...")
    create_communication_patterns()

    print("5. Creating data flow diagram...")
    create_data_flow_diagram()

    print(f"\nAll diagrams generated and saved to: {output_dir}")
    print("\nGenerated files:")
    for file in output_dir.glob("*.png"):
        print(f"  - {file.name}")


if __name__ == "__main__":
    main()
