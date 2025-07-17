import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, ArrowStyle, ConnectionPatch
import os

os.makedirs("docs/figures", exist_ok=True)

fig, ax = plt.subplots(figsize=(10, 6))

# Define block positions
system_pos = (0.5, 0.6, 0.2, 0.1)
esn_pos = (0.3, 0.4, 0.2, 0.1)
pid_pos = (0.3, 0.2, 0.2, 0.1)
sum_pos = (0.5, 0.3, 0.05, 0.05)
target_pos = (0.7, 0.3, 0.05, 0.05)

# Draw blocks
ax.add_patch(FancyBboxPatch((system_pos[0], system_pos[1]), system_pos[2], system_pos[3], boxstyle="round,pad=0.02", ec="black", fc="lightblue", label="System (Pendulum/Arm)"))
ax.add_patch(FancyBboxPatch((esn_pos[0], esn_pos[1]), esn_pos[2], esn_pos[3], boxstyle="round,pad=0.02", ec="black", fc="lightgreen", label="ESN Controller"))
ax.add_patch(FancyBboxPatch((pid_pos[0], pid_pos[1]), pid_pos[2], pid_pos[3], boxstyle="round,pad=0.02", ec="black", fc="lightcoral", label="PID Controller"))
ax.add_patch(plt.Circle((sum_pos[0] + sum_pos[2]/2, sum_pos[1] + sum_pos[3]/2), 0.02, ec="black", fc="white", label="Sum"))
ax.add_patch(plt.Circle((target_pos[0] + target_pos[2]/2, target_pos[1] + target_pos[3]/2), 0.02, ec="black", fc="white", label="Target"))

# Draw arrows
ax.annotate("", xy=(system_pos[0], system_pos[1] + system_pos[3]/2), xytext=(esn_pos[0] + esn_pos[2], esn_pos[1] + esn_pos[3]/2),
            arrowprops=dict(arrowstyle="->", lw=1.5, label="State"))
ax.annotate("", xy=(system_pos[0], system_pos[1] + system_pos[3]/2), xytext=(pid_pos[0] + pid_pos[2], pid_pos[1] + pid_pos[3]/2),
            arrowprops=dict(arrowstyle="->", lw=1.5, label="State"))
ax.annotate("", xy=(sum_pos[0] + sum_pos[2]/2, sum_pos[1] + sum_pos[3]), xytext=(system_pos[0] + system_pos[2]/2, system_pos[1]),
            arrowprops=dict(arrowstyle="->", lw=1.5, label="Control u"))
ax.annotate("", xy=(esn_pos[0] + esn_pos[2]/2, esn_pos[1]), xytext=(sum_pos[0] + sum_pos[2]/2, sum_pos[1]),
            arrowprops=dict(arrowstyle="->", lw=1.5, label="u_ESN"))
ax.annotate("", xy=(pid_pos[0] + pid_pos[2]/2, pid_pos[1]), xytext=(sum_pos[0] + sum_pos[2]/2, sum_pos[1]),
            arrowprops=dict(arrowstyle="->", lw=1.5, label="u_PID"))
ax.annotate("", xy=(target_pos[0] + target_pos[2]/2, target_pos[1]), xytext=(pid_pos[0] + pid_pos[2]/2, pid_pos[1] + pid_pos[3]),
            arrowprops=dict(arrowstyle="->", lw=1.5, label="Target"))

# Labels
ax.text(system_pos[0] + system_pos[2]/2, system_pos[1] + system_pos[3]/2, "System\n(Pendulum/Arm)", ha="center", va="center")
ax.text(esn_pos[0] + esn_pos[2]/2, esn_pos[1] + esn_pos[3]/2, "ESN\nController", ha="center", va="center")
ax.text(pid_pos[0] + pid_pos[2]/2, pid_pos[1] + pid_pos[3]/2, "PID\nController", ha="center", va="center")
ax.text(sum_pos[0] + sum_pos[2]/2, sum_pos[1] + sum_pos[3]/2, "+", ha="center", va="center")
ax.text(target_pos[0] + target_pos[2]/2, target_pos[1] + target_pos[3]/2, "Target", ha="center", va="center")

# Axis settings
ax.set_xlim(0, 1)
ax.set_ylim(0, 0.8)
ax.set_axis_off()
plt.savefig("docs/figures/esn_pid_diagram.png", dpi=300, bbox_inches="tight")
plt.close()