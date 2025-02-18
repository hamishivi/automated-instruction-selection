import matplotlib.pyplot as plt
import numpy as np

# Define cost function
def compute_cost(method, x, y):
    if method in ["Random (unbal.)", "Random (bal.)"]:
        return 6 * 8e9 * 2048 * x * 2
    elif method == "LESS":
        return 6 * 8e9 * 2048 * x * 2 * 2 + 6 * 8e9 * 2048 * y * 3
    elif method == "Embed (NV)":
        return 6 * 8e9 * 2048 * x * 2 + 2 * 8e9 * 2048 * y
    elif method == "Embed (GTR)":
        return 6 * 8e9 * 2048 * x * 2 + 2 * 110e6 * 2048 * y
    elif method in ["Top-PPL", "Mid-PPL", "RDS (ours)"]:
        return 6 * 8e9 * 2048 * x * 2 + 2 * 8e9 * 2048 * y
    elif method == "IFD":
        return 2 * 8e9 * 2048 * 200000 + 1000 * 6 * 8e9 * 2048 + 2 * 8e9 * 2048 * y + 6 * 8e9 * 2048 * x * 2

# First dataset (x = 10000, y = 200000)
x_1 = 10000
y_1 = 200000
scores_1 = {
    "Random (unbal.)": 40.9,
    "Random (bal.)": 42.2,
    "LESS": 44.2,
    "Embed (NV)": 42.7,
    "Embed (GTR)": 42.1,
    "Top-PPL": 34.2,
    "Mid-PPL": 39.4,
    "RDS (ours)": 46.4,
    "IFD": 38.7
}

costs_1 = {method: compute_cost(method, x_1, y_1) for method in scores_1.keys()}

# Second dataset (x = 326000, y = 5.8M)
x_2 = 10000
y_2 = 5.8e6
scores_2 = {
    "Random (unbal.)": 40.73,
    "Random (bal.)": 41.1,
    "Embed (NV)": 41.6,
    "Embed (GTR)": 46.4,
    "Top-PPL": 30.4,
    "Mid-PPL": 38.9,
    "RDS (ours)": 50.5,
    "IFD": 35.8
}

costs_2 = {method: compute_cost(method, x_2, y_2) for method in scores_2.keys()}

# x_3 = 326000
# y_3 = 5.8e6
# scores_3 = {
#     "Random (unbal.)": 44.5,
#     "Random (bal.)": 47.5,
#     "Embed (NV)": 45.3,
#     "Embed (GTR)": 48.0,
#     "Top-PPL": 36.6,
#     "Mid-PPL": 41.3,
#     "RDS (ours)": 50.9,
#     "IFD": 35.7
# }

# costs_3 = {method: compute_cost(method, x_3, y_3) for method in scores_3.keys()}

# Prepare data for Pareto frontier calculation
all_scores = list(scores_1.values()) + list(scores_2.values()) # + list(scores_3.values())
all_costs = list(costs_1.values()) + list(costs_2.values()) # + list(costs_3.values())

# Convert to numpy arrays for processing
points = np.array(list(zip(all_costs, all_scores)))

# Sort points by cost (ascending), then by score (descending)
sorted_indices = np.lexsort((-points[:, 1], points[:, 0]))
sorted_points = points[sorted_indices]

# Compute Pareto frontier: keep a point if it has the highest score seen so far for a given cost
pareto_points = []
max_score = -np.inf

for cost, score in sorted_points:
    if score > max_score:
        pareto_points.append((cost, score))
        max_score = score

pareto_points = np.array(pareto_points)

# Plot with Pareto frontier shading above the line
plt.figure(figsize=(12, 7))
plt.rcParams.update({'font.size': 24})  # Increase general font size

# Adjusted label offset for a slight upward shift
label_offset = -0.5  # Slightly adjust upwards


# set y_max to 52
plt.ylim(30, 52)

# Colors for differentiation
colors = plt.cm.tab10(np.linspace(0, 1, len(scores_1)))
less_color = None  # Store the color used for LESS

# Plot all points and connect them with lines, adding labels slightly below the middle
for (method, score_1), color in zip(scores_1.items(), colors):
    if method in scores_2 or method == "LESS":  # LESS only appears once
        score_2 = scores_2.get(method, None)
        cost_1 = costs_1[method]
        cost_2 = costs_2.get(method, None)
        # cost_3 = costs_3.get(method, None)
        # score_3 = scores_3.get(method, None)

        # Scatter points, bold for RDS
        if method == "RDS (ours)":
            plt.scatter(cost_1, score_1, color=color, edgecolors='black', linewidth=4, s=100)
        else:
            plt.scatter(cost_1, score_1, color=color, linewidth=4,)

        # Store LESS color
        if method == "LESS":
            less_color = color

        if score_2 is not None:
            if method == "RDS (ours)":
                plt.scatter(cost_2, score_2, color=color, edgecolors='black', linewidth=4, s=100)
            else:
                plt.scatter(cost_2, score_2, color=color, linewidth=4,)

            # Connect points with a line if there are two points
            plt.plot([cost_1, cost_2], [score_1, score_2], color=color, linestyle="--", linewidth=5)

            # Calculate the middle of the line for label placement
            mid_cost = np.sqrt(cost_1 * cost_2)  # Geometric mean for log scale placement
            mid_score = (score_1 + score_2) / 2 - label_offset  # Move slightly up

            # Place label slightly above the middle of the line
            if 'rds' in method.lower():
                plt.text(mid_cost, mid_score - 2, "RDS+", fontsize=20, fontweight='bold' if method == "RDS (ours)" else 'normal',
                     verticalalignment='bottom', horizontalalignment='center', color=color, 
                     )
            elif 'random (bal.)' in method.lower():
                plt.text(mid_cost + 2e18, mid_score - 1.4, method, fontsize=20, fontweight='bold' if method == "RDS (ours)" else 'normal',
                     verticalalignment='bottom', horizontalalignment='center', color=color, 
                     )
            elif 'random (unbal.)' in method.lower():
                plt.text(mid_cost + 2e18, mid_score - 2, method, fontsize=20, fontweight='bold' if method == "RDS (ours)" else 'normal',
                     verticalalignment='bottom', horizontalalignment='center', color=color, 
                     )
            elif 'ifd' in method.lower():
                plt.text(mid_cost, mid_score - 1, method, fontsize=20, fontweight='bold' if method == "RDS (ours)" else 'normal',
                     verticalalignment='top', horizontalalignment='center', color=color, 
                     )
            elif 'embed (gtr)' in method.lower():
                plt.text(mid_cost + 2.5e18, mid_score - 1, method, fontsize=20, fontweight='bold' if method == "RDS (ours)" else 'normal',
                     verticalalignment='bottom', horizontalalignment='center', color=color, 
                     )
            else:
                plt.text(mid_cost, mid_score, method, fontsize=20, fontweight='bold' if method == "RDS (ours)" else 'normal',
                     verticalalignment='bottom', horizontalalignment='center', color=color, 
                     )
        
        # if cost_3 is not None:
        #     if method == "RDS (ours)":
        #         plt.scatter(cost_3, score_3, color=color, edgecolors='black', linewidth=2, s=100)
        #     else:
        #         plt.scatter(cost_3, score_3, color=color)

        #     # Connect points with a line if there are two points
        #     plt.plot([cost_2, cost_3], [score_2, score_3], color=color, linestyle="--")

        #     # Calculate the middle of the line for label placement
        #     mid_cost = np.sqrt(cost_2 * cost_3)  # Geometric mean for log scale placement
        #     mid_score = (score_2 + score_3) / 2 - label_offset  # Move slightly up

        #     # Place label slightly above the middle of the line
        #     if 'rds' in method.lower():
        #         plt.text(mid_cost, mid_score - 2, method, fontsize=9, fontweight='bold' if method == "RDS (ours)" else 'normal',
        #              verticalalignment='bottom', horizontalalignment='center', color=color, 
        #              )
        #     elif 'random' in method.lower():
        #         plt.text(mid_cost, mid_score - 1.4, method, fontsize=9, fontweight='bold' if method == "RDS (ours)" else 'normal',
        #              verticalalignment='bottom', horizontalalignment='center', color=color, 
        #              )
        #     elif 'ifd' in method.lower():
        #         plt.text(mid_cost, mid_score - .3, method, fontsize=9, fontweight='bold' if method == "RDS (ours)" else 'normal',
        #              verticalalignment='top', horizontalalignment='center', color=color, 
        #              )
        #     else:
        #         plt.text(mid_cost, mid_score, method, fontsize=9, fontweight='bold' if method == "RDS (ours)" else 'normal',
        #              verticalalignment='bottom', horizontalalignment='center', color=color, 
        #              )

# Label the LESS point to the right of the point in green
less_cost = costs_1["LESS"]
less_score = scores_1["LESS"]
plt.text(less_cost * 1.1, less_score, "LESS", fontsize=20, verticalalignment='center', horizontalalignment='left',
         color=less_color)

# Shade the area **above** the Pareto frontier
plt.fill_between(pareto_points[:, 0], pareto_points[:, 1], y2=max(all_scores) + 5, color='red', alpha=0.2, label="Suboptimal Region")

plt.xlabel("Estimated Overall FLOPs Cost")
plt.ylabel("Average Performance")
plt.xscale("log")  # Ensure x-axis is log scale
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.savefig('pareto_plot.pdf',bbox_inches='tight')
