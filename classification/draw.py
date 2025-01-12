
# Re-importing necessary libraries due to environment reset
import matplotlib.pyplot as plt

# Data for the intervals and corresponding Acc@1 values
intervals = [2, 3, 4, 5, 6, 7, 8]
accuracy = [83.016, 82.072, 81.248, 79.908, 80.000, 78.61, 78.71]

# Plotting the line chart
plt.figure(figsize=(8, 5), dpi=500)
plt.plot(intervals, accuracy, marker='o', linestyle='-', color='b')
plt.title('Effect of Pruning Interval on Top-1 Accuracy', fontsize=20)
plt.xlabel('Interval', fontsize=20)
plt.ylabel('Acc@1 (%)', fontsize=20)
plt.xticks(intervals, fontsize=14)
plt.yticks(fontsize=14)
plt.grid(True)
plt.tight_layout()
plt.savefig("ablation_2_1.png")


# Enhancing the plot by adding lines for each n group, enlarging symbols, and refining overall style for clarity
# ablation study 2.2
# Refining the legend and title capitalization as per the request

# n	m	Acc@1
# 1	2	83.02
# 2	4	82.47
# 	5	81.9
# 	6	81.71
# 	7	80.73
# 	8	80.81
# 3	6	82.19
# 	7	81.55
# 	8	81.66
# 4	8	82.32

# n_values = [1, 2, 2, 2, 2, 2, 3, 3, 3, 4]
# m_values = [2, 4, 5, 6, 7, 8, 6, 7, 8, 8]
# accuracy_values = [83.02, 82.47, 81.9, 81.71, 80.73, 80.81, 82.19, 81.55, 81.66, 82.32]
# colors = {1: 'b', 2: 'g', 3: 'r', 4: 'purple'}  # Mapping n to colors
# plt.figure(figsize=(10, 6), dpi=500)

# # Plot each group of n with lines and larger markers
# for n in set(n_values):
#     # Filter data for each unique 'n' value
#     x = [m for i, m in enumerate(m_values) if n_values[i] == n]
#     y = [acc for i, acc in enumerate(accuracy_values) if n_values[i] == n]
#     plt.plot(x, y, color=colors[n], marker='X', markersize=10, linestyle='-', linewidth=2, label=f'n = {n}')  # Simplified legend

# # Updated plot aesthetics
# plt.title('Acc@1 for pruning with continuous n pixels out of m intervals', fontsize=20)
# plt.xlabel('Pruning Interval (m)', fontsize=20)
# plt.ylabel('Acc@1 (%)', fontsize=20)
# plt.legend(title="n", fontsize=14, title_fontsize=15)  # Updated legend title
# plt.xticks(fontsize=14)
# plt.yticks(fontsize=14)
# plt.grid(True, linestyle='--', alpha=0.7)
# plt.tight_layout()
# plt.savefig('ablation_2_2.png')



# 