import matplotlib.pyplot as plt
import numpy as np

# Data
cities = ['Beijing', 'Berlin', 'NYC', 'Wuhan']
pop_density = [1300, 4100, 11300, 1200]  # people/km² 
res_ind = [30.9, 48.9, 70.2, 18.2]
GDProduct = [0.700 , 0.236, 1.286, 0.296]
industrial = [29.1, 26.1, 37.6, 17.6]


fig, ax = plt.subplots(figsize=(12, 5)) 

scatter = ax.scatter(pop_density, res_ind, s=200, alpha=0.6, c=['red', 'blue', 'green', 'orange'], edgecolors='black', linewidths=2)


for i, city in enumerate(cities):
    ax.annotate(city, (pop_density[i], res_ind[i]), 
                xytext=(10, 10), textcoords='offset points',
                fontsize=12, fontweight='bold')

ax.set_xlabel('Population Density (people/km²)', fontsize=12, fontweight='bold')
ax.set_ylabel('Residential + Industrial (%)', fontsize=12, fontweight='bold')
ax.set_title('Urban Land Use vs Population Density', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, linestyle='--')
z = np.polyfit(pop_density, res_ind, 1)
p = np.poly1d(z)
ax.plot(pop_density, p(pop_density), "r--", alpha=0.8, linewidth=2, label=f'Trend: y={z[0]:.4f}x+{z[1]:.1f}')
ax.legend()

plt.tight_layout()
plt.savefig('city_comparison.png', dpi=300, bbox_inches='tight')
plt.show()


# Data
cities = ['Beijing', 'Berlin', 'NYC', 'Wuhan']
pop_density = [1300, 4100, 11300, 1200]  # people/km² 
herbaceous_veg = [48.6, 33.1, 13.9, 50.3]  # Fill in your % values from results

fig, ax = plt.subplots(figsize=(10, 6))

scatter = ax.scatter(pop_density, herbaceous_veg, s=200, alpha=0.6, 
                    c=['red', 'blue', 'green', 'orange'],
                    edgecolors='black', linewidths=2)

for i, city in enumerate(cities):
    ax.annotate(city, (pop_density[i], herbaceous_veg[i]),
                xytext=(10, 10), textcoords='offset points',
                fontsize=12, fontweight='bold')

# Optional: Add trendline
z = np.polyfit(pop_density, herbaceous_veg, 1)
p = np.poly1d(z)
ax.plot(pop_density, p(pop_density), "r--", alpha=0.5, linewidth=2, 
        label=f'Trend: y={z[0]:.4f}x+{z[1]:.1f}')

# Optional: Add correlation coefficient
correlation = np.corrcoef(pop_density, herbaceous_veg)[0, 1]
ax.text(0.95, 0.95, f'R = {correlation:.3f}', transform=ax.transAxes, 
        fontsize=12, verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

ax.set_xlabel('Population Density (people/km²)', fontsize=12, fontweight='bold')
ax.set_ylabel('Herbaceous Vegetation (%)', fontsize=12, fontweight='bold')
ax.set_title('Herbaceous Vegetation vs Population Density', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, linestyle='--')
ax.legend()

plt.tight_layout()
plt.savefig('herbaceous_vegetation_comparison.png', dpi=300, bbox_inches='tight')
plt.show()


fig, ax = plt.subplots(figsize=(7, 5))

# Scatter plot (green dots, like before)
sc = ax.scatter(GDProduct, industrial, color='green', label='Cities')

# Annotate each point with the city name
for x, y, name in zip(GDProduct, industrial, cities):
    ax.annotate(name, (x, y),
                textcoords="offset points", xytext=(5, 5),
                fontsize=9)

# Trendline (least-squares linear fit)
m, b = np.polyfit(GDProduct, industrial, 1)
x_line = np.linspace(min(GDProduct), max(GDProduct), 100)
y_line = m * x_line + b
ax.plot(x_line, y_line, linestyle='--', color='black', label='Trendline')

# Correlation coefficient
correlation = np.corrcoef(GDProduct, industrial)[0, 1]
ax.text(0.95, 0.95, f'R = {correlation:.3f}',
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(boxstyle='round',
                  facecolor='lightgreen',
                  alpha=0.5))

# Labels and styling (same format as your example)
ax.set_xlabel('GDProduct', fontsize=12, fontweight='bold')
ax.set_ylabel('Industrial (%)', fontsize=12, fontweight='bold')
ax.set_title('Industrial vs GDProduct', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, linestyle='--')
ax.legend()

plt.tight_layout()
plt.savefig('gdproduct_vs_industrial.png', dpi=300, bbox_inches='tight')
plt.show()
