import matplotlib.pyplot as plt
import numpy as np

platforms = ['IBM', 'Rigetti', 'IonQ']
values = [2.0, 2.272, 2.364]

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(platforms, values)
ax.set_ylabel('CHSH S')
ax.set_title('Bar Order Test')

for i, (p, v) in enumerate(zip(platforms, values)):
    print(f'Bar {i}: {p} = {v}')

plt.savefig('test_bars.png', dpi=150, bbox_inches='tight')
print('Saved test_bars.png')
plt.close()
