import matplotlib.pyplot as plt
from collections import OrderedDict

data = {0: 10, 10001: 1628, 1: 635, 20001: 12, 20002: 621, 10000: 10, 20000: 2, 30002: 1, 30003: 108, 10002: 16, 2: 22}
sorted_data = OrderedDict()

for i in sorted(data):
    sorted_data[str(i)] = data[i]

plt.bar(range(len(list(sorted_data.keys()))), list(sorted_data.values()), tick_label=list(sorted_data.keys()))

i = 0
for a, b in sorted_data.items(): # zip 函数
    plt.text(i, b+0.05, f"{b}", ha='center', fontsize=10)
    i += 1

plt.show()
