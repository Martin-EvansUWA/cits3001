import matplotlib
import matplotlib.pyplot as plt
import numpy as np

#time
time_dict = { 200: 0.4373147487640381, 400: 0.8761138916015625, 600: 1.506394624710083, 800: 12.971395492553711, 1000: 18.15818500518799, 1200: 39.24370765686035, 1400: 39.55875563621521, 1600: 163.84463763237, 1800: 246.96680569648743, 2000: 247.4340672492981, 2200: 248.05865788459778, 2400: 250.0582275390625}

values = sorted(time_dict.items())
x, y = zip(*values)

x = np.array(x)
y = np.array(y)

plt.scatter(x,y)
plt.ylim(0,None)

plt.title(" Time to reach certain distances.")
plt.xlabel("Distance (x position)")
plt.ylabel("Time taken (seconds)")

plt.show()
plt.savefig("time_to_reach_certain_distances2.png")


#deaths
death_dict = {200: 0, 400: 0, 600: 0, 800: 0, 1000: 0, 1200: 11, 1400: 11, 1600: 54, 1800: 168, 2000: 322, 2200: 322, 2400: 322, 2600: 528}
values = sorted(death_dict.items())
x, y = zip(*values)

x = np.array(x)
y = np.array(y)

plt.scatter(x,y)
plt.ylim(0,None)

plt.title(" Time to reach certain distances.")
plt.xlabel("Distance (x position)")
plt.ylabel("Time taken (seconds)")

plt.show()
plt.savefig("time_to_reach_certain_distances2.png")

# Memory
avg = 60.59648739518905

# plot memory bar graph with Robert's Algorithm


score_dictr = {200: 0.0, 400: 300.0, 600: 300.0, 800: 400.0, 1000: 700.0, 1200: 400.0, 1400: 400.0, 1600: 400.0, 1800: 900.0, 2000: 800.0, 2200: 800.0, 2400: 800.0, 2600: 1200.0, 2800: 1500.0, 3000: 1500.0}

