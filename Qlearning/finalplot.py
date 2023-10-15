import matplotlib
import matplotlib.pyplot as plt
import numpy as np

#time

ddqn_time_dict = { 200: 0.4373147487640381, 400: 0.8761138916015625, 600: 1.506394624710083, 800: 12.971395492553711, 1000: 18.15818500518799, 1200: 39.24370765686035, 1400: 39.55875563621521, 1600: 163.84463763237, 1800: 246.96680569648743, 2000: 247.4340672492981, 2200: 248.05865788459778, 2400: 250.0582275390625}

mcts_time_dict = {200: 43.653846979141235, 400: 89.02893209457397, 600: 180.82742190361023, 800: 278.49342679977417, 1000: 368.5176239013672, 1200: 605.2430918216705, 1400: 741.5620839595795, 1600: 917.7260229587555, 1800: 1113.151396036148, 2000: 1393.452437877655, 2200: 1744.5309858322144, 2400: 2012.3721940517426, 2600: 2285.299206972122, 2800: 2618.422122001648, 3000: 2991.2072670459747}


mcts_values = sorted(mcts_time_dict.items())
ddqn_values = sorted(ddqn_time_dict.items())

x1, y1 = zip(*mcts_values)
x2, y2 = zip(*ddqn_values)

x1 = np.array(x1)
y1 = np.array(y1)

x2 = np.array(x2)
y2 = np.array(y2)


fig, ax = plt.subplots()


ax.scatter(x1, y1, label='MCTS')
ax.scatter(x2, y2, label='DDQN', marker='s', color='r')
ax.legend()
ax.set_xlabel('Distance (x coordinate)')
ax.set_ylabel('Time taken (seconds)')


plt.title("Time taken to reach certain distances.")
plt.savefig("finalgraphs/Time taken to reach certain distances.")

plt.cla()

# plot time against alg


# steps

ddqn_steps_dict = {200: 27, 400: 105, 600: 164, 800: 563, 1000: 592, 1200: 2518, 1400: 2547, 1600: 52433, 1800: 59554, 2000: 169664, 2200: 169712, 2400: 170012}

mcts_steps_dict =  {200: 11570, 400: 23539, 600: 47593, 800: 73031, 1000: 96444, 1200: 158134, 1400: 193452, 1600: 238985, 1800: 289347, 2000: 361067, 2200: 451040, 2400: 519867, 2600: 589938, 2800: 675690, 3000: 771653}


mcts_values = sorted(mcts_steps_dict.items())
ddqn_values = sorted(ddqn_steps_dict.items())

x1, y1 = zip(*mcts_values)
x2, y2 = zip(*ddqn_values)

x1 = np.array(x1)
y1 = np.array(y1)

x2 = np.array(x2)
y2 = np.array(y2)


fig, ax = plt.subplots()


ax.scatter(x1, y1, label='MCTS')
ax.scatter(x2, y2, label='DDQN', marker='s', color='r')
ax.legend()
ax.set_xlabel('Distance (x coordinate)')
ax.set_ylabel('Number of moves taken (steps)')


plt.title("Steps taken to reach certain distances.")
plt.savefig("finalgraphs/Steps taken to reach certain distances.")

plt.cla()
# deaths

ddqn_deaths_dict = {200: 0, 400: 0, 600: 0, 800: 0, 1000: 0, 1200: 2, 1400: 2, 1600: 234, 1800: 375, 2000: 845, 2200: 845, 2400: 845, 2600: 845}

mcts_deaths_dict =  {200: 9, 400: 31, 600: 37, 800: 53, 1000: 57, 1200: 125, 1400: 185, 1600: 242, 1800: 303, 2000: 413, 2200: 413, 2400: 421, 2600: 439, 2800: 489, 3000: 489}



mcts_values = sorted(mcts_deaths_dict.items())
ddqn_values = sorted(ddqn_deaths_dict.items())

x1, y1 = zip(*mcts_values)
x2, y2 = zip(*ddqn_values)

x1 = np.array(x1)
y1 = np.array(y1)

x2 = np.array(x2)
y2 = np.array(y2)


fig, ax = plt.subplots()


ax.scatter(x1, y1, label='MCTS')
ax.scatter(x2, y2, label='DDQN', marker='s', color='r')
ax.legend()
ax.set_xlabel('Distance (x coordinate)')
ax.set_ylabel('Times diead (deaths)')


plt.title("Deaths committed to reach certain distances.")
plt.savefig("finalgraphs/Deaths commmitted to reach certain distances.")

plt.cla()


# score

ddqn_score_dict = {200: 0.0, 400: 300.0, 600: 300.0, 800: 400.0, 1000: 700.0, 1200: 400.0, 1400: 400.0, 1600: 400.0, 1800: 900.0, 2000: 800.0, 2200: 800.0, 2400: 800.0, 2600: 1200.0, 2800: 1500.0, 3000: 1500.0}

mcts_score_dict =  {200: 0, 400: 200, 600: 200, 800: 200, 1000: 300, 1200: 300, 1400: 400, 1600: 400, 1800: 800, 2000: 1000, 2200: 1000, 2400: 1000, 2600: 1000, 2800: 1100, 3000: 1100}



mcts_values = sorted(ddqn_score_dict.items())
ddqn_values = sorted(mcts_score_dict.items())

x1, y1 = zip(*mcts_values)
x2, y2 = zip(*ddqn_values)

x1 = np.array(x1)
y1 = np.array(y1)

x2 = np.array(x2)
y2 = np.array(y2)


fig, ax = plt.subplots()


ax.scatter(x1, y1, label='MCTS')
ax.scatter(x2, y2, label='DDQN', marker='s', color='r')
ax.legend()
ax.set_xlabel('Distance (x coordinate)')
ax.set_ylabel('Average score (points)')


plt.title("Average scores at certain distances.")
plt.savefig("finalgraphs/Average scores at certain distances.")

plt.cla()

# memory usage


ddqn = 430.28 # megabytes
mcts = 222.5 # megabytes

data = {"DDQN": 430.28, "MCTS": 222.5}

algorithms = list(data.keys())
mem_usage = list(data.values()  )

x = np.arange(2)
fig = plt.figure()


fig, ax = plt.subplots()
plt.bar(algorithms, mem_usage,  color = 'maroon')
plt.ylabel(" Memory usage (MB)")
plt.xlabel(" Algorithm ")
plt.title(" Memory usage by algorithm")

plt.savefig("finalgraphs/memory_usage")


# laptop speed for level complete

times = [1027.444876909256, 419.7998788356781, 2391.431916952133, 4022.487316131592]

time_avg = sum(times) / len(times)
