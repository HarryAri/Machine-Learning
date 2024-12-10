import heapq

buses_and_locations = list(map(int, input().strip().split()))
B = buses_and_locations[0]
L = buses_and_locations[1]

bus_schedule = [buses_and_locations]

for i in range(B):
    locations = list(map(int, input().strip().split()))
    times = list(map(int, input().strip().split()))
    bus_schedule.append(locations)
    bus_schedule.append(times)

bus_schedule_graph = {}
bus_schedule_graph[(0, 0)] = []
for i in range(1, len(bus_schedule) - 1, 2):
    for j in range(len(bus_schedule[i]) - 1):
        if (bus_schedule[i][j], bus_schedule[i + 1][j]) not in bus_schedule_graph:
            bus_schedule_graph[(bus_schedule[i][j], bus_schedule[i + 1][j])] = []
        bus_schedule_graph[(bus_schedule[i][j], bus_schedule[i + 1][j])].append(((bus_schedule[i][j + 1],
                                                                                  bus_schedule[i + 1][j + 1]),
                                                                                 bus_schedule[i + 1][j + 1] -
                                                                                 bus_schedule[i + 1][j]))

    if (bus_schedule[i][-1], bus_schedule[i+1][-1]) not in bus_schedule_graph:
        bus_schedule_graph[(bus_schedule[i][-1], bus_schedule[i + 1][-1])] = []


keys = sorted(list(bus_schedule_graph.keys()), key=lambda x: (x[0], x[1]))
for i in range(len(keys)):
    for j in range(len(keys)):
        if keys[i][0] == keys[j][0] and keys[i][1] < keys[j][1]:
            bus_schedule_graph[keys[i]].append((keys[j], (keys[j][1] - keys[i][1])))

priority_queue = [(0, (0, 0))]
fastest_times = {}

for key in bus_schedule_graph:
    fastest_times[key] = float('inf')

while priority_queue:
    time, key = heapq.heappop(priority_queue)
    if key[0] == L - 1:
        print(time)
        break
    for key_next, weigth in bus_schedule_graph[key]:
        candidate = time + weigth
        if candidate < fastest_times[key_next]:
            fastest_times[key_next] = candidate
            heapq.heappush(priority_queue, (candidate, key_next))