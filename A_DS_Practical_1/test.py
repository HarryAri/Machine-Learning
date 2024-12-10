# Read input
buses_and_locations = list(map(int, input().strip().split()))
B = buses_and_locations[0]  # number of buses
L = buses_and_locations[1]  # number of locations

bus_schedule = [buses_and_locations]

# Read the bus schedules
for i in range(B):
    locations = list(map(int, input().strip().split()))
    times = list(map(int, input().strip().split()))
    bus_schedule.append(locations)
    bus_schedule.append(times)

# Graph Construction
bus_schedule_graph = {}
bus_schedule_graph[(0, 0)] = []  # Start at (0, 0)

# Build the graph where each edge contains the time at the bus stop
for i in range(1, len(bus_schedule) - 1, 2):
    for j in range(len(bus_schedule[i]) - 1):
        if (bus_schedule[i][j], bus_schedule[i + 1][j]) not in bus_schedule_graph:
            bus_schedule_graph[(bus_schedule[i][j], bus_schedule[i + 1][j])] = []
        # The weight is the time spent at the current bus stop (next stop time - current stop time)
        bus_schedule_graph[(bus_schedule[i][j], bus_schedule[i + 1][j])].append(
            ((bus_schedule[i][j + 1], bus_schedule[i + 1][j + 1]), bus_schedule[i + 1][j + 1] - bus_schedule[i + 1][j]))

    # Handle last edge to make sure it's in the graph
    if (bus_schedule[i][-1], bus_schedule[i+1][-1]) not in bus_schedule_graph:
        bus_schedule_graph[(bus_schedule[i][-1], bus_schedule[i + 1][-1])] = []

# Initialize distances (maximize time spent at bus stops)
distances = {key: -float('inf') for key in bus_schedule_graph}  # Use negative infinity to maximize
distances[(0, 0)] = 0  # Start at (0, 0)

# Relax all edges L-1 times
for _ in range(L - 1):
    for key in bus_schedule_graph:
        for neighbor, weight in bus_schedule_graph[key]:
            if distances[key] + weight > distances[neighbor]:
                distances[neighbor] = distances[key] + weight

                # Check if we have reached the destination location (L-1)
                if neighbor[0] == L - 1:
                    # Output the time when reaching the destination
                    print(distances[neighbor])
                    exit()  # Exit once we have reached the destination
