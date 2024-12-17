# following algorithm

actors_movies_num = list(map(int, input().strip().split()))
print(actors_movies_num)
actors_num = actors_movies_num[0]
movies_num = actors_movies_num[1]
actors_names = []
actress_names = []
co_stars = {}

for i in range(actors_num):
    actress = input()
    actress_names.append(actress)
    co_stars[actress] = set()

for i in range(actors_num):
    actor = input()
    actors_names.append(actor)
    co_stars[actor] = set()

for i in range(movies_num):
    movie = input()
    names = []
    cast_num = int(input())

    for j in range(cast_num):
        name = input()
        names.append(name)

    for j in range(len(names)):
        for v in range(len(names)):

            if names[v] != names[j]:
                if (names[j] in actors_names and names[v] in actress_names) or (names[j] in actress_names and names[v] in actors_names):
                    co_stars[names[j]].add(names[v])

matched = []
unmatched = []

for key in co_stars:
    unmatched.append(key)



print(co_stars)











