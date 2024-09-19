from matplotlib import pyplot as plt

a = []
t = [[0],[0],[0]]
with open("debug.log", "r") as f:
    for l in f.readlines():
        print(l)
        l = l.split()
        c_ls, s_mxs = l[0], l[1]
        c_l = int(c_ls.split(":")[1])
        s_mas, s_mis = s_mxs.split(":")[1].split(",")
        s_ma, s_mi = float(s_mas), float(s_mis)
        if c_l < t[0][-1]:
            a.append(t)
            t = [[0],[0],[0]]
        t[0].append(c_l)
        t[1].append(s_ma)
        t[2].append(s_mi)
    a.append(t)

print(a)
for p in a:
    plt.plot(p[0], p[1])
    plt.plot(p[0], p[2])

plt.show()