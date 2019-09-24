noisy1 = np.array([
    [0, 1, 0, 0, 0, 0, 1, 1, 1, 0],
    [0, 1, 1, 0, 1, 1, 1, 0, 1, 0],
    [1, 0, 0, 1, 0, 1, 1, 1, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [1, 1, 1, 1, 1, 1, 0, 1, 0, 1],
    [0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
    [1, 0, 0, 1, 0, 1, 0, 0, 0, 0],
    [1, 1, 1, 1, 0, 0, 1, 0, 1, 1],
    [0, 0, 1, 0, 1, 1, 1, 1, 1, 0],
    [1, 1, 1, 1, 0, 0, 1, 0, 1, 1]
])    
a = time.time()
lb, previous_G = get_lower_bound(noisy1, None, None)
b = time.time()
print('First: {:.5f}'.format(b-a))
print(lb)
for (u, v, wt) in previous_G.edges.data('weight'):
    print(u, v, wt)
noisy2 = np.array([
    [0, 1, 0, 0, 0, 1, 1, 1, 1, 0],
    [0, 1, 1, 0, 1, 1, 1, 0, 1, 0],
    [1, 0, 0, 1, 0, 1, 1, 1, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [1, 1, 1, 1, 1, 1, 0, 1, 0, 1],
    [0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
    [1, 0, 0, 1, 0, 1, 0, 0, 0, 0],
    [1, 1, 1, 1, 0, 0, 1, 0, 1, 1],
    [0, 0, 1, 0, 1, 1, 1, 1, 1, 0],
    [1, 1, 1, 1, 0, 0, 1, 0, 1, 1]
])
a = time.time()
lb, new_G = get_lower_bound(noisy2, 5, previous_G)
b = time.time()
print('Second: {:.5f}'.format(b-a))
print(lb)
lb, new_G = get_lower_bound(noisy2, None, None)
b = time.time()
print('Second: {:.5f}'.format(b-a))
print(lb)
for (u, v, wt) in new_G.edges.data('weight'):
    print(u, v, wt)
exit()


class bcolors:
    #https://pypi.org/project/colorama/
    MAGENTA = '\033[4m\033[35m'
    ENDC = '\033[0m'

print(bcolors.MAGENTA, 'PhISCS_I in seconds: {:.3f}'.format(c_time), bcolors.ENDC)