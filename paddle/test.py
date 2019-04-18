scores_scope = {}
scope_val = []
for i in range(10):  # save the nums of score scope is (i, i+0.05]
    i = round(i/10, 2)  # remain 2 float
    scope_val.append([i, i+0.05])
    scope_val.append([i+0.05, i+0.1])
scope_val.append([0.95, 1.0])

print(scope_val)




