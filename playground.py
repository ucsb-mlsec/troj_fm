# bert base
d = 768
layer = 12
n = 64
v = 30000

# bert large
# d = 1024
# layer = 24
# n = 64
# v = 30000

# llama2 7b
# d = 4096
# layer = 32
# n = 512
# v = 30000
forward = layer * (12 * n * (d ** 2) + 2 * (n ** 2) + (n ** 2) * d + 8 * n * d + 12 * (d ** 2) + 13 * d) + (
        v + 514) * d

entire = layer * (36 * n * (d ** 2) + 4 * (n ** 2) + 3 * (n ** 2) * d + 8 * n * d + 12 * (d ** 2) + 13 * d) + (
        v + 514) * d

our = layer * (24 * n * (d ** 2) + 4 * (n ** 2) + 3 * (n ** 2) * d + 8 * n * d) + v * d + n * (d ** 2)

percent = our / entire
print(percent)
