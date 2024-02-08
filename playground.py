# # llama2 7b
# d = 4096
# K = 32
# L = 512
# V = 30000
# a = 32
# b = 1
# # llama2 13b
# d = 5120
# K = 40
# L = 512
# V = 30000
# a = 40
# b = 1
# # llama2 70b
# d = 8196
# K = 80
# L = 512
# V = 30000
# a = 80
# b = 1
# # OPT 1.3b
# d = 2048
# K = 24
# L = 512
# V = 50272
# a = 32
# b = 1
# # OPT 66b
d = 9216
K = 64
L = 512
V = 50272
a = 72
b = 1
# C_{w}=bLd+Kb[(\frac{9}{a}+27)Ld^2 + 2L^2(3d+2a) + 12Ld]+bLV(3d+2)+L(12d^2+13d)+Vd
entire = b*L*d+K*b*((9/a+27)*L*(d**2)+2*L**2*(3*d+2*a)+12*L*d)+b*L*V*(3*d+2)+L*(12*(d**2)+13*d)+V*d

# C_{p}=bLd+Kb[(\frac{6}{a}+18)Ld^2 + 2L^2(2d+2a) + 8Ld]+bLV(2d+2)+Vd
our = b*L*d+K*b*((6/a+18)*L*(d**2)+2*L**2*(2*d+2*a)+8*L*d)+b*L*V*(2*d+2)+V*d

percent = our / entire
print(percent)
