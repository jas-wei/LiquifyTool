from read_svg_cubics import read_svg_cubics

P, C, I, F, S, W, D, viewBox = read_svg_cubics("example.svg")

print("Control points:")
print(P)

print("Cubic segments:")
print(C)

# Get the 4 control points of the first segment
first_curve = P[C[0]]
print(first_curve)