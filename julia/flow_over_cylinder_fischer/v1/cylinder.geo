// Geometry parameters
L = 1.0;
R = 0.5;
V_inf = 1.0;

// Domain boundaries
x_min = -5.0;
x_max = 15.0;
y_min = -5.0;
y_max = 5.0;

// Cylinder center
cx = 5.0;
cy = 0.0;

// Characteristic lengths
h_far = 0.3;
h_near = 0.02;

// Domain corners
Point(1) = {x_min, y_min, 0, h_far};
Point(2) = {x_max, y_min, 0, h_far};
Point(3) = {x_max, y_max, 0, h_far};
Point(4) = {x_min, y_max, 0, h_far};

// Cylinder boundary points
Point(5) = {cx, cy, 0, h_near};
Point(6) = {cx + R, cy, 0, h_near};
Point(7) = {cx, cy + R, 0, h_near};
Point(8) = {cx - R, cy, 0, h_near};
Point(9) = {cx, cy - R, 0, h_near};

// Domain boundary lines
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};

// Cylinder arcs
Circle(5) = {6, 5, 7};
Circle(6) = {7, 5, 8};
Circle(7) = {8, 5, 9};
Circle(8) = {9, 5, 6};

// Line loops
Line Loop(1) = {1, 2, 3, 4};
Line Loop(2) = {5, 6, 7, 8};

// Surface
Plane Surface(1) = {1, 2};

// Physical entities
Physical Line("inlet") = {4};
Physical Line("outlet") = {2};
Physical Line("walls") = {1, 3};
Physical Line("cylinder") = {5, 6, 7, 8};
Physical Surface("domain") = {1};
