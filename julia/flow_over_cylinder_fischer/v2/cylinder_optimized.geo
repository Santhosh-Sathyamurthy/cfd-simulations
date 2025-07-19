// Geometry parameters
L = 1.0;
R = 0.5;

// Domain boundaries
x_min = -2.0;
x_max = 10.0;
y_min = -3.0;
y_max = 3.0;

// Cylinder center
cx = 5.0;
cy = 0.0;

// Mesh sizes
h_far = 0.15;
h_near = 0.02;
h_mid = 0.03;

// Domain corners
Point(1) = {x_min, y_min, 0, h_far};
Point(2) = {x_max, y_min, 0, h_far};
Point(3) = {x_max, y_max, 0, h_far};
Point(4) = {x_min, y_max, 0, h_far};

// Cylinder points
Point(5) = {cx, cy, 0, h_near};
Point(6) = {cx + R, cy, 0, h_near};
Point(7) = {cx, cy + R, 0, h_near};
Point(8) = {cx - R, cy, 0, h_near};
Point(9) = {cx, cy - R, 0, h_near};

// Intermediate refinement box
Point(10) = {cx + 2*R, cy + 1.5*R, 0, h_mid};
Point(11) = {cx + 2*R, cy - 1.5*R, 0, h_mid};
Point(12) = {cx + 4*R, cy + 1.5*R, 0, h_mid};
Point(13) = {cx + 4*R, cy - 1.5*R, 0, h_mid};

// Domain lines
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};

// Cylinder
Circle(5) = {6, 5, 7};
Circle(6) = {7, 5, 8};
Circle(7) = {8, 5, 9};
Circle(8) = {9, 5, 6};

// Refinement box
Line(9) = {10, 11};
Line(10) = {11, 13};
Line(11) = {13, 12};
Line(12) = {12, 10};

// Line loops
Line Loop(1) = {1, 2, 3, 4};
Line Loop(2) = {5, 6, 7, 8};
Line Loop(3) = {9, 10, 11, 12};

// Surfaces
Plane Surface(1) = {1, 2, 3};
Plane Surface(2) = {3};

// Physical entities
Physical Line("inlet") = {4};
Physical Line("outlet") = {2};
Physical Line("walls") = {1, 3};
Physical Line("cylinder") = {5, 6, 7, 8};
Physical Surface("domain") = {1, 2};

// Mesh settings
Mesh.CharacteristicLengthFromPoints = 1;
Mesh.CharacteristicLengthExtendFromBoundary = 1;
Mesh.Algorithm = 6;  // Frontal-Delaunay
