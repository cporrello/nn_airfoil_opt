% MATLAB SCRIPT FOR VISUALIZING OPTIMIZED AIRFOIL PROFILE SINCE I LIKE THE
% DEFAULT MATLAB PLOT SETTINGS BETTER THAN MATPLOTLIB.PYPLOT
clc; clear; close all;

% Load data for optimized airfoil profiles
xs_fcn = readmatrix("xs_fcn.csv"); 
ys_fcn = readmatrix("ys_fcn.csv");
ps_fcn = readmatrix("ps_fcn.csv");
xs_geofno = readmatrix("xs_geofno.csv"); 
ys_geofno = readmatrix("ys_geofno.csv");
ps_geofno = readmatrix("ps_geofno.csv");

% Compute pressure coefficients
pinf=1.0; Minf=1.0; gamma=1.4;
cp_fcn = ps_fcn;  % (ps_fcn-pinf)/(0.5*gamma*pinf*Minf^2.0);
cp_geofno = ps_geofno;  % (ps_geofno - pinf)/(0.5*gamma*pinf*Minf^2.0);

% Plotting routine for airfoils
t = tiledlayout(3, 1);

ax2 = nexttile([2 1]);
hold on;
plot(xs_fcn, cp_fcn, "DisplayName", "FCN")
plot(xs_geofno, cp_geofno, "DisplayName", "Geo-FNO")
ylabel("$C_p$", "FontSize", 16, "Interpreter", "latex")
grid on

ax1 = nexttile();
hold on;
plot(xs_fcn, ys_fcn, "DisplayName", "FCN")
plot(xs_geofno, ys_geofno, "DisplayName", "Geo-FNO")

% Display options
grid on


% Display options
xlabel("$x/c$", "FontSize", 16, "Interpreter", "latex")
ylabel("$y/c$", "FontSize", 16, "Interpreter", "latex")
legend("Interpreter", "latex", "FontSize", 14, "Location", "Southeast")
grid on

% Link axes and axes display options
linkaxes([ax1, ax2], "x")
axis(ax1, [0 1 -0.1 0.1])
set(ax2, 'XTicklabel', [])

% Legend options
hl = legend("Interpreter", "latex", "FontSize", 14, "Location", "Southeast");
hl.Layout.Tile = "North";
hl.Orientation = "horizontal";

% Resize grid
t.TileSpacing = "tight";
t.Padding = "tight";

% Save figure
% exportgraphics(gcf, "optimization_tilechart.pdf", "ContentType", "vector")