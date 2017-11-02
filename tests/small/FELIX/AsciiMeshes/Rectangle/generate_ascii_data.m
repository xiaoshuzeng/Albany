H = 1.06; % km
L = 50;   % km
W = 10;   % km

nex = 50;
ney = 10;

npx = nex + 1;
npy = ney;  % periodic in y, so one less point

np = npx*npy;

x = linspace(0,L,npx);
H_min = min(0.001,H/npx); % make sure there are no flat elements at the end

% Geometry (ice thickness and surface height)
func_thickness = @(x)(max(H_min,H*sqrt(1-x/L)));
s = func_thickness(x);

surface_height = [];
for i=1:npy
  surface_height = [surface_height; s'];
end
ice_thickness = surface_height;

sh_fid = fopen("surface_height.ascii",'w');
th_fid = fopen("ice_thickness.ascii",'w');

fprintf(sh_fid,'%d\n',np);
fprintf(th_fid,'%d\n',np);

fprintf(sh_fid,'%f\n',surface_height);
fprintf(th_fid,'%f\n',ice_thickness);

fclose(sh_fid);
fclose(th_fid);

% Hydrology (surface water input and water thickness)
% I have only a given h on a 50 elem grid (except for the first point).
% For more/less, we need to interpolate/extrapolate

dx_ref = L/50;
x_ref = linspace(dx_ref,L,50);
h_ref = [1.5963930E-02
         2.5435650E-02
         3.2408990E-02
         3.7700180E-02
         4.1870780E-02
         4.5312760E-02
         4.8265970E-02
         5.0874460E-02
         5.3226960E-02
         5.5380660E-02
         5.7374360E-02
         5.9235800E-02
         6.0985390E-02
         6.2638820E-02
         6.4208570E-02
         6.5704610E-02
         6.7133690E-02
         6.8501870E-02
         6.9817640E-02
         7.1085700E-02
         7.2310280E-02
         7.3494800E-02
         7.4642380E-02
         7.5755810E-02
         7.6837500E-02
         7.7889580E-02
         7.8914020E-02
         7.9912620E-02
         8.0886900E-02
         8.1838220E-02
         8.2768000E-02
         8.3677380E-02
         8.4567430E-02
         8.5438340E-02
         8.6291610E-02
         8.7128300E-02
         8.7949390E-02
         8.8755490E-02
         8.9547250E-02
         9.0325260E-02
         9.1090150E-02
         9.1842480E-02
         9.2582740E-02
         9.3311390E-02
         9.4028840E-02
         9.4735520E-02
         9.5431850E-02
         9.6117220E-02
         9.6757260E-02
         9.6947700E-02];

wt = interp1(x_ref,h_ref,x,"extrap");

rm = 25;  % mm/day
rs = 60;  % mm/(day*km)
sm = 0.5; % km
func_water_input = @(x)(max(0,rm-rs*abs(s-sm)));
swi = func_water_input(x);

surface_water_input = [];
water_thickness = [];
for i=1:npy
  surface_water_input = [surface_water_input; swi'];
  water_thickness = [water_thickness; wt'];
end

swi_fid = fopen("surface_water_input.ascii",'w');
wth_fid = fopen("water_thickness.ascii",'w');

fprintf(swi_fid, '%d\n', np);
fprintf(wth_fid, '%d\n', np);

fprintf(swi_fid, '%f\n', surface_water_input);
fprintf(wth_fid, '%f\n', water_thickness);

fclose(swi_fid);
fclose(wth_fid);
