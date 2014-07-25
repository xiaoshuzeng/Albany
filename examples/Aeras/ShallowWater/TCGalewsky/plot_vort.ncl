load "~/local/ncl/lib/ncarg/nclscripts/csm/gsn_code.ncl"
load "~/local/ncl/lib/ncarg/nclscripts/csm/gsn_csm.ncl"

begin
;******************************************************
;  Open a netCDF file containing the grid and data from the HOMME
;  (High-Order Multiscale Modeling Environment) model, formerly
;  called SEAM (Spectral Element Atmosphere Model).
;******************************************************


ffile = "~/albany/Albany-rep/build/examples/Aeras/ShallowWater/TCGalewsky/TCGalewsky.nc";
 f=addfile(ffile,"r");
;how to access file's dimension directly???

numnodes=dimsizes(f->node_num_map);

numtimes = dimsizes(f->time_whole)

x=f->coordx;
  y=f->coordy;
  z=f->coordz;
 lon=new(numnodes,double);
lat=new(numnodes,double);

  sval = csc2s(x,y,z);

print("numnodes = "+numnodes);

  lat = sval(0,:);
  lon = sval(1,:);

  print("how many time frames = "+numtimes)


  outdata_max_error = f->vals_nod_var3(numtimes - 1,:) - f->vals_nod_var3(0,:);

  print("max abs error = " + max(abs(outdata_max_error)))

 outdata = f->vals_nod_var3(numtimes-1,:);

;******************************************************
; create plot
;******************************************************



wks = gsn_open_wks("pdf","var1") ; open workstation
gsn_define_colormap(wks,"BlAqGrYeOrReVi200"); choose colormap

res = True ; plot modifications desired
;res@gsnMaximize  = True  ; Maximize size of plot in frame

;---This resource not needed in V6.1.0
;res@gsnSpreadColors=True  ; Use full colormap, but start

 res@sfXArray= lon;  Required to tell NCL where to
 res@sfYArray= lat  ;overlay data on globe.
  
  res@cnFillOn=True;Turn on contour fill
;  res@cnFillMode  = "AreaFill"  ; Style of fill. You can also
; use "CellFill" and "RasterFill"

  res@cnLinesOn=False; Turn off contour lines
  res@cnLineLabelsOn=False; Turn off contour line labels
  res@lbLabelAutoStride=True; Clean up labelbar labels.

; res@tiMainString = names(0); Title


 mincont = min(outdata);
 maxcont = max(outdata);
 
 numcont = 100;

 res@cnLevelSelectionMode = "ManualLevels";
  res@cnMinLevelValF = mincont;
  res@cnMaxLevelValF = maxcont
  res@cnLevelSpacingF = (maxcont - mincont)/numcont;
  
  print("field's max, min = "+ max(outdata)+"   "+ min(outdata))

  res@mpOutlineOn = False;
  res@mpFillOn = False;

  map = gsn_csm_contour_map_ce(wks,outdata,res)
end
