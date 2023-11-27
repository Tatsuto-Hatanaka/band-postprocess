set terminal pdfcairo enhanced font "Helvetica,15" lw 1.2 size 5,4

set title "Band structure of 1D-chain"
#set xlabel ""
set ylabel "Energy (eV)"

x1 = 1.0
x2 = 1.5
x3 = 2.0
xmax = x3

xmin = 0
xmax = xmax
ymin = -2.5
ymax = 2.5
set xrange [xmin:xmax]
set yrange [ymin:ymax]
set xtics ("(0,0,0)" 0, "(1,0,0)" x1, "(1,1,0)" x2, "(1,1,1)" x3)
set ytics 0.5

# unset key
set key box opaque
#set key right top font",20"
set arrow 100 nohead from 0,0 to xmax,0 dt (5,5) lc "black"
set arrow 1 nohead from x1,ymin to x1,ymax dt (5,5) lc "black"
set arrow 2 nohead from x2,ymin to x2,ymax dt (5,5) lc "black"

set style fill transparent solid 0.2 noborder
set output "figures/chain1d_spinless_pbands.pdf"
plot "chain1d_spinless_bands.dat" u 1:2 w l lw 1.2 lc rgb "black" notitle,\
    "chain1d_spinless_bands.dat" u 1:2:($3/50) w circles lc rgb "red" title "orb1",\
    "chain1d_spinless_bands.dat" u 1:2:($4/50) w circles lc rgb "blue" title "orb2"
