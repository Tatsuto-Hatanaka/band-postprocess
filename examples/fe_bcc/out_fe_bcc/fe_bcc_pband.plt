set terminal pdfcairo enhanced font "Helvetica,15" lw 1.2 size 5,4

set title "Band structure of bcc Fe"
#set xlabel ""
set ylabel "Energy (eV)"

# G-H-N-G-P-H/P-N
x1 = 0.348857
x2 = 0.595537
x3 = 0.842216
x4 = 1.144336
x5 = 1.446455
x6 = 1.620884
xmax = x6

xmin = 0
xmax = xmax
ymin = -9
ymax = 4
set xrange [xmin:xmax]
set yrange [ymin:ymax]
set xtics ("{/Symbol G}" 0, "H" x1, "N" x2, "{/Symbol G}" x3, "P" x4, "H/P" x5, "N" x6)
set ytics 2

# unset key
set key
#set key right top font",20"
set arrow 100 nohead from 0,0 to xmax,0 dt (5,5) lc "black"
set arrow 1 nohead from x1,ymin to x1,ymax dt (5,5) lc "black"
set arrow 2 nohead from x2,ymin to x2,ymax dt (5,5) lc "black"
set arrow 3 nohead from x3,ymin to x3,ymax dt (5,5) lc "black"
set arrow 4 nohead from x4,ymin to x4,ymax dt (5,5) lc "black"
set arrow 5 nohead from x5,ymin to x5,ymax dt (5,5) lc "black"

set style fill transparent solid 0.2 noborder
set output "figures/fe_bcc_pbands_up.pdf"
plot "fe_bcc_bands_up.dat" u 1:2 w l lw 1.2 lc rgb "black" notitle,\
    "fe_bcc_bands_up.dat" u 1:2:($5/50) w circles lc rgb "red" title "d",\
    "fe_bcc_bands_up.dat" u 1:2:($4/50) w circles lc rgb "blue" title "p",\
    "fe_bcc_bands_up.dat" u 1:2:($3/50) w circles lc rgb "green" title "s"