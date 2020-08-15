import gizeh as gz
from math import pi

L = 200 
surface = gz.Surface(L, L, bg_color=(0 ,.3, .6)) 
r = 80 

yin_yang = gz.Group([
     gz.arc(r, pi/2, 3*pi/2, fill = (1,1,1)), 
     gz.arc(r, -pi/2, pi/2, fill = (0,0,0)), 
     gz.arc(r/2, -pi/2, pi/2, fill = (1,1,1), xy = [0,-r/2]), 
     gz.arc(r/2, pi/2, 3*pi/2, fill = (0,0,0), xy = [0, r/2]),  
     gz.circle(r/8, xy = [0,  +r/2], fill = (1,1,1)), 
     gz.circle(r/8, xy = [0,  -r/2], fill = (0,0,0)) ]) 
    
yin_yang.translate([L/2,L/2]).draw(surface)
surface.write_to_png("yin_yang.png")
