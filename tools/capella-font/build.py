"""Draws a substitute for the missing "capella" music font (see KIRKKOKASIKIRJA.md). Needs fonttools: python build.py -> capella.ttf"""
import math
from fontTools.fontBuilder import FontBuilder
from fontTools.pens.ttGlyphPen import TTGlyphPen
S=1000/128.0   # font units per wmf unit
def ell(cx,cy,rx,ry,rot=0,n=40,rev=False):
    pts=[]
    c,s=math.cos(math.radians(rot)),math.sin(math.radians(rot))
    for i in range(n):
        a=2*math.pi*i/n*(-1 if rev else 1)
        x,y=rx*math.cos(a),ry*math.sin(a)
        pts.append((cx+x*c-y*s,cy+x*s+y*c))
    return pts
def stroke(pts,w):
    """polyline -> list of quad polygons (ccw) plus round joints"""
    polys=[]
    for (x0,y0),(x1,y1) in zip(pts,pts[1:]):
        dx,dy=x1-x0,y1-y0;L=math.hypot(dx,dy) or 1
        nx,ny=-dy/L*w/2,dx/L*w/2
        q=[(x0+nx,y0+ny),(x0-nx,y0-ny),(x1-nx,y1-ny),(x1+nx,y1+ny)]
        polys.append(q)
    for (x,y) in pts: polys.append(ell(x,y,w/2,w/2,n=10))
    return polys
def bez(p0,p1,p2,p3,n=14):
    out=[]
    for i in range(n+1):
        t=i/n;u=1-t
        out.append((u**3*p0[0]+3*u*u*t*p1[0]+3*u*t*t*p2[0]+t**3*p3[0],u**3*p0[1]+3*u*u*t*p1[1]+3*u*t*t*p2[1]+t**3*p3[1]))
    return out
def rect(x0,y0,x1,y1): return [(x0,y0),(x1,y0),(x1,y1),(x0,y1)]
def orient(poly,ccw=True):
    a=sum(poly[i][0]*poly[(i+1)%len(poly)][1]-poly[(i+1)%len(poly)][0]*poly[i][1] for i in range(len(poly)))
    return poly if (a>0)==ccw else poly[::-1]
G={}  # code -> (advance_wmf, [polys in wmf units, y up])
def add(code,adv,polys,ccw_all=True): G[code]=(adv,[orient(p) for p in polys])
# heads
add(0xe5,39,[ell(19.5,0,19.5,13.5,22)])
add(0xe4,39,[ell(19.5,0,19.5,13.5,22), orient(ell(19.5,0,15,6.5,35),False)])
add(0xe3,52,[ell(26,0,26,14), orient(ell(26,0,10,11.5,-35,),False)])
# dot (center 45 units right of origin)
add(0x2e,60,[ell(45,0,6,6)])
# flags: up flag from stem top, hanging down/right ; down flag mirrored
def flag(up=True):
    pts=bez((2,0),(4,-30),(30,-40),(24,-82))+bez((24,-82),(30,-55),(8,-46),(2,-40))[1:]
    poly=pts
    if not up: poly=[(x,-y) for x,y in pts]
    return [poly]
add(0xe6,30,flag(True)); add(0xea,30,flag(False))
# accidentals
def flat():
    stem=rect(0,-14,3.5,66)
    bowl=bez((3.5,-14),(26,6),(30,22),(14,30))+bez((14,30),(8,32),(5,29),(3.5,26))[1:]
    bowl2=bez((3.5,6),(14,14),(20,22),(3.5,18))
    return [stem]+stroke(bez((3.5,-12),(24,4),(30,20),(12,26)),5)+stroke(bez((12,26),(5,28),(3.5,24),(3.5,18)),4)
add(0x51,32,flat())
def sharp():
    p=[rect(6,-40,9.5,36),rect(16,-36,19.5,40)]
    p+=stroke([(0,-4),(26,6)],6)+stroke([(0,-18),(26,-8)],6)
    p=[orient(q) for q in p]
    return p
add(0x53,30,sharp())
def natural():
    return [rect(2,-30,5.5,38),rect(16,-38,19.5,30)]+stroke([(2,-12),(19,-4)],5)+stroke([(2,6),(19,14)],5)
add(0x52,24,natural())
# rests
add(0x4a,36,[rect(0,0,36,16)])
add(0x4b,26,stroke([(8,46),(20,26),(8,10),(20,-6)],6)+stroke(bez((20,-6),(0,-8),(2,-34),(14,-30)),5)+stroke([(8,46),(20,40)],6))
def eighth():
    p=stroke([(4,20),(16,-36)],4.5)+[ell(8,12,7,6)]+stroke(bez((8,12),(14,10),(18,6),(20,0)),3.5)
    return p
add(0x4c,24,eighth())
# time signature C and cut C
def C(cut=False):
    p=[]
    arc=[]
    for i in range(0,101):
        a=math.radians(40+280*i/100)
        arc.append((20+22*math.cos(a)*0.95,22*math.sin(a)))
    p+=stroke(arc,7)
    if cut: p+=[rect(18,-38,22,38)]
    return p
add(0x3a,44,C(False)); add(0x3b,44,C(True))
# digits (height 64 centered)
def seg(pts,w=6): return stroke(pts,w)
dig={}
def curve(*pp):
    out=[]
    for i in range(0,len(pp)-3,3): out+=bez(pp[i],pp[i+1],pp[i+2],pp[i+3])
    return out
D={ '2':seg(curve((3,16),(3,32),(30,36),(30,16))+curve((30,16),(30,2),(10,-10),(3,-30)))+seg([(3,-30),(32,-30)]),
    '3':seg(curve((3,26),(8,38),(30,36),(28,18))+curve((28,18),(26,6),(14,4),(10,3))+curve((10,3),(26,3),(32,-6),(30,-18))+curve((30,-18),(28,-36),(8,-36),(2,-24))),
    '4':seg([(26,-32),(26,34),(2,-12),(34,-12)]),
    '6':seg(curve((28,30),(14,38),(2,26),(3,-4))+curve((3,-4),(3,-40),(32,-40),(32,-18))+curve((32,-18),(32,0),(8,6),(3,-6))),
    '8':seg(ell(17,18,12,13,0,24)[:]+[ell(17,18,12,13,0,24)[0]])+seg(ell(17,-18,14,15,0,24)[:]+[ell(17,-18,14,15,0,24)[0]]),
}
for k,v in D.items(): add(ord(k),40,v)
# treble clef (origin G line)
def treble():
    P=[]
    P+=stroke(bez((18,-70),(18,-90),(-4,-90),(-4,-70)),6)   # bottom curl
    sp=bez((18,-70),(22,-40),(26,20),(20,60))
    P+=stroke(sp,6)
    P+=stroke(bez((20,60),(16,100),(26,130),(22,150)),6)   # upper loop top
    P+=stroke(bez((22,150),(16,120),(4,100),(8,70)),6)
    P+=stroke(bez((8,70),(12,30),(40,10),(42,-18)),6)
    P+=stroke(bez((42,-18),(44,-48),(6,-52),(2,-24)),6)
    P+=stroke(bez((2,-24),(0,-8),(14,4),(26,2)),6)
    P+=[ell(-4,-72,5,5)]
    return P
add(0x41,64,treble())
# bass clef (origin F line)
def bass():
    P=[ell(4,0,8,8)]
    P+=stroke(bez((4,8),(30,12),(46,-6),(38,-30)),7)
    P+=stroke(bez((38,-30),(32,-50),(14,-64),(0,-72)),7)
    P+=[ell(54,20,5,5),ell(54,-20,5,5)]
    return P
add(0x45,60,bass())
# unknown: E1 -> small triangle marker, others
add(0xe1,50,[[(0,0),(30,0),(15,30)]])
# unit conversions
glyphs={}
order=['.notdef','space']
fb=FontBuilder(1000,isTTF=True)
names=['.notdef','space']+['g%02x'%c for c in G]
cmap={}
for c in G:
    n='g%02x'%c
    cmap[0xF000+c]=n
    cmap[c]=n if c not in (0x20,) else 'space'
cmap[0x20]='space'
fb.setupGlyphOrder(names)
fb.setupCharacterMap(cmap)
adv={};tt={}
pen=TTGlyphPen(None);tt['.notdef']=pen.glyph();adv['.notdef']=(500,0)
pen=TTGlyphPen(None);tt['space']=pen.glyph();adv['space']=(300,0)
for c,(a,polys) in G.items():
    pen=TTGlyphPen(None)
    for poly in polys:
        pts=[(int(round(x*S)),int(round(y*S))) for x,y in poly]
        pen.moveTo(pts[0])
        for p in pts[1:]: pen.lineTo(p)
        pen.closePath()
    tt['g%02x'%c]=pen.glyph(); adv['g%02x'%c]=(int(a*S),0)
fb.setupGlyf(tt)
fb.setupHorizontalMetrics(adv)
fb.setupHorizontalHeader(ascent=700,descent=-400)
fb.setupNameTable({'familyName':'capella','styleName':'Regular'})
fb.setupOS2(sTypoAscender=700,sTypoDescender=-400,usWinAscent=900,usWinDescent=600)
fb.setupPost()
fb.save('capella.ttf')
print('ok',len(G))
