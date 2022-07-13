# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 23:48:04 2022

@author: USER
"""

forthga = 165.53
oxhmata = 805.03

p = 33.86

gefyraOxhmata = oxhmata * (p/100) * 13
porthmeioOxhmata = oxhmata * (100-p)/100 * 6.50

gefyraForthga = forthga * (p/100) * 20
porthmeioForthga = forthga * (100-p)/100 * 11

Totalgefyra = gefyraOxhmata + gefyraForthga
Totalporthmeio = porthmeioOxhmata + porthmeioForthga


print(Totalgefyra)
print(Totalporthmeio) 