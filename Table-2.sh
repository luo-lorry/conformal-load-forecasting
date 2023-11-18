#!/bin/bash

for  Dataset in An Ch 
do
  for Weighted in False True CP_or_CQR in CQR CP
  do
    for CP_or_CQR in CP CQR
    do
      for  SCORE in cqr qr cqr_new  
      do
        for DiGAE_or_GAE_or_LGNN in GAE DiGAE LGNN 
        do
          for GNNCONV in GraphConv SAGEConv GCNConv GATConv 
          do
            python3 code/main.py --Conditional True --SCORE $SCORE --GNNCONV GraphConv --CP_or_CQR $CP_or_CQR --DiGAE_or_GAE_or_LGNN $DiGAE_or_GAE_or_LGNN --SEEDNUM 1 --Weighted $Weighted --Dataset $Dataset
          done
        done
      done
    done
  done
done
