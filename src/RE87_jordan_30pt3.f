C 
C Axisymmetric hurricane model originally by Rotunno & Emanuel:
C 
C Rotunno, R., and K.A. Emanuel, 1987:  An air-sea interaction theory
C for tropical cyclones, Part II: Evolutionary study using axisymmetric
C nonhydrostatic numerical model.  J. Atmos. Sci., 44, 542-561,
C doi:10.1175/1520-0469(1987)044<0542:AAITFT>2.0.CO;2
C
C Incorporating modifications by Craig:

C Craig, G. C., 1995: Radiation and polar lows. Quart. J. Roy. Meteor.
C Soc., 121 (521), 79–94, doi:10.1002/qj.49712152105

C Craig, G. C., 1996: Numerical experiments on radiation and tropical
C cyclones. Quart. J. Roy. Meteor. Soc., 122 (530), 415–422,
C doi:10.1002/qj.49712253006
C
C This version used by Harris et al. for development of moist
C local APE budgets.
C
C The latest version of the RE87 model is available
C at https://emanuel.mit.edu/products
C
C  INPUT PARAMETERS/typical value/
C
C  IREAD/0/     - number of dumps to read from input file (unit 10) and
C               - copied to output file (unit 11).
C               - dump number IREAD is initial state
C  ISTART/1/    - first timestep of run 
C  ISTOP/5/     - last timestep of run
C
C  IPRINT/3600/ - interval (number of timesteps) to print fields
C  IPLOT/5/     - interval to output fields for plotting (unit 20)
C  ITMAX/180/   - interval to determine maximum values of v and w
C  IBUFF/1800/  - interval to dump fields to output file (unit 11)
C
C  F/.000136/   - Coriolis parameter
C  CD/.0011/    - surface drag coefficient (momentum)
C  CE/.0011/    - surface drag coefficient (heat and moisture)
C  VGUST/1./    - 'gustiness factor', square of minimum surface wind 
C                 for calculation of heat and moisture fluxes
C
C  OBRAD/.TRUE./- apply radiation outer boundary condition if true,
C                 otherwise wall
C  CSTAR/30/    - wave speed for open boundary condition
C  RALPHA/0./   - sponge layer damping coefficient for outer boundary
C
C  XVL/200/     - vertical mixing length
C  ALPHA/.01/   - sponge layer damping coefficient for upper boundary
C  XLDCP/2500/  - latent heat of vaporization
C  RADT/86400/  - time scale for radiative damping
C
C  DT/20/       - long timestep
C  NS/10/       - number of short timesteps per long step
C  EPS/.1/      - time smoothing constant
C  EP/.1/       - something to do with small timestepping
C
C  RB/1500 000/ - horizontal size of domain
C  ZB/15 000/   - vertical size of domain
C  NZSP/2/      - thickness of sponge layer in vertical levels
C  NRSP/30/     - thickness of outer sponge layer
C  NZTR/7/      - vertical level of top of initial vortex
C
C  VM/0/        - maximum wind speed for upper level cold core vortex
C  RM/500 000/  - radius of maximum wind for upper level vortex
C  VSO/10/      - maximum wimd speed for initial low level vortex
C  RSO/50 000/  - radius of maximum wind for initial low level vortex
C
C  TB/267.3,274.2,.../  - initial temperature sounding
C  QVB/.0013,.0007,.../ - initial water vapor mixing ratio profile
C
C  TBS/279/     - sea surface temperature
C  TBT/410/     - upper boundary temperature
C  PDS/1000/    - ambient surface pressure
C  QVBT/0/      - upper boundary moisture content
C 
      PROGRAM HURAXI
      IMPLICIT NONE
      INTEGER M,N,MP1,MM1,NP1,NM1
      PARAMETER(M=1620,N=44,MP1=M+1,MM1=M-1,NP1=N+1,NM1=N-1)
      INTEGER IREAD,ISTART,ISTOP,IPRINT,IPLOT,ITMAX,IBUFF,IBUFF2,
     $ NS,NZSP,NRSP,NZTR,
     $ I,J,K,MOLD,NOLD,ISTOLD,ISTOPOLD,IBUFFOLD,ITIME,
     $ NSMALL,IWMAX,JWMAX,IVMAX,JVMAX,NWET
      REAL QRLW(M,N), QRSW(M,N), UB(N), QI(M,N), QR(M,N),
     $ U(MP1,N), V(M,N) , W(M,NP1), P(M,N), T(M,N)  , QV(M,N) , QL(M,N),
     $ PRESS(M,N), RC2(N), ZC2(N), THETAV(M,N), SUBGT(M,N), SUBGQV(M,N),
     $ CPTDR(N), CPTDZ(N), QS(N)  ,
     $ RHOTVB(N), RHOTVW(NP1), RHOT(N), RHOW(NP1), RTB(N), RQVB(N) ,
     $ DTSR(M), DTSZ(M,N), DTSRW(N), DTSZW(N), DTLR(M), DTLZ(N),DTSV(M),
     $ A(N), B(N), C(N), D(N), E(N), WS(M,NP1), PS(M,N),ZTAU(N),RTAU(M)
      REAL F,CSTAR,RALPHA,XVL,ALPHA,RADT,TDIF,DT,EPS,EP,RB,ZB,
     $ VM,RM,VSO,RSO,TBS,TBT,PDS,QVBT,
     $ DR,DZ2,DR2,PI,C2,CP,CV,RD,
     $ PNS,ESS,QSP,QVBS,TVBS,CC1,CC2,CC5,CC3,TIME,TSTOP,
     $ PNT,PDT,ES,PNW,TVW,ZSP,RSP,DTSF,DTSG,DTL2,CC4,
     $ HZT,RHZT,PSO,PSM,ALSO,BESO,ALSM,BESM,RSRO,RSRM,FS,FM,ZSZT,
     $ PDST,PNST,TEMP,QSS,R1,QVD,UTEMP,VTEMP,WTEMP,PTEMP,TTEMP,
     $ QVTEMP,QLTEMP,QITEMP,QRTEMP,WMAX,VMAX,TINHRS
      LOGICAL OBRAD, WRT
      REAL
     $ UA(MP1,N) , WA(M,NP1), VA(M,N)  , TA(M,N), QVA(M,N), QLA(M,N),
     $ U1(MP1,N) , V1(M,N)  , W1(M,NP1), T1(M,N), QV1(M,N), QL1(M,N),
     $ QIA(M,N), QRA(M,N), QI1(M,N), QR1(M,N),
     $ XKM(M,NP1), P1(M,N)  , R(MP1)   , RS(M)  , Z(NP1)  , ZS(N)   ,
     $ RTBW(N)   , RQVBW(N) , TAU(M,N) , TB(N), QVB(N), PN(N), PD(N),
     $ TSURF(M), QSURF(M), PNSURF(M),RDR, RDZ, RDR2, RDZ2, DTS, DTL, DZ,
     $ XVL2, XHL2, CD, CE, CT, XLDCP, XKAPPA, A1, G, DUM2(M,N)
     $ ,CERS(M), CDRS(M), CTRS(M), VGUST, DTHETAV(M,N), DUM1(M,N)

      REAL
     $ UAOUT(MP1,N),VAOUT(M,N),WAOUT(M,NP1),
     $ QVAOUT(M,N),QLAOUT(M,N),TAOUT(M,N),
     $ TZOUT(M,NP1),QVZOUT(M,NP1),
     $ MICT(M,40),MICQV(M,40),tempold(M,N), change(M,N),lagt(M,N),
     $ qvold(M,N),qlold(M,N),qrold(M,N),qiold(M,N),qvchange(M,N),
     $ qlchange(M,N),qichange(M,N),qrchange(M,N)



C 
C This common block DOES NOT pass any variables to the subroutines. It's only
C purpose is to prevent problems when the program searches for an array element
C which is outside the array.
      COMMON/PROB/ DUM1, U, W, V, T, QV, QL, QI, QR, DUM2
C
      DATA
     $ IREAD/0/, ISTART/1/, ISTOP/150005/,
     $ IPRINT/180000/, IPLOT/180000/, ITMAX/150/, IBUFF/600/,
     $ IBUFF2/60000/,
     $ F/.0000614/, CD/.0011/, CE/.0012/, CT/.001/, VGUST/1./,
     $ OBRAD/.FALSE./, CSTAR/30./, RALPHA/0.003/,
     $ XVL/200./, ALPHA/0.01/, XLDCP/2488.8/, RADT/43200./,
     $ DT/6. /, NS/10/, EPS/0.1/, EP/0.1/, NWET/40/,
     $ RB/4050000./, ZB/27500./, NZSP/8/, NRSP/360/, NZTR/24/,
     $ VM/0.0/, RM/500000./, VSO/12./, RSO/75000./
C      DATA
C     $ TB/300.643, 305.796, 311.701, 317.361, 323.305, 328.874,
C     $ 333.636, 337.505, 340.726, 343.821, 347.762, 355.278,
C     $ 369.884, 393.034, 425.129, 460.385, 496.593, 533.422,
C     $ 570.984, 609.369, 649.088, 689.834/
       DATA
     $ TB/299.347, 301.472, 303.971, 306.768, 309.787, 312.952, 316.187,
     $ 319.416, 322.562, 325.55, 328.302, 330.744, 332.8, 334.473,
     $ 335.879, 337.143, 338.393, 339.754, 341.351, 343.312, 345.773,
     $ 349.0, 353.348, 359.167, 366.81, 376.513, 388.133, 401.459, 
     $ 416.279, 432.381, 449.552, 467.579, 486.252, 505.357, 524.682,
     $ 544.015, 563.144, 581.857, 598.914, 618.470, 638.267, 658.417,
     $ 678.725, 700.588/
C      DATA
C     $ QVB/0.0150949, 0.00934729, 0.00592932, 0.00370308, 0.00248573,
C     $ 0.00128430, 0.000531237, 0.000191826, 7.47406e-05, 2.74812e-05,
C     $ 7.18929e-06, 3.81717e-06, 3.00000e-06, 3.00000e-06, 3.00000e-06,
C     $ 3.00000e-06, 3.00000e-06, 3.00000e-06, 3.00e-06, 3*3.00e-06/
       DATA
     $ QVB/0.0169637, 0.0133191, 0.0106305, 0.00844096, 0.00668493,
     $ 0.00529694, 0.00421152, 0.00336317, 0.00268641, 0.00211576,
     $ 0.00158572, 0.00109603, 0.000719503, 0.000446384, 0.000276679,
     $ 0.000162555, 0.000104012, 6.29258e-05, 3.92961e-05, 2.24082e-05,
     $ 1.22623e-05, 6.34626e-06, 4.66020e-06, 3.61288e-06, 3.20429e-06,
     $ 19*3.00000e-06/
      DATA TBS/302.15/, TBT/711.969/, PDS/1015.1/, QVBT/.000003/
      DR  = RB / FLOAT(M)
      DZ  = ZB / FLOAT(N)
      DTS = DT / FLOAT(NS)
      DTL = DT
      DZ2 = DZ * DZ
      DR2 = DR * DR
      RDR = 1. / DR
      RDZ = 1. / DZ
      RDR2= 1. / DR2
      RDZ2= 1. / DZ2
      PI  = 4. * ATAN(1.)
      C2  = 90000.
      CP  = 1004.5
      CV  = 717.5
      RD  = 287.
      XKAPPA= RD / CP
      G     = 9.81
      A1    = 7.5 * ALOG(10.)
      PNS   = (PDS/1000.) ** XKAPPA
      ESS   =  6.11 * EXP( A1* (TBS*PNS-273.)/( TBS*PNS-36.)  )
      QSP   = .622 * ESS /( PDS -ESS )
      QVBS  = 0.027939
      TVBS  = TBS * ( 1. + .61 * QVBS )
      CC1 = .61 * G * DZ /(2.* CP * TVBS )/(1. + .61 *QVBS   )
      CC2 = 1. - G * DZ / (2. * CP * TVBS * PNS )
      CC5 = - TBS / PNS
      CC3   =  G * DZ / CP
      XHL2  = .04 * DR2
      XVL2  = XVL * XVL
      TIME  = DT * ( FLOAT(ISTART) - 1. )
C
C        DEFINE GRID ARRAYS
C
      DO 10 I = 1 , MP1
      R(I) = ( FLOAT(I) - 1.0 ) * DR
      IF( I .NE. MP1 )  RS(I) = ( FLOAT(I) - 0.5 ) * DR
   10 CONTINUE
      DO 20 J = 1 , NP1
      Z(J) = ( FLOAT(J) - 1.0 ) * DZ
      IF( J .NE. NP1)  ZS(J) = ( FLOAT(J) - 0.5 ) * DZ
   20 CONTINUE
      TSTOP  = DT * FLOAT(ISTOP)
      WRITE(*,*) '      TSTART = ',TIME
      WRITE(*,*) '       TSTOP = ',TSTOP
      WRITE(*,*) '      ISTART = ',ISTART
      WRITE(*,*) '       ISTOP = ',ISTOP
      WRITE(*,*) '       IBUFF = ',IBUFF
      WRITE(*,*) '      IPRINT = ',IPRINT
      WRITE(*,*) '          DT = ',DT
      WRITE(*,*) '         EPS = ',EPS
      WRITE(*,*) '          XL = ',XVL
      WRITE(*,*) '           F = ',F
      WRITE(*,*) '          RB = ',RB
      WRITE(*,*) '          ZB = ',ZB
      WRITE(*,*) '           M = ',M
      WRITE(*,*) '           N = ',N
      WRITE(*,*) '          CD = ',CD
      WRITE(*,*) '          CE = ',CE
      WRITE(*,*) '       IPLOT = ',IPLOT
      WRITE(*,*) '       IREAD = ',IREAD
C
C        BASE STATE (HYDROSTATIC EQUATION FOR PN)
C
      PN(1) = PNS -.5*CC3/(.5*(TB(1)+TBS)*(1.+.61*.5*(QVB(1)+QVBS)))
      PD(1) = 1000. * PN(1) ** ( 1. / XKAPPA )
      DO 40 J = 2 , N
      PN(J) = PN(J-1) - CC3/(.5*(TB(J)+TB(J-1))*
     $                          (1. + .61*.5*(QVB(J)+QVB(J-1)) ) )
      PD(J) = 1000. * PN(J) ** ( 1. / XKAPPA )
   40 CONTINUE
      PNT = PN(N) -.5*CC3/(.5*(TB(N)+TBT)*(1.+.61*.5*(QVB(N)+QVBT)))
      PDT = 1000. * PNT ** ( 1. / XKAPPA )
      PRINT 945
  945 FORMAT(1H ,'     Z           P            PI         PT
     $QVB         QVS')
      PRINT 940, ZB, PDT, PNT, TBT, QVBT, 0.
      DO 50 J = N , 1, -1
      ES = 6.11 * EXP( A1* ( PN(J)*TB(J) - 273.)/( PN(J)*TB(J) - 36.))
      QS(J) = .622 * ES /( PD(J)-ES )
      PRINT 940, ZS(J), PD(J), PN(J), TB(J), QVB(J), QS(J)
  940 FORMAT(1H , 6(F11.5,1X) )
   50 CONTINUE
      PRINT 940, 0., PDS, PNS, TBS, QVBS, QSP
C
C       ARRAYS FOR SMALL TIME STEP AND BUOYANCY CALCULATION
C
      DO 60 J = 1 , N
      RC2(J)    =  RDR*DTS* C2/( CP * TB(J)* (1. + .61 * QVB(J) ) )
      CPTDR(J)  =  DTS*RDR * CP * TB(J)* (1. + .61 * QVB(J) )
      RTB(J)    = .5  * G / TB(J)
      RQVB(J)   = .61 * G * .5 / ( 1. + .61 * QVB(J) )
      RHOTVB(J) = (1000./RD) * PN(J) ** (CV /RD)
      RHOT(J)   = 100. * RHOTVB(J) / ( TB(J) * ( 1. + .61 * QVB(J) ) )
      ZC2(J)    = RDZ*RC2(J)/RHOTVB(J)/RDR
      IF( J.EQ.1 )  GO TO 60
      PNW = .5 * ( PN(J) + PN(J-1) )
      TVW = .5 * ( TB(J) + TB(J-1) )*( 1. + .305*( QVB(J)+QVB(J-1) ) )
      RHOTVW(J)=  (1000./RD) * PNW ** (CV /RD)
      RHOW(J)  = 100. * RHOTVW(J) / TVW
      RTBW(J) = G / ( TB(J) + TB(J-1) )
      RQVBW(J)= .305*G/( 1. + .305*(QVB(J)+QVB(J-1)) )
      CPTDZ(J)=DTS*RDZ*CP*.5*(TB(J)+TB(J-1))*(1.+.305*(QVB(J)+QVB(J-1)))
   60 CONTINUE
      RHOTVW(1  )= (1000./RD) * PNS ** (CV /RD)
      RHOTVW(NP1)= (1000./RD) * PNT ** (CV /RD)
      RHOW(1  ) = 100. * RHOTVW(1  ) / TVBS
      RHOW(NP1)=  100. * RHOTVW(NP1) / TBT
C
C         SPONGE LAYER DAMPING COEFFICIENT
C
      DO 63 J = 1 , N
      ZSP = ( ZS(J) - ZS(N-NZSP) ) / ( ZS(N) - ZS(N-NZSP) )
      IF(ZSP.LT.0.) ZTAU(J) = 0.
      IF(ZSP.GE.0..AND.ZSP.LE..5) ZTAU(J)=-.5*ALPHA*(1.-COS(ZSP*PI) )
      IF(ZSP.GT..5) ZTAU(J)=-.5*ALPHA*( 1. + PI*(ZSP-.5) )
   63 CONTINUE
      DO 64 I = 1 , M
      RSP = ( RS(I) - RS(M-NRSP) ) / ( RS(M) - RS(M-NRSP) )
      IF(RSP.LT.0.) RTAU(I) = 0.
      IF(RSP.GE.0..AND.RSP.LE..5) RTAU(I)=-.5*RALPHA*(1.-COS(RSP*PI) )
      IF(RSP.GT..5) RTAU(I)=-.5*RALPHA*( 1. + PI*(RSP-.5) )
   64 CONTINUE
      DO 65 J = 1 , N
      DO 65 I = 1 , M
      TAU(I,J) = MIN( ZTAU(J) , RTAU(I) )
   65 CONTINUE
C
C         ARRAYS TO INCREASE EFFICIENCY
C
      DTSF = .5 * DTS * F
      DTSG = .5 * DTS * G
      DTL2 = .5 * DTL
      DO 70 I = 1 , M
      DTLR(I)  = .5  * DTL * RDR /RS(I)
      DTSV(I)  = .5  * DTS  / RS(I)
      IF( I .NE. 1 )  DTSR(I)  = .25 * DTS * RDR / R(I)
   70 CONTINUE
      DO 75 J = 1 , N
      DTLZ(J)  = .5  * DTL * RDZ / RHOT(J)
      DTSZW(J) = .25 * DTS * RDZ / RHOW(J)
      DTSRW(J) = .25 * DTS * RDR / RHOW(J)
   75 CONTINUE
      DO 80 J = 1 , N
      DO 80 I = 2 , M
      DTSZ(I,J)  = .25 * DTS * RDZ /( RHOT(J) * R(I) )
   80 CONTINUE
C
C        ARRAYS FOR SEMI - IMPLICIT SMALL TIME STEP
C
      CC4=.25*(1.+EP)**2
      DO 85 J = 2 , N
      A(J)=   CC4*CPTDZ(J)*RHOTVW(J+1)*    ZC2(J)
      B(J)=1.+CC4*CPTDZ(J)*RHOTVW(J  )*(ZC2(J)+ZC2(J-1))
      C(J)=   CC4*CPTDZ(J)*RHOTVW(J-1)*   ZC2(J-1)
   85 CONTINUE
      E(1)=0.
      D(1)=0.
      DO 90 J = 2 , N
      E(J) = A(J)/(B(J)-C(J)*E(J-1))
   90 CONTINUE
C
C       INITIAL CONDITIONS
C
C
C      BAROCLINIC VORTEX
C
      HZT = ZS(N-NZSP)/ZS(NZTR)
      RHZT=1./HZT
      PSO=2.
      PSM=5.
      ALSO=PSO/((PSO-1.)**(1.-1./PSO))
      BESO =(PSO-1.)**(1./PSO)
      ALSM=PSM/((PSM-1.)**(1.-1./PSM))
      BESM =(PSM-1.)**(1./PSM)
      DO 95 I = 1 , M
      RSRO=RS(I)/(RSO*BESO)
      RSRM=RS(I)/(RM *BESM)
      FS = ALSO*RSRO/(1.+RSRO**PSO)
      FM = ALSM*RSRM/(1.+RSRM**PSM)
      DO 95 J = 1 , N
      ZSZT=ZS(J)/ZS(NZTR)
      V(I,J)=VM*ZSZT*FM + VSO*FS*(1.-ZSZT)
      IF( J. LE. NZTR) GO TO 96
      V(I,J)= VM*HZT*FM*(1.-ZS(J)/ZS(N-NZSP))/(1.-RHZT)
     $        *(1.+RHZT-(1.-ZS(J)/ZS(N-NZSP))/(1.-RHZT))+VSO
     $ *HZT*FS*(1.-ZS(J)/ZS(N-NZSP))*(RHZT-ZS(J)/ZS(N-NZSP))/(1.-RHZT)
   96 CONTINUE
      IF ( ( J .GT. N-NZSP) .OR. ( I .GT. M-NRSP) ) V(I,J)=0.
   95 CONTINUE
      DO 100 J = 1 , N
      DO 100 I = 1 , M
      V1(I,J)   = V(I,J)
      QV(I,J)   = QVB(J)
      QV1(I,J)  = QV(I,J)
      QL(I,J)   = 0.
      QL1(I,J)  = 0.
      QI(I,J)   = 0.
      QI1(I,J)  = 0.
      QR(I,J)   = 0.
      QR1(I,J)  = 0.
      QRLW(I,J)  = 0.
      QRSW(I,J)  = 0.
  100 CONTINUE
C
C       INTEGRATE INWARD TO ADJUST PRESSURE TO VORTEX
C
      DO 105 J = 1 , N
       P(M,J) = 0.
       P1(M,J)=P(M,J)
       DO 105 I = M-1, 1, -1
        P(I,J) = P(I+1,J) - ( DTS / CPTDR(J) ) * .5 *
     $   ( V(I,J)*V(I,J)/RS(I) + V(I+1,J)*V(I+1,J)/RS(I+1)
     $    + F * ( V(I,J) + V(I+1,J) )            )
  105 CONTINUE
      DO 106 I = 1 , M
      DO 107 J = 2 , NM1
      T(I,J) = TB(J) + .25*( CPTDZ(J+1)*(P(I,J+1)-P(I,J  ))
     $                  +CPTDZ(J  )*(P(I,J  )-P(I,J-1)) )/(DTS*RTB(J))
      T1(I,J)=T(I,J)
  107 CONTINUE
      T(I,1) = TB(1) + .5*CPTDZ(2)*(P(I,2)-P(I,1  ))/(DTS*RTB(1))
      T(I,N) = TB(N) + .5*CPTDZ(N)*(P(I,N)-P(I,NM1))/(DTS*RTB(N))
      T1(I,1)=T(I,1)
      T1(I,N)=T(I,N)
  106 CONTINUE
      DO 110 J = 1 , N
      DO 110 I = 1 , MP1
      U(I,J)  = 0.0
      U1(I,J) = U(I,J)
  110 CONTINUE
      DO 120 I = 1 , M
      DO 120 J = 1 , NP1
      W(I,J)  = 0.0
      W1(I,J) = W(I,J)
  120 CONTINUE
      DO 505 J = 1 , N
      UA(1,J) = 0.
      UA(MP1,J) = 0.
      UB(J) = 0.
  505 CONTINUE
      DO J = 1,N
         DO I = 1,M
            DTHETAV(I,J) = 0.
         ENDDO
      ENDDO
      WRITE(49) V, P, T, QV
C
C        READ IN DUMP FILE
C
      IF (IREAD.LE.0) THEN
         WRITE(11) M,N,ISTART,ISTOP,IBUFF
         WRITE(11) RB,ZB,DT,PN,TB,QVB
         WRITE(12) 120,24,ISTART,ISTOP,IBUFF2
         WRITE(12) RB,ZB,DT,PN(1:24),TB(1:24),QVB(1:24)
      ELSE
         READ(10) MOLD,NOLD,ISTOLD,ISTOPOLD,IBUFFOLD
         READ(10)
         WRITE(11) M,N,ISTOLD,ISTOP,IBUFF
         WRITE(11) RB,ZB,DT,PN,TB,QVB
         WRITE(12) 120,24,ISTOLD,ISTOP,IBUFF2
         WRITE(12) RB,ZB,DT,PN(1:24),TB(1:24),QVB(1:24)
         DO 8000 K = 1, IREAD
            READ(10) U,V,W,T,QV,QL,QI,QR,P,QRLW,QRSW
            WRITE(11) U,V,W,T,QV,QL,QI,QR,P,QRLW,QRSW
            WRITE(12) U(1:120,1:24),V(1:120,1:24),W(1:120,1:24),
     &      T(1:120,1:24),QV(1:120,1:24),QL(1:120,1:24),P(1:120,1:24),
     &      QRLW(1:120,1:24)
 8000    CONTINUE
         DO 8004 J = 1, N
            DO 8002 I = 1, M
               U1(I,J) = U(I,J)
               V1(I,J) = V(I,J)
               W1(I,J) = W(I,J)
               T1(I,J) = T(I,J)
               QV1(I,J) = QV(I,J)
               QL1(I,J) = QL(I,J)
               QI1(I,J) = QI(I,J)
               QR1(I,J) = QR(I,J)
               P1(I,J) = P(I,J)
 8002       CONTINUE
 8004    CONTINUE
         DO J = 1, N
            U1(MP1,J) = U(MP1,J)
         END DO
         DO 8006 I = 1, M
             W1(I,NP1) = W(I,NP1)
 8006    CONTINUE
      END IF







C===================================================================
C        BEGIN TIME MARCH
C
C===================================================================
      DO 500 ITIME = ISTART , ISTOP
      TIME = TIME + DT
C      WRITE(*,*) 'TIME = ',TIME
      IF (MOD(TIME,3600.) .EQ. 0) WRITE(*,*) 'TIME (HRS) = ',TIME,
     &  NINT(TIME/3600.)


C store temperature at beginning of each timestep
      change = 0
      tempold = T




C
C       SURFACE TEMP AND WATER-VAPOR MIXING RATIO
C
      DO 6 I = 1 , M
      CDRS(I)=.0011+4.E-5 *SQRT(.25*(U1(I,1)+U1(I+1,1))**2
     $ +V1(I,1)**2)
C      CDRS(I) = CD
      CTRS(I)= CT
      CERS(I) = 1.2*CT
    6 CONTINUE
      DO 515 I = 1 , M
      PNSURF(I) = (P1(I,1)-CC1*(QV1(I,1)-QVB(1)))/CC2
      PDST = 1000. * (PNS +PNSURF(I)) ** (1./.2856)
      QSURF(I) = .622 * ESS /(PDST -ESS)
      TSURF(I) = TBS + CC5 *PNSURF(I)
  515 CONTINUE
      CALL DIFF
     $ (UA , WA, VA  , TA, QVA, QLA,
     $ U1 , V1  , W1, T1, QV1, QL1,
     $ XKM, P1  , R   , RS  , Z  , ZS,
     $ RTBW   , RQVBW , TAU , TB, QVB, PN, PD,
     $ TSURF  , QSURF , RDR, RDZ, RDR2, RDZ2 , DTS, DTL, DZ   ,
     $ XVL2, XHL2, CD, CE, CT, XLDCP, XKAPPA, A1, G
     $ ,CERS, CDRS, CTRS, VGUST, QIA, QRA, QI1, QR1,
     $ UAOUT,VAOUT,WAOUT,QVAOUT,QLAOUT,TAOUT,TZOUT,
     $ QVZOUT)

C       IF (MOD(ITIME,IBUFF2).EQ.0) 
C     &   WRITE(13) UAOUT(1:120,1:24),VAOUT(1:120,1:24),
C     &   WAOUT(1:120,1:24),QVAOUT(1:120,1:24),QLAOUT(1:120,1:24),
C     &   TAOUT(1:120,1:24),TZOUT(1:120,1:24),QVZOUT(1:120,1:24),
C     &   UAOUT(1:120,1:24)




C
C     RADIATION (R < 1 DEG/DAY)
C     NO RADIATION
 
      DO 150 I = 1 , M
      DO 150 J = 1 , N
      SUBGT(I,J)=TA(I,J)
      SUBGQV(I,J)=QVA(I,J)
      TDIF=T1(I,J)-TB(J)
      QRLW(I,J) = -DTL*TDIF/RADT
      IF(TDIF.GT. 1.) QRLW(I,J) = -DTL/RADT
      IF(TDIF.LT.-1.) QRLW(I,J) =  DTL/RADT
  150 CONTINUE

      IF (MOD(ITIME,IBUFF).EQ.0) 
     &WRITE(16) TSURF, QSURF, PNSURF

      IF (ITIME.EQ.1) THEN
      WRITE(24) M,N,NWET,ISTART,ISTOP,IBUFF,NRSP,NZSP
      WRITE(24) DR,DZ,DT,DTS,F,G,RD,CP,XLDCP*CP,C2,
     &TBS,TBT,PNS,PNT,QVBS,QVBT
      WRITE(24) RHOT,RHOW,R,RS,PN,TB,QVB,RHOTVW
      ENDIF
      IF (MOD(ITIME+1,IBUFF).EQ.0) 
     &WRITE(24) U, V, W, P, T, QV, QL, QR, QI
      IF (MOD(ITIME,IBUFF).EQ.0) 
     &WRITE(24) U, V, W, P, T, QV, QL, QR, QI,
     &UA, VA, WA, TA, QVA, QLA, QRA, QIA, DTHETAV, TZOUT, QVZOUT, QRLW
      IF (MOD(ITIME-1,IBUFF).EQ.0 .AND. ITIME.GT.5) 
     &WRITE(24) U, V, W, P, T, QV, QL, QR, QI

C
C        FORCING FOR U EQUATION
C
      DO 510 J = 1 , N
      DO 510 I = 2 , M
      UA(I,J)    = UA(I,J)
     1 - DTSR(I)   *(
     2 ( R(I+1)*U(I+1,J)+R(I  )*U(I  ,J) )*( U(I+1,J)-U(I  ,J) )
     3+( R(I  )*U(I  ,J)+R(I-1)*U(I-1,J) )*( U(I  ,J)-U(I-1,J) ) )
     4 - DTSZ(I,J) *(
     5 RHOW(J+1)*(RS(I)*W(I,J+1)+RS(I-1)*W(I-1,J+1))*(U(I,J+1)-U(I,J))
     6+RHOW(J  )*(RS(I)*W(I,J  )+RS(I-1)*W(I-1,J  ))*(U(I,J)-U(I,J-1)))
     6+ DTSV(I) * V(I,J)*V(I,J) +  DTSV(I-1) * V(I-1,J)*V(I-1,J)
     7  +  DTSF    * ( V(I,J) + V(I-1,J) )
  510 CONTINUE
C
C       OUTER BOUNDARY
C
      IF ( OBRAD ) THEN
      DO 525 J = 1 , N
      UA(MP1,J) =
     $ -AMAX1( U(MP1,J) + CSTAR , 0. ) * DTL * RDR * (U1(MP1,J)-U1(M,J))
     $+ DTL * (V(M,J)*V(M,J)/RS(M) +  F * V(M,J)   )
  525 CONTINUE
C
C       DRAG AT J = 1
C
      UA(MP1,1) = UA(MP1,1)
     $ - CD * DTL * RDZ * U1(MP1,1) * SQRT( U1(MP1,1)**2
     $  + V1(M,1)**2 )
      END IF
C
C       FORCING FOR W EQUATION
C
      DO 550 J = 2 , N
      DO 550 I = 1 , MM1
      WA(I,J)    = WA(I,J)
     1 -DTSRW(J)*(
     2 R(I+1)*(RHOT(J )*U(I+1,J)+RHOT(J-1)*U(I+1,J-1))*(W(I+1,J)-W(I,J))
     3+R(I)*(RHOT(J)*U(I,J)+RHOT(J-1)*U(I,J-1))*(W(I,J)-W(I-1,J)))/RS(I)
     4 -DTSZW(J)*(
     5 (RHOW(J+1)*W(I,J+1)+RHOW(J  )*W(I  ,J  ))*(W(I,J+1)-W(I,J  ))
     6+(RHOW(J-1)*W(I,J-1)+RHOW(J  )*W(I  ,J  ))*(W(I,J  )-W(I,J-1)) )
     7  + DTS*RTBW(J)  * ( T(I,J) - TB(J)  + T(I,J-1)  - TB(J-1)  )
     8  + DTS*0.305*G* (QV(I,J) - QVB(J) + QV(I,J-1) - QVB(J-1) )
     9  - DTSG*(QL(I,J)+QL(I,J-1)+QI(I,J)+QI(I,J-1)+QR(I,J)+QR(I,J-1))
  550 CONTINUE
C
C       OUTER BOUNDARY
C
      IF ( OBRAD ) THEN
      DO 520 J = 2 , N
      UB(J) =      AMAX1( RHOT(J  ) * ( U(MP1,J) + U(M,J  ) )
     $                   +RHOT(J-1) * ( U(M,J-1) + U(M,J-1) )  , 0.  )
  520 CONTINUE
      END IF
      I = M
      DO 555 J = 2 , N
      WA(I,J)    = WA(I,J)
     1 -DTSRW(J) * UB(J) * ( W1(I  ,J) - W1(I-1,J) )
     4 -DTSZW(J)*(
     5 (RHOW(J+1)*W(I,J+1)+RHOW(J  )*W(I  ,J  ))*(W(I,J+1)-W(I,J  ))
     6+(RHOW(J-1)*W(I,J-1)+RHOW(J  )*W(I  ,J  ))*(W(I,J  )-W(I,J-1)) )
     7  + DTS*RTBW(J)  * ( T(I,J) - TB(J)  + T(I,J-1)  - TB(J-1)  )
     8  + DTS*0.305*G* (QV(I,J) - QVB(J) + QV(I,J-1) - QVB(J-1) )
     9  - DTSG*(QL(I,J)+QL(I,J-1)+QI(I,J)+QI(I,J-1)+QR(I,J)+QR(I,J-1))
  555 CONTINUE
C
C       FORCING FOR  V, T, QV, QL  EQUATIONS
C
      DO 540 J = 1 , N
      DO 540 I = 1 , MM1
      VA(I,J)    = VA(I,J)
     1 -DTLR(I)* ( R(I+1) * U(I+1,J) * ( V(I+1,J  ) - V(I  ,J  ) )
     2            +R(I  ) * U(I  ,J) * ( V(I  ,J  ) - V(I-1,J  ) ) )
     3 -DTLZ(J)* ( RHOW(J+1)*W(I,J+1)* ( V(I  ,J+1) - V(I  ,J  ) )
     4            +RHOW(J  )*W(I,J  )* ( V(I  ,J  ) - V(I  ,J-1) ) )
     5 -DTL2*(F+ V(I,J)/RS(I))*(R(I+1)*U(I+1,J)+R(I)*U(I,J))/RS(I)
      TA(I,J)    = TA(I,J)
     1 -DTLR(I)* ( R(I+1) * U(I+1,J) * ( T(I+1,J  ) - T(I  ,J  ) )
     2            +R(I  ) * U(I  ,J) * ( T(I  ,J  ) - T(I-1,J  ) ) )
     3 -DTLZ(J)* ( RHOW(J+1)*W(I,J+1)* ( T(I  ,J+1) - T(I  ,J  ) )
     4            +RHOW(J  )*W(I,J  )* ( T(I  ,J  ) - T(I  ,J-1) ) )
     5 +QRLW(I,J)+QRSW(I,J)

C lagrangian change of theta
      lagt(I,J) = 
     1 (-DTLR(I)* ( R(I+1) * U(I+1,J) * ( T(I+1,J  ) - T(I  ,J  ) )
     2            +R(I  ) * U(I  ,J) * ( T(I  ,J  ) - T(I-1,J  ) ) )
     3 -DTLZ(J)* ( RHOW(J+1)*W(I,J+1)* ( T(I  ,J+1) - T(I  ,J  ) )
     4            +RHOW(J  )*W(I,J  )* ( T(I  ,J  ) - T(I  ,J-1) ) )
     5 +QRLW(I,J)+QRSW(I,J))/6



      QVA(I,J)    = QVA(I,J)
     1 -DTLR(I)* ( R(I+1) * U(I+1,J) * ( QV(I+1,J  ) - QV(I  ,J  ) )
     2            +R(I  ) * U(I  ,J) * ( QV(I  ,J  ) - QV(I-1,J  ) ) )
     3 -DTLZ(J)* ( RHOW(J+1)*W(I,J+1)* ( QV(I  ,J+1) - QV(I  ,J  ) )
     4            +RHOW(J  )*W(I,J  )* ( QV(I  ,J  ) - QV(I  ,J-1) ) )
      QLA(I,J)    = QLA(I,J)
     1 -DTLR(I)* ( R(I+1) * U(I+1,J) * ( QL(I+1,J  ) - QL(I  ,J  ) )
     2            +R(I  ) * U(I  ,J) * ( QL(I  ,J  ) - QL(I-1,J  ) ) )
     3 -DTLZ(J)* ( RHOW(J+1)*W(I,J+1)* ( QL(I  ,J+1) - QL(I  ,J  ) )
     4            +RHOW(J  )*W(I,J  )* ( QL(I  ,J  ) - QL(I  ,J-1) ) )
      QIA(I,J)    = QIA(I,J)
     1 -DTLR(I)* ( R(I+1) * U(I+1,J) * ( QI(I+1,J  ) - QI(I  ,J  ) )
     2            +R(I  ) * U(I  ,J) * ( QI(I  ,J  ) - QI(I-1,J  ) ) )
     3 -DTLZ(J)* ( RHOW(J+1)*W(I,J+1)* ( QI(I  ,J+1) - QI(I  ,J  ) )
     4            +RHOW(J  )*W(I,J  )* ( QI(I  ,J  ) - QI(I  ,J-1) ) )
      QRA(I,J)    = QRA(I,J)
     1 -DTLR(I)* ( R(I+1) * U(I+1,J) * ( QR(I+1,J  ) - QR(I  ,J  ) )
     2            +R(I  ) * U(I  ,J) * ( QR(I  ,J  ) - QR(I-1,J  ) ) )
     3 -DTLZ(J)* ( RHOW(J+1)*W(I,J+1)* ( QR(I  ,J+1) - QR(I  ,J  ) )
     4            +RHOW(J  )*W(I,J  )* ( QR(I  ,J  ) - QR(I  ,J-1) ) )
  540 CONTINUE
C
C       OUTER BOUNDARY
C
      IF ( OBRAD ) THEN
      DO 530 J = 1 , N
      UB(J) =  AMAX1( R(MP1) * U(MP1,J) + R(M) * U(M,J) , 0. )
  530 CONTINUE
      END IF
      I = M
      DO 545 J = 1 , N
      VA(M,J)    = VA(M,J)
     1 -DTLR(M) * UB(J) * ( V1(M  ,J  ) - V1(M-1,J  ) )
     3 -DTLZ(J)* ( RHOW(J+1)*W(M,J+1)* ( V(M  ,J+1) - V(M  ,J  ) )
     4            +RHOW(J  )*W(M,J  )* ( V(M  ,J  ) - V(M  ,J-1) ) )
     5 -DTL2*(F+ V(M,J)/RS(M))*(R(M+1)*U(M+1,J)+R(M)*U(M,J))/RS(M)
      TA(M,J)    = TA(M,J)
     1 -DTLR(M) * UB(J) * ( T1(M  ,J  ) - T1(M-1,J  ) )
     3 -DTLZ(J)* ( RHOW(J+1)*W(M,J+1)* ( T(M  ,J+1) - T(M  ,J  ) )
     4            +RHOW(J  )*W(M,J  )* ( T(M  ,J  ) - T(M  ,J-1) ) )
     5 +QRLW(M,J)+QRSW(M,J)
      QVA(M,J)    = QVA(M,J)
     1 -DTLR(M) * UB(J) * ( QV1(M  ,J  ) - QV1(M-1,J  ) )
     3 -DTLZ(J)* ( RHOW(J+1)*W(M,J+1)* ( QV(M  ,J+1) - QV(M  ,J  ) )
     4            +RHOW(J  )*W(M,J  )* ( QV(M  ,J  ) - QV(M  ,J-1) ) )
      QLA(M,J)    = QLA(M,J)
     1 -DTLR(M) * UB(J) * ( QL1(M  ,J  ) - QL1(M-1,J  ) )
     3 -DTLZ(J)* ( RHOW(J+1)*W(M,J+1)* ( QL(M  ,J+1) - QL(M  ,J  ) )
     4            +RHOW(J  )*W(M,J  )* ( QL(M  ,J  ) - QL(M  ,J-1) ) )
      QIA(M,J)    = QIA(M,J)
     1 -DTLR(M) * UB(J) * ( QI1(M  ,J  ) - QI1(M-1,J  ) )
     3 -DTLZ(J)* ( RHOW(J+1)*W(M,J+1)* ( QI(M  ,J+1) - QI(M  ,J  ) )
     4            +RHOW(J  )*W(M,J  )* ( QI(M  ,J  ) - QI(M  ,J-1) ) )
      QRA(M,J)    = QRA(M,J)
     1 -DTLR(M) * UB(J) * ( QR1(M  ,J  ) - QR1(M-1,J  ) )
     3 -DTLZ(J)* ( RHOW(J+1)*W(M,J+1)* ( QR(M  ,J+1) - QR(M  ,J  ) )
     4            +RHOW(J  )*W(M,J  )* ( QR(M  ,J  ) - QR(M  ,J-1) ) )
  545 CONTINUE
C
C        TIME SMOOTHER
C
      DO 160 J = 1 , N
      DO 160 I = 1 , M
      U(I+1,J)= U(I+1,J)+ EPS * ( U1(I+1,J)-2.*U(I+1,J) )
      V(I,J)  = V(I,J)  + EPS * ( V1(I,J)  -2.*V(I,J)   )
      W(I,J+1)= W(I,J+1)+ EPS * ( W1(I,J+1)-2.*W(I,J+1) )
      P(I,J)  = P(I,J)  + EPS * ( P1(I,J)  -2.*P(I,J)   )
      T(I,J)  = T(I,J)  + EPS * ( T1(I,J)  -2.*T(I,J)   )
      QV(I,J) = QV(I,J) + EPS * ( QV1(I,J) -2.*QV(I,J)  )
      QL(I,J) = QL(I,J) + EPS * ( QL1(I,J) -2.*QL(I,J)  )
      QI(I,J) = QI(I,J) + EPS * ( QI1(I,J) -2.*QI(I,J)  )
      QR(I,J) = QR(I,J) + EPS * ( QR1(I,J) -2.*QR(I,J)  )
  160 CONTINUE
C
C         SMALL TIME STEP
C
      DO 170 NSMALL = 1 , NS
      DO 171 J = 1 , N
      DO 171 I = 2 , M
      U1(I,J)=U1(I,J)-CPTDR(J)*(P1(I,J)-P1(I-1,J))+UA(I,J)
  171 CONTINUE
      DO 172 J = 2 , N
      DO 172 I = 1 , M
      WS(I,J)=W1(I,J)-.5*(1.-EP)*CPTDZ(J)*(P1(I,J)-P1(I,J-1))+WA(I,J)
  172 CONTINUE
      DO 173 J = 1 , N
      DO 173 I = 1 , M
      PS(I,J)=P1(I,J)-RC2(J) *
     1   ( R(I+1)*U1(I+1,J)  -R(I)*U1(I,J)  )/RS(I)
     2               -.5*(1.-EP)*ZC2(J) *
     3    ( RHOTVW(J+1)*W1(I,J+1)-RHOTVW(J)*W1(I,J))
     $        + DTHETAV(I,J) 
     $        + RC2(J)*DR/(DTL*TB(J)*(1.+.61*QVB(J)))*
     $        ((1.+0.61*QV1(I,J))*(QRLW(I,J)+SUBGT(I,J))
     $        + 0.61*T1(I,J)*SUBGQV(I,J))
  173 CONTINUE
      DO 175 I = 1 , M
      DO 177 J = 2 , N
      D(J) = (WS(I,J) - CPTDZ(J)*.5*(1.+EP)*(PS(I,J)-PS(I,J-1))
     $                + C(J) * D(J-1) ) * E(J) /A(J)
  177 CONTINUE
      DO 174 J = N , 1 , -1
      W1(I,J) = E(J) * W1(I,J+1) + D(J)
      P1(I,J) = PS(I,J) -.5*(1.+EP)*ZC2(J)*
     $    ( RHOTVW(J+1)*W1(I,J+1)-RHOTVW(J)*W1(I,J))
  174 CONTINUE
  175 CONTINUE
  170 CONTINUE
C
C        OUTER BOUNDARY
C
      DO 195 J = 1 , N
      U1(MP1,J) = U1(MP1,J) + UA(MP1,J)
  195 CONTINUE
C
C        ADVANCE V , T , QV , QL , QI , QR
C
      DO J=1,N
         DO I=1,M
            V1(I,J)  = V1(I,J)  + VA(I,J)
            T1(I,J)  = T1(I,J)  + TA(I,J)
            THETAV(I,J) = T1(I,J) * (1. + 0.61*QV1(I,J))
            QV1(I,J) = QV1(I,J) + QVA(I,J)
            QL1(I,J) = QL1(I,J) + QLA(I,J)
            QI1(I,J) = QI1(I,J) + QIA(I,J)
            QR1(I,J) = QR1(I,J) + QRA(I,J)
            QV1(I,J) = MAX(1.E-100, QV1(I,J))
            QL1(I,J) = MAX(1.E-100, QL1(I,J))
            QI1(I,J) = MAX(1.E-100, QI1(I,J))
            QR1(I,J) = MAX(1.E-100, QR1(I,J))
            IF (I .GE. M-NRSP/10) THEN
              U1(I,J) = 0.
              V1(I,J) = 0.
              W1(I,J) = 0.
            END IF
         END DO
      END DO


C store qv and ql before phase change
      qvold = QV1
      qlold = QL1


C
C        CONDENSATION / EVAPORATION
C              NB PRESS is pressure in mb!
C
      DO 490 J=1,NWET
      DO 490 I=1,M
      PNST= PN(J) + P1(I,J)
      PRESS(I,J)= 1000. * PNST ** (1./XKAPPA)
      TEMP= PNST * T1(I,J)
      ES  = 6.11 * EXP(A1 * ( TEMP  - 273. ) / ( TEMP  - 36. ) )
      QSS = .622 * ES /(PRESS(I,J)-ES)

      MICT(I,J)=0
      MICQV(I,J)=0


      IF( .NOT.(QV1(I,J) .LT. QSS .AND. QL1(I,J).LE. 1.E-8) ) THEN
         R1 = 1./(1. + XLDCP * 237. * A1 * QSS /( TEMP - 36. )**2 )
         QVD = R1 * ( QV1(I,J) - QSS )
C    If next line is included, there is NO EVAPORATION
C        IF(QVD.LT.0.) QVD=0.
         IF ( (QL1(I,J)   + QVD) .LT. 0. ) THEN
C
C    EVAPORATE
C
            T1 (I,J) = T1(I,J)  - XLDCP * QL1(I,J) /PNST
            MICT(I,J) =   - XLDCP * QL1(I,J) /PNST

            QV1(I,J) = QV1(I,J) + QL1(I,J)
            MICQV(I,J) =  QL1(I,J)

            QL1(I,J) = 0.
         ELSE
C
C    CONDENSE
C
            T1(I,J)   = T1(I,J)  + XLDCP * QVD / PNST
            MICT(I,J) = XLDCP * QVD / PNST

            QV1(I,J)  = QV1(I,J)         - QVD
            MICQV(I,J) = - QVD
            QL1(I,J)  = QL1(I,J)         + QVD
         END IF
         QV1(I,J) = MAX(1.E-100, QV1(I,J))
         QL1(I,J) = MAX(1.E-100, QL1(I,J))
      END IF
  490 CONTINUE

C
C        MICROPHYSICS
C
      tempold = T1
      qvold = QV1
      qlold = QL1
      qiold = QI1
      qrold = QR1

      IF (MOD(ITIME,IBUFF).EQ.0) THEN
         WRT = .TRUE.
      ELSE
         WRT = .FALSE.
      END IF

      CALL PRECIP(WRT,M,N,M,P1,QV1,QL1,QI1,QR1,T1,ZS,PRESS,PN,DTL,NWET)

C compare QV1 and QL1 with value before phase change calculation
      change = T1 - tempold
      qvchange = QV1 - qvold
      qlchange = QL1 - qlold
      qrchange = QR1 - qrold
      qichange = QI1 - qiold

      IF (MOD(ITIME,IBUFF).EQ.0) 
     &   WRITE(24) MICT,MICQV,change,qvchange,qlchange,qrchange,qichange
C compare new T with to find dtheta

      change = (T1-tempold)/6
      tempold = 0



C
C        TIME FLIP
C
      DO 190 J=1,N
      DO 190 I=1,M
      DTHETAV(I,J) = RC2(J)*DR/(DTL*TB(J)*(1.+.61*QVB(J)))*
     $      (T1(I,J)*(1. + 0.61*QV1(I,J)) - THETAV(I,J))
      UTEMP    = U(I+1,J) + EPS * U1(I+1,J)
      VTEMP    = V(I,J)   + EPS * V1(I,J)
      WTEMP    = W(I,J+1) + EPS * W1(I,J+1)
      PTEMP    = P(I,J)   + EPS * P1(I,J)
      TTEMP    = T(I,J)   + EPS * T1(I,J)
      QVTEMP   = QV(I,J)  + EPS * QV1(I,J)
      QLTEMP   = QL(I,J)  + EPS * QL1(I,J)
      QITEMP   = QI(I,J)  + EPS * QI1(I,J)
      QRTEMP   = QR(I,J)  + EPS * QR1(I,J)
      U(I+1,J) = U1(I+1,J)
      V(I,J)   = V1(I,J)
      W(I,J+1) = W1(I,J+1)
      P(I,J)   = P1(I,J)
      T(I,J)   = T1(I,J)
      QV(I,J)  = QV1(I,J)
      QL(I,J)  = QL1(I,J)
      QI(I,J)  = QI1(I,J)
      QR(I,J)  = QR1(I,J)
      U1(I+1,J)= UTEMP
      V1(I,J)  = VTEMP
      W1(I,J+1)= WTEMP
      T1(I,J)  = TTEMP
      P1(I,J)  = PTEMP
      QV1(I,J) = QVTEMP
      QL1(I,J) = QLTEMP
      QI1(I,J) = QITEMP
      QR1(I,J) = QRTEMP
  190 CONTINUE


C
C      COMPUTE WMAX , VMAX
C
      IF (MOD(ITIME,ITMAX).EQ.0) THEN
         IWMAX=1
         JWMAX=1
         WMAX=0.
         IVMAX=1
         JVMAX=1
         VMAX=0.
         DO 210 I = 1 , M
         DO 210 J = 2 , N
            IF(W(I,J).GT.WMAX) THEN
               WMAX = W(I,J)
               IWMAX = I
               JWMAX = J
            END IF
            IF(V(I,J).GT.VMAX) THEN
               VMAX = V(I,J)
               IVMAX = I
               JVMAX = J
            END IF
  210    CONTINUE
         TINHRS = TIME / 3600.
         IF(MOD(ITIME,ITMAX).EQ.0) WRITE(21,*) TINHRS,VMAX,WMAX,
     $   1000.*(PNS+(P1(1,1)-CC1*(QV1(1,1)-QVB(1)))/CC2)**(1./XKAPPA),
     $   R(IVMAX),Z(JVMAX)
      ENDIF
      IF (MOD(ITIME,IBUFF).EQ.0 .OR. MOD(ITIME,IBUFF2).EQ.0) THEN
         DO J = 1, N
            DO I = 1, M
               IF (ABS(U(I,J)).LT.1.E-35) U(I,J) = 0.
               IF (ABS(V(I,J)).LT.1.E-35) V(I,J) = 0.
               IF (ABS(W(I,J)).LT.1.E-35) W(I,J) = 0.
               IF (ABS(T(I,J)).LT.1.E-35) T(I,J) = 0.
               IF (ABS(QV(I,J)).LT.1.E-35) QV(I,J) = 0.
               IF (ABS(QL(I,J)).LT.1.E-35) QL(I,J) = 0.
               IF (ABS(QI(I,J)).LT.1.E-35) QI(I,J) = 0.
               IF (ABS(QR(I,J)).LT.1.E-35) QR(I,J) = 0.
               IF (ABS(P(I,J)).LT.1.E-35) P(I,J) = 0.
               IF (ABS(QRLW(I,J)).LT.1.E-35) QRLW(I,J) = 0.
               IF (ABS(QRSW(I,J)).LT.1.E-35) QRSW(I,J) = 0.
            END DO
         END DO
         DO J = 1, N
            IF (ABS(U(MP1,J)).LT.1.E-35) U(MP1,J) = 0.
         END DO
         DO I = 1, M
            IF (ABS(W(I,NP1)).LT.1.E-35) W(I,NP1) = 0.
         END DO
C        IF (MOD(ITIME,IBUFF).EQ.0) 
C    &   WRITE(11) U,V,W,T,QV,QL,QI,QR,P,QRLW,QRSW,
C    &   UAOUT,VAOUT,WAOUT,QVAOUT,QLAOUT,TAOUT,
C    &   TZOUT,QVZOUT,MICT,MICQV,change,lagt,qvchange,qlchange,U

C        IF (MOD(ITIME,IBUFF2).EQ.0) 
C    &   WRITE(12) U(1:120,1:24),V(1:120,1:24),W(1:120,1:24),
C    &   T(1:120,1:24),QV(1:120,1:24),QL(1:120,1:24),
C    &   QR(1:120,1:24),QI(1:120,1:24),
C    &   P(1:120,1:24),QRLW(1:120,1:24),U(1:120,1:24)
      ENDIF
      IF(ITIME.GT.ISTART) GO TO 500
         DTL = 2. * DT
         NS  = 2  * NS
         DTL2 = .5 * DTL
         DO 17 I = 1 , M
   17       DTLR(I)  = .5  * DTL * RDR /RS(I)
         DO 27 J = 1 , N
   27       DTLZ(J)  = .5  * DTL * RDZ / RHOT(J)
  500 CONTINUE
      STOP
      END
      SUBROUTINE DIFF
     $ (UA , WA, VA  , TA, QVA, QLA,
     $ U1 , V1  , W1, T1, QV1, QL1,
     $ XKM, P1  , R   , RS  , Z  , ZS,
     $ RTBW   , RQVBW , TAU , TB, QVB, PN, PD,
     $ TSURF  , QSURF , RDR, RDZ, RDR2, RDZ2 , DTS, DTL, DZ,
     $ XVL2, XHL2, CD, CE, CT, XLDCP, XKAPPA, A1, G
     $ ,CERS, CDRS, CTRS, VGUST, QIA, QRA, QI1, QR1,
     $ UAOUT,VAOUT,WAOUT,QVAOUT,QLAOUT,TAOUT,TZOUT,
     $ QVZOUT)

      IMPLICIT NONE
      INTEGER M,N,MP1,MM1,NP1,NM1
      PARAMETER(M=1620,N=44,MP1=M+1,MM1=M-1,NP1=N+1,NM1=N-1)
      INTEGER I,J
      REAL TT,AA
      REAL
     $ TRR(M,N  )  , TTT(M,N)  , TZZ(M,NP1),
     $ TRZ(MP1,NP1), TRT(MP1,N), TZT(M,NP1),
     $ TR(MP1, N)  , TZ(M,NP1) ,
     $ QVR(MP1, N) , QVZ(M,NP1), QLR(MP1,N), QLZ(M,NP1),
     $ XKMH(M,NP1) , TE(M,N)   , UDR(M,N)  , WDZ(M,N)  , VDR(MP1,N),
     $ UZWR(MP1,N) , VDZ(M,N)  , DEFH2(M,N), DEF2(M,N) , DTDZ(M,N),
     $ QIR(MP1,N), QIZ(M,NP1), QRR(MP1,N), QRZ(M,NP1)
      REAL
     $ UA(MP1,N) , WA(M,NP1), VA(M,N)  , TA(M,N), QVA(M,N), QLA(M,N),
     $ U1(MP1,N) , V1(M,N)  , W1(M,NP1), T1(M,N), QV1(M,N), QL1(M,N),
     $ XKM(M,NP1), P1(M,N)  , R(MP1)   , RS(M)  , Z(NP1)  , ZS(N)   ,
     $ RTBW(N)   , RQVBW(N) , TAU(M,N) , TB(N), QVB(N), PN(N), PD(N),
     $ TSURF(M)  , QSURF(M) , RDR, RDZ, RDR2, RDZ2, DTS, DTL, DZ,
     $ XVL2, XHL2, CD, CE, CT, XLDCP, XKAPPA, A1, G
     $ ,CERS(M), CDRS(M), CTRS(M), VGUST, QIA(M,N), QRA(M,N), QI1(M,N),
     $ QR1(M,N)

      REAL
     $ UAOUT(MP1,N),VAOUT(M,N),WAOUT(M,NP1),
     $ QVAOUT(M,N),QLAOUT(M,N),TAOUT(M,N),
     $ TZOUT(M,NP1),QVZOUT(M,NP1)

      DO 500 J = 1 , N
      DO 500 I = 1 , M
      TE(I,J)= T1(I,J) * (1.+ XLDCP * QV1(I,J) /
     $        ( T1(I,J) * ( PN(J) + P1(I,J) )  )  )
  500 CONTINUE
      DO 110 J = 1 , N
      DO 110 I = 1 , M
      UDR(I,J) = RDR * ( U1(I+1,J) - U1(I,J) )
      WDZ(I,J) = RDZ * ( W1(I,J+1) - W1(I,J) )
  110 CONTINUE
      DO 120 J = 1 , N
      DO 125 I = 2 , M
      VDR(I,J) = R(I) * RDR * ( V1(I,J)/RS(I) - V1(I-1,J)/RS(I-1) )
  125 CONTINUE
      VDR(1  ,J) = 0.
      VDR(MP1,J) = VDR(M,J)
  120 CONTINUE
      DO 130 J = 2 , N
      DO 135 I = 2 , M
      UZWR(I,J) = RDZ*(U1(I,J)-U1(I,J-1)) + RDR*(W1(I,J)-W1(I-1,J))
  135 CONTINUE
      UZWR(1  ,J) = 0.
      UZWR(MP1,J) = UZWR(M,J)
  130 CONTINUE
      DO 140 J = 2 , N
      DO 140 I = 1 , M
      VDZ(I,J) = RDZ * ( V1(I,J) - V1(I,J-1) )
  140 CONTINUE
      DO 150 J = 2 , N
      DO 150 I = 1 , M
      DEFH2(I,J) = UDR(I,J)*UDR(I,J) + UDR(I,J-1)*UDR(I,J-1)
     $ + ( .25/(RS(I)*RS(I)) )*(
     $  (U1(I+1,J  )+U1(I,J  )) *  (U1(I+1,J  )+U1(I,J  ))
     $ +(U1(I+1,J-1)+U1(I,J-1)) *  (U1(I+1,J-1)+U1(I,J-1)) )
     $ + .25*( VDR(I+1,J  )*VDR(I+1,J  ) + VDR(I,J  )*VDR(I,J  )
     $        +VDR(I+1,J-1)*VDR(I+1,J-1) + VDR(I,J-1)*VDR(I,J-1) )
  150 CONTINUE
      DO 160 J = 2 , N
      DO 160 I = 1 , M
      DEF2(I,J) = DEFH2(I,J) +
     $    WDZ(I,J)*WDZ(I,J) + WDZ(I,J-1)*WDZ(I,J-1)
     $ +.5*( UZWR(I+1,J)*UZWR(I+1,J) + UZWR(I,J)*UZWR(I,J) )
     $ + VDZ(I,J) * VDZ(I,J)
  160 CONTINUE
      DO 170 J = 2 , N
      DO 170 I = 1 , M
      DTDZ(I,J) = RDZ*(
     $ 2.*RTBW(J) * (T1(I,J) - T1(I,J-1)  )
     $+2.*RQVBW(J)* (QV1(I,J)- QV1(I,J-1) ) )
      IF( (QL1(I,J) * QL1(I,J-1)) .GE. 1.E-8 ) THEN
      TT = .5 * (  T1(I,J  ) * ( PN(J  )+P1(I,J  ) )
     $           + T1(I,J-1) * ( PN(J-1)+P1(I,J-1) )  )
      AA=(1. + 4355.4   *(QV1(I,J)+QV1(I,J-1)) / TT      )
     $  /(1. + 6742307.7*(QV1(I,J)+QV1(I,J-1))/(TT*TT)   )
     $  * (2. / (TB(J)+TB(J-1)) )
      DTDZ(I,J) = RDZ* G * ( AA * ( TE(I,J) - TE(I,J-1) )
     $ - ( QL1(I,J) + QV1(I,J) - QL1(I,J-1) - QV1(I,J-1) ) )
      END IF
  170 CONTINUE
      DO 180 J = 2 , N
      DO 180 I = 1 , M
      XKM(I,J) = 0.
      IF(DEF2(I,J).GT.DTDZ(I,J))
     $  XKM(I,J)= XVL2 * SQRT( DEF2(I,J) - DTDZ(I,J))
      XKMH(I,J) = XHL2 * SQRT( DEFH2(I,J) )
      IF(XKMH(I,J).LT.XKM(I,J))   XKMH(I,J) = XKM(I,J)
      IF(XKM(I,J).GE. .4 * DZ *DZ/DTL) XKM(I,J) = .4 *DZ*DZ/DTL
  180 CONTINUE
      DO 190 I = 1 , M
      XKM(I,  1) = XKM(I,2)
      XKM(I,NP1) = XKM(I,N)
      XKMH(I,  1) = XKMH(I,2)
      XKMH(I,NP1) = XKMH(I,N)
  190 CONTINUE
C
C        CALCULATE STRESS
C
C        TRR(M,N)
C
      DO 200 J = 1 , N
      DO 200 I = 1 , M
      TRR(I,J) =  ( XKMH(I,J+1) + XKMH(I,J) ) * UDR(I,J)
  200 CONTINUE
C
C        TTT(M,N)
C
      DO 210 J = 1 , N
      TTT(1,J) = 0.
      DO 210 I = 2 , M
      TTT(I,J) =
     $.5*(XKMH(I,J+1)+XKMH(I,J)+XKMH(I-1,J+1)+XKMH(I-1,J))*U1(I,J)/R(I)
  210 CONTINUE
C
C        TRZ(M,NP1)
C
      DO 220 J=2,N
      DO 220 I=2,M
      TRZ(I,J)=.5* ( XKM(I-1,J) + XKM(I,J) ) *  UZWR(I,J)
  220 CONTINUE
      DO 230 I =2,M
      TRZ(I,1  )=.5*(CDRS(I)+CDRS(I-1))*U1(I,1)*SQRT(
     $ MAX( U1(I,1)**2 + .25*(V1(I,1)+V1(I-1,1))**2 , VGUST ) )
      TRZ(I,NP1) = 0.
  230 CONTINUE
      DO 235 J = 1 , NP1
      TRZ(1  ,J) = 0.
      TRZ(MP1,J) = TRZ(M,J) * R(M) / R(MP1)
  235 CONTINUE
C
C        TRT(M,N)
C
      DO 245 J=1,N
      DO 240 I=2,M
      TRT(I,J) =
     $.25*( XKMH(I,J+1)+XKMH(I,J)+XKMH(I-1,J+1)+XKMH(I-1,J))* VDR(I,J)
  240 CONTINUE
      TRT(1  ,J) = 0.
      TRT(MP1,J) = TRT(M,J) * R(M)**2 / R(MP1)**2
  245 CONTINUE
C
C        TZT(M,NP1)
C
      DO 250 I = 1 , M
      DO 255 J = 2 , N
      TZT(I,J) = XKM(I,J) * VDZ(I,J)
  255 CONTINUE
      TZT(I,1  )=CDRS(I)*V1(I,1)*SQRT(
     $ MAX( .25*(U1(I,1)+U1(I+1,1))**2 + V1(I,1)**2 , VGUST ) )
      TZT(I,NP1)= 0.
  250 CONTINUE
C
C          TZZ(M,N)
C
      DO 260 J = 1 , N
      DO 260 I = 1 , M
      TZZ(I,J) = ( XKM(I,J+1) + XKM(I,J) )*WDZ(I,J)
  260 CONTINUE
C
C       TEMPERATURE FLUX
C
C          TR(MP1,N)
C
      DO 270 J=1,N
      DO 275 I=2,M
      TR(I,J) = .25*( XKMH(I,J+1)+XKMH(I,J)+XKMH(I-1,J+1)+XKMH(I-1,J))
     $                          *RDR*(T1(I,J)-T1(I-1,J))
  275 CONTINUE
      TR(1  ,J)=0.0
      TR(MP1,J)= TR(M,J) * R(M) / R(MP1)
  270 CONTINUE
C
C         TZ(M,NP1)
C
      DO 280 I=1,M
      DO 285 J=2,N
      TZ(I,J) = XKM(I,J) * RDZ * (T1(I,J) - T1(I,J-1))
  285 CONTINUE
      TZ(I,1  ) =( T1(I,1)-TSURF(I))*CTRS(I)*SQRT(
     $  MAX( .25*( U1(I+1,1) + U1(I,1) )**2 + V1(I,1)**2 , VGUST ) )
      TZ(I,NP1) = 0.
  280 CONTINUE
C
C       QV FLUX
C
C         QVR(MP1,N)
C
      DO 370 J=1,N
      DO 375 I=2,M
      QVR(I,J)= .25*(XKMH(I,J+1)+XKMH(I,J)+XKMH(I-1,J+1)+XKMH(I-1,J))
     $                            *RDR*(QV1(I,J)-QV1(I-1,J))
  375 CONTINUE
      QVR(1  ,J)=0.0
      QVR(MP1,J)= QVR(M,J) * R(M) / R(MP1)
  370 CONTINUE
C
C         QVZ(M,NP1)
C
      DO 380 I=1,M
      DO 385 J=2,N
      QVZ(I,J) = XKM(I,J) * RDZ * (QV1(I,J) - QV1(I,J-1))
  385 CONTINUE
      QVZ(I,1  ) =( QV1(I,1)-QSURF(I))*CERS(I)*SQRT(
     $ MAX( .25*( U1(I+1,1) + U1(I,1) )**2 + V1(I,1)**2 , VGUST ) )
      QVZ(I,NP1) = 0.
  380 CONTINUE
C
C       QL FLUX
C
C        QLR(MP1,N)
C
      DO 570 J=1,N
      DO 575 I=2,M
      QLR(I,J)= .25*(XKMH(I,J+1)+XKMH(I,J)+XKMH(I-1,J+1)+XKMH(I-1,J))
     $                          *RDR*(QL1(I,J)-QL1(I-1,J))
  575 CONTINUE
      QLR(1  ,J)=0.0
      QLR(MP1,J)= QLR(M,J) * R(M) / R(MP1)
  570 CONTINUE
C
C         QLZ(M,NP1)
C
      DO 580 I=1,M
      DO 585 J=2,N
      QLZ(I,J) = XKM(I,J) * RDZ * (QL1(I,J) - QL1(I,J-1))
  585 CONTINUE
      QLZ(I,1  ) = 0.
      QLZ(I,NP1) = 0.
  580 CONTINUE
C
C       QI FLUX
C
C        QIR(MP1,N)
C
      DO 670 J=1,N
      DO 675 I=2,M
      QIR(I,J)= .25*(XKMH(I,J+1)+XKMH(I,J)+XKMH(I-1,J+1)+XKMH(I-1,J))
     $                          *RDR*(QI1(I,J)-QI1(I-1,J))
  675 CONTINUE
      QIR(1  ,J)=0.0
      QIR(MP1,J)= QIR(M,J) * R(M) / R(MP1)
  670 CONTINUE
C
C         QIZ(M,NP1)
C
      DO 680 I=1,M
      DO 685 J=2,N
      QIZ(I,J) = XKM(I,J) * RDZ * (QI1(I,J) - QI1(I,J-1))
  685 CONTINUE
      QIZ(I,1  ) = 0.
      QIZ(I,NP1) = 0.
  680 CONTINUE
C
C       QR FLUX
C
C        QRR(MP1,N)
C
      DO 770 J=1,N
      DO 775 I=2,M
      QRR(I,J)= .25*(XKMH(I,J+1)+XKMH(I,J)+XKMH(I-1,J+1)+XKMH(I-1,J))
     $                          *RDR*(QR1(I,J)-QR1(I-1,J))
  775 CONTINUE
      QRR(1  ,J)=0.0
      QRR(MP1,J)= QRR(M,J) * R(M) / R(MP1)
  770 CONTINUE
C
C         QRZ(M,NP1)
C
      DO 780 I=1,M
      DO 785 J=2,N
      QRZ(I,J) = XKM(I,J) * RDZ * (QR1(I,J) - QR1(I,J-1))
  785 CONTINUE
      QRZ(I,1  ) = 0.
      QRZ(I,NP1) = 0.
  780 CONTINUE
      DO 400 J = 1 , N
      DO 400 I = 2 , M
      UA(I,J)= -DTS*(
     1 - RDR * ( RS(I) * TRR(I,J) - RS(I-1) * TRR(I-1,J) ) / R(I)
     2 + TTT(I,J) / R(I)
     3 - RDZ * ( TRZ(I,J+1) - TRZ(I,J) )   )
     4 + DTS * .5 * ( TAU(I,J) + TAU(I-1,J) ) * U1(I,J)
  400 CONTINUE
      DO 410  J = 2 , N
      DO 410  I = 1 , M
      WA(I,J) = -DTS*(
     1 - RDR * ( R(I+1)*TRZ(I+1,J) - R(I)*TRZ(I,J) ) / RS(I)
     2 - RDZ * ( TZZ(I,J) - TZZ(I,J-1) )   )
     3 + DTS * .5 * ( TAU(I,J) + TAU(I,J-1) ) * W1(I,J)
  410 CONTINUE
      DO 420  J = 1 , N
      DO 420  I = 1 , M
      VA(I,J) = -DTL * (
     1 - RDR * (R(I+1)*R(I+1)*TRT(I+1,J)-R(I)*R(I)*TRT(I,J))
     2                                           /(RS(I)*RS(I))
     3 - RDZ * ( TZT(I,J+1) - TZT(I,J) )   )
     4 +DTL * TAU(I,J) * V1(I,J)
      TA(I,J) = -DTL * (
     1 - RDR * ( R(I+1) * TR(I+1,J) - R(I) * TR(I,J) ) / RS(I)
     2 - RDZ * (  TZ(I,J+1) - TZ(I,J) )    )
C   3 + DTL * TAU(I,J) * ( T1(I,J) - TB(J) )
      QVA(I,J) = -DTL * (
     1 - RDR * ( R(I+1) * QVR(I+1,J) - R(I) * QVR(I,J) ) / RS(I)
     2 - RDZ * (  QVZ(I,J+1) - QVZ(I,J) )    )
C    3 + DTL * TAU(I,J) *  QV1(I,J)
      QLA(I,J) = -DTL * (
     1 - RDR * ( R(I+1) * QLR(I+1,J) - R(I) * QLR(I,J) ) / RS(I)
     2 - RDZ * (  QLZ(I,J+1) - QLZ(I,J) )    )
C    3 + DTL * TAU(I,J) *  QL1(I,J)
      QIA(I,J) = -DTL * (
     1 - RDR * ( R(I+1) * QIR(I+1,J) - R(I) * QIR(I,J) ) / RS(I)
     2 - RDZ * (  QIZ(I,J+1) - QIZ(I,J) )    )
C    3 + DTL * TAU(I,J) *  QI1(I,J)
      QRA(I,J) = -DTL * (
     1 - RDR * ( R(I+1) * QRR(I+1,J) - R(I) * QRR(I,J) ) / RS(I)
     2 - RDZ * (  QRZ(I,J+1) - QRZ(I,J) )    )
C    3 + DTL * TAU(I,J) *  QR1(I,J)
  420 CONTINUE

      UAOUT = UA
      VAOUT = VA
      WAOUT = WA

      QVAOUT = QVA
      QLAOUT = QLA
      TAOUT = TA

      TZOUT = TZ
      QVZOUT = QVZ

      RETURN

      



      END
CLL   SUBROUTINE PRECIP-------------------------------------------------
CLL   PURPOSE: CALCULATE THE PRECIPITATION TERM FOR THE CLOUD WATER
CLL            EQUATION.  CALCULATE THE EVAPORATION TERM FOR THE WATER
CLL            VAPOUR EQUATION.  CALCULATE THE MELTING OF PRECIPITATION
CLL            LATENT HEAT TERM.  CALCULATE THE EVAPORATION OF RAIN AND
CLL            SNOW LATENT HEAT TERM.
CLL   MODIFICATIONS:  Diagnostic statements removed.
CLL                   Workspace variables dimensioned by parameters M,N.
CLL                   Orography and cloud fraction removed.  23/10/92
CLL                   Variables eliminated:  IDL,OROG,C,CF,HTDROP,LDIAG,
CLL                                          HI,DH,DHR,VPR,VPI
CLL                   Surface precipitation rates PPR, PPS and THREF
CLL                   commented out. 24/10/92
C*L   ARGUMENTS:--------------------------------------------------------
      SUBROUTINE PRECIP(WRT,NX,NZ,NXY,PEX,Q,QL,QI,QR,TH,HT,P,P0,DT,NWET)
C    $   ,PPR,PPS)
      IMPLICIT NONE
C  INPUT VARIABLES:
      LOGICAL WRT
      INTEGER NX,NZ,               ! X,Y,Z DIMENSIONS
     $  NXY                        ! NO. OF POINTS IN FIELD
      INTEGER NWET                 ! NO. LEVELS WITH MOISTURE
      REAL PEX(NXY,NZ),            ! EXNER PRES. DEVIATION
     $  Q(NXY,NZ),                 ! HUMIDTY MIXING RATIO
     $  QL(NXY,NZ),                ! CLOUD WATER M.R.
     $  QI(NXY,NZ),                ! ICE M.R.
     $  QR(NXY,NZ),                ! RAIN WATER M.R.
     $  TH(NXY,NZ),                ! POTL. TEMP. DEVIATION
     $  HT(NZ),                    ! LEVEL HEIGHT
     $  P(NXY,NZ),                 ! PRESSURE (MB)
     $  P0(NZ),                    ! BASIC STATE EXNER PRESSURE
     $  DT                         ! TIMESTEP
C  OUTPUT VARIABLES:
C     REAL PPR(NXY),PPS(NXY)       ! GRID-SCALE PRECIPITATION RATE
C  WORKSPACE:
      INTEGER MMM,NNN
      PARAMETER(MMM=1620,NNN=44)
      REAL PEXTOT(MMM,NNN),TK(MMM,NNN),RHO(MMM,NNN),DTR(MMM),DTI(MMM),
     $  QS(MMM),ESI(MMM),ESW(MMM),FQI(MMM,NNN),FQR(MMM,NNN),
     $  THSTORE(MMM,NNN),QSTORE(MMM,NNN),QLSTORE(MMM,NNN),
     $  QRSTORE(MMM,NNN),QISTORE(MMM,NNN),QRFALL(MMM,NNN),
     $  QIFALL(MMM,NNN),THNUCL(MMM,NNN),QNUCL(MMM,NNN),QLNUCL(MMM,NNN),
     $  QINUCL(MMM,NNN),THRIM(MMM,NNN),QLRIM(MMM,NNN),QIRIM(MMM,NNN),
     $  THDEPSUB(MMM,NNN),QDEPSUB(MMM,NNN),QLDEPSUB(MMM,NNN),
     $  QIDEPSUB(MMM,NNN),THCAPT(MMM,NNN),QRCAPT(MMM,NNN),
     $  QICAPT(MMM,NNN),THSNEVAP(MMM,NNN),QSNEVAP(MMM,NNN),
     $  QISNEVAP(MMM,NNN),THMELT(MMM,NNN),QRMELT(MMM,NNN),
     $  QIMELT(MMM,NNN),THEVAP(MMM,NNN),QEVAP(MMM,NNN),QREVAP(MMM,NNN),
     $  QLACCR(MMM,NNN),QRACCR(MMM,NNN),QLAUTO(MMM,NNN),QRAUTO(MMM,NNN)
C  LOCAL VARIABLES:
      INTEGER K,L,L1,L2
      REAL QC,TEMP7,TEMPI,TEMPC,QSI,DQI,DQIL,DPR,PR02,PR04,
     $  DHI,RHOD,TEM1,TEM2,TEM3
C  LOCAL CONSTANTS:
      REAL EPSLON,CP,RGAS,WATLHE,WATLHF,CPILH,CPILF,CZERO
      PARAMETER (EPSLON=0.622,CP=1004.5,RGAS=287.,
     $  WATLHE=2.5E6,WATLHF=3.34E5,
     $  CPILH=WATLHE/CP,CPILF=(WATLHE+WATLHF)/CP,CZERO=273.16)
      REAL CONST,CONW,A1
C*----------------------------------------------------------------------
      L1 = 1
      L2 = NXY
      CONST = 100.0/RGAS
      CONW = RGAS/(EPSLON*WATLHE)
      A1 = 7.5*ALOG(10.)
CL----------------------------------------------------------------------
CL       Compute Exner pressure, temperature,density, fallout of water
CL       and ice, saturation vapour pressures over water and ice
CL----------------------------------------------------------------------
      DO K=1,NWET
       DO L=L1,L2
        PEXTOT(L,K) = P0(K)+PEX(L,K)
        TK(L,K) = TH(L,K)*PEXTOT(L,K)
        RHO(L,K) = CONST*P(L,K)/TK(L,K)
        THSTORE(L,K) = 0.0
        QSTORE(L,K) = 0.0
        QLSTORE(L,K) = 0.0
        QRSTORE(L,K) = 0.0
        QISTORE(L,K) = 0.0
        QRFALL(L,K) = 0.0
        QIFALL(L,K) = 0.0
        THNUCL(L,K) = 0.0
        QNUCL(L,K) = 0.0
        QLNUCL(L,K) = 0.0
        QINUCL(L,K) = 0.0
        THRIM(L,K) = 0.0
        QLRIM(L,K) = 0.0
        QIRIM(L,K) = 0.0
        THDEPSUB(L,K) = 0.0
        QDEPSUB(L,K) = 0.0
        QLDEPSUB(L,K) = 0.0
        QIDEPSUB(L,K) = 0.0
        THCAPT(L,K) = 0.0
        QRCAPT(L,K) = 0.0
        QICAPT(L,K) = 0.0
        THSNEVAP(L,K) = 0.0
        QSNEVAP(L,K) = 0.0
        QISNEVAP(L,K) = 0.0
        THMELT(L,K) = 0.0
        QRMELT(L,K) = 0.0
        QIMELT(L,K) = 0.0
        THEVAP(L,K) = 0.0
        QEVAP(L,K) = 0.0
        QREVAP(L,K) = 0.0
        QLACCR(L,K) = 0.0
        QRACCR(L,K) = 0.0
        QLAUTO(L,K) = 0.0
        QRAUTO(L,K) = 0.0
       END DO
      END DO
      DO K=1,NWET-1
       DO L=L1,L2
        FQI(L,K) = 3.23*(RHO(L,K)*QI(L,K))**0.17
        FQR(L,K) = 33.2*RHO(L,K)**(-0.15)*QR(L,K)**0.25
       END DO
      END DO
      DO 100 K=1,NWET-1
       DHI = DT/(HT(K+1)-HT(K))
       DO L=L1,L2
        DTR(L) = MAX(MAX(FQR(L,K+1),FQR(L,K))*DHI,1.0)
        DTI(L) = MAX(MAX(FQI(L,K+1),FQI(L,K))*DHI,1.0)
       END DO
       DO L=L1,L2
        QRSTORE(L,K) = QR(L,K)
        QISTORE(L,K) = QI(L,K)
        QR(L,K) = MAX(QR(L,K)+(RHO(L,K+1)*QR(L,K+1)*FQR(L,K+1)
     $    -RHO(L,K)*QR(L,K)*FQR(L,K))*DHI/(DTR(L)*RHO(L,K)),0.0)
        QI(L,K) = MAX(QI(L,K)+(RHO(L,K+1)*QI(L,K+1)*FQI(L,K+1)
     $    -RHO(L,K)*QI(L,K)*FQI(L,K))*DHI/(DTI(L)*RHO(L,K)),0.0)
        QRFALL(L,K) = QR(L,K) - QRSTORE(L,K)
        QIFALL(L,K) = QI(L,K) - QISTORE(L,K)
       END DO
       DO L=L1,L2
        ESW(L) = 6.11*EXP(A1*(TK(L,K)-273.)/(TK(L,K)-36.))
        ESI(L) = ESW(L)*(MIN(TK(L,K),CZERO)/CZERO)**2.66
        QS(L) = EPSLON*ESW(L)/(P(L,K)-(1.0-EPSLON)*ESW(L))
       END DO
CL----------------------------------------------------------------------
CL       Ice and Snow Formation, Evaporation and Melting
CL----------------------------------------------------------------------
       DO 141 L=L1,L2
        QSI = EPSLON*ESI(L)/(P(L,K)-(1.0-EPSLON)*ESI(L))
        TEMPC = TK(L,K)-CZERO
        ESI(L) = ESI(L)*100.0
        IF (TK(L,K).LE.CZERO) THEN
CL----------------------------------------------------------------------
C            Nucleation of snow
CL----------------------------------------------------------------------
         QISTORE(L,K) = QI(L,K)
         QLSTORE(L,K) = QL(L,K)
         QSTORE(L,K) = Q(L,K)
         THSTORE(L,K) = TH(L,K)
         DQI = 1.E-12*0.01*EXP(-0.6*TEMPC)/RHO(L,K)
         DQI = MAX(MIN(DQI,Q(L,K)+QL(L,K)-QSI),0.0)
         QI(L,K) = max(0.,QI(L,K)+DQI)
         DQIL = MIN(DQI,QL(L,K))
         QL(L,K) = QL(L,K)-DQIL
         TH(L,K) = TH(L,K)+WATLHF*DQIL/(CP*PEXTOT(L,K))
         DQI = DQI-DQIL
         TH(L,K) = TH(L,K)+CPILF*DQI/PEXTOT(L,K)
         Q(L,K) = Q(L,K)-DQI
         QINUCL(L,K) = QI(L,K) - QISTORE(L,K)
         QLNUCL(L,K) = QL(L,K) - QLSTORE(L,K)
         QNUCL(L,K) = Q(L,K) - QSTORE(L,K)
         THNUCL(L,K) = TH(L,K) - THSTORE(L,K)
CL----------------------------------------------------------------------
C            Deposition/Sublimation of snow
CL----------------------------------------------------------------------
         QISTORE(L,K) = QI(L,K)
         QLSTORE(L,K) = QL(L,K)
         QSTORE(L,K) = Q(L,K)
         THSTORE(L,K) = TH(L,K)
         DQI = 4.0*TK(L,K)**2*ESI(L)*(Q(L,K)+QL(L,K)-QSI)/
     $   (QSI*((4.2E10-1.24E8*TK(L,K))*ESI(L)+2.23E7*TK(L,K)**3))*
     $   (306.7*(RHO(L,K)*QI(L,K))**0.6667*EXP(-TEMPC/24.54)+
     $   1.24E4*RHO(L,K)**1.22*(1.0-TEMPC/102.0)*QI(L,K)**0.92)/RHO(L,K)
         DQI = MAX(MIN(DQI*DT,Q(L,K)+QL(L,K)-QSI),-QI(L,K))
         QI(L,K) = max(0.,QI(L,K)+DQI/DTI(L))
         DQIL = MAX(MIN(DQI,QL(L,K)),0.0)
         QL(L,K) = QL(L,K)-DQIL
         TH(L,K) = TH(L,K)+WATLHF*DQIL/(CP*PEXTOT(L,K))
         DQI = DQI-DQIL
         Q(L,K) = Q(L,K)-DQI
         TH(L,K) = TH(L,K)+CPILF*DQI/PEXTOT(L,K)
         QIDEPSUB(L,K) = QI(L,K) - QISTORE(L,K)
         QLDEPSUB(L,K) = QL(L,K) - QLSTORE(L,K)
         QDEPSUB(L,K) = Q(L,K) - QSTORE(L,K)
         THDEPSUB(L,K) = TH(L,K) - THSTORE(L,K)
CL----------------------------------------------------------------------
C            Riming of snow by cloud water
CL----------------------------------------------------------------------
         QISTORE(L,K) = QI(L,K)
         QLSTORE(L,K) = QL(L,K)
         THSTORE(L,K) = TH(L,K)
         DQI = QL(L,K)/(1.0+39.7*RHO(L,K)**0.776*QI(L,K)**1.176*
     $     EXP(TEMPC/45.56)*DT)
         QI(L,K) = max(0.,QI(L,K)+(QL(L,K)-DQI)/DTI(L))
         TH(L,K) = TH(L,K)+WATLHF*(QL(L,K)-DQI)/(CP*PEXTOT(L,K))
         QL(L,K) = DQI
         QIRIM(L,K) = QI(L,K) - QISTORE(L,K)
         QLRIM(L,K) = QL(L,K) - QLSTORE(L,K)
         THRIM(L,K) = TH(L,K) - THSTORE(L,K)
CL----------------------------------------------------------------------
C            Capture of rain by snow
CL----------------------------------------------------------------------
         QISTORE(L,K) = QI(L,K)
         QRSTORE(L,K) = QR(L,K)
         THSTORE(L,K) = TH(L,K)
         TEM1 = QR(L,K)**0.25
         TEM2 = QI(L,K)**0.3333
         TEM3 = EXP(-0.04*TEMPC)
         RHOD = RHO(L,K)**(-0.08)
         DQI = QR(L,K)/(1.0+22.76*ABS(33.2*TEM1*RHOD*RHOD-
     $     3.23*(RHO(L,K)*QI(L,K))**0.17)*(0.1336*RHOD*RHOD*TEM1*
     $     TEM1*TEM2*TEM3*TEM3+0.3269*RHOD*TEM1*TEM2*TEM2*TEM3+
     $     0.5*RHO(L,K)*QI(L,K))*RHO(L,K)*DT/DTR(L))
         QI(L,K) = max(0.,QI(L,K)+(QR(L,K)-DQI)*DTR(L)/DTI(L))
         TH(L,K) = TH(L,K)+DTR(L)*WATLHF*(QR(L,K)-DQI)/(CP*PEXTOT(L,K))
         QR(L,K) = DQI
         QICAPT(L,K) = QI(L,K) - QISTORE(L,K)
         QRCAPT(L,K) = QR(L,K) - QRSTORE(L,K)
         THCAPT(L,K) = TH(L,K) - THSTORE(L,K)
        END IF
  141  CONTINUE
       DO 150 L=L1,L2
        IF (TK(L,K).GT.CZERO) THEN
         TEMPC = TK(L,K)-CZERO
         TEMPI = (RHO(L,K)*QI(L,K))**0.8775*DT/RHO(L,K)
CL----------------------------------------------------------------------
C            Evaporate melting snow
CL----------------------------------------------------------------------
         DPR = 5.0658E-3*(0.493+0.0213*TEMPC)*TEMPI/QS(L)
         DPR = MIN(DPR*MAX(QS(L)-Q(L,K)-QL(L,K),0.0)/(1.0+DPR),QI(L,K))
         QISTORE(L,K) = QI(L,K)
         QSTORE(L,K) = Q(L,K)
         THSTORE(L,K) = TH(L,K)
         QI(L,K) = QI(L,K)-DPR/DTI(L)
         Q(L,K) = Q(L,K)+DPR
         TH(L,K) = TH(L,K)-DPR*CPILF/PEXTOT(L,K)
         QISNEVAP(L,K) = QI(L,K) - QISTORE(L,K)
         QSNEVAP(L,K) = Q(L,K) - QSTORE(L,K)
         THSNEVAP(L,K) = TH(L,K) - THSTORE(L,K)
CL----------------------------------------------------------------------
C            Melting of snow
C           Use wet bulb temp. (deg. C) in snow melt calc.
C           TW = T - (1-RH)*R*T**2/(EPSLON*WATLHE)
CL----------------------------------------------------------------------
         TEMP7 = MAX(TEMPC-CONW*TK(L,K)*TK(L,K)*
     $     MAX(QS(L)-Q(L,K)-QL(L,K),0.0)/QS(L),0.0)
         DPR = MIN(4.654E-3*TEMP7*TEMPI,QI(L,K))
         QISTORE(L,K) = QI(L,K)
         QRSTORE(L,K) = QR(L,K)
         THSTORE(L,K) = TH(L,K)
         QI(L,K) = max(0.,QI(L,K)-DPR/DTI(L))
         QR(L,K) = max(0.,QR(L,K)+DPR/DTR(L))
         TH(L,K) = TH(L,K)-WATLHF*DPR/(CP*PEXTOT(L,K))
         QIMELT(L,K) = QI(L,K) - QISTORE(L,K)
         QRMELT(L,K) = QR(L,K) - QRSTORE(L,K)
         THMELT(L,K) = TH(L,K) - THSTORE(L,K)
        END IF
  150  CONTINUE
       DO 181 L=L1,L2
CL----------------------------------------------------------------------
C            Evaporation of rain
CL----------------------------------------------------------------------
        QRSTORE(L,K) = QR(L,K)
        QSTORE(L,K) = Q(L,K)
        THSTORE(L,K) = TH(L,K)
        PR02 = 2.01477993*QR(L,K)**0.25*RHO(L,K)**0.17
        PR04 = PR02*PR02
        ESW(L) = ESW(L)*100.0
        DPR = 40.216*TK(L,K)**2*ESW(L)*(1.215+13.434*PR02)*PR04*DT/
     $    (QS(L)*RHO(L,K)*(RHO(L,K)*ESW(L)*(5.6E11-1.03E8*TK(L,K))+
     $    2.0E7*TK(L,K)**3))
        DPR = MIN(DPR*MAX(QS(L)-Q(L,K)-QL(L,K),0.0)/(1.0+DPR),QR(L,K))
        QR(L,K) = QR(L,K)-DPR/DTR(L)
        Q(L,K) = Q(L,K)+DPR
        TH(L,K) = TH(L,K)-DPR*CPILH/PEXTOT(L,K)
        QREVAP(L,K) = QR(L,K) - QRSTORE(L,K)
        QEVAP(L,K) = Q(L,K) - QSTORE(L,K)
        THEVAP(L,K) = TH(L,K) - THSTORE(L,K)
CL----------------------------------------------------------------------
C            Accretion of cloud on rain
CL----------------------------------------------------------------------
        QRSTORE(L,K) = QR(L,K)
        QLSTORE(L,K) = QL(L,K)
        DPR = QL(L,K)/(1.0+3.98*RHO(L,K)**(-0.32)*QR(L,K)*DT)
CKesslerDPR = MIN(2.2*QL(L,K)*QR(L,K)**0.875*DT,QL(L,K))
        QR(L,K) = QR(L,K)+(QL(L,K)-DPR)/DTR(L)
        QL(L,K) = max(0.,DPR)
        QRACCR(L,K) = QR(L,K) - QRSTORE(L,K)
        QLACCR(L,K) = QL(L,K) - QLSTORE(L,K)
CL----------------------------------------------------------------------
C            Autoconversion of cloud to rain
CL----------------------------------------------------------------------
        QRSTORE(L,K) = QR(L,K)
        QLSTORE(L,K) = QL(L,K)
        QC = MIN(3.34E-4,QL(L,K))
        DPR = MIN(4.96*(RHO(L,K)*QL(L,K))**0.333*DT*
     $    QL(L,K)*QL(L,K),QL(L,K)-QC)
CKesslerDPR = MIN(0.001*(QL(L,K)-QC)*DT,QL(L,K))
        QL(L,K) = QL(L,K)-DPR
        QR(L,K) = max(0.,QR(L,K)+DPR/DTR(L))
        QRAUTO(L,K) = QR(L,K) - QRSTORE(L,K)
        QLAUTO(L,K) = QL(L,K) - QLSTORE(L,K)
  181  CONTINUE
  100 CONTINUE
C     DO L=L1,L2
C      PPS(L) = 3.23*(RHO(L,1)*QI(L,1))**1.17
C      PPR(L) = 33.2*RHO(L,1)**0.85*QR(L,1)**1.25
C     END DO
      IF ( WRT ) THEN
      WRITE(19) QRFALL,QIFALL,THNUCL,QNUCL,QLNUCL,QINUCL,THRIM,QLRIM,
     &QIRIM,THDEPSUB,QDEPSUB,QLDEPSUB,QIDEPSUB,THCAPT,QRCAPT,QICAPT,
     &THSNEVAP,QSNEVAP,QISNEVAP,THMELT,QRMELT,QIMELT,THEVAP,QEVAP,
     &QREVAP,QLACCR,QRACCR,QLAUTO,QRAUTO
      END IF

      RETURN
      END
