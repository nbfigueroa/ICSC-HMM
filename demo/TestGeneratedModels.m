%Test generated models for Test8-workskinda
seq1 = RobotdataAR.seq(1);
seq2 = RobotdataAR.seq(2);
seq3 = RobotdataAR.seq(3);
seq4 = RobotdataAR.seq(4);
seq5 = RobotdataAR.seq(5);

C1 = bestARCH.Psi(1,end).theta(1,5);
dataIDs = bestARCH.Psi(1,end).stateSeq(1,1).z == 5;
C1.data = seq1(:,dataIDs);
Y_C1 = generateDataFromModel(C1);
figure; 
subplot(2,1,1), plotDataMin(C1.data)
subplot(2,1,2), plotDataMin(Y_C1)   

C2 = bestARCH.Psi(1,end).theta(1,8);
dataIDs = bestARCH.Psi(1,end).stateSeq(1,3).z == 8;
C2.data = seq3(:,dataIDs);
Y_C2 = generateDataFromModel(C2);
figure; 
subplot(2,1,1), plotDataMin(C2.data)
subplot(2,1,2), plotDataMin(Y_C2) 

C3a = bestARCH.Psi(1,end).theta(1,10);
dataIDs = bestARCH.Psi(1,end).stateSeq(1,4).z == 10;
C3a.data = seq4(:,dataIDs);
Y_C3a = generateDataFromModel(C3a);
figure; 
subplot(2,1,1), plotDataMin(C3a.data)
subplot(2,1,2), plotDataMin(Y_C3a) 

C3b = bestARCH.Psi(1,end).theta(1,10);
dataIDs = bestARCH.Psi(1,end).stateSeq(1,5).z == 10;
C3b.data = seq5(:,dataIDs);
Y_C3b = generateDataFromModel(C3b);
figure;
subplot(2,1,1), plotDataMin(C3b.data)
subplot(2,1,2), plotDataMin(Y_C3b) 
%%
B1 = bestARCH.Psi(1,end).theta(1,3);
dataIDs = bestARCH.Psi(1,end).stateSeq(1,1).z == 3;
B1.data = seq1(:,dataIDs);
Y_B1 = generateDataFromModel(B1);
figure; 
subplot(2,1,1), plotDataMin(B1.data)
subplot(2,1,2), plotDataMin(Y_B1) 


B2 = bestARCH.Psi(1,end).theta(1,9);
dataIDs = bestARCH.Psi(1,end).stateSeq(1,4).z == 9;
dataIDs(1,1:900) = 0;
B2.data = seq4(:,dataIDs);
Y_B2 = generateDataFromModel(B2);
figure; 
subplot(2,1,1), plotDataMin(B2.data)
subplot(2,1,2), plotDataMin(Y_B2)

E = bestARCH.Psi(1,end).theta(1,1);
dataIDs = bestARCH.Psi(1,end).stateSeq(1,1).z == 1;
dataIDs(1,1:1400) = 0;
E.data = seq1(:,dataIDs);
Y_E = generateDataFromModel(E);
figure; 
subplot(2,1,1), plotDataMin(E.data)
subplot(2,1,2), plotDataMin(Y_E)
