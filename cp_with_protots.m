% 
% Calculating credibility- and confidence-values of conformal prediction for vectorial data by means
% of the trained prototypes.
%
% Inputs:
%   mTrData: training data
%   vTrLabels: true labels of training data
%   mTeData: test data
%   vTeLabels: true labels of test data
%   mProtos: trained prototypes on training data 
%   vProtoLabels: corresponding labels of the prototypes
%
% Outputs:
%   oResult: 
%       oResult.mP_Values: p-values for each test data point and each possible
%           label
%       oResult.vConfidence: the confidence value of the predicted label which
%           is the second largest p-value
%       oResult.vCredibility: the credibility value of the predicted label
%           which is the largest p-value
%       oResult.vPred_Region: contains the predicted label for each data
%           point, namely the label with largest p-value
%       oResult.vPred_Region_threshold: contains a set of possible labels whose
%           p-value are larger than to a given threshold (iEps).
%
%
%   Xibin Zhu (c) 2014 (10.06.2014)
%   Bielefeld University, CITEC, Theoretical Computer Science
%   Mail: xzhu@techfak.uni-bielefeld.de
% 
%
function [oResult] = cp_with_protots(mTrData, vTrLabels, mTeData, vTeLabels, mProtos, vProtoLabels)

iNrTrData = size(mTrData,1);
iNrTestData = size(mTeData,1);
vUniq_Labels = unique(vTrLabels);
iNrAllLabels = length(vUniq_Labels);

vNonConformTrain = zeros(iNrTrData,1);
mNonConformTest = zeros(iNrTestData,iNrAllLabels);

fNonConform = @(x,y)non_conformity_measure(mProtos, vProtoLabels, x,y); % function pointer 

% non-conformity of training data
vNonConformTrain = fNonConform(mTrData, vTrLabels);

% Calculating non-conformity of test data with respect to all possible labels:
%   Trying to use every possible label to label test data and calculate the
%   corresponding non-conformity
for i = 1:iNrAllLabels
        vPredLabels = ones(iNrTestData,1)*vUniq_Labels(i); % generate a vector of a possible label for all data points
        mNonConformTest(:,i) = fNonConform(mTeData, vPredLabels);
end

oResult = determine_confidence_credibility(vNonConformTrain, mNonConformTest, 0.95, 1);
oResult.accuracy = 1 - sum(oResult.vPred_Region ~= vTeLabels)/iNrTestData;

end


% Using d+/d- as non conformity measure, or as alternative using (d+ - d-)/(d+ + d-) 
function [vAlphas] = non_conformity_measure(mPrototypes, vProtoLabels, mData, vLabels)

    nr_Data = length(vLabels);
    vAlphas = zeros(nr_Data, 1); 	% non-conformities 
    
    % Euclidean distance 
    distPrototypesToData = calc_euclidean_distance_protos_to_data(mPrototypes, mData);
    
    for i = 1:nr_Data
        pos_idx = vProtoLabels == vLabels(i); % indices of protos with same label (postive prototypes)
        
        d_pos = min(distPrototypesToData(pos_idx, i));
        d_neg = min(distPrototypesToData(~pos_idx, i));

        vAlphas(i) = d_pos/d_neg;
        
        if isnan(vAlphas(i))
            error('Non-conformal value: NaN!!!');
        end
    end
    
end


% calculate the confidence and credibility of each test data point
% iEps: significance level, default: 0.95
% bTest: should be alway 1, if confidence/credibilty of test data are calculated
function [oResult] = determine_confidence_credibility(vNonConformityTrain, mNonConformityTest, iEps, bTest)

    if  any( all(isnan(mNonConformityTest)) )
        error('mNonConformityTest contains at least one NaN!!!');
    end
    
    iNrTrainData = length(vNonConformityTrain);

    iNrTestData = size(mNonConformityTest, 1);
    iNrLabels = size(mNonConformityTest, 2);
    
    mP_Values = -1*ones(iNrTestData, iNrLabels); % initialized by -1
    vConfidence = zeros(iNrTestData, 1);
    vCredibility = zeros(iNrTestData, 1);
 
    vPred_Region = zeros(iNrTestData, 1);
    vPred_Region_threshold = cell(iNrTestData, 1);
    
    % p-values of all possible labels for each test data point
    for i=1:iNrTestData
        
        if bTest
           logidx = logical( ones(iNrTrainData,1) ); 
        else
           logidx = logical( ones(iNrTrainData,1) );  logidx(i)=0; 
        end
        
        for j=1:iNrLabels
            mP_Values(i,j) = length( find(vNonConformityTrain(logidx) >= mNonConformityTest(i, j)) )/(iNrTrainData+1); % can be zero, if data are labeled by a wrong label
        end
        
        if all(mP_Values(i,:)==0)  % if all p-values are zero, than take the indix of smallest alpha as label
          [~, I] = min( mNonConformityTest(i,:) );
           mP_Values(i, I) = exp(-5); % set corresponding r-value to a small value!
           vPred_Region(i) = I;
           
           vCredibility(i) = exp(-5);
           vConfidence(i)  = 1-0;
        else
            [sorted_r_values,  I]    = sort(mP_Values(i,:));
            vCredibility(i) = sorted_r_values(iNrLabels);         % the largest p-value
            vConfidence(i)  = 1 - sorted_r_values(iNrLabels - 1); % the second largest p-value

            vPred_Region(i) = I(iNrLabels);     % the label with the largest p-value !!! if p-values for all labels are same, it will take the last one as label.
        end
        vPred_Region_threshold{i} = find(mP_Values(i,:) > iEps); % the labels whose p-values large than the threshold
        
    end
    
    oResult.mP_Values    = mP_Values;
    oResult.vConfidence  = vConfidence;
    oResult.vCredibility = vCredibility;
    oResult.vPred_Region = vPred_Region; % contrains only the most possible label for each data point
    oResult.vPred_Region_threshold = vPred_Region_threshold; % contains a set of possible labels for each data point whose p-values are above the given threshold iEps
    
end


function distPrototypesToData = calc_euclidean_distance_protos_to_data( mPrototypes, mData )
%
% Calculates the distance matrix for a given (n,d)-matrix and (p,d)-matrix
% Output is a (p,n)-matrix of the distance between p neurons and n data points
%
% Alexander Hasenfuss (c) 2005
%
%

    [nr_data, dim_data] = size( mData );
    [nr_protos,dim_proto] = size( mPrototypes );

    if dim_data ~= dim_proto
        error('Dimension mismatched!');
    end;

    distPrototypesToData = zeros(nr_protos,nr_data);

    for i = 1:nr_protos
      temp = (mData - (ones(nr_data,1) * mPrototypes(i,:))).^2;
      distPrototypesToData(i,:) = sqrt( sum( temp, 2 ) )';  
    end
end