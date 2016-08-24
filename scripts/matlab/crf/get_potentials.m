function [nodePot, edgePot, edgeStruct] = get_potentials(X)
close all

% figure;
X = rgb2gray(X);
X = single(X);
% imagesc(X); axis image; axis off;
% colormap(jet)
% title('Original Img');

%% Represent denoising as UGM

[nRows,nCols] = size(X);
nNodes = nRows*nCols;
nStates = 2;

adj = sparse(nNodes,nNodes);

% Add Down Edges
ind = 1:nNodes;
exclude = sub2ind([nRows nCols],repmat(nRows,[1 nCols]),1:nCols); % No Down edge for last row
ind = setdiff(ind,exclude);
adj(sub2ind([nNodes nNodes],ind,ind+1)) = 1;

% Add Right Edges
ind = 1:nNodes;
exclude = sub2ind([nRows nCols],1:nRows,repmat(nCols,[1 nRows])); % No right edge for last column
ind = setdiff(ind,exclude);
adj(sub2ind([nNodes nNodes],ind,ind+nRows)) = 1;

% Add Up/Left Edges
adj = adj+adj';
edgeStruct = UGM_makeEdgeStruct(adj,nStates);

% Learned optimal sub-modular parameters
Xstd = UGM_standardizeCols(reshape(X,[1 1 nNodes]),1);
nodePot = zeros(nNodes,nStates);
nodePot(:,1) = exp(-1-2.5*Xstd(:));
nodePot(:,2) = 1;

edgePot = zeros(nStates,nStates,edgeStruct.nEdges);
for e = 1:edgeStruct.nEdges
    n1 = edgeStruct.edgeEnds(e,1);
    n2 = edgeStruct.edgeEnds(e,2);

    pot_same = exp(1.8 + .3*1/(1+abs(Xstd(n1)-Xstd(n2))));
    edgePot(:,:,e) = [pot_same 1;1 pot_same];
end
end
