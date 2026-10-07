function make_interactive_reference(volumeFolder, outMat, legacyPath)
% Reference outputs of the two interactive legacy tools, for the web-app port:
%  (1) Demo_DispersionCorrectionManual.m : ln|scan| of Y frame 1 of Data200 at several
%      slider values (pchip equispacing, then yOCTInterfToScanCpx)
%  (2) yOCTMeasureFocusDrift.m reconstructCenterBScan + renderTileBScan : dB image of the
%      default tile/B-scan the "Choose Focus Positions" window opens on
%  (3) yOCTMeasureFocusDrift_fitDrift on example clicks
% legacyPath: folder of the original myOCT MATLAB library
addpath(genpath(legacyPath));
volumeFolder = [volumeFolder '/'];

%% (1) manual dispersion tool
filePath = [volumeFolder 'Data200/'];
[interf, dimensions] = yOCTLoadInterfFromFile(filePath, 'BScanAvgFramesToProcess', 1, 'YFramesToProcess', 1);
[interfe, dimensionse] = yOCTEquispaceInterf(interf, dimensions);
sliderVals = [log10(100), log10(8.949e7), -log10(1.2e8)];
D.lg = zeros([size(interf,1)/2, size(interf,2), numel(sliderVals)]);
for i = 1:numel(sliderVals)
    val = sliderVals(i);
    beta = sign(val) * 10^(abs(val));
    scanCpxe = yOCTInterfToScanCpx(interfe, dimensionse, 'dispersionQuadraticTerm', beta);
    D.lg(:,:,i) = log(abs(scanCpxe));
    D.beta(i) = beta;
end
D.sliderVals = sliderVals;
D.interfe = interfe;
D.lambdaEq = dimensionse.lambda.values;

%% (2) focus window default B-scan (copy of reconstructCenterBScan + renderTileBScan maths)
json = awsReadJSON([volumeFolder 'ScanInfo.json']);
[dimOneTile_mm, ~] = yOCTProcessTiledScan_createDimStructure(volumeFolder, NaN);
[~, xi0] = min(abs(json.xCenters_mm)); [~, yi0] = min(abs(json.yCenters_mm));
[~, zi0] = min(abs(json.zDepths));
idx = find(abs(json.gridZcc - json.zDepths(zi0)) < 1e-9 & abs(json.gridXcc - json.xCenters_mm(xi0)) < 1e-9 ...
    & abs(json.gridYcc - json.yCenters_mm(yi0)) < 1e-9, 1);
folderPath = [volumeFolder json.octFolders{idx} '/'];
yIInFile = max(1, round(length(dimOneTile_mm.y.values) / 2));
beta = 8.949e7;
reconstructConfig = {'dispersionQuadraticTerm', beta};
[intFrame, dimFrame] = yOCTLoadInterfFromFile([{folderPath}, reconstructConfig, ...
    {'dimensions', dimOneTile_mm, 'YFramesToProcess', yIInFile, 'octSystem', json.octSystem}]);
[scan1, ~] = yOCTInterfToScanCpx([{intFrame} {dimFrame} reconstructConfig]);
scan1 = abs(scan1);
[scan1, ~] = yOCTOpticalPathCorrection(scan1, dimFrame, json);
F.imLog = mag2db(abs(scan1) + eps);
F.baseLo = prctile(F.imLog(:), 5);
F.baseHi = prctile(F.imLog(:), 99.8);
F.z_mm = dimFrame.z.values; F.x_mm = dimFrame.x.values;
F.folder = json.octFolders{idx}; F.yIInFile = yIInFile;

%% (3) drift fit on example clicks
zPix_um = median(diff(dimOneTile_mm.z.values)) * 1e3;
nZ = length(dimOneTile_mm.z.values);
clicks = [0 433; 0.05 452; 0.1 470; 0.15 420; 0.2 507];
zAll = -0.03:0.01:0.25;
[G.focus, G.diag] = yOCTMeasureFocusDrift_fitDrift(clicks(:,1), clicks(:,2), zAll, zPix_um, nZ, ...
    'tissueRefractiveIndex', 1.33);
[G.focusOne, ~] = yOCTMeasureFocusDrift_fitDrift(0, 433, json.zDepths, zPix_um, nZ, 'tissueRefractiveIndex', 1.33);
G.clicks = clicks; G.zAll = zAll; G.zPix_um = zPix_um; G.nZ = nZ;
G.rejectedIdx = G.diag.rejectedIdx; G.driftSlope = G.diag.driftSlope; G.tissueRI = G.diag.tissueRI;
G.regime = G.diag.driftRegime;
save(outMat, 'D', 'F', 'G', '-v7.3');
fprintf('saved %s\n', outMat);
end
